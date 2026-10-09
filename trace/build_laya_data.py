"""Risk-model dataset for laya from the bot's traces.

Rows: the first action of each game turn in a "risk state" -- HP below 80%, or the bot in a
non-calm strategy, or deeper than Dlvl 4 (20% of turns, 96.5% of the turns within K of a death).
Question (noul): will the hero die within the next K game turns? Gold: the actual outcome.

Split by game (crc32 of the trace path), 1 in 5 games held out:
  train.jsonl       all positives + NEG_RATE of the negatives (laya fine-tuning; odds shifted, see
                    NEG_RATE -- multiply the model's odds by it to get natural-rate probabilities)
  test.jsonl        every STRIDE-th risk state of the held-out games (natural rate)
  features.parquet  the same blstats/strategy features for every STRIDE-th risk state of all
                    games, with split and row id -- the tabular baselines train and test on these
-> /data/laya/nethack/trace/laya_data/
"""
import json
import os
import random
import re
import subprocess
import sys
import zlib
from bisect import bisect_right
from collections import deque
from multiprocessing import Pool

import pandas as pd

from analyze_traces import K, STRIDE, games

OUT = "/data/laya/nethack/trace/laya_data"
NEG_RATE = 0.02
HP, MAXHP, DEPTH, AC, XL, TIME, HUNGER = 10, 11, 12, 16, 18, 20, 21
CALM = {"<base>", "Inventory.pickup_and_drop_items",
        "GlobalLogic.tour_strategy.<locals>.exploration_strategy.<locals>.<lambda>",
        "ExplorationLogic.search_neighbors_for_traps", "Agent.eat_corpses_from_ground",
        "ExplorationLogic.go_to_strategy", "ExplorationLogic.explore_stairs", "GlobalLogic.offer_corpses",
        "ExplorationLogic.untrap_traps"}
QUESTION = {"type": "noul", "instructions": f"Will the hero die within the next {K} game turns?"}
_LOCALS = re.compile(r"\.<locals>|\.<lambda>|\.f$")


def short(strategy):
    return _LOCALS.sub("", strategy)


def state_text(identity, stack, scr, hp_then, msgs, b):
    """Most predictive first: laya drops the end of a state that does not fit its window."""
    x, y = b[0], b[1] + 1  # map row y is tty row y + 1
    rows = [scr[r] if r < len(scr) else "" for r in range(max(1, y - 5), min(22, y + 6))]
    crop = "\n".join(row.ljust(x + 13)[max(0, x - 12):x + 13].rstrip() for row in rows)
    then = f"{hp_then[0]}/{hp_then[1]}" if hp_then else "unknown"
    return (f"Character: {identity}. Bot behaviour: {' > '.join(map(short, stack)) or 'explore/descend'}.\n"
            f"HP {b[HP]}/{b[MAXHP]}, 10 turns ago {then}. Dlvl {b[DEPTH]}, XL {b[XL]}, AC {b[AC]}, turn {b[TIME]}.\n"
            f"Status: {scr[22].strip() if len(scr) > 22 else ''} | {scr[23].strip() if len(scr) > 23 else ''}\n"
            f"Recent messages: {' | '.join(msgs) or 'none'}\n"
            f"Map around the hero (@):\n{crop}")


def build_game(job):
    path, res = job
    test = zlib.crc32(path.encode()) % 5 == 0
    rng = random.Random(path)
    lines = subprocess.run(["zstdcat", path], capture_output=True, check=True).stdout.splitlines()[1:]
    if len(lines) != res["steps"]:
        return [], [], []
    died = res["status"] == "completed" and bool(res["cause_of_death"])
    msgs = deque(maxlen=4)
    hp_turns, hp_vals = [], []
    train, held, feats = [], [], []
    k = 0
    for line in lines:
        r = json.loads(line)
        m = r.get("m")
        if m and (not msgs or msgs[-1] != m[:160]):
            msgs.append(m[:160])
        if "scr" not in r:
            continue
        b = r["b"]
        turn = b[TIME]
        i = bisect_right(hp_turns, turn - 10)
        hp_then = hp_vals[i - 1] if i else None
        hp_turns.append(turn)
        hp_vals.append((b[HP], b[MAXHP]))
        strat = r["s"][-1] if r["s"] else "<base>"
        hp_frac = b[HP] / max(b[MAXHP], 1)
        if not (hp_frac < 0.8 or strat not in CALM or b[DEPTH] > 4):
            continue
        k += 1
        y = int(died and res["turns"] - turn <= K)
        rid = f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}/{os.path.basename(path)}#{turn}"
        sampled = k % STRIDE == 0
        if sampled:
            feats.append({"id": rid, "split": "test" if test else "train", "y": y, "identity": res["identity"],
                          "hp_frac": hp_frac, "maxhp": b[MAXHP], "depth": b[DEPTH], "xl": b[XL], "ac": b[AC],
                          "hunger": b[HUNGER], "stack": len(r["s"]), "strategy": strat,
                          "hp_drop10": (hp_then[0] / max(hp_then[1], 1) - hp_frac) if hp_then else 0.0})
        if test and not sampled:
            continue
        if not test and not (y or rng.random() < NEG_RATE):
            continue
        row = {"id": rid, "state": state_text(res["identity"], r["s"], r["scr"], hp_then, list(msgs), b),
               "questions": {"die_soon": QUESTION},
               "gold": {"die_soon": {"probabilities": {"true": float(y), "false": float(1 - y)}, "label": str(bool(y)).lower()}}}
        (held if test else train).append(row)
    return train, held, feats


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = games()
    with Pool(16) as pool:
        parts = pool.map(build_game, jobs, chunksize=4)
    train = [r for p in parts for r in p[0]]
    held = [r for p in parts for r in p[1]]
    random.Random(0).shuffle(train)
    for name, rows in (("train", train), ("test", held)):
        with open(f"{OUT}/{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    feats = pd.DataFrame([f for p in parts for f in p[2]])
    feats.to_parquet(f"{OUT}/features.parquet")
    meta = {"K": K, "stride": STRIDE, "neg_rate": NEG_RATE, "games": len(jobs),
            "test_games": sum(zlib.crc32(p.encode()) % 5 == 0 for p, _ in jobs),
            "train_rows": len(train), "train_pos": sum(r["gold"]["die_soon"]["label"] == "true" for r in train),
            "test_rows": len(held), "test_pos": sum(r["gold"]["die_soon"]["label"] == "true" for r in held),
            "feature_rows": len(feats)}
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(json.dumps(meta))
    print(train[0]["state"])


if __name__ == "__main__":
    sys.exit(main())
