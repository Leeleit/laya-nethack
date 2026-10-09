"""risk-v2 dataset: time to death as a distribution, from the v2 traces (monsters, inventory).

Changes from build_laya_data.py (risk-v1), each from what v1 or the literature showed:
  - target: one ordinal `score` question over BUCKETS of turns until death -- discrete-time survival
    (Kvamme & Borgan 2019; Gensheimer & Narasimhan 2019) instead of one fixed horizon; P(T <= h) for
    any bucket edge h is a cumulative sum, so each bot decision can read its own horizon (Fedus et al.
    2019 on learning several horizons at once). Edges from the bot's own warning times: median
    2 turns from the last HP >= 50% to death, 8 from the last HP >= 80%, 90th pct 108.
  - state: the whole 1024-token window (v1 used ~234 tokens): named monsters with offsets, six
    messages, HP 10 and 30 turns ago, the inventory, the full level map -- most predictive first,
    because laya cuts the end of a state that does not fit.
  - val split: games held out of training at the natural class rates, for calibration (v1 fitted its
    temperature on the rebalanced training slice and over-predicted death ~2x).

Rows: risk states as in v1. Split by crc32 of the trace path: test 1 in 5 games, val 1 in 10 of the
rest, train the others. Train keeps a random KEEP[bucket] share of the risk states (bucket weights in
meta.json undo it): all of the last 2 turns, fewer further back -- neighbouring turns are near
copies, the independent examples are the deaths, not the rows, and every row costs ~540 tokens of
training. Val/test keep every STRIDE-th risk state.
-> /data/laya/nethack/trace/v2_data/{train,val,test}.jsonl, features.parquet, meta.json
   (python build_v2.py [--root <runs dir>] [--out <dir>])
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import zlib
from bisect import bisect_right
from collections import deque
from functools import partial
from multiprocessing import Pool

import pandas as pd

from analyze_traces import STRIDE, games
from build_laya_data import AC, CALM, DEPTH, HP, HUNGER, MAXHP, TIME, XL, short
from state_fmt import compact_inv, hero_map, status, surroundings, terse_state

ROOT = "/data/laya/nethack/nethackers/runs/traces2"
OUT = "/data/laya/nethack/trace/v2_data"
EDGES = [2, 7, 30, 100]  # bucket k: T <= EDGES[k]; last bucket T > 100
# train share kept per bucket; the far one makes ~40% of the training rows (~84% of risk states)
KEEP = [1.0, 0.5, 1 / 3, 1 / 3, 0.03]
LEVELS = ["dies within 2 turns", "dies in 3 to 7 turns", "dies in 8 to 30 turns",
          "dies in 31 to 100 turns", "survives more than 100 turns"]
QUESTION = {"type": "score", "criteria": LEVELS,
            "instructions": "If the bot keeps playing as it does now, when will the hero die?"}
# --terse: the humans' short layout and wording (human_q6 terse ttd)
QUESTION_TERSE = {"type": "score", "criteria": ["at most 2", "3-7", "8-30", "31-100", "over 100"],
                  "instructions": "The bot plays. Turns until the hero dies?"}
NLE_VERSION = "3.6.6"


def bucket(t):
    return next((k for k, e in enumerate(EDGES) if t <= e), len(EDGES))


def split_of(path):
    c = zlib.crc32(path.encode())
    return "test" if c % 5 == 0 else "val" if (c // 5) % 10 == 0 else "train"


def where(dx, dy):
    if max(abs(dx), abs(dy)) == 1:
        return "adjacent " + ("north" if dy < 0 else "south" if dy > 0 else "") + ("west" if dx < 0 else "east" if dx > 0 else "")
    parts = [f"{abs(dx)} {'east' if dx > 0 else 'west'}" if dx else "", f"{abs(dy)} {'south' if dy > 0 else 'north'}" if dy else ""]
    return " ".join(p for p in parts if p)


def state_text(identity, stack, rec, hp_back, msgs, inv):
    b, scr = rec["b"], rec["scr"]
    x, y = b[0], b[1] + 1  # map row y is tty row y + 1
    hp = lambda v: f"{v[0]}/{v[1]}" if v else "unknown"
    mons = "; ".join(f"{n}{' (' + k + ')' if k else ''} {where(dx, dy)}" for n, dx, dy, k in rec.get("mon", [])) or "none"
    crop = hero_map(scr, x, y)
    # identity.title(): Val-Hum-Neu-Fem, as the human rows write it (one character, one spelling)
    return (f"Character: {identity.title()}. Bot behaviour: {' > '.join(map(short, stack)) or 'explore/descend'}.\n"
            f"HP {b[HP]}/{b[MAXHP]}; 10 turns ago {hp(hp_back[0])}; 30 turns ago {hp(hp_back[1])}. "
            f"Dlvl {b[DEPTH]}, XL {b[XL]}, AC {b[AC]}, turn {b[TIME]}.\n"
            f"Status: {status(scr)}\n"
            f"Monsters in view: {mons}\n"
            f"Recent messages: {' | '.join(msgs) or 'none'}\n"
            f"Map around the hero (@):\n{crop}\n"
            f"Inventory: {'; '.join(inv) or 'unknown'}")


def terse_text(identity, stack, rec, hp_back, msgs, inv, deepest):
    """The bot's state in the humans' short layout (state_fmt.terse_state): the map in words with the monsters by
    name ('soldier ant 2', 'little dog (pet) 1'), the bot's strategy, the inventory without its gold line (the
    status has $) and without articles (compact_inv)."""
    b, scr = rec["b"], rec["scr"]
    x, y = b[0], b[1] + 1  # map row y is tty row y + 1
    names = {(x + dx, y + dy): f"{n} ({k})" if k else n for n, dx, dy, k in rec.get("mon", [])}
    s = dict(hp=b[HP], maxhp=b[MAXHP], dlvl=b[DEPTH], deepest=deepest, xl=b[XL], ac=b[AC], turn=b[TIME])
    inv = compact_inv("; ".join(it for it in inv if not it.startswith("$ - ")) or "unknown")
    return terse_state(identity.title(), NLE_VERSION, s, hp_back, scr, msgs, surroundings(scr, x, y, names), inv,
                       bot=" > ".join(map(short, stack)) or "explore/descend")


BOT_UI = re.compile(r"^#|Showing known terrain|^You see no objects here\.$|^Never mind\.$|"
                    r"^(What|Which|In what direction|Do you|Are you|Really|Call|Name|Type)\b.*\?")


def walk(lines):
    """Every trace record with a screen ("scr", the first action of a game turn), with the context
    state_text needs: (record index, record, HP 10 and 30 turns back, recent messages, inventory).
    Recent messages leave out the bot's own interface lines (BOT_UI: its "# terrain" look, its ":" answer
    "You see no objects here.", prompts with their typed echo) -- ~40% of its messages (msg_freq.py)."""
    msgs, inv = deque(maxlen=6), []
    hp_turns, hp_vals = [], []
    for idx, line in enumerate(lines):
        r = json.loads(line)
        m = r.get("m")
        if m and BOT_UI.search(m):
            m = None
        if m and (not msgs or msgs[-1] != m[:160]):
            msgs.append(m[:160])
        if "inv" in r:
            inv = r["inv"]
        if "scr" not in r:
            continue
        b = r["b"]
        back = []
        for d in (10, 30):
            i = bisect_right(hp_turns, b[TIME] - d)
            back.append(hp_vals[i - 1] if i else None)
        hp_turns.append(b[TIME])
        hp_vals.append((b[HP], b[MAXHP]))
        yield idx, r, back, msgs, inv


def build_game(job, features_only=False, terse=False):
    path, res = job
    split = split_of(path)
    rng = random.Random(path)
    lines = subprocess.run(["zstdcat", path], capture_output=True, check=True).stdout.splitlines()[1:]
    if len(lines) != res["steps"] or not (res["status"] == "completed" and res["cause_of_death"]):
        return [], [], [], [], {}
    rows = {"train": [], "val": [], "test": []}
    feats, k, deepest = [], 0, 0
    for _, r, back, msgs, inv in walk(lines):
        b = r["b"]
        turn = b[TIME]
        deepest = max(deepest, b[DEPTH])
        strat = r["s"][-1] if r["s"] else "<base>"
        hp_frac = b[HP] / max(b[MAXHP], 1)
        if not (hp_frac < 0.8 or strat not in CALM or b[DEPTH] > 4):
            continue
        k += 1
        ttd = res["turns"] - turn
        bk = bucket(ttd)
        rid = f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}/{os.path.basename(path)}#{turn}"
        sampled = k % STRIDE == 0
        if sampled:
            mons = r.get("mon", [])
            feats.append({"id": rid, "split": split, "ttd": ttd, "bucket": bk, "identity": res["identity"],
                          "hp_frac": hp_frac, "maxhp": b[MAXHP], "depth": b[DEPTH], "xl": b[XL], "ac": b[AC],
                          "hunger": b[HUNGER], "stack": len(r["s"]), "strategy": strat,
                          "hp_drop10": (back[0][0] / max(back[0][1], 1) - hp_frac) if back[0] else 0.0,
                          "hp_drop30": (back[1][0] / max(back[1][1], 1) - hp_frac) if back[1] else 0.0,
                          "hostiles": sum(1 for x in mons if x[3] != "pet"),
                          "hostiles_adjacent": sum(1 for x in mons if x[3] != "pet" and max(abs(x[1]), abs(x[2])) == 1)})
        if features_only:
            continue
        if split == "train":
            if rng.random() >= KEEP[bk]:
                continue
        elif not sampled:
            continue
        text = (terse_text(res["identity"], r["s"], r, back, list(msgs), inv, deepest) if terse else
                state_text(res["identity"], r["s"], r, back, list(msgs), inv))
        rows[split].append({"id": rid, "state": text, "questions": {"ttd": QUESTION_TERSE if terse else QUESTION},
                            "gold": {"ttd": {"probabilities": {str(i): float(i == bk) for i in range(len(LEVELS))},
                                             "label": str(bk)}}})
    return rows["train"], rows["val"], rows["test"], feats, {"split": split}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", default=[ROOT], help="collected runs (<root>/<run>/evidence.json, trace/)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--features-only", action="store_true", help="only features.parquet (e.g. for plan_branches.py)")
    ap.add_argument("--terse", action="store_true", help="the humans' short layout and wording (terse_text)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    jobs = [j for root in a.root for j in games(root)]
    with Pool(a.workers) as pool:
        parts = pool.map(partial(build_game, features_only=a.features_only, terse=a.terse), jobs, chunksize=4)
    meta = {"edges": EDGES, "levels": (QUESTION_TERSE if a.terse else QUESTION)["criteria"], "keep": KEEP,
            "stride": STRIDE, "games_found": len(jobs), "roots": a.root,
            "games_used": {s: sum(1 for p in parts if p[4].get("split") == s) for s in ("train", "val", "test")}}
    for i, name in enumerate(("train", "val", "test")):
        rows = [r for p in parts for r in p[i]]
        if name == "train":
            random.Random(0).shuffle(rows)
        with open(f"{out}/{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        meta[f"{name}_rows"] = len(rows)
        meta[f"{name}_buckets"] = pd.Series([r["gold"]["ttd"]["label"] for r in rows]).value_counts().sort_index().to_dict()
    # multiply the model's bucket probabilities by these and renormalise to undo the train sampling
    meta["bucket_weights"] = KEEP
    feats = pd.DataFrame([f for p in parts for f in p[3]])
    feats.to_parquet(f"{out}/features.parquet")
    meta["feature_rows"] = len(feats)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(json.dumps(meta))
    if not a.features_only:
        print([r for p in parts for r in p[0]][0]["state"])


if __name__ == "__main__":
    sys.exit(main())
