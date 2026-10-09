"""Behaviour-cloning dataset: laya learns to play NetHack by itself from the bot's recorded games.

Every game turn (a trace record with a screen) is a state; the bot's first command of the turn (bc_vocab.parse,
interface keys and its information queries dropped) is the answer to the `command` question, and its arguments
answer `item_cmd` / `item` / `direction` / `eat_here` (bc_vocab.labelled). Commands outside the vocabulary
(travel, #loot, test engravings: ~1 % of turns) give no row. The second command of a fast hero's turn has no
screen in the trace and gives no row either. Half the states (SHOW_STACK) show the bot's strategy stack, the
others "unknown": one model for laya inside the bot (bot + model, the goal) and alone.

Train keeps a share KEEP of each command (moves and searches are ~90 % of turns and neighbouring turns are near
copies); meta.json has the shares, to undo the resampling at play time: p(command) / KEEP (bc_vocab.UNDO),
renormalised.
Split by crc32 of the trace path: test 1 game in 20.
  python bc_build.py --root <run dir with trace/> [...] --out bc_data [--identity val-dwa-law-fem] [--workers 31]
-> <out>/{train,test}.jsonl, meta.json
"""
import argparse
import collections
import json
import os
import random
import subprocess
import zlib
from bisect import bisect_right
from collections import deque
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from bc_vocab import HP, KEEP, MAXHP, TIME, command_of, key, labelled, message_ok, parse, state_text

# share of states that show the bot's strategy stack (laya inside the bot); the rest say "unknown" (laya alone)
SHOW_STACK = 0.5
# share of states that show the hero's last commands; the rest say "unknown". bc-v1 showed them always and copied
# the previous command where the bot changed it (15-20 % of such turns; copycat, de Haan 2019, Wen 2020)
SHOW_LAST = 0.5


def split_of(path):
    return "test" if zlib.crc32(path.encode()) % 20 == 0 else "train"


def build_game(path, identity):
    split = split_of(path)
    rng = random.Random(path)
    lines = subprocess.run(["zstdcat", path], capture_output=True, check=True).stdout.splitlines()[1:]
    msgs, inv, last = deque(maxlen=6), [], deque(maxlen=6)
    hp_turns, hp_vals = [], []
    rows, counts = [], collections.Counter()
    state, steps = None, []

    def close(nxt=""):
        cmds = parse(steps, nxt)
        first = next((c for c in cmds if command_of(c) is not None), None)
        if first is not None:
            qs = labelled(first, state[2])
            cmd = qs["command"][1]["label"]
            counts[cmd] += 1
            if rng.random() < KEEP[cmd]:
                rows.append({"id": state[0], "state": state[1],
                             "questions": {n: q for n, (q, _) in qs.items()},
                             "gold": {n: g for n, (_, g) in qs.items()}})
        last.extend(c for c in map(command_of, cmds) if c)  # what took effect, as a player's own history

    for idx, line in enumerate(lines):
        r = json.loads(line)
        m = r.get("m")
        if "scr" in r and state is not None:
            close(m or "")
        if message_ok(m) and (not msgs or msgs[-1] != m[:160]):
            msgs.append(m[:160])
        if "inv" in r:
            inv = r["inv"]
        if "scr" in r:
            b = r["b"]
            back = []
            for d in (10, 30):
                i = bisect_right(hp_turns, b[TIME] - d)
                back.append(hp_vals[i - 1] if i else None)
            hp_turns.append(b[TIME])
            hp_vals.append((b[HP], b[MAXHP]))
            rid = f"{os.path.basename(path)}#{idx}"
            stack = r["s"] if rng.random() < SHOW_STACK else None
            shown = list(last) if rng.random() < SHOW_LAST else None
            state = (rid, state_text(identity, b, r["scr"], r.get("mon", []), back, list(msgs), inv, shown,
                                     stack), list(inv))
            steps = []
        if state is not None:
            steps.append((key(r.get("a")), m or ""))
    if state is not None:
        close()
    return split, rows, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, nargs="+", help="run dirs with trace/*.jsonl.zst")
    ap.add_argument("--out", default="bc_data")
    ap.add_argument("--identity", default="val-dwa-law-fem")
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--workers", type=int, default=31)
    a = ap.parse_args()
    files = sorted(str(p) for root in a.root for p in Path(root, "trace").glob("*.jsonl.zst"))
    if a.games:
        files = files[:a.games]
    os.makedirs(a.out, exist_ok=True)
    out = {s: open(f"{a.out}/{s}.jsonl", "w", encoding="utf-8", newline="\n") for s in ("train", "test")}
    nrows, nq, counts, games = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    with Pool(a.workers) as pool:
        for split, rows, c in pool.imap_unordered(partial(build_game, identity=a.identity), files, chunksize=2):
            games[split] += 1
            counts.update(c)
            for r in rows:
                out[split].write(json.dumps(r) + "\n")
                nq.update(f"{split}:{q}" for q in r["questions"])
            nrows[split] += len(rows)
    for f in out.values():
        f.close()
    meta = {"games": games, "rows": nrows, "questions": nq, "keep": KEEP,
            "turn_commands": dict(counts.most_common())}
    Path(a.out, "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
