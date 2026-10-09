"""How accurate are the command labels of bc_vocab.parse? Each label is checked against what the game did next
(blstats of the record after the command key, and the messages up to the next turn):
  move <dir>   the hero moved one square that way, or stayed (an attack, a door, a boulder: allowed) --
               moved some other way = a wrong label (or a teleport trap / being displaced)
  search(5)    the hero did not move
  down / up    Dlvl changed by +1 / -1 by the next turn
  pickup       a "<letter> - <item>" line, or nothing there, follows
  throw        the inventory shrank or a projectile message follows
Prints the share of each outcome per command, and random parsed turns (keys, messages, label) to read by eye.
  python bc_audit.py --root R [--games 100] [--show 40] [--out results/bc_audit.txt]
"""
import argparse
import collections
import json
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

from bc_vocab import DIRS, DIR_OPTS, command_of, key, parse

LABEL = None  # --label: show random turns of this label only
DELTA = {"k": (0, -1), "u": (1, -1), "l": (1, 0), "n": (1, 1), "j": (0, 1), "b": (-1, 1), "h": (-1, 0), "y": (-1, -1)}
DEPTH = 12


def game(path):
    lines = subprocess.run(["zstdcat", path], capture_output=True, check=True).stdout.splitlines()[1:]
    recs = [json.loads(x) for x in lines]
    out, samples = collections.Counter(), []
    starts = [i for i, r in enumerate(recs) if "scr" in r] + [len(recs)]
    rng = random.Random(path)
    for s, e in zip(starts, starts[1:]):
        steps = [(key(r.get("a")), r.get("m", "")) for r in recs[s:e]]
        cmds = parse(steps, recs[e].get("m", "") if e < len(recs) else "")
        c = next((c for c in cmds if command_of(c) is not None), None)
        if c is None:
            continue
        lab = command_of(c)
        i = s + c.step
        b0 = recs[i]["b"]
        b1 = recs[i + 1]["b"] if i + 1 < len(recs) else b0
        bend = recs[e]["b"] if e < len(recs) else recs[-1]["b"]
        msgs = " | ".join(r["m"] for r in recs[i + 1:e + 1] if r.get("m"))
        dx, dy = b1[0] - b0[0], b1[1] - b0[1]
        if lab in DIR_OPTS and c.cmd in DIRS:
            want = DELTA[c.cmd]
            res = "moved that way" if (dx, dy) == want else "stayed" if (dx, dy) == (0, 0) else \
                "level changed" if b1[DEPTH] != b0[DEPTH] else "moved elsewhere"
        elif lab in ("search", "search5"):
            res = "did not move" if (dx, dy) == (0, 0) else "moved"
        elif lab in ("down", "up"):
            d = bend[DEPTH] - b0[DEPTH]
            res = "Dlvl +1" if d == 1 else "Dlvl -1" if d == -1 else f"Dlvl {d:+d}"
        elif lab == "pickup":
            res = "got an item" if " - " in msgs else "nothing" if "nothing" in msgs.lower() or "There is" in msgs \
                else "other"
        elif lab == "throw":
            res = "thrown" if any(w in msgs for w in ("hits", "misses", "miss", "kill", "destroy")) else "other"
        else:
            res = "-"
        out[(lab if lab not in DIR_OPTS or lab in ("up", "down") else "move", res)] += 1
        if (lab == LABEL and rng.random() < 0.05) or (not LABEL and rng.random() < 0.002):
            samples.append(f"{lab:9s} {res:16s} keys {''.join(k for k, _ in steps)!r:30s} msgs {msgs[:110]!r}")
    return out, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--out", default="results/bc_audit.txt")
    ap.add_argument("--label", help="show random turns of this label only")
    a = ap.parse_args()
    global LABEL
    LABEL = a.label
    files = sorted(str(p) for p in Path(a.root, "trace").glob("*.jsonl.zst"))[:a.games]
    tot, samples = collections.Counter(), []
    with Pool(24) as pool:
        for c, s in pool.imap_unordered(game, files):
            tot.update(c)
            samples += s
    by = collections.defaultdict(dict)
    for (lab, res), v in tot.items():
        by[lab][res] = v
    lines = [f"{len(files)} games; label: share of each outcome (turns)"]
    for lab, d in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(d.values())
        lines.append(f"  {lab:9s} {n:8d}  " + ", ".join(f"{r} {v / n:.2%}" for r, v in sorted(d.items(), key=lambda kv: -kv[1])))
    random.Random(0).shuffle(samples)
    lines.append(f"random parsed turns ({min(a.show, len(samples))} of {len(samples)}):")
    lines += ["  " + s for s in samples[:a.show]]
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
