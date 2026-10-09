"""What the bot's recorded games offer for teaching laya to play by itself (behaviour cloning):
per game the bot's actions (NLE action index per step), how many steps carry a screen (laya_trace writes "scr"
only on the first action of a game turn), the actions taken inside a turn (prompts, menus, multi-key commands),
and the bot's strategy stack (HiHack-style high-level labels).
  python bc_inventory.py --root R [--out results/bc_inventory.txt]
"""
import argparse
import collections
import json
from multiprocessing import Pool
from pathlib import Path

import nle.nethack as nh
import zstandard


def game(path):
    with open(path, "rb") as fh:
        lines = zstandard.ZstdDecompressor().stream_reader(fh).read().splitlines()[1:]
    acts, acts_scr, acts_in, tops = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    n = scr = 0
    turns = set()
    for line in lines:
        r = json.loads(line)
        n += 1
        a = r.get("a")
        acts[a] += 1
        if "scr" in r:
            scr += 1
            acts_scr[a] += 1
        else:
            acts_in[a] += 1
        turns.add(r["b"][20])
        s = r.get("s") or []
        tops[s[-1] if s else "-"] += 1
    return n, scr, len(turns), acts, acts_scr, acts_in, tops


def name(a):
    try:
        return str(nh.actions.ACTIONS[a]).split(".", 1)[-1] if a is not None else "None"
    except (IndexError, TypeError):
        return str(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="a run dir with trace/*.jsonl.zst")
    ap.add_argument("--out", default="results/bc_inventory.txt")
    a = ap.parse_args()
    files = sorted(Path(a.root, "trace").glob("*.jsonl.zst"))
    with Pool(31) as pool:
        got = pool.map(game, files, chunksize=4)
    n = sum(g[0] for g in got)
    scr = sum(g[1] for g in got)
    turns = sum(g[2] for g in got)
    tot = [collections.Counter() for _ in range(4)]
    for g in got:
        for c, t in zip(g[3:], tot):
            t.update(c)
    acts, acts_scr, acts_in, tops = tot
    lines = [f"{len(files)} games, {n:,} bot actions, {turns:,} game turns ({n / turns:.2f} actions a turn), "
             f"{scr:,} actions with a screen ({scr / n:.0%})",
             f"distinct actions: {len(acts)}; top 30 (all / at a turn start / inside a turn):"]
    for a_, c in acts.most_common(30):
        lines.append(f"  {name(a_):28s} {c / n:6.2%}  {acts_scr[a_] / max(scr, 1):6.2%}  {acts_in[a_] / max(n - scr, 1):6.2%}")
    k = 0
    for k, (a_, c) in enumerate(acts.most_common()):
        if sum(c2 for _, c2 in acts.most_common(k + 1)) >= 0.99 * n:
            break
    lines.append(f"actions covering 99% of steps: {k + 1}")
    lines.append(f"strategy at the top of the bot's stack ({len(tops)} distinct), top 25:")
    for s, c in tops.most_common(25):
        lines.append(f"  {s:40s} {c / n:6.2%}")
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
