"""The bot's game turns parsed into commands (bc_vocab.parse): what a laya policy has to choose.

Per game turn (a record with a screen to the next): how many commands it holds; the first command's
distribution (count prefixes and prompt kinds folded in); per command key the prompts it answers.
  python bc_commands.py --root R [--out results/bc_commands.txt]
"""
import argparse
import collections
import json
import re
from multiprocessing import Pool
from pathlib import Path

import zstandard

from bc_vocab import DIRS, key, parse

LETTERS = re.compile(r"\[[^\]]*\]")


def kind(prompt):
    p = prompt.lower()
    if "direction" in p:
        return "dir"
    if "what do you want to" in p and "[" in p and "write in" not in p:
        return "item"
    if "[yn" in p:
        return "yn"
    return "text" if p.endswith(":") or p.endswith("?") else "other"


def label(c):
    if c.cmd in DIRS:
        return "move" + (" (count)" if c.count else "")
    if c.cmd == "s":
        return f"search {c.count or 1}"
    sig = "+".join(kind(p) for p, _ in c.args[:3])
    return c.cmd + (f" [{sig}]" if sig else "") + (" menu" if c.menu and c.cmd != "_" else "")


def game(path):
    with open(path, "rb") as fh:
        lines = zstandard.ZstdDecompressor().stream_reader(fh).read().splitlines()[1:]
    per_turn, first, prompts, all_cmds = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    cur = None

    def flush(nxt=""):
        cmds = parse(cur, nxt)
        per_turn[min(len(cmds), 3)] += 1
        if cmds:
            first[label(cmds[0])] += 1
        for c in cmds:
            all_cmds[label(c)] += 1
            for p, k in c.args:
                prompts[(c.cmd, LETTERS.sub("[..]", p)[:60])] += 1

    for line in lines:
        r = json.loads(line)
        if "scr" in r:
            if cur:
                flush(r.get("m", ""))
            cur = []
        if cur is not None:
            cur.append((key(r.get("a")), r.get("m", "")))
    if cur:
        flush()
    return per_turn, first, prompts, all_cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/bc_commands.txt")
    ap.add_argument("--games", type=int, default=0)
    a = ap.parse_args()
    files = sorted(Path(a.root, "trace").glob("*.jsonl.zst"))
    if a.games:
        files = files[:a.games]
    tot = [collections.Counter() for _ in range(4)]
    with Pool(31) as pool:
        for got in pool.imap_unordered(game, files, chunksize=4):
            for t, g in zip(tot, got):
                t.update(g)
    per_turn, first, prompts, all_cmds = tot
    n = sum(per_turn.values())
    nc = sum(all_cmds.values())
    lines = [f"{len(files)} games, {n:,} turns, {nc:,} commands; commands per turn: "
             + ", ".join(f"{k if k < 3 else '3+'}: {v / n:.1%}" for k, v in sorted(per_turn.items()))]
    lines.append("first command of a turn: share of turns (cumulative)")
    acc = 0
    for s, v in first.most_common(60):
        acc += v
        lines.append(f"  {v / n:7.3%} {acc / n:7.2%}  {s}")
    lines.append(f"all commands ({len(all_cmds)} labels), top 60: share of commands")
    for s, v in all_cmds.most_common(60):
        lines.append(f"  {v / nc:7.3%}  {s}")
    lines.append("prompts answered, top 50: (command, prompt) count")
    for (c, p), v in prompts.most_common(50):
        lines.append(f"  {v:9,d}  {c:6s} {p}")
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
