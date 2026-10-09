"""The bot's turns as key sequences: the action vocabulary of a laya policy that plays by itself.

A turn = the bot's keys from an action with a screen ("scr": laya_trace writes it on the first action of a game
turn) up to the next one. Reported raw and with the --More-- dismissals (space) dropped, which a thin interface
layer would press: the distinct sequences, how many cover 80/90/95/99 % of the turns, the most common ones, and
the commands (the first key of a turn) with how many distinct continuations each has.
  python bc_turns.py --root R [--out results/bc_turns.txt]       (needs zstandard)
"""
import argparse
import collections
import json
from multiprocessing import Pool
from pathlib import Path

import zstandard

from bc_vocab import key


def game(path):
    with open(path, "rb") as fh:
        lines = zstandard.ZstdDecompressor().stream_reader(fh).read().splitlines()[1:]
    raw, clean = collections.Counter(), collections.Counter()
    cur = None
    for line in lines:
        r = json.loads(line)
        if "scr" in r:
            if cur:
                raw["".join(map(key, cur))] += 1
                clean["".join(k for k in map(key, cur) if k != "␣")] += 1
            cur = []
        if cur is not None:
            cur.append(r.get("a"))
    return raw, clean


def coverage(c):
    n, acc, out = sum(c.values()), 0, {}
    for k, (_, v) in enumerate(c.most_common(), 1):
        acc += v
        for q in (0.8, 0.9, 0.95, 0.99):
            if q not in out and acc >= q * n:
                out[q] = k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/bc_turns.txt")
    a = ap.parse_args()
    files = sorted(Path(a.root, "trace").glob("*.jsonl.zst"))
    raw, clean = collections.Counter(), collections.Counter()
    with Pool(31) as pool:
        for r, c in pool.imap_unordered(game, files, chunksize=4):
            raw.update(r)
            clean.update(c)
    n = sum(raw.values())
    lines = [f"{len(files)} games, {n:,} turns"]
    for name, c in (("raw", raw), ("without --More-- spaces", clean)):
        cov = coverage(c)
        lines.append(f"{name}: {len(c):,} distinct sequences; sequences covering 80/90/95/99 %: "
                     + " / ".join(str(cov[q]) for q in (0.8, 0.9, 0.95, 0.99)))
    lines.append("most common (without spaces): share, sequence")
    for s, v in clean.most_common(60):
        lines.append(f"  {v / n:7.3%}  {s!r}")
    first = collections.defaultdict(collections.Counter)
    for s, v in clean.items():
        first[s[:1]][s] += v
    lines.append("commands (first key): share of turns, distinct continuations, top continuations")
    for k, c in sorted(first.items(), key=lambda kv: -sum(kv[1].values()))[:30]:
        tot = sum(c.values())
        lines.append(f"  {k!r:6s} {tot / n:7.3%} {len(c):6d}  " + ", ".join(f"{s!r} {v / tot:.0%}" for s, v in c.most_common(4)))
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
