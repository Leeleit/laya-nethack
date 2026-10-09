"""Print the bot's turns that contain a given key, with the top-line message each key was pressed at
(to work out the grammar of its key sequences: info queries, prompts, menus).
  python bc_peek.py --root R KEY [KEY ...] [--n 8] [--games 20]
"""
import argparse
import json
from pathlib import Path

import zstandard

from bc_vocab import key


def turns(path):
    with open(path, "rb") as fh:
        lines = zstandard.ZstdDecompressor().stream_reader(fh).read().splitlines()[1:]
    cur = None
    for line in lines:
        r = json.loads(line)
        if "scr" in r:
            if cur:
                yield cur
            cur = []
        if cur is not None:
            cur.append((key(r.get("a")), r.get("m", ""), r["b"][20], r["b"][0], r["b"][1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--games", type=int, default=20)
    a = ap.parse_args()
    files = sorted(Path(a.root, "trace").glob("*.jsonl.zst"))[:a.games]
    for k in a.keys:
        print(f"===== {k!r}")
        shown = 0
        for f in files:
            for t in turns(f):
                if any(s[0] == k for s in t):
                    print("  " + " | ".join(f"{s[0]}{' «' + s[1][:70] + '»' if s[1] else ''}" for s in t))
                    shown += 1
                    if shown >= a.n:
                        break
            if shown >= a.n:
                break


if __name__ == "__main__":
    main()
