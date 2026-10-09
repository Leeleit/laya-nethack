"""Compare two laya_trace files action by action (a, b): e.g. forkrun's parent trace against the recorded game.

  python fork_check.py <trace.jsonl.zst> <recorded.jsonl.zst> [...pairs]
"""
import io
import json
import sys

import zstandard


def steps(path):
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        lines = text.read().splitlines()
    return [(r["a"], r["b"]) for r in map(json.loads, lines[1:])]


def main():
    args = sys.argv[1:]
    bad = 0
    for x, y in zip(args[::2], args[1::2]):
        a, b = steps(x), steps(y)
        diff = next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), None)
        same = diff is None and len(a) == len(b)
        bad += not same
        print(f"{x}: {len(a)} vs {len(b)} steps, " + ("identical" if same else f"first difference at step {diff}"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
