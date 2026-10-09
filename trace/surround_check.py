"""Eyeball state_fmt.surroundings: the screen's map rows next to the text, for a few states of the first
games of a states file.
  python surround_check.py <states.jsonl> [n states]
"""
import sys
import json

from state_fmt import surroundings

path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
shown = 0
for line in open(path, encoding="utf-8"):
    g = json.loads(line)
    for s in g["states"][::7]:
        scr = s["screen"]
        if len(sys.argv) > 3 and not any(ch in r for r in scr[1:22] for ch in sys.argv[3]):
            continue  # only states whose map shows one of these symbols
        print("=" * 80)
        print("\n".join(r.rstrip() for r in scr[1:22] if r.strip()))
        print(f"hero at column {s['x']}, row {s['y']}: {scr[s['y']][s['x']]!r}")
        print(surroundings(scr, s["x"], s["y"]))
        shown += 1
        if shown >= n:
            sys.exit()
