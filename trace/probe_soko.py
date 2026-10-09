"""probe: Sokoban in bot traces -- per game, the Sokoban levels visited (dnum 4), turns there, the deepest Dlvl, the
turn the trip ended, and the rings / wands / amulets carried at the end.
  python probe_soko.py <trace dir>"""
import glob
import json
import re
import subprocess
import sys
from collections import Counter

from bot_profile import DEPTH, DNUM, TIME

LEVNO = 24
ITEM = re.compile(r"\b(ring|wand|amulet|bag)\b")
n, visited, full = 0, 0, Counter()
for path in sorted(glob.glob(sys.argv[1] + "/*.jsonl.zst")):
    lines = subprocess.run(["zstd", "-dcq", path], capture_output=True, check=True).stdout.splitlines()[1:]
    soko, b, inv, first, last, deepest = set(), None, [], None, None, 0
    for line in lines:
        r = json.loads(line)
        b = r.get("b", b)
        inv = r.get("inv", inv)
        if b is None:
            continue
        deepest = max(deepest, b[DEPTH] if b[DNUM] in (0, 1) else 0)
        if b[DNUM] == 4:
            soko.add(b[LEVNO])
            first = first or b[TIME]
            last = b[TIME]
    n += 1
    if soko:
        visited += 1
        full[len(soko)] += 1
        items = [x[:40] for x in inv if ITEM.search(x)]
        print(f"{path[-18:]} soko levels {sorted(soko)} turns {first}-{last} end turn {b[TIME]} deepest {deepest} "
              f"items {items}")
print(f"{n} games, {visited} visited Sokoban, levels visited {dict(full)}")
