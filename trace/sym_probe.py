"""What an unusual map symbol of some players stands for (a '7' some players draw): over their games'
frames, what a map cell showed (symbol, colour) just before it became the symbol and just after it stopped, with the
message row then, and which map symbols are drawn in the symbol's colour (doors are brown).
  python sym_probe.py <games.json> <name-substring> <symbol> [--max 5]   (needs nle)"""
import argparse
import collections
import json

import numpy as np
from nle import _pyconverter as converter

ap = argparse.ArgumentParser()
ap.add_argument("games")
ap.add_argument("name")
ap.add_argument("sym")
ap.add_argument("--max", type=int, default=5)
a = ap.parse_args()
sym = ord(a.sym)
into, out_of, msgs, same_col = (collections.Counter() for _ in range(4))
games = [g for g in json.load(open(a.games)) if a.name in g["name"]][:a.max]
n = 4096
chars, colors = np.zeros((n, 24, 80), np.uint8), np.zeros((n, 24, 80), np.int8)
curs = np.zeros((n, 2), np.int16)
ts, acts, scores = np.zeros(n, np.int64), np.zeros(n, np.uint8), np.zeros(n, np.int32)
for g in games:
    conv = converter.Converter(24, 80, g.get("ttyrec_version", 1))
    prev = pcol = None
    for part, path in enumerate(g["files"]):
        conv.load_ttyrec(path, gameid=0, part=part)
        while True:
            remaining = conv.convert(chars, colors, curs, ts, acts, scores)
            for f in range(n - remaining):
                cur, col = chars[f, 1:22], colors[f, 1:22]
                if prev is not None:
                    for r, c in np.argwhere((cur == sym) & (prev != sym)):
                        into[(chr(prev[r, c]), int(pcol[r, c]))] += 1
                        msgs[chars[f, 0].tobytes().decode("latin-1").strip()[:60]] += 1
                    for r, c in np.argwhere((prev == sym) & (cur != sym)):
                        out_of[(chr(cur[r, c]), int(col[r, c]))] += 1
                if f % 50 == 0:
                    want = col[cur == sym]
                    if want.size:
                        for ch in np.unique(cur[col == want[0]]):
                            same_col[(chr(ch), int(want[0]))] += 1
                prev, pcol = cur.copy(), col.copy()
            if remaining == 0:
                continue
            break
print(len(games), "games of", a.name)
print("became", a.sym, "from (symbol, colour):", into.most_common(12))
print("stopped being", a.sym, "into:", out_of.most_common(12))
print("symbols in its colour (every 50th frame):", same_col.most_common(15))
print("message rows then:", *msgs.most_common(15), sep="\n  ")
