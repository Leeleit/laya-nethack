"""A human game's ttyrec frame times against its xlogfile start / end (do the listed files hold frames of other
games?): per file, the first and last frame time, how many frames fall before start / after end, and the status
line's turn at the first and last frame inside the game.
  python ts_probe.py <games.json> <name> <start>   (needs nle)"""
import argparse
import json
import re

import numpy as np
from nle import _pyconverter as converter

ap = argparse.ArgumentParser()
ap.add_argument("games")
ap.add_argument("name")
ap.add_argument("start", type=int)
a = ap.parse_args()
g = next(g for g in json.load(open(a.games)) if g["name"] == a.name and g["start"] == a.start)
print("xlog start", g["start"], "end", g["end"], "turns", g["turns"])
T = re.compile(rb"T:(\d+)")
n = 4096
chars, colors = np.zeros((n, 24, 80), np.uint8), np.zeros((n, 24, 80), np.int8)
curs = np.zeros((n, 2), np.int16)
ts, acts, scores = np.zeros(n, np.int64), np.zeros(n, np.uint8), np.zeros(n, np.int32)
conv = converter.Converter(24, 80, g.get("ttyrec_version", 1))
for part, path in enumerate(g["files"]):
    conv.load_ttyrec(path, gameid=0, part=part)
    first = last = None
    before = after = inside = 0
    turns = []
    while True:
        remaining = conv.convert(chars, colors, curs, ts, acts, scores)
        for f in range(n - remaining):
            t = int(ts[f])
            first = t if first is None else first
            last = t
            sec = t / 1e6 if t > 1e12 else t
            if sec < g["start"]:
                before += 1
            elif sec > g["end"]:
                after += 1
            else:
                inside += 1
                m = T.search(chars[f, 23].tobytes())
                if m:
                    turns.append(int(m.group(1)))
        if remaining == 0:
            continue
        break
    print(path.rsplit("/", 1)[-1], "ts", first, "..", last, f"before {before} inside {inside} after {after}",
          "turns", turns[:1], turns[-1:])
