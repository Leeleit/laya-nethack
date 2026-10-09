"""Raw screens of one human game (nao_corridor game list) around the first frames whose screen contains a substring:
each row with its length, to see what the extractor sees.
  python frame_dump.py <games.json> <name> <start> <substring> [--n 2] [--wide 160]"""
import argparse
import json

import numpy as np
from nle import _pyconverter as converter

ap = argparse.ArgumentParser()
ap.add_argument("games")
ap.add_argument("name")
ap.add_argument("start", type=int)
ap.add_argument("sub")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--wide", type=int, default=160)
ap.add_argument("--endswith", action="store_true", help="a row must end with the substring")
ap.add_argument("--at", help="FILE_PART:FRAME[,FRAME...]: those frames of the file whose name has FILE_PART "
                             "(inv_trace.py prints them); sub is then ignored")
a = ap.parse_args()
g = next(g for g in json.load(open(a.games)) if g["name"] == a.name and g["start"] == a.start)
n, W = 4096, a.wide
chars, colors = np.zeros((n, 24, W), np.uint8), np.zeros((n, 24, W), np.int8)
curs = np.zeros((n, 2), np.int16)
ts, acts, scores = np.zeros(n, np.int64), np.zeros(n, np.uint8), np.zeros(n, np.int32)
conv = converter.Converter(24, W, g.get("ttyrec_version", 1))
shown, sub = 0, a.sub.encode("latin-1")
for part, path in enumerate(g["files"]):
    conv.load_ttyrec(path, gameid=0, part=part)
    while shown < a.n:
        remaining = conv.convert(chars, colors, curs, ts, acts, scores)
        for f in range(n - remaining):
            if a.at:
                want, frames = a.at.rsplit(":", 1)
                hit = want in path and str(f) in frames.split(",")
            else:
                hit = any(chars[f, r].tobytes().rstrip().endswith(sub) if a.endswith else sub in chars[f, r].tobytes()
                          for r in range(24))
            if shown < a.n and hit:
                shown += 1
                print(f"==== {path} frame {f}")
                for r in range(24):
                    row = chars[f, r].tobytes().decode("latin-1").rstrip()
                    print(f"{r:2d} {len(row):3d} |{row}")
        if remaining == 0:
            continue
        break
    if shown >= a.n:
        break
