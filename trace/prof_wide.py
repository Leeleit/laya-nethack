"""Time of nao_corridor.game_states on the first games of a list at two terminal widths (80 vs WIDE), and a profile
of the wide run: where the extractor spends its time.   python prof_wide.py <games.json> [--n 8]"""
import argparse
import cProfile
import json
import pstats
import random
import time

import nao_corridor as nc

ap = argparse.ArgumentParser()
ap.add_argument("games")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()
games = json.load(open(a.games))
random.Random(0).shuffle(games)  # the order nao_corridor states takes them in
games = games[:a.n]
wide = nc.WIDE
for w in (80, wide):
    nc.WIDE = w
    t = time.time()
    out = [nc.game_states(g) for g in games]
    st = sum(len(o["states"]) for o in out)
    print("errors:", [o["error"] for o in out if "error" in o])
    print(f"width {w}: {time.time() - t:.1f} s, {st} states")
nc.WIDE = wide
pr = cProfile.Profile()
pr.enable()
for g in games:
    nc.game_states(g)
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
