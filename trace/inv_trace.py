"""How a human game's inventory was built: every inventory menu nao_corridor.game_states committed (its classes, its
letters, whether the screen edge cut it), then the states' inventories.
  python inv_trace.py <games.json> <name> [--start S]   (needs nle)"""
import argparse
import json

import nao_corridor as nc

ap = argparse.ArgumentParser()
ap.add_argument("games")
ap.add_argument("name")
ap.add_argument("--start", type=int)
ap.add_argument("--frames", action="store_true", help="every menu frame too")
a = ap.parse_args()
for g in json.load(open(a.games)):
    if g["name"] == a.name and (a.start is None or g["start"] == a.start):
        print("########", g["name"], g["start"])
        r = nc.game_states(dict(g, debug=2 if a.frames else 1))
        for s in r["states"]:
            print(f"STATE T{s['turn']}: {len(s['inv'])} items: {' | '.join(s['inv'])[:400]}")
