"""States of a nao_corridor.py states file whose inventory has an item containing a substring: the game, turn, item
and the screen's message row.   python probe_find.py <states.jsonl> <substring> [--n 5]"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("states")
ap.add_argument("sub")
ap.add_argument("--n", type=int, default=5)
a = ap.parse_args()
shown, total = 0, 0
for line in open(a.states, encoding="utf-8"):
    g = json.loads(line)
    for s in g["states"]:
        hit = [it for it in s["inv"] if a.sub in it] + ([s["screen"][0]] if a.sub in s["screen"][0] else [])
        if hit:
            total += 1
            if shown < a.n:
                shown += 1
                print(g["name"], g["start"], s["turn"], hit, "| msg:", s["screen"][0][:150])
print("states:", total)
