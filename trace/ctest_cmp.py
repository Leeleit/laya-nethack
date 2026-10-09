"""collect.py resume check: results of a collection interrupted and resumed vs the same seeds in runs/val1 (image before patch_arena.py)."""
import json
import sys

o, ref = sys.argv[1], sys.argv[2]
a = {r["trajectory_id"]: r for r in json.load(open(f"{o}/results.json"))}
b = {r["trajectory_id"]: r for r in json.load(open(f"{ref}/results.json"))}
k = ("status", "steps", "turns", "max_depth", "progress", "cause_of_death")
for s in sorted(a):
    d = [(x, a[s][x], b[s][x]) for x in k if a[s][x] != b[s][x]]
    print(s, "same" if not d else f"DIFF {d}")
print("lines in partial:", sum(1 for _ in open(f"{o}/results.partial.jsonl")))
