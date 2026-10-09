"""end: the deaths end_kind files as "monster" -- a killer name not in nle's monster table. The commonest.
  python end_unmapped.py nao/val_dwa_states2.jsonl
"""
import collections
import json
import sys

from human_q6 import end_kind, outcome

c = collections.Counter()
n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    g = json.loads(line)
    if not g["states"]:
        continue
    k = min(2, len(g["states"]))
    n += k
    if end_kind(g["death"]) == "monster":
        c[outcome(g, g["states"][0])["killer"]] += k
print(f"{sum(c.values()) / n:.1%} of {n} rows")
for name, k in c.most_common(40):
    print(f"  {k / n:.2%}  {name}")
