"""How human games end (xlogfile "death"), normalised: articles, "called/named <name>", ", while ..." cut -- to choose
the categories of the humans' "what ends this game" question.
  python death_kinds.py nao/val_dwa_any4.json [top]
"""
import collections
import json
import re
import sys

games = json.load(open(sys.argv[1]))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 60


def norm(d):
    d = d.split(", while ")[0]
    d = re.sub(r" (called|named) .*", "", d)
    d = re.sub(r"\b(an?|the) ", "", d)
    d = re.sub(r"\binvisible |\bhallucinogen-distorted ", "", d)
    return d


c = collections.Counter(norm(g["death"]) for g in games)
n = len(games)
cum = 0
for k, v in c.most_common(top):
    cum += v
    print(f"{v:6d} {100 * v / n:5.2f} % (cum {100 * cum / n:5.1f} %)  {k}")
print(f"{len(c)} distinct ends")
