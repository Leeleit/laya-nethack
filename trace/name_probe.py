"""probe: does a model's loss on human states depend on the player's name being in the state text (memorising
players)? Splits predict.py rows by "name in the state" and prints log loss / accuracy per question.
  python name_probe.py <cases.jsonl> <preds.parquet>"""
import json
import sys

import numpy as np
import pandas as pd

rows = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    named = r["id"].split("@")[0] in r["state"]
    for q, g in r["gold"].items():
        rows[(r["id"], q)] = (int(g["label"]), named)
p = pd.read_parquet(sys.argv[2])
out = []
for i, q, lg in zip(p.id, p.question, p.logits):
    if (i, q) in rows:
        z = np.asarray(lg, float)
        z = np.exp(z - z.max())
        z /= z.sum()
        y, named = rows[(i, q)]
        out.append(dict(q=q, named=named, ll=-np.log(max(z[y], 1e-12)), acc=float(z.argmax() == y)))
d = pd.DataFrame(out)
print(d.groupby(["q", "named"]).agg(n=("ll", "size"), log_loss=("ll", "mean"), accuracy=("acc", "mean")).round(3))
