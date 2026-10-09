"""Raw log loss / accuracy of several predict.py outputs on the same cases (only the rows all of them have), against
the cases' gold labels -- a quick check of one checkpoint against another on a task (no calibration, no baselines).
  python logloss_cmp.py <cases.jsonl> name=preds.parquet ...
"""
import json
import sys

import numpy as np
import pandas as pd

gold = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    for q, g in r["gold"].items():
        gold[(r["id"], q)] = int(np.argmax([g["probabilities"].get(str(i), 0) for i in range(len(g["probabilities"]))]))
preds = {}
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    p = pd.read_parquet(path)
    if "question" not in p:  # older predict.py output: one question per case
        only = dict(gold)  # (id, q) -> label; the case's one question
        qid = {i: q for i, q in only}
        p["question"] = [qid.get(i) for i in p.id]
    preds[name] = {(i, q): np.asarray(l, float) for i, q, l in zip(p.id, p.question, p.logits) if (i, q) in gold}
common = set.intersection(*(set(v) for v in preds.values()))
for question in sorted({q for _, q in common}):
    keys = sorted(k for k in common if k[1] == question)
    y = np.array([gold[k] for k in keys])
    print(f"{question}: {len(keys)} common cases; labels {np.bincount(y).tolist()}")
    for name, v in preds.items():
        z = np.stack([v[k] for k in keys])
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
        print(f"  {name:16s} log loss {ll.mean():.4f}  accuracy {(p.argmax(1) == y).mean():.3f}  "
              f"mean predicted {p.mean(0).round(3).tolist()}")
