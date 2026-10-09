"""How much a trained risk model leans on the maps: its test cases with the level map cut out ("nolevel") or with both
the level map and the crop around the hero cut out ("nomaps"), predicted by the same checkpoint, against its
predictions on the full text. The model was trained with the maps, so this overstates what a model trained without
them would lose -- an upper bound of the maps' worth, not a retrained comparison.
  python map_ablation.py make <cases.jsonl> <out prefix>      -> <prefix>_nolevel.jsonl, <prefix>_nomaps.jsonl
  python map_ablation.py score <cases.jsonl> name=preds.parquet ...   (log loss, accuracy, AUC of P(dies <= 30 turns))
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LEVEL = re.compile(r"\nLevel map:\n.*\Z", re.S)
CROP = re.compile(r"\nMap around the hero \(@\):\n.*?(?=\n[A-Z][A-Za-z ()]*: )", re.S)


def make(cases, prefix):
    with open(f"{prefix}_nolevel.jsonl", "w", encoding="utf-8", newline="\n") as a, \
            open(f"{prefix}_nomaps.jsonl", "w", encoding="utf-8", newline="\n") as b:
        for line in open(cases, encoding="utf-8"):
            r = json.loads(line)
            st = LEVEL.sub("", r["state"])
            a.write(json.dumps(dict(r, state=st)) + "\n")
            b.write(json.dumps(dict(r, state=CROP.sub("", st))) + "\n")


def score(cases, preds):
    gold = {}
    for line in open(cases, encoding="utf-8"):
        r = json.loads(line)
        for q, g in r["gold"].items():
            gold[(r["id"], q)] = int(g["label"])
    loaded = {}
    for arg in preds:
        name, path = arg.split("=", 1)
        p = pd.read_parquet(path)
        if "question" not in p:
            p["question"] = "ttd"
        loaded[name] = {(i, q): np.asarray(l, float) for i, q, l in zip(p.id, p.question, p.logits) if (i, q) in gold}
    keys = sorted(set.intersection(*(set(v) for v in loaded.values())))
    y = np.array([gold[k] for k in keys])
    print(f"{len(keys)} common cases; labels {np.bincount(y).tolist()}")
    for name, v in loaded.items():
        z = np.stack([v[k] for k in keys])
        z -= z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
        auc = roc_auc_score(y <= 2, p[:, :3].sum(1))
        print(f"  {name:10s} log loss {ll:.4f}  accuracy {(p.argmax(1) == y).mean():.3f}  "
              f"AUC P(dies <= 30 turns) {auc:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2], sys.argv[3:])
