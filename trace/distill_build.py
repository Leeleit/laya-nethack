"""A data set whose train targets mix the old target with the predictions of teacher models (the mean of their softmax
probabilities): gold.probabilities = alpha * teacher + (1 - alpha) * old target, per question, in the option order
finetune_fast._target reads them. Labels, states, questions, val and test stay as they are.
  python distill_build.py <data dir> <out dir> --alpha 0.5 --teachers preds/c17e-human_dwa24-train.parquet ...
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def keys_of(q):
    """Option keys in the order of the logits (finetune_fast._target)."""
    t, crit = q["type"], q.get("criteria", {})
    if t == "choice":
        return list(crit)
    if t == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(crit) if isinstance(crit, list) else 4)]


def teacher_probs(paths):
    dfs = [pd.read_parquet(p) for p in paths]
    base = dfs[0]
    for d in dfs[1:]:
        assert (d.id.to_numpy() == base.id.to_numpy()).all() and (d.question.to_numpy() == base.question.to_numpy()).all()
    out = {}
    for k, row in enumerate(zip(*(d.logits.to_numpy() for d in dfs))):
        z = np.stack([np.asarray(x, float) for x in row])
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        out[(base.id.iat[k], base.question.iat[k])] = p.mean(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--teachers", nargs="+", required=True)
    a = ap.parse_args()
    src, dst = Path(a.data), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    tp = teacher_probs(a.teachers)
    n = l1 = 0
    with open(src / "train.jsonl") as f, open(dst / "train.jsonl", "w", newline="\n") as g:
        for line in f:
            r = json.loads(line)
            for qid, gq in r["gold"].items():
                ks = keys_of(r["questions"][qid])
                old = np.array([gq["probabilities"].get(k, 0.0) for k in ks], float)
                old = old / old.sum() if old.sum() > 0 else np.full(len(ks), 1 / len(ks))
                p = tp[(r["id"], qid)]
                assert len(p) == len(ks), (r["id"], qid, len(p), len(ks))
                new = a.alpha * p + (1 - a.alpha) * old
                gq["probabilities"] = {k: float(v) for k, v in zip(ks, new)}
                n += 1
                l1 += float(np.abs(new - old).sum())
            g.write(json.dumps(r) + "\n")
    assert n == len(tp), f"{n} targets rewritten, {len(tp)} teacher predictions"
    for s in ("val.jsonl", "test.jsonl"):
        if (src / s).exists():
            shutil.copyfile(src / s, dst / s)
    meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
    meta["distill"] = {"from": str(src), "alpha": a.alpha, "teachers": a.teachers}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1), newline="\n")
    print(f"{n} targets, mean L1 change {l1 / n:.4f} -> {dst}")


if __name__ == "__main__":
    main()
