"""Ensemble of saved predictions: per row and question, the mean of the models' softmax probabilities, written back
as logits (log of the mean) so deep_checks.py / seed_avg.py read it as one more model.
  python ens_preds.py <data name> <out model> <model> <model> [...]
  -> preds/<out>-<data>-{val,test}.parquet
"""
import sys

import numpy as np
import pandas as pd


def main():
    data, out, models = sys.argv[1], sys.argv[2], sys.argv[3:]
    assert len(models) >= 2
    for s in ("val", "test"):
        dfs = [pd.read_parquet(f"preds/{m}-{data}-{s}.parquet") for m in models]
        base = dfs[0][["id", "question"]].reset_index(drop=True)
        for d in dfs[1:]:  # same rows in the same order, or the mean mixes different questions
            assert (d.id.to_numpy() == base.id.to_numpy()).all() and (d.question.to_numpy() == base.question.to_numpy()).all()
        logits = []
        for row in zip(*(d.logits.to_numpy() for d in dfs)):
            z = np.stack([np.asarray(x, float) for x in row])
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            logits.append(np.log(p.mean(0)))
        base["logits"] = logits
        base.to_parquet(f"preds/{out}-{data}-{s}.parquet")
        print(f"{s}: {len(base)} rows -> preds/{out}-{data}-{s}.parquet")


if __name__ == "__main__":
    main()
