"""What the soft ttd / depth targets cost under the hard-label log loss, per Dlvl band: the mean -log of the target's own
mass on the true bin (a model that learns the target exactly pays this over a one-hot model), the target's mean mass
per bin against the hard labels' shares, and the same for the models' predictions; then each model's hard-label log
loss as is and smoothed by the target's kernel (p @ K, K[c] = the mean target of the rows whose true bin is c): a model
fit to hard labels, so smoothed, is roughly what it would predict had it learned the soft target.
  python soft_cost.py human_dwa20t [rate=c17r gbm=gbm-c17r ...] [--out ../results/soft_cost20.txt]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from curve_eval import BANDS, labels, load_rows, softmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("models", nargs="*", help="label=model (preds/<model>-<data>-<split>.parquet, raw logits)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = load_rows(a.data)
    out = []
    for q in ("ttd", "depth"):
        sel, y, k = labels(rows, q)
        soft = np.array([[rows[i]["gold"][q]["probabilities"].get(str(c), 0.0) for c in range(k)] for i in sel])
        dl = np.array([rows[i]["dlvl"] for i in sel])
        pred = {}
        for spec in a.models:
            lab, m = spec.split("=", 1)
            df = pd.concat([pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{s}.parquet") for s in ("val", "test")])
            lg = df[df.question == q].set_index("id").logits
            pred[lab] = softmax(np.stack([np.asarray(lg[rows[i]["id"]], float) for i in sel]))
        out.append(f"==== {q}: {len(y)} rows, {k} bins")
        kern = np.stack([soft[y == c].mean(0) if (y == c).any() else np.eye(k)[c] for c in range(k)])
        for lo, hi in BANDS:
            m = (dl >= lo) & (dl <= hi)
            if not m.any():
                continue
            own = -np.log(np.clip(soft[m, y[m]], 1e-12, 1))
            out.append(f"  Dlvl {lo}-{hi}: {m.sum()} rows; target's -log mass on the true bin {own.mean():.4f} "
                       f"(its mass there {soft[m, y[m]].mean():.3f})")
            out.append("    bin    hard  target  " + "  ".join(f"{lab:>6s}" for lab in pred))
            hard = np.bincount(y[m], minlength=k) / m.sum()
            for c in range(k):
                out.append(f"    {c:3d}  {hard[c]:6.3f}  {soft[m, c].mean():6.3f}  "
                           + "  ".join(f"{p[m, c].mean():6.3f}" for p in pred.values()))
            ll = lambda p: -np.log(np.clip(p[np.arange(len(p)), y[m]], 1e-12, 1)).mean()
            out.append("    hard-label log loss, raw / smoothed by K: "
                       + "  ".join(f"{lab} {ll(p[m]):.4f} / {ll(p[m] @ kern):.4f}" for lab, p in pred.items()))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
