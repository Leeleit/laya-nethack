"""Which signature has a model's calibration map (CPU, from ready predictions on the train rows)? Per question, one
cell per (Dlvl band, option): d = mean model - observed share, against
  pool  = share over all bands - share in this band (the model leaning to the whole set's distribution),
  blur  = mean old target - observed share (the soft target's spread, learnt as it is),
  mode  = observed share (big options shrunk, small ones raised);
least squares over the cells weighted by the band's rows, each alone and all three; R2 and coefficients.
  python cal_reg.py human_dwa24/train.jsonl --tag human_dwa24-train --models c17e c17es1 c17es2 c17es3 \
      --out ../results/cal_reg24.txt"""
import argparse
from pathlib import Path

import numpy as np

from curve_eval import QS
from leak_total import NAMES, load, probs


def wls(x, y, w):
    x = np.column_stack([np.ones(len(y)), x])
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)[0]
    res = y - x @ beta
    r2 = 1 - (w * res ** 2).sum() / (w * (y - np.average(y, weights=w)) ** 2).sum()
    return beta, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d, _, _ = load(a.data, targets=True)
    out = [f"{' '.join(a.data)}; mean of {' '.join(a.models)}; cells (band x option), d = mean model - observed share"]
    for q in QS:
        t = d[q]
        k = t["old"].shape[1]
        p = np.mean([probs([f"preds/{m}-{x}.parquet" for x in a.tag], q, t["ids"]) for m in a.models], 0)
        yh = np.eye(k)[t["y"]]
        allshare = yh.mean(0)
        rows = []
        for b in range(6):
            kb = t["bands"] == b
            if kb.sum() < 50:
                continue
            obs, tgt, mod = yh[kb].mean(0), t["old"][kb].mean(0), p[kb].mean(0)
            for j in range(k):
                rows.append((b, j, kb.sum(), mod[j] - obs[j], allshare[j] - obs[j], tgt[j] - obs[j], obs[j]))
        r = np.array(rows, float)
        w, y = r[:, 2], r[:, 3]
        out.append(f"\n==== {q}: {len(r)} cells, weighted mean |d| {np.average(np.abs(y), weights=w):.4f}")
        for name, cols in (("pool", [4]), ("blur", [5]), ("mode", [6]), ("all three", [4, 5, 6])):
            beta, r2 = wls(r[:, cols], y, w)
            out.append(f"  {name:<10} R2 {r2:6.3f}   coef " + " ".join(f"{c:+.3f}" for c in beta[1:]))
        for b in range(6):
            m = r[:, 0] == b
            if m.any():
                beta, r2 = wls(r[m][:, [4, 5, 6]], y[m], w[m])
                out.append(f"  band {NAMES[b]:<6} |d| {np.abs(y[m]).mean():.4f}  all three R2 {r2:6.3f}  pool / blur / mode "
                           + " ".join(f"{c:+.3f}" for c in beta[1:]))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
