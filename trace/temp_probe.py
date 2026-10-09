"""probe: is a model's log loss on human_baseline.py's test rows a calibration problem? The best single temperature
on the test rows themselves (an optimistic diagnosis, not a fair score) and the mean confidence vs accuracy.
  python temp_probe.py <human_baseline parquet> <model column prefix>"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

df, name = pd.read_parquet(sys.argv[1]), sys.argv[2]
for q, g in df.groupby("question"):
    cols = sorted(c for c in g.columns if c.startswith(name + "_") and g[c].notna().all())
    p = np.clip(g[cols].to_numpy(float), 1e-9, 1)
    y = g.label.to_numpy()

    def ll(t):
        z = np.log(p) / t
        z -= z.max(1, keepdims=True)
        q_ = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return -np.log(q_[np.arange(len(y)), y]).mean()

    best = minimize_scalar(ll, bounds=(0.3, 5), method="bounded")
    print(f"{q}: log loss {ll(1):.4f} -> {best.fun:.4f} at T={best.x:.2f}; mean max prob {p.max(1).mean():.3f}, "
          f"accuracy {(p.argmax(1) == y).mean():.3f}; mean predicted {p.mean(0).round(3).tolist()} vs labels "
          f"{np.bincount(y, minlength=p.shape[1]) / len(y)}")
