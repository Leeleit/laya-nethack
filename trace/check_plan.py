"""One-off: plan_branches.py --points N reproduces the earlier one-point plans (plans/points.parquet)."""
import sys

import pandas as pd

a = pd.read_parquet(sys.argv[1] + "/points.parquet")
b = pd.read_parquet("/data/laya/nethack/trace/plans/points.parquet")
m = a[a.point == 0].merge(b, on=["run", "seed"])
print(f"new {len(a)} points ({a[a.point == 0].shape[0]} first), old {len(b)}, matched {len(m)}, "
      f"same k {(m.k_x == m.k_y).mean():.3f}")
