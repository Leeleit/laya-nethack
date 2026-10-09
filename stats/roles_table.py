"""Ascension rate per role (and race within role) on the same cleaned sample as analyze.py, raw and by player
skill (prior ascensions), plus analyze.py's skill-adjusted odds ratios.  -> results/roles_table.txt"""
import json

import pandas as pd

from analyze import load, wilson

df = load()
R = json.load(open("/data/laya/nethack/stats/results.json"))
adj = R.get("model_ascend_by_role_adjusted", {})
lines = [f"{len(df)} games, {int(df.asc.sum())} ascensions"]


def table(keys, title):
    lines.append(f"\n== {title}")
    lines.append(f"{'':12s} {'games':>8s} {'asc %':>7s} {'95% CI':>15s} | {'0 prior':>7s} {'1-4':>6s} {'5-19':>6s} {'20+':>6s}")
    rows = []
    for k, x in df.groupby(keys, observed=True):
        if len(x) < 2000:
            continue
        k = k if isinstance(k, str) else "-".join(k)
        w = wilson(int(x.asc.sum()), len(x))
        by = {s: 100 * v.asc.mean() for s, v in x.groupby("skill", observed=True)}
        rows.append((w[0] if isinstance(w, (list, tuple)) else w, k, len(x), w, by))
    for _, k, n, w, by in sorted(rows, key=lambda r: -r[0]):
        p, lo, hi = (w if isinstance(w, (list, tuple)) else (w, w, w))[:3]
        lines.append(f"{k:12s} {n:8d} {100 * p:7.2f} {f'[{100 * lo:.2f}, {100 * hi:.2f}]':>15s} | "
                     + " ".join(f"{by.get(s, float('nan')):6.2f}" for s in ("0", "1-4", "5-19", "20+")))


table("role", "role")
table(["role", "race"], "role-race")
lines.append("\nskill-adjusted odds ratio of ascending vs Arc (analyze.py): " + json.dumps(adj)[:1500])
print("\n".join(lines))
open("/data/laya/nethack/stats/roles_table.txt", "w").write("\n".join(lines) + "\n")
