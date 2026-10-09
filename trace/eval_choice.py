"""How good are the decisions? Exact off-policy evaluation on the held-out branch points.

Every test point has the outcome of every option that ran there (build_choice.py), so a policy restricted
to those options is evaluated exactly: its survival rate is the mean of the outcomes it picks. Policies:
  continue  -- the bot as it is (the recorded game)
  always X  -- X wherever it ran, else continue
  logit     -- per-option logistic regression on the tabular features of the point (train points)
  <model>   -- argmax of a laya checkpoint's P(alive | option) (predict.py logits, noul [false, true])
  oracle    -- the best outcome among the options (the ceiling)
Paired bootstrap over test points for the difference to continue.
  python eval_choice.py --data D --plans P --choice C [name=preds.parquet ...]   -> C/eval.txt
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from plan_branches import NUM

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="build_v2.py output (features.parquet)")
ap.add_argument("--plans", required=True)
ap.add_argument("--choice", required=True, help="build_choice.py output")
ap.add_argument("preds", nargs="*", help="name=predict.py parquet on <choice>/test.jsonl")
a = ap.parse_args()

f = pd.read_parquet(f"{a.data}/features.parquet")
pts = pd.read_parquet(f"{a.plans}/points.parquet")
oc = pd.read_parquet(f"{a.choice}/outcomes.parquet")
K = ["run", "seed", "point"]
feat = f.set_index("id")[NUM].astype(float)
pts["x"] = [feat.loc[f"{r}/{fl}#{t}"].to_numpy() for r, fl, t in zip(pts.run, pts.file, pts.turn)]
X = pts.set_index(K).x
oc = oc.join(X, on=K)
opts = sorted(oc.option.unique())
alive = oc.pivot_table(index=K, columns="option", values="alive", aggfunc="first")
split = oc.drop_duplicates(K).set_index(K).split
test = alive[split.reindex(alive.index) == "test"]

scores = {}  # policy -> per-option score table on the test points (higher = pick)
for o in opts:
    if o != "continue":
        scores[f"always {o}"] = pd.DataFrame({c: (1.0 if c == o else 0.5 if c == "continue" else 0.0)
                                              for c in opts}, index=test.index)
xt = np.stack(X.reindex(test.index))
cols = {}
for o in opts:
    tr = oc[(oc.option == o) & (oc.split == "train")]
    cols[o] = LogisticRegression(max_iter=2000).fit(np.stack(tr.x), tr.alive.astype(int)).predict_proba(xt)[:, 1]
scores["logit"] = pd.DataFrame(cols, index=test.index)
for arg in a.preds:
    name, path = arg.split("=", 1)
    p = pd.read_parquet(path)
    parts = p.id.str.split("#")
    p["run"], p["seed"], p["point"] = parts.str[0], parts.str[1].astype(int), parts.str[2].astype(int)
    z = np.stack(p.logits)
    p["p"] = 1 / (1 + np.exp(z[:, 0] - z[:, 1]))
    scores[name] = p.pivot_table(index=K, columns="question", values="p").reindex(test.index)


def realized(score):
    s = score.reindex(columns=test.columns).where(test.notna())  # only options that ran at the point
    pick = s.idxmax(axis=1)
    return np.array([test.at[i, c] for i, c in zip(test.index, pick)], float), pick


rng = np.random.default_rng(0)
boots = [rng.integers(0, len(test), len(test)) for _ in range(1000)]
base = test["continue"].to_numpy(float)
best = test.max(axis=1).to_numpy(float)
lines = [f"test points {len(test)}; options ran: " + ", ".join(f"{o} {int(test[o].notna().sum())}" for o in opts),
         f"{'continue':18s} alive after the horizon {base.mean():.3f}",
         f"{'oracle':18s} {best.mean():.3f}  (+{best.mean() - base.mean():.3f})"]
for name, sc in scores.items():
    r, pick = realized(sc)
    d = [r[b].mean() - base[b].mean() for b in boots]
    lines.append(f"{name:18s} {r.mean():.3f}  diff {r.mean() - base.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, "
                 f"{np.percentile(d, 97.5):+.3f}]  picks {pick.value_counts().to_dict()}")
print("\n".join(lines))
open(f"{a.choice}/eval.txt", "w").write("\n".join(lines) + "\n")
