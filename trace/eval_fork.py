"""How good are the decisions, on forkrun.py data (K outcomes per option at every test point)?

A policy picks one option per test point among the options that ran there; its value at the point is that
option's survival over the K reps (an unbiased estimate of its expected survival), the policy's value the mean
over points. Paired bootstrap over games (a game's points resampled together) for the difference to continue.
Policies:
  continue, always X        -- the bot, and X wherever it ran
  logit / gbm               -- per-option tabular models of survival (train points, rep means as weights)
  <name>=preds              -- argmax of a laya checkpoint (predict.py on test.jsonl: noul per option, or
                               test_choice.jsonl: one choice question "best")
  oracle (in-sample)        -- the best rep mean: inflated, it also picks rep noise
  oracle (split reps)       -- picks by the even reps, scored on the odd reps: an honest ceiling for the
                               options at this K (scored on half the reps, so noisier)
  python eval_fork.py --data D --plans P --choice C [--horizon 300] [name=preds.parquet ...]  -> C/eval.txt
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from build_fork import OPTIONS
from plan_branches import NUM

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="build_v2.py output (features.parquet)")
ap.add_argument("--plans", required=True)
ap.add_argument("--choice", required=True, help="build_fork.py output")
ap.add_argument("--horizon", type=int, default=300)
ap.add_argument("preds", nargs="*", help="name=predict.py parquet")
a = ap.parse_args()

K = ["run", "seed", "point"]
col = f"alive_{a.horizon}"
oc = pd.read_parquet(f"{a.choice}/outcomes.parquet")
f = pd.read_parquet(f"{a.data}/features.parquet")
pts = pd.read_parquet(f"{a.plans}/points.parquet")
feat = f.set_index("id")[NUM].astype(float)
pts["x"] = [feat.loc[f"{r}/{fl}#{t}"].to_numpy() for r, fl, t in zip(pts.run, pts.file, pts.turn)]
X = pts.set_index(K).x
mean = oc.pivot_table(index=K, columns="option", values=col, aggfunc="mean")
split = oc.drop_duplicates(K).set_index(K).split.reindex(mean.index)
opts = [o for o in OPTIONS if o in mean]
test = mean[split == "test"][opts]

# honest oracle: choose on even reps, score on odd reps (continue's rep 0 is the parent)
half = {p: oc[oc.rep % 2 == p].pivot_table(index=K, columns="option", values=col, aggfunc="mean")
        .reindex(index=test.index, columns=opts) for p in (0, 1)}

scores = {}
for o in opts[1:]:
    scores[f"always {o}"] = pd.DataFrame({c: (1.0 if c == o else 0.5 if c == "continue" else 0.0) for c in opts},
                                         index=test.index)
xt = np.stack(X.reindex(test.index))
for name, make in (("logit", lambda: LogisticRegression(max_iter=2000)),
                   ("gbm", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05))):
    cols = {}
    for o in opts:
        tr = oc[(oc.option == o) & (oc.split == "train")].join(X, on=K)
        y = tr[col].astype(int)
        # a rare option (upstairs, dig_down) can have too few or one-class rows: its train mean
        cols[o] = make().fit(np.stack(tr.x), y).predict_proba(xt)[:, 1] if len(tr) >= 50 and y.nunique() == 2 \
            else np.full(len(xt), y.mean() if len(tr) else 0.0)
    scores[name] = pd.DataFrame(cols, index=test.index)
for arg in a.preds:
    name, path = arg.split("=", 1)
    p = pd.read_parquet(path)
    parts = p.id.str.split("#")
    p["run"], p["seed"], p["point"] = parts.str[0], parts.str[1].astype(int), parts.str[2].astype(int)
    if (p.question == "best").all():  # one choice question: logits over the point's options in OPTIONS order
        rows = {}
        for (r, s, i), lg in zip(zip(p.run, p.seed, p.point), p.logits):  # (as many logits as options there)
            if (r, s, i) not in test.index:
                continue
            ran = [o for o in opts if pd.notna(test.at[(r, s, i), o])]
            assert len(ran) == len(lg), ((r, s, i), ran, len(lg))
            rows[r, s, i] = dict(zip(ran, lg))
        scores[name] = pd.DataFrame.from_dict(rows, orient="index").reindex(test.index)
    else:
        z = np.stack(p.logits)
        p["p"] = 1 / (1 + np.exp(z[:, 0] - z[:, 1]))
        scores[name] = p.pivot_table(index=K, columns="question", values="p").reindex(test.index)


def realized(score, value=test):
    s = score.reindex(columns=opts).where(test.notna())  # only options that ran at the point
    pick = s.idxmax(axis=1)
    return np.array([value.at[i, c] for i, c in zip(test.index, pick)], float), pick


rng = np.random.default_rng(0)
# resample games, not points: a game's points (up to 3) share its hero and dungeon, so a bootstrap over points
# gave intervals too narrow
game = pd.factorize(pd.Series(list(zip(test.index.get_level_values("run"), test.index.get_level_values("seed")))))[0]
members = [np.flatnonzero(game == g) for g in range(game.max() + 1)]
boots = [np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))]) for _ in range(1000)]
base = test["continue"].to_numpy(float)


def line(name, r, pick=None):
    ok = ~np.isnan(r)
    d = [np.nanmean(r[b]) - np.nanmean(base[b]) for b in boots]
    s = f"{name:22s} {np.nanmean(r):.4f}  diff {np.nanmean(r) - base[ok].mean():+.4f} " \
        f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
    return s + (f"  picks {pick.value_counts().to_dict()}" if pick is not None else "")


reps = oc[oc.split == "test"].groupby(K + ["option"]).size()
lines = [f"horizon {a.horizon}; test points {len(test)}; reps per option: median {int(reps.median())}; options ran: "
         + ", ".join(f"{o} {int(test[o].notna().sum())}" for o in opts),
         line("continue", base),
         line("oracle (in-sample)", test.max(axis=1).to_numpy(float))]
r_half, _ = realized(half[0].fillna(-1), half[1])
lines.append(line("oracle (split reps)", r_half) + "  (vs continue on the same odd reps: "
             f"{np.nanmean(r_half) - np.nanmean(half[1]['continue'].to_numpy(float)):+.4f})")
for name, sc in scores.items():
    r, pick = realized(sc)
    lines.append(line(name, r, pick))
print("\n".join(lines))
open(f"{a.choice}/eval_h{a.horizon}.txt", "w").write("\n".join(lines) + "\n")
