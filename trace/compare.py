"""Risk models side by side on the same held-out risk states (build_laya_data.py's test games).

  python compare.py name=preds.parquet[:rebalanced] ...

Tabular baselines on the blstats/strategy features: logistic regression and gradient boosting
(early-stopped on held-out *training games*, not rows). A ':rebalanced' laya run was trained on
build_laya_data's NEG_RATE negative sampling; its odds are multiplied back by NEG_RATE.
Metrics on all test states and on slices, with 95% intervals from a bootstrap over test games;
dAUC is the paired difference to the logit on the same resampled games.
-> /data/laya/nethack/trace/results/compare.json
"""
import json
import sys
import zlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

D = "/data/laya/nethack/trace/laya_data"
OUT = "/data/laya/nethack/trace/results/compare.json"
BOOT = 200


def auc(y, p):
    n1 = y.sum()
    if n1 == 0 or n1 == len(y):
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def metrics(y, p):
    # rank metrics on the raw p: clipping first ties every p below 1e-6 (most of an uncalibrated
    # model's raw outputs) and cost risk-v1 ~0.09 AUC in an earlier version of this function
    base = y.mean()
    brier = np.mean((p - y) ** 2)
    c = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": auc(y, p), "ap": float(average_precision_score(y, p)) if y.any() else np.nan,
            "brier_skill": float(1 - brier / (base * (1 - base))),
            "log_loss": float(-np.mean(y * np.log(c) + (1 - y) * np.log(1 - c))),
            "mean_p": float(p.mean()), "rate": float(base)}


def design(f, top):
    X = pd.DataFrame({"hp_frac": f.hp_frac, "hp_drop10": f.hp_drop10, "log_maxhp": np.log1p(f.maxhp),
                      "depth": f.depth, "xl": f.xl, "ac": f.ac, "hunger": f.hunger, "stack": f["stack"]})
    s = pd.Categorical(f.strategy.where(f.strategy.isin(top), "other"), categories=sorted(top) + ["other"])
    r = pd.Categorical(f.identity.str.slice(0, 3), categories=sorted(ROLES))
    return pd.concat([X, pd.get_dummies(s, prefix="s", drop_first=True, dtype=float).set_index(X.index),
                      pd.get_dummies(r, prefix="r", drop_first=True, dtype=float).set_index(X.index)],
                     axis=1).astype(float)


def main():
    global ROLES
    meta = json.load(open(f"{D}/meta.json"))
    f = pd.read_parquet(f"{D}/features.parquet")
    f["game"] = f.id.str.split("#").str[0]
    ROLES = set(f.identity.str.slice(0, 3))
    tr, te = f[f.split == "train"], f[f.split == "test"].reset_index(drop=True)
    top = list(tr.strategy.value_counts().index[:15])
    Xtr, Xte = design(tr, top), design(te, top)
    preds = {}

    glm = sm.GLM(tr.y, sm.add_constant(Xtr), family=sm.families.Binomial()).fit()
    preds["logit"] = glm.predict(sm.add_constant(Xte, has_constant="add")).to_numpy()

    games = tr.game.unique()
    val_games = set(np.random.default_rng(1).choice(games, len(games) // 5, replace=False))
    v = tr.game.isin(val_games).to_numpy()
    gbm = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=False, random_state=0)
    gbm.fit(Xtr[~v], tr.y[~v])
    yv = tr.y[v].to_numpy()
    losses = [-np.mean(yv * np.log(np.clip(p[:, 1], 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - p[:, 1], 1e-6, 1)))
              for p in gbm.staged_predict_proba(Xtr[v])]
    best = int(np.argmin(losses))
    preds["gbm"] = list(gbm.staged_predict_proba(Xte))[best][:, 1]

    laya_names = []
    for arg in sys.argv[1:]:
        name, spec = arg.split("=", 1)
        path, _, mode = spec.partition(":")
        pr = te[["id"]].merge(pd.read_parquet(path), on="id", how="left")
        if "logits" in pr:  # predict.py: raw [false, true] logits, no temperature
            z = np.stack(pr["logits"].to_numpy())
            pr["p"] = 1 / (1 + np.exp(z[:, 0] - z[:, 1]))
        p = pr["p"].to_numpy()
        assert not np.isnan(p).any(), f"{name}: predictions missing for some test ids"
        if mode == "rebalanced":
            odds = np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)) * meta["neg_rate"]
            p = odds / (1 + odds)
        preds[name] = p
        laya_names.append(name)

    # Does laya add anything the tabular model lacks? Stack logit-scale inputs with a logistic regression,
    # cross-fitted over test games (fit on 4/5 of them, predict the rest). 'logit(recal)' is the same with
    # the tabular model alone, so a stack gain is new information, not just recalibration.
    fold = te.game.map(lambda k: zlib.crc32(k.encode()) % 5).to_numpy()
    lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
    for name, cols in [("logit(recal)", ["logit"])] + [(f"logit+{n}", ["logit", n]) for n in laya_names]:
        Z = sm.add_constant(np.column_stack([lg(preds[c]) for c in cols]))
        p = np.empty(len(te))
        for k in range(5):
            fit = sm.GLM(te.y[fold != k], Z[fold != k], family=sm.families.Binomial()).fit()
            p[fold == k] = fit.predict(Z[fold == k])
        preds[name] = p

    slices = {"all": np.ones(len(te), bool), "depth>4": (te.depth > 4).to_numpy(),
              "fight2": (te.strategy == "Agent.fight2").to_numpy(), "hp<0.5": (te.hp_frac < 0.5).to_numpy()}
    y = te.y.to_numpy()
    g = te.game.to_numpy()
    ug = np.unique(g)
    idx_by_game = {k: np.flatnonzero(g == k) for k in ug}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx_by_game[k] for k in rng.choice(ug, len(ug))]) for _ in range(BOOT)]

    R = {"meta": meta, "test_rows": len(te), "test_games": len(ug), "gbm_best_iter": best + 1, "slices": {}}
    for sname, m in slices.items():
        out = {"rows": int(m.sum()), "positives": int(y[m].sum())}
        for name, p in preds.items():
            point = metrics(y[m], p[m])
            bs = [metrics(y[b][m[b]], p[b][m[b]]) for b in boots]
            out[name] = {k: [round(point[k], 4)] + [round(float(np.nanpercentile([x[k] for x in bs], q)), 4) for q in (2.5, 97.5)]
                         if k in ("auc", "ap", "brier_skill") else round(point[k], 5) for k in point}
            if name != "logit":  # paired: the same resampled games for both models
                d = [auc(y[b][m[b]], p[b][m[b]]) - auc(y[b][m[b]], preds["logit"][b][m[b]]) for b in boots]
                out[name]["d_auc_vs_logit"] = [round(point["auc"] - out["logit"]["auc"][0], 4)] + \
                    [round(float(np.nanpercentile(d, q)), 4) for q in (2.5, 97.5)]
        R["slices"][sname] = out
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=1)
    for sname, out in R["slices"].items():
        print(f"== {sname}: {out['rows']} states, {out['positives']} positives")
        for name in preds:
            o = out[name]
            print(f"  {name:16s} AUC {o['auc']}  dAUC {o.get('d_auc_vs_logit', '-')}  AP {o['ap']}  BSS {o['brier_skill']}  "
                  f"logloss {o['log_loss']}  mean_p {o['mean_p']}")


if __name__ == "__main__":
    sys.exit(main())
