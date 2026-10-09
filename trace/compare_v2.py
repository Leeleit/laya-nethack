"""risk-v2 against tabular models, per horizon h: P(hero dies within h turns), h = the bucket edges.

  python compare_v2.py name=<val logits.parquet>,<test logits.parquet> ...

Everything fitted outside the test games:
  - tabular baselines (logistic regression, gradient boosting early-stopped on val games) per horizon,
    on blstats/strategy features plus the v2 extras (HP 30 turns ago, hostile monsters in view);
  - laya's calibration: bias-corrected temperature scaling (Alexandari et al. 2020) on the val games,
    started from meta's bucket_weights (the training sample's correction);
  - stacks 'logit+<name>' and the control 'logit(recal)': logistic regressions on logit-scale inputs,
    fitted on val, so a stack gain is information the tabular model lacks, not recalibration.
Test metrics per horizon and slice; 95% intervals and paired dAUC (vs the logit) from a bootstrap
over test games. -> /data/laya/nethack/trace/results/compare_v2.json
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

from compare import BOOT, auc, metrics

D = "/data/laya/nethack/trace/v2_data"
OUT = os.environ.get("COMPARE_OUT", "/data/laya/nethack/trace/results/compare_v2.json")


def design(f, top, roles):
    X = pd.DataFrame({"hp_frac": f.hp_frac, "hp_drop10": f.hp_drop10, "hp_drop30": f.hp_drop30,
                      "log_maxhp": np.log1p(f.maxhp), "depth": f.depth, "xl": f.xl, "ac": f.ac,
                      "hunger": f.hunger, "stack": f["stack"], "hostiles": f.hostiles,
                      "hostiles_adjacent": f.hostiles_adjacent})
    s = pd.Categorical(f.strategy.where(f.strategy.isin(top), "other"), categories=sorted(top) + ["other"])
    r = pd.Categorical(f.identity.str.slice(0, 3), categories=roles)
    return pd.concat([X, pd.get_dummies(s, prefix="s", drop_first=True, dtype=float).set_index(X.index),
                      pd.get_dummies(r, prefix="r", drop_first=True, dtype=float).set_index(X.index)],
                     axis=1).astype(float)


def lg(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


_BOOT = None  # (boots, yt, P, slices) of the current horizon; set before the pool forks


def boot_one(i):
    """Metrics of every (slice, model) on bootstrap resample i, plus each model's AUC gap to logit."""
    boots, yt, P, slices = _BOOT
    b = boots[i]
    out = {}
    for sname, m in slices.items():
        sel = b[m[b]]
        y = yt[sel]
        a0 = auc(y, P["logit"][1][sel])
        for name, (_, p) in P.items():
            x = metrics(y, p[sel])
            out[sname, name] = (x, x["auc"] - a0)
    return out


def bucket_probs(logits, T, bias):
    z = logits / T + bias
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def fit_bcts(logits, y, w):
    """Bias-corrected temperature scaling (Alexandari et al., ICML 2020): one temperature plus a bias
    per bucket, by log loss on held-out games -- the label-shift fix for a model trained at other
    bucket rates. Starts from the sampling correction log(w); bucket 0's bias is fixed at log(w[0])."""

    def unpack(v):
        return np.exp(v[0]), np.concatenate([[np.log(w[0])], v[1:]])

    def nll(v):
        T, b = unpack(v)
        return -np.mean(np.log(bucket_probs(logits, T, b)[np.arange(len(y)), y] + 1e-12))

    v = minimize(nll, np.concatenate([[0.0], np.log(w[1:])]), method="L-BFGS-B").x
    T, b = unpack(v)
    return T, b, nll(v), nll(np.concatenate([[0.0], np.log(w[1:])]))


def main():
    meta = json.load(open(f"{D}/meta.json"))
    edges, w = meta["edges"], np.array(meta["bucket_weights"])
    f = pd.read_parquet(f"{D}/features.parquet")
    f["game"] = f.id.str.split("#").str[0]
    tr, va, te = (f[f.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    top = list(tr.strategy.value_counts().index[:15])
    roles = sorted(set(f.identity.str.slice(0, 3)))
    X = {s: design(d, top, roles) for s, d in (("train", tr), ("val", va), ("test", te))}

    laya = {}  # name -> {split: bucket probabilities aligned with the split's feature rows}
    R = {"meta": meta, "rows": {s: len(d) for s, d in (("train", tr), ("val", va), ("test", te))},
         "calibration": {}, "horizons": {}}
    for arg in sys.argv[1:]:
        name, paths = arg.split("=", 1)
        z = {}
        for split, path, d in zip(("val", "test"), paths.split(","), (va, te)):
            lp = d[["id"]].merge(pd.read_parquet(path), on="id", how="left")["logits"]
            assert lp.notna().all(), f"{name}: logits missing for some {split} ids"
            z[split] = np.stack(lp.to_numpy())
        yb = va.bucket.to_numpy()
        T, b, nll_fit, nll_weights_only = fit_bcts(z["val"], yb, w)
        R["calibration"][name] = {"T": round(float(T), 4), "bias": [round(float(x), 3) for x in b],
                                  "val_nll": round(float(nll_fit), 5), "val_nll_weights_only": round(float(nll_weights_only), 5)}
        laya[name] = {s: bucket_probs(z[s], T, b) for s in z}

    g = te.game.to_numpy()
    ug = np.unique(g)
    by_game = {k: np.flatnonzero(g == k) for k in ug}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([by_game[k] for k in rng.choice(ug, len(ug))]) for _ in range(BOOT)]
    slices = {"all": np.ones(len(te), bool), "depth>4": (te.depth > 4).to_numpy(),
              "fight2": (te.strategy == "Agent.fight2").to_numpy(), "hp<0.5": (te.hp_frac < 0.5).to_numpy()}

    for hi, h in enumerate(edges):
        y = {s: (d.ttd <= h).astype(int).to_numpy() for s, d in (("train", tr), ("val", va), ("test", te))}
        P = {}  # name -> (val, test) probabilities
        glm = sm.GLM(y["train"], sm.add_constant(X["train"]), family=sm.families.Binomial()).fit()
        P["logit"] = [glm.predict(sm.add_constant(X[s], has_constant="add")).to_numpy() for s in ("val", "test")]
        gbm = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, min_samples_leaf=200,
                                             l2_regularization=1.0, early_stopping=False, random_state=0)
        gbm.fit(X["train"], y["train"])
        yv = y["val"]
        losses = [-np.mean(yv * np.log(np.clip(p[:, 1], 1e-6, 1)) + (1 - yv) * np.log(np.clip(1 - p[:, 1], 1e-6, 1)))
                  for p in gbm.staged_predict_proba(X["val"])]
        best = int(np.argmin(losses))
        P["gbm"] = [list(gbm.staged_predict_proba(X[s]))[best][:, 1] for s in ("val", "test")]
        for name, bp in laya.items():
            P[name] = [bp[s][:, :hi + 1].sum(1) for s in ("val", "test")]
        for name, cols in [("logit(recal)", ["logit"])] + [(f"logit+{n}", ["logit", n]) for n in laya]:
            Zv, Zt = (sm.add_constant(np.column_stack([lg(P[c][i]) for c in cols])) for i in (0, 1))
            fit = sm.GLM(yv, Zv, family=sm.families.Binomial()).fit()
            P[name] = [fit.predict(Zv), fit.predict(Zt)]

        yt = y["test"]
        out = {"gbm_best_iter": best + 1}
        # the bootstrap is independent per resample and single-threaded numpy/sklearn: fan it out over
        # processes (fork shares the arrays copy-on-write); half the cores
        global _BOOT
        _BOOT = (boots, yt, P, slices)
        with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 2) // 2)) as pool:
            res = pool.map(boot_one, range(len(boots)))
        for sname, m in slices.items():
            o = {"rows": int(m.sum()), "positives": int(yt[m].sum())}
            for name, (_, p) in P.items():
                point = metrics(yt[m], p[m])
                bs = [r[sname, name][0] for r in res]
                o[name] = {k: [round(point[k], 4)] + [round(float(np.nanpercentile([x[k] for x in bs], q)), 4)
                                                     for q in (2.5, 97.5)] if k in ("auc", "ap", "brier_skill")
                           else round(point[k], 5) for k in point}
                if name != "logit":
                    d = [r[sname, name][1] for r in res]
                    o[name]["d_auc_vs_logit"] = [round(point["auc"] - o["logit"]["auc"][0], 4)] + \
                        [round(float(np.nanpercentile(d, q)), 4) for q in (2.5, 97.5)]
            out[sname] = o
        R["horizons"][str(h)] = out

    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=1)
    print("calibration", R["calibration"], "rows", R["rows"])
    for h, out in R["horizons"].items():
        for sname in slices:
            o = out[sname]
            print(f"== h={h} {sname}: {o['rows']} states, {o['positives']} positives")
            for name in o:
                if isinstance(o[name], dict):
                    x = o[name]
                    print(f"  {name:16s} AUC {x['auc']}  dAUC {x.get('d_auc_vs_logit', '-')}  AP {x['ap'][0]}  "
                          f"BSS {x['brier_skill'][0]}  mean_p {x['mean_p']} rate {x['rate']}")


if __name__ == "__main__":
    sys.exit(main())
