"""v6's hand-picked parts that can be counted without training laya. Tabular proxy: the features of
human_baseline.py, GBM / a small MLP -- what the labels carry, not what laya would learn.
  1. masses of v6's ttd / depth bins on train (forced bounds vs equal mass)
  2. ttd: number of bins K (equal mass in log(1+turns left), v5's 1000 / 5000 forced in): GBM on hard labels;
     val scores comparable across K -- density log score in log(1+turns left) (mass of a bin spread over its width)
     and log loss folded into v5's three classes (<=1000, 1001-5000, >5000)
  3. ttd: soft targets, sigma / bin width 0 / 0.25 / 0.75 / 2 at K = 19 (v6) and 64: MLP (GBM takes hard labels only)
  4. end: share of "another monster", mutual information of end and v5's ttd; joint (end x ttd3) GBM vs the product of
     separate ones on val
Raw turns left and levels gained are inverted from the soft targets: cumulative mass up to the true bin k is
Phi((hi_k - x) / sigma) (outer bins open, so no renormalisation) -> x = hi_k - sigma * Phi^-1(sum p_0..k).
  python bins_proxy.py <human_data6d dir>
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtri
from sklearn.ensemble import HistGradientBoostingClassifier

from human_baseline import features

D = sys.argv[1]
meta = json.load(open(f"{D}/meta.json"))["v6"]
TB, DB = meta["ttd_bounds"], meta["depth_gain_bounds"]


def edges(b, log):
    hi = [np.log1p(x) if log else x + 0.5 for x in b]
    return np.array([-np.inf] + hi), np.array(hi + [np.inf])


def invert(p, k, b, log):
    lo, hi = edges(b, log)
    n = len(b)
    if log:
        w = np.where((k > 0) & (k < n), hi[np.minimum(k, n)] - lo[k], np.where(k == 0, hi[1] - lo[1], hi[-2] - lo[-2]))
    else:
        w = np.where((k > 0) & (k < n), hi[np.minimum(k, n)] - lo[k], 1.0)
    # from the larger tail (precise: the targets are differences of cdfs, the far bins underflow to 0):
    # mass above bin k = Phi((x - hi_k) / sigma), mass below = Phi((lo_k - x) / sigma)
    c = np.cumsum(p, 1)
    i = np.arange(len(k))
    below = np.where(k > 0, c[i, np.maximum(k - 1, 0)], 0.0)
    above = np.where(k < n, p.sum(1) - c[i, k], 0.0)
    s = 0.75 * w
    up = hi[np.minimum(k, n)] + s * ndtri(np.clip(above, 1e-300, 1))
    down = lo[k] - s * ndtri(np.clip(below, 1e-300, 1))
    return np.where(above >= below, up, down)


def load(split):
    rows = [json.loads(line) for line in open(f"{D}/{split}.jsonl")]
    f = pd.DataFrame([features(r["state"]) for r in rows]).astype(float)
    out = pd.DataFrame({"player": [r["id"].split("@")[0] for r in rows]})
    for q, b, log in (("ttd", TB, True), ("depth", DB, False)):
        has = np.array([q in r["gold"] for r in rows])
        p = np.zeros((len(rows), len(b) + 1))
        k = np.zeros(len(rows), int)
        for i, r in enumerate(rows):
            if has[i]:
                p[i] = [r["gold"][q]["probabilities"][str(j)] for j in range(len(b) + 1)]
                k[i] = int(r["gold"][q]["label"])
        x = np.full(len(rows), np.nan)
        x[has] = invert(p[has], k[has], b, log)
        out[q] = x
        out[q + "_k"] = np.where(has, k, -1)
    out["end"] = [r["gold"]["end"]["label"] for r in rows]
    return f, out


t0 = time.time()
Ftr, Ytr = load("train")
Fva, Yva = load("val")
print(f"loaded {len(Ytr)} / {len(Yva)} rows in {time.time() - t0:.0f}s")
# the inversion must give back the true bin
for q, b, log in (("ttd", TB, True), ("depth", DB, False)):
    lo, hi = edges(b, log)
    m = Ytr[q].notna()
    k = np.searchsorted(hi, Ytr.loc[m, q].to_numpy(), side="left")
    print(f"inversion {q}: bin recovered for {(k == Ytr.loc[m, q + '_k']).mean():.4f} of {m.sum()} rows")
Ytr["depth"] = Ytr.depth.round()
Yva["depth"] = Yva.depth.round()

print("\n== 1. bin masses on train")
for q in ("ttd", "depth"):
    s = Ytr[Ytr[q + "_k"] >= 0][q + "_k"].value_counts(normalize=True).sort_index()
    print(f"  {q}: " + " ".join(f"{v:.3f}" for v in s))

LMAX = Ytr.ttd.max()  # ttd here is already log(1 + turns left)


def bounds_k(k):
    """equal mass in log1p on train with v5's 1000 / 5000 forced in (no bound within 15 % of them)"""
    v = Ytr.ttd.dropna().to_numpy()
    lq = np.quantile(v, np.arange(1, k) / k)
    keep = [x for x in lq if min(abs(x - np.log1p(1000)), abs(x - np.log1p(5000))) > 0.15]
    return np.unique(np.r_[keep, np.log1p(1000), np.log1p(5000)])


def scores(p, x, hi):
    """density log score in log1p (outer bins closed at 0 and the train maximum) and v5's 3-class log loss"""
    lo_e = np.r_[0.0, hi]
    hi_e = np.r_[hi, LMAX]
    w = np.maximum(hi_e - lo_e, 1e-6)
    k = np.searchsorted(hi, x, side="left")
    p = np.clip(p, 1e-9, 1)
    dens = -(np.log(p[np.arange(len(k)), k]) - np.log(w[k])).mean()
    v5 = np.searchsorted([np.log1p(1000), np.log1p(5000)], hi_e - 1e-9, side="left")  # coarse class of each bin
    p3 = np.stack([p[:, v5 == c].sum(1) for c in range(3)], 1)
    k3 = np.searchsorted([np.log1p(1000), np.log1p(5000)], x, side="left")
    return dens, -np.log(np.clip(p3[np.arange(len(k3)), k3], 1e-9, 1)).mean()


mtr, mva = Ytr.ttd.notna().to_numpy(), Yva.ttd.notna().to_numpy()
xtr, xva = Ytr.ttd[mtr].to_numpy(), Yva.ttd[mva].to_numpy()
Xtr, Xva = Ftr[mtr].to_numpy(), Fva[mva].to_numpy()


def gbm(X, y):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True,
                                          validation_fraction=0.1, random_state=0).fit(X, y)


print("\n== 2. ttd, number of bins (GBM, hard labels): density log score (lower better), v5 3-class log loss")
for k in (3, 6, 12, 19, 32, 64):
    hi = np.array([np.log1p(1000), np.log1p(5000)]) if k == 3 else bounds_k(k)
    if k == 19:
        hi = np.log1p(np.array(TB, float))  # v6's own bounds
    t1 = time.time()
    ytr = np.searchsorted(hi, xtr, side="left")
    g = gbm(Xtr, ytr)
    p = np.zeros((len(xva), len(hi) + 1))
    p[:, g.classes_] = g.predict_proba(Xva)
    d, l3 = scores(p, xva, hi)
    print(f"  K {len(hi) + 1:3d}: density {d:.4f}  v5-ttd log loss {l3:.4f}  ({time.time() - t1:.0f}s)", flush=True)

print("\n== 3. ttd, soft targets (MLP, 5 seeds): sigma = r x the true bin's width")
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Ttr, Tva = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor((Xva - mu) / sd, dtype=torch.float32)


def soft_targets(x, hi, r):
    k = np.searchsorted(hi, x, side="left")
    lo_e, hi_e = np.r_[-np.inf, hi], np.r_[hi, np.inf]
    w_in = np.r_[hi[1] - hi[0], np.diff(hi), hi[-1] - hi[-2]]
    if r == 0:
        t = np.zeros((len(x), len(hi) + 1))
        t[np.arange(len(x)), k] = 1
        return t
    s = (r * w_in[k])[:, None]
    from scipy.special import ndtr
    t = ndtr((hi_e[None] - x[:, None]) / s) - ndtr((lo_e[None] - x[:, None]) / s)
    return t / t.sum(1, keepdims=True)


def mlp(T, X, Xv, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                              torch.nn.ReLU(), torch.nn.Linear(256, T.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    T = torch.tensor(T, dtype=torch.float32)
    for ep in range(15):
        perm = torch.randperm(len(X))
        for j in range(0, len(X), 512):
            i = perm[j:j + 512]
            loss = -(T[i] * torch.log_softmax(net(X[i]), -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return torch.softmax(net(Xv), -1).numpy()


torch.set_num_threads(8)
for k in (19, 64):
    hi = np.log1p(np.array(TB, float)) if k == 19 else bounds_k(64)
    for r in (0, 0.25, 0.75, 2.0):
        t1 = time.time()
        res = [scores(mlp(soft_targets(xtr, hi, r), Ttr, Tva, s), xva, hi) for s in range(5)]
        d, l3 = np.mean(res, 0)
        sd_d, sd_l = np.std(res, 0)
        print(f"  K {len(hi) + 1:3d} r {r:4.2f}: density {d:.4f} (+-{sd_d:.4f})  v5-ttd log loss {l3:.4f} "
              f"(+-{sd_l:.4f})  ({time.time() - t1:.0f}s)", flush=True)

print("\n== 4. end")
sh = Ytr.end.value_counts(normalize=True)
print(f"  'another monster' {sh.get('monster', 0):.3f}, 'something else' {sh.get('other', 0):.3f}; "
      f"entropy of end {-(sh * np.log2(sh)).sum():.2f} bits ({len(sh)} options)")
k3tr = np.searchsorted([np.log1p(1000), np.log1p(5000)], Ytr.ttd.to_numpy(), side="left")
ct = pd.crosstab(Ytr.end[mtr], k3tr[mtr], normalize=True).to_numpy()
pe, pt = ct.sum(1, keepdims=True), ct.sum(0, keepdims=True)
mi = np.nansum(ct * np.log2(ct / (pe * pt)))
print(f"  mutual information end x v5-ttd (train labels): {mi:.3f} bits")
codes = {c: i for i, c in enumerate(sorted(Ytr.end.unique()))}
etr = Ytr.end.map(codes).to_numpy()[mtr]
eva = Yva.end.map(codes).fillna(-1).astype(int).to_numpy()[mva]
k3va = np.searchsorted([np.log1p(1000), np.log1p(5000)], xva, side="left")
ok = eva >= 0
ne = len(codes)
ge, gt, gj = gbm(Xtr, etr), gbm(Xtr, k3tr[mtr]), gbm(Xtr, etr * 3 + k3tr[mtr])


def pr(g, X, n):
    p = np.full((len(X), n), 1e-9)
    p[:, g.classes_] = g.predict_proba(X)
    return p / p.sum(1, keepdims=True)


pe_, pt_, pj_ = pr(ge, Xva, ne), pr(gt, Xva, 3), pr(gj, Xva, ne * 3)
sep = -(np.log(pe_[np.arange(len(eva)), np.maximum(eva, 0)]) + np.log(pt_[np.arange(len(k3va)), k3va]))[ok].mean()
joint = -np.log(pj_[np.arange(len(eva)), np.maximum(eva, 0) * 3 + k3va])[ok].mean()
print(f"  val, end + v5-ttd: separate GBMs {sep:.4f}, one joint GBM {joint:.4f} nats ({ok.sum()} rows)")
