"""Where and why C (--multi) loses to A on the humans' val: training loss curves, calibration with a
temperature fitted on other players (2-fold by player, 20 splits), sharpness, segments (Dlvl, turn, XL, label).
  python c_probe.py <preds.parquet> <val.jsonl> <train logs dir> [label=column_prefix ...]
"""
import re
import sys
import json
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from human_baseline import features

pq, val, logs = sys.argv[1:4]
# label=column prefix; the first one is the reference; default: the A / B / C run
pairs = sys.argv[4:] or ["A=eff-a-human_data5d-val", "B=eff-b-human_data5d-val", "C=eff-c-human_data5d-val"]
M = dict(p.split("=", 1) for p in pairs)
REF = next(iter(M))
M["gbm"] = "gbm"

print("== training loss (mean over questions, by tenth of the run)")
for m in M:
    try:
        ls = [float(x) for x in re.findall(r"loss ([\d.]+) \|", open(f"{logs}/eff-{m.lower()}.train.log").read())]
    except FileNotFoundError:
        continue
    tenths = [np.mean(c) for c in np.array_split(ls, 10)]
    print(f"  {m}: first window {ls[0]:.4f}; tenths " + " ".join(f"{t:.4f}" for t in tenths))

df = pd.read_parquet(pq)
st = {}
for line in open(val):
    r = json.loads(line)
    st[r["id"]] = features(r["state"])
F = pd.DataFrame.from_dict(st, orient="index")
df = df.join(F, on="id")
df["player"] = df.id.str.split("@").str[0]


def probs(g, m):
    cols = [c for c in g.columns if c.startswith(M[m] + "_") and g[c].notna().all()]
    return np.clip(g[cols].to_numpy(float), 1e-9, 1)


def scaled(p, t):
    z = np.log(p) / t
    z -= z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def nll(p, y):
    return -np.log(p[np.arange(len(y)), y])


def best_t(p, y):
    return minimize_scalar(lambda t: nll(scaled(p, t), y).mean(), bounds=(0.3, 5), method="bounded").x


def crossfit(p, y, players, n=20):
    """per-row loss with T fitted on the other half of the players, averaged over n random halvings"""
    out = np.zeros(len(y))
    for s in range(n):
        half = np.array([zlib.crc32(f"{s}:{x}".encode()) % 2 for x in players])
        for h in (0, 1):
            t = best_t(p[half != h], y[half != h])
            out[half == h] += nll(scaled(p[half == h], t), y[half == h])
    return out / n


def boot(d, players, n=1000, seed=0):
    codes, uniq = pd.factorize(players)
    rng = np.random.default_rng(seed)
    sums = np.bincount(codes, weights=d, minlength=len(uniq))
    cnts = np.bincount(codes, minlength=len(uniq))
    v = []
    for _ in range(n):
        k = rng.integers(0, len(uniq), len(uniq))
        v.append(sums[k].sum() / cnts[k].sum())
    return np.percentile(v, [2.5, 97.5])


for q, g in df.groupby("question"):
    g = g.reset_index(drop=True)
    y = g.label.to_numpy()
    pl = g.player.to_numpy()
    print(f"\n== {q}: {len(g)} rows, {g.player.nunique()} players")
    raw, cal = {}, {}
    for m in M:
        p = probs(g, m)
        raw[m], cal[m] = nll(p, y), crossfit(p, y, pl)
        ent = -(p * np.log(p)).sum(1).mean()
        print(f"  {m:4s} log loss {raw[m].mean():.4f} -> T on other players {cal[m].mean():.4f} (T on all "
              f"{best_t(p, y):.2f}); mean entropy {ent:.3f}, mean max p {p.max(1).mean():.3f}, "
              f"accuracy {(p.argmax(1) == y).mean():.3f}")
    others = [m for m in M if m not in (REF, "gbm")]
    for a, b in [(m, REF) for m in others] + [(m, "gbm") for m in [REF] + others]:
        for name, L in (("raw", raw), ("T", cal)):
            d = L[a] - L[b]
            lo, hi = boot(d, pl)
            print(f"  {a}-{b} {name:3s} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  (+ = {a} worse)")
    pr = probs(g, REF)
    for m in others:
        pm = probs(g, m)
        print(f"  {REF} vs {m}: total variation {np.abs(pr - pm).sum(1).mean() / 2:.3f}, same argmax "
              f"{(pr.argmax(1) == pm.argmax(1)).mean():.3f}")
    segs = {"Dlvl": pd.cut(g.dlvl, [0, 4, 10, 20, 99]), "turn": pd.cut(g.turn, [-2, 0, 5000, 15000, 30000, 10 ** 7]),
            "XL": pd.cut(g.xl, [0, 8, 14, 30]), "label": g.label}
    for m in others:
        for sname, s in segs.items():
            rows = []
            for k, idx in g.groupby(s, observed=True).groups.items():
                i = np.asarray(idx)
                rows.append(f"{k}: n {len(i)}, raw {(raw[m][i] - raw[REF][i]).mean():+.3f}, "
                            f"T {(cal[m][i] - cal[REF][i]).mean():+.3f}")
            print(f"  {m}-{REF} by {sname}: " + "; ".join(rows))
