"""Equal-share bins of the player's experience and result before a game (the bins must be even).
experience = n, the account's earlier NAO games; result = (k + a) / (n + a + b), k earlier ascensions, a, b the
beta-binomial prior fitted on train rows (k = 0 at 87 % of rows: raw k / n cannot be cut into equal shares, the
smoothed rate can -- 0 of 5 and 0 of 1000 differ). Edges = quantiles of train rows; per bin and split: share of
rows, distinct players, this game's ascended share with a Wilson 95 % interval.
  python skill_bins.py human_data15d ../results/skill_dist_15d [--bins 5 10] [--out ../results/skill_bins_15d.txt]
"""
import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln

GAMES = "/data/laya/nethack/data/games.parquet"
Z = 1.959964


def wilson(k, n):
    p = k / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return c - h, c + h


def load(data, ends, asc):
    out = {}
    for split in ("train", "val", "test"):
        n, k, y, p = [], [], [], []
        for row in open(Path(data, f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(row)
            name, start = r["id"].split("#", 1)[0].split("@", 1)
            name, start = name.lower(), float(start)
            n.append(bisect.bisect_left(ends.get(name, []), start))
            k.append(bisect.bisect_left(asc.get(name, []), start))
            y.append(r["gold"]["ascends"]["label"] == "true")
            p.append(name)
        out[split] = (np.array(n), np.array(k), np.array(y), np.array(p))
    return out


def fit_prior(n, k):
    """Beta-binomial maximum likelihood over rows with n >= 1 (log a, log b)."""
    m = n >= 1
    n, k = n[m].astype(float), k[m].astype(float)
    nll = lambda t: -(betaln(k + np.exp(t[0]), n - k + np.exp(t[1])) - betaln(np.exp(t[0]), np.exp(t[1]))).sum()  # noqa: E731
    t = minimize(nll, [np.log(0.5), np.log(50.0)], method="Nelder-Mead").x
    return np.exp(t[0]), np.exp(t[1])


def table(title, x, edges, sets):
    out = [title]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        cells = []
        for split, (xs, y, p) in sets.items():
            sel = (xs >= lo) & ((xs < hi) if i < len(edges) - 2 else (xs <= hi))
            if sel.sum():
                a, b = wilson(y[sel].sum(), sel.sum())
                cells.append(f"{split} {sel.mean():5.1%} pl {len(set(p[sel])):4d} asc {y[sel].mean():6.2%} "
                             f"[{a:.2%}, {b:.2%}]")
            else:
                cells.append(f"{split} empty")
        out.append(f"  {i + 1:2d} [{lo:.4g}, {hi:.4g}{']' if i == len(edges) - 2 else ')'}  " + " | ".join(cells))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("prefix", help="unused, kept for the call in the docstring")
    ap.add_argument("--bins", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = pd.read_parquet(GAMES, columns=["source", "name", "death", "endtime"])
    d = d[(d.source == "nao") & d.endtime.notna()]
    ends = {nm: np.sort(g.endtime.to_numpy()) for nm, g in d.groupby(d.name.str.lower())}
    asc = {nm: np.sort(g.endtime.to_numpy()) for nm, g in d[d.death == "ascended"].groupby(d.name.str.lower())}
    s = load(a.data, ends, asc)
    pa, pb = fit_prior(s["train"][0], s["train"][1])
    out = [f"beta-binomial prior on train: a {pa:.4f}, b {pb:.2f}, mean {pa / (pa + pb):.4%} (no earlier game -> this)"]
    exp = {sp: (v[0].astype(float), v[2], v[3]) for sp, v in s.items()}
    res = {sp: ((v[1] + pa) / (v[0] + pa + pb), v[2], v[3]) for sp, v in s.items()}
    for K in a.bins:
        q = np.linspace(0, 100, K + 1)
        e_exp = np.unique(np.percentile(exp["train"][0], q))
        e_res = np.unique(np.percentile(res["train"][0], q))
        out += table(f"== experience n, {K} equal-share bins on train ({len(e_exp) - 1} distinct; share of rows, "
                     f"players, this game ascended)", "n", e_exp, exp)
        out += table(f"== result (k + a) / (n + a + b), {K} equal-share bins on train ({len(e_res) - 1} distinct)",
                     "r", e_res, res)
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
