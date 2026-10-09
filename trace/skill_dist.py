"""Distribution of the player's experience and result before each game (for the strength in the question).
Per row of a dataset: n = the account's NAO games (any role, games.parquet) that ended before this game started,
k = of them ascended; result = k / n with a Wilson 95 % interval. Per split: experience n and log10(n + 1), result
(n >= 1): mean, std, 95 % CI of the mean (normal), quantiles; the Wilson interval's width; this game's ascended share
by quintiles of experience, and by result (k = 0 split by experience, k > 0 by quintiles of k / n), with 95 % CIs.
Train's n, k, y -> <out>.npz (skill_plot.py draws them).
  python skill_dist.py human_data15d [--out ../results/skill_dist_15d]
"""
import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd

GAMES = "/data/laya/nethack/data/games.parquet"
Z = 1.959964


def wilson(k, n):
    p = k / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return c - h, c + h


def describe(name, x):
    m, s = x.mean(), x.std(ddof=1)
    q = np.percentile(x, [1, 10, 25, 50, 75, 90, 99])
    return (f"  {name:16s} n {len(x):6d}  mean {m:9.4f}  std {s:9.4f}  95% CI of mean [{m - Z * s / np.sqrt(len(x)):.4f}, "
            f"{m + Z * s / np.sqrt(len(x)):.4f}]\n  {'':16s} p1 {q[0]:.4g}  p10 {q[1]:.4g}  p25 {q[2]:.4g}  p50 {q[3]:.4g}  "
            f"p75 {q[4]:.4g}  p90 {q[5]:.4g}  p99 {q[6]:.4g}")


def line(label, y):
    a, b = wilson(y.sum(), len(y))
    return f"    {label:28s} rows {len(y):6d}  {y.mean():6.2%}  [{a:.2%}, {b:.2%}]"


def by_quantiles(name, x, y, k=5):
    edges = np.unique(np.percentile(x, np.linspace(0, 100, k + 1)))
    out = [f"  this game ascended, by {name} (equal-share bins; Wilson 95 %)"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & ((x < hi) | (hi == edges[-1]))
        if sel.sum():
            out.append(line(f"[{lo:.4g}, {hi:.4g}{']' if hi == edges[-1] else ')'}", y[sel]))
    return out


def result_bins(k, n, y):
    out = ["  this game ascended, by result (Wilson 95 %)", line("no earlier game", y[n == 0])]
    for lo, hi in ((1, 9), (10, 99), (100, 999), (1000, 10 ** 9)):
        sel = (k == 0) & (n >= lo) & (n <= hi)
        if sel.sum():
            out.append(line(f"k = 0, n {lo}-{hi if hi < 10 ** 9 else 'max'}", y[sel]))
    pos = k > 0
    out.append(f"    k > 0: {pos.mean():.2%} of rows; by quintiles of k / n among them:")
    out += by_quantiles("k / n", k[pos] / n[pos], y[pos])[1:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = pd.read_parquet(GAMES, columns=["source", "name", "death", "endtime"])
    d = d[(d.source == "nao") & d.endtime.notna()]
    ends, asc = {}, {}
    for name, g in d.groupby(d.name.str.lower()):
        ends[name] = np.sort(g.endtime.to_numpy())
        asc[name] = np.sort(g.endtime[g.death == "ascended"].to_numpy())
    out = []
    for split in ("train", "val", "test"):
        n, k, y = [], [], []
        for row in open(Path(a.data, f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(row)
            name, start = r["id"].split("#", 1)[0].split("@", 1)
            name, start = name.lower(), float(start)
            n.append(bisect.bisect_left(ends.get(name, []), start))
            k.append(bisect.bisect_left(asc.get(name, []), start))
            y.append(r["gold"]["ascends"]["label"] == "true")
        n, k, y = np.array(n), np.array(k), np.array(y)
        has = n >= 1
        lo, hi = wilson(k[has], n[has])
        out += [f"== {split}: {len(n)} rows, no earlier game {(~has).mean():.2%}, k > 0 {(k > 0).mean():.2%}",
                describe("experience n", n.astype(float)), describe("log10(n + 1)", np.log10(n + 1.0)),
                describe("result k/n", k[has] / n[has]), describe("Wilson width", hi - lo),
                *by_quantiles("experience log10(n+1)", np.log10(n + 1.0), y), *result_bins(k, n, y)]
        if split == "train" and a.out:
            np.savez(a.out + ".npz", n=n, k=k, y=y)
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out + ".txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
