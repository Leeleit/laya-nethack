"""Which held-out games carry the spread of G over seeds of one recipe (T-band loss, T inside the band cross-fitted over
players, as seed_g.py). For seed s, d_s = G_s - mean over seeds; game g's part of d_s is its rows' share of it, and its
share of the seed variance is sum_s d_s * part_gs / sum_s d_s^2 (the shares add up to 1 over games).
Also: G with every game weighing the same inside a cell; G on at most --cap rows per game and band (20 random draws,
mean SD); and cancellation over the 8 cells -- SD of G against the SD it would have if the cells' seed deviations were
independent (sqrt(sum of cell variances) / 8).
  python seed_games.py human_dwa30t --models c30e c30es1 c30es2 c30es3 [--without c30e] [--drop-top 1] [--out ...]
"""
import argparse
import bisect
import collections
from pathlib import Path

import numpy as np

from curve_eval import crossfit, labels, load_rows
from deep_checks import BANDS, NAMES
from seed_var import zs

QS = ("ttd", "depth", "end", "ascends")
DEEP = ("21-24", "25-29")


def read(a):
    import pandas as pd
    rows = load_rows(a.data)
    if a.drop_top:
        c = collections.Counter(r["player"] for r in rows if r["asc"])
        top = {p for p, _ in c.most_common(a.drop_top)}
        rows = [r for r in rows if r["player"] not in top]
    dfs = {m: pd.concat([pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{s}.parquet") for s in ("val", "test")])
           for m in a.models}
    cells = {}
    for q in QS:
        sel, y, _ = labels(rows, q)
        pl = np.array([rows[i]["player"] for i in sel])
        gm = np.array([rows[i]["id"].split("#")[0].lower() for i in sel])
        band = np.array([NAMES[bisect.bisect_left(BANDS, rows[i]["dlvl"])] for i in sel])
        z = {m: zs(dfs[m], rows, sel, q) for m in a.models}
        for name in DEEP:
            mk = band == name
            cells[q, name] = {"game": gm[mk], "loss": np.stack([crossfit(z[m][mk], y[mk], pl[mk])[0] for m in a.models])}
    return cells


def report(cells, models, idx, cap, label):
    L = {k: c["loss"][idx] for k, c in cells.items()}
    G = np.mean([v.mean(1) for v in L.values()], 0)
    d = G - G.mean()
    out = [f"\n== {label}: seeds {', '.join(models[i] for i in idx)}",
           "  G per seed " + " ".join(f"{x:.4f}" for x in G) + f"  SD {G.std(ddof=1):.4f}"]
    part = collections.defaultdict(lambda: np.zeros(len(idx)))
    nrows = collections.Counter()
    for k, c in cells.items():
        dev = L[k] - L[k].mean(0)
        n = dev.shape[1]
        for g in np.unique(c["game"]):
            mk = c["game"] == g
            part[g] += dev[:, mk].sum(1) / n / len(cells)
            nrows[g] += int(mk.sum())
    games = list(part)
    share = np.array([float(d @ part[g]) for g in games]) / float(d @ d)
    order = np.argsort(-np.abs(share))
    cum = np.cumsum(np.abs(share[order])) / np.abs(share).sum()
    k50 = int(np.searchsorted(cum, 0.5)) + 1
    rg = np.array([nrows[g] for g in games])
    out.append(f"  {len(games)} games, G rows per game median {np.median(rg):.0f}, p90 {np.percentile(rg, 90):.0f}, "
               f"max {rg.max()}; half of |variance share| in {k50} games ({rg[order[:k50]].sum() / rg.sum():.1%} of rows)")
    out.append("  top 10 games: " + "  ".join(f"{share[i]:+.3f} ({rg[i]} rows)" for i in order[:10]))
    out.append(f"  top 10 sum {share[order[:10]].sum():+.3f}, top 50 sum {share[order[:50]].sum():+.3f}")
    eq = []
    for k, c in cells.items():
        gs, inv = np.unique(c["game"], return_inverse=True)
        s = np.stack([np.bincount(inv, weights=L[k][j]) for j in range(len(idx))])
        eq.append((s / np.bincount(inv)).mean(1))
    Ge = np.mean(eq, 0)
    out.append("  equal weight per game: G " + " ".join(f"{x:.4f}" for x in Ge) + f"  SD {Ge.std(ddof=1):.4f}")
    rng = np.random.default_rng(0)
    sds = []
    for _ in range(20):
        gc = []
        for k, c in cells.items():
            keep = np.zeros(len(c["game"]), bool)
            for g, ix in pd_groups(c["game"]).items():
                keep[rng.permutation(ix)[:cap]] = True
            gc.append(L[k][:, keep].mean(1))
        sds.append(np.mean(gc, 0).std(ddof=1))
    out.append(f"  at most {cap} rows per game and band: SD of G over seeds, mean of 20 draws {np.mean(sds):.4f}")
    cv = np.array([v.mean(1) for v in L.values()])
    ind = np.sqrt(cv.var(1, ddof=1).sum()) / len(cv)
    out.append(f"  cells: SD per cell " + " ".join(f"{x:.4f}" for x in cv.std(1, ddof=1))
               + f"; SD of G {G.std(ddof=1):.4f} against independent cells {ind:.4f} (ratio {G.std(ddof=1) / ind:.2f})")
    return out


def pd_groups(g):
    out = collections.defaultdict(list)
    for i, x in enumerate(g):
        out[x].append(i)
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--without", nargs="*", default=[])
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--drop-top", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cells = read(a)
    out = [f"{a.data}: G cells {', '.join(QS)} x {', '.join(DEEP)}, T-band" + (", top player dropped" if a.drop_top else "")]
    out += report(cells, a.models, list(range(len(a.models))), a.cap, "all seeds")
    if a.without:
        idx = [i for i, m in enumerate(a.models) if m not in a.without]
        out += report(cells, a.models, idx, a.cap, f"without {', '.join(a.without)}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
