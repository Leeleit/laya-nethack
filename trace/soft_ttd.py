"""Which soft ttd target, judged on the hard labels by cross-validation over players on train (as soft_variants.py --cv
for depth): the GBM of human_tab.py fit to each candidate target (a row per bin, weighted by the target's mass), its
log loss per Dlvl band (raw and with T fitted inside the band, cross-fitted over players), the difference to orig and to
the hard-label fit with a 95 % bootstrap over players and the SE over the folds.
  hard    the one-hot label
  orig    the set's own target: N(log(1 + left), 0.75 x the true bin's width in log(1 + left)) (human_q6.py)
  log75   N(log(1 + left), s) with one s for all rows: 0.75 x the median width of the inner bins
  log375  the same with 0.375 x the median width
The turns left are recovered from each row's stored target (the value in the true bin whose orig target is closest).
  python soft_ttd.py human_dwa24 --cv 5 [--out ../results/soft_ttd24cv.txt]
"""
import argparse
import collections
import json
import math
import os
import re
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from curve_eval import BANDS, boot_idx, ci, crossfit
from human_tab import load

VARIANTS = ("hard", "orig", "log75", "log375")


def edges(texts):
    """bin edges in log(1 + turns left) from the option texts: 'at most N', 'a-b', 'over N'"""
    b = [int(re.search(r"(\d+)$", t).group(1)) for t in texts[:-1]]
    assert texts[-1] == f"over {b[-1]}", texts[-1]
    lo = np.array([-math.inf] + [math.log1p(x) for x in b])
    hi = np.array([math.log1p(x) for x in b] + [math.inf])
    return lo, hi


def masses(lo, hi, x, s):
    """mass of N(x, s) on each bin; x and s broadcast over leading axes"""
    x, s = np.asarray(x)[..., None], np.asarray(s)[..., None]
    return ndtr((hi - x) / s) - ndtr((lo - x) / s)


def orig_sigma(lo, hi, k):
    """0.75 x the width of bin k; the outer bins take their inner neighbour's width (human_q6.py)"""
    n = len(lo)
    w = hi - lo
    return 0.75 * np.where(k == 0, w[1], np.where(k == n - 1, w[n - 2], w[np.clip(k, 1, n - 2)]))


def recover(lo, hi, soft, k):
    """log(1 + left) inside bin k whose orig target is closest to the stored one: two passes of a 64-point grid"""
    n = soft.shape[1]
    s = orig_sigma(lo, hi, k)
    a = np.where(k == 0, 0.0, lo[k])
    b = np.where(k == n - 1, math.log1p(3e6), hi[k])
    for _ in range(2):
        g = a[:, None] + (b - a)[:, None] * np.linspace(0, 1, 64)[None, :]
        err = np.abs(masses(lo, hi, g, s[:, None]) - soft[:, None, :]).sum(-1)
        j = err.argmin(1)
        step = (b - a) / 63
        x = g[np.arange(len(g)), j]
        a, b = np.maximum(x - step, np.where(k == 0, 0.0, lo[k])), np.minimum(x + step, np.where(k == n - 1, math.log1p(3e6), hi[k]))
    return x, err[np.arange(len(err)), j]


def fit(job):
    name, xt, rows, w, xh, threads, K = job
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True, random_state=0)
    m.fit(xt[rows[:, 0]], rows[:, 1], sample_weight=w)
    p = np.full((len(xh), K), 1e-9)
    p[:, m.classes_] = np.clip(m.predict_proba(xh), 1e-9, 1)
    return name, p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tr = [json.loads(x) for x in open(Path(a.train, "train.jsonl"), encoding="utf-8")]
    tr = [r for r in tr if "ttd" in r["gold"]][: a.max_train or None]
    texts = tr[0]["questions"]["ttd"]["criteria"]
    assert all(r["questions"]["ttd"]["criteria"] == texts for r in tr)
    K = len(texts)
    lo, hi = edges(texts)
    soft = np.array([[r["gold"]["ttd"]["probabilities"].get(str(c), 0.0) for c in range(K)] for r in tr])
    y = np.array([int(r["gold"]["ttd"]["label"]) for r in tr])
    x, err = recover(lo, hi, soft, y)
    s75 = 0.75 * float(np.median((hi - lo)[1:-1]))
    targets = {"hard": np.eye(K)[y], "orig": masses(lo, hi, x, orig_sigma(lo, hi, y)),
               "log75": masses(lo, hi, x, np.full(len(x), s75)), "log375": masses(lo, hi, x, np.full(len(x), s75 / 2))}
    out = [f"{len(tr)} train rows; recovered log(1 + left): |orig(x) - stored target| median {np.median(err):.2e}, "
           f"p99 {np.percentile(err, 99):.3f}, max {err.max():.3f}; label of the recovered value = stored label in "
           f"{(np.searchsorted(hi, x) == y).mean():.4f}; log sigma 0.75 / 0.375 x median inner width: {s75:.3f} / "
           f"{s75 / 2:.3f}; orig sigma per bin " + " ".join(f"{v:.2f}" for v in orig_sigma(lo, hi, np.arange(K)))]
    X, _ = load(Path(a.train, "train.jsonl"))
    xt = X.loc[[r["id"] for r in tr]].to_numpy()
    pl = np.array([r["id"].split("@")[0].lower() for r in tr])
    dl = np.array([int(re.search(r"\bDlvl (\d+)", r["state"]).group(1)) for r in tr])
    fold = np.array([zlib.crc32(p.encode()) % a.cv for p in pl])
    out[0] += f"; {a.cv} folds by player, scored on train"
    jobs = []
    for v in VARIANTS:
        t = targets[v]
        out.append(f"  {v:7s}: mean mass per bin over train " + " ".join(f"{q:.3f}" for q in t.mean(0)))
        for f in range(a.cv):
            tm = fold != f
            j, c = np.nonzero(t[tm] > 1e-3)
            jobs.append(((v, f), xt[tm], np.stack([j, c], 1), t[tm][j, c], xt[~tm], a.threads, K))
    with Pool(a.workers) as pool:
        parts = dict(pool.imap_unordered(fit, jobs))
    prob = {}
    for v in VARIANTS:
        prob[v] = np.zeros((len(y), K))
        for f in range(a.cv):
            prob[v][fold == f] = parts[v, f]
    raw = {v: -np.log(prob[v][np.arange(len(y)), y]) for v in VARIANTS}
    agg = collections.defaultdict(list)  # (variant, ref) -> per band from Dlvl 5, the fold means of the T-band difference
    for blo, bhi in BANDS:
        m = (dl >= blo) & (dl <= bhi)
        if not m.any():
            continue
        b = boot_idx(pl[m])
        tb = {v: crossfit(np.log(prob[v][m]), y[m], pl[m]) for v in VARIANTS}
        out.append(f"\nDlvl {blo}-{bhi}: {m.sum()} rows, {len(set(pl[m]))} players; observed mass in the first / last two "
                   f"bins: " + " / ".join(f"{(y[m] == c).mean():.3f}" for c in (0, 1, K - 2, K - 1)))
        for v in VARIANTS:
            line = (f"  {v:7s} raw {raw[v][m].mean():.4f}  T-band {tb[v][0].mean():.4f} (T {tb[v][1]:.2f})  first / last "
                    f"two bins " + " / ".join(f"{prob[v][m][:, c].mean():.3f}" for c in (0, 1, K - 2, K - 1)))
            for ref in [r for r in ("hard", "orig") if v != "hard" and r != v]:
                dt = tb[v][0] - tb[ref][0]
                fm = np.array([dt[fold[m] == f].mean() for f in range(a.cv)])
                line += (f"\n           {v} - {ref}: raw {ci(raw[v][m] - raw[ref][m], b)}  T-band {ci(dt, b)}  "
                         f"fold SE {fm.std(ddof=1) / np.sqrt(a.cv):.4f}")
                if blo >= 5:
                    agg[v, ref].append(fm)
            out.append(line)
    out.append("\nmean over the bands from Dlvl 5 on, equal weights, T-band (SE over the folds); per band in brackets:")
    for (v, ref), fms in agg.items():
        fm = np.mean(fms, 0)
        per = " ".join(f"{q.mean():+.4f}({q.std(ddof=1) / np.sqrt(len(q)):.4f})" for q in fms)
        out.append(f"  {v:7s} - {ref:5s} {fm.mean():+.4f} (SE {fm.std(ddof=1) / np.sqrt(len(fm)):.4f})   [{per}]")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
