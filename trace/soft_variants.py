"""Which soft depth target, judged on the hard labels: the GBM of human_tab.py fit to each candidate target (a row per
bin, weighted by the target's mass), its log loss per Dlvl band (raw and with T fitted on the other half of the
players), the difference to the hard-label fit with a 95 % bootstrap over players, and the mass on the last bins.
  hard      the one-hot label
  orig      the set's own target: N(gain, 0.75 x the true bin's width) over the gain bins (human_q6.py)
  trunc     orig cut at the reachable gain (deepest Dlvl of any train game minus this row's deepest) and renormalised
  log       N(log(1 + gain), s) with one s for all rows: 0.75 x the median width of the inner bins in log(1 + gain)
  logtrunc  log, cut as trunc
  half      trunc with half its sigma (0.375 x the true bin's width)
  narrow    trunc with sigma 0.75 of one level for every bin
The true gain is recovered from each row's stored target (the integer in the true bin whose orig target is closest); in
the open last bin every gain gives the same target, so its lowest gain is taken.
--cv N scores on the train rows themselves, N folds by player (picking a target this way leaves the held set for the
models trained on it).
  python soft_variants.py human_dwa20 human_dwa20t [--out ../results/soft_variants20.txt]
  python soft_variants.py human_dwa20 --cv 5 [--out ../results/soft_variants20cv.txt]
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
import pandas as pd
from scipy.special import ndtr

from curve_eval import BANDS, boot_idx, ci, crossfit, load_rows
from human_tab import load

VARIANTS = ("hard", "orig", "trunc", "log", "logtrunc", "half", "narrow")


def ranges(texts):
    """gain range of each bin from its option text (the deepest Dlvl so far is the bin-1 text minus 1)"""
    d = int(texts[1]) - 1
    out = [(0, 0)]
    for t in texts[1:]:
        if t.endswith("+"):
            out.append((int(t[:-1]) - d, math.inf))
        elif "-" in t:
            a, b = t.split("-")
            out.append((int(a) - d, int(b) - d))
        else:
            out.append((int(t) - d, int(t) - d))
    return d, out


def mass(lo, hi, x, s):
    return ndtr((hi - x) / s) - ndtr((lo - x) / s)


def orig(rg, g, cap=math.inf, scale=0.75, flat=False):
    k = next(i for i, (a, b) in enumerate(rg) if a <= g <= b)
    w = (rg[k][1] - rg[k][0] + 1) if 0 < k < len(rg) - 1 and not flat else 1.0
    lo = np.array([-math.inf] + [a - 0.5 for a, _ in rg[1:]])
    hi = np.array([b + 0.5 for _, b in rg[:-1]] + [math.inf])
    hi = np.minimum(hi, cap + 0.5)
    m = np.where(lo < hi, mass(lo, hi, g, scale * w), 0.0)
    return m / m.sum()


def logt(rg, g, s, cap=math.inf):
    lo = np.array([-math.inf] + [math.log1p(a - 0.5) for a, _ in rg[1:]])
    hi = np.array([math.log1p(b + 0.5) for _, b in rg[:-1]] + [math.inf])
    hi = np.minimum(hi, math.log1p(cap + 0.5))
    m = np.where(lo < hi, mass(lo, hi, math.log1p(g), s), 0.0)
    return m / m.sum()


def recover(rg, soft, k):
    a, b = rg[k]
    cands = range(int(a), int(min(b, a + 40)) + 1)
    return min(cands, key=lambda g: np.abs(orig(rg, g) - soft).sum())


def fit(job):
    name, xt, rows, w, xh, threads = job
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True, random_state=0)
    m.fit(xt[rows[:, 0]], rows[:, 1], sample_weight=w)
    p = np.full((len(xh), K), 1e-9)
    p[:, m.classes_] = np.clip(m.predict_proba(xh), 1e-9, 1)
    return name, p / p.sum(1, keepdims=True)


def main():
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("held", nargs="?", default="")
    ap.add_argument("--cv", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--bands", nargs="*", default=[], help="Dlvl bands as lo-hi (default curve_eval.BANDS)")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tr = [json.loads(x) for x in open(Path(a.train, "train.jsonl"), encoding="utf-8")]
    tr = [r for r in tr if "depth" in r["gold"]][: a.max_train or None]
    X, _ = load(Path(a.train, "train.jsonl"))
    K = len(tr[0]["questions"]["depth"]["criteria"])
    info, err = [], []
    for r in tr:
        d, rg = ranges(r["questions"]["depth"]["criteria"])
        soft = np.array([r["gold"]["depth"]["probabilities"].get(str(c), 0.0) for c in range(K)])
        k = int(r["gold"]["depth"]["label"])
        g = recover(rg, soft, k) if k < K - 1 else int(rg[k][0])
        if k < K - 1:
            err.append(np.abs(orig(rg, g) - soft).sum())
        info.append((r["id"], d, rg, g, k))
    final = np.array([d + g for _, d, _, g, _ in info])
    cap = int(final.max())
    widths = [math.log1p(b + 0.5) - math.log1p(a - 0.5) for a, b in info[0][2][1:-1]]
    s = 0.75 * float(np.median(widths))
    out = [f"{len(info)} train rows; recovered gain: |orig(g) - stored target| median {np.median(err):.2e}, "
           f"max {max(err):.3f} (open bin aside); reachable cap (deepest Dlvl of any train game) {cap}, rows ending "
           f"deeper than 50 / 53: {(final > 50).sum()} / {(final > 53).sum()}; log sigma {s:.3f} "
           f"(0.75 x median inner width {np.median(widths):.3f})"]
    targets = {"hard": lambda d, rg, g, k: np.eye(K)[k], "orig": lambda d, rg, g, k: orig(rg, g),
               "trunc": lambda d, rg, g, k: orig(rg, g, cap - d), "log": lambda d, rg, g, k: logt(rg, g, s),
               "logtrunc": lambda d, rg, g, k: logt(rg, g, s, cap - d),
               "half": lambda d, rg, g, k: orig(rg, g, cap - d, 0.375),
               "narrow": lambda d, rg, g, k: orig(rg, g, cap - d, flat=True)}
    xt = X.loc[[i for i, *_ in info]].to_numpy()
    if a.cv:
        pl = np.array([i.split("@")[0].lower() for i, *_ in info])
        dl = np.array([int(re.search(r"\bDlvl (\d+)", r["state"]).group(1)) for r in tr])
        y = np.array([k for *_, k in info])
        fold = np.array([zlib.crc32(p.encode()) % a.cv for p in pl])
        out[0] += f"; {a.cv} folds by player, scored on train"
    else:
        held = load_rows(a.held)
        sel = [i for i, r in enumerate(held) if "depth" in r["gold"]]
        Xh = pd.concat([load(Path(a.held, f"{sp}.jsonl"))[0] for sp in ("val", "test")])
        xh = Xh.reindex(columns=X.columns).loc[[held[i]["id"] for i in sel]].to_numpy()
        y = np.array([int(held[i]["gold"]["depth"]["label"]) for i in sel])
        pl = np.array([held[i]["player"] for i in sel])
        dl = np.array([held[i]["dlvl"] for i in sel])
    vs = ["hard"] + [v for v in a.variants if v != "hard"]
    bands = [tuple(map(int, x.split("-"))) for x in a.bands] or BANDS
    jobs = []
    for v in vs:
        t = np.stack([targets[v](d, rg, g, k) for _, d, rg, g, k in info])
        out.append(f"  {v:8s}: mean mass per bin over train " + " ".join(f"{x:.3f}" for x in t.mean(0)))
        for f in range(a.cv) if a.cv else [None]:
            tm = fold != f if a.cv else np.ones(len(t), bool)
            j, c = np.nonzero(t[tm] > 1e-3)
            jobs.append(((v, f), xt[tm], np.stack([j, c], 1), t[tm][j, c], xt[~tm] if a.cv else xh, a.threads))
    with Pool(a.workers) as pool:
        parts = dict(pool.imap_unordered(fit, jobs))
    prob = {}
    for v in vs:
        if a.cv:
            prob[v] = np.zeros((len(y), K))
            for f in range(a.cv):
                prob[v][fold == f] = parts[v, f]
        else:
            prob[v] = parts[v, None]
    loss = {}
    for v in vs:
        z = np.log(prob[v])
        raw = -np.log(prob[v][np.arange(len(y)), y])
        loss[v] = (raw, z)
    agg = collections.defaultdict(list)  # variant -> per band from Dlvl 5, the fold means of the T-band difference
    for lo, hi in bands:
        m = (dl >= lo) & (dl <= hi)
        if not m.any():
            continue
        b = boot_idx(pl[m])
        tb = {v: crossfit(loss[v][1][m], y[m], pl[m]) for v in vs}
        out.append(f"\nDlvl {lo}-{hi}: {m.sum()} rows, {len(set(pl[m]))} players; observed mass in bins 0 / 1 / 9 / 10 / 11: "
                   + " / ".join(f"{(y[m] == c).mean():.3f}" for c in (0, 1, 9, 10, 11)))
        for v in vs:
            line = (f"  {v:8s} raw {loss[v][0][m].mean():.4f}  T-band {tb[v][0].mean():.4f} (T {tb[v][1]:.2f})  "
                    f"bins 0 / 1 / 9 / 10 / 11 " + " / ".join(f"{prob[v][m][:, c].mean():.3f}" for c in (0, 1, 9, 10, 11)))
            if v != "hard":
                dt = tb[v][0] - tb["hard"][0]
                line += f"  | {v} - hard: raw {ci(loss[v][0][m] - loss['hard'][0][m], b)}  T-band {ci(dt, b)}"
                if a.cv:
                    fm = np.array([dt[fold[m] == f].mean() for f in range(a.cv)])
                    line += f"  fold SE {fm.std(ddof=1) / np.sqrt(a.cv):.4f}"
                    if lo >= 5:
                        agg[v].append(fm)
            out.append(line)
    if agg:
        out.append("\nmean over the bands from Dlvl 5 on, equal weights, T-band X - hard (SE over the folds):")
        for v, fms in agg.items():
            fm = np.mean(fms, 0)
            out.append(f"  {v:8s} {fm.mean():+.4f} (SE {fm.std(ddof=1) / np.sqrt(len(fm)):.4f})")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
