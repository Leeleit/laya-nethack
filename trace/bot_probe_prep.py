"""Before the probe "does the human-data base carry over to the bot's risk" (bot_data2: val1-3, the question "turns until
the hero dies", 5 buckets): what the data holds per Dlvl band, a tabular reference per band, and how much of the test
the GPU has to predict.
  1. rows, games and bucket shares per split and band; train is thinned by bucket (meta keep), so a model fitted on it
     learns p * keep and must be divided by keep before any loss on the unthinned val / test;
  2. gradient boosting (compare_v2's features) per band on test: log loss raw and with T fitted inside the band
     (cross-fitted over games), 95 % bootstrap over games; the band's bucket prior from train as the floor;
  3. the same on a share of the test rows (crc32 of the id), to see how much the intervals widen.
  python bot_probe_prep.py [--data bot_data2] [--threads 14] [--out ../results/bot_probe_prep.txt]
"""
import argparse
import bisect
import collections
import json
import os
import re
import zlib
from pathlib import Path

import numpy as np

DLVL = re.compile(r"Dlvl (\d+)")
LABEL = re.compile(r'"label": "(\d)"')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bot_data2")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--far", type=int, default=10, help="fit on every row of buckets 0-3 and 1/far of bucket 4")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from compare_v2 import design
    from curve_eval import boot_idx, ci, crossfit, nll
    from deep_checks import BANDS, NAMES

    meta = json.load(open(Path(a.data, "meta.json"), encoding="utf-8"))
    keep, levels = np.array(meta["keep"]), meta["levels"]
    f = pd.read_parquet(Path(a.data, "features.parquet"))
    f["game"] = f.id.str.split("#").str[0]
    f["band"] = [bisect.bisect_left(BANDS, d) for d in f.depth.to_numpy()]
    # train.jsonl thins every risk state, features.parquet keeps every stride-th: the thinned rows are read from the
    # jsonl itself, not matched to the features
    ids, dl, lab = [], [], []
    for line in open(Path(a.data, "train.jsonl"), encoding="utf-8"):
        ids.append(line[8:line.index('"', 8)])  # {"id": "<id>", ...
        dl.append(int(DLVL.search(line).group(1)))
        lab.append(int(LABEL.findall(line)[-1]))
    th = pd.DataFrame({"game": [i.split("#")[0] for i in ids], "bucket": lab,
                       "band": [bisect.bisect_left(BANDS, d) for d in dl]})
    out = [f"{a.data}: {len(f)} feature rows (every {meta['stride']}th risk state); train rows thinned by bucket "
           f"{keep.round(3).tolist()} from every risk state -> {len(th)} (the set laya learns from); buckets {levels}"]

    out.append("\n== 1. rows, games, bucket shares by band (train = all rows of the train games; 'thinned' = laya's)")
    for s in ("train", "val", "test"):
        d = f[f.split == s]
        out.append(f"  {s}: {len(d)} rows, {d.game.nunique()} games")
        for b, name in enumerate(NAMES):
            e = d[d.band == b]
            if not len(e):
                continue
            sh = np.bincount(e.bucket, minlength=5) / len(e)
            line = (f"    Dlvl {name:5s} {len(e):7d} rows {e.game.nunique():5d} games  buckets "
                    + " ".join(f"{x:.4f}" for x in sh) + f"  deaths in band {e[e.ttd <= 2].game.nunique()}")
            if s == "train":
                t = th[th.band == b]
                sh = np.bincount(t.bucket, minlength=5) / len(t)
                line += (f"  | thinned {len(t)} rows, per game median {int(t.groupby('game').size().median())}, "
                         "buckets " + " ".join(f"{x:.3f}" for x in sh))
            out.append(line)
    t = th.groupby("game").size()
    out.append(f"  thinned train rows per game: median {int(t.median())}, p90 {int(t.quantile(0.9))}, max {t.max()}")

    tr, va, te = (f[f.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    top = list(tr.strategy.value_counts().index[:15])
    roles = sorted(set(f.identity.str.slice(0, 3)))
    rng = np.random.default_rng(0)
    fit = (tr.bucket < 4) | (rng.random(len(tr)) < 1 / a.far)
    w = np.where(tr.bucket[fit] < 4, 1.0, float(a.far))
    Xtr, Xva, Xte = (design(d, top, roles) for d in (tr[fit], va, te))
    gbm = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0,
                                         early_stopping=False, random_state=0)
    gbm.fit(Xtr, tr.bucket[fit], sample_weight=w)
    yv = va.bucket.to_numpy()
    losses = [nll(np.log(np.clip(p, 1e-9, 1)), yv).mean() for p in gbm.staged_predict_proba(Xva)]
    best = int(np.argmin(losses))
    # one stage at a time: all 600 stages of the test would not fit in memory
    zt = next(np.log(np.clip(p, 1e-9, 1)) for i, p in enumerate(gbm.staged_predict_proba(Xte)) if i == best)
    out.append(f"\n== 2. gradient boosting (compare_v2 features), fitted on {fit.sum()} train rows (bucket 4 at 1/{a.far},"
               f" weighted back), best iteration {best + 1} on val; test by band")
    yt, gt, bt = te.bucket.to_numpy(), te.game.to_numpy(), te.band.to_numpy()
    prior = {b: np.bincount(tr.bucket[tr.band == b], minlength=5) / max((tr.band == b).sum(), 1) for b in range(6)}

    def band_lines(mask_all, label):
        res = {}
        for b, name in enumerate(NAMES):
            m = mask_all & (bt == b)
            if m.sum() < 20 or len(set(yt[m])) < 2:
                continue
            bb = boot_idx(gt[m])
            raw = nll(zt[m], yt[m])
            tb, T = crossfit(zt[m], yt[m], gt[m])
            pr = -np.log(np.clip(prior[b][yt[m]], 1e-9, 1))
            lo_hi = ci(tb, bb)
            res[name] = lo_hi
            out.append(f"    {label} Dlvl {name:5s} {m.sum():7d} rows {len(set(gt[m])):5d} games  prior {pr.mean():.4f}"
                       f"  raw {ci(raw, bb)}  T-band {lo_hi} (T {T:.2f})")
        return res

    full = band_lines(np.ones(len(te), bool), "all ")
    out.append("\n== 3. the same on a share of the test rows (crc32 of the id): the interval half-widths against all rows")
    for k in (5, 10, 25):
        sub = np.array([zlib.crc32(i.encode()) % k == 0 for i in te.id])
        part = band_lines(sub, f"1/{k:<2d}")
        for name, s in part.items():
            def half(x):
                lo, hi = x.split("[")[1].rstrip("]").split(",")
                return (float(hi) - float(lo)) / 2
            out.append(f"      Dlvl {name}: half-width x{half(s) / half(full[name]):.2f}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
