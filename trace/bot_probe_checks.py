"""Checks before the bot-risk probe (bot_data2), on top of bot_probe_prep.py:
  1. which bot_data2 games the BC model saw: bc_build splits val1 by crc32 of the trace path (test 1 in 20), so most
     val1 games of any bot_data2 split were BC training games (their screens, not their outcomes);
  2. deaths by band counted on the game's last row (every game in bot_data2 ends in death), against the count of
     games with a bucket-0 row in the band (rows are every stride-th risk state and miss part of the deaths);
  3. two gradient boostings that differ only in the seed of the bucket-4 subsample: the interval of their paired
     difference by band (T fitted inside the band, cross-fitted over games) on all test rows and on 1/5 of them.
  python bot_probe_checks.py [--data bot_data2] [--bc-root /data/laya/nethack/nethackers/runs] [--threads 14] [--out ...]
"""
import argparse
import bisect
import os
import zlib
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bot_data2")
    ap.add_argument("--bc-root", default="/data/laya/nethack/nethackers/runs")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--far", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from compare_v2 import design
    from curve_eval import boot_idx, ci, crossfit, nll
    from deep_checks import BANDS, NAMES

    f = pd.read_parquet(Path(a.data, "features.parquet"))
    f["game"] = f.id.str.split("#").str[0]
    f["turn"] = f.id.str.split("#").str[1].astype(int)
    f["band"] = [bisect.bisect_left(BANDS, d) for d in f.depth.to_numpy()]
    out = [f"{a.data}: {len(f)} feature rows, {f.game.nunique()} games"]

    out.append("\n== 1. games the BC model trained on (bc_build: val1 only, test = crc32 of the trace path % 20 == 0)")
    g = f.groupby("game").split.first().reset_index()
    g["run"] = g.game.str.split("/").str[0]
    val1 = g.run.str.startswith("val-dwa-law-fem-100000-")
    path = a.bc_root + "/val1/" + g.run + "/trace/" + g.game.str.split("/").str[1]
    g["bc"] = np.where(val1, ["bc-test" if zlib.crc32(p.encode()) % 20 == 0 else "bc-train" for p in path], "-")
    for s in ("train", "val", "test"):
        d = g[g.split == s]
        c = d.bc.value_counts()
        out.append(f"  {s}: {len(d)} games, val1 {int(val1[g.split == s].sum())} (bc-train {c.get('bc-train', 0)},"
                   f" bc-test {c.get('bc-test', 0)}), val2-3 {c.get('-', 0)}")

    out.append("\n== 2. deaths by band: band of the game's last row (its ttd = turns to death) against games with a "
               "bucket-0 row in the band")
    last = f.sort_values("turn").groupby("game").tail(1)
    for s in ("val", "test"):
        d, l = f[f.split == s], last[last.split == s]
        out.append(f"  {s}: last row ttd median {int(l.ttd.median())}, p90 {int(l.ttd.quantile(0.9))}")
        for b, name in enumerate(NAMES):
            lb = l[l.band == b]
            out.append(f"    Dlvl {name:5s} deaths {len(lb):5d} (val1 {lb.game.str.startswith('val-dwa-law-fem-100000-').sum():4d})"
                       f"  games with a bucket-0 row {d[(d.band == b) & (d.bucket == 0)].game.nunique():5d}")

    tr, va, te = (f[f.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    top = list(tr.strategy.value_counts().index[:15])
    roles = sorted(set(f.identity.str.slice(0, 3)))
    Xva, Xte = design(va, top, roles), design(te, top, roles)
    yv, yt, gt, bt = va.bucket.to_numpy(), te.bucket.to_numpy(), te.game.to_numpy(), te.band.to_numpy()
    zs = []
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        fit = (tr.bucket < 4) | (rng.random(len(tr)) < 1 / a.far)
        w = np.where(tr.bucket[fit] < 4, 1.0, float(a.far))
        gbm = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=0.05, min_samples_leaf=200,
                                             l2_regularization=1.0, early_stopping=False, random_state=0)
        gbm.fit(design(tr[fit], top, roles), tr.bucket[fit], sample_weight=w)
        losses = [nll(np.log(np.clip(p, 1e-9, 1)), yv).mean() for p in gbm.staged_predict_proba(Xva)]
        best = int(np.argmin(losses))
        zs.append(next(np.log(np.clip(p, 1e-9, 1)) for i, p in enumerate(gbm.staged_predict_proba(Xte)) if i == best))
        out.append(f"\n  seed {seed}: {fit.sum()} fitted rows, best iteration {best + 1}")
    out.append("\n== 3. paired difference seed 1 - seed 0 by band (T-band, cross-fitted over games; raw beside); all test "
               "rows and 1/5 of them (crc32 of the id)")
    sub = np.array([zlib.crc32(i.encode()) % 5 == 0 for i in te.id])
    for b, name in enumerate(NAMES):
        line = f"  Dlvl {name:5s}"
        for label, mask in (("all", np.ones(len(te), bool)), ("1/5", sub)):
            m = mask & (bt == b)
            if m.sum() < 20 or len(set(yt[m])) < 2:
                continue
            bb = boot_idx(gt[m])
            t0, _ = crossfit(zs[0][m], yt[m], gt[m])
            t1, _ = crossfit(zs[1][m], yt[m], gt[m])
            raw = nll(zs[1][m], yt[m]) - nll(zs[0][m], yt[m])
            line += f"  {label}: {m.sum():6d} rows T-band {ci(t1 - t0, bb)} raw {ci(raw, bb)}"
        out.append(line)
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
