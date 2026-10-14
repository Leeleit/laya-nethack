"""D = mean of models A minus mean of models B per Dlvl band and question: the per-row losses are averaged (not the
predictions), with the 95 % bootstrap over the band's players, val and test apart, and every pairwise difference a - b.
Read against the seed noise of several seed files (one pair of seeds each, deep_checks.py preds output): for a
difference of two-seed means the threshold is sqrt(2) x RMS -- the cell's RMS over the pairs, and the pooled one over
the deep bands 13+ (bands 1-4 and 5-12 pooled between themselves). D counts when it passes both thresholds and has one
sign in val, in test and in every pairwise difference; where any model of the cell has T-band above its raw loss, D
must also pass on raw (raw thresholds, the same sign).
  python seed_avg.py human_dwa20t --a m20e m20es1 --b c17e c17es1 --seeds ../results/seed24_drop1.txt
      ../results/seed25_drop1.txt [--drop-top 1] [--out ../results/avg25_drop1.txt]
"""
import argparse
import bisect
import collections
from pathlib import Path

import numpy as np

from deep_checks import BANDS, NAMES
from protocol20 import cells

POOLS = (("1-4", "5-12"), ("13-20", "21-24", "25-29", "30+"))


def thresholds(seed_files):
    """(question, band) -> (cell, pooled) for T-band and for raw, sqrt(2) x RMS over the pairs"""
    seeds = [cells(p)[0] for p in seed_files]
    out = {}
    for i in (1, 0):  # T-band, raw
        sq = collections.defaultdict(list)
        for s in seeds:
            for (q, b, _), v in s.items():
                sq[q, b].append(v[i] ** 2)
        noise = {k: np.sqrt(np.mean(v)) for k, v in sq.items()}
        for (q, b), n in noise.items():
            pool = next(p for p in POOLS if b in p)
            pooled = np.sqrt(np.mean([noise[q, x] ** 2 for x in pool if (q, x) in noise]))
            out.setdefault((q, b), {})[i] = (np.sqrt(2) * n, np.sqrt(2) * pooled)
    return out


def main():
    import pandas as pd
    from curve_eval import boot_idx, ci, crossfit, labels, load_rows, nll
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--drop-top", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    thr = thresholds(a.seeds)
    rows = load_rows(a.data)
    out = [f"{a.data}: D = mean({', '.join(a.a)}) - mean({', '.join(a.b)}); seed files {', '.join(a.seeds)}; "
           "thresholds sqrt(2) x RMS over the pairs"]
    if a.drop_top:
        c = collections.Counter(r["player"] for r in rows if r["asc"])
        top = {p for p, _ in c.most_common(a.drop_top)}
        gone = [r for r in rows if r["player"] in top]
        rows = [r for r in rows if r["player"] not in top]
        out.append(f"left out: the {a.drop_top} player(s) with the most ascended rows -- {len(gone)} rows")
    models = list(dict.fromkeys(a.a + a.b))
    dfs = {m: pd.concat([pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{s}.parquet") for s in ("val", "test")])
           for m in models}
    out.append(f"{len(rows)} rows")
    head = (f"{'band':6s} {'D T-band':>8s} {'95% CI':>19s} {'val':>8s} {'test':>8s}  pairwise a - b (T-band)"
            f"{'':14s}{'cell':>7s} {'pooled':>7s} {'D raw':>8s} {'cell':>7s} {'pooled':>7s}  verdict")
    for q in ("ttd", "depth", "end", "ascends"):
        sel, y, _ = labels(rows, q)
        pl = np.array([rows[i]["player"] for i in sel])
        band = np.array([bisect.bisect_left(BANDS, rows[i]["dlvl"]) for i in sel])
        split = np.array([rows[i]["split"] for i in sel])
        z = {}
        for m in models:
            lg = dfs[m][dfs[m].question == q].set_index("id").logits
            z[m] = np.stack([np.asarray(lg[rows[i]["id"]], float) for i in sel])
        out += [f"\n==== {q}", head]
        for b, name in enumerate(NAMES):
            mk = band == b
            if mk.sum() < 20 or (q, name) not in thr:
                continue
            loc = {m: crossfit(z[m][mk], y[mk], pl[mk])[0] for m in models}
            raw = {m: nll(z[m][mk], y[mk]) for m in models}
            d = np.mean([loc[m] for m in a.a], 0) - np.mean([loc[m] for m in a.b], 0)
            dr = np.mean([raw[m] for m in a.a], 0) - np.mean([raw[m] for m in a.b], 0)
            bb = boot_idx(pl[mk])
            v, t = (d[split[mk] == s].mean() for s in ("val", "test"))
            pw = [(loc[x] - loc[r]).mean() for x in a.a for r in a.b]
            (tc, tp), (rc, rp) = thr[q, name][1], thr[q, name][0]
            D, R = d.mean(), dr.mean()
            worse = [m for m in models if loc[m].mean() > raw[m].mean()]
            passes = abs(D) > tc and abs(D) > tp
            signs = v * D > 0 and t * D > 0 and all(p * D > 0 for p in pw)
            verdict = ""
            if passes and signs:
                verdict = "counts"
                if worse and not (abs(R) > rc and abs(R) > rp and R * D > 0):
                    verdict = "T-band only (" + ", ".join(worse) + " T-band > raw)"
            elif passes:
                verdict = "signs differ"
            elif abs(D) > tc and signs:
                verdict = "cell only"
            out.append(f"{name:6s} {ci(d, bb)} {v:+8.4f} {t:+8.4f}  " + " ".join(f"{p:+.4f}" for p in pw)
                       + f"  {tc:7.4f} {tp:7.4f} {ci(dr, bb)} {rc:7.4f} {rp:7.4f}  {verdict}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
