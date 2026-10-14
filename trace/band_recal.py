"""How much of a model's held-out loss per Dlvl band is its band mean missing the targets' mean (CPU, from ready
predictions)? Per model, question and band: adj = log(mean target) - log(mean model probability) per option, added to
the model's held-out logits of the band; held-out log loss before and after, raw and with T inside the band, after -
before with a bootstrap over players.
  --train given: adj from the model's own train rows (old targets), T inside the band by 2-fold over players (20
          halvings); no held-out row is seen by adj.
  no --train: adj and T both fitted on one half of the held-out players (their soft targets) and applied to the other
          half, 20 halvings (adj first, then T on the adjusted logits of the fitting half) -- for models without train
          predictions.
  --obs: the mean target is the observed label share (+1/K pseudo-count spread over the K options) instead of the mean
          soft target -- for comparing models trained on different targets (the old target's tails are not the data's).
  --fit-bands: upper Dlvl edges of the bands adj is fitted on (train mode only), e.g. 3 7 19 23 28; losses are still
          read by the usual bands -- a gain must not rest on the fit bands coinciding with the read ones.
  --a / --b: two groups of models (seeds): D = mean of A's row losses - mean of B's, before and after the correction,
          per band (a bootstrap over players).
  --split-report: also read each input file's rows separately from the same fitted losses; fitting and the dropped
          players still use all held-out inputs together.
  python band_recal.py --train human_dwa24/train.jsonl --train-tag human_dwa24-train \
      --test human_dwa20t/val.jsonl human_dwa20t/test.jsonl --test-tag human_dwa20t-val human_dwa20t-test \
      --models c17e c17es1 c17es2 c17es3 --drop-top 1 --out ../results/band_recal24_drop1.txt
  python band_recal.py --test human_dwa20t/val.jsonl human_dwa20t/test.jsonl --test-tag human_dwa20t-val human_dwa20t-test \
      --a m20e m20es1 m20es2 m20es3 --b c17e c17es1 c17es2 c17es3 --drop-top 1 --out ../results/band_recal_mix_drop1.txt
predictions: preds/<model>-<tag>.parquet"""
import argparse
import bisect
import collections
import json
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from curve_eval import QS, boot_idx, ci, crossfit, nll
from leak_total import BANDS, NAMES, load, probs


def adj_of(mean, p):
    return np.log(np.clip(mean, 1e-6, 1)) - np.log(np.clip(p.mean(0), 1e-6, 1))


def source(target, y, k, obs):
    """-> the mean the model's mean is moved to: observed label share (+1/K pseudo-count) or mean soft target"""
    if obs:
        c = np.bincount(y, minlength=k) + 1.0 / k
        return c / c.sum()
    return target.mean(0)


def fit_t(z, y):
    return minimize_scalar(lambda t: nll(z, y, t).mean(), bounds=(0.3, 5), method="bounded").x


def held_fit(z, y, target, players, obs, n=20):
    """-> raw and T loss after adj, both fitted on the other half of the players; T loss before (same halvings)"""
    raw, tl, t0l = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for s in range(n):
        half = np.array([zlib.crc32(f"{s}:{x}".encode()) % 2 for x in players])
        for h in (0, 1):
            m, o = half != h, half == h
            z2 = z + adj_of(source(target[m] if target is not None else None, y[m], z.shape[1], obs), np.exp(z[m]))
            raw[o] += nll(z2[o], y[o])
            tl[o] += nll(z2[o], y[o], fit_t(z2[m], y[m]))
            t0l[o] += nll(z[o], y[o], fit_t(z[m], y[m]))
    return raw / n, tl / n, t0l / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=[])
    ap.add_argument("--train-tag", nargs="+", default=[])
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--test-tag", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--a", nargs="+", default=[])
    ap.add_argument("--b", nargs="+", default=[])
    ap.add_argument("--obs", action="store_true")
    ap.add_argument("--fit-bands", type=int, nargs="+", default=[])
    ap.add_argument("--drop-top", type=int, default=0)
    ap.add_argument("--split-report", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    models = list(dict.fromkeys(a.models + a.a + a.b))
    held = not a.train
    assert not (held and a.fit_bands), "--fit-bands needs --train"
    fit_edges = tuple(a.fit_bands) or BANDS
    if not held:
        tr, keys_tr, _ = load(a.train, targets=not a.obs)
    te, keys_te, lk = load(a.test, targets=held and not a.obs)
    split_of, split_names = {}, []
    if a.split_report:
        for path in a.test:
            name = Path(path).stem
            assert name not in split_names, "split input names must be unique"
            split_names.append(name)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row_id = json.loads(line)["id"]
                    assert row_id not in split_of, "split inputs must have disjoint row ids"
                    split_of[row_id] = name
    how = ("adj and T fitted on the other half of the held-out players, 20 halvings" if held else
           "adj from the model's train rows, T inside the band by 2-fold over players")
    out = [f"held-out {' '.join(a.test)}; per band, adj = log(mean target) - log(mean model) added to the held-out "
           f"logits; {how}; target mean = " + ("observed label share (+1/K)" if a.obs else "mean soft target")]
    if a.fit_bands:
        out[0] += f"; adj fitted on bands with upper Dlvl edges {' '.join(map(str, fit_edges))}, read by the usual bands"
    dropped = []
    if a.drop_top:  # as deep_checks.py: the players with the most ascended rows
        asc = te["ascends"]
        yes = lk["ascends"].index("true")
        cnt = collections.Counter(p for p, v in zip(asc["players"], asc["y"]) if v == yes)
        dropped = [p for p, _ in cnt.most_common(a.drop_top)]
        out[0] += "; left out the top player(s) by ascended rows"
    for q in QS:
        t = te[q]
        if not held:
            assert keys_tr[q] == keys_te[q], q
            d = tr[q]
            fb_tr = np.array([bisect.bisect_left(fit_edges, x) for x in d["dlvl"]])
        keep = np.array([p not in dropped for p in t["players"]])
        pl = np.array([p for p, k in zip(t["players"], keep) if k])
        bands, y = t["bands"][keep], t["y"][keep]
        splits = np.array([split_of[x] for x in t["ids"]])[keep] if a.split_report else None
        fb_te = np.array([bisect.bisect_left(fit_edges, x) for x in t["dlvl"][keep]])
        loss = collections.defaultdict(dict)  # band -> model -> (raw0, raw1, T0, T1, t0, t1, players)
        for m in models:
            pte = probs([f"preds/{m}-{x}.parquet" for x in a.test_tag], q, t["ids"])[keep]
            z = np.log(np.clip(pte, 1e-12, 1))
            if not held:
                ptr = probs([f"preds/{m}-{x}.parquet" for x in a.train_tag], q, d["ids"])
                z2 = z.copy()
                for f in np.unique(fb_te):
                    kf = fb_tr == f
                    z2[fb_te == f] += adj_of(source(None if a.obs else d["old"][kf], d["y"][kf], z.shape[1], a.obs),
                                             ptr[kf])
            for b in range(6):
                kt = bands == b
                if not kt.any():
                    continue
                zb, yb, pb = z[kt], y[kt], pl[kt]
                r0 = nll(zb, yb)
                if held:
                    tgt = None if a.obs else t["old"][keep][kt]
                    r1, c1, c0 = held_fit(zb, yb, tgt, pb, a.obs)
                    t0 = t1 = float("nan")
                else:
                    r1 = nll(z2[kt], yb)
                    c0, t0 = crossfit(zb, yb, pb)
                    c1, t1 = crossfit(z2[kt], yb, pb)
                loss[b][m] = (r0, r1, c0, c1, t0, t1, pb)
        out.append(f"\n==== {q}")
        out.append(f"  {'':<16}{'raw before':>11} {'after':>7}  {'after - before [95 %]':<26}"
                   f"{'T-band before':>14} {'after':>7}  {'after - before [95 %]':<26} T before / after")
        for b in range(6):
            if not loss[b]:
                continue
            bi = boot_idx(next(iter(loss[b].values()))[6])
            for m in models:
                r0, r1, c0, c1, t0, t1, _ = loss[b][m]
                out.append(f"  {NAMES[b] + ' ' + m:<16}{r0.mean():11.4f} {r1.mean():7.4f}  {ci(r1 - r0, bi):<26}"
                           f"{c0.mean():14.4f} {c1.mean():7.4f}  {ci(c1 - c0, bi):<26} {t0:.2f} / {t1:.2f}")
            for name, grp in (("A", a.a), ("B", a.b)) if a.a and a.b else (("mean", models),):
                if len(grp) > 1 or a.split_report:
                    mr = np.mean([loss[b][m][1] - loss[b][m][0] for m in grp], 0)
                    mc = np.mean([loss[b][m][3] - loss[b][m][2] for m in grp], 0)
                    label = NAMES[b] + ("" if name == "mean" else " " + name) + " mean"
                    if len(grp) > 1:
                        out.append(f"  {label:<16}{'':11} {'':7}  {ci(mr, bi):<26}{'':14} {'':7}  "
                                   f"{ci(mc, bi):<26}")
                    if splits is not None:
                        for split in split_names:
                            ks = splits[bands == b] == split
                            if ks.any():
                                out.append(f"  split {label} {split} n={ks.sum()} "
                                           f"raw after-before {mr[ks].mean():+.4f} "
                                           f"T-band after-before {mc[ks].mean():+.4f}")
            if a.a and a.b:
                for k, label in ((0, "raw"), (2, "T-band")):
                    d0 = np.mean([loss[b][m][k] for m in a.a], 0) - np.mean([loss[b][m][k] for m in a.b], 0)
                    d1 = np.mean([loss[b][m][k + 1] for m in a.a], 0) - np.mean([loss[b][m][k + 1] for m in a.b], 0)
                    out.append(f"  {NAMES[b] + ' D ' + label:<16}A - B before {ci(d0, bi)}   after {ci(d1, bi)}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
