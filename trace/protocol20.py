"""The fixed reading rule over deep_checks.py outputs: a difference counts when its T-band value is beyond twice the
seed-noise threshold of the same cell (|seed1 - all| of two trainings on the same data) and has the same sign in val
and in test (both files with --drop-top 1). One pair of seeds is one draw of the noise, so a cell where the pair
happens to agree gives a near-zero threshold; a difference must also pass the pooled threshold: twice the RMS of the
seed differences over the deep bands (13+) of the same question.
The cross-fitted T of a band is unstable over folds of players and can make a model worse than its raw logits: where
any model of a cell has T-band above raw, the difference must also pass on raw log loss (same two thresholds, taken
from the raw seed differences, and the same sign as T-band); the raw difference is printed next to T-band always.
With several seed files (one pair of seeds each) the noise of a cell is the RMS of its values over the pairs, the
pooled threshold the RMS over the deep cells of all pairs.
Mean predicted probabilities (P(yes) of ascends, P(ascend) in end, P(no deeper), P(ttd <= 1000)) are read the same
way on the difference of the two means, on the bands 21+ (the --pool-bands when those are early) with the pooled
threshold over them; the files give no val / test split for them, so no sign rule. Of these only the mean P(yes) of
ascends on 21+ belongs to the reading rule; the others are for exploring.
  python protocol20.py ../results/seed20_drop1.txt [../results/seed24_drop1.txt ...] ../results/fix20_drop1.txt
      [--bands 21-24 25-29 30+]
"""
import argparse
import re

import numpy as np

NUM = r"([+-]?\d+\.\d+)"
PAIR = re.compile(r"^    (\w+) - (\w+): raw " + NUM + r" \[[^]]*\].*T-band " + NUM + r" \[" + NUM + ", " + NUM
                  + r"\]\s+\| val " + NUM + r" \[[^]]*\]\s+\| test " + NUM)
MODEL = re.compile(r"^    (\w+)\s+raw " + NUM + r"\s+T-all " + NUM + r"\s+T-band " + NUM)
MEAN = re.compile(r"^    (P\([^)]*\)) mean: observed \d\.\d+  (.*)$")
DEEP = ["21-24", "25-29", "30+"]


def cells(path, means=None):
    """(question, band, pair) -> (raw, T-band, lo, hi, val, test); (question, band) -> models with T-band above raw;
    means: (question, band, event) -> {model: mean predicted probability}, filled when given."""
    out, worse, q, band = {}, {}, None, None
    for line in open(path, encoding="utf-8"):
        if line.startswith("==== "):
            q = line.split()[1].rstrip(":")
        elif line.startswith("  Dlvl "):
            band = line.split()[1].rstrip(":")
        elif m := PAIR.match(line):
            a, b, r, d, lo, hi, v, t = m.groups()
            out[q, band, f"{a} - {b}"] = tuple(map(float, (r, d, lo, hi, v, t)))
        elif m := MODEL.match(line):
            if float(m.group(4)) > float(m.group(2)):
                worse.setdefault((q, band), []).append(m.group(1))
        elif means is not None and (m := MEAN.match(line)):
            w = m.group(2).split()
            means[q, band, m.group(1)] = {w[i]: float(w[i + 1]) for i in range(0, len(w), 2)}
    return out, worse


def mean_diffs(pairs, means):
    """(question, band, event, pair) -> difference of the two models' mean probabilities"""
    out = {}
    for (q, b, ev), mm in means.items():
        for p in pairs:
            x, r = p.split(" - ")
            if x in mm and r in mm:
                out[q, b, ev, p] = mm[x] - mm[r]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", nargs="+", help="seed files, then the target file")
    ap.add_argument("--bands", nargs="*", default=[])
    ap.add_argument("--pool-bands", nargs="*", default=["13-20", "21-24", "25-29", "30+"],
                    help="bands the pooled threshold is taken over (1-4 5-12 for a change aimed at the early bands)")
    a = ap.parse_args()
    if len(a.seed) < 2:
        ap.error("need at least one seed file and the target file")
    seed_means = [{} for _ in a.seed[:-1]]
    seeds = [cells(p, sm)[0] for p, sm in zip(a.seed[:-1], seed_means)]
    target_means = {}
    target, worse = cells(a.seed[-1], target_means)
    print("seed pairs: " + "; ".join(f"{p}: {', '.join(sorted({k[2] for k in s}))}" for p, s in zip(a.seed, seeds)))

    def thresholds(i, ss=seeds):
        sq = {}
        for s in ss:
            for (q, b, _), v in s.items():
                sq.setdefault((q, b), []).append(v[i] ** 2)
        noise = {k: np.sqrt(np.mean(v)) for k, v in sq.items()}
        return noise, {q: 2 * np.sqrt(np.mean([noise[q, b] ** 2 for b in a.pool_bands if (q, b) in noise]))
                       for q, _ in noise}

    noise, pool = thresholds(1)
    rnoise, rpool = thresholds(0)
    print("pooled thresholds, T-band: " + "  ".join(f"{q} {t:.4f}" for q, t in pool.items()))
    print("pooled thresholds, raw:    " + "  ".join(f"{q} {t:.4f}" for q, t in rpool.items()))
    if len(seeds) > 1:
        for p, s in zip(a.seed, seeds):
            for i, name in ((1, "T-band"), (0, "raw   ")):
                print(f"  this pair only, {name}: " + "  ".join(f"{q} {t:.4f}" for q, t in thresholds(i, [s])[1].items())
                      + f"  ({p})")
    print(f"{'question':8s} {'band':6s} {'pair':16s} {'T-band':>8s} {'95% CI':>19s} {'val':>8s} {'test':>8s} "
          f"{'2x seed':>8s} {'pooled':>7s} {'raw':>8s} {'2x seed':>8s} {'pooled':>7s}  verdict")
    for (q, b, p), (r, d, lo, hi, v, t) in target.items():
        if a.bands and b not in a.bands:
            continue
        thr, rthr = 2 * noise.get((q, b), float("nan")), 2 * rnoise.get((q, b), float("nan"))
        same = v * d > 0 and t * d > 0
        ci = lo * hi > 0  # the players bootstrap excludes 0
        lit = abs(d) > thr and same
        verdict = ("counts" if abs(d) > pool[q] else "cell only") if lit else ""
        if verdict == "counts" and worse.get((q, b)) and not (abs(r) > rthr and abs(r) > rpool[q] and r * d > 0):
            verdict = "T-band only (" + ", ".join(worse[q, b]) + " T-band > raw)"
        verdict += (", ci" if ci else "") if verdict else ("ci only" if ci else "")
        print(f"{q:8s} {b:6s} {p:16s} {d:+8.4f} [{lo:+.4f}, {hi:+.4f}] {v:+8.4f} {t:+8.4f} {thr:8.4f} {pool[q]:7.4f} "
              f"{r:+8.4f} {rthr:8.4f} {rpool[q]:7.4f}  {verdict}")

    # mean predicted probabilities: the seed shift of a cell is the difference of its pair's two means
    sq = {}
    for s, sm in zip(seeds, seed_means):
        for (q, b, ev, _), d in mean_diffs({k[2] for k in s}, sm).items():
            sq.setdefault((q, b, ev), []).append(d ** 2)
    mnoise = {k: np.sqrt(np.mean(v)) for k, v in sq.items()}
    mbands = DEEP if set(a.pool_bands) & set(DEEP) else a.pool_bands  # the early pool for a change aimed there
    mpool = {(q, ev): 2 * np.sqrt(np.mean([mnoise[q, b, ev] ** 2 for b in mbands if (q, b, ev) in mnoise]))
             for q, _, ev in mnoise}
    diffs = mean_diffs({k[2] for k in target}, target_means)
    if diffs:
        print("\nmean predicted probability, difference of the two means, bands " + ", ".join(mbands)
              + " (the pooled threshold over them)")
        print(f"{'question':8s} {'band':6s} {'event':14s} {'pair':16s} {'diff':>8s} {'2x seed':>8s} {'pooled':>7s}  verdict")
    for (q, b, ev, p), d in diffs.items():
        if b not in mbands or a.bands and b not in a.bands or (q, b, ev) not in mnoise:
            continue
        thr, pl = 2 * mnoise[q, b, ev], mpool[q, ev]
        verdict = ("counts" if abs(d) > pl else "cell only") if abs(d) > thr else ""
        print(f"{q:8s} {b:6s} {ev:14s} {p:16s} {d:+8.4f} {thr:8.4f} {pl:7.4f}  {verdict}")


if __name__ == "__main__":
    main()
