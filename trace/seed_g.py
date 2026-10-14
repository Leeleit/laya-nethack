"""Several seeds per arm, one main metric. G = mean T-band log loss over the 8 cells ttd, depth, end, ascends x Dlvl
21-24, 25-29 (each cell weighs the same; T fitted inside the band per model, cross-fitted over players); --g-questions
narrows G to some of the questions, and then each other question gets a guard: its mean over all bands under G's rule;
--band-guard guards each question of G the same way by its mean over the bands outside 21-24, 25-29.
D = mean_B(G) - mean_A(G). Training noise from the spread of G over each arm's seeds: sigma^2 = pooled variance
(ddof 1), SE_train = sqrt(sigma^2 (1/nA + 1/nB)); test noise SE_test from a bootstrap over players, the players of
21-24 and 25-29 resampled together. G counts when D + 2 sqrt(SE_train^2 + SE_test^2) < 0, D < 0 in val and in test,
and the same on raw G. Harm (guard and cells) counts on T-band alone; when only raw is worse, the same rule on one T per
model and question (cross-fitted over all bands) decides whether it is harm or a calibration shift.
Below it, for exploration only: every cell with its own training noise (threshold 2 sqrt(sigma_c^2 (1/nA + 1/nB)),
pooled one over 13+ and over 1-4 / 5-12), the sign in val / test, how many of the nA x nB pairwise differences share
D's sign, each arm's seed spread, the newest pair B[-1] - A[-1], and G of extra models (ensembles).
  python seed_g.py human_dwa20t --a c17e c17es1 c17es2 --b m20e m20es1 m20es2 [--extra c17ens3 m20ens3]
      [--drop-top 1] [--out ../results/g26_drop1.txt]
"""
import argparse
import bisect
import collections
from pathlib import Path

import numpy as np

from deep_checks import BANDS, NAMES

QS = ("ttd", "depth", "end", "ascends")
DEEP = ("21-24", "25-29")
POOLS = (("1-4", "5-12"), ("13-20", "21-24", "25-29", "30+"))
NBOOT = 2000


def var1(x):
    """variance over seeds (ddof 1); 0 for a single seed, which then adds nothing to the pooled variance"""
    x = np.asarray(x, float)
    return float(x.var(ddof=1)) if len(x) > 1 else 0.0


def main():
    import pandas as pd
    from curve_eval import crossfit, labels, load_rows, nll
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--g-questions", nargs="+", default=list(QS), choices=QS,
                    help="the questions whose 21-24 and 25-29 cells make G (a change aimed at one question)")
    ap.add_argument("--band-guard", action="store_true",
                    help="each question of G also guarded by its mean over the bands outside 21-24, 25-29")
    ap.add_argument("--drop-top", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    nA, nB = len(a.a), len(a.b)
    assert min(nA, nB) >= 1 and nA + nB >= 3  # one seed on one side: its variance comes from the other side
    rows = load_rows(a.data)
    GQ = tuple(q for q in QS if q in a.g_questions)
    gname = f"{len(GQ) * len(DEEP)} cells {', '.join(GQ)} x 21-24, 25-29"
    out = [f"{a.data}: G = mean T-band log loss over {', '.join(GQ)} x Dlvl 21-24, 25-29; "
           f"D = mean({', '.join(a.b)}) - mean({', '.join(a.a)})"]
    if a.drop_top:
        c = collections.Counter(r["player"] for r in rows if r["asc"])
        top = {p for p, _ in c.most_common(a.drop_top)}
        gone = [r for r in rows if r["player"] in top]
        rows = [r for r in rows if r["player"] not in top]
        out.append(f"left out: the {a.drop_top} player(s) with the most ascended rows -- {len(gone)} rows")
    models = list(dict.fromkeys(a.a + a.b + a.extra))
    dfs = {m: pd.concat([pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{s}.parquet") for s in ("val", "test")])
           for m in models}
    out.append(f"{len(rows)} rows")

    cell = {}  # (q, band name) -> dict(pl, split, loc {m: per-row loss}, raw {m: ...})
    for q in QS:
        sel, y, _ = labels(rows, q)
        pl = np.array([rows[i]["player"] for i in sel])
        band = np.array([bisect.bisect_left(BANDS, rows[i]["dlvl"]) for i in sel])
        split = np.array([rows[i]["split"] for i in sel])
        z = {}
        for m in models:
            lg = dfs[m][dfs[m].question == q].set_index("id").logits
            z[m] = np.stack([np.asarray(lg[rows[i]["id"]], float) for i in sel])
        glob = {m: crossfit(z[m], y, pl)[0] for m in models}  # one T per model and question
        for b, name in enumerate(NAMES):
            mk = band == b
            if mk.sum() < 20:
                continue
            cell[q, name] = {"pl": pl[mk], "split": split[mk],
                             "loc": {m: crossfit(z[m][mk], y[mk], pl[mk])[0] for m in models},
                             "raw": {m: nll(z[m][mk], y[mk]) for m in models},
                             "glob": {m: glob[m][mk] for m in models}}

    def mean_of(cl, kind, m, split=None):
        v = []
        for key in cl:
            c = cell[key]
            x = c[kind][m] if split is None else c[kind][m][c["split"] == split]
            v.append(x.mean())
        return float(np.mean(v))

    def resampler(cl, seed):
        """players of the cells, resampled together: (index of a player, NBOOT x players weights)"""
        pls = sorted({p for key in cl for p in cell[key]["pl"]})
        rng = np.random.default_rng(seed)
        return ({p: i for i, p in enumerate(pls)},
                np.stack([np.bincount(rng.integers(0, len(pls), len(pls)), minlength=len(pls)) for _ in range(NBOOT)]))

    def stats(cl, kind, idx, W):
        """D on the mean over the cells, its noise and the val / test signs"""
        mA = np.array([mean_of(cl, kind, m) for m in a.a])
        mB = np.array([mean_of(cl, kind, m) for m in a.b])
        D = mB.mean() - mA.mean()
        s2 = ((nA - 1) * var1(mA) + (nB - 1) * var1(mB)) / (nA + nB - 2)
        se_train = np.sqrt(s2 * (1 / nA + 1 / nB))
        bd = np.zeros(NBOOT)
        for key in cl:
            c = cell[key]
            codes = np.array([idx[p] for p in c["pl"]])
            d = np.mean([c[kind][m] for m in a.b], 0) - np.mean([c[kind][m] for m in a.a], 0)
            s = np.bincount(codes, weights=d, minlength=W.shape[1])
            n = np.bincount(codes, minlength=W.shape[1])
            bd += (W @ s) / (W @ n)
        bd /= len(cl)
        se_test = bd.std(ddof=1)
        dv, dt = (np.mean([mean_of(cl, kind, m, s) for m in a.b]) - np.mean([mean_of(cl, kind, m, s) for m in a.a])
                  for s in ("val", "test"))
        return dict(mA=mA, mB=mB, D=D, se_train=se_train, se_test=se_test, ci=np.percentile(bd, [2.5, 97.5]),
                    thr=2 * np.sqrt(se_train ** 2 + se_test ** 2), dv=dv, dt=dt)

    gcl = [(q, name) for q in GQ for name in DEEP]
    g_idx, W = resampler(gcl, 0)

    def g_of(kind, m, split=None):
        return mean_of(gcl, kind, m, split)

    out.append(f"\n==== main: G ({gname})")
    verdicts = {}
    for kind, label in (("loc", "T-band"), ("raw", "raw")):
        st = stats(gcl, kind, g_idx, W)
        gA, gB, D, se_train, se_test, thr, dv, dt = (st[k] for k in ("mA", "mB", "D", "se_train", "se_test", "thr",
                                                                      "dv", "dt"))
        lo, hi = st["ci"]
        verdicts[label] = D + thr < 0 and dv < 0 and dt < 0
        out.append(f"  {label:6s} G per seed A " + " ".join(f"{x:.4f}" for x in gA) + "  B " + " ".join(f"{x:.4f}" for x in gB))
        out.append(f"         D {D:+.4f}  test bootstrap [{lo:+.4f}, {hi:+.4f}]  val {dv:+.4f} test {dt:+.4f}  "
                   f"SD seeds A {np.sqrt(var1(gA)):.4f} B {np.sqrt(var1(gB)):.4f}  SE_train {se_train:.4f} SE_test {se_test:.4f}"
                   f"  D + 2 SE_total {D + thr:+.4f}  {'passes' if verdicts[label] else 'does not pass'}")
        out.append(f"         newest pair {a.b[-1]} - {a.a[-1]}: {g_of(kind, a.b[-1]) - g_of(kind, a.a[-1]):+.4f};  "
                   + "  ".join(f"{m} - {a.a[0]}: {g_of(kind, m) - g_of(kind, a.a[0]):+.4f}" for m in a.a[1:]) + "  "
                   + "  ".join(f"{m} - {a.b[0]}: {g_of(kind, m) - g_of(kind, a.b[0]):+.4f}" for m in a.b[1:]))
        for m in a.extra:
            out.append(f"         extra {m}: G {g_of(kind, m):.4f}  minus mean A {g_of(kind, m) - gA.mean():+.4f}"
                       f"  minus mean B {g_of(kind, m) - gB.mean():+.4f}")
    out.append(f"  G counts: {'yes' if verdicts['T-band'] and verdicts['raw'] else 'no'}"
               f" (T-band {'passes' if verdicts['T-band'] else 'no'}, raw {'passes' if verdicts['raw'] else 'no'})")

    def guard(q, cl):
        idx, Wq = resampler(cl, 1)
        sides = {}
        for kind, label in (("loc", "T-band"), ("raw", "raw"), ("glob", "glob T")):
            st = stats(cl, kind, idx, Wq)
            D, thr = st["D"], st["thr"]
            sides[label] = ("worse" if D - thr > 0 and st["dv"] > 0 and st["dt"] > 0 else
                            "better" if D + thr < 0 and st["dv"] < 0 and st["dt"] < 0 else "")
            out.append(f"  {q:8s} {label:6s} D {D:+.4f}  test bootstrap [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}]"
                       f"  val {st['dv']:+.4f} test {st['dt']:+.4f}  SD seeds A {np.sqrt(var1(st['mA'])):.4f}"
                       f" B {np.sqrt(var1(st['mB'])):.4f}  SE_train {st['se_train']:.4f} SE_test {st['se_test']:.4f}"
                       f"  2 SE_total {thr:.4f}  {sides[label] or '-'}")
        tb, rw, gl = sides["T-band"], sides["raw"], sides["glob T"]
        verdict = ("worse" if tb == "worse" else "better" if tb == rw == "better" else
                   ("worse (raw only, global T worse)" if gl == "worse" else
                    "no (raw only worse, global T not: a calibration shift)") if rw == "worse" else "no")
        out.append(f"  {q:8s} counts: {verdict} ({len(cl)} bands)")

    rest = [q for q in QS if q not in GQ]
    if rest:
        out.append("\n==== guard: each question outside G, mean over its bands under G's thresholds, players of all bands "
                   "resampled together. Worse counts on T-band alone; better needs T-band and raw; worse on raw only "
                   "is settled by one T per model and question (global T)")
        for q in rest:
            guard(q, [(q, name) for name in NAMES if (q, name) in cell])
    if a.band_guard:
        out.append("\n==== band guard: each question of G, mean over its bands outside 21-24, 25-29 under G's thresholds, "
                   "the same rule as the guard above")
        for q in GQ:
            guard(q, [(q, name) for name in NAMES if name not in DEEP and (q, name) in cell])

    out.append("\n==== cells (exploration): threshold 2 sqrt(sigma_c^2 (1/nA + 1/nB)), sigma_c^2 pooled over the arms' "
               "seeds; pooled over 13+ and over 1-4 / 5-12; sign in val / test; pairwise differences with D's sign. "
               "Better needs T-band and raw; worse counts on T-band alone, and worse on raw only is settled by the same "
               "thresholds on one T per model and question (D glob)")
    for q in QS:
        out.append(f"\n  {q}")
        out.append(f"  {'band':6s} {'D T-band':>9s} {'val':>8s} {'test':>8s} {'cell':>7s} {'pooled':>7s} {'D raw':>8s}"
                   f" {'cell':>7s} {'pooled':>7s}  pairwise  SD A    SD B    {a.b[-1]} - {a.a[-1]}  {'D glob':>8s}  verdict")
        var = {}
        for kind in ("loc", "raw", "glob"):
            for name in NAMES:
                if (q, name) in cell:
                    c = cell[q, name]
                    vA = var1([c[kind][m].mean() for m in a.a])
                    vB = var1([c[kind][m].mean() for m in a.b])
                    var[kind, name] = ((nA - 1) * vA + (nB - 1) * vB) / (nA + nB - 2)
        k = np.sqrt(1 / nA + 1 / nB)
        for name in NAMES:
            if (q, name) not in cell:
                continue
            c = cell[q, name]
            pool = next(p for p in POOLS if name in p)
            r = {}
            for kind in ("loc", "raw", "glob"):
                d = np.mean([c[kind][m] for m in a.b], 0) - np.mean([c[kind][m] for m in a.a], 0)
                tc = 2 * np.sqrt(var[kind, name]) * k
                tp = 2 * np.sqrt(np.mean([var[kind, x] for x in pool if (kind, x) in var])) * k
                r[kind] = (d.mean(), tc, tp, d)
            D, tc, tp, d = r["loc"]
            v, t = (d[c["split"] == s].mean() for s in ("val", "test"))
            pw = sum((c["loc"][x].mean() - c["loc"][y].mean()) * D > 0 for x in a.b for y in a.a)
            sdA = np.sqrt(var1([c["loc"][m].mean() for m in a.a]))
            sdB = np.sqrt(var1([c["loc"][m].mean() for m in a.b]))
            newest = c["loc"][a.b[-1]].mean() - c["loc"][a.a[-1]].mean()
            R, rc, rp, rd = r["raw"]
            Gl, gc, gp, gd = r["glob"]

            def worse(x, xc, xp, xd):
                return x > xc and x > xp and all(xd[c["split"] == s].mean() > 0 for s in ("val", "test"))
            ok = abs(D) > tc and abs(D) > tp and v * D > 0 and t * D > 0
            ok_raw = abs(R) > rc and abs(R) > rp and R * D > 0
            if ok and D > 0:
                verdict = "counts (worse)"
            elif ok and ok_raw:
                verdict = "counts (better)"
            elif worse(R, rc, rp, rd):
                verdict = ("counts (worse: raw only, global T worse)" if worse(Gl, gc, gp, gd) else
                           "raw only worse, global T not (calibration shift)")
            else:
                verdict = "T-band only" if ok else "cell only" if abs(D) > tc and v * D > 0 and t * D > 0 else ""
            out.append(f"  {name:6s} {D:+9.4f} {v:+8.4f} {t:+8.4f} {tc:7.4f} {tp:7.4f} {R:+8.4f} {rc:7.4f} {rp:7.4f}"
                       f"  {pw}/{nA * nB}     {sdA:.4f}  {sdB:.4f}  {newest:+.4f}  {Gl:+8.4f}  {verdict}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
