"""Where the spread between seeds of one recipe comes from, on one held-out set (val + test).
Per cell (question x Dlvl band): T-band log loss of each seed, the SD over seeds against the test noise of one seed pair
(bootstrap over players, the same draws for every pair). For G (the 8 cells ttd, depth, end, ascends x 21-24, 25-29):
every seed pair, its test noise, and how many players carry half of the pair's per-player contributions.
Disagreement: per cell, the mean over rows of the Jensen-Shannon divergence between two seeds' raw distributions
against the same divergence between their band means (a shift of the whole band shows up in the second).
With --train: raw loss of each seed on its own train rows by cell, and the held-out minus train gap.
  python seed_var.py human_dwa30t --models c30e c30es1 c30es2 c30es3 --train human_dwa30/train.jsonl [--drop-top 1]
      [--out ../results/seed_var30.txt]
"""
import argparse
import bisect
import collections
import itertools
import json
import re
from pathlib import Path

import numpy as np

from curve_eval import crossfit, labels, load_rows, nll, softmax
from deep_checks import BANDS, NAMES

QS = ("ttd", "depth", "end", "ascends")
DEEP = ("21-24", "25-29")
NBOOT = 500


def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: (a * (np.log(np.clip(a, 1e-12, 1)) - np.log(np.clip(b, 1e-12, 1)))).sum(-1)
    return (kl(p, m) + kl(q, m)) / 2


def train_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        dl = re.search(r"\bDlvl (\d+)", r["state"])
        rows.append({"id": r["id"], "player": r["id"].split("@")[0].lower(), "q": r["questions"],
                     "gold": r["gold"], "dlvl": int(dl.group(1)) if dl else -1})
    return rows


def zs(df, rows, sel, q):
    lg = df[df.question == q].set_index("id").logits
    return np.stack([np.asarray(lg[rows[i]["id"]], float) for i in sel])


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--train", default="")
    ap.add_argument("--drop-top", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    M = a.models
    pairs = list(itertools.combinations(range(len(M)), 2))
    rows = load_rows(a.data)
    out = [f"{a.data}: seeds {', '.join(M)}"]
    if a.drop_top:
        c = collections.Counter(r["player"] for r in rows if r["asc"])
        top = {p for p, _ in c.most_common(a.drop_top)}
        rows = [r for r in rows if r["player"] not in top]
        out.append(f"left out: the {a.drop_top} player(s) with the most ascended rows")
    out.append(f"{len(rows)} rows, {len({r['player'] for r in rows})} players")
    dfs = {m: pd.concat([pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{s}.parquet") for s in ("val", "test")])
           for m in M}

    cell = {}
    for q in QS:
        sel, y, _ = labels(rows, q)
        pl = np.array([rows[i]["player"] for i in sel])
        band = np.array([bisect.bisect_left(BANDS, rows[i]["dlvl"]) for i in sel])
        z = {m: zs(dfs[m], rows, sel, q) for m in M}
        for b, name in enumerate(NAMES):
            mk = band == b
            if mk.sum() < 20:
                continue
            cell[q, name] = {"pl": pl[mk], "y": y[mk], "z": [z[m][mk] for m in M],
                             "loc": [crossfit(z[m][mk], y[mk], pl[mk])[0] for m in M],
                             "raw": [nll(z[m][mk], y[mk]) for m in M]}

    players = sorted({p for c in cell.values() for p in c["pl"]})
    pidx = {p: i for i, p in enumerate(players)}
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(players), len(players)), minlength=len(players))
                  for _ in range(NBOOT)])
    for c in cell.values():
        c["codes"] = np.array([pidx[p] for p in c["pl"]])
        c["n"] = np.bincount(c["codes"], minlength=len(players))

    def boot_mean(c, x):
        s = np.bincount(c["codes"], weights=x, minlength=len(players))
        return (W @ s) / np.maximum(W @ c["n"], 1)

    out.append("\n==== cells: T-band loss per seed; SD over seeds; test noise of one seed pair (bootstrap SD of the pair "
               "difference, mean over pairs); ratio; raw SD over seeds")
    out.append(f"  {'q':8s}{'band':7s}{'rows':>7s}  " + " ".join(f"{m:>8s}" for m in M)
               + "    SD seeds  SE pair   ratio   SD raw")
    for (q, name), c in cell.items():
        loc = np.array([x.mean() for x in c["loc"]])
        raw = np.array([x.mean() for x in c["raw"]])
        se = np.mean([boot_mean(c, c["loc"][j] - c["loc"][i]).std(ddof=1) for i, j in pairs])
        sd = loc.std(ddof=1)
        out.append(f"  {q:8s}{name:7s}{len(c['y']):7d}  " + " ".join(f"{x:8.4f}" for x in loc)
                   + f"    {sd:.4f}    {se:.4f}  {sd / se:6.2f}   {raw.std(ddof=1):.4f}")

    gcl = [(q, n) for q in QS for n in DEEP if (q, n) in cell]
    out.append("\n==== G seed pairs: difference, its test noise (bootstrap over players of the 8 cells together), "
               "players carrying half of the sum of |per-player contributions| and their share of the G rows")
    for kind in ("loc", "raw"):
        for i, j in pairs:
            contrib = np.zeros(len(players))
            rowsof = np.zeros(len(players))
            bd = np.zeros(NBOOT)
            for key in gcl:
                c = cell[key]
                d = c[kind][j] - c[kind][i]
                contrib += np.bincount(c["codes"], weights=d, minlength=len(players)) / len(d) / len(gcl)
                rowsof += c["n"] / len(d) / len(gcl)
                bd += boot_mean(c, d) / len(gcl)
            D = contrib.sum()
            order = np.argsort(-np.abs(contrib))
            cum = np.cumsum(np.abs(contrib[order])) / np.abs(contrib).sum()
            k = int(np.searchsorted(cum, 0.5)) + 1
            top5 = contrib[order[:5]].sum()
            out.append(f"  {'T-band' if kind == 'loc' else 'raw':6s} {M[j]} - {M[i]}: {D:+.4f}  SE_test {bd.std(ddof=1):.4f}"
                       f"  half of |contrib| in {k} players ({rowsof[order[:k]].sum():.1%} of rows)"
                       f"  top-5 players {top5:+.4f}  rest {D - top5:+.4f}")

    out.append("\n==== disagreement per cell, mean over seed pairs: rows -- mean JS between the two seeds' raw "
               "distributions; band -- JS between their band-mean distributions; mean |dlog p of the true option|")
    for (q, name), c in cell.items():
        P = [softmax(z) for z in c["z"]]
        rj = np.mean([js(P[i], P[j]).mean() for i, j in pairs])
        bj = np.mean([float(js(P[i].mean(0), P[j].mean(0))) for i, j in pairs])
        dl = np.mean([np.abs(c["raw"][i] - c["raw"][j]).mean() for i, j in pairs])
        out.append(f"  {q:8s}{name:7s} rows {rj:.5f}  band {bj:.5f}  band/rows {bj / rj:.3f}  |dlog p| {dl:.4f}")

    if a.train:
        tr = train_rows(Path(a.train))
        tdf = {m: pd.read_parquet(f"preds/{m}-{Path(a.train).parent.name}-{Path(a.train).stem}.parquet") for m in M}
        out.append(f"\n==== own train rows ({len(tr)}): raw loss per seed; held-out raw loss minus train raw loss per seed")
        for q in QS:
            sel = [i for i, r in enumerate(tr) if q in r["gold"]]
            crit = tr[sel[0]]["q"][q]["criteria"]
            keys = list(crit) if isinstance(crit, dict) else [str(k) for k in range(len(crit))]
            y = np.array([keys.index(tr[i]["gold"][q]["label"]) for i in sel])
            band = np.array([bisect.bisect_left(BANDS, tr[i]["dlvl"]) for i in sel])
            loss = [nll(zs(tdf[m], tr, sel, q), y) for m in M]
            for b, name in enumerate(NAMES):
                mk = band == b
                if (q, name) not in cell or mk.sum() < 20:
                    continue
                t = np.array([x[mk].mean() for x in loss])
                h = np.array([x.mean() for x in cell[q, name]["raw"]])
                out.append(f"  {q:8s}{name:7s}{mk.sum():7d}  train " + " ".join(f"{x:.4f}" for x in t)
                           + "  gap " + " ".join(f"{x:+.4f}" for x in h - t)
                           + f"  SD train {t.std(ddof=1):.4f}  SD gap {(h - t).std(ddof=1):.4f}")

    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
