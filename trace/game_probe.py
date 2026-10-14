"""Delta_game, next to mem_gate.py: does the teacher know the train games better than new games, beyond what a reference
that never saw those games does?
  Delta_game = [teacher(unseen) - teacher(new)] - [R(unseen) - R(new)]
unseen: Dlvl 21-29 states of train games the teacher never saw (human_mem29/unseen.jsonl; "same games" = the games with
a seen row in that band, as in mem_gate.py); new: Dlvl 21-29 rows of the held-out players (human_dwa20t val + test,
without the --drop-top players with the most ascended rows, as in deep_checks.py). Teacher = the mean softmax of its
seeds. Raw log loss, and under one temperature per model and question fitted on the new rows; the mean over each
group's rows per (question, band); bootstrap over the players of both groups at once (they are disjoint).
Main: the mean over ttd, end, ascends x 21-24, 25-29 (depth is left out: the teacher and o17 learnt different depth
targets); G over the 8 cells next to it.
  python game_probe.py --teacher c17e c17es1 c17es2 c17es3 --refs o17 gbmcf --drop-top 1 --out ../results/game29.txt
predictions: preds/<m>-human_mem29-unseen.parquet and preds/<m>-human_dwa20t-{val,test}.parquet for every model."""
import argparse
import bisect
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from curve_eval import QS, labels, load_rows, softmax
from mem_gate import BANDS, DEEP, game, rows_of

MAIN = ("ttd", "end", "ascends")
TS = np.exp(np.linspace(np.log(0.25), np.log(6.0), 241))


def logp(rows, dfs, q):
    """-> (row indices with q, labels, log of the mean softmax of dfs on those rows)"""
    sel, y, _ = labels(rows, q)
    ps = []
    for df in dfs:
        lg = df[df.question == q].set_index("id").logits
        ps.append(softmax(np.stack([np.asarray(lg[rows[i]["id"]], float) for i in sel])))
    return sel, y, np.log(np.clip(np.mean(ps, 0), 1e-12, 1))


def nll_t(lp, y, t):
    z = lp / t
    z = z - z.max(1, keepdims=True)
    return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", nargs="+", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--seen", default="human_mem29/seen.jsonl")
    ap.add_argument("--unseen", default="human_mem29/unseen.jsonl")
    ap.add_argument("--new", default="human_dwa20t")
    ap.add_argument("--drop-top", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seen, unseen = rows_of(a.seen), rows_of(a.unseen)
    seen_games = {b: {game(r["id"]) for r in seen if r["band"] == b} for b in DEEP}
    new = load_rows(a.new)
    head = []
    if a.drop_top:
        c = collections.Counter(r["player"] for r in new if r["asc"])
        top = {p for p, _ in c.most_common(a.drop_top)}
        head.append(f"left out of the new rows: the {a.drop_top} player(s) with the most ascended rows -- "
                    f"{sum(r['player'] in top for r in new)} rows")
        new = [r for r in new if r["player"] not in top]
    for r in new:
        r["band"] = bisect.bisect_left(BANDS, r["dlvl"]) if r["dlvl"] > 0 else -1
    new = [r for r in new if r["band"] in DEEP]
    groups = {"unseen same games": [r for r in unseen if game(r["id"]) in seen_games[r["band"]]],
              "unseen all": unseen, "new": new}
    players = sorted({r["player"] for rs in groups.values() for r in rs})
    assert not {r["player"] for r in unseen} & {r["player"] for r in new}, "a held-out player among the train games"
    pidx = {p: i for i, p in enumerate(players)}
    rng = np.random.default_rng(0)
    ws = np.array([np.ones(len(players))] + [np.bincount(rng.integers(0, len(players), len(players)),
                                                         minlength=len(players)) for _ in range(1000)])
    uname = f"{Path(a.unseen).parent.name}-{Path(a.unseen).stem}"
    models = sorted(set(a.teacher) | set(a.refs))
    pr = {m: {"unseen": pd.read_parquet(f"preds/{m}-{uname}.parquet"),
              "new": pd.concat([pd.read_parquet(f"preds/{m}-{Path(a.new).name}-{s}.parquet") for s in ("val", "test")])}
          for m in models}
    arms = [("teacher", a.teacher)] + [(m, [m]) for m in a.refs]
    out = head + [f"rows in 21-24 / 25-29: " + "; ".join(
        f"{g} {sum(r['band'] == 3 for r in rs)} / {sum(r['band'] == 4 for r in rs)} "
        f"({len({r['player'] for r in rs})} players)" for g, rs in groups.items()),
        f"teacher = mean softmax of {' '.join(a.teacher)}; raw log loss and under one T per model and question fitted on "
        f"the new rows; 95 % bootstrap over players"]
    cell = {}  # (arm, calib, q, band, group) -> means under every weight
    temps = {}
    for arm, ms in arms:
        for q in QS:
            got = {g: logp(rs, [pr[m]["new" if g == "new" else "unseen"] for m in ms], q) for g, rs in groups.items()}
            _, yn, lpn = got["new"]
            t = TS[np.argmin([nll_t(lpn, yn, t).mean() for t in TS])]
            temps[arm, q] = t
            for g, rs in groups.items():
                sel, y, lp = got[g]
                for calib, tt in (("raw", 1.0), ("T", t)):
                    loss = nll_t(lp, y, tt)
                    for b in DEEP:
                        k = np.array([rs[i]["band"] == b for i in sel])
                        codes = np.array([pidx[rs[i]["player"]] for i in sel[k]])
                        s = np.bincount(codes, weights=loss[k], minlength=len(players))
                        n = np.bincount(codes, minlength=len(players))
                        cell[arm, calib, q, b, g] = (ws * s).sum(1) / (ws * n).sum(1)
    out.append("T on the new rows: " + "; ".join(f"{arm} " + " ".join(f"{q} {temps[arm, q]:.2f}" for q in QS)
                                                 for arm, _ in arms))

    def fmt(d):
        lo, hi = np.percentile(d[1:], [2.5, 97.5])
        return f"{d[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]"

    for ug in ("unseen same games", "unseen all"):
        out.append(f"\n==== {ug} against new")
        for calib in ("raw", "T"):
            out.append(f"  {calib}")
            out.append(f"    {'cell':<16}" + "".join(f"{arm + ' u - n':>30}" for arm, _ in arms))
            gap = {arm: {(q, b): cell[arm, calib, q, b, ug] - cell[arm, calib, q, b, "new"] for q in QS for b in DEEP}
                   for arm, _ in arms}
            for q in QS:
                for b in DEEP:
                    out.append(f"    {q + ' ' + DEEP[b]:<16}" + "".join(f"{fmt(gap[arm][q, b]):>30}" for arm, _ in arms))
            for name, qs in (("main (ttd, end, ascends)", MAIN), ("G (8 cells)", QS)):
                for arm, _ in arms[1:]:
                    d = np.mean([gap["teacher"][q, b] - gap[arm][q, b] for q in qs for b in DEEP], 0)
                    out.append(f"    Delta_game {name}, reference {arm}: {fmt(d)}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
