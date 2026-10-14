"""P(ttd <= 1000) per Dlvl (CPU, from ready predictions): the observed share against the models' mean and the mean old
target, with the share of rows at that Dlvl -- is a model's band miss a rare hump smoothed toward its neighbours (peak
lower and wider, too low at the peak and too high past it) or a level shift? Dlvl above --max pooled into one row.
  python ttd_curve.py human_dwa24/train.jsonl --tag human_dwa24-train --models c17e c17es1 c17es2 c17es3 gbmcf \
      --out ../results/ttd_curve24.txt
  python ttd_curve.py human_dwa20t/val.jsonl human_dwa20t/test.jsonl --tag human_dwa20t-val human_dwa20t-test \
      --models c17e c17es1 c17es2 c17es3 gbmcf --drop-top 1 --out ../results/ttd_curve20t_drop1.txt"""
import argparse
import collections
from pathlib import Path

import numpy as np

from leak_total import probs
from ttd_miss import rows_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--drop-top", type=int, default=0)
    ap.add_argument("--min", type=int, default=13)
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = rows_of(a.data)
    keep = np.ones(len(d["id"]), bool)
    if a.drop_top:  # as deep_checks.py: the players with the most ascended rows
        cnt = collections.Counter(p for p, v in zip(d["player"], d["ascend"]) if v)
        top = [p for p, _ in cnt.most_common(a.drop_top)]
        keep = np.array([p not in top for p in d["player"]])
    hi = d["hi"][0]
    ps = {m: probs([f"preds/{m}-{t}.parquet" for t in a.tag], "ttd", d["id"])[:, :hi].sum(1) for m in a.models}
    lv = np.minimum(d["dlvl"], a.max)
    out = [f"{' '.join(a.data)}; P(ttd <= 1000) per Dlvl (>= {a.max} pooled): observed / mean old target / models"
           + ("; left out the top player(s)" if a.drop_top else "")]
    out.append(f"  {'Dlvl':>5}{'rows':>7}{'share':>7}  {'obs':>6}  {'target':>6}  " + "  ".join(f"{m:>8}" for m in a.models))
    n = keep.sum()
    for x in range(a.min, a.max + 1):
        g = keep & (lv == x)
        if not g.any():
            continue
        out.append(f"  {x:>5}{g.sum():7d}{g.sum() / n:7.3f}  {d['short'][g].mean():6.3f}  {d['target'][g].mean():6.3f}  "
                   + "  ".join(f"{ps[m][g].mean():8.3f}" for m in a.models))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
