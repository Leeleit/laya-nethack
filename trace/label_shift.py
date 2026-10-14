"""Label shares by question and Dlvl band, train against held-out (val + test), with players, games and rows per band.
TV = total variation distance between the two shares; a model sharpened on train pays for it where TV is large.
  python label_shift.py human_dwa30/train.jsonl human_dwa30t [--drop-top 1] [--out ../results/label_shift30.txt]
"""
import argparse
import bisect
import collections
from pathlib import Path

import numpy as np

from curve_eval import load_rows
from deep_checks import BANDS, NAMES
from seed_var import QS, train_rows


def shares(rows, q):
    sel = [r for r in rows if q in r["gold"]]
    crit = sel[0]["q"][q]["criteria"]
    keys = list(crit) if isinstance(crit, dict) else [str(k) for k in range(len(crit))]
    out = {}
    for b, name in enumerate(NAMES):
        rs = [r for r in sel if bisect.bisect_left(BANDS, r["dlvl"]) == b]
        if not rs:
            continue
        c = collections.Counter(r["gold"][q]["label"] for r in rs)
        out[name] = (np.array([c[k] for k in keys], float) / len(rs), len(rs), len({r["player"] for r in rs}),
                     len({r["id"].split("#")[0] for r in rs}))
    return keys, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("held")
    ap.add_argument("--drop-top", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tr = train_rows(Path(a.train))
    he = load_rows(a.held)
    if a.drop_top:
        c = collections.Counter(r["player"] for r in he if r["asc"])
        top = {p for p, _ in c.most_common(a.drop_top)}
        he = [r for r in he if r["player"] not in top]
    out = [f"{a.train} against {a.held}: label shares per band (train | held), TV; rows / players / games"]
    for q in QS:
        keys, st = shares(tr, q)
        _, sh = shares(he, q)
        out.append(f"\n  {q}: options {' '.join(keys)}")
        for name in NAMES:
            if name not in st or name not in sh:
                continue
            pt, nt, plt, gt = st[name]
            ph, nh, plh, gh = sh[name]
            tv = 0.5 * np.abs(pt - ph).sum()
            out.append(f"  {name:6s} TV {tv:.3f}  train " + " ".join(f"{x:.3f}" for x in pt) + "  | held "
                       + " ".join(f"{x:.3f}" for x in ph) + f"   train {nt}/{plt}/{gt}  held {nh}/{plh}/{gh}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
