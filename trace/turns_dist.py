"""Where start-scums end and real games begin (a "real game" for the player's experience / result): NAO
games' turns -- share at or under each threshold, what such games end with, and the ascension rate of the accounts'
later games when counting games of > T turns only.
  python turns_dist.py [--out ../results/turns_dist.txt]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GAMES = "/data/laya/nethack/data/games.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = pd.read_parquet(GAMES, columns=["source", "death", "turns", "maxlvl"])
    d = d[(d.source == "nao") & d.turns.notna()]
    t = d.turns.to_numpy()
    out = [f"NAO games with turns: {len(d)}"]
    for th in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        sel = t <= th
        top = d.death[sel].value_counts(normalize=True).head(3)
        out.append(f"  turns <= {th:5d}: {sel.mean():6.2%}   maxlvl 1 {np.mean(d.maxlvl[sel] <= 1):6.1%}   top ends: "
                   + ", ".join(f"{k[:24]} {v:.0%}" for k, v in top.items()))
    hist = np.histogram(np.log10(t[t >= 1]), bins=np.arange(0, 5.6, 0.2))
    out.append("histogram of log10(turns): " + "  ".join(f"{e:.1f}:{c / len(t):.1%}" for e, c in zip(hist[1], hist[0])))
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
