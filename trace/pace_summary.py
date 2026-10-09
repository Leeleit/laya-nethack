"""Summary of ttyrec_pace.py over every train player: where bot-named accounts (bot / taeb in the name)
sit against everyone else, how many accounts and train games each candidate rule takes.
  python pace_summary.py ../results/ttyrec_pace_14d.txt [--out ../results/pace_summary_14d.txt]
"""
import argparse
from pathlib import Path

import pandas as pd

COLS = "p train files ev med p90 p99 g5 g30 idle".split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pace")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = [ln.split() for ln in open(a.pace, encoding="utf-8")]
    d = pd.DataFrame([r for r in rows if len(r) == 10 and r[0] != "player"], columns=COLS)
    d[COLS[1:]] = d[COLS[1:]].astype(float)
    named = d.p.str.contains("bot|taeb")
    out = [f"{len(d)} players, {named.sum()} named bot/taeb ({int(d[named].train.sum())} train games)"]
    for title, s in (("named", d[named]), ("not named, >= 20000 events", d[~named & (d.ev >= 20000)]),
                     ("not named, all", d[~named])):
        out += [f"== {title}: {len(s)} players, quantiles",
                s[["med", "p90", "p99", "g5", "g30", "idle"]].quantile([.01, .05, .1, .25, .5, .75]).round(4).to_string()]
    rules = {"g30 == 0 & g5 <= 0.005": (d.g30 == 0) & (d.g5 <= 0.005),
             "g30 == 0 & g5 <= 0.001": (d.g30 == 0) & (d.g5 <= 0.001),
             "p99 <= 1 & g5 <= 0.001": (d.p99 <= 1) & (d.g5 <= 0.001),
             "g30 == 0 & g5 <= 0.005 & ev >= 20000": (d.g30 == 0) & (d.g5 <= 0.005) & (d.ev >= 20000)}
    out.append("== rules: accounts (named / not named), train games (named / not named)")
    for k, c in rules.items():
        out.append(f"  {k:40s} {c.sum():5d} ({(c & named).sum()} / {(c & ~named).sum()})  "
                   f"{int(d[c].train.sum())} ({int(d[c & named].train.sum())} / {int(d[c & ~named].train.sum())})")
    out.append("== named accounts the first rule misses")
    out.append(d[named & ~rules["g30 == 0 & g5 <= 0.005"]].sort_values("train", ascending=False).to_string(index=False))
    out.append("== not named, g30 == 0 & g5 <= 0.005, by train games")
    out.append(d[rules["g30 == 0 & g5 <= 0.005"] & ~named].sort_values("train", ascending=False).head(60)
               .to_string(index=False))
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
