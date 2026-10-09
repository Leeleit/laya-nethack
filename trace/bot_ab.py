"""Paired A/B of two bot runs on the same seeds (a jf_config flag on vs off): games are deterministic, so a
seed's two games are the same until the flag first acts and every difference is the flag's.
  python bot_ab.py <base run dir> <arm run dir> --profiles <base.parquet> <arm.parquet> [--out results/ab.txt]
                   [--endgame <base.parquet> <arm.parquet>]
run dir: evidence.json (collect.py); profiles: bot_profile.py's per-arrival parquet (with the seed column);
endgame: bot_endgame.py's per-game parquet (milestones: Medusa passed, castle reached / crossed, Gehennom).
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from bot_profile import cause


def results(run):
    return {r["trajectory_id"]: r for r in json.load(open(Path(run, "evidence.json")))["results"]}


def paired(d):
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return f"{m:+.4f} (se {se:.4f}, t {m / se:+.2f})" if se else f"{m:+.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("arm")
    ap.add_argument("--profiles", nargs=2, required=True)
    ap.add_argument("--endgame", nargs=2)
    ap.add_argument("--out", default="results/ab.txt")
    a = ap.parse_args()
    B, R = results(a.base), results(a.arm)
    seeds = sorted(set(B) & set(R))
    g = pd.DataFrame([dict(seed=s, p0=B[s]["progress"], p1=R[s]["progress"], d0=B[s]["max_depth"],
                           d1=R[s]["max_depth"], s0=B[s]["steps"], s1=R[s]["steps"],
                           c0=cause(B[s].get("cause_of_death")), c1=cause(R[s].get("cause_of_death")))
                      for s in seeds])
    ch = g[(g.s0 != g.s1) | (g.p0 != g.p1)]
    out = [f"A/B {a.base} -> {a.arm}: {len(g)} paired seeds, {len(ch)} games changed", "",
           f"progress {g.p0.mean():.4f} -> {g.p1.mean():.4f}, paired diff {paired(g.p1 - g.p0)}",
           f"changed games only: {ch.p0.mean():.4f} -> {ch.p1.mean():.4f}; better {(ch.p1 > ch.p0).sum()}, "
           f"worse {(ch.p1 < ch.p0).sum()}, same {(ch.p1 == ch.p0).sum()}",
           f"max depth median {g.d0.median():.0f} -> {g.d1.median():.0f}; reached Dlvl 25+ {(g.d0 >= 25).mean():.1%}"
           f" -> {(g.d1 >= 25).mean():.1%}; ended on Dlvl 21-24 (Medusa's band) {g.d0.between(21, 24).sum()} -> "
           f"{g.d1.between(21, 24).sum()}", ""]
    out.append("end causes in changed games (base -> arm counts)")
    c0, c1 = Counter(ch.c0), Counter(ch.c1)
    for c in sorted(set(c0) | set(c1), key=lambda c: -(c0[c] + c1[c]))[:15]:
        out.append(f"  {c:32} {c0[c]:4d} -> {c1[c]:4d}")
    P0, P1 = (pd.read_parquet(p) for p in a.profiles)
    P0, P1 = P0[P0.seed.isin(seeds)], P1[P1.seed.isin(seeds)]
    j = P0.merge(P1, on=["seed", "depth"], suffixes=("0", "1"))
    out += ["", "first arrival at Dlvl d, seeds that reached it in both runs: AC median base -> arm, paired mean "
            "AC diff (negative = better), share of games with better AC"]
    for d in (4, 6, 8, 10, 12, 15, 18, 20, 22, 24, 25, 26, 27):
        s = j[j.depth == d]
        if len(s) < 10:
            continue
        diff = s.ac1 - s.ac0
        out.append(f"  Dlvl {d:2d}  n {len(s):4d}  AC {s.ac0.median():3.0f} -> {s.ac1.median():3.0f}  "
                   f"diff {paired(diff)}  better {(diff < 0).mean():.0%}")
    if a.endgame:
        E0, E1 = (pd.read_parquet(p).set_index("seed") for p in a.endgame)
        e = E0.join(E1, lsuffix="0", rsuffix="1", how="inner")
        e = e[e.index.isin(seeds)]
        out += ["", f"milestones ({len(e)} paired games): share base -> arm, paired diff"]
        for col in ("reached_medusa", "passed_medusa", "reached_castle", "crossed_castle", "gehennom",
                    "gehennom_levelport"):
            x0, x1 = e[col + "0"].astype(float), e[col + "1"].astype(float)
            out.append(f"  {col:20} {x0.mean():6.1%} -> {x1.mean():6.1%}  {paired(x1 - x0)}")
        out.append("  where games end (base -> arm): " + "; ".join(
            f"{k} {(e.section0 == k).sum()} -> {(e.section1 == k).sum()}" for k in
            sorted(set(e.section0) | set(e.section1), key=lambda k: -(e.section0 == k).sum())))
    txt = "\n".join(out)
    Path(a.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
