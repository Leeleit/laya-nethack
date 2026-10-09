"""plan.json for forkrun.py from plan_branches.py's points.parquet (all points of a game in one entry).

  python fork_plan.py --points plans_val/points.parquet --out fork_val/plan.json --reps 4 [--seeds 100928,100515]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

OPTIONS = ["continue", "pray", "elbereth", "quaff_heal", "stairs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--identity", default="val-dwa-law-fem")
    ap.add_argument("--solution", default="/data/laya/nethack/nethackers/leader-traced")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--h", type=int, default=300)
    ap.add_argument("--options", default=",".join(OPTIONS))
    ap.add_argument("--seeds", default="")
    ap.add_argument("--trace", action="store_true", help="also keep the parents' full traces")
    a = ap.parse_args()
    d = pd.read_parquet(a.points)
    d = d[d.identity == a.identity]
    if a.seeds:
        d = d[d.seed.isin([int(s) for s in a.seeds.split(",")])]
    games = [{"seed": int(seed), "points": [{"point": int(r.point), "k": int(r.k), "b": [int(v) for v in r.b]}
                                            for r in g.sort_values("k").itertuples()]}
             for seed, g in d.groupby("seed")]
    plan = {"identity": a.identity, "solution": a.solution, "options": a.options.split(","), "reps": a.reps,
            "h": a.h, "trace": a.trace, "games": games}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(plan))
    print(a.out, len(games), "games", sum(len(g["points"]) for g in games), "points")


if __name__ == "__main__":
    main()
