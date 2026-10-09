"""Branch points for the counterfactual collection: risk alarms of every recorded game.

Alarm = P(death within 7 turns) from a gradient-boosting model on the dataset's tabular features (build_v2.py)
above the threshold that flags ALARM_BUDGET of the safe (death > 100 turns away) training states -- the bot's
own HP rule's rate (alarm_v2.py). GBM rather than laya: as good at this horizon (test AUC 0.967 vs 0.958) and it
scores every state on the CPU in seconds. Up to --points alarms per game, each at least HORIZON turns after the
previous one so the branches' horizons do not overlap; pass i of the collection branches the i-th point of
every game. A game keeps its train/val/test split.

  python plan_branches.py [--root R --data D --plans P --points N]
      -> P/points.parquet, P/<option>/p<i>/<run>.json (collect.py --branch)
  python plan_branches.py --plans P --filter <branch runs prefix>
      after a pray pass: drop the quaff_heal/stairs branches its logs show unavailable (no log entry: kept)
"""
import argparse
import glob
import json
import os
import subprocess
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

D = "/data/laya/nethack/trace/"
OPTIONS = ["pray", "elbereth", "quaff_heal", "stairs"]
HORIZON = 300  # game turns after firing; the branch run's result is the state then
ALARM_BUDGET = 0.0162
TIME = 20
WORKERS = 16  # CPU processes: 16 on a shared 32-vCPU server gave the throughput of 20; 24: -40 %, 28: -90 %
NUM = ["hp_frac", "maxhp", "depth", "xl", "ac", "hunger", "stack", "hp_drop10", "hp_drop30", "hostiles",
       "hostiles_adjacent"]


def locate(job):
    """(runs root, run, trace file, alarm turns) -> seed, b0 and per turn the step index k of its first record
    and the blstats there."""
    root, run, name, turns = job
    first, n, b0, last = {}, 0, None, None
    # streamed: a long game's trace is ~40 MB of JSON, ~10x that as parsed records
    with subprocess.Popen(["zstdcat", f"{root}/{run}/trace/{name}"], stdout=subprocess.PIPE) as zc:
        next(zc.stdout)  # header
        for i, line in enumerate(zc.stdout):
            r = json.loads(line)
            last, n = r["b"], i + 1
            b0 = b0 or last
            if "scr" in r and last[TIME] in turns and last[TIME] not in first:
                first[last[TIME]] = (i, last)
    if name.split(".")[0].isdigit():  # <seed>.jsonl.zst
        return {"seed": int(name.split(".")[0]), "b0": b0, "at": first}
    results = json.load(open(f"{root}/{run}/results.json"))
    # older traces are named <pid>-<episode>: the seed is the result with the same step count, ties broken by the last turn
    cand = [r for r in results if r["steps"] == n]
    if len(cand) > 1:
        gap = [abs(r["turns"] - last[TIME]) for r in cand]
        cand = [r for r, g in zip(cand, gap) if g == min(gap)]
    return {"seed": cand[0]["trajectory_id"] if len(cand) == 1 else None, "b0": b0, "at": first}


def pick(turns):
    out = []
    for t in sorted(turns):
        if not out or t >= out[-1] + HORIZON:
            out.append(t)
    return out


def plan(a):
    f = pd.read_parquet(f"{a.data}/features.parquet")
    f["run"] = f.id.str.split("/").str[0]
    f["file"] = f.id.str.split("/").str[1].str.split("#").str[0]
    f["turn"] = f.id.str.split("#").str[1].astype(int)
    tr = (f.split == "train").to_numpy()
    top = f[tr].strategy.value_counts().index[:30]
    X = f[NUM].astype(float).assign(strategy=pd.Categorical(f.strategy.where(f.strategy.isin(top), "other")).codes)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0,
                                         categorical_features=[len(NUM)], random_state=0)
    gbm.fit(X[tr], (f.ttd[tr] <= 7).astype(int))
    f["p7"] = gbm.predict_proba(X)[:, 1]
    tau = float(np.quantile(f.p7[tr & (f.ttd > 100).to_numpy()], 1 - ALARM_BUDGET))
    al = f[f.p7 > tau]
    games = al.groupby(["run", "file"]).turn.apply(lambda t: pick(t.tolist()))
    games = games[games.str.len() > 0]
    with Pool(WORKERS) as pool:
        loc = pool.map(locate, [(a.root, r, fl, set(t)) for (r, fl), t in games.items()], chunksize=4)
    meta = f.drop_duplicates(["run", "file"]).set_index(["run", "file"])
    at = al.set_index(["run", "file", "turn"])
    rows = []
    for ((run, name), turns), lc in zip(games.items(), loc):
        if lc["seed"] is None:
            continue
        for i, t in enumerate(turns[:a.points]):
            k, b = lc["at"][t]
            ff = at.loc[(run, name, t)]
            rows.append({"run": run, "file": name, "split": meta.loc[(run, name)].split, "identity": meta.loc[(run, name)].identity,
                         "point": i, "turn": t, "ttd": int(ff.ttd), "p7": float(ff.p7), "seed": lc["seed"], "k": k,
                         "b": b, "b0": lc["b0"]})
    pts = pd.DataFrame(rows)
    os.makedirs(a.plans, exist_ok=True)
    pts.to_parquet(f"{a.plans}/points.parquet")
    for opt in OPTIONS:
        for (i, run), g in pts.groupby(["point", "run"]):
            os.makedirs(f"{a.plans}/{opt}/p{i}", exist_ok=True)
            json.dump([{"seed": int(r.seed), "k": int(r.k), "b": list(map(int, r.b)), "b0": list(map(int, r.b0)),
                        "option": opt, "h": HORIZON} for r in g.itertuples()], open(f"{a.plans}/{opt}/p{i}/{run}.json", "w"))
    print(f"tau {tau:.4f}; games {f.groupby(['run', 'file']).ngroups}, with an alarm {len(games)}; "
          f"{len(pts)} points, per pass {pts.point.value_counts().sort_index().to_dict()}, split {pts.split.value_counts().to_dict()}; "
          f"turns to death at the point: median {pts.ttd.median():.0f}")


def filter_plans(a):
    avail = {}  # (pass, run, seed) -> options available at the branch point, from the pray pass
    for p in glob.glob(f"{a.filter}-pray-p*/*/branch_log.jsonl"):
        i, run = int(p.split("-pray-p")[1].split("/")[0]), os.path.basename(os.path.dirname(p))
        for r in map(json.loads, open(p)):
            if "avail" in r:
                avail[i, run, r["seed"]] = set(r["avail"])
    for opt in ("quaff_heal", "stairs"):
        kept = total = 0
        for p in glob.glob(f"{a.plans}/{opt}/p*/*.json"):
            i, run = int(p.split("/p")[-1].split("/")[0]), os.path.basename(p)[:-5]
            spec = json.load(open(p))
            keep = [e for e in spec if opt in avail.get((i, run, e["seed"]), {opt})]
            kept, total = kept + len(keep), total + len(spec)
            json.dump(keep, open(p, "w"))
        print(f"{opt}: kept {kept} of {total} branches ({len(avail)} points with availability known)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/laya/nethack/nethackers/runs/traces2", help="recorded runs")
    ap.add_argument("--data", default=D + "v2_data", help="build_v2.py output (features.parquet)")
    ap.add_argument("--plans", default=D + "plans")
    ap.add_argument("--points", type=int, default=1, help="branch points per game")
    ap.add_argument("--filter", default=None, metavar="BRANCH_PREFIX", help="e.g. runs/vb1 (runs/vb1-pray-p0/...)")
    a = ap.parse_args()
    filter_plans(a) if a.filter else plan(a)
