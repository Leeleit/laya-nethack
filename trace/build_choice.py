"""Counterfactual dataset from the branch runs (collect.py --branch): what each emergency option does at a risk
alarm, against the bot carrying on.

For every branch point (plan_branches.py) and every option that ran there: is the hero alive HORIZON turns
after the decision, and how deep did it get. "continue" is the recorded game itself from the same state
(the games are deterministic, so it needs no rerun). Rows are risk-v1-style `noul` questions, one per
option, on the same state text as risk-v2 (build_v2.walk/state_text):
  "If the bot <does X> now, is the hero still alive 300 game turns later?"
One rollout per option: a label is one draw of a noisy outcome, so models are judged on held-out games, and
the per-option survival rates are paired comparisons on the same points.
  python build_choice.py --root R --plans P --branches B --out O
      -> O/{train,val,test}.jsonl, O/outcomes.parquet; prints paired rates
"""
import argparse
import json
import os
import subprocess
from multiprocessing import Pool

import pandas as pd

from build_laya_data import DEPTH, TIME
from build_v2 import state_text, walk
from plan_branches import HORIZON, OPTIONS, WORKERS

ACTION = {"continue": "keeps playing as it does now",
          "pray": "prays to its god now",
          "elbereth": "engraves Elbereth in the dust now",
          "quaff_heal": "drinks a known healing potion now",
          "stairs": "takes the staircase it is standing on now",
          "elbereth_rest": "engraves Elbereth and rests on it until healed or left alone",
          "upstairs": "walks to the nearest up staircase and climbs it",
          "dig_down": "digs a hole down through the floor to the next level"}


def question(option):
    return {"type": "noul",
            "instructions": f"If the bot {ACTION[option]}, is the hero still alive {HORIZON} game turns later?"}


def control(job):
    """State texts at a game's branch points and the recorded game's outcome from each."""
    root, run, name, ks, identity = job
    lines = subprocess.run(["zstdcat", f"{root}/{run}/trace/{name}"], capture_output=True,
                           check=True).stdout.splitlines()[1:]
    open_ = {}  # k -> [state, t0, max depth so far]
    for idx, r, back, msgs, inv in walk(lines):
        if idx in ks:
            open_[idx] = [state_text(identity, r["s"], r, back, list(msgs), inv), r["b"][TIME], 0]
        for v in open_.values():
            if r["b"][TIME] <= v[1] + HORIZON:
                v[2] = max(v[2], r["b"][DEPTH])
    return open_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="recorded runs the points come from")
    ap.add_argument("--plans", required=True, help="plan_branches.py output")
    ap.add_argument("--branches", required=True, help="branch runs prefix: <prefix>-<option>-p<i>/<run>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pts = pd.read_parquet(f"{a.plans}/points.parquet")
    orig = {}
    for run in pts.run.unique():
        for r in json.load(open(f"{a.root}/{run}/results.json")):
            orig[run, r["trajectory_id"]] = r
    games = pts.groupby(["run", "file"])
    jobs = [(a.root, run, name, set(g.k), g.identity.iloc[0]) for (run, name), g in games]
    with Pool(WORKERS) as pool:
        ctl = dict(zip([j[1:3] for j in jobs], pool.map(control, jobs, chunksize=4)))
    out = []
    for p in pts.itertuples():
        state, t0, depth = ctl[p.run, p.file][p.k]
        o = orig[p.run, p.seed]
        out.append({"run": p.run, "seed": p.seed, "point": p.point, "split": p.split, "option": "continue",
                    "turn": t0, "alive": o["turns"] > t0 + HORIZON, "depth": depth, "state": state})
    for opt in OPTIONS:
        for i, run in pts[["point", "run"]].drop_duplicates().itertuples(index=False):
            d = f"{a.branches}-{opt}-p{i}/{run}"
            if not os.path.exists(f"{d}/evidence.json"):
                continue
            res = {r["trajectory_id"]: r for r in json.load(open(f"{d}/results.json"))}
            logs = {}  # a seed replayed after an interrupted collection logs again: its last entry counts
            for lg in map(json.loads, open(f"{d}/branch_log.jsonl")):
                logs[lg.get("seed")] = lg
            for lg in logs.values():
                r = res.get(lg.get("seed"))  # "no match" entries carry "seeds"
                if not lg.get("done") or r is None or r["status"] != "completed":
                    continue
                died = r["cause_of_death"] is not None and r["turns"] <= lg["turn"] + HORIZON
                out.append({"run": run, "seed": lg["seed"], "point": i, "option": opt, "turn": lg["turn"],
                            "alive": not died, "depth": r["max_depth"]})
    df = pd.DataFrame(out)
    first = df[df.option == "continue"].set_index(["run", "seed", "point"])
    k3 = list(zip(df.run, df.seed, df.point))
    df["split"] = [first.split.get(k) for k in k3]
    df["state"] = [first.state.get(k) for k in k3]
    df = df[df.state.notna()]
    os.makedirs(a.out, exist_ok=True)
    df.drop(columns="state").to_parquet(f"{a.out}/outcomes.parquet")
    for split in ("train", "val", "test"):
        with open(f"{a.out}/{split}.jsonl", "w") as fh:
            for (run, seed, i), g in df[df.split == split].groupby(["run", "seed", "point"]):
                fh.write(json.dumps({
                    "id": f"{run}#{seed}#{i}", "state": g.state.iloc[0],
                    "questions": {o: question(o) for o in g.option},
                    "gold": {o: {"probabilities": {"true": float(v), "false": float(1 - v)}, "label": str(bool(v)).lower()}
                             for o, v in zip(g.option, g.alive)}}) + "\n")
    wide = df.pivot_table(index=["run", "seed", "point"], columns="option", values="alive", aggfunc="first")
    for opt in OPTIONS:
        if opt in wide:
            m = wide[opt].notna()
            print(f"{opt:10s} ran at {m.sum():5d} points: alive after {HORIZON} turns {wide[opt][m].mean():.3f} "
                  f"vs continue {wide['continue'][m].mean():.3f}; better {((wide[opt] > wide['continue']) & m).sum()}, "
                  f"worse {((wide[opt] < wide['continue']) & m).sum()}")
    print(df.groupby("split").size().to_dict(), "rows;", len(wide), "points")


if __name__ == "__main__":
    main()
