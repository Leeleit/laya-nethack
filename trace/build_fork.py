"""Choice dataset from forkrun.py output: K outcomes per (branch point, option) instead of one.

Outcome of a branch = is the hero alive h game turns after the decision, for every h in HORIZONS (the children
run 300 turns; a death turn gives all shorter horizons for free). "continue" rep 0 is the parent (the recorded
game); the other reps are reseeded children. Unavailable options are dropped; crashed / timed-out / truncated
children are counted and dropped.

Rows are build_choice-style noul questions on the risk-v2 state text at the point's step k, one per option,
"If the bot X, is the hero still alive H game turns later?", with a SOFT gold: the share of the K outcomes
alive (H = --horizon).
  python build_fork.py --fork F [F2 ...] --plans P --root R --out O [--horizon 300]
      -> O/outcomes.parquet (one row per outcome), O/{train,val,test}.jsonl, O/stats.txt
stats.txt (docs/branches.md): per horizon, each option's paired effect against continue and how much
of the spread of that effect between points is signal rather than outcome noise (from the reps).
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from build_choice import ACTION, control
from plan_branches import WORKERS

HORIZONS = (25, 50, 100, 200, 300)
OPTIONS = ["continue", "pray", "elbereth", "quaff_heal", "stairs", "elbereth_rest", "upstairs", "dig_down"]


def question(option, h):
    return {"type": "noul",
            "instructions": f"If the bot {ACTION[option]}, is the hero still alive {h} game turns later?"}


def choice_question(options, h):
    return {"type": "choice",
            "instructions": f"Which should the bot do now to keep the hero alive for the next {h} game turns?",
            "criteria": {o: ACTION[o] for o in options}}


def p_best(alive, n, draws=4000, seed=0):
    """P(option is the best) under independent Beta(1 + alive, 1 + dead) posteriors of each option's survival
    (cost-sensitive choice: a soft target that stays honest when K is small)."""
    rng = np.random.default_rng(seed)
    s = rng.beta(1 + np.asarray(alive, float)[:, None], 1 + (np.asarray(n) - np.asarray(alive))[:, None],
                 size=(len(n), draws))
    return np.bincount(s.argmax(0), minlength=len(n)) / draws


def outcomes(fork, pts):
    """One row per usable outcome; dropped children counted by status."""
    rows, dropped = [], {}
    key = pts.set_index(["seed", "point"])
    for res in sorted((fork / "results").glob("*.json")):
        g = json.loads(res.read_text())
        seed = g["seed"]
        gd = fork / "games" / str(seed)
        fired = {}
        if (gd / "points.jsonl").exists():
            for p in map(json.loads, (gd / "points.jsonl").read_text().splitlines()):
                if p.get("fired"):
                    fired[p["point"]] = p
        for i, p in fired.items():
            if (seed, i) not in key.index:
                continue
            t0 = p["fire_turn"]
            died = g["cause"] is not None
            rows.append({"seed": seed, "point": i, "option": "continue", "rep": 0, "fire_turn": t0,
                         "died": died, "end_turn": g["turns"], "max_depth": g["max_depth"]})
        for f in gd.glob("p*-r*.json"):
            c = json.loads(f.read_text())
            if c["point"] not in fired:
                continue
            st = c["status"]
            if st not in ("died", "alive"):
                dropped[st] = dropped.get(st, 0) + 1
                continue
            rows.append({"seed": seed, "point": c["point"], "option": c["option"], "rep": c["rep"],
                         "fire_turn": c["fire_turn"], "died": st == "died", "end_turn": c["end_turn"],
                         "max_depth": c["max_depth"]})
    df = pd.DataFrame(rows)
    for h in HORIZONS:
        df[f"alive_{h}"] = ~(df.died & (df.end_turn <= df.fire_turn + h))
    df = df.join(key[["run", "file", "split", "k", "identity"]], on=["seed", "point"])
    return df, dropped


def signal_share(adv):
    """adv: points x reps advantage samples (NaN-padded). Share of the between-point variance of the mean
    advantage that is not rep noise: 1 - E[var_within / n] / var(point means)."""
    n = adv.notna().sum(axis=1)
    ok = n >= 2
    m = adv[ok].mean(axis=1)
    noise = (adv[ok].var(axis=1, ddof=1) / n[ok]).mean()
    total = m.var(ddof=1)
    return (1 - noise / total) if total > 0 else float("nan"), int(ok.sum())


def bootstrap_bound(df, target=300, cuts=(100, 200)):
    """Before spending GPU time on a value model: how much of the label noise of alive@target could a
    bootstrap (alive@cut x V(state at cut)) remove at most? A perfect V removes E[alive_cut * Var(alive@target |
    s_cut)] of the within-point variance; that is at most the tail's Bernoulli variance p_cut * q (1 - q), q the
    point's survival from cut to target among its reps alive at cut (Jensen). Equivalent rep multiplier
    = total noise / remaining noise."""
    out = []
    g = df.groupby(["seed", "point", "option"])
    total = (g[f"alive_{target}"].mean() * (1 - g[f"alive_{target}"].mean())).mean()
    for c in cuts:
        pc = g[f"alive_{c}"].mean()
        alive_c = df[df[f"alive_{c}"]].groupby(["seed", "point", "option"])[f"alive_{target}"].mean()
        tail = (pc.reindex(alive_c.index) * alive_c * (1 - alive_c)).sum() / len(pc)
        out.append(f"  cut {c}: tail noise {tail:.4f} of {total:.4f} ({tail / total:.0%}); a perfect V is worth at "
                   f"most x{total / (total - tail):.2f} reps")
    return f"bootstrap bound for alive@{target} (within-point noise):\n" + "\n".join(out)


def stats(df):
    out = []
    for h in HORIZONS:
        col = f"alive_{h}"
        mean = df.groupby(["seed", "point", "option"])[col].mean().unstack("option")
        out.append(f"horizon {h}: continue alive {mean['continue'].mean():.3f} on {mean['continue'].notna().sum()} points")
        for o in OPTIONS[1:]:
            if o not in mean:
                continue
            both = mean[[o, "continue"]].dropna()
            d = both[o] - both["continue"]
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            # per-rep advantages against the continue mean of the same point
            reps = df[df.option == o].pivot_table(index=["seed", "point"], columns="rep", values=col)
            adv = reps.sub(mean["continue"].reindex(reps.index), axis=0).dropna(how="all")
            share, npts = signal_share(adv)
            out.append(f"  {o:13s} {len(d):5d} points: alive {both[o].mean():.3f} vs {both['continue'].mean():.3f}, "
                       f"effect {d.mean():+.4f} ± {1.96 * se:.4f}; better at {(d > 0).mean():.2f} of points, "
                       f"worse at {(d < 0).mean():.2f}; signal share of the spread {share:.2f} ({npts} points)")
        best = mean.max(axis=1) - mean["continue"]
        out.append(f"  oracle over rep means: {best.mean():+.4f} (an upper bound: rep noise inflates it)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True, nargs="+",
                    help="forkrun.py outputs (runs of other options at the same points; a (point, option, rep) "
                         "found twice is the same deterministic game, the first is kept)")
    ap.add_argument("--plans", required=True, help="plan_branches.py output (points.parquet)")
    ap.add_argument("--root", required=True, help="recorded runs (state texts at the points)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=300, help="horizon of the training questions")
    a = ap.parse_args()
    pts = pd.read_parquet(f"{a.plans}/points.parquet")
    parts = [outcomes(Path(f), pts) for f in a.fork]
    df = pd.concat([p[0] for p in parts]).drop_duplicates(["seed", "point", "option", "rep"]).reset_index(drop=True)
    dropped = {k: sum(p[1].get(k, 0) for p in parts) for p in parts for k in p[1]}
    os.makedirs(a.out, exist_ok=True)
    df.to_parquet(f"{a.out}/outcomes.parquet")
    text = (f"{len(df)} outcomes at {df.groupby(['seed', 'point']).ngroups} points; dropped children {dropped}\n"
            + stats(df) + "\n" + bootstrap_bound(df))
    Path(f"{a.out}/stats.txt").write_text(text + "\n")
    print(text)

    used = pts.merge(df[["seed", "point"]].drop_duplicates(), on=["seed", "point"])
    jobs = [(a.root, run, name, set(g.k), g.identity.iloc[0]) for (run, name), g in used.groupby(["run", "file"])]
    with Pool(WORKERS) as pool:
        ctl = dict(zip([j[1:3] for j in jobs], pool.map(control, jobs, chunksize=4)))
    col = f"alive_{a.horizon}"
    mean = df.groupby(["run", "file", "seed", "point", "k", "split", "option"])[col].agg(["mean", "size", "sum"])
    for split in ("train", "val", "test"):
        with open(f"{a.out}/{split}.jsonl", "w") as fh, open(f"{a.out}/{split}_choice.jsonl", "w") as fc:
            if split not in mean.index.get_level_values("split"):
                continue
            part = mean.xs(split, level="split")
            for (run, name, seed, i, k), g in part.groupby(level=["run", "file", "seed", "point", "k"]):
                g = g.droplevel(["run", "file", "seed", "point", "k"]).reindex([o for o in OPTIONS if o in
                                                                                g.index.get_level_values(-1)])
                opts = list(g.index)
                rid, state = f"{run}#{seed}#{i}", ctl[run, name][k][0]
                fh.write(json.dumps({
                    "id": rid, "state": state,
                    "questions": {o: question(o, a.horizon) for o in opts},
                    "gold": {o: {"probabilities": {"true": float(v), "false": float(1 - v)},
                                 "label": str(bool(v >= 0.5)).lower(), "n": int(n)}
                             for o, v, n in zip(opts, g["mean"], g["size"])}}) + "\n")
                if len(opts) > 1:
                    pb = p_best(g["sum"].to_numpy(), g["size"].to_numpy(), seed=seed * 7 + i)
                    fc.write(json.dumps({
                        "id": rid, "state": state, "questions": {"best": choice_question(opts, a.horizon)},
                        "gold": {"best": {"probabilities": {o: float(v) for o, v in zip(opts, pb)},
                                          "label": opts[int(pb.argmax())]}}}) + "\n")
    print({s: int((mean.index.get_level_values("split") == s).sum()) for s in ("train", "val", "test")},
          "option rows")


if __name__ == "__main__":
    main()
