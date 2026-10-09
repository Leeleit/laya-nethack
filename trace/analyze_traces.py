"""Where and how the bot dies, from its decision traces (nethackers/collect.py).

One row per game turn (the first action of each turn). Label: the game ended in death within
K turns of this one. Questions:
  1. which strategy was in control at the death, and in the last turns before it;
  2. P(death within K) by HP fraction while fighting (fight2) -- the bot's own risk curve;
  3. a baseline risk model on blstats + strategy only (game-grouped split): the bar laya,
     which also sees the screen, has to beat.
-> trace/results/trace_stats.json, trace/{turns,games}.parquet
"""
import glob
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = "/data/laya/nethack/nethackers/runs/traces"
OUT = "/data/laya/nethack/trace"
K = 20  # turns
STRIDE = 5  # turns per row in the risk model
# NLE blstats indices
HP, MAXHP, DEPTH, AC, XL, TIME, HUNGER, DNUM = 10, 11, 12, 16, 18, 20, 21, 23
R = {"K_turns": K}


def load_game(args):
    path, result = args
    lines = subprocess.run(["zstdcat", path], capture_output=True, check=True).stdout.splitlines()
    recs = [json.loads(x) for x in lines[1:]]
    if len(recs) != result["steps"]:
        return None  # not this episode
    rows = []
    for r in recs:
        if "scr" not in r:
            continue
        b = r["b"]
        rows.append((b[TIME], b[DEPTH], b[DNUM], b[HP], b[MAXHP], b[XL], b[AC], b[HUNGER],
                     r["s"][-1] if r["s"] else "<base>", len(r["s"])))
    df = pd.DataFrame(rows, columns=["turn", "depth", "dnum", "hp", "maxhp", "xl", "ac", "hunger", "strategy", "stack"])
    last = [r["s"][-1] if r["s"] else "<base>" for r in recs[-50:]]
    return df, last


def games(root=ROOT):
    """(trace file, its arena result) pairs. <seed>.jsonl.zst traces (collect.py in the laya-frozen image) name
    their seed; older <pid>-<episode> ones carry none, so they are matched on the step count, a tie (46 of
    runs/val1's 1000 games share a step count with another) broken by the trace's last game turn."""
    jobs = []
    for d in sorted(glob.glob(f"{root}/*/evidence.json")):
        run = os.path.dirname(d)
        ident = os.path.basename(run).rsplit("-", 2)[0]
        results = json.load(open(d))["results"]
        by_id = {x["trajectory_id"]: x for x in results}
        by_steps, used = {}, set()
        for x in results:
            by_steps.setdefault(x["steps"], []).append(x)
        for t in sorted(glob.glob(f"{run}/trace/*.jsonl.zst")):
            stem = os.path.basename(t).split(".")[0]
            if stem.isdigit():
                if int(stem) in by_id:
                    jobs.append((t, dict(by_id[int(stem)], identity=ident)))
                continue
            n = int(subprocess.run(f"zstdcat {t} | wc -l", shell=True, capture_output=True, text=True).stdout) - 1
            cand = [x for x in by_steps.get(n, []) if x["trajectory_id"] not in used]
            if len(cand) > 1:
                tail = subprocess.run(f"zstdcat {t} | tail -n 1", shell=True, capture_output=True, text=True).stdout
                last = json.loads(tail)["b"][20]  # game turn
                gap = [abs(x["turns"] - last) for x in cand]
                cand = [x for x, g in zip(cand, gap) if g == min(gap)]
            if len(cand) == 1:
                used.add(cand[0]["trajectory_id"])
                jobs.append((t, dict(cand[0], identity=ident)))
    return jobs


def auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def build():
    """Per-turn and per-game tables from the traces; cached as parquet (delete them to rebuild)."""
    if os.path.exists(f"{OUT}/turns.parquet") and os.path.exists(f"{OUT}/games.parquet"):
        return pd.read_parquet(f"{OUT}/turns.parquet"), pd.read_parquet(f"{OUT}/games.parquet")
    jobs = games()
    with Pool(16) as pool:
        loaded = pool.map(load_game, jobs, chunksize=4)
    frames, final = [], []
    for g, ((path, res), out) in enumerate(zip(jobs, loaded)):
        if out is None:
            continue
        df, last = out
        died = res["status"] == "completed" and bool(res["cause_of_death"])
        df["game"] = g
        df["identity"] = res["identity"]
        df["died"] = died
        df["to_end"] = res["turns"] - df["turn"]
        df["y"] = (died & (df["to_end"] <= K)).astype(int)
        frames.append(df)
        final.append({"game": g, "identity": res["identity"], "died": died, "cause": res["cause_of_death"],
                      "max_depth": res["max_depth"], "progress": res["progress"], "turns": res["turns"],
                      "last_strategy": last[-1], "last50": last})
    t = pd.concat(frames, ignore_index=True)
    f = pd.DataFrame(final)
    t["hp_frac"] = t["hp"] / t["maxhp"].clip(lower=1)
    t["role"] = t["identity"].str.slice(0, 3)
    t.to_parquet(f"{OUT}/turns.parquet")
    f.to_parquet(f"{OUT}/games.parquet")
    return t, f


def main():
    t, f = build()
    R["games"] = len(f)
    R["turn_rows"] = len(t)
    R["died_share"] = float(f["died"].mean())
    R["mean_progress"] = float(f["progress"].mean())

    # 1. who was in control at the death
    d = f[f["died"]]
    R["strategy_at_death"] = d["last_strategy"].value_counts(normalize=True).round(4).head(15).to_dict()
    R["strategy_in_last50_steps_before_death"] = (
        pd.Series([s for x in d["last50"] for s in set(x)]).value_counts() / len(d)).round(3).head(15).to_dict()
    R["strategy_share_all_turns"] = t["strategy"].value_counts(normalize=True).round(4).head(15).to_dict()

    # 2. the fight's risk curve by HP fraction
    fight = t[t["strategy"] == "Agent.fight2"]
    bins = pd.cut(fight["hp_frac"], [0, .1, .2, .3, .4, .5, .66, .8, 1.01])
    R["fight2_p_death_within_K_by_hp_frac"] = {str(k): [round(float(v.mean()), 4), int(len(v))]
                                                for k, v in fight.groupby(bins, observed=True)["y"]}
    R["p_death_within_K_by_depth_band"] = {str(k): [round(float(v.mean()), 4), int(len(v))] for k, v in t.groupby(
        pd.cut(t["depth"], [0, 4, 9, 14, 20, 24, 27, 60]), observed=True)["y"]}

    # 3. baseline risk model, split by game; every STRIDE-th turn of a game (neighbouring turns are near
    # copies, and the full 9M x 45 design matrix does not fit statsmodels' IRLS in memory)
    t = t[t.groupby("game").cumcount() % STRIDE == 0].reset_index(drop=True)
    rng = np.random.default_rng(0)
    test_games = set(rng.choice(t["game"].unique(), size=t["game"].nunique() // 5, replace=False))
    top = t["strategy"].value_counts().index[:15]
    X = pd.DataFrame({"hp_frac": t["hp_frac"], "log_maxhp": np.log1p(t["maxhp"]), "depth": t["depth"],
                      "xl": t["xl"], "ac": t["ac"], "hunger": t["hunger"], "stack": t["stack"]})
    # reference levels dropped: '<base>' strategy (sorts first) and role arc
    X = pd.concat([X, pd.get_dummies(t["strategy"].where(t["strategy"].isin(top), "other"), prefix="s",
                                     drop_first=True, dtype=float),
                   pd.get_dummies(t["role"], prefix="r", drop_first=True, dtype=float)], axis=1).astype(float)
    test = t["game"].isin(test_games).to_numpy()
    y = t["y"][test].to_numpy()
    base = y.mean()

    def score(p):
        brier = float(np.mean((p - y) ** 2))
        return {"auc": round(auc(y, p), 4), "brier": round(brier, 6),
                "brier_skill": round(1 - brier / (base * (1 - base)), 4),  # > 0: better than the base rate
                "log_loss": round(float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))), 5)}

    R["risk_models"] = {"train_rows": int((~test).sum()), "test_rows": int(test.sum()),
                        "test_games": len(test_games), "test_positive_rate": round(float(base), 5)}
    Xc = sm.add_constant(X)
    glm = sm.GLM(t["y"][~test], Xc[~test], family=sm.families.Binomial()).fit()
    R["risk_models"]["logit"] = dict(score(glm.predict(Xc[test]).to_numpy()),
                                     coef={k: round(float(v), 3) for k, v in glm.params.items()})
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0)
    gbm.fit(X[~test], t["y"][~test])
    R["risk_models"]["gbm"] = score(gbm.predict_proba(X[test])[:, 1])

    os.makedirs(f"{OUT}/results", exist_ok=True)
    with open(f"{OUT}/results/trace_stats.json", "w") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "risk_models"}, indent=1))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "coef"} if isinstance(v, dict) else v)
                      for k, v in R["risk_models"].items()}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
