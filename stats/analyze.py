"""Correlations between game outcome and what happened in it, from ~2.5M real NetHack games.

Sample: NAO 3.4.3 + NAO/Hardfought 3.6.x, without start-scums (quit/escaped by turn 100),
wizard/explore mode, the nh361dev test server and ascensions under 1000 turns.
Player skill confounds everything, so each game carries leakage-free skill covariates:
how many games and ascensions the same player (server+name) finished before it.
Inference: Wilson intervals for shares, logistic regression with player-clustered SEs.
"""
import json, re, sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

DATA = "/data/laya/nethack/data/games.parquet"
OUT = "/data/laya/nethack/stats/results.json"
BRANCH = {0: "Dungeons", 1: "Gehennom", 2: "Mines", 3: "Quest", 4: "Sokoban", 5: "Ludios", 6: "Vlad", 7: "Planes"}
ACH = {"sokoban_prize": 0x400, "mines_luckstone": 0x200, "medusa_killed": 0x800, "entered_gehennom": 0x2}
R = {}


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return [round(float(k / n), 4) if n else None, round(float(lo), 4), round(float(hi), 4), int(n)]


def load():
    df = pd.read_parquet(DATA)
    df["fam"] = df["version"].str.slice(0, 3)
    d = df["death"].fillna("")
    df["outcome"] = np.select([d.str.startswith("ascended"), d.str.startswith("quit"), d.str.startswith("escaped")],
                              ["ascended", "quit", "escaped"], "died")
    n0 = len(df)
    flags = df["flags"].fillna(0).astype(int)
    keep = (df["fam"].isin(["3.4", "3.6"]) & (df["file"] != "xlogfile.nh361dev") & ((flags & 3) == 0)
            & ~(df["outcome"].isin(["quit", "escaped"]) & (df["turns"] <= 100))
            & ~((df["outcome"] == "ascended") & (df["turns"] < 1000)))
    df = df[keep].copy()
    R["sample"] = {"raw_games": n0, "kept_games": len(df), "by_family": df["fam"].value_counts().to_dict(),
                   "by_outcome": df["outcome"].value_counts().to_dict()}
    df["asc"] = (df["outcome"] == "ascended").astype(int)
    df["player"] = df["source"] + "/" + df["name"].astype(str)
    df["t"] = df["endtime"].fillna(df["deathdate"].fillna(0) * 1e6)  # old lines lack endtime
    df = df.sort_values(["player", "t"], kind="stable")
    g = df.groupby("player", sort=False)
    df["prior_games"] = g.cumcount()
    df["prior_asc"] = g["asc"].cumsum() - df["asc"]
    df["skill"] = pd.cut(df["prior_asc"], [-1, 0, 4, 19, 10**9], labels=["0", "1-4", "5-19", "20+"])
    df["exp"] = pd.cut(df["prior_games"], [-1, 9, 99, 999, 10**9], labels=["<10", "10-99", "100-999", "1000+"])
    return df


def skill_baseline(df):
    R["ascension_rate_by_prior_ascensions"] = {s: wilson(int(x.sum()), len(x)) for s, x in df.groupby("skill", observed=True)["asc"]}
    R["ascension_rate_by_prior_games"] = {s: wilson(int(x.sum()), len(x)) for s, x in df.groupby("exp", observed=True)["asc"]}


def depth_survival(df):
    """P(go deeper | reached Dlvl d) and P(ascend | reached d), by player skill."""
    out = {}
    for s, x in [("all", df)] + [(f"skill {k}", v) for k, v in df.groupby("skill", observed=True)]:
        m = x["maxlvl"].to_numpy()
        a = x["asc"].to_numpy()
        rows = {}
        for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]:
            reached = m >= d
            n = int(reached.sum())
            if n < 50:
                continue
            rows[d] = {"reached": n, "p_deeper": wilson(int((m[reached] > d).sum()), n)[:3],
                       "p_ascend": wilson(int(a[reached].sum()), n)[:3]}
        out[s] = rows
    R["depth_survival"] = out


def killer(s):
    s = s.lower()
    s = re.sub(r",? while .*$", "", s)
    s = re.sub(r"^killed by (an? |the )?", "", s)
    s = re.sub(r" called .*$", "", s)
    s = re.sub(r"^(the )?invisible ", "", s)
    return s.strip()


def killers(df):
    dead = df[df["outcome"] == "died"].copy()
    dead["killer"] = dead["death"].map(killer)
    lev = dead["deathlev"]
    dead["band"] = np.select(
        [dead["deathdnum"] == 7, dead["deathdnum"] == 1, lev <= 4, lev <= 9, lev <= 14, lev <= 24],
        ["Planes", "Gehennom", "Dlvl 1-4", "Dlvl 5-9", "Dlvl 10-14", "Dlvl 15-24"], "Dlvl 25+ (Castle area)")
    R["deaths_by_band"] = dead["band"].value_counts().to_dict()
    R["top_killers_by_band"] = {b: {k: round(float(v), 4) for k, v in x["killer"].value_counts(normalize=True).head(12).items()}
                                for b, x in dead.groupby("band")}
    exp = dead[dead["skill"].isin(["5-19", "20+"])]
    R["top_killers_experienced_players"] = {k: round(float(v), 4) for k, v in exp["killer"].value_counts(normalize=True).head(20).items()}
    R["death_branch"] = dead["deathdnum"].map(BRANCH).value_counts().to_dict()


def logit(df, y, terms, name, cluster="player"):
    """Logistic regression with player-clustered SEs; returns odds ratios with 95% CI."""
    X = pd.get_dummies(df[terms], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    groups = pd.factorize(df[cluster])[0]
    res = sm.GLM(df[y].astype(float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = res.conf_int()
    R[name] = {"n": int(res.nobs), "players": int(groups.max() + 1), "positives": int(df[y].sum()),
               "odds_ratios": {k: [round(float(np.exp(res.params[k])), 3), round(float(np.exp(ci.loc[k, 0])), 3),
                                   round(float(np.exp(ci.loc[k, 1])), 3), float(f"{res.pvalues[k]:.2g}")]
                               for k in res.params.index if k != "const"}}


def roles(df):
    R["ascension_rate_by_role_raw"] = {r: wilson(int(x.sum()), len(x)) for r, x in df.groupby("role")["asc"]}
    R["reach_dlvl10_by_role_raw"] = {r: wilson(int((x >= 10).sum()), len(x)) for r, x in df.groupby("role")["maxlvl"]}
    # Reference category = first alphabetically (Arc); skill and experience held fixed.
    logit(df, "asc", ["role", "race", "align", "fam", "skill", "exp"], "model_ascend_by_role_adjusted")
    df = df.assign(reach10=(df["maxlvl"] >= 10).astype(int))
    logit(df, "reach10", ["role", "race", "align", "fam", "skill", "exp"], "model_reach_dlvl10_by_role_adjusted")


def milestones(df):
    """Among games that killed Medusa (same checkpoint), do Sokoban / Mines' End relate to ascending?"""
    x = df[df["achieve"].notna()].copy()
    ach = x["achieve"].astype(int)
    for k, bit in ACH.items():
        x[k] = ((ach & bit) > 0).astype(int)
    m = x[x["medusa_killed"] == 1]
    R["milestones_raw_given_medusa"] = {
        f"{k}={v}": wilson(int(g["asc"].sum()), len(g)) for k in ["sokoban_prize", "mines_luckstone"] for v, g in m.groupby(k)}
    logit(m, "asc", ["sokoban_prize", "mines_luckstone", "role", "fam", "skill", "exp"], "model_ascend_given_medusa")
    m36 = m[m["fam"] == "3.6"]
    logit(m36, "asc", ["sokoban_prize", "mines_luckstone", "role", "skill", "exp"], "model_ascend_given_medusa_36only")


def main():
    df = load()
    print(json.dumps(R["sample"]), flush=True)
    for f in (skill_baseline, depth_survival, killers, roles, milestones):
        f(df)
        print("done", f.__name__, flush=True)
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    print("->", OUT)


if __name__ == "__main__":
    sys.exit(main())
