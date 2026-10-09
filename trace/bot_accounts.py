"""Are the busiest "human" players of a dataset bots? (Several top accounts have "bot" or "taeb" in the
name; TAEB is a NetHack bot framework.) Per player with >= --min games in the dataset's train, from all of
the account's NAO games in games.parquet (xlogfile): games, share Val-Dwa, role-race combinations, ascensions,
median turns, seconds per turn (realtime / turns), the pause from one game's end to the next one's start (median,
share < 60 s), hours of the day (UTC) with >= 1 % of its games, days active; the same medians over players with
10-49 games as the reference. A bot plays without breaks around the clock, a person sleeps.
  python bot_accounts.py human_data14d [--min 50] [--out ../results/bot_accounts_14d.txt]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

GAMES = "/data/laya/nethack/data/games.parquet"


def profile(g):
    g = g.sort_values("starttime")
    st, en = g.starttime.to_numpy(), g.endtime.to_numpy()
    pause = st[1:] - en[:-1]
    pause = pause[np.isfinite(pause)]
    spt = (g.realtime / g.turns).replace([np.inf, -np.inf], np.nan).dropna()
    spt = spt[spt > 0]
    hours = pd.to_datetime(g.starttime.dropna(), unit="s").dt.hour.value_counts(normalize=True)
    days = pd.to_datetime(g.starttime.dropna(), unit="s").dt.date.nunique()
    return {
        "games": len(g),
        "val_dwa": ((g.role == "Val") & (g.race == "Dwa")).mean(),
        "combos": g.groupby(["role", "race"]).ngroups,
        "asc": (g.death == "ascended").mean(),
        "turns": g.turns.median(),
        "s/turn": spt.median() if len(spt) else np.nan,
        "pause": np.median(pause) if len(pause) else np.nan,
        "p<60s": (pause < 60).mean() if len(pause) else np.nan,
        "hours": int((hours >= 0.01).sum()),
        "days": days,
        "top death": g.death.str.replace(r"^killed by (an? )?", "", regex=True).value_counts().index[0][:28],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--min", type=int, default=50)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    train = collections.defaultdict(set)
    for line in open(Path(a.data, "train.jsonl"), encoding="utf-8"):
        game = json.loads(line)["id"].split("#", 1)[0]
        train[game.split("@", 1)[0].lower()].add(game)
    n = {p: len(v) for p, v in train.items()}
    d = pd.read_parquet(GAMES, columns=["source", "name", "role", "race", "death", "turns", "realtime",
                                        "starttime", "endtime"])
    d = d[d.source == "nao"]
    d["lname"] = d.name.str.lower()
    by = dict(tuple(d[d.lname.isin(set(n))].groupby("lname")))
    rows = []
    for p, k in n.items():
        if p in by and k >= 10:
            rows.append(dict(player=p, train=k, **profile(by[p])))
    t = pd.DataFrame(rows).sort_values("train", ascending=False)
    big, ref = t[t.train >= a.min], t[(t.train >= 10) & (t.train < a.min)]
    pd.set_option("display.width", 250)
    fmt = {"val_dwa": "{:.2f}".format, "asc": "{:.3f}".format, "s/turn": "{:.2f}".format, "p<60s": "{:.2f}".format,
           "pause": "{:.0f}".format, "turns": "{:.0f}".format}
    out = [f"players with >= {a.min} train games: {len(big)} (of {len(n)} in train, {sum(1 for p in n if p not in by)} "
           f"not found in games.parquet); train games are Val-Dwa games of the dataset, `games` all NAO games",
           big.to_string(index=False, formatters=fmt), "",
           f"reference, players with 10-{a.min - 1} train games ({len(ref)}): medians",
           ref.drop(columns=["player", "top death"]).median().to_string()]
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
