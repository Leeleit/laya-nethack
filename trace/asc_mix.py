"""Why the share of ascended rows differs between train / val / test (14d: 2.4 / 5.4 / 3.8 %) and how the
player mix bears on it. Per split: rows, games, ascended share by rows and by games; train cut to --cap games a player
(random, seed 0; val/test are built so) -- if that brings train to val/test, the cap explains the gap; ascended share
by the player's train-split game count (1, 2-9, 10-99, 100+); the player's strength before the game -- ascensions of
the account on NAO (any role, games.parquet) that ended before this game started: 0 / 1-4 / 5-19 / 20+ -- its mix
per split and the ascended share in each.
  python asc_mix.py human_data15d [--cap 20] [--out ../results/asc_mix_15d.txt]
"""
import argparse
import bisect
import collections
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

GAMES = "/data/laya/nethack/data/games.parquet"
LEVELS = [(0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"), (20, 10 ** 9, "20+")]
SIZES = [(1, 1, "1"), (2, 9, "2-9"), (10, 99, "10-99"), (100, 10 ** 9, "100+")]


def bucket(x, table):
    return next(name for lo, hi, name in table if lo <= x <= hi)


def share(rows):
    if not rows:
        return "n 0"
    games = {}
    for r in rows:
        games[r["game"]] = r["asc"]
    return (f"rows {len(rows):6d} asc {np.mean([r['asc'] for r in rows]):6.2%} | "
            f"games {len(games):6d} asc {np.mean(list(games.values())):6.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = pd.read_parquet(GAMES, columns=["source", "name", "death", "starttime", "endtime"])
    d = d[(d.source == "nao") & (d.death == "ascended") & d.endtime.notna()]
    asc_end = {n.lower(): sorted(g.endtime.tolist()) for n, g in d.groupby(d.name.str.lower())}
    data = {}
    for split in ("train", "val", "test"):
        rows = []
        for line in open(Path(a.data, f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            game = r["id"].split("#", 1)[0]
            name, start = game.split("@", 1)
            prior = bisect.bisect_left(asc_end.get(name.lower(), []), float(start))
            rows.append({"game": game, "player": name.lower(), "asc": r["gold"]["ascends"]["label"] == "true",
                         "prior": prior})
        data[split] = rows
    out = ["== ascended share"]
    for split, rows in data.items():
        out.append(f"  {split:5s} {share(rows)}")
    by_player = collections.defaultdict(set)
    for r in data["train"]:
        by_player[r["player"]].add(r["game"])
    rng, keep = random.Random(0), set()
    for p in sorted(by_player):
        g = sorted(by_player[p])
        keep.update(rng.sample(g, a.cap) if len(g) > a.cap else g)
    out.append(f"  train cut to {a.cap} games a player: {share([r for r in data['train'] if r['game'] in keep])}")
    out.append("== by the player's games in train (train rows)")
    for lo, hi, name in SIZES:
        out.append(f"  {name:6s} {share([r for r in data['train'] if lo <= len(by_player[r['player']]) <= hi])}")
    out.append("== by the player's ascensions before the game (share of the split's rows; ascended share)")
    for split, rows in data.items():
        out.append(f"  {split}")
        for lo, hi, name in LEVELS:
            sel = [r for r in rows if lo <= r["prior"] <= hi]
            out.append(f"    {name:5s} {len(sel) / len(rows):6.1%} of rows   {share(sel)}")
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
