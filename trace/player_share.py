"""How unevenly a human dataset's rows come from players: if a few players give much of train, the
model learns mostly their play. Per split: players, games, rows; games per player (median, p90, p99, max); the
share of rows from the top 1 / 10 / 50 / 100 / 1% players; the effective number of players 1 / sum(p^2); rows of
players with >= 10 / 100 / 1000 games; the top 15 players with their games, rows and labels of `ascends`.
  python player_share.py human_data14d [--out ../results/player_share_14d.txt]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = []
    for split in ("train", "val", "test"):
        rows, games, asc = collections.Counter(), collections.defaultdict(set), collections.defaultdict(collections.Counter)
        for line in open(Path(a.data, f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            game = r["id"].split("#", 1)[0]
            p = game.split("@", 1)[0].lower()
            rows[p] += 1
            games[p].add(game)
            if "ascends" in r["gold"]:
                asc[p][r["gold"]["ascends"]["label"]] += 1
        n = np.array(sorted(rows.values(), reverse=True), float)
        g = np.array([len(v) for v in games.values()])
        share = n / n.sum()
        top = lambda k: share[:k].sum()  # noqa: E731
        by_games = {p: len(v) for p, v in games.items()}
        heavy = lambda k: sum(rows[p] for p in rows if by_games[p] >= k) / n.sum()  # noqa: E731
        out.append(f"== {split}: {len(rows)} players, {sum(g)} games, {int(n.sum())} rows")
        out.append(f"  games per player: median {np.median(g):.0f}, p90 {np.percentile(g, 90):.0f}, "
                   f"p99 {np.percentile(g, 99):.0f}, max {g.max()}")
        out.append(f"  rows from the top 1 / 10 / 50 / 100 / 1% ({max(1, len(n) // 100)}) players: "
                   f"{top(1):.1%} / {top(10):.1%} / {top(50):.1%} / {top(100):.1%} / {top(max(1, len(n) // 100)):.1%}")
        out.append(f"  effective number of players 1/sum(p^2): {1 / (share ** 2).sum():.0f}")
        out.append(f"  rows of players with >= 10 / 100 / 1000 games: {heavy(10):.1%} / {heavy(100):.1%} / {heavy(1000):.1%}")
        all_asc = sum(asc.values(), collections.Counter())
        out.append(f"  ascends labels, all: {dict(sorted(all_asc.items()))}")
        if split == "train":
            out.append("  top 15: player  games  rows  share  ascends labels")
            for p, k in rows.most_common(15):
                out.append(f"    {p:20s} {by_games[p]:6d} {k:6d} {k / n.sum():6.2%}  {dict(sorted(asc[p].items()))}")
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
