"""How far the bot gets: arena progress (BALROG: estimated P(ascend | best milestone reached)),
ascensions, depth -- over all collected trace runs, overall and for one identity."""
import glob
import json
import sys

import pandas as pd

rows = []
for p in glob.glob("/data/laya/nethack/nethackers/runs/traces2/*/results.json"):
    for r in json.load(open(p)):
        rows.append({"identity": r["character"] if isinstance(r["character"], str) else p.split("/")[-2].rsplit("-", 2)[0],
                     "progress": r["progress"], "ascended": bool(r["ascended"]), "depth": r["max_depth"],
                     "turns": r["turns"], "milestone": r["milestone"]})
df = pd.DataFrame(rows)
df["role"] = df.identity.str[:3]
print(f"{len(df)} games, ascended {df.ascended.sum()}; progress mean {df.progress.mean():.3f}, median {df.progress.median():.3f}, "
      f"p90 {df.progress.quantile(.9):.3f}, max {df.progress.max():.3f}; depth median {df.depth.median():.0f}, max {df.depth.max()}")
print("share reaching depth >=", {d: round((df.depth >= d).mean(), 3) for d in (10, 20, 25, 30, 40, 50)})
print(df.groupby("role").agg(games=("progress", "size"), progress=("progress", "mean"), depth=("depth", "median"),
                             asc=("ascended", "sum")).sort_values("progress", ascending=False).round(3).to_string())
for ident in sys.argv[1:]:
    x = df[df.identity == ident]
    print(ident, len(x), "games: progress", round(x.progress.mean(), 3), "depth median", x.depth.median(),
          "max", x.depth.max(), "milestones", x.milestone.value_counts().head(5).to_dict())
