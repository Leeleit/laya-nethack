"""Data hygiene overview of games.parquet: versions, flags, outcomes, start-scumming."""
import pandas as pd

df = pd.read_parquet("/data/laya/nethack/data/games.parquet")
df["fam"] = df["version"].str.slice(0, 3)
df["outcome"] = df["death"].fillna("").map(
    lambda d: "ascended" if d.startswith("ascended") else "quit" if d.startswith("quit")
    else "escaped" if d.startswith("escaped") else "died")
pd.set_option("display.width", 200, "display.max_columns", 20)

print("== games by version family x source")
print(pd.crosstab(df["fam"], df["source"], margins=True))
print("\n== flags values by family (top)")
for fam, g in df.groupby("fam"):
    print(fam, g["flags"].value_counts(dropna=False).head(8).to_dict())
print("\n== outcome by family")
print(pd.crosstab(df["fam"], df["outcome"], margins=True))
print("\n== short games (turns<=100) by outcome, 3.4+/3.6")
s = df[df["fam"].isin(["3.4", "3.6"])]
print(pd.crosstab(s["turns"].le(100), s["outcome"]))
print("\n== ascensions with flags bit0/bit1 (wizard/explore)")
a = df[df["outcome"] == "ascended"]
print(a.groupby("fam")["flags"].apply(lambda f: f.fillna(0).astype(int).map(lambda v: v & 3).value_counts().to_dict()))
print("\n== achieve bits among 3.6 ascensions (share set)")
a36 = a[(a["fam"] == "3.6") & a["achieve"].notna()]
print({hex(1 << b): round(float(a36["achieve"].astype(int).map(lambda v: (v >> b) & 1).mean()), 3) for b in range(14)})
print("\n== distinct players", df["name"].nunique(), " players with >=1 ascension", a["name"].nunique())
print("== date range", df["deathdate"].min(), df["deathdate"].max())
