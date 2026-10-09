"""Bot accounts among the "human" players -> a file for human_build.py --exclude-players,
a name and the reason a line:
  machine pace: ttyrec_pace.py's p99 of the gaps <= 1 s and gaps > 5 s <= 0.1 % (people: p99 >= 2.0 s even at the 1st
    percentile, > 5 s >= 0.2 %);
  taeb: "taeb" in the name -- TAEB, a NetHack bot framework; its accounts play at a person's pace (it thinks slowly), so
    only the name (with 0 ascensions and play around the clock, bot_accounts.py) tells them;
  manual: MANUAL, accounts excluded by judgement rather than a rule.
  python bot_list.py ../results/ttyrec_pace_14d.txt ../results/ttyrec_pace_14d_val.txt ../results/ttyrec_pace_14d_test.txt \
      --out nao/bot_players.txt
"""
import argparse
from pathlib import Path

MANUAL = {}  # name -> reason (none listed here)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pace", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", nargs="*", default=[], help="jsonl files whose players are checked for 'taeb' by name")
    a = ap.parse_args()
    why, games = dict(MANUAL), {}
    for path in a.pace:
        for line in open(path, encoding="utf-8"):
            f = line.split()
            if len(f) != 10 or f[0] == "player":
                continue
            p, p99, g5 = f[0], float(f[6]), float(f[7])
            games[p] = games.get(p, 0) + int(f[1])
            if p99 <= 1.0 and g5 <= 0.001:
                why.setdefault(p, f"machine pace (p99 {p99} s, gaps > 5 s {g5})")
            elif "taeb" in p:
                why.setdefault(p, "taeb in the name")
    # "taeb" in the name over every player of these datasets, not only the paced ones (>= --min games)
    for path in a.data:
        for line in open(path, encoding="utf-8"):
            p = line.split('"id": "', 1)[1].split("@", 1)[0].lower()
            if "taeb" in p:
                why.setdefault(p, "taeb in the name")
                games[p] = games.get(p, 0)
    lines = [f"{p} {games.get(p, 0)} {w}" for p, w in sorted(why.items(), key=lambda kv: -games.get(kv[0], 0))]
    Path(a.out).write_text("# name, games in the paced splits, why (bot_list.py)\n" + "\n".join(lines) + "\n",
                           encoding="utf-8")
    print(f"{len(lines)} accounts, {sum(games.get(p, 0) for p in why)} games")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
