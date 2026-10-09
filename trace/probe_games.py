"""A probe games list: the named games (name@start) plus the first N of the list after
nao_corridor's shuffle-free order taken at random with a fixed seed.
  python probe_games.py nao/val_dwa_any4.json nao/probe_games.json --n 400 --keys name@start
"""
import argparse
import json
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("games")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--keys", nargs="*", default=[])
    a = p.parse_args()
    games = json.load(open(a.games))
    keys = set(a.keys)
    named = [g for g in games if f"{g['name']}@{g['start']}" in keys]
    rest = [g for g in games if f"{g['name']}@{g['start']}" not in keys]
    random.Random(1).shuffle(rest)
    json.dump(named + rest[:a.n], open(a.out, "w"))
    print(f"{len(named)} named + {min(a.n, len(rest))} random -> {a.out}")


if __name__ == "__main__":
    main()
