"""Stage 2 of the learning curve: player variety vs volume. A subset of train with at most CAP games per
player, as many rows as a random subset (subset_games.py) of the same share: games in the order of crc32 of
"name@start" (the same hash as subset_games.py, so no new randomness), a game taken while its player has fewer than CAP
games, until the rows reach the target. --stats prints how far the cap reaches (rows of the whole capped pool).
  python subset_cap.py <train.jsonl> <out.jsonl> <target rows> [--cap 20]
  python subset_cap.py <train.jsonl> --stats [--cap 20]
"""
import argparse
import zlib
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out", nargs="?")
    ap.add_argument("target", nargs="?", type=int)
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    games = defaultdict(list)  # "name@start" -> line numbers
    lines = open(a.data, encoding="utf-8").read().splitlines()
    for n, line in enumerate(lines):
        games[line.split('"id": "', 1)[1].split("#", 1)[0]].append(n)
    order = sorted(games, key=lambda g: (zlib.crc32(g.encode()) % 1000, zlib.crc32(g.encode())))
    per = Counter(g.split("@", 1)[0] for g in games)
    if a.stats:
        pool = sum(len(games[g]) for g in order if True)  # all rows
        taken, seen = 0, Counter()
        for g in order:
            p = g.split("@", 1)[0]
            if seen[p] < a.cap:
                seen[p] += 1
                taken += len(games[g])
        print(f"{len(lines)} rows, {len(games)} games, {len(per)} players; players over {a.cap} games: "
              f"{sum(v > a.cap for v in per.values())} with {sum(v for v in per.values() if v > a.cap)} games; "
              f"capped pool {sum(min(v, a.cap) for v in per.values())} games, {taken} rows ({taken / pool:.1%})")
        return
    keep, seen, rows = set(), Counter(), 0
    for g in order:
        if rows >= a.target:
            break
        p = g.split("@", 1)[0]
        if seen[p] < a.cap:
            seen[p] += 1
            keep.update(games[g])
            rows += len(games[g])
    if rows < a.target:
        raise SystemExit(f"the capped pool has {rows} rows < target {a.target}")
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for n in sorted(keep):
            f.write(lines[n] + "\n")
    print(f"{rows} rows (target {a.target}), {sum(seen.values())} games, {len(seen)} players, cap {a.cap}")


if __name__ == "__main__":
    main()
