"""A look at the new state fields (hp_ago, msgs, mcol) next to the screen they came from.
  python states3_look.py <states jsonl> [n states] [seed]
"""
import json
import random
import sys

rows = [s for line in open(sys.argv[1], encoding="utf-8") for s in json.loads(line)["states"]]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
print(f"{len(rows)} states; with HP 30 turns ago {sum(bool(s['hp_ago'][1]) for s in rows)}, with msgs "
      f"{sum(bool(s['msgs']) for s in rows)}, with monsters {sum(bool(s['mcol']) for s in rows)}; "
      f"msgs a state {sum(len(s['msgs']) for s in rows) / len(rows):.1f}")
for s in rng.sample(rows, n):
    print("=" * 80)
    print(f"T{s['turn']} HP {s['hp']}/{s['maxhp']} 10, 30 turns ago {s['hp_ago']}")
    print(f"row 0: {s['screen'][0]!r}")
    for m in s["msgs"]:
        print(f"  msg: {m!r}")
    for k, c in s["mcol"].items():
        x, y = map(int, k.split(","))
        print(f"  mon {s['screen'][y][x]!r} at {k} colour {c}")
