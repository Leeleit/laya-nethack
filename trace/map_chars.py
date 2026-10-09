"""Which characters the human level maps (nao_corridor states, tty rows 1-21) hold and how often, with examples: to
find screen-scraping debris on the map (a '7' where a wall should be).
  python map_chars.py <states.jsonl> [--chars 6789] [--n 6]"""
import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("states")
ap.add_argument("--chars", help="only these characters (default: all, counts only)")
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
count, states, example = collections.Counter(), 0, collections.defaultdict(list)
for line in open(a.states, encoding="utf-8"):
    g = json.loads(line)
    for s in g["states"]:
        states += 1
        for ch in {ch for r in s["screen"][1:22] for ch in r[:80]}:
            count[ch] += 1
            if a.chars and ch in a.chars and len(example[ch]) < a.n:
                row = next(r for r in s["screen"][1:22] if ch in r[:80])
                example[ch].append(f"{g['name']}@{g['start']} T{s['turn']}: {row.strip()[:70]}")
print(f"{states} states; share of states with the character on the map:")
for ch, n in count.most_common():
    if not a.chars or ch in a.chars:
        print(f"  {ch!r:6} {100 * n / states:7.3f} %")
for ch, ex in example.items():
    print(ch, *ex, sep="\n    ")
