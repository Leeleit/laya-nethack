"""A few human states as the model reads them (human_build.state_text, --map text --terse --compact) from a states
file's first games (the re-extracted fields hp_ago, msgs, mcol).
  python human_look.py <states jsonl> [n states] [games]
"""
import json
import random
import sys

import human_build

human_build.MAP, human_build.COMPACT, human_build.TERSE = "text", True, True
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
games = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        games.append(json.loads(line))
        if len(games) >= (int(sys.argv[3]) if len(sys.argv) > 3 else 200):
            break
pairs = [(g, s) for g in games for s in g["states"] if s.get("mcol")]
for g, s in random.Random(0).sample(pairs, min(n, len(pairs))):
    print("=" * 80)
    print(human_build.state_text(g, s))
