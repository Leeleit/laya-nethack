"""After the first divergence of a replay, does the game drift apart or is it a one-off turn-counter glitch?

  python replay_tail.py <run dir> <trace file> <seed>   (inside the laya-replay image)
"""
import sys
from pathlib import Path

from nethackers.arena.environment import NLEEnvironment
from nethackers.arena.seeds import trajectory_spec
from replay_check import read_trace

NAMES = "x y str25 str125 dex con int wis cha score hp hpmax depth gold ene enemax ac hd xp exp time hunger cap dnum dlevel cond align".split()
run, name, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
recs = read_trace(run / "trace" / name)
import json
char = next(r["character"] for r in json.load(open(run / "results.json")) if r["trajectory_id"] == seed)
env = NLEEnvironment(max_steps=len(recs) + 10, no_progress_timeout=10 ** 9, character=char)
obs = env.reset(trajectory_spec("public", "local", seed))
bad, fields, first = 0, {}, None
for i, r in enumerate(recs):
    bl = [int(v) for v in obs["blstats"]]
    d = [NAMES[k] for k, (x, y) in enumerate(zip(bl, r["b"])) if x != y]
    if d:
        bad += 1
        first = first if first is not None else i
        for k in d:
            fields[k] = fields.get(k, 0) + 1
        if bad <= 6:
            print(i, [(NAMES[k], x, y) for k, (x, y) in enumerate(zip(bl, r["b"])) if x != y], "| a", r["a"], r.get("m", "")[:60])
    obs, _, term, trunc = env.step(r["a"])
    if term or trunc:
        print("env ended at", i + 1, "of", len(recs))
        break
m = env.metrics()
print(f"first {first}, mismatched steps {bad}/{len(recs)}, fields {fields}; end depth {m.max_depth} turns {m.turns} status {m.end_status}")
