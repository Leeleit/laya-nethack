"""Does replaying a trace's recorded actions on the same seeds reproduce the game exactly?

Runs inside the arena image (its NLE and nethackers). For each trace file of a collected run: find its
seed (results.json entry with the same number of steps), rebuild the arena's NLEEnvironment with
trajectory_spec("public", "local", seed) -- the derivation eval_batch uses -- feed the recorded actions and
compare the blstats before every action with the recorded ones.

  python replay_check.py <run dir> [max games]   -> one line per game: OK / first diverging step
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import zstandard
from nethackers.arena.environment import NLEEnvironment
from nethackers.arena.seeds import trajectory_spec


def read_trace(p):
    with open(p, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        lines = text.read().splitlines()
    return [json.loads(x) for x in lines[1:]]


def replay(seed, character, recs):
    env = NLEEnvironment(max_steps=len(recs) + 10, no_progress_timeout=10 ** 9, character=character)
    try:
        obs = env.reset(trajectory_spec("public", "local", seed))
        for i, r in enumerate(recs):
            bl = [int(v) for v in obs["blstats"]]
            if bl != r["b"]:
                diff = [(k, x, y) for k, (x, y) in enumerate(zip(bl, r["b"])) if x != y] + [("prev a", recs[i - 1]["a"], recs[i - 1].get("m", ""))]
                return i, diff
            obs, _, term, trunc = env.step(r["a"])
            if (term or trunc) and i != len(recs) - 1:
                return i + 1, "ended early"
        return None, env.metrics()
    finally:
        env.close() if hasattr(env, "close") else None


def main():
    run = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    results = json.load(open(run / "results.json"))
    by_steps = {}
    for r in results:
        by_steps.setdefault(r["steps"], []).append(r)
    ok = 0
    files = sorted((run / "trace").glob("*.jsonl.zst"))[:limit]
    for p in files:
        recs = read_trace(p)
        cands = by_steps.get(len(recs), [])
        verdict = f"no results.json entry with {len(recs)} steps"
        for r in cands:
            at, info = replay(r["trajectory_id"], r["character"], recs)
            if at is None:
                verdict = f"OK seed {r['trajectory_id']}: {len(recs)} steps, depth {info.max_depth}, turns {info.turns}, " \
                          f"result turns {r['turns']} depth {r['max_depth']}"
                ok += 1
                break
            verdict = f"DIVERGES seed {r['trajectory_id']} at step {at}/{len(recs)}: {info}"
        print(f"{p.name}: {verdict}", flush=True)
    print(f"reproduced {ok}/{len(files)}")


if __name__ == "__main__":
    main()
