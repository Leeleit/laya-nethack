"""Does the bot reproduce its own games? Compare per-seed action sequences of collected runs.

  python repro_cmp.py <run dir> <run dir> ...   -> per seed: steps and the first step where the actions differ
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def actions(p):
    raw = subprocess.run(["zstdcat", "-f", str(p)], capture_output=True, check=True).stdout.decode()
    return [json.loads(x)["a"] for x in raw.splitlines()[1:]]


def by_seed(run):
    run = Path(run)
    steps = {r["steps"]: r["trajectory_id"] for r in json.load(open(run / "results.json"))}
    out = {}
    for p in (run / "trace").glob("*.jsonl*"):
        a = actions(p)
        out[steps.get(len(a), f"?{len(a)}")] = a
    return out


runs = [by_seed(r) for r in sys.argv[1:]]
for seed in sorted(runs[0], key=str):
    seqs = [r.get(seed) for r in runs]
    ref = seqs[0]
    cells = []
    for s in seqs[1:]:
        if s is None:
            cells.append("missing")
            continue
        first = next((i for i, (x, y) in enumerate(zip(ref, s)) if x != y), None)
        if first is None and len(ref) == len(s):
            cells.append("same")
        else:
            cells.append(f"differs at {first if first is not None else min(len(ref), len(s))}")
    print(seed, len(ref), hashlib.md5(bytes(str(ref), "ascii")).hexdigest()[:8], cells)
