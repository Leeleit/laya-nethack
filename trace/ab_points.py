"""Branch points of a paired flag A/B (bot_ab.py's runs): games are deterministic, so a seed's two traces are the
same up to the step where the flag first acts -- a fork with one outcome per option. For every seed: that step k,
the state there (risk-v2's state text, build_v2.walk: the last screen at or before k) and both games' outcomes
(arena progress, deepest Dlvl, death; with bot_endgame.py parquets also the milestones).
  python ab_points.py <base run dir> <arm run dir> [--endgame base.parquet arm.parquet] --out <points.jsonl>
    [--workers N]
-> one JSON line per seed whose games diverge: {seed, k, turn, state, base: {...}, arm: {...}}
"""
import argparse
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from build_v2 import state_text, walk


def lines_of(path):
    return subprocess.run(["zstd", "-dcq", str(path)], capture_output=True, check=True).stdout.splitlines()


def game_fields(line):
    """a record without the bot's strategy stack "s": the flag's copy names its strategies differently long before
    it acts differently (seen: the stacks differ at turn 50, the actions and screens much later)"""
    r = json.loads(line)
    r.pop("s", None)
    return r


def diverge(job):
    """the first record index where the two games differ (action, screen, blstats, monsters), and the state at it"""
    seed, base, arm = job
    a, b = lines_of(base), lines_of(arm)
    identity = json.loads(a[0]).get("identity", "unknown")
    a, b = a[1:], b[1:]
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y and game_fields(x) != game_fields(y)), None)
    if k is None:
        return seed, None
    state, turn = None, None
    for idx, r, back, msgs, inv in walk(a[:k + 1]):
        state, turn = state_text(identity, r["s"], r, back, list(msgs), inv), r["b"][20]
    return seed, dict(k=k, turn=turn, state=state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("arm")
    ap.add_argument("--endgame", nargs=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    res = [{r["trajectory_id"]: r for r in json.load(open(Path(d, "evidence.json")))["results"]}
           for d in (a.base, a.arm)]
    eg = [pd.read_parquet(f).set_index("seed") for f in a.endgame] if a.endgame else None
    jobs = []
    for seed in sorted(set(res[0]) & set(res[1])):
        # older traces are named <pid>-<episode>: the seed -> trace map is bot_endgame's (bot_profile.match)
        p = [e.at[seed, "game"] if e is not None and seed in e.index else
             next(iter(Path(d, "trace").glob(f"{seed}.jsonl*")), None) for d, e in zip((a.base, a.arm), eg or (None, None))]
        if all(p):
            jobs.append((seed, p[0], p[1]))
    cols = ["section", "reached_medusa", "passed_medusa", "reached_castle", "crossed_castle", "gehennom", "sokoban"]
    n = 0
    with Pool(a.workers) as pool, open(a.out, "w", encoding="utf-8", newline="\n") as out:
        for seed, d in pool.imap_unordered(diverge, jobs, chunksize=2):
            if d is None:
                continue
            for name, r, e in zip(("base", "arm"), res, eg or (None, None)):
                x = r[seed]
                o = dict(progress=x["progress"], max_depth=x["max_depth"], turns=x["turns"],
                         died=bool(x.get("cause_of_death")), cause=x.get("cause_of_death"))
                if e is not None and seed in e.index:
                    o.update({c: (e.at[seed, c].item() if hasattr(e.at[seed, c], "item") else e.at[seed, c])
                              for c in cols if c in e.columns})
                d[name] = o
            out.write(json.dumps(dict(seed=seed, **d)) + "\n")
            n += 1
    print(f"{len(jobs)} paired seeds, {n} diverge -> {a.out}")


if __name__ == "__main__":
    main()
