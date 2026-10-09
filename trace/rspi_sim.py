"""Adaptive allocation of replays (docs/branches.md): how should rollouts be spent at a branch point? Offline, on forkrun.py data.

Per point, each option's even reps are the pool to spend from, its odd reps score the pick (held out). With a
budget of B rollouts per point:
  uniform  -- B / |options| reps of every option
  race     -- RSPI-style (Dimitrakakis & Lagoudakis 2008): one rep of every option, then every further rollout
              goes to the less-sampled of the two options with the best mean so far
The value of a pick is its held-out survival; reported against continue and the full-pool pick.
  python rspi_sim.py --choice C [--horizon 300]  -> C/rspi_h<H>.txt
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--choice", required=True, help="build_fork.py output")
ap.add_argument("--horizon", type=int, default=300)
a = ap.parse_args()
col = f"alive_{a.horizon}"
oc = pd.read_parquet(f"{a.choice}/outcomes.parquet")
K = ["run", "seed", "point"]

points = []
for key, g in oc.groupby(K):
    pool, held = {}, {}
    for o, go in g.groupby("option"):
        go = go.sort_values("rep")
        pool[o] = go[go.rep % 2 == 0][col].to_numpy(float)
        held[o] = go[go.rep % 2 == 1][col].mean()
    if len(pool) > 1 and all(len(v) for v in pool.values()) and not any(np.isnan(v) for v in held.values()):
        points.append((list(pool), pool, held))
n_pool = min(len(v) for _, pool, _ in points for v in pool.values())


def pick_uniform(opts, pool, b):
    r = max(1, b // len(opts))
    return max(opts, key=lambda o: (pool[o][:r].mean(), o == "continue"))


def pick_race(opts, pool, b):
    used = {o: 1 for o in opts}
    for _ in range(max(0, b - len(opts))):
        m = {o: pool[o][:used[o]].mean() for o in opts}
        top = sorted(opts, key=lambda o: (-m[o], used[o]))[:2]
        o = min((t for t in top if used[t] < len(pool[t])), key=lambda t: used[t], default=None)
        if o is None:
            break
        used[o] += 1
    return max(opts, key=lambda o: (pool[o][:used[o]].mean(), o == "continue"))


base = np.mean([held["continue"] for _, _, held in points])
full = np.mean([held[pick_uniform(opts, pool, len(opts) * n_pool)] for opts, pool, held in points])
lines = [f"horizon {a.horizon}: {len(points)} points, pool {n_pool} reps per option; held-out survival: "
         f"continue {base:.4f}, full-pool pick {full:.4f} ({full - base:+.4f})"]
for per in range(1, n_pool + 1):
    for name, fn in (("uniform", pick_uniform), ("race", pick_race)):
        v = np.mean([held[fn(opts, pool, len(opts) * per)] for opts, pool, held in points])
        lines.append(f"  budget {per} x options  {name:8s} {v:.4f} ({v - base:+.4f})")
print("\n".join(lines))
open(f"{a.choice}/rspi_h{a.horizon}.txt", "w").write("\n".join(lines) + "\n")
