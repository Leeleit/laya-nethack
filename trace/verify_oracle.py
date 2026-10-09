"""Independent check of the greedy oracle used as a teacher.

The oracle labels a move "good" if it strictly reduces the Chebyshev distance to the stairs
(View.good). Here the ground truth is a BFS shortest path over the actual map (8-connected,
walls and unseen cells blocked, monsters treated as passable), computed without that function:
  1. label set == BFS-optimal move set, on the oracle's own path and on random-walk states;
  2. oracle episode length == BFS distance from the start (empty rooms);
  3. Monster room: what ignoring monsters costs (extra steps, fights, HP lost, deaths).
"""
import argparse, json, random, statistics
from collections import Counter, deque

import gymnasium as gym
import minihack  # noqa: F401
from nle import nethack

from baseline import BLOCKED, DELTA, DIRS, View, greedy

KEYS = ("chars", "screen_descriptions", "blstats", "message")


def bfs(view):
    """Shortest-path distance (in moves) from every passable cell to the stairs."""
    if view.stairs is None:
        return None
    H, W = view.obs["chars"].shape
    dist = {view.stairs: 0}
    q = deque([view.stairs])
    while q:
        x, y = q.popleft()
        for dx, dy in DELTA.values():
            n = (x + dx, y + dy)
            if n in dist or not (0 <= n[0] < W and 0 <= n[1] < H):
                continue
            if n != (view.x, view.y) and view.desc(*n) in BLOCKED:
                continue
            dist[n] = dist[(x, y)] + 1
            q.append(n)
    return dist


def check_labels(view, stats):
    dist = bfs(view)
    if dist is None or (view.x, view.y) not in dist:
        stats["states_without_stairs_or_path"] += 1
        return None
    here = dist[(view.x, view.y)]
    optimal = {d for d in DIRS if dist.get((view.x + DELTA[d][0], view.y + DELTA[d][1]), 1e9) == here - 1}
    labelled = {d for d in DIRS if view.good(d)}
    stats["states"] += 1
    if labelled == optimal:
        stats["exact"] += 1
    else:
        stats["missing" if optimal - labelled else "extra"] += 1
        if len(stats["examples"]) < 3:
            stats["examples"].append({"pos": [view.x, view.y], "stairs": list(view.stairs),
                                      "labelled": sorted(labelled), "optimal": sorted(optimal)})
    return here


def run(env_id, policy, episodes, seed0, max_steps=100):
    env = gym.make(env_id, observation_keys=KEYS)
    idx = {d: env.unwrapped.actions.index(getattr(nethack.CompassDirection, s))
           for d, s in zip(DIRS, ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])}
    st = Counter(examples=[])
    st["examples"] = []
    ends, excess, fights, hp_lost = Counter(), [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        v = View(obs)
        start = check_labels(v, st)
        hp0, nfight = v.hp, 0
        for t in range(1, max_steps + 1):
            obs, _, term, trunc, info = env.step(idx[policy(v)])
            if term or trunc:
                end = info["end_status"].name
                ends[end] += 1
                if end == "TASK_SUCCESSFUL" and start is not None:
                    excess.append(t - start)
                break
            v = View(obs)
            nfight += any(w in v.message for w in ("You hit", "You miss", "You kill", "You destroy"))
            check_labels(v, st)
        else:
            ends["MAX_STEPS"] += 1
        fights.append(nfight)
        hp_lost.append(hp0 - v.hp)
    env.close()
    return {"states_checked": st["states"], "labels_exact": st["exact"],
            "labels_missing_optimal": st["missing"], "labels_extra_nonoptimal": st["extra"],
            "states_without_stairs_or_path": st["states_without_stairs_or_path"],
            "mismatch_examples": st["examples"], "ends": dict(ends),
            "steps_minus_bfs": dict(Counter(excess)) if excess else None,
            "episodes_with_fights": sum(f > 0 for f in fights),
            "hp_lost_mean": round(statistics.mean(hp_lost), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="MiniHack-Room-Random-5x5-v0,MiniHack-Room-Random-15x15-v0,"
                                      "MiniHack-Room-Monster-15x15-v0")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=5000)  # disjoint from baseline.py seeds
    ap.add_argument("--out", default="/data/laya/nethack/results/verify_oracle.json")
    a = ap.parse_args()
    random.seed(a.seed)
    out = {}
    for env_id in a.envs.split(","):
        for name, pol in (("oracle", greedy), ("random-walk", lambda v: random.choice(DIRS))):
            r = run(env_id, pol, a.episodes, a.seed)
            out[f"{env_id}/{name}"] = r
            print(env_id, name, json.dumps(r), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
