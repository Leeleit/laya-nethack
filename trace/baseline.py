"""Zero-shot laya on MiniHack room tasks: reach the staircase down.

Two input formats for the same decision (one `choice` over 8 compass moves):
  text  - state lists visible things with direction and distance; options are bare moves.
  cands - options themselves say what the neighbouring square holds and which targets lie that way.
Baselines: random move, greedy oracle (step that brings the stairs closest).
"""
import argparse, json, math, random, statistics, time
from collections import Counter

import gymnasium as gym
import minihack  # noqa: F401  (registers MiniHack envs)
import numpy as np
from nle import nethack

DIRS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
DELTA = {"north": (0, -1), "northeast": (1, -1), "east": (1, 0), "southeast": (1, 1),
         "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1)}
BORING = {"", "floor of a room", "dark part of a room"}
BLOCKED = {"", "wall", "dark part of a room"}
INSTR = ("You are playing NetHack. Your goal is to reach the staircase down. "
         "Moving into a monster attacks it. Which direction do you move next?")


def compass(dx, dy):
    if dx == 0 and dy == 0:
        return "here"
    sector = round(math.atan2(-dy, dx) / (math.pi / 4)) % 8
    return ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"][sector]


def dist_word(d):
    return "adjacent" if d <= 1 else "very near" if d <= 3 else "near" if d <= 7 else "far" if d <= 15 else "very far"


class View:
    """What the agent sees this step, decoded from one NLE observation."""

    def __init__(self, obs):
        self.obs = obs
        bl = obs["blstats"]
        self.x, self.y = int(bl[nethack.NLE_BL_X]), int(bl[nethack.NLE_BL_Y])
        self.hp, self.hpmax = int(bl[nethack.NLE_BL_HP]), int(bl[nethack.NLE_BL_HPMAX])
        self.depth = int(bl[nethack.NLE_BL_DEPTH])
        self.message = bytes(obs["message"]).rstrip(b"\0").decode(errors="replace").strip()
        self.things = []  # (desc, dx, dy) for notable visible cells
        sd, chars = obs["screen_descriptions"], obs["chars"]
        for r, c in zip(*np.nonzero(chars != ord(" "))):
            if (c, r) == (self.x, self.y):
                continue
            d = self.desc(c, r)
            if d not in BORING and d != "wall":
                self.things.append((d, int(c) - self.x, int(r) - self.y))
        self.things.sort(key=lambda t: max(abs(t[1]), abs(t[2])))
        self.stairs = next(((self.x + dx, self.y + dy) for d, dx, dy in self.things if d == "staircase down"), None)

    def desc(self, c, r):
        if not (0 <= r < self.obs["chars"].shape[0] and 0 <= c < self.obs["chars"].shape[1]):
            return ""
        return bytes(self.obs["screen_descriptions"][r, c]).rstrip(b"\0").decode(errors="replace")

    def neighbour(self, d):
        dx, dy = DELTA[d]
        return self.desc(self.x + dx, self.y + dy)

    def good(self, d):
        """True if moving `d` is not blocked and strictly closes the distance to the stairs."""
        if self.stairs is None:
            return None
        dx, dy = DELTA[d]
        nx, ny = self.x + dx, self.y + dy
        if self.neighbour(d) in BLOCKED:
            return False
        cheb = lambda a, b: max(abs(a - self.stairs[0]), abs(b - self.stairs[1]))
        return cheb(nx, ny) < cheb(self.x, self.y)

    def status(self):
        return f"HP {self.hp}({self.hpmax}), dungeon level {self.depth}."

    def text_state(self):
        lines = [f"Message: {self.message or '(none)'}", f"Status: {self.status()}", "You see:"]
        for d, dx, dy in self.things:
            ns = f"{abs(dy)} {'south' if dy > 0 else 'north'}" if dy else ""
            ew = f"{abs(dx)} {'east' if dx > 0 else 'west'}" if dx else ""
            lines.append(f"- {d}: {dist_word(max(abs(dx), abs(dy)))} {compass(dx, dy)} ({', '.join(p for p in (ew, ns) if p)})")
        walls = [d for d in DIRS if self.neighbour(d) in BLOCKED]
        if walls:
            lines.append("Blocked: " + ", ".join(walls) + ".")
        return "\n".join(lines)

    def cands_state(self):
        return f"Message: {self.message or '(none)'}\nStatus: {self.status()}"

    def cands_options(self):
        opts = {}
        for d in DIRS:
            n = self.neighbour(d)
            here = "blocked by a wall" if n in BLOCKED else n
            toward = [f"{t} ({dist_word(max(abs(dx), abs(dy)))})" for t, dx, dy in self.things if compass(dx, dy) == d]
            opts[d] = f"move {d}: {here}" + (f"; toward {', '.join(toward)}" if toward else "")
        return opts


def question(view, fmt):
    crit = view.cands_options() if fmt == "cands" else {d: f"move one square {d}" for d in DIRS}
    return {"move": {"type": "choice", "instructions": INSTR, "criteria": crit}}


def run(env_id, agent_name, policy, episodes, max_steps, seed0):
    env = gym.make(env_id, observation_keys=("chars", "screen_descriptions", "blstats", "message"))
    idx = {d: env.unwrapped.actions.index(getattr(nethack.CompassDirection, s))
           for d, s in zip(DIRS, ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])}
    ends, steps_win, good, judged, ms, picks = Counter(), [], 0, 0, [], Counter()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        for t in range(1, max_steps + 1):
            v = View(obs)
            t0 = time.perf_counter()
            d = policy(v)
            ms.append((time.perf_counter() - t0) * 1000)
            picks[d] += 1
            g = v.good(d)
            if g is not None:
                judged += 1
                good += g
            obs, rew, term, trunc, info = env.step(idx[d])
            if term or trunc:
                # The env's own step limit also ends with term=True, status ABORTED.
                end = info["end_status"].name if term else "TRUNCATED"
                ends[end] += 1
                if end == "TASK_SUCCESSFUL":
                    steps_win.append(t)
                break
        else:
            ends["MAX_STEPS"] += 1
    env.close()
    return {"env": env_id, "agent": agent_name, "episodes": episodes,
            "success": round(ends["TASK_SUCCESSFUL"] / episodes, 3), "ends": dict(ends),
            "steps_on_success": round(statistics.mean(steps_win), 1) if steps_win else None,
            "good_move_rate": round(good / judged, 3) if judged else None,
            "ms_p50": round(statistics.median(ms), 2),
            "picks": dict(picks.most_common())}


def greedy(v):
    if v.stairs is None:
        return random.choice(DIRS)
    ok = [d for d in DIRS if v.good(d)]
    return random.choice(ok) if ok else random.choice(DIRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="MiniHack-Room-Random-5x5-v0,MiniHack-Room-Random-15x15-v0,"
                                      "MiniHack-Room-Monster-15x15-v0")
    ap.add_argument("--models", default="laya,laya-multilingual,laya-typed-decisions")
    ap.add_argument("--formats", default="text,cands")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--models-dir", default="/data/laya/repo/models")
    ap.add_argument("--out", default="/data/laya/nethack/results/baseline.json")
    ap.add_argument("--show", action="store_true", help="print one state/options example per format and exit")
    a = ap.parse_args()
    envs = a.envs.split(",")
    random.seed(a.seed)

    if a.show:
        env = gym.make(envs[-1], observation_keys=("chars", "screen_descriptions", "blstats", "message"))
        v = View(env.reset(seed=a.seed)[0])
        for fmt in a.formats.split(","):
            print(f"===== {fmt}\n--- state\n{v.text_state() if fmt == 'text' else v.cands_state()}")
            print("--- options\n" + "\n".join(f"{k}: {c}" for k, c in question(v, fmt)["move"]["criteria"].items()))
        return

    import laya, torch
    results = []
    for env_id in envs:
        results.append(run(env_id, "random", lambda v: random.choice(DIRS), a.episodes, a.max_steps, a.seed))
        results.append(run(env_id, "greedy-oracle", greedy, a.episodes, a.max_steps, a.seed))
        print(json.dumps(results[-2]), json.dumps(results[-1]), sep="\n", flush=True)
    for m in a.models.split(","):
        agent = laya.load(f"{a.models_dir}/{m}", device="cuda")
        for fmt in a.formats.split(","):
            def policy(v, fmt=fmt):
                state = v.text_state() if fmt == "text" else v.cands_state()
                return agent.predict(state, question(v, fmt))["answers"]["move"]["choice"]
            for env_id in envs:
                results.append(run(env_id, f"{m}/{fmt}", policy, a.episodes, a.max_steps, a.seed))
                print(json.dumps(results[-1]), flush=True)
        del agent
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
