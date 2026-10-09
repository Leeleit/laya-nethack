"""Counterfactual branches by fork(): each game is played once; at its branch points the game is snapshotted and
every option runs K times from the snapshot (no replay of the prefix, K outcomes per option).

Runs inside a container of the laya-frozen image with NLE and the bot in ONE process:
the same seeds, objective and bot seeding as the arena (arena/trajectory.py, arena/sandbox.py), so the parent
plays the recorded game.

At a branch point the bot's own preemption check (agent thread, between two actions) forks a *template*; the
parent plays on untouched. The template is a frozen snapshot that forks one child per (option, rep) whenever a
CPU slot is free. A child:
  - is the agent thread alone (fork keeps only the calling thread) and drives NLE itself (adapter.step -> env.step);
  - gets a private copy of NetHack's directory (settings.hackdir in libnethack.so points there) and private
    offsets of the open files;
  - rep > 0: reseeds NetHack's core/display RNG (new levels still come from the separate lgen streams) and the
    bot's RNG, the same reseed for every option of a rep (common random numbers);
  - runs the option with the bot's own code (laya_branch._run), plays on, stops at death or h game turns later.
rep 0 of an option = the old branch method (same game, no reseed); rep 0 of "continue" = the parent itself.

plan.json: {"identity", "solution", "options": [...], "reps": K, "h": 300,
            "games": [{"seed", "points": [{"point", "k", "b"}]}]}       k: index of the bot's act() call
out/results/<seed>.json           the finished game, written last (a game is the unit of resume); "stopped": the
                                   parent stopped h turns after its last point (only that much is an outcome)
out/games/<seed>/points.jsonl      a line per plan point (the parent)
out/games/<seed>/p<i>-<option>-r<rep>.json   a branch outcome (the child, or its template if it crashed)
One writer per file: concurrent appends to one file are not atomic on every file system (a Windows bind
mount interleaved them).
out/btrace/<seed>-p<i>-<option>-r<rep>.jsonl.zst   the child's steps (laya_trace records, no header)
out/trace/<seed>.jsonl.zst         the parent's trace (--trace)

  docker exec <container> /opt/nethackers/.venv/bin/python forkrun.py --plan P --out D [--parallel 16] [--games 8]
"""
import argparse
import ctypes
import importlib
import json
import multiprocessing
import os
import random
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

import nle.nethack as nhc
import numpy as np
import zstandard
from nethackers.arena.environment import make_environment
from nethackers.arena.sandbox import _readonly
from nethackers.arena.seeds import trajectory_spec
from nethackers.harness.seeds import validation_spec

BL_TIME, BL_DEPTH = nhc.NLE_BL_TIME, nhc.NLE_BL_DEPTH
CHILD_WALL = 120.0     # s; a child has no hang guard (the arena driver loop is gone), the template kills it.
                       # Children live <= ~10 s (p99.9 6 s); a bot livelock (no step, 100 % CPU: 2 in ~60k) held
                       # a run's tail for 30 min at the old 1800 s
CHILD_STEPS = 100_000  # bot actions after firing
FINISHED = 17          # a child's exit code after it wrote its outcome and released its slot
MAX_SEED = (1 << 63) - 1


class G:
    """Per-process state (a game process, its templates and children are forks of one another)."""
    cfg = out = sem = None
    waiting = None   # templates with children still to start: new games wait for them (they hold snapshots)
    work = None      # bot steps of the whole tree (parents and children), counted by 100: the throughput
    seed = env = bot = adapter_mod = trace_mod = branch_mod = None
    steps = 0
    obs = None
    points = []      # plan points of this game not reached yet, by k
    armed = None     # the point matched at its step k, waiting for the bot's next preemption check
    agent = None     # the agent the branch strategy is installed on (the driver restarts agents)
    templates = []
    last_fire = None  # game turn of the parent's last snapshot
    child = None     # set in a branch child
    keep = []        # file objects a child must never flush or close (the parent's)


def ro(obs):
    """What the arena hands the bot: a private, read-only copy (it arrives pickled from the env process)."""
    return _readonly({k: v.copy() for k, v in obs.items()})


def write_line(path, rec):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def gdir():
    return G.out / "games" / str(G.seed)


def put(path, rec):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, path)


def point_rec(p, **kw):
    write_line(gdir() / "points.jsonl", {"seed": G.seed, "point": p["point"], "k": p["k"], **kw})


def zst(path):
    with open(path, "rb") as src, open(str(path) + ".zst", "wb") as dst:
        zstandard.ZstdCompressor(level=3).copy_stream(src, dst)
    os.unlink(path)


def copy_dir(src):
    dst = src.rstrip("/") + f".{os.getpid()}.{time.monotonic_ns()}"
    shutil.copytree(src, dst, symlinks=True)
    return dst


# ---- the parent game ---------------------------------------------------------------------------------------------

def install(agent):
    """The branch strategy on top of the agent's global strategy (as laya_branch.install)."""
    Strategy = importlib.import_module(type(agent).__module__.rsplit(".", 1)[0] + ".strategy").Strategy

    def branch():
        p = G.armed
        if p is None:
            yield False
        G.armed = None
        role = snapshot(agent, p)  # None in the parent; the option in a child; a template never returns
        if role is None or role == "continue":
            yield False
        yield True
        G.branch_mod._run(agent, role)

    gl = agent.global_logic
    orig = gl.global_strategy
    gl.global_strategy = lambda: orig().preempt(agent, [Strategy(branch)])
    G.agent = agent


def snapshot(agent, p):
    fire_turn = int(agent.blstats.time)
    point_rec(p, fired=True, fire_step=G.steps, fire_turn=fire_turn)
    if G.trace_mod._fh is not None:
        G.trace_mod._fh.flush()  # a child must not inherit unwritten parent trace data
    snap = copy_dir(G.env._env.nethack._vardir)  # frozen NetHack directory; the parent keeps writing its own
    pid = os.fork()
    if pid:
        G.templates.append(pid)
        G.last_fire = fire_turn
        return None
    try:
        return template(p, fire_turn, snap)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        os._exit(3)


def check_arm(step, obs):
    while G.points and G.points[0]["k"] < step:  # passed by while another point was armed
        point_rec(G.points.pop(0), fired=False, note="skipped")
    if G.points and G.points[0]["k"] == step:
        p = G.points.pop(0)
        if [int(v) for v in obs["blstats"]] == p["b"]:
            G.armed = p
        else:
            point_rec(p, fired=False, note="no match")


def add_work():
    with G.work.get_lock():
        G.work.value += 100


def game(g):
    cfg = G.cfg
    G.seed = seed = g["seed"]
    gdir().mkdir()
    # the children of points already reached go first: otherwise a new game (blocking acquire) wins every freed
    # slot over the polling templates, and the snapshots pile up in memory until the end (games were replaying
    # on 7 of 14 slots while the children's 7 stood idle most of the time: 59 % of the CPU)
    while G.waiting.value > 0:
        time.sleep(0.2)
    G.sem.acquire()  # a CPU slot while playing
    try:
        spec = trajectory_spec("public", "local", seed)
        vs = validation_spec(cfg["identity"], n=1, start=seed)
        char = vs.batch[0][1]  # as arena/run.run_prepared
        env = G.env = make_environment(vs.max_steps, vs.no_progress_timeout, None if char == "-" else char)
        # as arena/sandbox._agent_process: seed the bot's RNGs, then import it from its own directory
        os.environ.pop("NETHACK_ARENA_SECRET", None)
        os.environ["LAYA_TRAJECTORY_ID"] = str(seed)
        os.chdir(cfg["solution"])
        random.seed(spec.bot_seed)
        np.random.seed(spec.bot_seed % (1 << 32))
        sys.path.insert(0, cfg["solution"])
        bot = importlib.import_module("bot")
        G.trace_mod = sys.modules["laya_trace"]
        G.branch_mod = sys.modules["laya_branch"]
        G.trace_mod.DIR = str(G.out / "trace")
        G.trace_mod._ENABLED = bool(cfg.get("trace"))
        b = G.bot = bot.make_agent()
        G.points = sorted(g["points"], key=lambda p: p["k"])
        obs = ro(env.reset(spec))
        b.reset(obs)
        G.adapter_mod = sys.modules[type(b._driver).__module__]
        steps, stopped = 0, False
        while steps < vs.max_steps:
            G.steps, G.obs = steps, obs
            check_arm(steps, obs)
            if b._driver._agent is not None and b._driver._agent is not G.agent:
                install(b._driver._agent)
            action = b.act(obs)
            o, _reward, terminated, truncated = env.step(action)
            steps += 1
            if steps % 100 == 0:
                add_work()
            obs = ro(o)
            if terminated or truncated:
                break
            # every point fired and the last one's horizon passed: the rest of the game is no outcome of anything
            # (a full parent trace, --trace, plays on)
            if G.last_fire is not None and not G.points and G.armed is None and not cfg.get("trace") and \
                    int(obs["blstats"][BL_TIME]) > G.last_fire + cfg["h"]:
                stopped = True
                break
        m = env.metrics()
        for p in G.points + ([G.armed] if G.armed else []):
            point_rec(p, fired=False, note="not reached")
        b.close()
        if cfg.get("trace"):
            zst(G.out / "trace" / f"{seed}.jsonl")
    finally:
        G.sem.release()
    for pid in G.templates:
        os.waitpid(pid, 0)
    put(G.out / "results" / f"{seed}.json", {
        "seed": seed, "steps": steps, "turns": m.turns, "max_depth": m.max_depth, "end_status": m.end_status,
        "cause": m.cause_of_death, "progress": m.progress, "milestone": m.milestone, "stopped": stopped})


# ---- the template: a frozen snapshot that forks the branch children ------------------------------------------------

def template(p, fire_turn, snap):
    cfg = G.cfg
    G.templates = []
    jobs = [(o, r) for r in range(cfg["reps"]) for o in cfg["options"] if not (o == "continue" and r == 0)]
    running = {}  # pid -> (option, rep, dir, start)

    def reap():
        now = time.monotonic()
        for pid, (o, r, d, t0) in list(running.items()):
            if now - t0 > CHILD_WALL:
                os.kill(pid, signal.SIGKILL)
        while running:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            if pid not in running:
                continue
            o, r, d, t0 = running.pop(pid)
            shutil.rmtree(d, ignore_errors=True)
            code = os.waitstatus_to_exitcode(status)
            if code != FINISHED:
                G.sem.release()  # the child died before its own release
                put(gdir() / f"p{p['point']}-{o}-r{r}.json", {
                    "seed": G.seed, "point": p["point"], "option": o, "rep": r, "fire_turn": fire_turn,
                    "status": "timeout" if now - t0 > CHILD_WALL else "crash", "exit": code,
                    "wall": round(now - t0, 1)})

    me = os.getpid()
    with G.waiting.get_lock():
        G.waiting.value += 1
    try:
        for o, r in jobs:
            while not G.sem.acquire(block=False):
                reap()
                time.sleep(0.2)
            d = copy_dir(snap)
            pid = os.fork()
            if pid == 0:
                running.clear()
                return child_setup(p, fire_turn, o, r, d)
            running[pid] = (o, r, d, time.monotonic())
    except BaseException:  # noqa: BLE001
        if os.getpid() != me:
            raise  # a child's own error
        traceback.print_exc()  # the template must not fall back into the bot and play on as a second parent
    finally:
        if os.getpid() == me:  # (a child leaves through the return above)
            with G.waiting.get_lock():
                G.waiting.value -= 1
    while running:
        reap()
        time.sleep(0.2)
    shutil.rmtree(snap, ignore_errors=True)
    os._exit(0)


# ---- a branch child --------------------------------------------------------------------------------------------------

def repoint_hackdir(nh, new):
    """NetHack builds every file path from fqn_prefix[], which points into settings.hackdir (src/nle.c)."""
    lib = ctypes.CDLL(nh.dlpath, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    buf = (ctypes.c_char * 4096).in_dll(lib, "settings")
    old = nh._vardir.rstrip("/").encode() + b"/"
    assert buf.value == old, (buf.value, old)
    path = new.rstrip("/").encode() + b"/"
    ctypes.memmove(buf, path + b"\0", len(path) + 1)
    nh._vardir = new  # the arena reads the death cause from <vardir>/xlogfile


def unshare_files(old, new):
    """fork() shares file offsets: reopen every regular file NetHack may read (nhdat) or write (its directory)."""
    for fd in [int(f) for f in os.listdir("/proc/self/fd")]:
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if not path.startswith("/") or path.startswith(("/proc/", "/dev/")) or not os.path.isfile(path):
            continue
        with open(f"/proc/self/fdinfo/{fd}") as fh:
            info = dict(line.split(":", 1) for line in fh.read().splitlines() if ":" in line)
        flags, pos = int(info["flags"].strip(), 8), int(info["pos"])
        acc = flags & os.O_ACCMODE
        target = new + path[len(old):] if path.startswith(old + "/") else path
        if acc != os.O_RDONLY and target == path:
            continue  # written by the parent only (its trace); a child never writes it
        nfd = os.open(target, acc | (flags & os.O_APPEND))
        os.lseek(nfd, pos, os.SEEK_SET)
        os.dup2(nfd, fd)
        os.close(nfd)


def child_setup(p, fire_turn, option, rep, d):
    cfg = G.cfg
    nh = G.env._env.nethack
    old = nh._vardir.rstrip("/")
    repoint_hackdir(nh, d)
    unshare_files(old, d)
    reseed = None
    if rep:
        s = np.random.SeedSequence([G.seed, p["point"], rep]).generate_state(4, dtype=np.uint64)
        reseed = [int(s[0]) & MAX_SEED, int(s[1]) & MAX_SEED]
        # NetHackChallenge blocks the seed methods on the instance: call the binding. The current lgen seed is
        # passed back: nle_set_seed re-inits only the lgen base, the per-dungeon level streams stay (nlernd.c).
        lgen = nh._pynethack.get_seeds()[3]
        nh._pynethack.set_seeds(reseed[0], reseed[1], False, lgen)
        random.seed(int(s[2]))
        np.random.seed(int(s[3]) % (1 << 32))
    tm = G.trace_mod
    G.keep.append(tm._fh)
    name = f"{G.seed}-p{p['point']}-{option}-r{rep}"
    tm._fh = open(G.out / "btrace" / f"{name}.jsonl", "w", buffering=1 << 20)
    tm._last_turn = tm._last_inv = None
    G.child = {"name": name, "point": p["point"], "option": option, "rep": rep, "reseed": reseed,
               "fire_turn": fire_turn, "fire_step": G.steps, "steps": G.steps, "obs": G.obs, "h": cfg["h"],
               "start": time.monotonic(), "avail": None}
    G.bot._driver._env.step = child_step
    if option != "continue":
        G.child["avail"] = G.branch_mod.available(G.bot._driver._agent)
        if option not in G.child["avail"]:
            finish("unavailable")
    return option


def child_step(action):
    c = G.child
    idx = G.adapter_mod._action_index(action)
    G.trace_mod.step(c["obs"], idx, G.bot._driver._agent)
    o, _reward, terminated, truncated = G.env.step(idx)
    c["steps"] += 1
    if c["steps"] % 100 == 0:
        add_work()
    if terminated:  # NLE zeroes blstats on the final observation: keep the last real one
        c["last_bl"] = [int(v) for v in c["obs"]["blstats"]]
    c["obs"] = obs = ro(o)
    if terminated:
        finish("died")
    if truncated:
        finish("truncated")
    if int(obs["blstats"][BL_TIME]) > c["fire_turn"] + c["h"]:
        finish("alive")
    if c["steps"] - c["fire_step"] > CHILD_STEPS:
        finish("steps")
    return G.adapter_mod._copy_observation(obs), 0.0, False, {}


def finish(status):
    c = G.child
    try:
        m = G.env.metrics()
        bl = c.get("last_bl") or [int(v) for v in c["obs"]["blstats"]]
        G.trace_mod._fh.close()
        zst(G.out / "btrace" / f"{c['name']}.jsonl")
        put(gdir() / f"p{c['point']}-{c['option']}-r{c['rep']}.json", {
            "seed": G.seed, "point": c["point"], "option": c["option"], "rep": c["rep"], "reseed": c["reseed"],
            "fire_step": c["fire_step"], "fire_turn": c["fire_turn"], "status": status,
            "end_turn": max(bl[BL_TIME], m.turns if status == "died" else 0), "end_step": c["steps"],
            "depth": bl[BL_DEPTH], "max_depth": m.max_depth, "end_status": m.end_status,
            "cause": m.cause_of_death, "avail": c["avail"], "b": bl,
            "wall": round(time.monotonic() - c["start"], 1)})
        G.sem.release()
        code = FINISHED
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 4
    os._exit(code)


# ---- launcher ----------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=16, help="CPU slots: playing games + running children")
    ap.add_argument("--games", type=int, default=0,
                    help="games in flight (default parallel: the slots balance themselves, see game())")
    ap.add_argument("--limit", type=int, default=0, help="only the first N games of the plan")
    a = ap.parse_args()
    G.cfg = cfg = json.loads(Path(a.plan).read_text())
    G.out = out = Path(a.out).resolve()
    for sub in ("results", "games", "btrace", "trace"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    done = {int(f.stem) for f in (out / "results").glob("*.json")}
    todo = [g for g in cfg["games"][:a.limit or None] if g["seed"] not in done]
    for g in todo:  # leftovers of games that did not finish
        s = g["seed"]
        shutil.rmtree(out / "games" / str(s), ignore_errors=True)
        for f in [*(out / "btrace").glob(f"{s}-*"), *(out / "trace").glob(f"{s}.*")]:
            f.unlink()
    print(f"{len(done)} games done, {len(todo)} to play", flush=True)
    ctx = multiprocessing.get_context("fork")
    G.sem, G.waiting, G.work = ctx.Semaphore(a.parallel), ctx.Value("i", 0), ctx.Value("q", 0)
    ctypes.CDLL(None).prctl(36, 1)  # PR_SET_CHILD_SUBREAPER: orphaned templates are reaped here
    width = a.games or a.parallel
    running = {}
    t0, finished = time.monotonic(), 0
    last = [t0, 0]

    def wait_one():
        nonlocal finished
        while True:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid:
                break
            now = time.monotonic()
            if now - last[0] >= 60:  # (the image's wall clock is frozen: elapsed minutes, not the time of day)
                print(f"min {(now - t0) / 60:.0f}: {(G.work.value - last[1]) / (now - last[0]):.0f} steps/s, "
                      f"{len(running)} games", flush=True)
                last[:] = [now, G.work.value]
            time.sleep(0.5)
        seed = running.pop(pid, None)
        if seed is None:
            return  # an adopted orphan
        code = os.waitstatus_to_exitcode(status)
        if code < 0:
            G.sem.release()  # killed while holding its slot
        if code != 0:
            print(f"game {seed} failed: exit {code}", flush=True)
        finished += 1
        if finished % 10 == 0:
            print(f"{finished}/{len(todo)} games, {finished / (time.monotonic() - t0) * 60:.1f}/min", flush=True)

    for g in todo:
        while len(running) >= width:
            wait_one()
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                game(g)
                code = 0
            except BaseException:  # noqa: BLE001
                traceback.print_exc()
            finally:
                os._exit(code)
        running[pid] = g["seed"]
    while running:
        wait_one()
    print(f"done: {finished} games in {time.monotonic() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
