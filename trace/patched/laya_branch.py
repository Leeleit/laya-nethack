"""Counterfactual branches for training laya: force one strategy-level decision at a chosen step.

Off unless /out/branch.json exists (collect.py --branch puts it there; plan_branches.py writes it). The arena
games are deterministic under a frozen wall clock (image laya-frozen), so the bot replays its recorded game
exactly up to the branch step; there, if the observation's blstats equal the recorded ones, the top-priority
strategy below runs the forced option once through the bot's own code, and the bot plays on by itself.

branch.json: [{"seed", "k": step index (0 = first act), "b": recorded blstats at k, "b0": at step 0,
               "option", "h": horizon in game turns}, ...]   at most one entry per game.
A game whose first observation matches no "b0", an unavailable option, and a branch "h" turns after firing go
idle: the bot sends ESC until the arena's no-progress timeout ends the episode, so the result (turns, depth,
death) is the state at firing + h. Each firing appends {"seed", "k", "option", "done", "turn", "step", "avail",
"note"} to /out/branch_log.jsonl.

Options (see _run): pray, elbereth, quaff_heal, stairs, elbereth_rest (engrave and hold the square like the
bot's own Elbereth rest), upstairs (walk to the nearest reachable "<" and climb it), dig_down
(dig a hole down with the bot's pick-axe or mattock).
"""
import importlib
import json
import os

PATH = "/out/branch.json"
LOG = "/out/branch_log.jsonl"
_BL_TIME = 20
_spec = None
_steps = 0
_cands = []        # entries whose "b0" matches this game's first observation (several seeds can share it)
_entry = None      # this game's entry: the candidate whose "b" matched at its step k
_armed = False
_fire_turn = None  # game turn the option ran at; None before firing
_idle = False


def _log(rec):
    try:
        with open(LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _healing(agent):
    mod = importlib.import_module(type(agent).__module__)
    return [i for i in mod.flatten_items(agent.inventory.items) if i.is_unambiguous() and
            i.category == mod.nh.POTION_CLASS and i.object.name in ('healing', 'extra healing', 'full healing')]


def _stairs(agent):
    mod = importlib.import_module(type(agent).__module__)
    here = agent.current_level().objects[agent.blstats.y, agent.blstats.x]
    if here in mod.G.STAIR_DOWN:
        return '>'
    if here in mod.G.STAIR_UP and agent.blstats.depth > 1:
        return '<'
    return None


def _upstairs(agent):
    """The nearest reachable '<' (dive_logic.retreat_upstairs without its HP/hostile conditions and its 12-step
    reach: a digger rarely knows a '<' that close -- in a local test it knew one at 1 of 8 points at all)."""
    pkg = type(agent).__module__.rsplit(".", 1)[0]
    mod, utils = importlib.import_module(type(agent).__module__), importlib.import_module(pkg + ".utils")
    level = agent.current_level()
    if agent.blstats.depth <= 1 or level.dungeon_number == mod.Level.SOKOBAN:
        return None
    dis = agent.bfs()
    ups = [p for p in zip(*utils.isin(level.objects, mod.G.STAIR_UP).nonzero()) if dis[p] >= 0]
    return min(ups, key=lambda p: dis[p]) if ups else None


def _dig_tool(agent):
    """A digging tool the bot can dig down with here (the conditions of dive_logic._dig_escape_action's level)."""
    dive = agent.global_logic.dive
    dl = importlib.import_module(type(dive).__module__)
    level = agent.current_level()
    if level.dungeon_number not in dl.MAIN_LINE or level.dungeon_number == dl.GEHENNOM or \
            level.key() in dive.undiggable:
        return None
    return dive.digging_tool()


def available(agent):
    """Options that can be executed in the current state."""
    out = ["pray"]
    if agent.can_engrave():
        out += ["elbereth", "elbereth_rest"]
    if _healing(agent):
        out.append("quaff_heal")
    if _stairs(agent):
        out.append("stairs")
    if _upstairs(agent) is not None:
        out.append("upstairs")
    if _dig_tool(agent) is not None:
        out.append("dig_down")
    return out


def _run(agent, option):
    if option == "pray":
        agent.pray()
    elif option == "elbereth":
        agent.engrave("Elbereth")
    elif option == "quaff_heal":
        agent.inventory.quaff(_healing(agent)[0])
    elif option == "stairs":
        agent.move(_stairs(agent))
    elif option == "elbereth_rest":
        # the bot's own Elbereth rest (dive_logic.elbereth_rest), entered now: once resting it holds the square
        # until HP >= ELBERETH_REST_UNTIL (0.85) or no hostile is near, instead of only below 0.4
        agent.engrave("Elbereth")
        agent.global_logic.dive._elbereth_resting = True
    elif option == "upstairs":
        y, x = _upstairs(agent)
        if (agent.blstats.y, agent.blstats.x) != (y, x):
            agent.go_to(y, x)
        if (agent.blstats.y, agent.blstats.x) == (y, x):
            agent.move("<")
    elif option == "dig_down":
        # the bot's own dig (Elbereth first when hostiles are near, up to 3 apply tries); the hole leaves the
        # level's monsters behind
        agent.global_logic.dive.dig_with_tool(_dig_tool(agent))
    else:
        raise ValueError(option)


def install(agent):
    """Wrap the agent's global strategy with the branch strategy on top. Called by bot.py after reset."""
    global _spec, _steps, _cands, _entry, _armed, _fire_turn, _idle
    _steps, _cands, _entry, _armed, _fire_turn, _idle = 0, [], None, False, None, False
    if _spec is None:
        _spec = json.load(open(PATH)) if os.path.exists(PATH) else []
    if not _spec:
        return
    Strategy = importlib.import_module(type(agent).__module__.rsplit(".", 1)[0] + ".strategy").Strategy

    def branch():
        global _fire_turn, _idle
        if not _armed or _fire_turn is not None:
            yield False
        yield True
        _fire_turn = int(agent.blstats.time)
        rec = {"seed": _entry.get("seed"), "k": _entry["k"], "option": _entry["option"], "turn": _fire_turn,
               "step": _steps}
        try:
            rec["avail"] = available(agent)  # lets plan_branches.py --filter drop branches that cannot run
            if _entry["option"] not in rec["avail"]:
                rec.update(done=False, note="unavailable")
                _idle = True
            else:
                _run(agent, _entry["option"])
                rec.update(done=True)
        finally:
            _log(rec)

    gl = agent.global_logic
    orig = gl.global_strategy
    gl.global_strategy = lambda: orig().preempt(agent, [Strategy(branch)])


def observe(observation):
    """Called by bot.py on every observation before the bot acts on it; True: send ESC instead (idle)."""
    global _steps, _cands, _entry, _armed, _idle
    if not _spec:
        return False
    if not _idle:
        bl = [int(v) for v in observation["blstats"]]
        if _steps == 0:
            _cands = [e for e in _spec if e["b0"] == bl]
            _idle = not _cands
        elif not _armed:
            _entry = next((e for e in _cands if _steps == e["k"] and bl == e["b"]), None)
            _armed = _entry is not None
            if not _armed and _steps >= max(e["k"] for e in _cands):  # diverged from every candidate
                _log({"seeds": [e.get("seed") for e in _cands], "step": _steps, "done": False, "note": "no match"})
                _idle = True
        if _fire_turn is not None and bl[_BL_TIME] > _fire_turn + _entry["h"]:
            _idle = True
    _steps += 1
    return _idle
