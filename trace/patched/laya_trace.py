"""Decision trace for training laya: one JSONL file per episode under /out/trace.

Named <seed>.jsonl in the laya-frozen image (image/patch_arena.py), else <pid>-<episode>.jsonl.
Off unless that directory exists. nethackers/collect.py creates it in the arena's /out mount;
`nethackers eval` and the official arena never do, so there this module is a no-op.

Line 1: {"identity": ..., "package": ...}. Then one line per action:
  a   action index the bot chose on this observation
  s   preempting strategies executing it, outermost first (empty: the base explore/dive strategy)
  b   blstats (NLE order)
  m   top-line message, if any
  On the first action of each game turn also:
  scr the 24 tty rows
  mon monsters on the map other than the hero: [name, dx, dy, kind] (kind: pet / detected / ''),
      nearest first, at most 16
  inv inventory lines ("a - ..."), only when they changed since the last record that carried them
"""
import json
import os
import re

import numpy as np

try:
    import nle.nethack as nh
except ImportError:  # outside the arena image
    nh = None

DIR = "/out/trace"
_ENABLED = os.path.isdir(DIR) and nh is not None
_BL_X, _BL_Y, _BL_TIME = 0, 1, 20  # nle.nethack.NLE_BL_*
_FUNC = re.compile(r"<function (\S+) at ")

_names = {}
_fh = None
_episodes = 0
_last_turn = None
_last_inv = None


def _name(strategy):
    key = id(strategy)
    if key not in _names:
        cfg = strategy.config
        while isinstance(cfg, dict):  # condition/until/preempt/every wrap 'strategy', repeat 'repeat', before '1'
            cfg = cfg.get("strategy", cfg.get("repeat", cfg.get("1")))
        m = _FUNC.match(str(cfg))
        _names[key] = m.group(1) if m else str(cfg)[:80]
    return _names[key]


def _monsters(glyphs, x0, y0):
    g = np.asarray(glyphs)
    # monster glyphs: normal, pet, invisible marker, detected (all below the corpse range), and ridden
    mask = (g < nh.GLYPH_BODY_OFF) | ((g >= nh.GLYPH_RIDDEN_OFF) & (g < nh.GLYPH_OBJ_OFF))
    mask[y0, x0] = False  # the hero
    out = []
    for y, x in zip(*np.nonzero(mask)):
        v = int(g[y, x])
        if v == nh.GLYPH_INVIS_OFF:
            name, kind = "unseen monster", "remembered"
        else:
            name = nh.permonst(nh.glyph_to_mon(v)).mname
            kind = "pet" if nh.glyph_is_pet(v) else "detected" if nh.glyph_is_detected_monster(v) else ""
        out.append([name, int(x) - x0, int(y) - y0, kind])
    out.sort(key=lambda r: max(abs(r[1]), abs(r[2])))
    return out[:16]


def begin(header):
    global _fh, _episodes, _last_turn, _last_inv
    if not _ENABLED:
        return
    end()
    _episodes += 1
    _last_turn = _last_inv = None
    try:
        # <seed>.jsonl when the arena passes the trajectory id (our laya-frozen image), else <pid>-<episode>.jsonl
        tid = os.environ.get("LAYA_TRAJECTORY_ID")
        name = f"{tid}.jsonl" if tid else f"{os.getpid()}-{_episodes}.jsonl"
        _fh = open(os.path.join(DIR, name), "w", buffering=1 << 20)
        _fh.write(json.dumps(header) + "\n")
    except OSError:
        _fh = None


def step(observation, action, agent):
    global _last_turn, _last_inv
    if _fh is None:
        return
    try:
        bl = [int(v) for v in observation["blstats"]]
        rec = {"a": int(action), "s": [_name(s) for s in getattr(agent, "strategy_stack", ())], "b": bl}
        msg = bytes(observation["message"]).split(b"\0")[0].decode("latin-1").strip()
        if msg:
            rec["m"] = msg
        if bl[_BL_TIME] != _last_turn:
            _last_turn = bl[_BL_TIME]
            rec["scr"] = [bytes(row).decode("latin-1").rstrip() for row in observation["tty_chars"]]
            rec["mon"] = _monsters(observation["glyphs"], bl[_BL_X], bl[_BL_Y])
            inv = [f"{chr(int(c))} - {s}" for c, s in zip(observation["inv_letters"], (
                bytes(r).split(b"\0")[0].decode("latin-1").strip() for r in observation["inv_strs"])) if s]
            if inv != _last_inv:
                _last_inv = rec["inv"] = inv
        _fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001 -- tracing must never cost the bot an action
        pass


def end():
    global _fh
    if _fh is not None:
        try:
            _fh.close()
        except OSError:
            pass
        _fh = None
