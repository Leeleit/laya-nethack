"""The bot's keys parsed into commands -- the action vocabulary of a laya policy that plays by itself.

A trace record carries the key the bot pressed ("a", an index into nle.nethack.ACTIONS) and the top-line
message of the observation it pressed it at ("m"). The game's own prompts tell the keys apart: a key pressed at
"What do you want to eat? [fg or ?*]" is an item, at "In what direction?" a direction, at "...? [yn] (n)" an
answer, at a getlin prompt ("What do you want to write in the dust here?") text up to Enter. Dropped as
interface, not decisions: --More-- (space), Esc, and the bot's free information queries -- look here (:),
/M (monsters on the map), #terrain, \\ (discoveries), ^X, #enhance, the autopickup toggle (@).

parse(steps) -> list of Cmd for one stretch of keys (e.g. one game turn: from a record with a screen to the next).
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# key code of each action index: [int(a) for a in nle.nethack.ACTIONS] (NetHackChallenge's 121 actions),
# saved so the dataset builders run without nle
ACTION_KEYS = json.loads((Path(__file__).parent / "bc_actions.json").read_text())

DIRS = {"k": "north", "u": "northeast", "l": "east", "n": "southeast",
        "j": "south", "b": "southwest", "h": "west", "y": "northwest"}
# a prompt ends with "?", "]" or ":" (getlin), optionally followed by the default answer "(n)"
PROMPT = re.compile(r"(\?|\]|:)( \(.\))?$")
TRAVEL_DONE = {".", ",", ";", ":"}
MENU_CMDS = {",", "D", "A", "#loot", "#force"}  # may open a selection menu answered by letters + Enter
INFO_EXT = {"te", "terrain", "overview", "attributes"}


def address(s):
    """laya_serve's address: a Unix socket path (the game container has no network but shares a directory with the server)
    or host:port (the local PC: Docker Desktop forwards host.docker.internal to the Windows host's 127.0.0.1)."""
    if "/" in s:
        return s
    host, port = s.rsplit(":", 1)
    return host, int(port)


def key(a):
    """A trace's "a" is an index into nle.nethack.ACTIONS (NetHackChallenge's action set), not a key code."""
    if a is None:
        return "?"
    c = ACTION_KEYS[a]
    if 33 <= c < 127:
        return chr(c)
    if c >= 128 and 33 <= c - 128 < 127:
        return "M-" + chr(c - 128)  # meta: extended commands
    return {32: "␣", 27: "⎋", 13: "⏎", 10: "⏎"}.get(c, f"<{c}>")


@dataclass
class Cmd:
    cmd: str                                   # command key ("l", "s", "t", "M-p", "#loot", ...)
    count: str = ""                            # count prefix ("5" in "5s")
    fight: bool = False                        # F prefix: attack in the direction without moving
    args: list = field(default_factory=list)   # (prompt, key) answered after the command
    menu: list = field(default_factory=list)   # keys pressed in selection menus (letters, Enter)
    step: int = 0                              # index of the command key in the stretch
    after: str = ""                            # the message right after the command key


def _menu(steps, i):
    """Length of a menu selection at i: keys without a message up to and including Enter, else 0."""
    j = i
    while j < len(steps) and j - i < 80:
        k, m = steps[j]
        if k == "⏎":
            return j - i + 1
        if m or k == "⎋" or len(k) > 1 and k != "␣":
            return 0
        j += 1
    return 0


def parse(steps, nxt=""):
    """steps: [(key, message)] -> [Cmd]; information queries and interface keys are dropped. nxt: the message of
    the record after the stretch (what a command at its very end led to: "b - a dagger.", "You can't ...")."""
    out, i, n = [], 0, len(steps)
    while i < n:
        k, m = steps[i]
        if k in ("␣", "⎋", ":", "\\", "<24>", "@", "M-e", "?"):
            i += 1
            if k == "M-e":  # #enhance: a skill menu, answered by letters + Enter when a skill can advance
                i += _menu(steps, i)
            continue
        if k == "/":  # what is: a selector key follows (M: list the monsters shown)
            i += 2
            continue
        if m and PROMPT.search(m):  # an answer to a prompt left open by an earlier stretch
            out.append(Cmd("answer", args=[(m, k)], step=i))
            i += 1
            continue
        if k == "#":
            j, name = i + 1, ""
            while j < n and steps[j][0] not in ("⏎", "⎋") and len(steps[j][0]) == 1:
                name += steps[j][0]
                j += 1
            i = j + 1
            if name in INFO_EXT:  # #terrain: a view menu, then Esc
                while i < n and steps[i][0] != "⎋":
                    i += 1
                i += 1
                continue
            c = Cmd("#" + name, step=i - 1)
        else:
            count, fight = "", False
            while k.isdigit() and i + 1 < n:
                count += k
                i += 1
                k, m = steps[i]
            if k == "F" and i + 1 < n and steps[i + 1][0] in DIRS:
                fight = True
                i += 1
                k, m = steps[i]
            c = Cmd(k, count=count, fight=fight, step=i)
            i += 1
        if c.cmd == "_":  # travel: cursor keys up to a confirming key
            while i < n:
                c.menu.append(steps[i][0])
                i += 1
                if c.menu[-1] in TRAVEL_DONE:
                    break
            out.append(c)
            continue
        last = None
        while i < n:
            k, m = steps[i]
            if k == "␣":
                i += 1
                continue
            if m and (PROMPT.search(m) or last and m.startswith(last)):
                if not (last and m.startswith(last)):
                    last = m
                c.args.append((last, k))
                i += 1
                continue
            if c.cmd in MENU_CMDS and (ln := _menu(steps, i)):
                c.menu += [s[0] for s in steps[i:i + ln]]
                i += ln
                continue
            break
        c.after = steps[c.step + 1][1] if c.step + 1 < n else nxt
        out.append(c)
    return out


# ---- what laya chooses: the command question and the questions for its arguments ----

COMMANDS = {  # short: the options share laya's 256-token head window with the start of the state
    **{d: None for d in DIRS.values()},
    "search": "search one turn (or wait)",
    "search5": "search five turns",
    "pickup": "pick up items here",
    "throw": None,
    "apply": "dig with a pick-axe",
    "eat": None,
    "elbereth": "engrave Elbereth",
    "pray": None,
    "down": "go downstairs",
    "up": "go upstairs",
    "kick": None,
    "item": "wear, take off, wield, quaff, read, put on, remove, zap or drop an item",
}
ITEM_CMDS = {"W": "wear", "T": "take off", "w": "wield", "q": "quaff", "r": "read", "P": "put on",
             "R": "remove", "z": "zap", "d": "drop", "D": "drop"}
SIMPLE = {",": "pickup", "t": "throw", "a": "apply", "e": "eat", "M-p": "pray", ">": "down", "<": "up",
          "<4>": "kick"}
DIR_OPTS = {**{v: k for k, v in DIRS.items()}, "up": "<", "down": ">", "here": "."}  # option -> key
DIR_OF_KEY = {v: k for k, v in DIR_OPTS.items()}
VERB = {"throw": "throws", "apply": "applies", "kick": "kicks", "zap": "zaps"}
BRACKET = re.compile(r"\s*\[([^\]]*)\]")
ITEM_TEXT = 48   # characters of an item's name in an option: 16 options must fit laya's 256-token head window
MAX_ITEMS = 16

# share of each command's turns bc_build.py keeps for training (moves and searches are ~90 % of turns, and
# neighbouring turns are near copies). laya learns p(c | s) * KEEP[c] (renormalised), so a player multiplies its
# probabilities by UNDO = 1 / KEEP (renormalised) to get the bot's own rates back (multiplying by KEEP
# instead weighs rare commands up ~44x against moves).
KEEP = {**{c: 0.15 for c in COMMANDS}, "pickup": 0.4, "throw": 0.5,
        **{c: 1.0 for c in ("apply", "eat", "elbereth", "pray", "down", "up", "kick", "item")}}
UNDO = {c: 1 / k for c, k in KEEP.items()}

Q_COMMAND = {"type": "choice", "criteria": COMMANDS,
             "instructions": "Which command should the hero do now? A direction moves there or attacks what is there."}
Q_ITEM_CMD = {"type": "choice", "instructions": "How should the hero use an item?",
              "criteria": {v: None for v in dict.fromkeys(ITEM_CMDS.values())}}


def allowed(prompt):
    """Keys a prompt offers: "[$aD or ?*]" -> $ a D; ranges "fjl-nr" -> f j l m n r; "- ajnps" -> - a j n p s."""
    m = BRACKET.search(prompt)
    if not m:
        return []
    s = m.group(1).split(" or ")[0].replace(" ", "")
    out, i = [], 0
    while i < len(s):
        if i + 2 < len(s) and s[i + 1] == "-" and s[i].isalpha() and s[i + 2].isalpha():
            out += [chr(c) for c in range(ord(s[i]), ord(s[i + 2]) + 1)]
            i += 3
        else:
            out.append(s[i])
            i += 1
    return out


def q_item(prompt, keys, inv):
    """Which item, at a game prompt "What do you want to <verb>? [...]"; options: the offered letters with names."""
    names = {line[0]: line[4:] for line in inv if len(line) > 4}
    names.setdefault("-", "your bare hands")
    return {"type": "choice", "instructions": BRACKET.sub("", prompt).strip(),
            "criteria": {k: names.get(k, "")[:ITEM_TEXT] or None for k in keys[:MAX_ITEMS]}}


def q_dir(verb, keys=None):
    opts = [DIR_OF_KEY[k] for k in keys if k in DIR_OF_KEY] if keys else list(DIR_OPTS)
    return {"type": "choice", "instructions": f"The hero {verb}. In what direction?",
            "criteria": {o: None for o in opts}}


def q_eat_here(prompt):
    return {"type": "noul", "instructions": BRACKET.sub("", prompt).replace(" (n)", "").strip()}


def gold_choice(label):
    return {"probabilities": {label: 1.0}, "label": label}


def gold_noul(v):
    return {"probabilities": {"true": float(v), "false": float(1 - v)}, "label": str(bool(v)).lower()}


# the game refused, or nothing happened: no time passed, the turn's real command comes after (bc_audit.py)
NOOP = ("You can't", "You cannot", "There is nothing here", "You don't have", "You have no", "Never mind",
        "The bag is empty")


def command_of(c):
    """Cmd -> its option in COMMANDS, or None: outside the vocabulary (travel, #loot, a test engraving, ...) or
    one of the bot's looks that only resemble a command (bc_audit.py): `,` whose menu it closed
    without taking anything (~35 % of its pickups), `a` on its bag (most applies without a direction)."""
    if c.after.startswith(NOOP):
        return None
    if c.cmd in DIRS:
        return DIRS[c.cmd]
    if c.cmd in ("s", "."):
        return "search5" if c.count and int(c.count) > 1 else "search"
    if any(k == "⎋" for _, k in c.args):
        return None  # cancelled at a prompt: the bot only looked at what it could choose
    if c.cmd == ",":
        return "pickup" if c.menu or " - " in c.after else None
    if c.cmd == "a":
        return "apply" if any("direction" in p for p, _ in c.args) else None
    if c.cmd == "E":
        text = "".join(k for p, k in c.args if "write in" in p and len(k) == 1 and k != "⏎")
        withwhat = next((k for p, k in c.args if "write with" in p), None)
        return "elbereth" if text == "Elbereth" and withwhat == "-" else None
    if c.cmd in ITEM_CMDS:
        return "item"
    return SIMPLE.get(c.cmd)


def labelled(c, inv):
    """The questions a Cmd answers and their gold: {name: (question, gold)}; empty outside the vocabulary."""
    cmd = command_of(c)
    if cmd is None:
        return {}
    out = {"command": (Q_COMMAND, gold_choice(cmd))}
    if cmd == "item":
        out["item_cmd"] = (Q_ITEM_CMD, gold_choice(ITEM_CMDS[c.cmd]))
    item = None
    for p, k in c.args:
        pl = p.lower()
        if "eat it?" in pl and "eat_here" not in out and k in "yn":
            out["eat_here"] = (q_eat_here(p), gold_noul(k == "y"))
        elif pl.startswith("what do you want to") and "[" in p and "item" not in out and cmd != "elbereth":
            keys = allowed(p)
            keys = keys if k in keys else [k] + keys
            keys = keys if keys.index(k) < MAX_ITEMS else [k] + [x for x in keys if x != k]
            q = q_item(p, keys, inv)
            out["item"] = (q, gold_choice(k))
            item = q["criteria"].get(k) or "it"
        elif "direction" in pl and "direction" not in out and k in DIR_OF_KEY:
            verb = VERB.get(ITEM_CMDS.get(c.cmd, cmd), "acts")
            verb += f" {item}" if item and verb != "kicks" else ""
            out["direction"] = (q_dir(verb, allowed(p) or None), gold_choice(DIR_OF_KEY[k]))
    return out


# ---- the state laya reads (the same text from a trace record and from a live observation) ----

NOISE = ("You see no objects here.", "Never mind.", "Showing known terrain", "Autopickup:", "What do you want",
         "#")


def message_ok(m):
    """Messages the state shows: not prompts, nor the output of the bot's information queries."""
    return bool(m) and not PROMPT.search(m) and not m.startswith(NOISE)


def where(dx, dy):  # as build_v2.where
    if max(abs(dx), abs(dy)) == 1:
        return "adjacent " + ("north" if dy < 0 else "south" if dy > 0 else "") + ("west" if dx < 0 else "east" if dx > 0 else "")
    parts = [f"{abs(dx)} {'east' if dx > 0 else 'west'}" if dx else "", f"{abs(dy)} {'south' if dy > 0 else 'north'}" if dy else ""]
    return " ".join(p for p in parts if p)


HP, MAXHP, DEPTH, AC, XL, TIME = 10, 11, 12, 16, 18, 20  # NLE blstats indices


_LOCALS = re.compile(r"\.<locals>|\.<lambda>|\.f$")  # as build_laya_data.short


def behaviour(stack):
    """The bot's strategy stack as risk-v2 showed it; None (laya without the bot) -> unknown."""
    if stack is None:
        return "unknown"
    return " > ".join(_LOCALS.sub("", s) for s in stack) or "explore/descend"


def state_text(identity, b, scr, mon, hp_back, msgs, inv, last, stack=None):
    """risk-v2's state (build_v2.state_text) plus the hero's last commands; the bot's strategy stack when laya
    plays inside the bot (bot + model), "unknown" when it plays alone (bc_build.py trains on both); last None:
    "unknown" too (bc-v2 hides it in part of the states against copycat repeats). Inventory
    last: laya cuts the end of a state that does not fit, and the item questions list the items they offer."""
    x, y = b[0], b[1] + 1  # map row y is tty row y + 1
    hp = lambda v: f"{v[0]}/{v[1]}" if v else "unknown"
    mons = "; ".join(f"{n}{' (' + k + ')' if k else ''} {where(dx, dy)}" for n, dx, dy, k in mon) or "none"
    rows = [scr[r] if r < len(scr) else "" for r in range(max(1, y - 5), min(22, y + 6))]
    crop = "\n".join(row.ljust(x + 13)[max(0, x - 12):x + 13].rstrip() for row in rows)
    level = "\n".join(r for r in scr[1:22] if r.strip())
    return (f"Character: {identity}. Bot behaviour: {behaviour(stack)}.\n"
            f"HP {b[HP]}/{b[MAXHP]}; 10 turns ago {hp(hp_back[0])}; 30 turns ago {hp(hp_back[1])}. "
            f"Dlvl {b[DEPTH]}, XL {b[XL]}, AC {b[AC]}, turn {b[TIME]}.\n"
            f"Status: {scr[22].strip() if len(scr) > 22 else ''} | {scr[23].strip() if len(scr) > 23 else ''}\n"
            f"Monsters in view: {mons}\n"
            f"Recent messages: {' | '.join(msgs) or 'none'}\n"
            f"Last commands: {'unknown' if last is None else ', '.join(last) or 'none'}\n"
            f"Map around the hero (@):\n{crop}\n"
            f"Level map:\n{level}\n"
            f"Inventory: {'; '.join(inv) or 'unknown'}")
