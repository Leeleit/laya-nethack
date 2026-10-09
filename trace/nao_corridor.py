"""The ascenders' corridor: experience level, AC, max HP, game turn and key items of human players who ascended
(NLD-NAO, alt.org) when they first reach each Dlvl -- the reference for the bot's profile (bot_profile.py).

Games: the ascensions of one role/race in the NAO xlogfiles (3.4.3 and 3.6.x). A game's ttyrecs are its player's
files created between its start (- 5 min) and end, as nle.dataset.populate_db.assign_ttyrecs_to_games does (one
game at a time per player). The frames come from NLE's own ttyrec converter (24x80); the bottom lines give
Dlvl / Xp (Exp) / AC / HP / T, inventory-style lines ("k - a wand of digging.", the 'i' menu, the final
"possessions identified" list) the items, counted as owned from their first sight on.
  list:    python nao_corridor.py list --nld /data/laya/nethack/nld/nld-nao --out nao_val_dwa.json
             [--outcome died --min-maxlvl 10 --sample 3000]   (the players who died, for the comparison)
  profile: python nao_corridor.py profile --games nao_val_dwa.json --out results/nao_corridor_val_dwa [--workers N]
           [--root /work/nld_asc]   (needs nle)
           -> <out>.txt (tables as bot_profile.py), <out>.parquet (a row per game and new deepest level)
  compare: python nao_corridor.py compare --asc A.parquet --died D.parquet [--bot bot_profile.parquet] --out X.txt
           -> ascended vs died soon after vs died deeper (vs the bot) at each Dlvl
"""
import argparse
import datetime
import json
import os
import random
import re
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

from state_fmt import ASK, MSG_BACK, UI_MSG

FIVE_MINS = 5 * 60
ITEMS = {  # as bot_profile.ITEMS
    "excalibur": r"excalibur",
    "reflection": r"shield of reflection|polished silver shield|silver dragon scale|amulet of reflection",
    "magic_res": r"cloak of magic resistance|gray dragon scale|magicbane",
    "dragon_mail": r"dragon scale mail",
    "levitation": r"levitation",
    "speed_boots": r"speed boots",
    "digging": r"wand of digging|pick-axe|dwarvish mattock",
    "unicorn_horn": r"unicorn horn",
    "blindfold": r"blindfold|towel",
    "lizard": r"lizard corpse",
}
STATUS = re.compile(rb"Dlvl:(\d+)\s.*?HP:(-?\d+)\((\d+)\).*?AC:(-?\d+)\s+(?:Xp|Exp):(\d+)(?:/\d+)?(?:\s+T:(\d+))?")
INV = re.compile(rb"(?:^|\s)[a-zA-Z] - (.+?)\s*$")


def stamp(name):  # "2019-07-25.22:03:29.ttyrec.bz2" (or HH_MM_SS) -> epoch, UTC as in populate_db
    ts = re.sub(r"(.*\.\d\d)_(\d\d)_(\d\d.*)", r"\1:\2:\3", name[:-11])
    try:
        return datetime.datetime.fromisoformat(ts).replace(tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        return None


def xlog(path):
    sep = ":" if path.endswith(".txt") else "\t"
    with open(path, encoding="latin-1") as f:
        for line in f:
            rec, key = {}, None
            for tok in line.rstrip("\n").split(sep):
                k, eq, v = tok.partition("=")
                if eq and k.isidentifier():
                    key, rec[k] = k, v
                elif key:  # a separator inside a value (':' in 3.4.3 death reasons)
                    rec[key] += sep + tok
            yield rec


def outcome_ok(death, outcome):
    if outcome == "ascended":
        return death == "ascended"
    if outcome == "any":  # ascended or died: a known fate (quitting and escaping are not)
        return not death.startswith(("quit", "escaped"))
    return death != "ascended" and not death.startswith(("quit", "escaped"))  # died


def list_games(a):
    root = Path(a.nld)
    users = {d.lower(): d for d in os.listdir(root / "nld-nao-unzipped")}
    stamps = {}  # user dir -> [(time, file)]
    games = []
    for xf in sorted(root.glob("xlogfile.*")):
        if xf.name.endswith("dev") or xf.name == "blacklist.txt":
            continue
        for r in xlog(str(xf)):
            if a.role != "any" and r.get("role") != a.role or a.race != "any" and r.get("race") != a.race:
                continue
            if not outcome_ok(r.get("death", ""), a.outcome):
                continue
            if not (r.get("version", "").startswith("3.4.3") or r.get("version", "").startswith("3.6")):
                continue
            if int(r.get("maxlvl", 0)) < a.min_maxlvl or a.max_maxlvl and int(r.get("maxlvl", 0)) > a.max_maxlvl:
                continue
            d = users.get(r["name"].lower())
            if d is None or "starttime" not in r or "endtime" not in r:  # old 3.4.3 lines have no times
                continue
            if d not in stamps:
                stamps[d] = [(t, f) for f in os.listdir(root / "nld-nao-unzipped" / d)
                             if f.endswith(".ttyrec.bz2") and (t := stamp(f)) is not None]
            start, end = int(r["starttime"]), int(r["endtime"])
            files = sorted((t, f) for t, f in stamps[d] if start - FIVE_MINS <= t <= end)
            if files:
                paths = [str(root / "nld-nao-unzipped" / d / f) for _, f in files]
                games.append(dict(name=r["name"], version=r["version"], turns=int(r["turns"]), start=start, end=end,
                                  maxlvl=int(r["maxlvl"]), death=r["death"], files=paths,
                                  bytes=sum(os.path.getsize(p) for p in paths),
                                  character="-".join(r.get(k, "?") for k in ("role", "race", "align", "gender")),
                                  deathlev=int(r.get("deathlev", 0))))
    if a.sample and len(games) > a.sample:
        games = sorted(random.Random(0).sample(games, a.sample), key=lambda g: g["start"])
    Path(a.out).write_text(json.dumps(games))
    mb = sum(g["bytes"] for g in games) / 1e6
    print(f"{len(games)} {a.outcome} games with ttyrecs, {sum(len(g['files']) for g in games)} files, {mb:.0f} MB bz2")


def list_aa(a):
    """AutoAscend's games in NLD-AA (<root>/nle_data/<run>/nle.<pid>.xlogfile, each line naming its ttyrec3 file):
    a summary of how deep it gets, per role and overall, and the games of one role/race in list's json format."""
    import collections
    games, by_role = [], collections.defaultdict(list)
    for xf in sorted(Path(a.nld_aa).glob("nle_data/*/*.xlogfile")):
        for r in xlog(str(xf)):
            if "ttyrecname" not in r or "maxlvl" not in r:
                continue
            death, maxlvl = r.get("death", ""), int(r["maxlvl"])
            by_role[r.get("role", "?")].append((maxlvl, death == "ascended", int(r.get("turns", 0))))
            if (a.role != "any" and r.get("role") != a.role or a.race != "any" and r.get("race") != a.race
                    or not outcome_ok(death, "any")):
                continue
            p = xf.parent / r["ttyrecname"]
            if p.exists():
                games.append(dict(name=f"aa{r.get('uid', '')}-{r['ttyrecname']}", version=r["version"],
                                  turns=int(r.get("turns", 0)), start=int(r.get("starttime", 0)),
                                  end=int(r.get("endtime", 0)), maxlvl=maxlvl, death=death, files=[str(p)],
                                  bytes=p.stat().st_size, ttyrec_version=3, deathlev=int(r.get("deathlev", 0)),
                                  character="-".join(r.get(k, "?") for k in ("role", "race", "align", "gender"))))
    Path(a.out).write_text(json.dumps(games))
    out = [f"AutoAscend (NLD-AA {a.nld_aa}): games per role, deepest level reached p25/median/p75/p95, "
           f"share reaching Dlvl >= 10 / 25 / 30, ascensions, median turns"]
    allg = [x for v in by_role.values() for x in v]
    for role, v in sorted(by_role.items()) + [("ALL", allg)]:
        d = sorted(x[0] for x in v)
        qq = lambda f: d[min(len(d) - 1, int(f * len(d)))]
        share = lambda k: sum(x >= k for x in d) / len(d)
        out.append(f"{role:4} {len(v):7d}  {qq(.25):3d} {qq(.5):3d} {qq(.75):3d} {qq(.95):3d}   "
                   f"{share(10):6.1%} {share(25):6.1%} {share(30):6.1%}  {sum(x[1] for x in v):4d}  "
                   f"{sorted(x[2] for x in v)[len(v) // 2]:7d}")
    out.append(f"selected {a.role}-{a.race}: {len(games)} games with ttyrecs, "
               f"{sum(g['bytes'] for g in games) / 1e6:.0f} MB")
    text = "\n".join(out) + "\n"
    print(text)
    if a.summary:
        Path(a.summary).write_text(text, encoding="utf-8")


def game(g):
    import numpy as np
    from nle import _pyconverter as converter

    n = 4096
    chars = np.zeros((n, 24, 80), np.uint8)
    colors = np.zeros((n, 24, 80), np.int8)  # dtypes as nle.dataset.dataset
    curs = np.zeros((n, 2), np.int16)
    ts, acts, scores = np.zeros(n, np.int64), np.zeros(n, np.uint8), np.zeros(n, np.int32)
    conv = converter.Converter(24, 80, g.get("ttyrec_version", 1))  # alt.org 1, NLE's own (NLD-AA) 3
    arrivals, xls, owned, deepest, top_xl, frames, seen = [], [], {}, 0, 0, 0, 0
    last, rows_done = None, set()
    for part, path in enumerate(g["files"]):
        conv.load_ttyrec(path, gameid=0, part=part)
        while True:
            remaining = conv.convert(chars, colors, curs, ts, acts, scores)
            end = n - remaining
            c = chars[:end]
            dash = ((c[:, :, 1:-1] == 45) & (c[:, :, :-2] == 32) & (c[:, :, 2:] == 32)).any(axis=2)  # " - "
            inv_rows = {}
            for f, r in np.argwhere(dash):
                inv_rows.setdefault(f, []).append(r)
            for f in range(end):
                frames += 1
                bottom = chars[f, 22:24].tobytes()
                if bottom != last:
                    last = bottom
                    m = STATUS.search(chars[f, 23].tobytes()) or STATUS.search(chars[f, 22:24].tobytes())
                    if m:
                        seen += 1
                        dl, hp, maxhp, ac, xl = (int(x) for x in m.groups()[:5])
                        t = int(m.group(6)) if m.group(6) else -1
                        if dl > deepest:
                            deepest = dl
                            arrivals.append(dict(depth=dl, turn=t, xl=xl, ac=ac, maxhp=maxhp, hp=hp,
                                                 **{k: k in owned for k in ITEMS}))
                        if xl > top_xl:
                            top_xl = xl
                            xls.append((xl, dl, t))
                for r in inv_rows.get(f, ()):
                    s = chars[f, r].tobytes()
                    if s in rows_done:
                        continue
                    rows_done.add(s)
                    im = INV.search(s)
                    if im:
                        txt = im.group(1).decode("latin-1")
                        for k, p in ITEMS.items():
                            if k not in owned and re.search(p, txt, re.I):
                                owned[k] = deepest
            if remaining == 0:
                continue
            break
    return dict(g, files=len(g["files"]), frames=frames, status_frames=seen, arrivals=arrivals, xls=xls, owned=owned)


# the menu patterns run on rows without trailing blanks: a \s* over a blank row of WIDE columns is quadratic
# an item line of a menu, matched at the menu's column (where its class headers start): elsewhere on the screen
# " f - ||0" is a kitten by a wall
ITEM_AT = re.compile(rb"([a-zA-Z$]) - (.+)$")
PAGE = re.compile(rb"\((\d+) of (\d+)\)$")  # a menu's page marker
# an object prompt ("What do you want to use or apply? [h or ?*]"): its "?" list is an untitled menu of a few classes
# like 'i' ("Tools, Wands" taken for the whole inventory)
PROMPT = re.compile(rb"\?\s*\[[^\]]*\]$")
# an inventory menu: a class header on some row ('i' shows them; so do pickup and container menus, excluded); some
# players' headers carry the class symbol: "Weapons  (')')"
CATS = re.compile(rb"\b(Coins|Amulets|Weapons|Armor|Comestibles|Scrolls|Spellbooks|Potions|Rings|Wands|Tools|"
                  rb"Gems/Stones|Gems|Boulders/Statues|Iron balls|Chains)(?:\s+\('.'\))?$")
NOT_INV = re.compile(rb"Pick up what|Take out what|Put in what|Contents of|What would you like to|Drop what|"
                     rb"Pick an object|Loot|Things that are here|There are several objects here|What do you want to|"
                     rb"Unpaid |Do what\?|Both of the above")  # a shop bill ("Unpaid Scrolls"); a chest
PERIOD = 1000  # game turns between periodic states
EVENTS = False  # states --events: the message log of the whole game too (game_states)
# game_states' checks (from reading the states): the tty layout -- message row 0, map rows 1-21, status
# rows 22-23 -- is the only one parsed; other interfaces (curses windows, 3-line status) put status or text elsewhere
STATUS_TOP = re.compile(r"\bSt:\d")
# the class symbol some players' menus put before an item -- any sign: some remap the classes ("] elven cloak",
# ": 3 food rations", ", amulet of reflection")
OBJSYM = re.compile(r"^[^\w\s] ")
# "q - 64 rocks." in a sentence of the message row; the period: a frame may catch the row half-written ("+3 smal")
PICKUP = re.compile(r"(?:^|\s)([a-zA-Z$]) - (.+?)\.$")
SENTENCE = re.compile(r"(?<=[.!?])\s{2}")
DEC = re.compile(r"[lmjktuvwna~]q{2,}|q{2,}[lmjktuvwnx]|q{4,}|~{4,}")  # DECgraphics left as letters
# the virtual terminal: players with wide terminals draw menus right of column 80 (were cut at 80); some terminals
# are wider than 160 (a menu at column 137 was seen): a menu reaching the last column is dropped whole
WIDE = 256
TALL = 64  # likewise for taller terminals: some menus run past row 23 (status stays at rows 22-23 under the map)
MORE = re.compile(r"--More--|>>\s*$")  # some players' --More-- is ">>"
# a state carries the HP HP_AGO turns ago, the last MSG_BACK messages and the colours of
# the monsters on the map, as the bot's rows do (its traces). Not the last few different HP
# values: regeneration makes them +1 steps (probe: "40/85 before 37, 38, 39"); T is unknown in 0.2 % of states.
# What is the interface, not a message (ASK, UI_MSG) -- state_fmt, which drops it from the screen's current row too
HP_AGO = (10, 30)  # HP that many turns ago, as the bot's rows (build_v2.walk)
MON_CH = re.compile(r"[A-Za-z@&;:']")  # monster letters on the map, as state_fmt.surroundings takes them
ODD_SYMS = "||||"  # corridors drawn as | (a custom symset): mazes and mines give ||| at most
# some players' closed door is a brown '7' (sym_probe.py: it turns into a brown open door '-' / '|'
# on "The door opens.", and they have no brown '+'): drawn as the default '+'
DOOR7, BROWN = ord("7"), 3
# a class-choice menu ("What kinds of thing..."): its entries are class names, not items
CLASS_ITEM = re.compile(rb"(?:^|\s)[a-zA-Z] - (All types|Auto-select every item|Items known to be \w+|Unpaid items|"
                        rb"Unidentified items|Coins|Amulets|Weapons|Armor|Comestibles|Scrolls|Spellbooks|Potions|Rings|"
                        rb"Wands|Tools|Gems/Stones|Gems|Boulders/Statues|Iron balls|Chains)$")


PHRASE = re.compile(r"[A-Za-z]{2,}[,.]? [A-Za-z]{2,}[,.]? [A-Za-z]{2,}")


def map_text(row):
    """Text over a map row (a window or a message): three words in a row. Monsters side by side are no text:
    "a run of 4+ letters with 3+ different ones" took a fight with a crowd (dZeFFF, FpBe, qvoiZ) for a
    window, and the state waited for a later frame -- after the fight."""
    return bool(PHRASE.search(row))


def dec_walls(rows):
    """DECgraphics letters left in map rows: q runs (horizontal walls) or three x one above the other (vertical);
    a single x by a wall is a grid bug."""
    if any(DEC.search(r) for r in rows):
        return True
    for c in range(max(map(len, rows), default=0)):
        run = 0
        for r in rows:
            run = run + 1 if len(r) > c and r[c] == "x" else 0
            if run >= 3:
                return True
    return False


def map_ok(rows):
    """Rows 1-21 are a clean map: no status line, no text over it (map_text), no DECgraphics letters (dec_walls)."""
    return not (any(STATUS_TOP.search(r) or "Dlvl:" in r or map_text(r) for r in rows) or dec_walls(rows))


def item_text(raw, row):
    """An inventory line's item without the class symbol, or None when the screen edge cut it (its row reaches
    the last column, or column 79-80 of a player's 80-column terminal: tty cuts a menu line at 79, as in
    "gauntlets of p")."""
    if len(row.rstrip()) in (79, 80, WIDE):
        return None
    return OBJSYM.sub("", raw.strip()).rstrip(".") or None


def hp_at(hp_turns, turn):
    """[HP, max HP] at `turn` from (turn, HP, max HP) entries in turn order: the last at or before it; None when the
    turn is unknown or before the first entry (the game's start)"""
    last = None
    for t, hp, maxhp in hp_turns:
        if t > turn:
            break
        last = [hp, maxhp]
    return last if turn >= 0 else None


def game_states(g):
    """A human game as laya states: the screen (24 rows), the cursor (the hero, when on the map), the status and the
    inventory as last shown, at each first arrival at a new deepest Dlvl and every PERIOD turns. The game's fate
    (death, deepest level, turns) comes from the xlogfile record in g.
    Inventory: an inventory menu (class headers; not a pick-up / container / "what do you want to" menu, on any of
    its pages) is read while it is up, over all its pages, and committed when the game is back (the cursor on the
    hero) -- an untitled one with 2+ classes seen to its last page is the whole inventory ('i') and replaces it,
    others ('I', a titled menu, one left early) update its letters; "q - 64 rocks." sentences of the message row add
    pickups ("3+ classes = the whole inventory" would take each page of a 2-page 'i', and partial titled menus, for
    all of it). A state is taken only on the tty layout with a clean map (map_ok).
    The screen is rendered WIDE columns wide (menus of wide terminals); a game with a custom symset gives no states."""
    import numpy as np
    from nle import _pyconverter as converter

    n = 2048
    chars = np.zeros((n, TALL, WIDE), np.uint8)
    colors = np.zeros((n, TALL, WIDE), np.int8)
    curs = np.zeros((n, 2), np.int16)
    ts, acts, scores = np.zeros(n, np.int64), np.zeros(n, np.uint8), np.zeros(n, np.int32)
    conv = converter.Converter(TALL, WIDE, g.get("ttyrec_version", 1))
    # only frames within the game's xlogfile start-end (ttyrec times, microseconds): an account's games share files
    # at their ends -- a game's first file is often all the previous game's (ts_probe.py: one Valkyrie got a
    # Wizard's inventory from it); before start there are only the character-choice screens
    t_lo, t_hi = (g["start"] * 10**6, (g["end"] + 1) * 10**6) if g.get("start") and g.get("end") else (0, 2**62)
    states, inv, deepest, next_t, pending, last, stat = [], {}, 0, PERIOD, None, None, None
    listing, classes, in_menu, last_msg, cut = {}, set(), False, None, False
    page, titled, excluded = None, None, False
    hp_turns, msg_back, msg_cur = deque(), deque(maxlen=MSG_BACK), ""  # hp_turns: (turn, HP) of the last HP_AGO turns
    events, ev_key = [], None
    try:
        for part, path in enumerate(g["files"]):
            conv.load_ttyrec(path, gameid=0, part=part)
            while True:
                remaining = conv.convert(chars, colors, curs, ts, acts, scores)
                end = n - remaining
                c = chars[:end]
                dash = ((c[:, :, 1:-1] == 45) & (c[:, :, :-2] == 32) & (c[:, :, 2:] == 32)).any(axis=2)
                for f in range(end):
                    if not t_lo <= ts[f] <= t_hi:  # another game's frame (see t_lo)
                        continue
                    # the converter's cursor is (row, column), not (column, row): read the other way round, states
                    # would be taken only with the hero on the diagonal row == column
                    y, x = int(curs[f, 0]), int(curs[f, 1])
                    # back in the game: the cursor on the hero, no menu line on the screen
                    on_map = 1 <= y <= 21 and 0 <= x < 80 and chars[f, y, x] == 64 and not dash[f].any()
                    rs = np.flatnonzero(dash[f])
                    menu = False
                    if not on_map and PROMPT.search(chars[f, 0].tobytes().rstrip()):
                        excluded = True
                    if len(rs):
                        text = [chars[f, r].tobytes().rstrip() for r in range(TALL)]
                        if any(NOT_INV.search(t) or CLASS_ITEM.search(t) for t in text):
                            excluded = True  # for all its pages: page 2 of "Pick up what?" has no title
                        heads = [m for t in text if (m := CATS.search(t))]
                        menu = bool(heads) and not excluded
                        if menu:  # read the menu as it stands; later frames overwrite half-drawn lines
                            in_menu = True
                            cut = cut or any(len(t) >= WIDE - 1 for t in text) or bool(text[-1])
                            classes.update(m.group(1) for m in heads)
                            cols = {m.start(1) for m in heads}
                            pm = next((m for t in text if (m := PAGE.search(t))), None)
                            if pm:
                                page = (int(pm.group(1)), int(pm.group(2)))
                            if titled is None and (not pm or page[0] == 1):  # 'i' starts with a class header
                                col0 = min(cols)
                                top = next((t[col0:].strip() for t in text if t[col0:].strip()), b"")
                                titled = not CATS.search(top)
                            for r in rs:
                                for col in cols:
                                    im = ITEM_AT.match(text[r], col)
                                    if im:
                                        it = item_text(im.group(2).decode("latin-1"), text[r].decode("latin-1"))
                                        if it:
                                            listing[im.group(1).decode()] = it
                                        break
                    if g.get("debug") == 2 and (len(rs) or in_menu):
                        print(f"  frame {f}: menu={menu} excluded={excluded} on_map={on_map} page={page} "
                              f"cursor=({x},{y}) {chr(chars[f, y, x]) if 0 <= y < 24 and 0 <= x < WIDE else ''!r}")
                    if in_menu and on_map:  # the menu is over (all its pages): commit it
                        full = not titled and len(classes) >= 2 and (page is None or page[0] == page[1])
                        if g.get("debug"):  # inv_trace.py
                            print(f"{path.rsplit('/', 1)[-1]} frame {f} T{stat and stat['turn']}: menu "
                                  f"{sorted(classes)} titled={titled} page={page} cut={cut} full={full}: "
                                  f"{sorted(listing)}")
                        if cut:  # the screen edge cut it: nothing
                            pass
                        elif full:
                            inv = dict(listing)
                        else:
                            inv.update(listing)
                        listing, classes, in_menu, cut, page, titled = {}, set(), False, False, None, None
                    if on_map:
                        excluded = False
                    # the message row: a frame may catch it half-written, so a message is kept when the row stops growing
                    # from it (replaced or cleared). Only with the cursor on the hero or the message row: while it is
                    # elsewhere on the map the player picks a spot (travel, ';') and 3.6's autodescribe writes what is
                    # under it there ("floor of a room", "tame housecat called <name>")
                    if not (menu or in_menu or excluded) and (on_map or y == 0):
                        row0 = MORE.sub("", chars[f, 0].tobytes().decode("latin-1")).strip()
                        if row0 != msg_cur:
                            if not (msg_cur and row0.startswith(msg_cur)):
                                if msg_cur and not ASK.search(msg_cur) and not UI_MSG.search(msg_cur) \
                                        and (not msg_back or msg_back[-1] != msg_cur):
                                    msg_back.append(msg_cur[:160])
                                # --events: every message, prompts with their answers too ("What do you want to
                                # drink? [fgh or ?*] g" -- the action; NAO's ttyrecs have no keys)
                                if EVENTS and msg_cur and (not events or events[-1][-1] != msg_cur[:160]):
                                    events.append(event(stat, msg_cur[:160]))
                            msg_cur = row0
                    if not (menu or in_menu or excluded) and dash[f, 0]:  # pickups on the message row, once each
                        msg = chars[f, 0].tobytes().decode("latin-1")
                        if msg != last_msg:
                            last_msg = msg
                            if len(msg.rstrip()) < WIDE:  # a full row may go on on the next one
                                for sentence in SENTENCE.split(MORE.sub("", msg.rstrip()).strip()):
                                    pm = PICKUP.search(sentence)
                                    if pm:
                                        inv[pm.group(1)] = OBJSYM.sub("", pm.group(2).strip())
                    bottom = chars[f, 22:24].tobytes()
                    # the status only once it is drawn: with the cursor still on rows 22-23 the frame may end
                    # mid-redraw -- "T:6" for T:6603, T gone (turn -1 mid-game), or old digits left after new ones
                    # (T:5681116 in a 128 445-turn game; junk_scan.py: 4.8 % / 11.5 % of the games' event logs).
                    # "Turns never go back" would make one such high reading stick to the game's end.
                    if bottom != last and not 22 <= y <= 23:
                        last = bottom
                        m = STATUS.search(chars[f, 23].tobytes()) or STATUS.search(bottom)
                        if m:
                            dl, hp, maxhp, ac, xl = (int(x) for x in m.groups()[:5])
                            t = int(m.group(6)) if m.group(6) else -1
                            # the cursor is not always left on the status ("T:2" for T:2686 with it on
                            # the message row): a turn that is the start of the last one's digits, or no T at all
                            # after one was seen, is the cut field -- the last turn stands (a real reading is kept as
                            # it is, so a stray high one does not stick)
                            if stat and stat["turn"] >= 0 and (
                                    t < 0 or (t < stat["turn"] and str(stat["turn"]).startswith(str(t)))):
                                t = stat["turn"]
                            if dl > deepest:
                                deepest = dl
                                pending = "arrival"
                            elif t >= next_t:
                                pending = "period"
                            if t >= next_t:
                                next_t = (t // PERIOD + 1) * PERIOD
                            stat = dict(dlvl=dl, hp=hp, maxhp=maxhp, ac=ac, xl=xl, turn=t, deepest=deepest)
                            if EVENTS and (dl, ac, xl, maxhp) != ev_key:  # a level, AC, XL or max HP change
                                ev_key = (dl, ac, xl, maxhp)
                                events.append(event(stat, ""))
                            if t >= 0:  # keep one entry at or before t - max(HP_AGO): the HP then
                                hp_turns.append((t, hp, maxhp))
                                while len(hp_turns) > 1 and hp_turns[1][0] <= t - max(HP_AGO):
                                    hp_turns.popleft()
                    if pending and stat and on_map:  # the state once the hero is on the map and no menu covers it
                        scr = chars[f, :24].copy()
                        scr[1:22, :80][(scr[1:22, :80] == DOOR7) & (colors[f, 1:22, :80] == BROWN)] = ord("+")
                        rows = [scr[r, :80 if 1 <= r <= 21 else WIDE].tobytes().decode("latin-1").rstrip()
                                for r in range(24)]  # the map is 80 columns; message and status may be wider
                        if STATUS_TOP.search(rows[22]) and STATUS.search(chars[f, 23].tobytes()) \
                                and map_ok(rows[1:22]):  # else wait for a frame that is clean
                            mcol = {f"{c},{r}": int(colors[f, r, c]) for r in range(1, 22)
                                    for c, ch in enumerate(rows[r]) if MON_CH.match(ch) and (c, r) != (x, y)}
                            states.append(dict(stat, kind=pending, x=x, y=y, screen=rows,
                                               inv=[f"{k} - {v}" for k, v in sorted(inv.items())],
                                               hp_ago=[hp_at(hp_turns, stat["turn"] - d) for d in HP_AGO],
                                               msgs=list(msg_back), mcol=mcol))
                            pending = None
                if remaining == 0:
                    continue
                break
    except Exception as e:  # a broken ttyrec: keep what was read
        return dict(g, files=len(g["files"]), states=odd_syms(states), error=repr(e)[:200],
                    **({"events": events} if EVENTS else {}))
    return dict(g, files=len(g["files"]), states=odd_syms(states), **({"events": events} if EVENTS else {}))


def event(stat, msg):
    """--events: [turn, Dlvl, HP, max HP, AC, XL, message] (status as last read; -1 / -99 before the first)."""
    s = stat or {}
    return [s.get("turn", -1), s.get("dlvl", -1), s.get("hp", -1), s.get("maxhp", -1), s.get("ac", -99),
            s.get("xl", -1), msg]


def odd_syms(states):
    """No states of a game drawn with a custom symset: corridors as |, or bytes >= 0x80 on the map -- 3.6's UTF-8 box
    walls (10 of 2000 Val-Hum games), whose 3 bytes a glyph the converter puts in 3 cells, so the map's columns shift;
    or a '7' left after the brown-door fix (DOOR7): a few players have it at doors without colour, unverified (4 of
    2000 games). Its maps would mislead."""
    return [] if any(ODD_SYMS in r or "7" in r or max(r, default=" ") >= "\x80"
                     for s in states for r in s["screen"][1:22]) else states


def states(a):
    """Every listed game's states (game_states) into one jsonl, a game per line, written as each game ends;
    rerunning skips the games already there. Soft stop: the file <out>.stop (or --until, epoch seconds -- but the
    clock stands still under faketime) -- no new games, the started ones are finished and written; with
    <out>.stop the file is removed and the exit code is 3 (a chain under `set -e` stops there, not on half data)."""
    import queue
    import time
    global EVENTS
    EVENTS = a.events  # before the pool forks
    games = json.loads(Path(a.games).read_text())
    random.Random(0).shuffle(games)  # any prefix is a random sample
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                g = json.loads(line)
                done.add((g["name"], g["start"]))
            except (ValueError, KeyError):  # a line cut by an interruption
                pass
    todo = [g for g in games if (g["name"], g["start"]) not in done]
    if a.limit:
        todo = todo[:a.limit]
    stop = Path(f"{out}.stop")
    if stop.exists():
        print(f"stale {stop} removed", flush=True)
        stop.unlink()
    print(f"{len(done)} games done, {len(todo)} to go", flush=True)
    # games are handed out one by one (a spare per worker), so a stop leaves no long queue to finish
    results, pending, sent = queue.Queue(), 0, 0
    t0, n, nstates, stopping = time.monotonic(), 0, 0, ""  # monotonic: real even under faketime
    # --events: the message logs go to <out>.events.jsonl ({name, start, events} a line; ~110 KB a game -- the states
    # file and its readers stay as they were), written before the game's states line (a rerun may repeat a log)
    evh = open(f"{out}.events.jsonl", "a", encoding="utf-8") if a.events else None
    with Pool(a.workers) as pool, out.open("a", encoding="utf-8") as fh:
        def send():
            nonlocal pending, sent
            if sent < len(todo):
                pool.apply_async(game_states, (todo[sent],), callback=results.put, error_callback=results.put)
                pending, sent = pending + 1, sent + 1
        for _ in range(2 * a.workers):
            send()
        while pending:
            g = results.get()
            pending -= 1
            if isinstance(g, BaseException):
                raise g
            if evh:
                evh.write(json.dumps({"name": g["name"], "start": g["start"], "events": g.pop("events", [])}) + "\n")
                evh.flush()
            fh.write(json.dumps(g) + "\n")
            fh.flush()
            n += 1
            nstates += len(g["states"])
            if n % 500 == 0:
                print(f"{n} games, {nstates} states, "
                      f"{(time.monotonic() - t0) / n * a.workers:.1f} s/game/worker", flush=True)
            if not stopping:
                stopping = "stop file" if stop.exists() else "deadline" if a.until and time.time() > a.until else ""
                if stopping:
                    print(f"{stopping}: no new games, finishing {pending}", flush=True)
            if not stopping:
                send()
    print(f"END {n} games, {nstates} states", flush=True)
    if stopping == "stop file":
        stop.unlink(missing_ok=True)
        if sent < len(todo):
            print(f"stopped: {len(todo) - sent} games left (rerun to continue)", flush=True)
            sys.exit(3)


def profile(a):
    """--raw: only read the games (needs nle) and dump them as json (for a container with nle but no pandas);
    --load: the tables from such a dump (any python with pandas)."""
    if a.load:
        games = done = json.loads(Path(a.load).read_text())
    else:
        games = json.loads(Path(a.games).read_text())
        if a.limit:
            games = games[:a.limit]
        if a.root:  # a copy of the user dirs, e.g. on Windows where ':' in the file names became '_'
            for g in games:
                g["files"] = [f"{a.root}/{Path(p).parent.name}/{Path(p).name.replace(':', '_')}" for p in g["files"]]
        with Pool(a.workers) as pool:
            done = [g for g in pool.imap_unordered(game, games) if g["arrivals"]]
        if a.raw:
            Path(a.raw).write_text(json.dumps(done))
            print(f"{len(done)} of {len(games)} games with status lines -> {a.raw}")
            return
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bot_profile import q
    rows = [dict(s, game=i, version=g["version"][:3], maxdepth=g["arrivals"][-1]["depth"], maxlvl=g["maxlvl"],
                 death=g.get("death", "ascended"), player=g["name"].lower(), start=g["start"])
            for i, g in enumerate(done) for s in g["arrivals"]]
    A = pd.DataFrame(rows)
    items = list(ITEMS)
    out = [f"corridor: {len(done)} of {len(games)} {a.games} games with status lines "
           f"(3.4.3: {sum(g['version'].startswith('3.4') for g in done)}, 3.6: "
           f"{sum(g['version'].startswith('3.6') for g in done)}); turns {q(pd.Series([g['turns'] for g in done]))}",
           "", "first arrival at each depth (median [p25–p75]; items: % of games that had shown them by then)",
           f"{'Dlvl':>4} {'games':>5}  {'Xp':12} {'AC':12} {'max HP':14} {'turn':18} "
           + " ".join(f"{k[:6]:>6}" for k in items)]
    for d, s in A.groupby("depth"):
        if len(s) >= 10:
            t = s.turn[s.turn >= 0]
            out.append(f"{d:4d} {len(s):5d}  {q(s.xl):12} {q(s.ac):12} {q(s.maxhp):14} "
                       f"{q(t) if len(t) else '—':18} " + " ".join(f"{s[k].mean():6.0%}" for k in items))
    out += ["", "first reaching each experience level: Dlvl then (median [p25–p75])", f"{'Xp':>3} {'games':>5}  Dlvl"]
    for x in range(1, 31):
        at = pd.Series([d for g in done for xl, d, _ in g["xls"] if xl == x])
        if len(at) >= 10:
            out.append(f"{x:3d} {len(at):5d}  {q(at)}")
    out += ["", "items ever shown in an inventory line (% of games; Dlvl when first shown, median)"]
    for k in items:
        at = pd.Series([g["owned"][k] for g in done if k in g["owned"]], dtype=float)
        out.append(f"{k:14s} {len(at) / len(done):5.0%}  Dlvl {at.median() if len(at) else float('nan'):.0f}")
    text = "\n".join(out) + "\n"
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".txt").write_text(text, encoding="utf-8")
    A.to_parquet(a.out + ".parquet", index=False)


def compare(a):
    """At the first arrival at Dlvl d: ascenders, players who died within two levels below d, players who died
    deeper, and the bot (bot_profile.py parquet), side by side."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bot_profile import q

    asc, died = pd.read_parquet(a.asc), pd.read_parquet(a.died)
    bot = pd.read_parquet(a.bot) if a.bot else None
    items = list(ITEMS)
    out = [f"first arrival at Dlvl d: humans who ascended / died within 2 levels below d / died deeper; the bot "
           f"(median [p25–p75]; items: % of games that had shown them by then). {a.asc}, {a.died}"
           + (f", {a.bot}" if a.bot else ""),
           f"{'Dlvl':>4} {'group':11} {'n':>5}  {'Xp':12} {'AC':12} {'max HP':14} {'turn':18} "
           + " ".join(f"{k[:6]:>6}" for k in items)]
    for d in a.depths:
        groups = [("ascended", asc[asc.depth == d])]
        s = died[died.depth == d]
        groups += [("died ≤ d+2", s[s.maxdepth <= d + 2]), ("died deeper", s[s.maxdepth > d + 2])]
        if bot is not None:
            groups.append(("bot", bot[bot.depth == d]))
        for name, g in groups:
            if len(g) < 5:
                continue
            t = g.turn[g.turn >= 0]
            out.append(f"{d:4d} {name:11} {len(g):5d}  {q(g.xl):12} {q(g.ac):12} {q(g.maxhp):14} "
                       f"{q(t) if len(t) else '—':18} " + " ".join(f"{g[k].mean():6.0%}" for k in items))
        out.append("")
    text = "\n".join(out)
    print(text)
    Path(a.out).write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--nld", required=True)
    p.add_argument("--role", default="Val")
    p.add_argument("--race", default="Dwa")
    p.add_argument("--outcome", choices=["ascended", "died", "any"], default="ascended",
                   help="died: any end but ascension, quitting or escaping; any: died or ascended")
    p.add_argument("--min-maxlvl", type=int, default=0, help="only games that reached this Dlvl")
    p.add_argument("--max-maxlvl", type=int, default=0,
                   help="only games that never went deeper than this Dlvl (the games --min-maxlvl 4 cut)")
    p.add_argument("--sample", type=int, default=0, help="at most this many games (a fixed random sample)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--games", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--root", default="", help="user dirs elsewhere than the listed paths (a copy)")
    p.add_argument("--raw", default="", help="only read the games and dump them here as json (no pandas needed)")
    p.add_argument("--load", default="", help="the tables from a --raw dump instead of reading the games")
    p = sub.add_parser("list-aa")
    p.add_argument("--nld-aa", required=True)
    p.add_argument("--role", default="Val")
    p.add_argument("--race", default="Dwa")
    p.add_argument("--out", required=True)
    p.add_argument("--summary", default="", help="also write the per-role summary here")
    p = sub.add_parser("states")
    p.add_argument("--games", required=True)
    p.add_argument("--out", required=True, help="jsonl, a game per line (appended; rerun to continue)")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--until", type=float, default=0, help="epoch seconds: take no new games after it")
    p.add_argument("--events", action="store_true",
                   help="also the game's whole message log with the status (game_states, event) -> <out>.events.jsonl")
    p = sub.add_parser("compare")
    p.add_argument("--asc", required=True, help="profile parquet of the ascensions")
    p.add_argument("--died", required=True, help="profile parquet of the deaths")
    p.add_argument("--bot", default="", help="bot_profile.py parquet")
    p.add_argument("--depths", type=int, nargs="+", default=[5, 8, 10, 12, 15, 18, 20, 22, 24, 25, 26, 27, 28])
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"list": list_games, "list-aa": list_aa, "profile": profile, "states": states, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
