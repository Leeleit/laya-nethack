"""The screen parts of a state text, shared by the human states (human_build.py) and the bot's (build_v2.py), so the
unified model reads one layout. Tokens saved where the text repeated itself (window_probe.py):
  - the status lines without what the state's first line already gives (HP, Dlvl, XL, AC, turn), without the
    player's name and rank title (noise, and a handle for memorising players; the rank only restates XL) and without
    the score (only some players show it; cut in the bot's rows);
  - the maps without their common left indent and without blank rows above and below (positions within the map
    are kept: blank rows inside it stay);
  - no separate crop around the hero when the whole level map fits a crop-sized box (it was the same picture twice);
  - only the map around the hero, no whole level map (hero_map).
"""
import re

CROP_DX, CROP_DY = 12, 5  # the crop around the hero: 25 columns x 11 rows
SAME_AS_LEVEL = "the whole level map below"  # the crop's place in rows of the older format (map_variants.py)
_REPEATED = re.compile(r"\s*\b(Dlvl:\d+|HP:-?\d+\(\d+\)|AC:-?\d+|Xp:\d+(/\d+)?|Exp:\d+|T:\d+|S:\d+)")
# the score ("showscore"): some players show it, and the bot's 80-column row cuts it ("Lawful S:4", "Neutral S:")
_SCORE = re.compile(r"\s+S(:\d*)?$")
# the bottom line's conditions (botl.c 3.4.3 / 3.6), to complete the last one when the 80th column cut it
_CONDITIONS = ["Satiated", "Hungry", "Weak", "Fainting", "Fainted", "Burdened", "Stressed", "Strained", "Overtaxed",
               "Overloaded", "Conf", "FoodPois", "Ill", "Blind", "Stun", "Hallu", "Slime", "Stone", "Strngl", "Deaf",
               "Lev", "Fly", "Ride"]


def status(scr):
    """'St:18/04 Dx:10 ... Lawful | $:211 Pw:20(20) Hungry' from tty rows 22-23."""
    top = scr[22] if len(scr) > 22 else ""
    bottom = scr[23] if len(scr) > 23 else ""
    i = top.find("St:")
    top = _SCORE.sub("", (top[i:] if i >= 0 else top).rstrip())
    words = _REPEATED.sub("", bottom).split()
    if len(bottom.rstrip()) >= 79 and words and words[-1] not in _CONDITIONS:  # tty cut it: "Satiated Burdene"
        fits = [c for c in _CONDITIONS if c.startswith(words[-1])]
        if len(fits) == 1:  # "S" could be several: then left cut
            words[-1] = fits[0]
    return f"{' '.join(top.split())} | {' '.join(words)}"


def _trim(rows):
    """Rows without blank rows above and below and without their common left indent."""
    rows = [r.rstrip() for r in rows]
    while rows and not rows[0]:
        rows.pop(0)
    while rows and not rows[-1]:
        rows.pop()
    indent = min((len(r) - len(r.lstrip(" ")) for r in rows if r), default=0)
    return "\n".join(r[indent:] for r in rows)


# the map in words: the level's notable features and the monsters in view, each with the
# number of moves to the nearest one along the known map, no number when the known map has no path to it
FEATURES = {">": "down stairs", "<": "up stairs", "{": "fountain", "_": "altar", "\\": "throne", "^": "trap",
            "}": "water or lava"}
_WALK = set(".#<>{_\\^+") | set(")[%?!/=*(`\"0$,:;&'~@") | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DIRS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def _moves(grid, x, y):
    """Moves from (x, y) to every cell reachable over the known map: floor, corridors, features, objects and
    monsters are walkable; walls, water and the unknown are not; doors (closed '+', open: '|' between cells above and
    below, '-' between cells left and right) are entered and left only orthogonally, as in NetHack."""
    h = len(grid)
    at = lambda c, r: grid[r][c] if 0 <= r < h and 0 <= c < len(grid[r]) else " "
    def door(c, r):
        ch = at(c, r)
        return ch == "+" or (ch == "|" and at(c, r - 1) in _WALK and at(c, r + 1) in _WALK) or \
            (ch == "-" and at(c - 1, r) in _WALK and at(c + 1, r) in _WALK)
    dist, todo = {(x, y): 0}, [(x, y)]
    for c, r in todo:  # breadth first: the list grows while it is walked
        for dx, dy in _DIRS:
            n = (c + dx, r + dy)
            if n in dist or not (at(*n) in _WALK or door(*n)):
                continue
            if dx and dy and (door(c, r) or door(*n)):
                continue
            dist[n] = dist[(c, r)] + 1
            todo.append(n)
    return dist


_GLYPHS = None
PET = 16  # the converter's colour of an inverse cell: tty draws pets so (hilite_pet; 3.6 also detected monsters)


def mon_name(ch, colour):
    """A map letter as a monster's name (mon_glyphs.py): the monster when its (letter, colour) is one
    monster's, else its class ('ant or other insect'); 'little dog (pet)' when tty drew it inverse."""
    global _GLYPHS
    if _GLYPHS is None:
        from nle import nethack
        _GLYPHS = {}
        for i in range(nethack.NUMMONS):
            p = nethack.permonst(i)
            c = nethack.class_sym.from_mlet(p.mlet)
            _GLYPHS.setdefault((c.sym, p.mcolor), []).append(p.mname)
            _GLYPHS.setdefault(c.sym, c.explain)
    if ch == "I":  # where an unseen monster was noticed: no class
        return "unseen monster"
    names = _GLYPHS.get((ch, colour & 15), [])
    name = names[0] if len(names) == 1 else _GLYPHS.get(ch, ch)
    return f"{name} (pet)" if colour & PET else name


def surroundings(scr, x, y, names=None):
    """'down stairs 12; fountain ? x2. Monsters: G 2; F 5' from tty rows 1-21 (x, y: the hero's column and tty row):
    moves to the nearest, '?' without a known path, 'x2' when several are seen; '' when there is nothing. Short words:
    the long form ('Features: ...', 'Monsters in view: ...', '(no known path)', '2 seen', 'moves') cost 23 tokens a
    state, this one 14 (around_tokens.py). names: {(column, tty row): monster name} -- the bot knows them,
    the humans' from (letter, colour) by mon_name -- in place of the letters."""
    names = names or {}
    grid = [scr[r] if r < len(scr) else "" for r in range(1, 22)]
    dist = _moves(grid, x, y - 1)
    feats, mons = {}, []
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if (c, r) == (x, y - 1):
                continue
            d = dist.get((c, r))
            if ch in FEATURES:
                feats.setdefault(FEATURES[ch], []).append(d)
            elif ch.isalpha() or ch in "@&;:'":
                # 'I': where an unseen monster was noticed
                mons.append((d if d is not None else 999, names.get((c, r + 1), "unseen monster" if ch == "I" else ch)))
    def say(name, ds):
        known = [d for d in ds if d is not None]
        return f"{name} {min(known) if known else '?'}{f' x{len(ds)}' if len(ds) > 1 else ''}"
    f = "; ".join(say(n, ds) for n, ds in sorted(feats.items(), key=lambda kv: min(d for d in kv[1] if d is not None)
                                                    if any(d is not None for d in kv[1]) else 999))
    m = "; ".join(f"{ch} {d if d < 999 else '?'}" for d, ch in sorted(mons)[:8])
    return ". ".join(p for p in (f, m and f"Monsters: {m}") if p)


# the short layout both sources share (human_build.py --terse, build_v2.py --terse)
ALIGN_WORD = re.compile(r" (Lawful|Neutral|Chaotic|Unaligned)\b")
# the game's hello: "Velkommen player, welcome to NetHack!  You are a lawful dwarven Valkyrie." / "..., the dwarven
# Valkyrie, welcome back to NetHack!" (the bot is "Agent"); what follows it ("Be careful!  New moon tonight.") stays
GREETING = re.compile(r"^\w+ [\w-]+, (?:welcome to NetHack!\s+You are an? [\w -]+?\.|the [\w -]+?, welcome back to "
                      r"NetHack!)\s*")
LETTER = re.compile(r"^([a-zA-Z$] - )?(.*)$")
# an item's words that cost tokens and tell nothing (inv_tokens.py): the articles, "(being worn)",
# "(weapon in hand)"; the inventory letters stay
_SHORT = [(re.compile(r"^an? "), ""), (re.compile(r"\(being worn\)"), "(worn)"),
          (re.compile(r"\(weapon in hands?\)"), "(wielded)")]
MSG_BACK = 6
ASK = re.compile(r"\?\s*(\[[^\]]*\](\s*\(\w\))?)?\s*\S?$")  # a prompt with the player's answer echoed: not a message
# the interface on the message row, not the game (probe of 200 games, states3_look.py): extended-command echo, menus'
# titles, counts, farlook answers, character choice; build_v2.BOT_UI drops the bot's own the same way. Also
# (msg_junk.py: 1.1 % of a dataset's messages without these, mostly the screen's current row): typed-in answers
# ("Call an emerald potion: acid pos", "What do you want to call the dog? Skitters"), "[HP+1=56]", direction errors
UI_MSG = re.compile(r"^#|^Unknown command|^Pick an object|^Things that are here:|^Count: \d|^Choosing |"
                    r"^Message History|^Where do you want to |^Never mind\.$|^You see no objects here\.$|"
                    r"^Autopickup: |^\S\s{4,}\S|"
                    r"^Call an? .*:|^What do you want to (call|name|write|engrave|add)|^Specify what\?|"
                    r"^Talk to whom\?|^In what direction\?|^Really |^Unknown direction|^Done\.$|^\[HP[+-]|"
                    # autodescribe left with the cursor over the hero (0.06 % of messages in the probe after the
                    # cursor rule): bare terrain names
                    r"^(floor of a room|dark part of a room|corridor|lit corridor|doorway|open door|broken door|wall|"
                    r"stone|staircase (up|down)|ladder (up|down)|fountain|altar|sink|grave|tree|water|lava|ice|air|"
                    r"cloud|throne|iron bars|drawbridge|moat)$")


def compact_inv(inv):
    """'a - a +1 long sword (weapon in hand); c - an uncursed +3 small shield (being worn)' -> 'a - +1 long sword
    (wielded); c - uncursed +3 small shield (worn)'; letters, counts, BUC, enchantment, names stay."""
    if inv == "unknown":
        return inv
    out = []
    for it in inv.split("; "):
        letter, it = LETTER.match(it).groups()
        for pat, rep in _SHORT:
            it = pat.sub(rep, it)
        out.append((letter or "") + it)
    return "; ".join(out)


def messages(msgs):
    """The last MSG_BACK messages, oldest first, without the game's hello, the interface (ASK, UI_MSG) and repeats."""
    out = []
    for m in msgs:
        m = GREETING.sub("", m).strip()
        if m and not ASK.search(m) and not UI_MSG.search(m) and (not out or out[-1] != m):
            out.append(m)
    return out[-MSG_BACK:]


def terse_state(character, version, s, hp_ago, scr, msgs, around, inv, bot=None):
    """'Val-Dwa-Law-Fem 3.4.3 / HP 31/46 (10 turns ago 40/46, 30 ago 46/46) Dlvl 5 deepest 5 XL 7 AC -5 turn 2000 /
    status / Bot: ... / messages / Around: ... / Inventory: ...'. s: hp, maxhp, dlvl, deepest, xl, ac, turn (-1:
    unknown); hp_ago: [HP, max HP] or None 10 and 30 turns ago."""
    ago = ", ".join(f"{w} {a[0]}/{a[1]}" for w, a in zip(("10 turns ago", "30 ago"), hp_ago or ()) if a)
    ago = f" ({ago})" if ago else ""
    turn = s["turn"] if s["turn"] >= 0 else "unknown"
    lines = [f"{character} {version}",
             f"HP {s['hp']}/{s['maxhp']}{ago} Dlvl {s['dlvl']} "
             f"deepest {s['deepest']} XL {s['xl']} AC {s['ac']} turn {turn}",
             ALIGN_WORD.sub("", status(scr)).replace(" | ", " ")]  # the alignment is in the first line already
    if bot:
        lines.append(f"Bot: {bot}")
    msgs = messages(msgs)
    if msgs:
        lines.append(" | ".join(msgs))
    if around:
        lines.append(f"Around: {around}")
    lines.append(f"Inventory: {inv}")
    return "\n".join(lines)


def hero_map(scr, x, y):
    """The map around the hero from tty rows 1-21 (x, y: the hero's column and tty row): the whole level map when it
    fits a crop-sized box, else the crop. No whole level map beside it: it told the risk model
    next to nothing (map_ablation.py) and was ~20 % of the humans' tokens."""
    level_rows = [scr[r] if r < len(scr) else "" for r in range(1, 22)]
    used = [r for r, row in enumerate(level_rows) if row.strip()]
    if not used or (used[-1] - used[0] <= 2 * CROP_DY and
                    max(len(level_rows[r].rstrip()) for r in used)
                    - min(len(level_rows[r]) - len(level_rows[r].lstrip(" ")) for r in used) <= 2 * CROP_DX + 1):
        return _trim(level_rows)
    lo, hi = max(1, y - CROP_DY), min(22, y + CROP_DY + 1)
    return _trim([(scr[r] if r < len(scr) else "").ljust(x + CROP_DX + 1)[max(0, x - CROP_DX):x + CROP_DX + 1]
                  for r in range(lo, hi)])
