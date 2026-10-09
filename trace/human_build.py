"""laya's dataset from human games (NLD-NAO): the states of nao_corridor.py states (screen, status, inventory last
shown, recent messages; at each new deepest Dlvl and every 1000 turns) with the game's outcome from its xlogfile
record. The reference is the humans, not the bot: what a player in this state goes on to do.

Questions: with --v6 those of human_q6.py (ttd: turns left, depth: the deepest Dlvl reached, end: what ends the game,
ascends), otherwise the first two ordinal ones:
  fate -- how much deeper than the deepest level so far the game gets: dies there or one level deeper / 2-5 deeper
          / 6+ deeper / ascends
  ttd  -- turns left: dies within 1000 / in 1000-5000 / later or ascends (only states with a turn counter)
--map, --compact and --terse choose the wording of the state (state_fmt.py).
Split by player (crc32 of the name): test 1 in 20, with --val val 1 in 20, the rest train -- no player in two splits.
  python human_build.py --states nao/val_dwa_states.jsonl [more.jsonl ...] --out human_data [--v6] [--per-band]
      [--per-game K] [--held-per-player K] [--val] [--exclude F ...] [--exclude-players F ...] [--workers N]
-> <out>/{train,val,test}.jsonl, meta.json
"""
import argparse
import bisect
import collections
import json
import random
import re
import zlib
from multiprocessing import Pool
from pathlib import Path

from human_q6 import Builder, outcome
from state_fmt import GREETING, compact_inv, hero_map, mon_name, status, surroundings, terse_state

MAP = "grid"  # --map: the map around the hero as a grid, or "text" -- features and monsters with moves to them
COMPACT = False  # --compact: the inventory without what costs tokens and tells nothing (state_fmt.compact_inv)
TERSE = False  # --terse: the state's head lines and v6's questions in fewer words (state_fmt.terse_state, Builder)

FATE = ["dies on its deepest level so far or one deeper", "dies 2 to 5 levels deeper", "dies 6 or more levels deeper",
        "ascends"]
TTD = ["dies within 1000 turns", "dies in 1000 to 5000 turns", "lives more than 5000 turns or ascends"]
Q_FATE = {"type": "score", "criteria": FATE,
          "instructions": "A human plays this game. How far does the game get from here?"}
Q_TTD = {"type": "score", "criteria": TTD, "instructions": "A human plays this game. How long until the hero dies?"}


def gold(k, n):
    return {"probabilities": {str(i): float(i == k) for i in range(n)}, "label": str(k)}


GLUED = re.compile(r"\.  (?=[a-zA-Z$] - )")  # "n - an egg.  o - a yellow gem": a pickup message stuck to an item
ITEM = re.compile(r"^([a-zA-Z$]) - (?:[^\w\s] )?(.+)$")  # with the class symbol some players' menus show (any sign)
ALIGN = re.compile(r"\b(Lawful|Neutral|Chaotic)\b")  # the alignment on the status line at the state's time
CUT = re.compile(r"(\b(an?|the|\d+)( (unc|uncurs|uncurse|uncursed|blessed|cursed|[+-]\d+))?|\b(unc|uncurs|uncurse|of|o|"
                 r"labeled|called|named))$")


def clean_inv(raw):
    """The inventory as last shown, without the screen-scraping debris found by reading states
    (data_checks.py: ~25% of the states had a game message inside an item, ~29% several items glued into one, ~4% items
    cut off, ~1-21% class symbols): items split where pickup messages are glued on, a game message after an item's
    ".  " dropped, class symbols dropped, cut-off items dropped (only those: blanking a listing with a few cut lines
    threw away whole late-game inventories), one item per letter (the later, newer one), in inventory order."""
    items = {}
    for part in (p for it in raw for p in GLUED.split(it)):
        m = ITEM.match(part.split(".  ")[0].strip())
        if not m:
            continue
        letter, what = m.group(1), m.group(2).rstrip(".")
        if letter == "$":  # gold as last shown is stale ("$ - 16" by $:161): the status has it
            continue
        if what.count("(") > what.count(")"):  # "(on left han": the line ran off the screen
            what = what[:what.rfind("(")].rstrip()
        if CUT.search(what):
            continue
        items[letter] = f"{letter} - {what}"
    if not items:
        return "unknown"
    order = "$abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "; ".join(items[k] for k in sorted(items, key=order.index))


def state_text(g, s):
    """risk-v2's fields where a ttyrec has them: status first, the map around the hero, the inventory. Status and
    map: state_fmt.py (no repeats, no player name, no whole level map)."""
    scr = s["screen"]
    # the monsters by name (newer extractions carry their colours: "mcol", "col,row": colour)
    names = {tuple(map(int, k.split(","))): mon_name(scr[int(k.split(",")[1])][int(k.split(",")[0])], c)
             for k, c in s.get("mcol", {}).items()}
    if MAP == "text":
        words = surroundings(scr, s["x"], s["y"], names)
        around = f"Around the hero: {words}\n" if words else ""
    else:
        around = f"Map around the hero (@):\n{hero_map(scr, s['x'], s['y'])}\n"
    turn = s["turn"] if s["turn"] >= 0 else "unknown"
    character = g["character"]
    am = ALIGN.search(scr[22] if len(scr) > 22 else "")
    if am:  # the xlogfile's alignment is the one at the end: some players convert (xlogfile Cha, the status Lawful)
        role, race, _, gender = character.split("-")
        character = f"{role}-{race}-{am.group(1)[:3]}-{gender}"
    msgs = s.get("msgs", []) + [scr[0].strip()]  # the earlier messages (newer extractions), then the state's own
    if g.get("name"):  # "Velkommen <name>, the human Valkyrie": the name is noise and a handle on the player
        msgs = [re.sub(rf"(?<!\w){re.escape(g['name'])}(?!\w)", "player", m, flags=re.I) for m in msgs]
    inv = compact_inv(clean_inv(s["inv"])) if COMPACT else clean_inv(s["inv"])
    if TERSE:  # the same facts in fewer tokens (fmt_tokens.py / tok_alt.py); no "human":
        # the state is the game's, who plays it is the question's ("A human plays")
        assert MAP == "text", "--terse writes the map in words (--map text)"
        return terse_state(character, g["version"], s, s.get("hp_ago"), scr, msgs, words, inv)
    msg = GREETING.sub("", msgs[-1])  # the game's hello repeats the first line
    return (f"Character: {character}. Human player, NetHack {g['version']}.\n"
            f"HP {s['hp']}/{s['maxhp']}. Dlvl {s['dlvl']} (deepest so far {s['deepest']}), XL {s['xl']}, "
            f"AC {s['ac']}, turn {turn}.\n"
            f"Status: {status(scr)}\n"
            f"Message: {msg or 'none'}\n"
            f"{around}"
            f"Inventory (as last shown): {inv}")


def bot_state(identity, rec, inv, deepest):
    """A bot trace record (laya_trace: "b", "scr") in the humans' layout, so the unified model's fate / ttd
    questions can be asked in the bot's games: 'what would a human's game make of this state'. identity:
    val-dwa-law-fem -> Val-Dwa-Law-Fem as in the data (nao_corridor: role-race-align-gender)."""
    from bot_profile import AC, DEPTH, HP, MAXHP, TIME, XL
    role, race, align, gender = (w.capitalize() for w in identity.split("-"))
    b = rec["b"]
    return state_text({"character": f"{role}-{race}-{align}-{gender}", "version": "3.6.6"},
                      {"screen": rec["scr"], "x": b[0], "y": b[1] + 1, "hp": b[HP], "maxhp": b[MAXHP],
                       "dlvl": b[DEPTH], "deepest": deepest, "xl": b[XL], "ac": b[AC], "turn": b[TIME], "inv": inv})


def rows_of(g, keep=None):
    asc = g["death"] == "ascended"
    for i, s in enumerate(g["states"]):
        if keep is not None and i not in keep:  # --workers: only the picked states are made
            continue
        gain = g["maxlvl"] - s["deepest"]
        fate = 3 if asc else 0 if gain <= 1 else 1 if gain <= 5 else 2
        qs, gd = {"fate": Q_FATE}, {"fate": gold(fate, len(FATE))}
        if s["turn"] >= 0:
            left = g["turns"] - s["turn"]
            qs["ttd"], gd["ttd"] = Q_TTD, gold(2 if asc or left > 5000 else 0 if left <= 1000 else 1, len(TTD))
        yield {"id": f"{g['name']}@{g['start']}#{i}", "state": state_text(g, s), "questions": qs, "gold": gd,
               "_o": outcome(g, s)}


SKIP, PER_GAME, VAL, PER_BAND = set(), 0, False, False  # --workers: main() sets them before the pool forks
# --per-band: Dlvl bands of the bot's losses (bot_bottlenecks.py: grind 1-4, 5-12, 13-20, Medusa 21-24,
# castle 25-29, Gehennom 30+) -- a game gives one random state of each band it was in, not 2 of all (2 random states of
# a game are mostly early ones: the deep levels, where the bot dies, came out thin)
BANDS = (4, 12, 20, 24, 29)


def game_rows(line):
    """--workers: one line of the states file -> (key, split, rows) / (key, None, None) left out / None unreadable.
    The --per-game states picked with the game's own seed (crc32 of name@start), so any worker count gives the same
    rows (not the same as the one-process build's shared random stream)."""
    try:
        g = json.loads(line)
    except ValueError:  # a line cut by an interruption
        return None
    if not g["states"]:
        return None
    key = f"{g['name']}@{g['start']}"
    if key in SKIP:
        return key, None, None
    c = zlib.crc32(g["name"].lower().encode()) % 20
    split = "test" if c == 0 else "val" if VAL and c == 1 else "train"
    if PER_BAND:
        by = collections.defaultdict(list)
        for i, s in enumerate(g["states"]):
            by[bisect.bisect_left(BANDS, s["dlvl"])].append(i)
        rnd = random.Random(zlib.crc32(key.encode()))
        rs = list(rows_of(g, {rnd.choice(v) for _, v in sorted(by.items())}))
    elif PER_GAME and len(g["states"]) > PER_GAME:
        keep = set(random.Random(zlib.crc32(key.encode())).sample(range(len(g["states"])), PER_GAME))
        rs = list(rows_of(g, keep))
    else:
        rs = list(rows_of(g))
    return key, split, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, nargs="+")
    ap.add_argument("--out", default="human_data")
    ap.add_argument("--max-train", type=int, default=0, help="a random sample of at most this many train rows")
    ap.add_argument("--per-game", type=int, default=0,
                    help="at most this many random states of a game, every split (nh-v1 learned the games: ~8 "
                         "states with one label each)")
    ap.add_argument("--held-per-player", type=int, default=0,
                    help="val and test: at most this many random states of a player (with all states one player "
                         "made 21%% of the test)")
    ap.add_argument("--val", action="store_true", help="a val split of held-out players (crc32 % 20 == 1)")
    ap.add_argument("--v6-from", default="", help="take v6's bins and end classes from this dataset's meta.json")
    ap.add_argument("--v6", action="store_true",
                    help="human_q6's questions (ttd, depth, end, ascends) instead of fate / ttd; bins from train")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="files of games to leave out, name@start a line (rank_check.py --bad-out: another game's frames)")
    ap.add_argument("--exclude-players", nargs="*", default=[],
                    help="files of players to leave out, a name a line, any case (bot_list.py: bots). Taken out after "
                         "every random pick, so everyone else's rows stay as without it")
    ap.add_argument("--map", choices=["grid", "text"], default="grid",
                    help="the map around the hero as a grid, or in words: features and monsters with moves to them")
    ap.add_argument("--compact", action="store_true",
                    help="the inventory without articles, '(worn)' / '(wielded)' (compact_inv)")
    ap.add_argument("--terse", action="store_true", help="the state's head lines and the questions in fewer words")
    ap.add_argument("--end-classes", type=int, default=12,
                    help="v6 end: the commonest monster classes named (v7: 30 -- 463 tokens, needs head_max_len 512)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parse and make the rows in this many processes (--per-game picks: game_rows)")
    ap.add_argument("--per-band", action="store_true",
                    help="one random state of each Dlvl band a game was in (BANDS) instead of --per-game, every split")
    ap.add_argument("--games-cap", type=int, nargs="*", default=[],
                    help="at most this many games (all splits) from each --states file, in file order; 0 = all")
    a = ap.parse_args()
    global MAP, COMPACT, TERSE
    MAP, COMPACT, TERSE = a.map, a.compact, a.terse
    skip = {line.strip() for path in a.exclude for line in open(path, encoding="utf-8") if line.strip()}
    seen, left_out = set(), collections.Counter()
    rows = {"train": [], "val": [], "test": []} if a.val else {"train": [], "test": []}
    games, labels = collections.Counter(), collections.Counter()
    pick = random.Random(2)
    assert a.workers > 1 or not (a.per_band or a.games_cap), "--per-band / --games-cap: with --workers"
    if a.workers > 1:
        global SKIP, PER_GAME, VAL, PER_BAND
        SKIP, PER_GAME, VAL, PER_BAND = skip, a.per_game, a.val, a.per_band
        caps = a.games_cap or [0] * len(a.states)
        assert len(caps) == len(a.states), "--games-cap: a number per --states file"
        with Pool(a.workers) as pool:
            for path, cap in zip(a.states, caps):
                taken = 0
                with open(path, encoding="utf-8") as f:
                    for res in pool.imap(game_rows, f, chunksize=64):
                        if res is None:
                            continue
                        key, split, rs = res
                        if split is None or key in seen:
                            left_out["excluded" if split is None else "duplicate"] += 1
                            continue
                        if cap and taken >= cap:  # the files are shuffled: the first cap games are a random sample
                            continue
                        taken += 1
                        seen.add(key)
                        games[split] += 1
                        rows[split] += rs
                left_out[f"games_from:{Path(path).name}"] = taken
    for path in a.states if a.workers <= 1 else []:
        for line in open(path, encoding="utf-8"):
            try:
                g = json.loads(line)
            except ValueError:  # a line cut by an interruption
                continue
            if not g["states"]:
                continue
            key = f"{g['name']}@{g['start']}"
            if key in skip or key in seen:  # a game listed twice in the games file comes out twice
                left_out["excluded" if key in skip else "duplicate"] += 1
                continue
            seen.add(key)
            c = zlib.crc32(g["name"].lower().encode()) % 20
            split = "test" if c == 0 else "val" if a.val and c == 1 else "train"
            games[split] += 1
            rs = list(rows_of(g))
            if a.per_game and len(rs) > a.per_game:
                rs = pick.sample(rs, a.per_game)
            rows[split] += rs
    if a.held_per_player:
        for split in ("val", "test"):
            by = collections.defaultdict(list)
            for r in rows.get(split, []):
                by[r["id"].split("@")[0].lower()].append(r)
            rows[split] = [r for rs in by.values() for r in
                           (pick.sample(rs, a.held_per_player) if len(rs) > a.held_per_player else rs)]
    bots = {line.split()[0].lower() for path in a.exclude_players for line in open(path, encoding="utf-8")
            if line.strip() and not line.startswith("#")}
    if bots:
        for split in rows:
            out = [r for r in rows[split] if r["id"].split("@")[0].lower() in bots]
            gone = {r["id"].split("#")[0] for r in out}
            games[split] -= len(gone)
            left_out[f"bot_games_{split}"] += len(gone)
            left_out[f"bot_rows_{split}"] += len(out)
            rows[split] = [r for r in rows[split] if r["id"].split("@")[0].lower() not in bots]
    if a.v6 and a.v6_from:
        q6 = Builder.from_meta(json.loads(Path(a.v6_from, "meta.json").read_text())["v6"],
                               [r["_o"] for r in rows["train"]], terse=TERSE)
    else:
        q6 = Builder([r["_o"] for r in rows["train"]], n_classes=a.end_classes, terse=TERSE) if a.v6 else None
    for split, rs in rows.items():
        for r in rs:
            o = r.pop("_o")
            if q6:
                r["questions"], r["gold"] = q6.questions(o)
            for q, g in r["gold"].items():
                if q != "ttd" or not q6:  # v6 ttd: 17 bins, its labels in meta would be noise
                    labels[f"{split}:{q}:{g['label']}"] += 1
    if a.max_train and len(rows["train"]) > a.max_train:
        rows["train"] = random.Random(0).sample(rows["train"], a.max_train)
    random.Random(1).shuffle(rows["train"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for split, rs in rows.items():
        with open(Path(a.out, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    meta = {"games": games, "left_out": dict(left_out), **({"v6": q6.meta()} if q6 else {"fate": FATE, "ttd": TTD}),
            "rows": {k: len(v) for k, v in rows.items()}, "labels": dict(sorted(labels.items()))}
    Path(a.out, "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
