"""Junk in the extracted human games: every check counts the games /
states it hits and keeps a few examples (name@start), so a category is judged on its count before anything is cut.
  python junk_scan.py nao/other_states3.jsonl [--events nao/other_states3.jsonl.events.jsonl] [--limit N]
States file (nao_corridor states): a game per line, {name, start, turns, character, version, states: [{dlvl, hp, maxhp,
ac, xl, turn, deepest, kind, x, y, screen (24 rows), inv, hp_ago, msgs, mcol}], error?}. Events file: {name, start,
events: [[turn, Dlvl, HP, max HP, AC, XL, msg]]}.
"""
import argparse
import collections
import json
import re

CTRL = re.compile(r"[\x00-\x1f\x7f]")
HIGH = re.compile(r"[\x80-\xff]")
INV = re.compile(r"^[a-zA-Z$#] - \S")
EXAMPLES = 5
# --bad-out: the games to exclude whole (human_build.py --exclude) -- interleaved games / a half-drawn
# turn, rerun duplicates, screens with junk; ~0.15 % of the games
BAD = {"duplicate game", "turn past the game's end", "turn goes back", "control chars on screen",
       "bytes >= 0x80 in message/status", "--More-- on message row", "HP <= 0", "odd inventory line",
       "status name is not the player's"}


class Tally:
    def __init__(self):
        self.games, self.states, self.ex = collections.Counter(), collections.Counter(), collections.defaultdict(list)
        self.bad = set()

    def hit(self, cat, key, n_states=0):
        if cat in BAD:
            self.bad.add(key)
        self.games[cat] += 1
        self.states[cat] += n_states
        if len(self.ex[cat]) < EXAMPLES:
            self.ex[cat].append(key)

    def report(self, title, n_games, n_states):
        print(f"== {title}: {n_games} games, {n_states} states")
        for cat, n in sorted(self.games.items(), key=lambda kv: -kv[1]):
            s = self.states[cat]
            print(f"  {cat:34s} {n:8d} games ({100 * n / max(n_games, 1):6.2f} %)"
                  + (f"  {s:9d} states ({100 * s / max(n_states, 1):6.2f} %)" if s else "")
                  + "  e.g. " + " ".join(self.ex[cat]))


def scan_states(path, limit, bad_out=""):
    t, seen, n_games, n_states = Tally(), set(), 0, 0
    kinds, versions = collections.Counter(), collections.Counter()
    for line in open(path, encoding="utf-8"):
        try:
            g = json.loads(line)
        except ValueError:
            t.hit("unreadable line", "?")
            continue
        n_games += 1
        key = f"{g['name']}@{g['start']}"
        versions[g.get("version")] += 1
        if (g["name"], g["start"]) in seen:
            t.hit("duplicate game", key)
        seen.add((g["name"], g["start"]))
        if g.get("error"):
            t.hit("broken ttyrec (error)", key)
        st = g["states"]
        n_states += len(st)
        if not st:
            t.hit("no states", key)
            continue
        name9 = g["name"][:9].lower()  # hitpointbar cuts the name to 9 letters
        bad = collections.Counter()
        last_turn = -1
        for s in st:
            kinds[s["kind"]] += 1
            scr = s["screen"]
            if len(scr) != 24:
                bad["screen not 24 rows"] += 1
                continue
            top, bot = scr[22], scr[23]
            if top.startswith("["):
                bad["hitpointbar [title] on status"] += 1
            # 3.4.3 drops "Name the Rank" when the row is too long: the row starts with St:
            if not top.lstrip("[").lower().startswith(name9) and not top.lstrip().startswith("St:"):
                bad["status name is not the player's"] += 1
            if any(CTRL.search(r) for r in scr):
                bad["control chars on screen"] += 1
            if HIGH.search(scr[0]) or HIGH.search(top) or HIGH.search(bot):
                bad["bytes >= 0x80 in message/status"] += 1
            if "--More--" in scr[0]:
                bad["--More-- on message row"] += 1
            if not (1 <= s["dlvl"] <= 60):
                bad["Dlvl out of 1..60"] += 1
            if s["dlvl"] > s["deepest"]:
                bad["Dlvl > deepest"] += 1
            if s["hp"] > s["maxhp"] or s["maxhp"] <= 0:
                bad["HP > max HP or max HP <= 0"] += 1
            if s["hp"] <= 0:
                bad["HP <= 0"] += 1
            if s["turn"] < 0:
                bad["no turn counter"] += 1
            elif s["turn"] < last_turn:
                bad["turn goes back"] += 1
            last_turn = max(last_turn, s["turn"])
            if g.get("turns") and s["turn"] > g["turns"] + 50:
                bad["turn past the game's end"] += 1
            if not (0 <= s["x"] < 80 and 1 <= s["y"] <= 21):
                bad["cursor off the map"] += 1
            elif s["x"] >= len(scr[s["y"]]) or scr[s["y"]][s["x"]] != "@":
                bad["no @ at the cursor"] += 1
            if any(not INV.match(i) for i in s["inv"]):
                bad["odd inventory line"] += 1
            if any(CTRL.search(m) or HIGH.search(m) for m in s["msgs"]):
                bad["odd chars in msgs"] += 1
        for cat, n in bad.items():
            t.hit(cat, key, n)
        if limit and n_games >= limit:
            break
    t.report(f"states {path}", n_games, n_states)
    print("  kinds:", dict(kinds))
    print("  versions:", dict(versions))
    if bad_out:
        with open(bad_out, "w", encoding="utf-8") as f:
            f.writelines(k + "\n" for k in sorted(t.bad))
        print(f"  {len(t.bad)} games to exclude -> {bad_out}")
    return seen


DIP = 3  # clean_cut_turns: the longest run of half-drawn turns taken for one


def clean_cut_turns(ev):
    """Logs extracted before nao_corridor.py's status fix (419k games): an event whose turn dips below the last one for
    at most DIP events with the same turn (then back at or above it: "T:2" for 2686, "T:5" between 477 and 535 after a
    count, "T:96" between 858 and 992, "T:4" twice between 40303 and 40307 on a level change) or is -1 after a turn was
    seen is a half-drawn status -- the last turn stands (the
    true one lies between it and the next). A real drop (a restored save, another game) stays below and is kept.
    In place; the fixed extractor reads only drawn statuses (its logs still have ~0.3 % games with such a dip)."""
    last, n = -1, len(ev)
    for i, e in enumerate(ev):
        t = e[0]
        if last >= 0 and 0 <= t < last:
            j = i + 1
            while j < n and j - i < DIP and ev[j][0] == t:
                j += 1
            if j == n or ev[j][0] >= last or ev[j][0] < 0:
                for k in range(i, j):
                    ev[k][0] = last
                t = last
        elif last >= 0 and t < 0:
            e[0] = t = last
        last = t
    return ev


def scan_events(path, limit, games, clean=False):
    t, n_games, n_ev = Tally(), 0, 0
    seen = set()
    for line in open(path, encoding="utf-8"):
        try:
            g = json.loads(line)
        except ValueError:
            t.hit("unreadable line", "?")
            continue
        n_games += 1
        key = f"{g['name']}@{g['start']}"
        k = (g["name"], g["start"])
        if k in seen:
            t.hit("duplicate log (rerun)", key)
        seen.add(k)
        if games is not None and k not in games:
            t.hit("log of a game not in states", key)
        ev = g["events"]
        n_ev += len(ev)
        if not ev:
            t.hit("empty log", key)
            continue
        if clean:
            clean_cut_turns(ev)
        bad = collections.Counter()
        last = -1
        for e in ev:
            msg = e[-1]
            if CTRL.search(msg):
                bad["control chars in msg"] += 1
            if HIGH.search(msg):
                bad["bytes >= 0x80 in msg"] += 1
            if "--More--" in msg:
                bad["--More-- in msg"] += 1
            if e[0] >= 0 and e[0] < last:
                bad["turn goes back"] += 1
            last = max(last, e[0])
            if e[0] < 0:
                bad["before the first status"] += 1
        for cat, n in bad.items():
            t.hit(cat, key, n)
        if limit and n_games >= limit:
            break
    t.report(f"events {path}", n_games, n_ev)
    return t.bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("states")
    p.add_argument("--events", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bad-out", default="", help="the games to exclude whole (categories in BAD), one name@start a line")
    p.add_argument("--events-bad", action="store_true", help="also the games whose log goes back in turns")
    p.add_argument("--clean-cut", action="store_true", help="logs from before the status fix: clean_cut_turns() first")
    a = p.parse_args()
    games = scan_states(a.states, a.limit, a.bad_out) if a.states != "-" else None
    if a.events:
        bad = scan_events(a.events, a.limit, None if a.limit else games, a.clean_cut)
        if a.bad_out and a.events_bad:  # logs from before the status fix only with --clean-cut
            with open(a.bad_out, "a", encoding="utf-8") as f:
                f.writelines(k + "\n" for k in sorted(bad))
            print(f"  {len(bad)} games from the log -> {a.bad_out}")


if __name__ == "__main__":
    main()
