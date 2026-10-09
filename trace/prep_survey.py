"""How humans build the hero up, from the message logs (the dead differ from the ascended by
AC and resistances, not XL, gained on Dlvl 5-20; the bot has neither): per kind of preparation event -- games with
it, events, Dlvl band, turn, XL, AC (and AC gained for wear / put on) -- for all games and for the games that went
deep (maxlvl >= 25) or ascended, and by the player's earlier games (skill: the same n as add_skill.py would give is
not at hand here, so the player's share of games in the file stands in: games by players with >= 100 games).
Logs cleaned as the question builder will (junk_scan.clean_cut_turns; junk lists out; events before the first status
and with control / high chars dropped).
  python prep_survey.py nao/val_dwa_states3e.jsonl --exclude nao/junk_games_dwa3e.txt --workers 15 \
      --out ../results/prep_survey_dwa.txt
"""
import argparse
import collections
import json
import re
import statistics
from multiprocessing import Pool

from junk_scan import clean_cut_turns

BAD_CHARS = re.compile(r"[\x00-\x1f\x7f-￿]")
KEY = re.compile(r'"name": "([^"]*)", "start": (\d+)')
END = ("You die...", "Do you want your possessions identified?")
BANDS = ((1, 4), (5, 12), (13, 20), (21, 24), (25, 29), (30, 99))
# a prompt, then within LOOK events what it gave (msg_after.py: a worn armor shows "You finish your dressing
# maneuver." or "You are now wearing a dwarvish cloak.", not its inventory line; the AC changes an event or two
# later, when the status is redrawn -- AC gained: before the prompt minus the AC AC_LAG turns after the result)
PROMPTED = {"wear": (re.compile(r"^What do you want to wear\?"),
                     re.compile(r"You finish your dressing maneuver|You are now wearing")),
            "put on": (re.compile(r"^What do you want to put on\?"),
                       re.compile(r"(?:^|  )[a-zA-Z] - .*\((?:on (?:left|right) hand|being worn)\)|You are now wearing"))}
LOOK, AC_LAG = 4, 3
# one message is the event. A message and its --More-- sequel: the first is often missing from the log
# (excal_check.py, 300 ascended Val-Dwa: "From the murky depths, a hand reaches up to bless the sword" in 23 %, the
# game names Excalibur in 70 %; the sequel "As the hand retreats, the fountain disappears!" is there) -- either one
# counts; for a lawful hero (every Val-Dwa) the sequel comes only with Excalibur. Not counted: speed and cold
# resistance -- a Valkyrie's own (cold from the start, speed at XL 7)
SAID = {"enchant armor": re.compile(r"glows silver for (?:a moment|a while)"),
        "excalibur": re.compile(r"a hand reaches up to bless the sword|As the hand retreats, the fountain disappears"),
        "fire res": re.compile(r"You feel a momentary chill"),
        "shock res": re.compile(r"Your health currently feels amplified"),
        "sleep res": re.compile(r"You feel wide awake"),
        "poison res": re.compile(r"You feel (?:especially )?healthy"),
        "disint res": re.compile(r"You feel very firm"),
        "telepathy": re.compile(r"You feel a strange mental acuity"),
        "tele control": re.compile(r"You feel in control of yourself|You feel centered in your personal space"),
        "reflection id": re.compile(r"amulet of reflection|shield of reflection"),
        "magic res id": re.compile(r"cloak of magic resistance|gray dragon scale"),
        "dragon armor": re.compile(r"dragon scale mail")}
# once a game: an item named (the inventory lists it again and again), Excalibur (its two messages)
ONCE = {"excalibur", "reflection id", "magic res id", "dragon armor"}


def band(d):
    return next(f"{lo}-{hi}" if hi < 99 else f"{lo}+" for lo, hi in BANDS if d <= hi)


def game(args):
    line, info = args
    g = json.loads(line)
    ev = [e for e in clean_cut_turns(g["events"]) if e[0] >= 0 and not BAD_CHARS.search(e[-1])]
    cut = next((i for i, e in enumerate(ev) if e[-1].startswith(END)), len(ev))
    ev = ev[:cut]
    out, seen_id = [], set()
    deepest, last_ac = 0, None
    for i, e in enumerate(ev):
        t, dl, hp, maxhp, ac, xl, msg = e
        if dl > deepest:
            if deepest:  # a new deepest level, not the start
                out.append(("descend", dl, t, xl, ac, None))
            deepest = dl
        # AC better by any means (worn armor, enchantment, rings, spells -- also short ones like protection);
        # the prompts miss a part (msg_after.py: a frame may skip the prompt), the status does not
        if last_ac is not None and -99 < ac < last_ac:
            out.append(("AC better", dl, t, xl, last_ac, last_ac - ac))
        if ac > -99:
            last_ac = ac
        for kind, (ask, got) in PROMPTED.items():
            if ask.search(msg):
                for j in range(i + 1, min(i + 1 + LOOK, len(ev))):
                    if got.search(ev[j][-1]):
                        later = [f for f in ev[j:] if f[0] <= ev[j][0] + AC_LAG]
                        out.append((kind, dl, t, xl, ac, ac - later[-1][4]))
                        break
        for kind, rx in SAID.items():
            if rx.search(msg) and kind not in seen_id:
                out.append((kind, dl, t, xl, ac, None))
                if kind in ONCE:
                    seen_id.add(kind)
    return g["name"], info, out


def jobs(path, games):
    for line in open(path, encoding="utf-8"):
        m = KEY.search(line[:300])
        if m and (m.group(1), int(m.group(2))) in games:
            yield line, games[(m.group(1), int(m.group(2)))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("states", help="the states file (game fate); its .events.jsonl is read")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    bad = {l.strip() for f in a.exclude for l in open(f, encoding="utf-8") if l.strip()}
    games, per_player = {}, collections.Counter()
    for line in open(a.states, encoding="utf-8"):
        g = json.loads(line)
        if f"{g['name']}@{g['start']}" in bad or not g["states"]:
            continue
        games[(g["name"], g["start"])] = ("ascended" if "ascended" in g["death"] else
                                          "deep" if g["maxlvl"] >= 25 else "other")
        per_player[g["name"]] += 1
        if a.limit and len(games) >= a.limit:
            break
    groups = collections.Counter(games.values())
    many = {n for n, c in per_player.items() if c >= 100}
    groups_many = collections.Counter(v for (n, _), v in games.items() if n in many)
    rows = collections.defaultdict(list)  # kind -> [(group, many, dl, t, xl, ac, dac)]
    games_with = collections.defaultdict(lambda: collections.Counter())
    with Pool(a.workers) as pool:
        for name, grp, out in pool.imap_unordered(game, jobs(a.states + ".events.jsonl", games), chunksize=64):
            for kind in {o[0] for o in out}:
                games_with[kind][grp] += 1
                games_with[kind]["all"] += 1
            for kind, dl, t, xl, ac, dac in out:
                rows[kind].append((grp, name in many, dl, t, xl, ac, dac))
    n_all = sum(groups.values())
    with open(a.out, "w", encoding="utf-8") as f:
        def out(x=""):
            print(x)
            f.write(x + "\n")
        out(f"{a.states}: {n_all} games ({dict(groups)}); players with >= 100 games: {len(many)}, their games "
            f"{dict(groups_many)}")
        out("\n== games with it: all / ascended / deep (maxlvl >= 25, not ascended) / other (share of the group)")
        for kind in sorted(rows, key=lambda k: -games_with[k]["all"]):
            gw = games_with[kind]
            out(f"{kind:14s} {gw['all']:6d} ({100 * gw['all'] / n_all:4.1f} %)  asc "
                f"{100 * gw['ascended'] / max(groups['ascended'], 1):5.1f} %  deep "
                f"{100 * gw['deep'] / max(groups['deep'], 1):5.1f} %  other "
                f"{100 * gw['other'] / max(groups['other'], 1):5.1f} %")
        out("\n== events: count, by Dlvl band (%), median turn / XL / AC at it, AC gained (wear / put on), "
            "share by players with >= 100 games")
        for kind in sorted(rows, key=lambda k: -len(rows[k])):
            r = rows[kind]
            bands = collections.Counter(band(x[2]) for x in r)
            bs = " ".join(f"{b} {100 * bands[b] / len(r):3.0f}" for b in
                          (f"{lo}-{hi}" if hi < 99 else f"{lo}+" for lo, hi in BANDS))
            dac = [x[6] for x in r if x[6] is not None]
            out(f"{kind:14s} {len(r):8d}  {bs}  turn {statistics.median(x[3] for x in r):7.0f}  XL "
                f"{statistics.median(x[4] for x in r):4.0f}  AC {statistics.median(x[5] for x in r):4.0f}"
                + (f"  AC gained median {statistics.median(dac):.0f}, > 0 in {100 * sum(d > 0 for d in dac) / len(dac):.0f} %"
                   if dac else "")
                + f"  many-games players {100 * sum(x[1] for x in r) / len(r):.0f} %")


if __name__ == "__main__":
    main()
