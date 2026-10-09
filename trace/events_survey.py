"""What the message logs hold, before any action question is built from them: per message (a sentence of
an event, digits -> #), in how many games it appears, and how much more often it comes in the last NEAR turns before
a death than in a random NEAR-turn window of a game. Also how often a prompt ("... [yn] (n)") carries its answer in
the log.
Logs cleaned as the question builder will: the junk lists out, junk_scan.clean_cut_turns(), events before the first
status and messages with control / high chars dropped.
  python events_survey.py nao/val_dwa_states3e.jsonl --exclude nao/junk_games_dwa3e.txt --workers 15 \
      --out ../results/events_survey_dwa.txt
"""
import argparse
import collections
import json
import random
import re
import zlib
from multiprocessing import Pool

from junk_scan import clean_cut_turns

NEAR = 100  # turns before a death
MIN_NEAR = 100  # messages near fewer deaths are not ranked
DIGITS = re.compile(r"\d+")
BAD_CHARS = re.compile(r"[\x00-\x1f\x7f-￿]")
PROMPT = re.compile(r"^(.*?\[[^\]]*\])\s*(\([^)]*\))?\s*(.*)$")
NOT_DEATH = ("ascended", "quit", "escaped")
KEY = re.compile(r'"name": "([^"]*)", "start": (\d+)')
END = ("You die...", "Do you want your possessions identified?")
# what the player did: an extended command's echo ("# pray") or an item prompt ("What do you want to drink?")
ACTION = re.compile(r"^(# [a-z]+$|What do you want to [\w ]+\?)")


def split(msg):
    """The sentences of a message row, digits -> # (two spaces part NetHack's messages on one row)."""
    return {DIGITS.sub("#", s.strip()) for s in msg.split("  ") if s.strip()}


def game(args):
    line, (end, died) = args
    g = json.loads(line)
    ev = [e for e in clean_cut_turns(g["events"]) if e[0] >= 0 and not BAD_CHARS.search(e[-1])]
    # the game ends at "You die..." (the DYWYPI questions and the next game after it are not play)
    cut = next((i for i, e in enumerate(ev) if e[-1].startswith(END)), len(ev))
    ev = ev[:cut]
    # the baseline: a NEAR-turn window at a random place of the game (the game's own seed)
    lo = random.Random(zlib.crc32(f"{g['name']}@{g['start']}".encode())).randint(0, max(end - NEAR, 0))
    seen, near, ref, prompts = set(), set(), set(), [0, 0]
    for e in ev:
        for s in split(e[-1]):
            m = PROMPT.match(s)
            if m:  # a prompt: answered if anything follows the bracket and the default
                prompts[0] += 1
                prompts[1] += bool(m.group(3))
                s = m.group(1)
            a = ACTION.match(s)
            if a:  # counted once a game whatever the prompt's item list
                s = "ACT " + a.group(1)
            seen.add(s)
            if died and end - e[0] <= NEAR:
                near.add(s)
            if lo <= e[0] < lo + NEAR:
                ref.add(s)
    return seen, near, ref, prompts, died


def jobs(path, games):
    for line in open(path, encoding="utf-8"):
        m = KEY.search(line[:300])
        if m and (m.group(1), int(m.group(2))) in games:
            yield line, games[(m.group(1), int(m.group(2)))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("states", help="the states file: game end turn and death; its .events.jsonl is read")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top", type=int, default=60)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    bad = {l.strip() for f in a.exclude for l in open(f, encoding="utf-8") if l.strip()}
    games = {}
    for line in open(a.states, encoding="utf-8"):
        g = json.loads(line)
        if f"{g['name']}@{g['start']}" in bad or not g["states"]:
            continue
        games[(g["name"], g["start"])] = (g["turns"], not any(w in g["death"] for w in NOT_DEATH))
        if a.limit and len(games) >= a.limit:
            break
    n_games = n_died = 0
    seen, near, ref, prompts = collections.Counter(), collections.Counter(), collections.Counter(), [0, 0]
    with Pool(a.workers) as pool:
        for s, nr, rf, pr, died in pool.imap_unordered(game, jobs(a.states + ".events.jsonl", games), chunksize=64):
            n_games += 1
            n_died += died
            seen.update(s)
            near.update(nr)
            ref.update(rf)
            prompts[0] += pr[0]
            prompts[1] += pr[1]
    with open(a.out, "w", encoding="utf-8") as f:
        def out(x=""):
            print(x)
            f.write(x + "\n")
        out(f"{a.states}: {n_games} games read ({len(games)} kept), {n_died} died; NEAR {NEAR} turns")
        out(f"prompts {prompts[0]}, with the answer in the log {prompts[1]} "
            f"({100 * prompts[1] / max(prompts[0], 1):.1f} %)")
        out(f"\n== most common (games with it; near death: died games with it in the last {NEAR} turns)")
        for s, n in seen.most_common(a.top):
            out(f"{n:8d} {100 * n / n_games:5.1f} %  near death {near[s]:7d}  {s[:110]}")
        out(f"\n== before deaths: share of the died games with it in the last {NEAR} turns over the share of all games "
            f"with it in a random {NEAR}-turn window (>= {MIN_NEAR} died games with it), top {a.top}")
        ranked = sorted((((near[s] / n_died) / ((ref[s] + 1) / n_games), s) for s in near if near[s] >= MIN_NEAR),
                        reverse=True)
        for lift, s in ranked[:a.top]:
            out(f"{lift:7.1f}x  near death {near[s]:6d}  random window {ref[s]:6d}  seen {seen[s]:7d}  {s[:100]}")
        out(f"\n== actions (extended commands, item prompts): games with it, died games with it in the last {NEAR} "
            f"turns, all games with it in a random {NEAR}-turn window, the ratio as above")
        for s, n in sorted(((s, n) for s, n in seen.items() if s.startswith("ACT ")), key=lambda x: -x[1])[:a.top]:
            nr, rf = near[s], ref[s]
            out(f"{n:8d} {100 * n / n_games:5.1f} %  near death {nr:6d}  random window {rf:6d}  "
                f"{(nr / max(n_died, 1)) / ((rf + 1) / n_games):6.1f}x  {s[4:]}")


if __name__ == "__main__":
    main()
