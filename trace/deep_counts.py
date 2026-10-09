"""How many human games reach the Dlvl bands where the bot dies: a deep-band
evaluation's precision is bounded by the games, not the states (every state of a game carries the game's fate:
Kish's design effect, n_eff ~ games). Val-Dwa from the states files (games with states, junk / another role's titles /
bots out, as human_all18): per band (a state on a Dlvl of the band; and maxlvl reaching it) games, players,
ascended, by split (crc32 of the name % 20: 0 test, 1 val, else train -- human_build's). Other roles from the list
(maxlvl only: no states / exclusion check) for scale.
  python deep_counts.py --out ../results/deep_counts.txt
"""
import argparse
import collections
import json
import re
import zlib

BANDS = ((1, 4), (5, 12), (13, 20), (21, 24), (25, 29), (30, 99))
HEAD = re.compile(r'"name": "([^"]*)".*?"start": (\d+).*?"maxlvl": (\d+), "death": "([^"]*)"')
DLVL = re.compile(r'"dlvl": (\d+)')
EMPTY = re.compile(r'"states": \[\]')


def band(d):
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= d <= hi)


def split(name):
    c = zlib.crc32(name.lower().encode()) % 20
    return "test" if c == 0 else "val" if c == 1 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["nao/val_dwa_states3e.jsonl", "nao/val_dwa_early.jsonl"])
    ap.add_argument("--exclude", nargs="+", default=["nao/bad_games_dwa3.txt", "nao/junk_games_dwa3e.txt",
                                                     "nao/bad_games_dwa_early.txt", "nao/junk_games_dwa_early.txt"])
    ap.add_argument("--bots", default="nao/bot_players_all.txt")
    ap.add_argument("--other-list", default="nao/other_any4_cut.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    skip = {ln.strip() for f in a.exclude for ln in open(f, encoding="utf-8") if ln.strip()}
    bots = {ln.split()[0].lower() for ln in open(a.bots, encoding="utf-8") if ln.strip() and not ln.startswith("#")}
    seen = set()
    by_state = collections.defaultdict(collections.Counter)   # (band, split) -> games / asc
    by_max = collections.defaultdict(collections.Counter)
    players = collections.defaultdict(set)
    states_n = collections.Counter()
    drop = collections.Counter()
    for path in a.states:
        for line in open(path, encoding="utf-8"):
            m = HEAD.search(line[:1000])
            if not m:
                drop["no head"] += 1
                continue
            name, start, maxlvl, death = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            key = f"{name}@{start}"
            if key in seen:
                drop["duplicate"] += 1
                continue
            seen.add(key)
            if key in skip:
                drop["excluded"] += 1
                continue
            if name.lower() in bots:
                drop["bot"] += 1
                continue
            if EMPTY.search(line[:2000]):
                drop["no states"] += 1
                continue
            s, asc = split(name), death == "ascended"
            bands = collections.Counter(band(int(d)) for d in DLVL.findall(line) if int(d) > 0)
            for b, n in bands.items():
                by_state[(b, s)]["games"] += 1
                by_state[(b, s)]["asc"] += asc
                states_n[(b, s)] += n
                players[(b, s)].add(name.lower())
            for b, (lo, hi) in enumerate(BANDS):
                if maxlvl >= lo:
                    by_max[(b, s)]["games"] += 1
                    by_max[(b, s)]["asc"] += asc
    other = collections.Counter()
    for g in json.load(open(a.other_list, encoding="utf-8")):
        for b, (lo, hi) in enumerate(BANDS):
            if g["maxlvl"] >= lo:
                other[(b, split(g["name"]))] += 1
    with open(a.out, "w", encoding="utf-8") as f:
        def out(x=""):
            print(x)
            f.write(x + "\n")
        out(f"Val-Dwa {a.states}: games {len(seen)}; dropped {dict(drop)}")
        out("band     split | games with a state there (ascended) | players | states | games reaching it by maxlvl"
            " | other roles reaching it (list)")
        for b, (lo, hi) in enumerate(BANDS):
            name = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
            for s in ("train", "val", "test"):
                g = by_state[(b, s)]
                out(f"{name:7s} {s:5s} | {g['games']:6d} ({g['asc']:5d}) | {len(players[(b, s)]):5d} | "
                    f"{states_n[(b, s)]:7d} | {by_max[(b, s)]['games']:6d} | {other[(b, s)]:7d}")
            h = {k: by_state[(b, 'val')][k] + by_state[(b, 'test')][k] for k in ('games', 'asc')}
            out(f"{name:7s} held  | {h['games']:6d} ({h['asc']:5d}) | "
                f"{len(players[(b, 'val')] | players[(b, 'test')]):5d} | "
                f"{states_n[(b, 'val')] + states_n[(b, 'test')]:7d} |")


if __name__ == "__main__":
    main()
