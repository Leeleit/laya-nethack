"""The bot's minotaur encounters (~14 % of its games end to a minotaur, in the mazes
and on the castle level): for each game, the first record per level with a minotaur within --dist squares, the
hero's state then (HP, XL, AC, depth, the bot's strategy), what it carries that could answer one (a digging tool,
known/unknown wands, scrolls that may be scare monster or teleportation), and the outcome: died within 50 / 300
turns, killed by the minotaur, left the level. These are the candidate fork points (the trace record index is
the step k of plan_branches.py).
  python minotaur_encounters.py <run dir with evidence.json and trace/> [...] [--out results/minotaur] [--workers N]
-> <out>.txt, <out>.parquet (a row per encounter)
"""
import argparse
import json
import re
import subprocess
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from bot_bottlenecks import top
from bot_profile import AC, DEPTH, DNUM, HP, MAXHP, TIME, XL, cause, match

LEVNO = 24
DIG = re.compile(r"\b(pick-axe|dwarvish mattock)\b")
WAND_DIG = re.compile(r"\bwand of digging\b")
WAND_UNKNOWN = re.compile(r"\b\w+ wand\b(?! of )")
SCROLL_UNKNOWN = re.compile(r"\bscrolls? labeled\b")
SCARE = re.compile(r"\bscrolls? of scare monster\b")
TELE = re.compile(r"\bscrolls? of teleportation\b")


def game(path, dist=7):
    lines = subprocess.run(["zstd", "-dcq", path], capture_output=True, check=True).stdout.splitlines()[1:]
    b, stack, inv, seen, enc, levels = None, [], [], set(), [], []
    for k, line in enumerate(lines):
        r = json.loads(line)
        stack = r.get("s", stack)
        inv = r.get("inv", inv)
        b = r.get("b", b)
        if b is None:
            continue
        key = (b[DNUM], b[LEVNO])
        if not levels or levels[-1][0] != key:
            levels.append((key, b[TIME]))
        if key in seen or "scr" not in r:
            continue
        near = [m for m in r.get("mon", []) if "minotaur" in m[0] and max(abs(m[1]), abs(m[2])) <= dist]
        if not near:
            continue
        seen.add(key)
        enc.append(dict(k=k, turn=b[TIME], depth=b[DEPTH], dnum=b[DNUM], level=key, hp=b[HP], maxhp=b[MAXHP],
                        xl=b[XL], ac=b[AC], dist=min(max(abs(m[1]), abs(m[2])) for m in near), n_mino=len(near),
                        strategy=top(stack), dig_tool=any(DIG.search(x) for x in inv),
                        wand_digging=any(WAND_DIG.search(x) for x in inv),
                        wands_unknown=sum(bool(WAND_UNKNOWN.search(x)) for x in inv),
                        scrolls_unknown=sum(bool(SCROLL_UNKNOWN.search(x)) for x in inv),
                        scare=any(SCARE.search(x) for x in inv), teleport=any(TELE.search(x) for x in inv),
                        b=list(b)))
    if b is None:
        return path, len(lines), 0, []
    end = b[TIME]
    for e in enc:
        later = [t for key, t in levels if t > e["turn"] and key != e["level"]]
        e["left_level_in"] = (min(later) - e["turn"]) if later else None
        e["end_turn"] = end
        del e["level"]
    return path, len(lines), end, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/minotaur")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    files, results = [], []
    for run in a.runs:
        files += sorted(str(p) for p in Path(run, "trace").glob("*.jsonl.zst"))
        results += json.load(open(Path(run, "evidence.json")))["results"]
    with Pool(a.workers) as pool:
        done = pool.map(game, files, chunksize=4)
    res = match(done, results)
    rows = []
    for path, _, _, enc in done:
        r = res.get(path)
        if r is None:
            continue
        c = cause(r.get("cause_of_death"))
        for e in enc:
            ttd = e["end_turn"] - e["turn"] if r.get("cause_of_death") else None
            rows.append(dict(e, game=path, seed=r["trajectory_id"], cause=c, ttd=ttd,
                             died50=ttd is not None and ttd <= 50, died300=ttd is not None and ttd <= 300,
                             killed_by_mino=ttd is not None and ttd <= 300 and "minotaur" in c))
    E = pd.DataFrame(rows)
    games = len(set(f for f, *_ in done))
    out = [f"minotaur encounters (first sighting within 7 squares per level): {len(E)} in {E.seed.nunique()} of "
           f"{games} games ({', '.join(a.runs)})", "",
           f"depth {E.depth.value_counts().sort_index().to_dict()}",
           f"died within 50 turns {E.died50.mean():.1%}, within 300 {E.died300.mean():.1%}, killed by a minotaur "
           f"within 300 {E.killed_by_mino.mean():.1%}; left the level within 300 turns "
           f"{(E.left_level_in.fillna(1e9) <= 300).mean():.1%}",
           f"hero then: HP {E.hp.median():.0f}/{E.maxhp.median():.0f}, XL {E.xl.median():.0f}, AC {E.ac.median():.0f}, "
           f"distance {E.dist.median():.0f}",
           f"carries: digging tool {E.dig_tool.mean():.0%}, wand of digging (known) {E.wand_digging.mean():.0%}, "
           f"unknown wands {(E.wands_unknown > 0).mean():.0%}, unknown scrolls {(E.scrolls_unknown > 0).mean():.0%}, "
           f"scare monster (known) {E.scare.mean():.0%}, teleportation (known) {E.teleport.mean():.0%}", "",
           "by the bot's strategy at the sighting (n, killed by a minotaur within 300 turns):"]
    for s, n in Counter(E.strategy).most_common(10):
        x = E[E.strategy == s]
        out.append(f"  {s:42} {n:4d}  {x.killed_by_mino.mean():.0%}")
    out += ["", "killed within 300 turns by what the hero carries:"]
    for col in ("dig_tool", "wand_digging", "scare", "teleport"):
        for v in (True, False):
            x = E[E[col] == v]
            if len(x):
                out.append(f"  {col}={v!s:5} n {len(x):4d}  killed by a minotaur {x.killed_by_mino.mean():.0%}, "
                           f"died {x.died300.mean():.0%}")
    text = "\n".join(out) + "\n"
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".txt").write_text(text, encoding="utf-8")
    E.to_parquet(a.out + ".parquet", index=False)


if __name__ == "__main__":
    main()
