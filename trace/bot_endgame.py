"""Where exactly the bot dies from Medusa down (Medusa, the castle, Gehennom: docs/bot.md): for every game, its Medusa level
and castle level (found from the trace's screens the way the bot finds them: Medusa -- the first Dungeons level at
depth >= 21 with lots of water; the castle -- 'too hard to dig' at depth >= 25 or a water level below Medusa), and
the section its game ended in: before Medusa / Medusa / the mazes between Medusa and the castle / the castle /
Gehennom. Per section: games reaching it, passing it, the causes and the bot's strategy at the end, turns spent.
  python bot_endgame.py <run dir with evidence.json and trace/> [...] [--out results/bot_endgame] [--workers N]
-> <out>.txt, <out>.parquet (a row per game)
"""
import argparse
import json
import subprocess
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from bot_bottlenecks import top
from bot_profile import DEPTH, DNUM, HP, MAXHP, TIME, cause, match

LEVNO = 24
CASTLE_SOUNDS = ("blades being honed", "loud snoring", "dice being thrown", "General MacArthur",  # barracks
                 "courtly conversation", "sceptre pounded", "Off with his head", "Queen Beruthiel")  # throne room
WATER = 150          # '}' squares seen on a level: Medusa's levels and the castle (moat) have hundreds, others ~0
SECTIONS = ["before Medusa", "Medusa", "mazes", "castle", "Gehennom"]


def game(path):
    lines = subprocess.run(["zstd", "-dcq", path], capture_output=True, check=True).stdout.splitlines()[1:]
    b, stack, levels, order, hard, msgs, entry = None, [], {}, [], set(), [], None
    for line in lines:
        r = json.loads(line)
        stack = r.get("s", stack)
        if r.get("m"):
            msgs = (msgs + [r["m"]])[-6:]
        if "b" in r:
            b = r["b"]
        if b is None:
            continue
        key = (b[DNUM], b[LEVNO])
        if key not in levels:
            if b[DNUM] == 1 and entry is None and order:   # the first Gehennom level: from where, and how
                last = levels[order[-1]]
                entry = dict(from_depth=last["depth"], from_dnum=last["dnum"],
                             levelport=any("teleport" in m for m in msgs))
            levels[key] = dict(depth=b[DEPTH], dnum=b[DNUM], arrived=b[TIME], water=0, strat=Counter())
            order.append(key)
        lv = levels[key]
        lv["left"] = b[TIME]
        lv["strat"][top(stack)] += 1
        if "scr" in r:
            lv["water"] = max(lv["water"], sum(row.count("}") for row in r["scr"][1:22]))
        if b[DNUM] == 0 and b[DEPTH] >= 25 and ("too hard to dig" in r.get("m", "") or
                                                any(w in r.get("m", "") for w in CASTLE_SOUNDS) or b[DEPTH] == 29 or
                                                any("astle" in x for x in r.get("s", []))):
            hard.add(key)
    if b is None:
        return path, len(lines), 0, None
    doom = {k: v for k, v in levels.items() if v["dnum"] == 0}
    castle = max((levels[k]["depth"] for k in hard), default=None)
    crossed = entry is not None and not entry["levelport"] and entry["from_dnum"] == 0 and entry["from_depth"] >= 25
    if castle is None and crossed:
        castle = entry["from_depth"]   # walked or fell from the castle into the Valley
    wet = sorted(v["depth"] for v in doom.values() if 21 <= v["depth"] <= 28 and v["water"] >= WATER and
                  (castle is None or v["depth"] < castle))
    medusa = wet[0] if wet else None
    if castle is None and medusa is not None:
        castle = next((d for d in wet if d > medusa), None)
    end = levels[(b[DNUM], b[LEVNO])]
    d = end["depth"]
    if b[DNUM] == 1:
        section = "Gehennom"
    elif castle is not None and d == castle:
        section = "castle"
    elif medusa is not None and d == medusa:
        section = "Medusa"
    elif medusa is not None and d > medusa:
        section = "mazes"
    elif medusa is None and castle is not None and d > castle - 4:
        section = "Medusa or mazes?"   # dug through Medusa without seeing her water
    elif medusa is None and d >= 21:
        section = "Dlvl 21+, unclear"   # no wet level and no castle seen
    else:
        section = "before Medusa"
    t = {s: sum(v["left"] - v["arrived"] for v in levels.values() if f(v)) for s, f in [
        ("t_medusa", lambda v: medusa is not None and v["dnum"] == 0 and v["depth"] == medusa),
        ("t_mazes", lambda v: medusa is not None and v["dnum"] == 0 and medusa < v["depth"] and v["depth"] != castle),
        ("t_castle", lambda v: castle is not None and v["dnum"] == 0 and v["depth"] == castle)]}
    return path, len(lines), b[TIME], dict(
        section=section, depth=d, medusa=medusa, castle=castle, turn=b[TIME], hp=b[HP], maxhp=b[MAXHP],
        turns_on_level=b[TIME] - end["arrived"], strategy=top(stack),
        level_strategy=", ".join(f"{s} {n / sum(end['strat'].values()):.0%}" for s, n in end["strat"].most_common(3)),
        reached_medusa=medusa is not None or castle is not None,
        passed_medusa=castle is not None or (medusa is not None and any(v["depth"] > medusa for v in doom.values())),
        reached_castle=castle is not None, crossed_castle=crossed, gehennom=entry is not None,
        sokoban=sum(v["dnum"] == 4 for v in levels.values()),   # Sokoban levels visited (dnum 4)
        gehennom_levelport=entry is not None and entry["levelport"],
        gehennom_from=entry["from_depth"] if entry else None, **t)


def section_report(G, name):
    s = G[G.section == name]
    if not len(s):
        return []
    died = s[s.died]
    return ["", f"== {name}: {len(s)} games end here ({len(s) / len(G):.1%}), died {len(died)}; Dlvl "
                f"{s.depth.median():.0f}, turns on the level median {s.turns_on_level.median():.0f} "
                f"[p75 {s.turns_on_level.quantile(.75):.0f}], HP at end {s.hp.median():.0f}/{s.maxhp.median():.0f}",
            "  causes:   " + "; ".join(f"{c} {n / len(s):.0%}" for c, n in Counter(s.cause).most_common(10)),
            "  strategy at the end: " + "; ".join(f"{c} {n / len(s):.0%}" for c, n in Counter(s.strategy).most_common(8)),
            "  cause x strategy: " + "; ".join(f"{c} / {t} {n}" for (c, t), n in
                                              Counter(zip(s.cause, s.strategy)).most_common(10))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/bot_endgame")
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
    for path, _, _, end in done:
        r = res.get(path)
        if r is None or end is None:
            continue
        rows.append(dict(end, game=path, seed=r["trajectory_id"], progress=r["progress"],
                         cause=cause(r.get("cause_of_death")), died=bool(r.get("cause_of_death"))))
    G = pd.DataFrame(rows)
    n = len(G)
    m, c = G[G.reached_medusa], G[G.reached_castle]
    out = [f"bot from Medusa down: {n} games ({', '.join(a.runs)}); progress mean {G.progress.mean():.3f}", "",
           f"Medusa level: Dlvl {m.medusa.value_counts().sort_index().to_dict()}; "
           f"castle: Dlvl {c.castle.value_counts().sort_index().to_dict()}; mazes between: "
           f"{(c.castle - c.medusa - 1).value_counts().sort_index().to_dict()}", "",
           f"reached Medusa {len(m) / n:.1%}; passed it {G.passed_medusa.mean():.1%} "
           f"({G.passed_medusa.sum() / max(len(m), 1):.0%} of those reaching)",
           f"reached the castle {len(c) / n:.1%} ({len(c) / max(G.passed_medusa.sum(), 1):.0%} of those past Medusa); "
           f"crossed it into the Valley {G.crossed_castle.mean():.1%} ({G.crossed_castle.sum() / max(len(c), 1):.0%} "
           f"of those reaching)",
           f"Gehennom {G.gehennom.mean():.1%}: by level teleport {G.gehennom_levelport.mean():.1%} (from Dlvl "
           f"{G[G.gehennom_levelport].gehennom_from.value_counts().sort_index().to_dict()}), "
           f"other {(G.gehennom & ~G.gehennom_levelport).mean():.1%} (from Dlvl "
           f"{G[G.gehennom & ~G.gehennom_levelport].gehennom_from.value_counts().sort_index().to_dict()})",
           *([f"visited Sokoban {(G.sokoban > 0).mean():.1%} (all 4 levels {(G.sokoban >= 4).mean():.1%}); of those: "
              f"reached the castle {G[G.sokoban > 0].reached_castle.mean():.1%}, crossed it "
              f"{G[G.sokoban > 0].crossed_castle.mean():.1%}, Gehennom {G[G.sokoban > 0].gehennom.mean():.1%}"]
             if (G.sokoban > 0).any() else []),
           "", "where games end: " + "; ".join(f"{k} {v / n:.1%}" for k, v in G.section.value_counts().items()),
           f"turns spent (median of those reaching): Medusa {m.t_medusa.median():.0f}, mazes "
           f"{G[G.passed_medusa].t_mazes.median():.0f}, castle {c.t_castle.median():.0f}"]
    for name in SECTIONS[1:] + ["Medusa or mazes?", "Dlvl 21+, unclear"]:
        out += section_report(G, name)
    text = "\n".join(out) + "\n"
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".txt").write_text(text, encoding="utf-8")
    G.to_parquet(a.out + ".parquet", index=False)


if __name__ == "__main__":
    main()
