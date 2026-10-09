"""What the maxlvl >= 4 threshold of the earlier human lists (--min-maxlvl 4) cuts off.
The same filters as nao_corridor.py list (known fate: died or ascended; 3.4.3 / 3.6; a ttyrec of the game exists) plus
turns > 50 and no bot accounts (nao/bot_players.txt, as 15d/16d); per group (Val-Dwa, Val-Hum, every other role): games
and turns with maxlvl < 4 vs >= 4, ascensions, where the cut games end (deathlev), turns.
  python maxlvl_cut.py [--nld /data/laya/nethack/nld/nld-nao] > ../results/maxlvl_cut.txt
"""
import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from nao_corridor import FIVE_MINS, outcome_ok, stamp, xlog


def group(r):
    if r.get("role") == "Val":
        return {"Dwa": "Val-Dwa", "Hum": "Val-Hum"}.get(r.get("race"), "Val-other")
    return "other roles"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nld", default="/data/laya/nethack/nld/nld-nao")
    ap.add_argument("--bots", default="nao/bot_players.txt")
    ap.add_argument("--min-turns", type=int, default=50)
    a = ap.parse_args()
    root = Path(a.nld)
    bots = {ln.split()[0].lower() for ln in open(a.bots) if ln.strip() and not ln.startswith("#")}
    users = {d.lower(): d for d in os.listdir(root / "nld-nao-unzipped")}
    stamps = {}
    turns = defaultdict(list)  # (group, cut) -> turns of each game
    asc = Counter()
    endlev = defaultdict(Counter)  # group -> deathlev of the cut games
    drop = Counter()
    for xf in sorted(root.glob("xlogfile.*")):
        if xf.name.endswith("dev") or xf.name == "blacklist.txt":
            continue
        for r in xlog(str(xf)):
            if not outcome_ok(r.get("death", ""), "any"):
                drop["quit/escaped"] += 1
                continue
            if not (r.get("version", "").startswith("3.4.3") or r.get("version", "").startswith("3.6")):
                drop["version"] += 1
                continue
            d = users.get(r["name"].lower())
            if d is None or "starttime" not in r or "endtime" not in r:
                drop["no ttyrec dir / times"] += 1
                continue
            if d not in stamps:
                stamps[d] = sorted(t for f in os.listdir(root / "nld-nao-unzipped" / d)
                                   if f.endswith(".ttyrec.bz2") and (t := stamp(f)) is not None)
            start, end = int(r["starttime"]), int(r["endtime"])
            if not any(start - FIVE_MINS <= t <= end for t in stamps[d]):
                drop["no ttyrec"] += 1
                continue
            if int(r["turns"]) <= a.min_turns:
                drop[f"turns <= {a.min_turns}"] += 1
                continue
            if r["name"].lower() in bots:
                drop["bot accounts"] += 1
                continue
            g, cut = group(r), int(r.get("maxlvl", 0)) < 4
            turns[g, cut].append(int(r["turns"]))
            asc[g, cut] += r["death"] == "ascended"
            if cut:
                endlev[g][int(r.get("deathlev", 0))] += 1
    print("dropped before the threshold:", dict(drop))
    for g in ("Val-Dwa", "Val-Hum", "Val-other", "other roles"):
        lo, hi = np.array(turns[g, True]), np.array(turns[g, False])
        if not len(lo) + len(hi):
            continue
        n = len(lo) + len(hi)
        print(f"\n{g}: {n} games (> {a.min_turns} turns, known fate, ttyrec, no bots)")
        print(f"  maxlvl < 4 (cut):  {len(lo)} games = {len(lo) / n:.1%}, ascended {asc[g, True]}; turns: median "
              f"{np.median(lo) if len(lo) else 0:.0f}, p90 {np.percentile(lo, 90) if len(lo) else 0:.0f}, "
              f"share of all turns {lo.sum() / (lo.sum() + hi.sum()):.2%}")
        print(f"  maxlvl >= 4 (kept): {len(hi)} games, ascended {asc[g, False]}; turns median {np.median(hi):.0f}")
        print(f"  the cut games end on Dlvl: " + ", ".join(f"{k}: {v}" for k, v in sorted(endlev[g].items())))
        b = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10 ** 9])
        h = np.histogram(lo, bins=b)[0]
        print("  the cut games by turns: " + ", ".join(f"{b[i] + 1}-{b[i + 1]}: {h[i]}" for i in range(len(h) - 1))
              + f", > {b[-2]}: {h[-1]}")


if __name__ == "__main__":
    main()
