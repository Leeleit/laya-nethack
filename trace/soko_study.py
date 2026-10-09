"""Why does the bot's SOKOBAN_TRIP (the authors' "solve Sokoban before diving", always on) lose games? The
paired A/B (bot_ab.py) only measured the end result; this takes the flag's games apart.

  A. phases of every flag game (its trace): reached Sokoban? (dnum 4) at which Xp / AC / HP / turn / deepest Dlvl;
     how many of its 4 levels, the prize level (levno 1), turns inside, hunger there (Weak or worse), the rings /
     wands / amulets / bags gained inside, whether the trip ended back in the Dungeons; where the game ended:
     before Sokoban / inside / after it before Medusa / later (with causes), and the paired base game's outcome --
     per phase, and for the games that finished Sokoban (selected: they survived the trip, so read as a ceiling).
     Later use of the loot: levitation ("float") messages, on the castle level especially.
  B. is "the flag helps here" predictable at the branch point (ab_points.py)? Features of the state there (HP, max HP,
     Dlvl, XL, AC, turn); cross-fitted gradient boosting of the paired progress difference; the policy "flag where
     predicted > 0" against never / always / the oracle, with a bootstrap over seeds.
  C. humans (NLD-NAO xlogfiles, achieve bits): share of Valkyrie games with the Sokoban prize (0x400) and Medusa
     killed (0x800) by outcome -- do the humans who get far do Sokoban?
  python soko_study.py --arm-endgame E1 E2 --base-endgame B1 B2 --points P1 P2 --nld DIR --out results/soko_study
-> <out>.txt, <out>.parquet (part A, a row per flag game)
"""
import argparse
import json
import re
import subprocess
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from bot_profile import AC, DEPTH, DNUM, HP, MAXHP, TIME, XL

LEVNO, HUNGER = 24, 21          # blstats: dungeon level number, hunger state (3 weak, 4 fainting)
LOOT = re.compile(r"\b(ring|wand|amulet|bag)\b", re.I)


def phases(path):
    lines = subprocess.run(["zstd", "-dcq", path], capture_output=True, check=True).stdout.splitlines()[1:]
    b, inv, deepest = None, [], 0
    o = dict(game=path, soko_levels=0, prize_level=False, soko_turns=0, weak_in_soko=0, left_soko=False,
             entry_turn=None, entry_xl=None, entry_ac=None, entry_hp=None, entry_maxhp=None, entry_deepest=None,
             loot_gained="", float_msgs=0, float_after_soko=0)
    levels, inv_in, inv_out, in_soko, last_soko_turn = set(), None, None, False, None
    for line in lines:
        r = json.loads(line)
        b = r.get("b", b)
        inv = r.get("inv", inv)
        if b is None:
            continue
        m = r.get("m", "")
        if "float" in m.lower():
            o["float_msgs"] += 1
            if levels and not in_soko:
                o["float_after_soko"] += 1
        if b[DNUM] == 4:
            if not levels:
                o.update(entry_turn=b[TIME], entry_xl=b[XL], entry_ac=b[AC], entry_hp=b[HP], entry_maxhp=b[MAXHP],
                         entry_deepest=deepest)
                inv_in = list(inv)
            levels.add(b[LEVNO])
            in_soko, last_soko_turn = True, b[TIME]
            if b[HUNGER] >= 3:
                o["weak_in_soko"] += 1
        else:
            if in_soko and b[DNUM] == 0:
                o["left_soko"], inv_out = True, list(inv)
            in_soko = False
            if b[DNUM] in (0, 1):
                deepest = max(deepest, b[DEPTH])
    if levels:
        o.update(soko_levels=len(levels), prize_level=min(levels) == 1, soko_turns=last_soko_turn - o["entry_turn"])
        gained = Counter(x for x in (inv_out or inv) if LOOT.search(x)) - Counter(x for x in inv_in if LOOT.search(x))
        o["loot_gained"] = "; ".join(x[:50] for x in gained.elements())
    o["in_soko_at_end"] = in_soko
    return o


def part_a(arm_eg, base_eg, workers):
    with Pool(workers) as pool:
        rows = pool.map(phases, list(arm_eg.game), chunksize=4)
    A = pd.DataFrame(rows).merge(arm_eg, on="game")
    base = base_eg.set_index("seed")
    for c in ("progress", "died", "section", "reached_castle", "crossed_castle", "cause"):
        A["base_" + c] = A.seed.map(base[c])
    A["entered"] = A.soko_levels > 0
    A["phase"] = np.select(
        [~A.entered & A.died & (A.section == "before Medusa"), ~A.entered, A.in_soko_at_end,
         A.section == "before Medusa"],
        ["died before reaching Sokoban", "never entered Sokoban (game went on)", "ended inside Sokoban",
         "after Sokoban, before Medusa"], default="after Sokoban, Medusa or deeper")
    out = [f"A. flag games: {len(A)}; entered Sokoban {A.entered.mean():.1%}, all 4 levels {(A.soko_levels >= 4).mean():.1%},"
           f" prize level {A.prize_level.mean():.1%}, left it back to the Dungeons {A.left_soko.mean():.1%}"]
    e = A[A.entered]
    if len(e):
        out += [f"  at entry (median): turn {e.entry_turn.median():.0f}, Xp {e.entry_xl.median():.0f}, AC "
                f"{e.entry_ac.median():.0f}, HP {e.entry_hp.median():.0f}/{e.entry_maxhp.median():.0f}, deepest Dlvl "
                f"{e.entry_deepest.median():.0f}; turns inside median {e.soko_turns.median():.0f} "
                f"[p75 {e.soko_turns.quantile(.75):.0f}]; weak with hunger inside {(e.weak_in_soko > 0).mean():.1%}",
                "  levels reached: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(e.soko_levels).items())),
                "  loot gained inside (top): " + "; ".join(f"{k} {v}" for k, v in Counter(
                    w for s in e.loot_gained for w in s.split("; ") if w).most_common(12))]
    out += ["", "where the flag games end, with the paired base game (progress arm vs base, base reached the castle):"]
    for ph, g in A.groupby("phase"):
        out.append(f"  {ph:38s} {len(g):5d} ({len(g) / len(A):.1%})  progress {g.progress.mean():.3f} vs base "
                   f"{g.base_progress.mean():.3f}; castle {g.reached_castle.mean():.1%} vs base "
                   f"{g.base_reached_castle.mean():.1%}")
        out.append("      causes: " + "; ".join(f"{c} {n}" for c, n in Counter(g.cause[g.died]).most_common(8)))
    full = A[A.prize_level & A.left_soko]
    if len(full):
        out += ["", f"finished Sokoban (prize level, came back; selected: survived the trip) {len(full)}: progress "
                    f"{full.progress.mean():.3f} vs their base games {full.base_progress.mean():.3f}; castle "
                    f"{full.reached_castle.mean():.1%} vs {full.base_reached_castle.mean():.1%}; crossed "
                    f"{full.crossed_castle.mean():.1%} vs {full.base_crossed_castle.mean():.1%}; levitation "
                    f"('float') after Sokoban in {(full.float_after_soko > 0).mean():.1%} of them"]
    out += [f"starvation: flag games {(A.cause == 'died of starvation').sum()}, of them weak inside Sokoban "
            f"{((A.cause == 'died of starvation') & (A.weak_in_soko > 0)).sum()}, ended inside "
            f"{((A.cause == 'died of starvation') & A.in_soko_at_end).sum()}; base games "
            f"{(A.base_cause == 'died of starvation').sum()}"]
    return A, out


STATE = re.compile(r"HP (\d+)/(\d+);.*?Dlvl (\d+), XL (\d+), AC (-?\d+), turn (\d+)", re.S)


def part_b(points, rng):
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    for p in points:
        m = STATE.search(p["state"])
        if m:
            rows.append(dict(zip(["hp", "maxhp", "dlvl", "xl", "ac", "turn"], map(int, m.groups())), seed=p["seed"],
                             diff=p["arm"]["progress"] - p["base"]["progress"],
                             dcastle=int(p["arm"].get("reached_castle", 0)) - int(p["base"].get("reached_castle", 0))))
    P = pd.DataFrame(rows)
    X, y = P[["hp", "maxhp", "dlvl", "xl", "ac", "turn"]].to_numpy(float), P["diff"].to_numpy()
    fold = rng.integers(0, 5, len(P))
    pred = np.zeros(len(P))
    for f in range(5):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
        pred[fold == f] = m.fit(X[fold != f], y[fold != f]).predict(X[fold == f])
    pick = pred > 0
    pol = {"never (the bot as is)": np.zeros(len(P)), "always (the flag)": y, "model: flag where predicted > 0":
           np.where(pick, y, 0), "oracle": np.maximum(y, 0)}
    boot = [rng.integers(0, len(P), len(P)) for _ in range(500)]
    out = ["", f"B. branch points {len(P)}: the flag better (progress) {(y > 0).mean():.1%}, worse {(y < 0).mean():.1%}; "
               f"state there median: Dlvl {P.dlvl.median():.0f}, XL {P.xl.median():.0f}, AC {P.ac.median():.0f}, "
               f"HP {P.hp.median():.0f}/{P.maxhp.median():.0f}, turn {P.turn.median():.0f}",
           "  mean progress gain over never, 5-fold cross-fitted, 95% bootstrap over points:"]
    for k, v in pol.items():
        lo, hi = np.percentile([v[t].mean() for t in boot], [2.5, 97.5])
        out.append(f"    {k:34s} {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]" +
                   (f"  (flag at {pick.mean():.1%} of points)" if k.startswith("model") else ""))
    for c in ("xl", "ac", "maxhp", "dlvl", "turn"):
        q = pd.qcut(P[c], 4, duplicates="drop")
        out.append(f"  by {c}: " + "; ".join(f"{iv}: {g['diff'].mean():+.3f} (n {len(g)})"
                                              for iv, g in P.groupby(q, observed=True)))
    return out


def part_c(nld):
    from nao_corridor import xlog
    recs = []
    for xf in sorted(Path(nld).glob("xlogfile.*")):
        if xf.name.endswith("dev"):
            continue
        for r in xlog(str(xf)):
            if r.get("role") == "Val" and "achieve" in r and r.get("version", "").startswith(("3.4.3", "3.6")):
                try:
                    recs.append(dict(race=r.get("race"), asc=r.get("death") == "ascended", maxlvl=int(r["maxlvl"]),
                                     ach=int(r["achieve"], 0), turns=int(r.get("turns", 0))))
                except ValueError:
                    pass
    H = pd.DataFrame(recs)
    H["soko"], H["medusa"] = (H.ach & 0x400) > 0, (H.ach & 0x800) > 0
    grp = np.select([H.asc, H.maxlvl >= 25, H.maxlvl >= 10], ["ascended", "died, Dlvl 25+", "died, Dlvl 10-24"],
                    default="died, Dlvl < 10")
    out = ["", f"C. humans, NLD-NAO Valkyries with an achieve field: {len(H)} games"]
    for gname, g in H.groupby(grp):
        out.append(f"  {gname:18s} {len(g):6d}  Sokoban prize {g.soko.mean():.1%}  Medusa killed {g.medusa.mean():.1%}"
                   f"  (Dwa {g[g.race == 'Dwa'].soko.mean():.1%}, Hum {g[g.race == 'Hum'].soko.mean():.1%})")
    deep = H[H.maxlvl >= 25]
    out.append(f"  reached Dlvl 25+: with the prize ascended {deep[deep.soko].asc.mean():.1%} (n {deep.soko.sum()}), "
               f"without {deep[~deep.soko].asc.mean():.1%} (n {(~deep.soko).sum()})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-endgame", nargs="+", required=True)
    ap.add_argument("--base-endgame", nargs="+", required=True)
    ap.add_argument("--points", nargs="+", required=True)
    ap.add_argument("--nld", default="/data/laya/nethack/nld/nld-nao")
    ap.add_argument("--out", default="/data/laya/nethack/results/soko_study")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    arm = pd.concat([pd.read_parquet(f) for f in a.arm_endgame])
    base = pd.concat([pd.read_parquet(f) for f in a.base_endgame])
    A, out = part_a(arm, base, a.workers)
    out += part_b([json.loads(x) for f in a.points for x in open(f, encoding="utf-8")], rng)
    out += part_c(a.nld)
    text = "\n".join(out) + "\n"
    print(text)
    Path(a.out + ".txt").write_text(text, encoding="utf-8")
    A.to_parquet(a.out + ".parquet", index=False)


if __name__ == "__main__":
    main()
