"""Where the bot stands at each depth: experience level, AC, max HP, game turn and key items when it first
reaches each Dlvl, and what separates the games that went on from those that died soon after. Groundwork for
handing laya the bot's strategic mechanics (descend or prepare, what to collect): docs/behaviour-cloning.md.

progress (BALROG, arena-src/nethackers/arena/progress.py) = the best milestone reached, Dlvl or Xp, each valued
by P(a human ascends | reached it); the profile also says which of the two set each game's progress.
  python bot_profile.py <run dir with evidence.json and trace/> [...] [--out results/bot_profile]
-> <out>.txt (tables), <out>.parquet (a row per game and new deepest level)
"""
import argparse
import json
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "arena-src"))
from nethackers.arena.progress import ACHIEVEMENTS  # noqa: E402

HP, MAXHP, DEPTH, AC, XL, TIME, DNUM = 10, 11, 12, 16, 18, 20, 23
# inventory lines; identified names and fixed appearances only (an unidentified cloak of magic resistance is a
# random cloak and is missed)
ITEMS = {
    "excalibur": r"excalibur",
    "reflection": r"shield of reflection|polished silver shield|silver dragon scale|amulet of reflection",
    "magic_res": r"cloak of magic resistance|gray dragon scale|magicbane",
    "dragon_mail": r"dragon scale mail",
    "levitation": r"levitation",
    "speed_boots": r"speed boots",
    "digging": r"wand of digging|pick-axe|dwarvish mattock",
    "unicorn_horn": r"unicorn horn",
    "blindfold": r"blindfold|towel",
    "lizard": r"lizard corpse",
}


def snap(b, inv):
    s = dict(depth=b[DEPTH], dnum=b[DNUM], turn=b[TIME], xl=b[XL], ac=b[AC], maxhp=b[MAXHP], hp=b[HP])
    for k, p in ITEMS.items():
        s[k] = any(re.search(p, x, re.I) for x in inv)
    return s


def game(path):
    lines = subprocess.run(["zstd", "-dcq", path], capture_output=True, check=True).stdout.splitlines()[1:]
    inv, arrivals, xls, deepest, top_xl, b = [], [], [], 0, 0, None
    for line in lines:
        r = json.loads(line)
        inv = r.get("inv", inv)
        b = r.get("b", b)
        if b is None:
            continue
        if b[DEPTH] > deepest:
            deepest = b[DEPTH]
            arrivals.append(snap(b, inv))
        if b[XL] > top_xl:
            top_xl = b[XL]
            xls.append((b[XL], b[DEPTH], b[TIME]))
    return path, len(lines), b[TIME], arrivals, xls, snap(b, inv)


def match(done, results):
    """trace -> arena result: <seed>.jsonl.zst by its seed, older <pid>-<episode> ones by the step count, a tie
    broken by the last game turn (as analyze_traces.games)."""
    by_id = {x["trajectory_id"]: x for x in results}
    used, out = set(), {}
    for path, steps, last_turn, *_ in done:
        stem = Path(path).name.split(".")[0]
        if stem.isdigit():
            out[path] = by_id.get(int(stem))
            continue
        cand = [x for x in results if x["steps"] == steps and x["trajectory_id"] not in used]
        if len(cand) > 1:
            gap = min(abs(x["turns"] - last_turn) for x in cand)
            cand = [x for x in cand if abs(x["turns"] - last_turn) == gap]
        if len(cand) == 1:
            used.add(cand[0]["trajectory_id"])
            out[path] = cand[0]
    return out


def q(s, fmt="{:.0f}"):
    return f"{fmt.format(s.median())} [{fmt.format(s.quantile(.25))}–{fmt.format(s.quantile(.75))}]"


def cause(c):
    return re.sub(r"^killed by (an? |the )?", "", c or "—")[:28]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/bot_profile")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    files, results = [], []
    for run in a.runs:
        files += sorted(str(p) for p in Path(run, "trace").glob("*.jsonl.zst"))
        results += json.load(open(Path(run, "evidence.json")))["results"]
    with Pool(a.workers) as pool:
        done = pool.map(game, files, chunksize=4)
    res = match(done, results)
    rows, games = [], []
    for path, steps, last_turn, arrivals, xls, end in done:
        r = res.get(path)
        if r is None:
            continue
        g = dict(game=path, seed=r["trajectory_id"], died=bool(r.get("cause_of_death")),
                 cause=cause(r.get("cause_of_death")),
                 progress=r["progress"], maxdepth=arrivals[-1]["depth"], maxxl=xls[-1][0],
                 p_dlvl=ACHIEVEMENTS.get(f"Dlvl:{arrivals[-1]['depth']}", 0.0),
                 p_xp=ACHIEVEMENTS.get(f"Xp:{xls[-1][0]}", 0.0), **{f"end_{k}": v for k, v in end.items()})
        g["xl_at"] = {x: (d, t) for x, d, t in xls}
        games.append(g)
        rows += [dict(s, game=path, seed=g["seed"], maxdepth=g["maxdepth"], died=g["died"]) for s in arrivals]
    G, A = pd.DataFrame(games), pd.DataFrame(rows)
    items = list(ITEMS)
    out = [f"bot profile: {len(G)} games of {len(files)} traces ({', '.join(a.runs)})", "",
           f"progress mean {G.progress.mean():.3f}; set by Dlvl {(G.p_dlvl >= G.p_xp).mean():.0%} of games, "
           f"by Xp {(G.p_xp > G.p_dlvl).mean():.0%}; mean of the Dlvl part {G.p_dlvl.mean():.3f}, "
           f"of the Xp part {G.p_xp.mean():.3f}",
           f"at the end: depth {q(G.maxdepth)}, Xp {q(G.maxxl)}, AC {q(G.end_ac)}, max HP {q(G.end_maxhp)}, "
           f"turn {q(G.end_turn)}; died {G.died.mean():.1%}", ""]
    out.append("first arrival at each depth (median [p25–p75]; items: % of games carrying them)")
    out.append(f"{'Dlvl':>4} {'games':>5} {'deeper':>6}  {'Xp':12} {'AC':12} {'max HP':14} {'turn':18} "
               + " ".join(f"{k[:6]:>6}" for k in items))
    for d, s in A.groupby("depth"):
        if len(s) < 10:
            continue
        deeper = (s.maxdepth > d).mean()
        out.append(f"{d:4d} {len(s):5d} {deeper:6.0%}  {q(s.xl):12} {q(s.ac):12} {q(s.maxhp):14} {q(s.turn):18} "
                   + " ".join(f"{s[k].mean():6.0%}" for k in items))
    out += ["", "at first arrival at Dlvl d: games that went 3+ levels deeper vs games that ended (died) within 2",
            f"{'Dlvl':>4} {'group':6} {'n':>4}  {'Xp':12} {'AC':12} {'max HP':14} {'turn':18} "
            + " ".join(f"{k[:6]:>6}" for k in items)]
    for d in (8, 10, 12, 15, 18, 20, 22, 24, 25, 26):
        s = A[A.depth == d]
        for name, part in (("on", s[s.maxdepth >= d + 3]), ("ended", s[s.maxdepth < d + 3])):
            if len(part) >= 5:
                out.append(f"{d:4d} {name:6} {len(part):4d}  {q(part.xl):12} {q(part.ac):12} {q(part.maxhp):14} "
                           f"{q(part.turn):18} " + " ".join(f"{part[k].mean():6.0%}" for k in items))
    out += ["", "first reaching each experience level: Dlvl and turn then (median [p25–p75])",
            f"{'Xp':>3} {'games':>5}  {'Dlvl':12} {'turn':18}"]
    for x in range(1, 31):
        at = [g["xl_at"][x] for g in games if x in g["xl_at"]]
        if len(at) >= 10:
            d, t = pd.Series([v[0] for v in at]), pd.Series([v[1] for v in at])
            out.append(f"{x:3d} {len(at):5d}  {q(d):12} {q(t):18}")
    out += ["", "causes of death by the deepest level reached (top 5, share of deaths in the band)"]
    D = G[G.died]
    for lo, hi in ((1, 9), (10, 19), (20, 24), (25, 27), (28, 60)):
        s = D[(D.maxdepth >= lo) & (D.maxdepth <= hi)]
        if len(s):
            top = s.cause.value_counts().head(5)
            out.append(f"Dlvl {lo}–{hi}: {len(s)} deaths; " + ", ".join(f"{c} {n / len(s):.0%}" for c, n in top.items()))
    text = "\n".join(out) + "\n"
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".txt").write_text(text, encoding="utf-8")
    A.to_parquet(a.out + ".parquet", index=False)


if __name__ == "__main__":
    main()
