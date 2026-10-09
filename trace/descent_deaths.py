"""Do the bot's deaths follow its descents? (What to branch on next, after the emergency options at risk alarms
turned out to leave ~no headroom: docs/branches.md.)

From the recorded games (laya_trace records: blstats per bot action): every change of Dlvl, the HP fraction just
before it, and whether the hero died within 100 / 300 turns of arriving. Reports how many deaths come soon after
an arrival, and the death rate after a descent by the HP fraction it was made with -- if low-HP descents are much
deadlier, "rest before going down" is a decision worth branching on.
  python descent_deaths.py --root R [--out results/descent_deaths.txt]
"""
import argparse
import json
import re
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from build_laya_data import DEPTH, HP, MAXHP, TIME

B = re.compile(rb'"b":\[([^\]]*)\]')


def game(path):
    """Arrivals on a new Dlvl: (turn, depth before, depth after, HP fraction just before)."""
    lines = subprocess.run(["zstdcat", str(path)], capture_output=True, check=True).stdout.splitlines()[1:]
    out, prev, last = [], None, None
    for line in lines:
        m = B.search(line)
        if not m:
            continue
        b = [int(v) for v in m.group(1).split(b",")]
        if b[MAXHP] <= 0:
            continue
        if prev is not None and b[DEPTH] != prev[DEPTH]:
            out.append((b[TIME], prev[DEPTH], b[DEPTH], prev[HP] / prev[MAXHP]))
        prev = last = b
    return path.name, out, last[TIME] if last else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="a run dir: results.json + trace/*.jsonl.zst")
    ap.add_argument("--out", default="results/descent_deaths.txt")
    a = ap.parse_args()
    root = Path(a.root)
    res = {str(r["trajectory_id"]): r for r in json.load(open(root / "results.json"))}
    files = sorted((root / "trace").glob("*.jsonl.zst"))
    with Pool(31) as pool:
        got = pool.map(game, files, chunksize=4)
    # trace file <seed>-<ep>.jsonl.zst -> results by the last turn (games() in analyze_traces matches the same way)
    by_turns = {}
    for r in res.values():
        by_turns.setdefault(r["turns"], []).append(r)
    rows, deaths = [], []
    for name, arr, end in got:
        cand = by_turns.get(end, [])
        died = len(cand) == 1 and bool(cand[0].get("cause_of_death"))
        if died:
            last_arrival = max((t for t, *_ in arr), default=0)
            deaths.append(end - last_arrival)
        for t, d0, d1, hp in arr:
            rows.append({"game": name, "turn": t, "down": d1 > d0, "hp": hp, "died": died,
                         "dead_in": (end - t) if died else np.inf})
    df = pd.DataFrame(rows)
    deaths = np.array(deaths)
    lines = [f"{len(got)} games, {len(deaths)} deaths matched to a trace; {len(df)} Dlvl changes "
             f"({int(df.down.sum())} down)",
             "turns from the last Dlvl change to death: " + ", ".join(
                 f"<= {k}: {np.mean(deaths <= k):.0%}" for k in (10, 30, 100, 300)) + f"; median {np.median(deaths):.0f}"]
    dn = df[df.down].copy()
    dn["bucket"] = pd.cut(dn.hp, [0, 0.33, 0.5, 0.75, 0.9, 1.01], right=False)
    for h in (100, 300):
        dn[f"d{h}"] = dn.dead_in <= h
    t = dn.groupby("bucket", observed=True).agg(descents=("hp", "size"), dead100=("d100", "mean"),
                                                 dead300=("d300", "mean"))
    lines += ["descents by HP fraction when going down: share dead within 100 / 300 turns of arriving",
              t.to_string(float_format=lambda v: f"{v:.3f}")]
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
