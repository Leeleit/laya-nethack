"""laya playing by itself (laya_play.py results) against the bot (an arena run's results.json): game
length, depth reached, the arena's progress score, how games end and the most common causes of death.
  python bc_compare.py <laya play dir> [--bot <run>/results.json] [--out results/bc_compare.txt]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

BOT = "/data/laya/nethack/nethackers/runs/val1/val-dwa-law-fem-100000-1000/results.json"


def summary(name, games):
    t = np.array([g["turns"] for g in games])
    d = np.array([g["max_depth"] for g in games])
    pr = np.array([g.get("progress") or 0 for g in games])
    q = lambda v: f"{np.percentile(v, 25):.0f} / {np.median(v):.0f} / {np.percentile(v, 75):.0f}"
    lines = [f"{name}: {len(games)} games",
             f"  turns       p25 / median / p75  {q(t)}",
             f"  max depth   p25 / median / p75  {q(d)};  reached Dlvl >= 5: {np.mean(d >= 5):.0%}, >= 10: {np.mean(d >= 10):.0%}",
             f"  progress    mean {pr.mean():.3f}, median {np.median(pr):.3f}",
             "  causes: " + "; ".join(f"{c} {v / len(games):.0%}" for c, v in collections.Counter(
                 (g.get("cause") or g.get("cause_of_death") or f"no death ({g.get('end_status')})") for g in games
             ).most_common(6))]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("play")
    ap.add_argument("--bot", default=BOT)
    ap.add_argument("--out", default="results/bc_compare.txt")
    a = ap.parse_args()
    laya = [json.loads(p.read_text()) for p in sorted(Path(a.play, "results").glob("*.json"))]
    bot = json.load(open(a.bot))
    lines = (summary(f"laya ({a.play})", laya) if laya else [f"laya: no games in {a.play}"]) + \
        summary(f"bot ({a.bot})", bot)
    if laya:
        cmds = collections.Counter()
        for g in laya:
            cmds.update(g.get("commands", {}))
        n = sum(cmds.values())
        lines.append("laya's commands: " + ", ".join(f"{c} {v / n:.1%}" for c, v in cmds.most_common()))
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
