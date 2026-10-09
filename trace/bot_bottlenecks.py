"""Where the bot loses its games: for every game of the given runs, the level band it ended on, what killed it,
the bot's strategy then, how long it had been on that level, its HP and the monsters around in its last turns,
and the unidentified potions / scrolls / rings / wands it carried (the castle's lift plan tries unknown ones).
The progress it would gain by one more level (BALROG table) ranks the bands: where an extra level is worth most.
  python bot_bottlenecks.py <run dir with evidence.json and trace/> [...] [--out results/bot_bottlenecks] [--workers N]
-> <out>.txt, <out>.parquet (a row per game)
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from bot_profile import AC, DEPTH, HP, MAXHP, TIME, XL, cause, match

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "arena-src"))
from nethackers.arena.progress import ACHIEVEMENTS  # noqa: E402

BANDS = [(1, 4, "Dlvl 1-4 (grind)"), (5, 12, "Dlvl 5-12"), (13, 20, "Dlvl 13-20"), (21, 24, "Dlvl 21-24 (Medusa)"),
         (25, 29, "Dlvl 25-29 (castle)"), (30, 99, "Dlvl 30+ (Gehennom)")]
UNKNOWN = {  # unidentified: an appearance, no "of <name>" (named ones carry "called"/"named": still unknown)
    "potion": re.compile(r"\bpotions?\b(?! of )", re.I),
    "scroll": re.compile(r"\bscrolls? labeled\b", re.I),
    "ring": re.compile(r"\b\w+ ring\b(?! of )", re.I),
    "wand": re.compile(r"\b\w+ wand\b(?! of )", re.I),
}


def band(d):
    return next(name for lo, hi, name in BANDS if lo <= d <= hi)


def top(strategy):
    """the innermost named strategy of the bot's stack, without Python's <locals>/<lambda> noise"""
    if not strategy:
        return "—"
    parts = [p for p in strategy[-1].split(".") if not p.startswith("<")]
    return ".".join(parts[-2:])[:40]


def game(path):
    lines = subprocess.run(["zstd", "-dcq", path], capture_output=True, check=True).stdout.splitlines()[1:]
    b, inv, stack, mons, arrived, hist = None, [], [], [], {}, []
    for line in lines:
        r = json.loads(line)
        inv = r.get("inv", inv)
        stack = r.get("s", stack)
        if "mon" in r:
            mons = r["mon"]
        if "b" in r:
            b = r["b"]
            arrived.setdefault(b[DEPTH], b[TIME])
            hist.append((b[TIME], b[HP], b[MAXHP]))
    if b is None:
        return path, len(lines), 0, None
    t = b[TIME]
    hp20 = next((h / m for tt, h, m in reversed(hist) if tt <= t - 20 and m), None)
    unknown = {k: sum(bool(p.search(x)) for x in inv) for k, p in UNKNOWN.items()}
    end = dict(depth=b[DEPTH], maxdepth=max(arrived), xl=b[XL], ac=b[AC], maxhp=b[MAXHP], turn=t,
               turns_on_level=t - arrived[b[DEPTH]], hp20=hp20, strategy=top(stack),
               monsters=", ".join(sorted({m[0] for m in mons}))[:80], **{f"unk_{k}": v for k, v in unknown.items()})
    return path, len(lines), t, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/bot_bottlenecks")
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
        d = end["maxdepth"]
        rows.append(dict(end, game=path, seed=r["trajectory_id"], progress=r["progress"],
                         cause=cause(r.get("cause_of_death")), died=bool(r.get("cause_of_death")), band=band(d),
                         gain=ACHIEVEMENTS.get(f"Dlvl:{d + 1}", 0.0) - ACHIEVEMENTS.get(f"Dlvl:{d}", 0.0)))
    G = pd.DataFrame(rows)
    out = [f"bot: where its {len(G)} games end ({', '.join(a.runs)}); progress mean {G.progress.mean():.3f}",
           "gain = progress of one more level (BALROG table) - the end level's; 'mass' = games x gain: where saving",
           "a game is worth most", ""]
    out.append(f"{'band':22} {'games':>6} {'share':>6} {'progress':>8} {'gain':>6} {'mass':>6}  "
               f"{'turns on level':16} {'HP 20 turns before':18}")
    for _, _, name in BANDS:
        s = G[G.band == name]
        if not len(s):
            continue
        out.append(f"{name:22} {len(s):6d} {len(s) / len(G):6.1%} {s.progress.mean():8.3f} {s.gain.mean():6.3f} "
                   f"{s.gain.sum():6.1f}  {s.turns_on_level.median():7.0f} [{s.turns_on_level.quantile(.75):.0f}]"
                   f"    {s.hp20.median():6.0%}")
    for _, _, name in BANDS:
        s = G[G.band == name]
        if len(s) < 20:
            continue
        out += ["", f"== {name}: {len(s)} games, end Dlvl {s.maxdepth.median():.0f}, Xp {s.xl.median():.0f}, "
                    f"AC {s.ac.median():.0f}, max HP {s.maxhp.median():.0f}, turn {s.turn.median():.0f}",
                "  causes:     " + "; ".join(f"{c} {n / len(s):.0%}" for c, n in Counter(s.cause).most_common(8)),
                "  strategy:   " + "; ".join(f"{c} {n / len(s):.0%}" for c, n in Counter(s.strategy).most_common(6)),
                "  cause x strategy: " + "; ".join(f"{c} / {t} {n}" for (c, t), n in
                                                  Counter(zip(s.cause, s.strategy)).most_common(6)),
                "  unknown items carried (mean / share with any): " + ", ".join(
                    f"{k} {s[f'unk_{k}'].mean():.1f} / {(s[f'unk_{k}'] > 0).mean():.0%}" for k in UNKNOWN)]
    text = "\n".join(out) + "\n"
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".txt").write_text(text, encoding="utf-8")
    G.to_parquet(a.out + ".parquet", index=False)


if __name__ == "__main__":
    main()
