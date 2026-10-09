"""Input pace of NAO accounts from their ttyrecs (follows bot_accounts.py): a person stops to read and
think, a bot does not (or pauses all alike). Per account, up to --files ttyrecs of >= 20 KB (random, seed 0):
frames' 12-byte headers (sec, usec, len); gaps between frames >= 0.02 s count as input events. Reported: events,
median gap, share of gaps > 5 s and > 30 s, share of the session's time spent in gaps > 5 s (idle), the gaps'
p90 / p99. Accounts: everyone with >= --min train games in the dataset plus --ref random players with 10..min-1.
  python ttyrec_pace.py human_data14d [--min 50] [--ref 60] [--files 15] [--out ../results/ttyrec_pace_14d.txt]
"""
import argparse
import bz2
import collections
import json
import os
import random
import struct
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = "/data/laya/nethack/nld/nld-nao/nld-nao-unzipped"


def gaps(path):
    data = bz2.open(path).read()
    t, o = [], 0
    while o + 12 <= len(data):
        sec, usec, n = struct.unpack_from("<III", data, o)
        t.append(sec + usec / 1e6)
        o += 12 + n
    g = np.diff(np.array(t))
    return g[(g >= 0.02) & (g < 3600)]  # >= 1 h: the player left (or a clock jump), not play


def account(job):
    name, dirname, k = job
    files = [os.path.join(ROOT, dirname, f) for f in sorted(os.listdir(os.path.join(ROOT, dirname)))
             if f.endswith(".ttyrec.bz2") and os.path.getsize(os.path.join(ROOT, dirname, f)) >= 20_000]
    random.Random(0).shuffle(files)
    gs = []
    for f in files[:k]:
        try:
            gs.append(gaps(f))
        except (OSError, EOFError, struct.error):
            pass
    g = np.concatenate(gs) if gs else np.zeros(0)
    if len(g) < 100:
        return name, None
    return name, {"files": len(gs), "events": len(g), "median": np.median(g), "p90": np.percentile(g, 90),
                  "p99": np.percentile(g, 99), ">5s": (g > 5).mean(), ">30s": (g > 30).mean(),
                  "idle": g[g > 5].sum() / g.sum()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--min", type=int, default=50)
    ap.add_argument("--ref", type=int, default=60)
    ap.add_argument("--files", type=int, default=15)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    games = collections.defaultdict(set)
    for line in open(Path(a.data, f"{a.split}.jsonl"), encoding="utf-8"):
        game = json.loads(line)["id"].split("#", 1)[0]
        games[game.split("@", 1)[0].lower()].add(game)
    n = {p: len(v) for p, v in games.items()}
    dirs = {d.lower(): d for d in os.listdir(ROOT)}
    big = sorted((p for p in n if n[p] >= a.min), key=lambda p: -n[p])
    mid = sorted(p for p in n if 10 <= n[p] < a.min)
    ref = random.Random(0).sample(mid, min(a.ref, len(mid)))
    jobs = [(p, dirs[p], a.files) for p in big + ref if p in dirs]
    with Pool(a.workers) as pool:
        res = dict(pool.imap_unordered(account, jobs))
    head = f"{'player':16s} {'train':>5s} {'files':>5s} {'events':>7s} {'median':>6s} {'p90':>5s} {'p99':>6s} " \
           f"{'>5s':>6s} {'>30s':>6s} {'idle':>5s}"
    out = [f"ttyrec pace: {len(big)} players with >= {a.min} train games, {len(ref)} random with 10-{a.min - 1}; "
           f"no ttyrec dir: {[p for p in big + ref if p not in dirs]}; too few events: "
           f"{[p for p in res if res[p] is None]}"]
    for title, group in (("== >= min", big), ("== reference", ref)):
        out += [title, head]
        for p in group:
            r = res.get(p)
            if r:
                out.append(f"{p:16s} {n[p]:5d} {r['files']:5d} {r['events']:7d} {r['median']:6.2f} {r['p90']:5.1f} "
                           f"{r['p99']:6.1f} {r['>5s']:6.3f} {r['>30s']:6.3f} {r['idle']:5.2f}")
        vals = [res[p] for p in group if res.get(p)]
        out.append("median   " + "  ".join(f"{k} {np.median([v[k] for v in vals]):.3f}"
                                           for k in ("median", "p90", "p99", ">5s", ">30s", "idle")))
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
