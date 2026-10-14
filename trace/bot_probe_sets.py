"""Inputs of the bot-risk probe (bot_data2, built with build_v2.py --terse), so the GPU predicts only what the reading uses.
  pred: bot_data2p/{val,test}.jsonl -- test: every row of Dlvl 13+ and 1/--share of Dlvl 1-12 (crc32 of the id);
        val: every row of Dlvl 13+.
  fam:  bot_data2p/fam.jsonl -- stride-sampled risk states of train games that the train set did not take (KEEP draw
        replayed from build_v2: rng = Random(path), one draw per risk state of a train game), Dlvl 13+, 25-29 at
        1/--share (crc32 of the id). Their counterpart "new" is the test rows of the same bands. Every replayed game
        must give exactly the ids train.jsonl has for it, else the run stops.
  python bot_probe_sets.py pred [--data bot_data2] [--out bot_data2p] [--share 5]
  python bot_probe_sets.py fam [--root <runs>/val1 <runs>/val2 <runs>/val3] [--workers 14]
val1 games (run name val-dwa-law-fem-100000-*) were BC training screens of bc-v1: the probe decides on val2/val3.
"""
import argparse
import collections
import json
import os
import random
import re
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

DLVL = re.compile(r"Dlvl (\d+)")
ROOT = "/data/laya/nethack/nethackers/runs"


def band(dl):
    return "1-12" if dl <= 12 else "25-29" if 25 <= dl <= 29 else "13+"


def keep_share(rid, share):
    return zlib.crc32(rid.encode()) % share == 0


def pred(a):
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for split in ("val", "test"):
        n = collections.Counter()
        with open(Path(a.data, f"{split}.jsonl"), encoding="utf-8") as f, \
                open(out / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as g:
            for line in f:
                rid = line.split('"id": "', 1)[1].split('"', 1)[0]
                dl = int(DLVL.search(line).group(1))
                deep = dl >= 13
                if deep or (split == "test" and keep_share(rid, a.share)):
                    g.write(line)
                    n["13+" if deep else "1-12"] += 1
        print(f"{split}: {dict(n)} rows -> {out / f'{split}.jsonl'}", flush=True)


def fam_game(job, share):
    import build_v2 as B
    path, res = job
    if B.split_of(path) != "train":
        return None
    import subprocess
    rng = random.Random(path)
    lines = subprocess.run(["zstdcat", path], capture_output=True, check=True).stdout.splitlines()[1:]
    if len(lines) != res["steps"] or not (res["status"] == "completed" and res["cause_of_death"]):
        return None
    taken, rows, k, deepest = [], [], 0, 0
    for _, r, back, msgs, inv in B.walk(lines):
        b = r["b"]
        turn = b[B.TIME]
        deepest = max(deepest, b[B.DEPTH])
        strat = r["s"][-1] if r["s"] else "<base>"
        hp_frac = b[B.HP] / max(b[B.MAXHP], 1)
        if not (hp_frac < 0.8 or strat not in B.CALM or b[B.DEPTH] > 4):
            continue
        k += 1
        bk = B.bucket(res["turns"] - turn)
        rid = f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}/{os.path.basename(path)}#{turn}"
        took = rng.random() < B.KEEP[bk]
        if took:
            taken.append(rid)
            continue
        if k % B.STRIDE or b[B.DEPTH] < 13 or (25 <= b[B.DEPTH] <= 29 and not keep_share(rid, share)):
            continue
        text = B.terse_text(res["identity"], r["s"], r, back, list(msgs), inv, deepest)
        rows.append({"id": rid, "state": text, "questions": {"ttd": B.QUESTION_TERSE},
                     "gold": {"ttd": {"probabilities": {str(i): float(i == bk) for i in range(len(B.LEVELS))},
                                      "label": str(bk)}}})
    return taken, rows


def fam(a):
    from analyze_traces import games
    want = collections.defaultdict(set)
    with open(Path(a.data, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            rid = line.split('"id": "', 1)[1].split('"', 1)[0]
            want[rid.split("#", 1)[0]].add(rid)
    jobs = [j for root in a.root for j in games(root)]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} games, {len(want)} train games in {a.data}/train.jsonl", flush=True)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    n, games_n, bad = collections.Counter(), 0, 0
    with Pool(a.workers) as pool, open(out / "fam.jsonl", "w", encoding="utf-8", newline="\n") as g:
        for res in pool.imap_unordered(partial(fam_game, share=a.share), jobs, chunksize=4):
            if res is None:
                continue
            taken, rows = res
            if not taken and not rows:
                continue
            gid = (taken or [r["id"] for r in rows])[0].split("#", 1)[0]
            if set(taken) != want.get(gid, set()):
                bad += 1
                continue
            games_n += 1
            for r in rows:
                g.write(json.dumps(r) + "\n")
                n[band(int(DLVL.search(r["state"]).group(1)))] += 1
    print(f"fam: {games_n} train games replayed, {bad} not matching train.jsonl, rows {dict(n)} -> {out / 'fam.jsonl'}")
    if bad:
        raise SystemExit("the KEEP replay does not reproduce train.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pred", "fam"))
    ap.add_argument("--data", default="bot_data2")
    ap.add_argument("--out", default="bot_data2p")
    ap.add_argument("--share", type=int, default=5)
    ap.add_argument("--root", nargs="+", default=[f"{ROOT}/val1", f"{ROOT}/val2", f"{ROOT}/val3"])
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="fam: only the first N games (a trial run)")
    a = ap.parse_args()
    if a.cmd == "pred":
        pred(a)
    else:
        fam(a)


if __name__ == "__main__":
    main()
