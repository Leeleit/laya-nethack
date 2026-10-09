"""Map around the hero vs no map, humans. Variants of every state, scored at inference:
  base   -- as built (crop around the hero + the whole level map when the level is larger than the crop; a small level
            is given once, as the level map, with SAME_AS_LEVEL in the crop's place)
  around -- only the map around the hero: the crop, or the small level map put in the crop's place; no whole level map
  none   -- no map at all
  python map_variants.py make <val.jsonl> <out dir>
  python map_variants.py score <val.jsonl> <base.parquet> <other.parquet> ...  (log loss per question, paired by player)
"""
import re
import sys
import json

import numpy as np
import pandas as pd

from state_fmt import SAME_AS_LEVEL

CROP = re.compile(r"Map around the hero \(@\):\n(.*?)\n(?=Inventory|Level map:|\Z)", re.S)
LEVEL = re.compile(r"\nLevel map:\n(.*)\Z", re.S)


def around(s):
    level = LEVEL.search(s)
    s = LEVEL.sub("", s)
    if SAME_AS_LEVEL in s and level:
        s = s.replace(SAME_AS_LEVEL, level.group(1).rstrip("\n"))
    return s


def none(s):
    return CROP.sub("", LEVEL.sub("", s))


def make(val, out):
    rows = [json.loads(line) for line in open(val)]
    for name, f in (("base", lambda s: s), ("around", around), ("none", none)):
        with open(f"{out}/{name}.jsonl", "w", newline="\n") as fo:
            for r in rows:
                fo.write(json.dumps({**r, "state": f(r["state"])}) + "\n")


def nll(val, pq):
    gold = {}
    for line in open(val):
        r = json.loads(line)
        gold[r["id"]] = r["gold"]
    df = pd.read_parquet(pq)
    out = {}
    for i, q, z in zip(df.id, df.question, df.logits):
        z = np.asarray(z, float)
        p = np.exp(z - z.max())
        p /= p.sum()
        out[(i, q)] = -np.log(max(p[int(gold[i][q]["label"])], 1e-9))
    return pd.Series(out)


def score(val, ref, rest):
    base = nll(val, ref)
    qs = np.array([q for _, q in base.index])
    codes, uniq = pd.factorize(np.array([i.split("@")[0] for i, _ in base.index]))
    rng = np.random.default_rng(0)
    for pq in [ref] + rest:
        d = nll(val, pq).reindex(base.index).to_numpy()
        for q in sorted(set(qs)):
            m = qs == q
            diff = d[m] - base.to_numpy()[m]
            sums = np.bincount(codes[m], weights=diff, minlength=len(uniq))
            cnts = np.bincount(codes[m], minlength=len(uniq))
            bs = []
            for _ in range(1000):
                k = rng.integers(0, len(uniq), len(uniq))
                bs.append(sums[k].sum() / max(cnts[k].sum(), 1))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"{pq.split('/')[-1]:14s} {q:5s} log loss {d[m].mean():.4f}  minus base {diff.mean():+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(*sys.argv[2:4])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4:])
