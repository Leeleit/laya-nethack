"""Check of finetune_fast.load_items' token cache: items from the cache must equal fresh tokenisation
field for field (ids as int32 arrays), in file order, for the file itself, a subset of its lines and --multi.
  small: the first N lines of --data -> tokcache_check/ (own cache dir), single and multi, miss / hit / subset hit
  --full: --data itself and --subset (e.g. train_g4) with the real cache dir: fresh vs miss vs hit timings + equality
  python tokcache_check.py --model ckpt/bc-v1 --data human_data14d/train.jsonl --subset human_data14d/train_g4.jsonl \
      --max-len 1152 --head-max-len 512 --workers 2 [--full]
"""
import argparse
import os
import shutil
import time

import numpy as np

from finetune_fast import load_items

HERE = os.path.dirname(os.path.abspath(__file__))


def same(a, b):
    if len(a) != len(b):
        return f"lengths {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if x.keys() != y.keys():
            return f"item {i}: keys {sorted(x)} != {sorted(y)}"
        if x["ids"].dtype != np.int32 or y["ids"].dtype != np.int32 or not np.array_equal(x["ids"], y["ids"]):
            return f"item {i}: ids differ"
        for k in x:
            if k != "ids" and x[k] != y[k]:
                return f"item {i}: {k} differs"
    return "equal"


def timed(*args, **kw):
    t = time.time()
    items, how = load_items(*args, **kw)
    return items, how, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--head-max-len", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=3000)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    m, L, H, W = a.model, a.max_len, a.head_max_len, a.workers

    d = os.path.join(HERE, "tokcache_check")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    small, sub, cache = os.path.join(d, "small.jsonl"), os.path.join(d, "sub.jsonl"), os.path.join(d, "cache")
    with open(a.data, "rb") as f:
        lines = [next(f) for _ in range(a.lines)]
    open(small, "wb").write(b"".join(lines))
    open(sub, "wb").write(b"".join(lines[1::2]))
    for multi in (False, True):
        fresh, how0, _ = timed(m, small, L, W, multi, H, "")
        miss, how1, _ = timed(m, small, L, W, multi, H, cache)
        hit, how2, _ = timed(m, small, L, W, multi, H, cache)
        sub_fresh, _, _ = timed(m, sub, L, W, multi, H, "")
        sub_hit, how3, _ = timed(m, sub, L, W, multi, H, cache)
        print(f"small multi={multi}: {len(fresh)} items; {how0}/{how1}/{how2}/sub {how3}; "
              f"miss {same(fresh, miss)}, hit {same(fresh, hit)}, subset hit {same(sub_fresh, sub_hit)}", flush=True)
        assert (how0, how1, how2, how3) == ("off", "miss", "hit", "hit")
    shards = len([n for n in os.listdir(os.path.join(cache, os.listdir(cache)[0])) if n.endswith(".meta.pkl")])
    print(f"cache keys {len(os.listdir(cache))} (single, multi), shards in the first {shards}", flush=True)

    if a.full:
        real = os.path.join(HERE, "tokcache")
        fresh, _, t0 = timed(m, a.data, L, W, False, H, "")
        miss, how1, t1 = timed(m, a.data, L, W, False, H, real)
        hit, how2, t2 = timed(m, a.data, L, W, False, H, real)
        print(f"full {a.data}: {len(fresh)} items; fresh {t0:.0f}s, {how1} {t1:.0f}s, {how2} {t2:.1f}s; "
              f"miss {same(fresh, miss)}, hit {same(fresh, hit)}", flush=True)
        del fresh, miss, hit
        sfresh, _, t3 = timed(m, a.subset, L, W, False, H, "")
        shit, how4, t4 = timed(m, a.subset, L, W, False, H, real)
        print(f"subset {a.subset}: {len(sfresh)} items; fresh {t3:.0f}s, {how4} {t4:.1f}s; {same(sfresh, shit)}",
              flush=True)


if __name__ == "__main__":
    main()
