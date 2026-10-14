"""Does the batch order explain how closely each seed fits its train rows? finetune_fast.py --batch B: items (one per
question with a gold label, file order) shuffled once by random.Random(seed), cut into batches of B; one epoch, cosine LR.
Per seed: where the items of the G cells (Dlvl 21-24, 25-29) fall in the epoch, and how closely the seed fits them
(raw loss on its own train rows); pooled over seeds, the slope of an item's loss residual (seed minus the mean of the
seeds) on its position in the epoch, and the fit each seed would get from that slope and its positions alone.
  python order_probe.py human_dwa30/train.jsonl c30e=0 c30es1=1 c30es2=2 c30es3=3 --items 357147 [--out ...]
"""
import argparse
import bisect
import json
import random
import re
from pathlib import Path

import numpy as np

from curve_eval import nll
from deep_checks import BANDS, NAMES

DEEP = ("21-24", "25-29")


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("models", nargs="+", help="name=seed")
    ap.add_argument("--items", type=int, required=True, help="the item count of the training log (a check)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    keys = []  # item index -> (row id, question, label index, band)
    for line in open(a.train, encoding="utf-8"):
        r = json.loads(line)
        dl = re.search(r"\bDlvl (\d+)", r["state"])
        band = NAMES[bisect.bisect_left(BANDS, int(dl.group(1)) if dl else -1)]
        for q, spec in r["questions"].items():
            if q in r["gold"]:
                crit = spec["criteria"]
                opts = list(crit) if isinstance(crit, dict) else [str(k) for k in range(len(crit))]
                keys.append((r["id"], q, opts.index(r["gold"][q]["label"]), band))
    assert len(keys) == a.items, (len(keys), a.items)
    deep = np.array([k[3] in DEEP for k in keys])
    n = len(keys)
    steps = -(-n // a.batch)
    out = [f"{a.train}: {n} items, {steps} steps of {a.batch}; G-cell items {deep.sum()}"]
    pos, res, loss = {}, {}, {}
    for spec in a.models:
        m, seed = spec.split("=")
        idx = list(range(n))
        random.Random(int(seed)).shuffle(idx)
        p = np.empty(n)
        p[np.array(idx)] = (np.arange(n) // a.batch) / steps
        pos[m] = p[deep]
        df = pd.read_parquet(f"preds/{m}-{Path(a.train).parent.name}-{Path(a.train).stem}.parquet")
        lg = df.set_index(["id", "question"]).logits
        sel = [k for k, d in zip(keys, deep) if d]
        z = [np.asarray(lg[(k[0], k[1])], float) for k in sel]
        y = np.array([k[2] for k in sel])
        loss[m] = np.array([nll(zz[None], yy[None])[0] for zz, yy in zip(z, y)])
    names = list(loss)
    mean = np.mean([loss[m] for m in names], 0)
    for m in names:
        res[m] = loss[m] - mean
    P = np.concatenate([pos[m] for m in names])
    R = np.concatenate([res[m] for m in names])
    slope = np.polyfit(P, R, 1)[0]
    bins = np.linspace(0, 1, 11)
    curve = [R[(P >= lo) & (P < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
    out.append(f"pooled: residual loss on position slope {slope:+.4f} per epoch; by tenth of the epoch "
               + " ".join(f"{x:+.3f}" for x in curve))
    for m in names:
        pred = slope * (pos[m].mean() - P.mean())
        out.append(f"  {m:8s} mean position {pos[m].mean():.4f}  in last tenth {(pos[m] >= 0.9).mean():.3f}  "
                   f"fit (raw loss) {loss[m].mean():.4f}  residual {res[m].mean():+.4f}  from position alone {pred:+.5f}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
