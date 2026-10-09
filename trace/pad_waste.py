"""How much of a training step is padding (finetune_fast.py: --batch random items, padded to the longest, then to a
multiple of 64) -- the ceiling of unpadding / packing. Items as the trainer builds them.
  python pad_waste.py <laya checkpoint> <jsonl> [--multi] [--batch 8] [--max-len 1152]
"""
import argparse
import random

import numpy as np

from finetune_fast import load_items

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("data")
ap.add_argument("--multi", action="store_true")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-len", type=int, default=1152)
a = ap.parse_args()
items, _ = load_items(a.ckpt, a.data, a.max_len, 4, a.multi)
n = np.array([len(it["ids"]) for it in items])
print(f"{len(n)} items: tokens median {np.median(n):.0f}, p10 {np.percentile(n, 10):.0f}, p90 {np.percentile(n, 90):.0f}, "
      f"max {n.max()}")
real = padded = 0
rng = random.Random(0)
for _ in range(20):
    idx = list(range(len(n)))
    rng.shuffle(idx)
    for j in range(0, len(idx), a.batch):
        b = n[idx[j:j + a.batch]]
        real += b.sum()
        padded += len(b) * (-(-b.max() // 64) * 64)
print(f"batch {a.batch}: real / padded tokens {real / padded:.3f} -> padding {1 - real / padded:.1%} of the step's tokens; "
      f"unpadded the same tokens cost x{padded / real:.2f} less in the token-proportional part")
