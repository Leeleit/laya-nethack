"""From seed_var.py outputs (run with --train): per seed, the raw loss on its own train rows and the held-out raw and
T-band loss, each averaged over the 8 cells of G; the least-squares slope of held on train over the seeds
(slope 0: fitting the train rows closer neither helps nor hurts held-out players; -1: every unit of closer fit is lost).
  python fit_held.py ../results/seed_var30_drop1.txt ../results/seed_var24_drop1.txt [--out ../results/fit_held.txt]
"""
import argparse
import re
from pathlib import Path

import numpy as np

QS = ("ttd", "depth", "end", "ascends")
DEEP = ("21-24", "25-29")


def parse(path):
    text = Path(path).read_text(encoding="utf-8")
    models = text.split("\n", 1)[0].split("seeds ", 1)[1].split(", ")
    tb, tr, gap = {}, {}, {}
    part = ""
    for line in text.splitlines():
        if line.startswith("===="):
            part = line
            continue
        f = line.split()
        if len(f) < 3 or f[0] not in QS or f[1] not in DEEP:
            continue
        key = f[0], f[1]
        if part.startswith("==== cells"):
            tb[key] = np.array(f[3:3 + len(models)], float)
        elif part.startswith("==== own train"):
            nums = re.findall(r"[+-]?\d+\.\d+", line)
            tr[key] = np.array(nums[:len(models)], float)
            gap[key] = np.array(nums[len(models):2 * len(models)], float)
    keys = [(q, b) for q in QS for b in DEEP]
    t = np.mean([tr[k] for k in keys], 0)
    raw = np.mean([tr[k] + gap[k] for k in keys], 0)
    loc = np.mean([tb[k] for k in keys], 0)
    return models, t, raw, loc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = ["mean over the 8 cells of G: train raw (own train rows), held raw, held T-band; slope of held on train"]
    for path in a.files:
        models, t, raw, loc = parse(path)
        out.append(f"\n{Path(path).name}")
        for m, x, y, z in zip(models, t, raw, loc):
            out.append(f"  {m:10s} train {x:.4f}  held raw {y:.4f}  held T-band {z:.4f}")
        sr, sl = (np.polyfit(t, v, 1)[0] for v in (raw, loc))
        out.append(f"  SD train {t.std(ddof=1):.4f}  slope raw {sr:+.2f}  slope T-band {sl:+.2f}"
                   f"  corr raw {np.corrcoef(t, raw)[0, 1]:+.2f}  corr T-band {np.corrcoef(t, loc)[0, 1]:+.2f}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
