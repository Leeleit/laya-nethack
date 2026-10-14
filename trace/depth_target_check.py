"""A set rebuilt with human_build.py --depth-target against the same set with the original target: same rows in the same
order, everything but gold.depth equal, the depth label equal, and the new target equal to soft_variants.py's version
of it (from the gain recovered out of the original target). Prints the worst gap and the mean mass on the last bins.
  python depth_target_check.py human_dwa20 human_dwa20n narrow
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from soft_variants import orig, ranges, recover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("target", choices=["trunc", "half", "narrow", "hard"])
    a = ap.parse_args()
    meta = json.loads(Path(a.new, "meta.json").read_text())["v6"]
    cap = meta.get("depth_cap")
    print(f"{a.new}: depth_target {meta.get('depth_target')}, depth_cap {cap}")
    for split in ("train", "val", "test"):
        n, worst, other, mass_b, mass_n = 0, 0.0, 0, [], []
        with open(Path(a.base, f"{split}.jsonl"), encoding="utf-8") as fb, \
                open(Path(a.new, f"{split}.jsonl"), encoding="utf-8") as fn:
            for lb, ln in zip(fb, fn, strict=True):
                rb, rn = json.loads(lb), json.loads(ln)
                gb, gn = rb["gold"].pop("depth"), rn["gold"].pop("depth")
                if rb != rn or gb["label"] != gn["label"]:
                    other += 1
                    continue
                crit = rb["questions"]["depth"]["criteria"]
                k, kk = int(gb["label"]), len(crit)
                d, rg = ranges(crit)
                pb = np.array([gb["probabilities"].get(str(c), 0.0) for c in range(kk)])
                pn = np.array([gn["probabilities"].get(str(c), 0.0) for c in range(kk)])
                g = recover(rg, pb, k) if k < kk - 1 else int(rg[k][0])
                c = max(cap, d + g) - d if cap is not None else math.inf
                want = (np.eye(kk)[k] if a.target == "hard" else
                        orig(rg, g, c, 0.375) if a.target == "half" else
                        orig(rg, g, c, flat=True) if a.target == "narrow" else orig(rg, g, c))
                if k < kk - 1:  # the open bin's gain is not recoverable
                    worst = max(worst, float(np.abs(pn - want).max()))
                mass_b.append(pb[-3:])
                mass_n.append(pn[-3:])
                n += 1
        print(f"  {split}: {n} rows, {other} differ outside gold.depth; worst |new - expected| {worst:.2e}; "
              f"mean mass on the last 3 bins {np.mean(mass_b, 0).round(3)} -> {np.mean(mass_n, 0).round(3)}")


if __name__ == "__main__":
    main()
