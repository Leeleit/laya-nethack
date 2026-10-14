"""A set rebuilt with another ttd target must differ from its base in gold.ttd's probabilities only: same ids in the same
order, every other field equal; the new target equals N(log(1 + left), meta's ttd_sigma) over the bins, with left
recovered from the base's orig target (soft_ttd.recover).
  python ttd_target_check.py human_dwa24 human_dwa28 [--out ../results/ttd_target_check28.txt]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from soft_ttd import edges, masses, recover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--sigma", type=float, default=0.0, help="for a merged set whose meta.json has no v6 (human_mix*)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sigma = a.sigma or json.loads(Path(a.new, "meta.json").read_text())["v6"]["ttd_sigma"]
    out = [f"{a.base} -> {a.new}: ttd sigma {sigma:.4f}"]
    for split in ("train", "val", "test"):
        if not Path(a.base, f"{split}.jsonl").exists():
            continue
        n = other = label = 0
        soft_b, soft_n, ys, texts = [], [], [], None
        with open(Path(a.base, f"{split}.jsonl"), encoding="utf-8") as fb, \
                open(Path(a.new, f"{split}.jsonl"), encoding="utf-8") as fn:
            for lb, ln in zip(fb, fn, strict=True):
                rb, rn = json.loads(lb), json.loads(ln)
                n += 1
                gb, gn = rb["gold"].pop("ttd", None), rn["gold"].pop("ttd", None)
                other += rb != rn
                if gb is None or gn is None:
                    other += (gb is None) != (gn is None)
                    continue
                label += gb["label"] != gn["label"]
                texts = texts or rn["questions"]["ttd"]["criteria"]
                K = len(texts)
                soft_b.append([gb["probabilities"].get(str(c), 0.0) for c in range(K)])
                soft_n.append([gn["probabilities"].get(str(c), 0.0) for c in range(K)])
                ys.append(int(gb["label"]))
        lo, hi = edges(texts)
        sb, sn, y = np.array(soft_b), np.array(soft_n), np.array(ys)
        x, err = recover(lo, hi, sb, y)
        want = masses(lo, hi, x, np.full(len(x), sigma))
        want /= want.sum(1, keepdims=True)
        d = np.abs(want - sn).sum(1)
        out.append(f"  {split}: {n} rows, other fields differ in {other}, ttd label differs in {label}; "
                   f"|new target - formula| L1 median {np.median(d):.2e}, max {d.max():.4f} (recovery error max "
                   f"{err.max():.4f}); mass on the last bin base {sb[:, -1].mean():.4f} new {sn[:, -1].mean():.4f}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
