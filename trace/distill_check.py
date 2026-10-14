"""Check of a distill_build.py set against its source: the same rows, states, questions and labels in the same order;
every train target = alpha * teacher + (1 - alpha) * old target; targets sum to 1; val and test byte for byte the same.
  python distill_check.py human_dwa24 human_dwa29d --out ../results/distill_check29.txt
"""
import argparse
import filecmp
import json
from pathlib import Path

import numpy as np

from distill_build import keys_of, teacher_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    meta = json.loads(Path(a.dst, "meta.json").read_text())["distill"]
    alpha, tp = meta["alpha"], teacher_probs(meta["teachers"])
    n = other = err = sums = l1 = 0.0
    n_rows = labels = 0
    src, dst = (open(Path(d, "train.jsonl"), encoding="utf-8") for d in (a.src, a.dst))
    for x, y in zip(src, dst):
        r, s = json.loads(x), json.loads(y)
        n_rows += 1
        gr, gs = r.pop("gold"), s.pop("gold")
        other += r != s or set(gr) != set(gs)
        for q, g in gr.items():
            ks = keys_of(r["questions"][q])
            old = np.array([g["probabilities"].get(k, 0.0) for k in ks])
            old = old / old.sum()
            new = np.array([gs[q]["probabilities"][k] for k in ks])
            want = alpha * tp[(r["id"], q)] + (1 - alpha) * old
            err, sums = max(err, np.abs(new - want).max()), max(sums, abs(new.sum() - 1))
            l1 += np.abs(new - old).sum()
            labels += g["label"] == gs[q]["label"] and set(g) == set(gs[q])
            n += 1
    assert sum(1 for _ in open(Path(a.dst, "train.jsonl"), encoding="utf-8")) == n_rows
    same = {s: filecmp.cmp(Path(a.src, s), Path(a.dst, s), shallow=False)
            for s in ("val.jsonl", "test.jsonl") if Path(a.src, s).exists()}
    text = (f"{a.dst} against {a.src}: alpha {alpha}, teachers {', '.join(meta['teachers'])}\n"
            f"train rows {n_rows}, rows differing outside gold {int(other)}; targets {int(n)}, label and keys kept "
            f"{labels}; max |target - formula| {err:.2e}, max |sum - 1| {sums:.2e}, mean L1 change {l1 / n:.4f}\n"
            + ", ".join(f"{s} identical: {v}" for s, v in same.items()))
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    return 0 if other == 0 and labels == n and err < 1e-9 and sums < 1e-9 and all(same.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
