"""One training file for one model on several NetHack question sets (laya takes the question and its options as
input, so one encoder + head answers all of them; joint training is the upper bound that weight merging is measured
against, Yadav et al. 2023). Each source is repeated `times` (fractions sample), the result shuffled.
  python mix_build.py --out mix_data/train.jsonl human_data/train.jsonl:1 v2_data/train.jsonl:1 \
      choice_fork_h300_all/train_choice.jsonl:5
-> the jsonl, <out>.meta.json (rows per source)
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="path:times (times may be a fraction: a random share)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    rows, meta = [], {}
    for s in a.sources:
        path, _, times = s.rpartition(":")
        times = float(times)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        whole, part = int(times), times - int(times)
        picked = lines * whole + rng.sample(lines, round(len(lines) * part))
        rows += picked
        meta[path] = {"lines": len(lines), "times": times, "rows": len(picked)}
    rng.shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(r + "\n")
    meta["total"] = len(rows)
    Path(a.out + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
