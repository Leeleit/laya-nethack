"""State text length and game version per Dlvl band of several sets: chars (median, p90), share of rows whose
state + first question exceed --max-len tokens (the model's tokenizer), and the version shares.
  python state_len.py human_dwa30/train.jsonl human_dwa30t/val.jsonl human_dwa30t/test.jsonl --model-dir ckpt/c30e
      [--max-len 1152] [--sample 3000] [--out ../results/state_len.txt]
"""
import argparse
import bisect
import collections
import json
import random
import re
from pathlib import Path

import numpy as np

from deep_checks import BANDS, NAMES


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-len", type=int, default=1152)
    ap.add_argument("--sample", type=int, default=3000, help="rows per band tokenised")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(Path(a.model_dir, "tokenizer")))
    out = [f"chars median / p90, tokens of state + first question median / p90, share over {a.max_len} tokens, versions"]
    for path in a.files:
        by = collections.defaultdict(list)
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            dl = re.search(r"\bDlvl (\d+)", r["state"])
            q = next(iter(r["questions"].values()))
            by[bisect.bisect_left(BANDS, int(dl.group(1)) if dl else -1)].append(
                (r["state"], q["instructions"] + " " + json.dumps(q.get("criteria", ""))))
        out.append(f"\n{path}")
        rng = random.Random(0)
        for b, name in enumerate(NAMES):
            rs = by.get(b, [])
            if not rs:
                continue
            ch = np.array([len(s) for s, _ in rs])
            sm = rng.sample(rs, min(a.sample, len(rs)))
            tk = np.array([len(tok(s + "\n" + q)["input_ids"]) for s, q in sm])
            ver = collections.Counter(s.split("\n", 1)[0].split()[-1] for s, _ in rs)
            out.append(f"  {name:6s} rows {len(rs):6d}  chars {np.median(ch):6.0f} / {np.percentile(ch, 90):6.0f}  tokens "
                       f"{np.median(tk):5.0f} / {np.percentile(tk, 90):5.0f}  over {(tk > a.max_len).mean():6.1%}  "
                       + " ".join(f"{v} {c / len(rs):.0%}" for v, c in ver.most_common(3)))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
