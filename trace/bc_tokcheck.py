"""Token lengths of a laya dataset as training sees them (finetune_single_device.build_training_item): per
question, the items dropped because their options did not fit the head window, and the length percentiles
(a state longer than max_len loses its end).
  python bc_tokcheck.py <checkpoint dir> <data.jsonl> [rows]
"""
import collections
import json
import sys

import numpy as np

from finetune_fast import load_model  # noqa: F401  (sets sys.path to the repo)
from finetune_single_device import build_training_item
from laya.agent import _fix_tokenizer_config
from transformers import AutoTokenizer


def main():
    ckpt, data = sys.argv[1:3]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    _fix_tokenizer_config(ckpt)
    cfg = json.load(open(f"{ckpt}/rl_agent_config.json"))
    tok = AutoTokenizer.from_pretrained(f"{ckpt}/tokenizer")
    lens, dropped, total = collections.defaultdict(list), collections.Counter(), collections.Counter()
    state_lens = []
    with open(data) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            state_lens.append(len(tok(row["state"], add_special_tokens=False)["input_ids"]))
            for q, spec in row["questions"].items():
                total[q] += 1
                it = build_training_item(tok, cfg, row["state"], spec, row["gold"][q])
                if it is None:
                    dropped[q] += 1
                else:
                    lens[q].append(len(it["ids"]))
    print(f"max_len {cfg['max_len']}, head_max_len {cfg['head_max_len']}")
    s = np.array(state_lens)
    print(f"state alone: median {np.median(s):.0f}, p90 {np.percentile(s, 90):.0f}, p99 {np.percentile(s, 99):.0f}")
    for q in total:
        v = np.array(lens[q]) if lens[q] else np.zeros(1)
        print(f"{q:10s} {total[q]:6d} rows, dropped {dropped[q]}, length median {np.median(v):.0f} "
              f"p90 {np.percentile(v, 90):.0f}, at max_len {np.mean(v >= cfg['max_len']):.1%}")


if __name__ == "__main__":
    main()
