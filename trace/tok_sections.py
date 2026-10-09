"""Where the state's tokens go: mean / median tokens per section of the state text (header, inventory,
level map, ...) over a split, with the tokenizer of a checkpoint. Shows what shortening the input could buy.
  python tok_sections.py <checkpoint dir> <rows.jsonl>
"""
import sys
import json
import re

import numpy as np
from transformers import AutoTokenizer

ckpt, path = sys.argv[1:3]
tok = AutoTokenizer.from_pretrained(f"{ckpt}/tokenizer")
SECTION = re.compile(r"^(Inventory|Level map|Map around|Message|Status|HP |Character)[^:\n]*", re.M)
per, total = {}, []
for line in open(path):
    s = json.loads(line)["state"]
    cuts = [m.start() for m in SECTION.finditer(s)] + [len(s)]
    names = [m.group(1) for m in SECTION.finditer(s)]
    total.append(len(tok(s, add_special_tokens=False)["input_ids"]))
    seen = {}
    for name, a, b in zip(names, cuts, cuts[1:]):
        seen[name] = seen.get(name, 0) + len(tok(s[a:b], add_special_tokens=False)["input_ids"])
    for name in set(per) | set(seen):
        per.setdefault(name, []).append(seen.get(name, 0))
total = np.array(total)
print(f"{len(total)} states: tokens mean {total.mean():.0f}, median {np.median(total):.0f}, p90 {np.percentile(total, 90):.0f}")
for name, v in sorted(per.items(), key=lambda kv: -np.mean(kv[1])):
    v = np.array(v + [0] * (len(total) - len(v)))
    print(f"  {name:12s} mean {v.mean():6.1f} ({v.mean() / total.mean():5.1%}), median {np.median(v):5.0f}, p90 {np.percentile(v, 90):5.0f}")

# the level map is redundant only where a separate window around the hero is shown (map larger than 25x11)
win, lvl = [], []
for line in open(path):
    s = json.loads(line)["state"]
    has_win = "the whole level map below" not in s
    m = re.search(r"\nLevel map:\n.*\Z", s, re.S)
    win.append(has_win)
    lvl.append(len(tok(m.group(0), add_special_tokens=False)["input_ids"]) if m else 0)
win, lvl = np.array(win), np.array(lvl)
print(f"window shown in {win.mean():.1%} of states; level map tokens there: {lvl[win].sum() / total.sum():.1%} of all "
      f"state tokens (mean {lvl[win].mean():.0f} where shown); elsewhere the level map is the only map")
