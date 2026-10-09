"""Tokens of the map-in-words line: what its fixed words cost over a split, and shorter spellings.
  python around_tokens.py <checkpoint dir> <rows.jsonl built with --map text>
"""
import re
import sys
import json

import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(f"{sys.argv[1]}/tokenizer")
n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
lines, states = [], []
for line in open(sys.argv[2]):
    s = json.loads(line)["state"]
    states.append(n(s))
    lines.append(re.search(r"\nAround the hero: (.*)\n", s).group(1))


def short(t):
    t = t.replace("Features: none. ", "").replace("Features: ", "").replace(". Monsters in view: none", "")
    t = t.replace("Monsters in view: none", "").replace("Monsters in view:", "Monsters:")
    t = re.sub(r" \(no known path(, (\d+) seen)?\)", lambda m: f" ?{' x' + m.group(2) if m.group(2) else ''}", t)
    t = re.sub(r", (\d+) seen", r" x\1", t)
    return re.sub(r" (\d+) moves?", r" \1", t)


total = sum(states)
full = np.array([n(f"\nAround the hero: {t}") for t in lines])
cut = np.array([n(f"\nAround the hero: {short(t)}") for t in lines])
print(f"{len(lines)} states, {total / len(lines):.0f} tokens a state on average")
print(f"the line now: mean {full.mean():.1f} tokens ({full.sum() / total:.1%} of the states), shorter: "
      f"{cut.mean():.1f} ({cut.sum() / total:.1%})")
for word in ["Features: none. ", "Features: ", "Monsters in view: none", "Monsters in view: ", " (no known path)",
             " seen", " moves", " move"]:
    k = sum(t.count(word.strip(" ")) for t in lines)
    print(f"  {word!r:24s} {n(word)} tokens x {k / len(lines):.2f} a state")
for t in lines[:12]:
    print(f"  {t}\n    -> {short(t)}")
