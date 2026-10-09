"""Where the inventory's tokens go: its share of the state and the cost of its recurring phrases.
  python inv_tokens.py <checkpoint dir> <rows.jsonl>
"""
import re
import sys
import json
import collections

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(f"{sys.argv[1]}/tokenizer")
n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
total = inv_total = items = letters = 0
phrases = collections.Counter()
for line in open(sys.argv[2]):
    s = json.loads(line)["state"]
    total += n(s)
    inv = s.split("Inventory (as last shown): ", 1)[1].split("\n", 1)[0]
    inv_total += n(inv)
    letters += n(inv) - n("; ".join(it[4:] if it[1:4] == " - " else it for it in inv.split("; ")))
    for it in inv.split("; "):
        items += 1
        for p in re.findall(r"\([^)]*\)", it):
            phrases[p] += 1
        for w in ("uncursed", "blessed", "cursed", "a ", "an ", "the ", "named", "called", "partly eaten"):
            if re.search(rf"(?<![\w-]){re.escape(w.strip())}\b", it[4:]):
                phrases[w.strip()] += 1
print(f"inventory {inv_total / total:.1%} of state tokens, {items} items, {inv_total / items:.1f} tokens an item; "
      f"the letters 'x - ' {letters / total:.1%} of state tokens")
for p, k in phrases.most_common(25):
    print(f"  {p!r:40s} {n(' ' + p)} tokens x {k} = {n(' ' + p) * k / total:.2%} of state tokens")
