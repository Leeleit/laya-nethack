"""What the inventory letters cost inside --compact: tokens a state of human_data9s (full inventory, short
map words) with the inventory as --compact writes it, and as --compact with the letters kept (the other rules applied
to the item after its letter).
  python letters_tokens.py <checkpoint dir> human_data9s/val.jsonl
Measured with compact_inv as it was then (dropping the letters); it keeps them since, so a rerun shows no difference.
"""
import re
import sys
import json

from transformers import AutoTokenizer

from human_build import compact_inv

tok = AutoTokenizer.from_pretrained(f"{sys.argv[1]}/tokenizer")
n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
LETTER = re.compile(r"^([a-zA-Z] - )(.*)$")


def with_letters(inv):
    if inv == "unknown":
        return inv
    out = []
    for it in inv.split("; "):
        m = LETTER.match(it)
        out.append(m.group(1) + compact_inv(m.group(2)) if m else compact_inv(it))
    return "; ".join(out)


head = "Inventory (as last shown): "
rows = full = compact = lettered = 0
for line in open(sys.argv[2]):
    s = json.loads(line)["state"]
    pre, inv = s.split(head, 1)
    inv, nl, post = inv.partition("\n")
    rows += 1
    full += n(s)
    compact += n(pre + head + compact_inv(inv) + nl + post)
    lettered += n(pre + head + with_letters(inv) + nl + post)
print(f"{rows} states, tokens a state: full inventory {full / rows:.1f}, --compact {compact / rows:.1f}, "
      f"--compact with letters {lettered / rows:.1f} (+{(lettered - compact) / rows:.1f})")
