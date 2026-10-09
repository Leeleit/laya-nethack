"""Token counts of spelling variants: each argument line 'a || b || c' prints the count of each variant.
  python tok_alt.py <checkpoint dir> <file with one 'a || b' line per comparison>
"""
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(f"{sys.argv[1]}/tokenizer")
for line in open(sys.argv[2], encoding="utf-8"):
    if line.strip():
        print("  |  ".join(f"{len(tok(v.strip(), add_special_tokens=False)['input_ids']):3d} {v.strip()!r}"
                           for v in line.split("||")))
