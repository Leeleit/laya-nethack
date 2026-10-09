"""end v7: how many monster classes fit laya's head at head_max_len 512, and what is left in "another
monster" for each count. Games of the states file with states, up to 2 rows a game (as --per-game 2).
  python v7_budget.py <checkpoint dir> nao/val_dwa_states2.jsonl [heads, default 256,512]
"""
import json
import sys

from transformers import AutoTokenizer

from human_q6 import Builder, outcome
from laya.common import build_sequence

tok = AutoTokenizer.from_pretrained(f"{sys.argv[1]}/tokenizer")
outs = []
for line in open(sys.argv[2], encoding="utf-8"):
    g = json.loads(line)
    if g["states"]:
        outs += [outcome(g, g["states"][0])] * min(2, len(g["states"]))
print(f"{len(outs)} rows")
HEADS = [int(h) for h in (sys.argv[3] if len(sys.argv) > 3 else "256,512").split(",")]
for n in (12, 20, 26, 30, 35, 40, 50, 99):
    b = Builder(outs, n_classes=n, terse=True)
    q = b.questions(outs[0])[0]["end"]
    qd = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
    other = sum(b.end_key(o["end"]) == "monster" for o in outs) / len(outs)
    out = []
    for head in HEADS:
        ids, _, st = build_sequence(tok, "", qd, 10 ** 6, head, state_ids=[], return_stats=True)
        out.append(f"head {head}: {len(ids)} tokens, capped {st['tokens_per_option'] is not None}, "
                   f"distinct {st['options_distinct']}/{st['options']}")
    print(f"classes {len(b.classes):2d}  options {len(b.end):2d}  another monster {other:.1%}  " + "; ".join(out))
