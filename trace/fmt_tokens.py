"""Where an example's tokens go: the state's lines by their head word and the question blocks (the
instructions and every option), as average tokens per example over a split.
  python fmt_tokens.py <checkpoint dir> <rows.jsonl>
"""
import sys
import json
import collections

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(f"{sys.argv[1]}/tokenizer")
n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
lines, examples, state_total = collections.Counter(), 0, 0
qcost = {}
for line in open(sys.argv[2]):
    r = json.loads(line)
    st = r["state"]
    head = {}
    for ln in st.split("\n"):
        key = ln.split(":")[0] if ":" in ln[:30] else ln.split(" ")[0]
        head[key] = head.get(key, 0) + n(ln)
    qs = [q for q in r["questions"] if q in r["gold"]]
    examples += len(qs)
    for k, v in head.items():
        lines[k] += v * len(qs)  # the state is encoded once per question
    state_total += n(st) * len(qs)
    for q in qs:
        if q not in qcost:
            d = r["questions"][q]
            opts = d["criteria"].values() if isinstance(d["criteria"], dict) else d["criteria"]  # choice / noul: texts
            qcost[q] = (n(d["instructions"]), [(o, n(o)) for o in opts])
qn = collections.Counter(q for line in open(sys.argv[2]) for q in json.loads(line)["gold"])
print(f"{examples} examples; state {state_total / examples:.0f} tokens an example")
for k, v in lines.most_common():
    print(f"  {k[:30]!r:34s} {v / examples:6.1f}")
for q, (ins, opts) in qcost.items():
    print(f"question {q} ({qn[q]} rows): instructions {ins}, options {sum(c for _, c in opts)} + markers {len(opts)}")
    for o, c in opts:
        print(f"    {c:3d}  {o}")
