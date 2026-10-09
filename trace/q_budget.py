"""Do a dataset's questions fit laya's head budget (head_max_len tokens for the instruction and the options)?
build_sequence silently caps every option when they do not fit: count per question how often that happens, the
option tokens, and whether options still differ after the cap.
  python q_budget.py <laya checkpoint> <jsonl> [rows]
"""
import collections
import json
import sys

from finetune_fast import load_model
from laya.common import build_sequence


def main():
    import torch
    cfg, tok, _ = load_model(sys.argv[1], torch.device("cpu"))
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    stats = collections.defaultdict(lambda: collections.Counter())
    for i, line in enumerate(open(sys.argv[2], encoding="utf-8")):
        if i >= n:
            break
        r = json.loads(line)
        for qid, q in r["questions"].items():
            qd = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria", {})}
            ids, markers, st = build_sequence(tok, "", qd, 10 ** 6, cfg["head_max_len"], state_ids=[], return_stats=True)
            s = stats[qid]
            s["n"] += 1
            s["capped"] += st["tokens_per_option"] is not None
            s["merged"] += st["options_distinct"] < st["options"]
            s["head_tokens"] = max(s["head_tokens"], len(ids))
            s["options"] = st["options"]
    print(f"head_max_len {cfg['head_max_len']}")
    for qid, s in stats.items():
        print(f"{qid:8s} {s['n']} rows, {s['options']} options, longest head {s['head_tokens']} tokens, "
              f"options capped in {s['capped']}, options no longer distinct in {s['merged']}")


if __name__ == "__main__":
    sys.exit(main())
