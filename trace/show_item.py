"""What the model reads for a dataset row: the single-question sequences (A/B) and the multi-question one (C), decoded
back from token ids, with lengths and marker positions.
  python show_item.py <laya checkpoint> <jsonl> [row index]
"""
import json
import sys

from finetune_fast import build_multi_item, build_training_item, load_model


def show(tok, ids, markers, title):
    sep = [i for i, t in enumerate(ids) if t == tok.sep_token_id]
    print(f"--- {title}: {len(ids)} tokens, markers at {markers}, [SEP] at {sep}")
    head = sep[-2] + 1 if len(sep) >= 2 else len(ids)  # through the [SEP] closing the last option block
    print(tok.decode(ids[:head]))
    print(f"... then the state: {len(ids) - head - 1} tokens + [SEP]")


def main():
    model_dir, data = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    import torch
    cfg, tok, _ = load_model(model_dir, torch.device("cpu"))
    cfg["max_len"] = 1152
    r = [json.loads(line) for line in open(data, encoding="utf-8")][k]
    for qid, q in r["questions"].items():
        it = build_training_item(tok, cfg, r["state"], q, r["gold"][qid])
        show(tok, it["ids"], it["markers"], f"A/B: one sequence for '{qid}' (qtype {it['qtype']}, target {it['target']})")
    m = build_multi_item(tok, cfg, r["state"], r["questions"], r["gold"])
    show(tok, m["ids"], m["markers"], f"C: one sequence for {m['qids']} (blocks {m['blocks']}, targets {m['targets']})")
    st = tok.decode(m["ids"][m["blocks"][-1][1]:-1])
    print("--- C, the state as decoded from its tokens (first 600 chars):")
    print(st[:600])


if __name__ == "__main__":
    sys.exit(main())
