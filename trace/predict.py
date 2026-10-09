"""Raw option logits of a laya checkpoint for every case of a JSONL file.

Tokenized exactly as in training (finetune_single_device.preprocess), so the logits are the ones the
loss saw: no temperature, no rounding (laya's predict API rounds probabilities to 4 decimals, which
erases the small death probabilities the calibration has to work with).

A checkpoint trained with finetune_fast.py --multi (multi_question in its config) gets every question of a case in
one sequence (build_multi_item), as in its training.

  python predict.py <checkpoint dir> <cases.jsonl> <out.parquet> [single|multi: override the checkpoint's mode]
  -> id, question, logits (one row per question)
"""
import json
import sys
import time

import pandas as pd
import torch

from finetune_fast import build_multi_item, collate_multi, load_model, logits_of
from finetune_single_device import collate, preprocess

TOKENS = 32768


def main():
    ckpt, data, out = sys.argv[1:4]
    cfg, tok, model = load_model(ckpt, torch.device("cuda", 0))
    model.eval()
    multi = cfg.get("multi_question", False) if len(sys.argv) < 5 else sys.argv[4] == "multi"  # override: diagnosis
    if multi:  # one item per case; its questions with gold in the case's order
        rows = [json.loads(line) for line in open(data)]
        items = [build_multi_item(tok, cfg, r["state"], r["questions"], r["gold"]) for r in rows]
        assert all(it and len(it["qids"]) == sum(q in r["gold"] for q in r["questions"])
                   for it, r in zip(items, rows)), "build_multi_item dropped a question"
    else:  # one item per (case, question with gold), in preprocess's order
        keys = [(r["id"], q) for r in map(json.loads, open(data)) for q in r["questions"] if q in r["gold"]]
        items = preprocess(tok, cfg, data)
        assert len(items) == len(keys), "preprocess dropped a question"
        ids, qids = [k[0] for k in keys], [k[1] for k in keys]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    logits = [None] * len(items)
    t0 = time.time()
    # batches of at most TOKENS padded tokens: a fixed 128 rows of ~1000-token v2 states filled the 32 GB card
    # and the caching allocator thrashed (risk-v2)
    batches, cur = [], []
    for i in order:  # ascending length, so the last item of a batch is its longest
        if cur and (len(cur) + 1) * len(items[i]["ids"]) > TOKENS:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)
    with torch.no_grad():
        for b in batches:
            z = logits_of(model, (collate_multi if multi else collate)([items[i] for i in b], tok.pad_token_id),
                          torch.device("cuda", 0)).cpu()
            if multi:  # rows of z: the questions of b's items in order
                rows_z = iter(z)
                for i in b:
                    logits[i] = [next(rows_z)[:len(mk)].tolist() for mk in items[i]["markers"]]
            else:
                for r, i in enumerate(b):
                    logits[i] = z[r, :len(items[i]["markers"])].tolist()
    if multi:
        ids = [r["id"] for r, it in zip(rows, items) for _ in it["qids"]]
        qids = [q for it in items for q in it["qids"]]
        logits = [x for per_case in logits for x in per_case]
    pd.DataFrame({"id": ids, "question": qids, "logits": logits}).to_parquet(out)
    print(f"{len(ids)} cases in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
