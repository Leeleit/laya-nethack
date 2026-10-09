"""Is finetune_fast.marker_logits (the last head layer only at the option markers) the same as DecisionModel.forward?
Both in eval on the same batches, fp32 (exactness) and fp16 autocast (as predict.py / laya_serve.py run), plus time.
  python head_check.py <laya checkpoint> <jsonl with state/questions/gold> [batch]
"""
import sys
import time

import torch

from finetune_fast import build_training_item, collate, load_model, marker_logits, pad_to

KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def main():
    import json
    model_dir, data = sys.argv[1], sys.argv[2]
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    device = torch.device("cuda", 0)
    cfg, tok, model = load_model(model_dir, device)
    cfg["max_len"] = 1152
    model.eval()
    items = []
    for line in open(data, encoding="utf-8"):
        row = json.loads(line)
        for qid, q in row["questions"].items():
            if qid in row["gold"] and (it := build_training_item(tok, cfg, row["state"], q, row["gold"][qid])):
                items.append(it)
    items.sort(key=lambda it: len(it["ids"]))
    batches = [pad_to(collate(items[j:j + bs], tok.pad_token_id) | {"pad_id": tok.pad_token_id}, 64)
               for j in range(0, len(items), bs)]
    print(f"{len(items)} items, {len(batches)} batches of {bs}", flush=True)
    for amp in (False, True):
        dz = dp = 0.0
        same = n = 0
        t_full = t_fast = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            for b in batches:
                args = [b[k].to(device) for k in KEYS]
                torch.cuda.synchronize(); t = time.time()
                full = model(*args)[0].float()
                torch.cuda.synchronize(); t_full += time.time() - t; t = time.time()
                fast = marker_logits(model, *args).float()
                torch.cuda.synchronize(); t_fast += time.time() - t
                m = args[3]
                dz = max(dz, float((full - fast).abs()[m].max()))
                dp = max(dp, float((full.softmax(-1) - fast.softmax(-1)).abs().max()))
                same += int((full.argmax(-1) == fast.argmax(-1)).sum())
                n += len(full)
        print(f"{'fp16 autocast' if amp else 'fp32'}: max |logit diff| {dz:.2e}, max |prob diff| {dp:.2e}, "
              f"argmax same {same}/{n}; time full {t_full:.2f}s, marker head {t_fast:.2f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
