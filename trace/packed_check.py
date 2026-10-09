"""packed.py against the padded sdpa path on the same batches: option logits (eval), loss and gradients
(no dropout), fwd+bwd time without compile.
  python packed_check.py <laya checkpoint> <jsonl> [--multi] [--rows 64] [--batch 8]
"""
import argparse
import json
import time

import torch

import packed
from finetune_fast import build_multi_item, collate, collate_multi, load_model, logits_of
from finetune_single_device import preprocess

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("data")
ap.add_argument("--multi", action="store_true")
ap.add_argument("--rows", type=int, default=64)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
dev = torch.device("cuda", 0)
cfg, tok, model = load_model(a.ckpt, dev)
cfg["max_len"] = 1152
rows = [json.loads(line) for line, _ in zip(open(a.data), range(a.rows))]
if a.multi:  # as load_items builds them (no fork pool: Windows)
    items = [build_multi_item(tok, cfg, r["state"], r["questions"], r["gold"]) for r in rows]
else:
    with open("tmp/packed_check_rows.jsonl", "w", newline="\n") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    items = preprocess(tok, cfg, "tmp/packed_check_rows.jsonl")[:a.rows]
model.eval()
col = collate_multi if a.multi else collate
batches = [col(items[i:i + a.batch], tok.pad_token_id) for i in range(0, len(items), a.batch)]
enc = model.encoder


def disable():
    enc.forward = enc._padded_forward
    enc.config._attn_implementation = "sdpa"


def run():
    with torch.no_grad():
        z = [logits_of(model, b, dev) for b in batches]
    small = batches[0] if a.multi else col(items[:4], tok.pad_token_id)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        if a.multi:
            lg = logits_of(model, small, dev)
        else:
            args = [small[k].to(dev) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            lg = model(*args)[0].float()  # DecisionModel.forward, the training path (eval: no dropout)
    mask, tg = small["marker_mask"].to(dev), small["target"].to(dev)
    loss = -(tg * torch.log_softmax(lg.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss.backward()
    g = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
         if p.grad is not None and any(s in n for s in ("layers.0.attn.Wqkv", "layers.10.mlp.Wi", "layers.19.attn.Wo",
                                                         "embeddings.tok", "head.layers.1.linear2", "scorer"))}
    torch.cuda.synchronize()
    t0 = time.time()
    for b in batches[:6]:
        model.zero_grad(set_to_none=True)
        lg = logits_of(model, b, dev) if a.multi else None
        if not a.multi:
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(*[b[k].to(dev) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask",
                                                    "qtype")])[0].float()
        lg.masked_fill(~b["marker_mask"].to(dev), 0).sum().backward()
    torch.cuda.synchronize()
    return z, float(loss), g, (time.time() - t0) / 6


packed.enable(model, 1280)
disable()
z0, l0, g0, t0 = run()
enc.forward = packed._forward.__get__(enc)
enc.config._attn_implementation = "laya_varlen"
z1, l1, g1, t1 = run()
dz = max(float((x - y).abs().max()) for x, y in zip(z0, z1))
dp = max(float((torch.softmax(x, -1) - torch.softmax(y, -1)).abs().max()) for x, y in zip(z0, z1))
print(f"{len(items)} items, {'multi' if a.multi else 'single'}: max |logit diff| {dz:.4f}, max |prob diff| {dp:.5f}")
print(f"loss padded {l0:.5f} packed {l1:.5f}")
for n in g0:
    cos = float(torch.nn.functional.cosine_similarity(g0[n].flatten(), g1[n].flatten(), 0))
    rel = float((g0[n] - g1[n]).norm() / g0[n].norm())
    print(f"  grad {n[:60]:60s} cos {cos:.5f} rel diff {rel:.4f}")
print(f"fwd+bwd per batch of {a.batch} (eager): padded {t0 * 1000:.0f} ms, packed {t1 * 1000:.0f} ms (x{t0 / t1:.2f})")
