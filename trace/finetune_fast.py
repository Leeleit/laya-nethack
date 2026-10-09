"""Single-GPU fine-tuning of a laya checkpoint, tuned for throughput on one large GPU.

Loss (--loss): `ce` (default), soft cross-entropy only; `rlcd`, the repo's RLCD (noisy-logit policy
gradient rewarded by proper scoring rules + soft CE). Sev ("Dissecting RLCD", github.com/LakoreAI/sev,
30 runs): the RL term is an evolution-strategies estimate of a noise-smoothed proper score whose optimum
is over-sharp at inference (fitted T rises with sigma -- risk-v1 needed T=6.15 for noul); CE-only matched
or beat RL+CE on accuracy, Brier and NLL.

Same precision (fp16 autocast + GradScaler; laya's BENCHMARKS.md: fp16 stays 3-10x closer to
fp32 than bf16 at the same speed) and optimiser as the repo's research/scripts/finetune_single_device.py
(its preprocess/collate are reused). Changes:
  - no gradient checkpointing (the 5090 has the memory; laya's docs/finetune_browser_agent.md: 1.25x);
  - optionally (--batch 0) steps of up to --max-tokens padded tokens, states of similar length
    together; default is the repo's 8 random states per step;
  - no calibration: risk-v1 showed a temperature fitted on the class-rebalanced training slice is
    wrong at the natural rate, so predict.py + compare_v2.py fit it on held-out games instead;
  - fused AdamW; torch.compile of the whole model (--compile default|reduce-overhead|off) on lengths
    padded to multiples of 64, so it compiles once per length bucket instead of once per shape
    (unbucketed variable lengths: 6x slower, docs/finetune_browser_agent.md). No float8: torchao
    measures 0.89-1.08x at ModernBERT-large's 1024-wide layers.

  python finetune_fast.py --data train.jsonl --model-dir <laya checkpoint> --output-dir <dir>
"""
import argparse
import glob
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tokenizer workers are forked
# a clone of the laya repository: $LAYA_REPO, else repo/ beside trace/'s parent or one level higher
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LAYA_REPO") or next(
    (p for p in (os.path.join(_HERE, "..", "repo"), os.path.join(_HERE, "..", "..", "repo")) if os.path.isdir(p)),
    os.path.join(_HERE, "..", "repo"))
sys.path[:0] = [REPO, os.path.join(REPO, "research", "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import finetune_single_device  # noqa: E402
import laya.common  # noqa: E402
from finetune_single_device import build_training_item, collate  # noqa: E402
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import (QTYPES, build_model, build_sequence, encode_text, proper_reward,  # noqa: E402
                         render_options, serialize_state)


def length_batches(items, max_tokens, rng, window=2000):
    """Shuffle, sort windows of `window` items by length, cut each into batches whose padded size
    (count x longest) stays within max_tokens, shuffle the batches."""
    idx = list(range(len(items)))
    rng.shuffle(idx)
    out = []
    for c in range(0, len(idx), window):
        batch, longest = [], 0
        for i in sorted(idx[c:c + window], key=lambda i: len(items[i]["ids"])):
            n = len(items[i]["ids"])
            if batch and max(longest, n) * (len(batch) + 1) > max_tokens:
                out.append(batch)
                batch, longest = [], 0
            batch.append(i)
            longest = max(longest, n)
        if batch:
            out.append(batch)
    rng.shuffle(out)
    return out


def pad_to(batch, mult):
    """Right-pad input_ids / attention_mask to a multiple of `mult` tokens (markers stay where they are)."""
    n, L = batch["input_ids"].shape
    extra = -L % mult
    if extra:
        batch["input_ids"] = torch.nn.functional.pad(batch["input_ids"], (0, extra), value=batch["pad_id"])
        batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], (0, extra))
        if "tok_type" in batch:
            batch["tok_type"] = torch.nn.functional.pad(batch["tok_type"], (0, extra), value=-1)
    return batch


def _target(t, crit, gold_q):
    """build_training_item's target for one question."""
    if t == "choice":
        target = [gold_q["probabilities"].get(k, 0.0) for k in crit]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    else:
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(len(crit) if isinstance(crit, list) else 4)]
    total = sum(target)
    return [v / total for v in target] if total > 0 else [1.0 / len(target)] * len(target)


def build_multi_item(tok, cfg, state, questions, gold):
    """Every question of a state in one sequence (GLiClass, 2508.07662: all labels in one pass, not one pass per
    text-label pair): [CLS] block_1 ... block_n state [SEP], block = `<type> question: instr [SEP] [MASK] opt0 ...
    [SEP]` exactly as build_sequence makes it for one question. The state keeps the room the longest single-question
    sequence leaves it, so no question sees more of the map than alone. -> None when no question is left.
    `gold` None: inference (targets are not built). Questions keep their order in `questions`; "qids" lists them."""
    ids, blocks, markers, qtypes, targets, qids, head = [tok.cls_token_id], [], [], [], [], [], 0
    for qid, q in questions.items():
        if gold is not None and qid not in gold:
            continue
        t, crit = q["type"], q.get("criteria", {})
        qd = {"t": t, "ins": q["instructions"], "crit": crit}
        block, mk = build_sequence(tok, "", qd, 10 ** 6, cfg["head_max_len"], state_ids=[])
        block, mk = block[1:-1], [m - 1 for m in mk]  # without [CLS] and the empty state's [SEP]
        if len(mk) != len(render_options(qd)):
            continue
        blocks.append((len(ids), len(ids) + len(block), QTYPES[t]))
        markers.append([len(ids) + m for m in mk])
        qtypes.append(QTYPES[t])
        targets.append(_target(t, crit, gold[qid]) if gold is not None else None)
        qids.append(qid)
        head = max(head, 1 + len(block))
        ids += block
    if not markers:
        return None
    st = encode_text(tok, serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    ids += st[:max(0, cfg["max_len"] - head - 1)] + [tok.sep_token_id]
    return {"ids": ids, "blocks": blocks, "markers": markers, "qtypes": qtypes, "targets": targets, "qids": qids}


def collate_multi(items, pad_id):
    """Multi-question rows -> the encoder inputs (tok_type: each block's question type, -1 elsewhere), marker_pos
    (B, M: every marker of a row), and per question (Q, K): q_index into the flat (B*M) marker logits, marker_mask,
    target, qtype -- so the loss code is the single-question one over questions instead of rows."""
    n, length = len(items), max(len(it["ids"]) for it in items)
    m_max = max(sum(map(len, it["markers"])) for it in items)
    k_max = max(len(mk) for it in items for mk in it["markers"])
    q_n = sum(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    tok_type = torch.full((n, length), -1, dtype=torch.long)
    mpos = torch.zeros((n, m_max), dtype=torch.long)
    q_index = torch.zeros((q_n, k_max), dtype=torch.long)
    q_mask = torch.zeros((q_n, k_max), dtype=torch.bool)
    target = torch.zeros((q_n, k_max), dtype=torch.float32)
    qtype = torch.zeros(q_n, dtype=torch.long)
    q = 0
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.as_tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        for s, e, t in it["blocks"]:
            tok_type[i, s:e] = t
        slot = 0
        for mk, tg, qt in zip(it["markers"], it["targets"], it["qtypes"]):
            k = len(mk)
            mpos[i, slot:slot + k] = torch.tensor(mk)
            q_index[q, :k] = torch.arange(i * m_max + slot, i * m_max + slot + k)
            q_mask[q, :k] = True
            if tg is not None:
                target[q, :k] = torch.tensor(tg)
            qtype[q] = qt
            slot += k
            q += 1
    return {"input_ids": ids, "attention_mask": att, "tok_type": tok_type, "marker_pos": mpos,
            "q_index": q_index, "marker_mask": q_mask, "target": target, "qtype": qtype}


def load_model(model_dir, device):
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = False
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return cfg, tok, model.to(device)


_TOK = _CFG = _MULTI = None


def _tok_init(model_dir, max_len, multi, head_max_len=0):
    global _TOK, _CFG, _MULTI
    _CFG = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    _CFG["max_len"] = max_len or _CFG["max_len"]
    _CFG["head_max_len"] = head_max_len or _CFG["head_max_len"]
    _MULTI = multi
    _TOK = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))


def _tok_chunk(job):
    """-> [(md5 of the line, its items)] in file order (a line may give no item)."""
    path, start, end = job
    out = []
    with open(path, "rb") as f:
        f.seek(start)
        for line in f.read(end - start).splitlines():
            row, items = json.loads(line), []
            if _MULTI:
                if it := build_multi_item(_TOK, _CFG, row["state"], row["questions"], row["gold"]):
                    it["ids"] = np.asarray(it["ids"], np.int32)
                    items.append(it)
            else:
                for qid, q in row["questions"].items():
                    if qid in row["gold"] and (it := build_training_item(_TOK, _CFG, row["state"], q, row["gold"][qid])):
                        it["ids"] = np.asarray(it["ids"], np.int32)
                        items.append(it)
            out.append((hashlib.md5(line).hexdigest(), items))
    return out


def _cache_key(model_dir, max_len, multi, head_max_len):
    """What the items depend on: the two lengths, multi, the tokenizer's files and the code that builds them."""
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    h = hashlib.sha256(json.dumps([max_len or cfg["max_len"], head_max_len or cfg["head_max_len"], multi]).encode())
    tok_dir = os.path.join(model_dir, "tokenizer")
    for name in sorted(os.listdir(tok_dir)):
        h.update(name.encode() + open(os.path.join(tok_dir, name), "rb").read())
    for mod in (finetune_single_device, laya.common):
        h.update(open(inspect.getsourcefile(mod), "rb").read())
    for fn in (build_multi_item, _target, _tok_chunk):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:16]


def _cache_read(d):
    """Every shard of a cache directory -> {md5 of a line: its items}; ids are views of the shard's one array."""
    lines = {}
    for meta in sorted(glob.glob(os.path.join(d, "*.meta.pkl"))):
        with open(meta, "rb") as f:
            m = pickle.load(f)
        ids = np.load(meta[:-len(".meta.pkl")] + ".ids.npy")
        for h, its in m.items():
            lines[h] = [dict(rest, ids=ids[o:o + n]) for o, n, rest in its]
    return lines


def _cache_write(d, groups, cached):
    """A new shard with the lines not cached yet; ids first, meta last (a shard counts once its meta exists)."""
    new, seen = [], set()
    for h, its in groups:
        if h not in cached and h not in seen:
            seen.add(h)
            new.append((h, its))
    if not new:
        return
    os.makedirs(d, exist_ok=True)
    meta, arrays, o = {}, [], 0
    for h, its in new:
        meta[h] = []
        for it in its:
            meta[h].append((o, len(it["ids"]), {k: v for k, v in it.items() if k != "ids"}))
            arrays.append(it["ids"])
            o += len(it["ids"])
    base = os.path.join(d, f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{len(new)}")
    with open(base + ".ids.npy.tmp", "wb") as f:
        np.save(f, np.concatenate(arrays) if arrays else np.zeros(0, np.int32))
    os.replace(base + ".ids.npy.tmp", base + ".ids.npy")
    with open(base + ".meta.pkl.tmp", "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(base + ".meta.pkl.tmp", base + ".meta.pkl")


def load_items(model_dir, data, max_len=0, workers=1, multi=False, head_max_len=0, cache=""):
    """preprocess() of a JSONL file in `workers` processes, in file order, token ids as int32 arrays (a million
    ~1000-token items as lists of Python ints would not fit in memory). multi: one build_multi_item per state.
    cache: a directory of tokenised lines keyed by their md5; when every line of `data` is there
    (the file itself or any subset of files tokenised before, e.g. train_g4 of train) nothing is tokenised,
    else the whole file is and its new lines are added. -> (items, "hit" | "miss" | "off")."""
    _fix_tokenizer_config(model_dir)
    cached, d = {}, ""
    if cache:
        d = os.path.join(cache, _cache_key(model_dir, max_len, multi, head_max_len))
        cached = _cache_read(d)
        with open(data, "rb") as f:
            hashes = [hashlib.md5(line).hexdigest() for line in f.read().splitlines()]
        if all(h in cached for h in hashes):
            return [it for h in hashes for it in cached[h]], "hit"
    size, chunk, jobs, start = os.path.getsize(data), 32 << 20, [], 0
    with open(data, "rb") as f:
        while start < size:
            end = size
            if start + chunk < size:
                f.seek(start + chunk)
                f.readline()
                end = f.tell()
            jobs.append((data, start, end))
            start = end
    with mp.get_context("fork").Pool(max(1, workers), _tok_init, (model_dir, max_len, multi, head_max_len)) as pool:
        groups = [g for part in pool.imap(_tok_chunk, jobs) for g in part]
    if cache:
        _cache_write(d, groups, cached)
    return [it for _, its in groups for it in its], ("miss" if cache else "off")


def save_model(model, tok, cfg, out_dir):
    """model.safetensors last: the run scripts take its presence for a finished training."""
    os.makedirs(out_dir, exist_ok=True)
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    cfg = dict(cfg, fine_tuned=True, temperature=[1.0, 1.0, 1.0])  # calibrated afterwards on held-out games
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(out_dir, "model.safetensors"))


def save_resume(a, model, opt, sched, scaler, tok, cfg, step):
    """<out>/resume.pt (model + optimiser state, for --resume after a stop) and <out>/snapshot/ (a usable fp16
    checkpoint of the model so far, e.g. to play with); each replaced whole, never half-written."""
    tmp = os.path.join(a.output_dir, "resume.pt.tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "step": step}, tmp)
    os.replace(tmp, os.path.join(a.output_dir, "resume.pt"))
    snap, new, old = (os.path.join(a.output_dir, s) for s in ("snapshot", "snapshot.tmp", "snapshot.old"))
    shutil.rmtree(new, ignore_errors=True)
    save_model(model, tok, dict(cfg, snapshot_step=step), new)
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(snap):
        os.replace(snap, old)
    os.replace(new, snap)
    shutil.rmtree(old, ignore_errors=True)


def marker_logits(model, input_ids, attention_mask, marker_pos, marker_mask, qtype):
    """DecisionModel.forward's option logits, the last head layer computed only at the markers.

    The scorer reads the head's output at the option markers alone, so the last pre-norm layer needs
    every position only as keys/values; its queries, out_proj and feed-forward run on the markers
    (as Julia-1's `_selected_head`). Eval only: no dropout, no act head (we do not use it)."""
    h = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = h + model.type_emb(qtype)[:, None, :]
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    if model.head is None:
        m = torch.gather(h, 1, idx)
    else:
        pad = ~attention_mask.bool()
        for layer in model.head.layers[:-1]:
            h = layer(h, src_key_padding_mask=pad)
        layer = model.head.layers[-1]
        attn, d = layer.self_attn, h.size(-1)
        x = layer.norm1(h)
        q = torch.nn.functional.linear(torch.gather(x, 1, idx), attn.in_proj_weight[:d], attn.in_proj_bias[:d])
        k, v = torch.nn.functional.linear(x, attn.in_proj_weight[d:], attn.in_proj_bias[d:]).chunk(2, -1)
        heads = lambda t: t.unflatten(-1, (attn.num_heads, attn.head_dim)).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(heads(q), heads(k), heads(v),
                                                             attn_mask=~pad[:, None, None, :])
        m = torch.gather(h, 1, idx) + attn.out_proj(o.transpose(1, 2).flatten(-2))
        m = m + layer.linear2(layer.activation(layer.linear1(layer.norm2(m))))
    return model.scorer(m).squeeze(-1).float().masked_fill(~marker_mask, -1e4)


def multi_logits(model, input_ids, attention_mask, tok_type, marker_pos, q_index, marker_mask):
    """Option logits (Q, K) of collate_multi's questions: DecisionModel.forward with the question-type embedding
    added per block (none on [CLS] and the state) instead of one type for the whole sequence."""
    h = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = h + model.type_emb(tok_type.clamp(min=0)) * (tok_type >= 0)[..., None]
    pad = ~attention_mask.bool()
    for layer in model.head.layers:
        h = layer(h, src_key_padding_mask=pad)
    m = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
    z = model.scorer(m).squeeze(-1).float().flatten()
    return z[q_index].masked_fill(~marker_mask, -1e4)


def logits_of(model, batch, device):
    if "q_index" in batch:
        args = [batch[k].to(device) for k in ("input_ids", "attention_mask", "tok_type", "marker_pos", "q_index",
                                               "marker_mask")]
        with torch.autocast("cuda", dtype=torch.float16):
            return multi_logits(model, *args)
    args = [batch[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    with torch.autocast("cuda", dtype=torch.float16):
        if model.training:
            return model(*args)[0].float()
        return marker_logits(model, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8,
                    help="random states per step, as the repo script; 0: length-grouped steps of --max-tokens "
                         "(faster; risk-v1: 0.908 vs 0.926 AUC at the same learning rate)")
    ap.add_argument("--max-tokens", type=int, default=8192, help="padded tokens per step (~21 GiB on ModernBERT-large)")
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (throughput tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["ce", "rlcd"], default="ce")
    ap.add_argument("--rps", type=float, default=0,
                    help="weight of the ranked probability score added to CE on score questions (ordered options; "
                         "the repo's RLCD reward has it at w_rps=1.0)")
    ap.add_argument("--multi", action="store_true",
                    help="every question of a state in one sequence (build_multi_item); --batch then counts states. "
                         "Recorded in the checkpoint (multi_question), predict.py follows it")
    ap.add_argument("--pack", action="store_true",
                    help="padding-free encoder (packed.py: the batch's sequences in one row, varlen flash attention)")
    ap.add_argument("--compile", choices=["default", "reduce-overhead", "off"], default="default")
    ap.add_argument("--pad-multiple", type=int, default=64)
    ap.add_argument("--keep-layers", type=int, default=0,
                    help="keep only the first N encoder layers (top-layer dropping: Poor Man's BERT, smolaya kept 20 of 28)")
    ap.add_argument("--max-len", type=int, default=0, help="tokens of question + state (default: the checkpoint's)")
    ap.add_argument("--head-max-len", type=int, default=0,
                    help="tokens of the question and its options (default: the checkpoint's; end v7 needs 512)")
    ap.add_argument("--workers", type=int, default=1, help="tokenizer processes")
    ap.add_argument("--save-epochs", action="store_true", help="a model copy in <output-dir>/epochN after each epoch")
    ap.add_argument("--save-every", type=float, default=0,
                    help="minutes between resumable saves (resume.pt, snapshot/; a rerun continues from them)")
    ap.add_argument("--tok-cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokcache"),
                    help="directory of tokenised lines reused across runs (load_items); '' to tokenise afresh")
    a = ap.parse_args()

    t0 = time.time()
    items, how = load_items(a.model_dir, a.data, a.max_len, a.workers, a.multi, a.head_max_len,
                            a.tok_cache)  # before CUDA: pool forks
    print(f"{len(items)} items, preprocess {time.time() - t0:.0f}s (cache {how})", flush=True)

    device = torch.device("cuda", 0)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    cfg, tok, model = load_model(a.model_dir, device)
    cfg["max_len"] = a.max_len or cfg["max_len"]
    cfg["head_max_len"] = a.head_max_len or cfg["head_max_len"]  # saved with the model: predict.py reads it there
    cfg["multi_question"] = a.multi
    if a.keep_layers:
        ec = model.encoder.config
        model.encoder.layers = model.encoder.layers[:a.keep_layers]
        ec.num_hidden_layers, ec.layer_types = a.keep_layers, ec.layer_types[:a.keep_layers]
    model.train()
    if a.pack:
        import packed
        packed.enable(model, cfg["max_len"] + 64, None if a.compile == "off" else a.compile)
    if a.compile != "off":
        torch._dynamo.config.recompile_limit = 64  # one graph per (rows, length bucket); the default 8 falls back to eager
        if a.pack:  # the encoder is compiled by packed.enable; DecisionModel.forward packs (data-dependent shapes)
            for part in model.head.layers:
                part.compile(mode=a.compile, dynamic=False)
        elif a.multi:  # multi_logits calls the parts, not DecisionModel.forward
            for part in (model.encoder, *model.head.layers):
                part.compile(mode=a.compile, dynamic=False)
        else:
            model.compile(mode=a.compile, dynamic=False)  # in place: state_dict keys stay as they are

    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr_encoder}, {"params": head, "lr": a.lr_head}],
                            weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    def batches(r):
        if a.batch:
            idx = list(range(len(items)))
            r.shuffle(idx)
            return [idx[j:j + a.batch] for j in range(0, len(idx), a.batch)]
        return length_batches(items, a.max_tokens, r)

    per_epoch = len(batches(random.Random(a.seed)))
    total = a.max_steps or per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)
    group = 4
    start = 0
    resume = os.path.join(a.output_dir, "resume.pt")
    if a.save_every and os.path.exists(resume):
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start = st["step"]
        del st
        print(f"resumed at step {start}", flush=True)
    os.makedirs(a.output_dir, exist_ok=True)
    step, tokens, t0 = 0, 0, time.time()
    t_win, t_save, run_loss, run_n = t0, t0, 0.0, 0
    for epoch in range(a.epochs):
        sigma = 0.4 + (0.1 - 0.4) * (epoch / max(1, a.epochs - 1))
        for b in batches(rng):  # the same order on a rerun: the steps before `start` are skipped
            if step < start:
                step += 1
                continue
            batch = (collate_multi if a.multi else collate)([items[i] for i in b], tok.pad_token_id)
            batch["pad_id"] = tok.pad_token_id
            batch = pad_to(batch, a.pad_multiple)
            logits = logits_of(model, batch, device)
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            loss = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            sc = batch["qtype"].to(device) == 1  # QTYPES["score"]: options are ordered
            if a.rps and sc.any():
                p = torch.softmax(logits.masked_fill(~mask, -1e4)[sc], -1)
                d = p.cumsum(-1) - target[sc].cumsum(-1)
                k = (mask[sc].sum(-1) - 1).clamp(min=1)  # RPS over K options: K-1 cumulative terms
                loss = loss + a.rps * ((d[:, :-1] ** 2).sum(-1) / k).sum() / len(logits)
            if a.loss == "rlcd":
                k = mask.sum(-1, keepdim=True).float()
                eps = torch.randn((group,) + logits.shape, device=device) * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
                z = logits.detach().unsqueeze(0) + eps
                q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
                with torch.no_grad():
                    r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                    adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
                logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
                loss = loss - (adv * logp).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens += int(batch["attention_mask"].sum())
            run_loss += loss.detach()  # no per-step host sync
            run_n += 1
            if step % 200 == 0:  # rates over these 200 steps
                dt = time.time() - t_win
                print(f"epoch {epoch + 1} step {step}/{total} loss {float(run_loss) / run_n:.4f} | {run_n / dt:.2f} steps/s "
                      f"{tokens / dt:.0f} tok/s | {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
                run_loss, run_n, tokens, t_win = 0.0, 0, 0, time.time()
            if a.save_every and time.time() - t_save > a.save_every * 60:
                save_resume(a, model, opt, sched, scaler, tok, cfg, step)
                print(f"resumable save at step {step}", flush=True)
                t_save = time.time()
            if step >= total:
                break
        # a copy after each epoch but the last (the curve's stage 3: when repeats turn into memorization);
        # skipped when a resume began past this epoch's end (the copy was written then)
        if a.save_epochs and epoch + 1 < a.epochs and step > start:
            save_model(model, tok, cfg, os.path.join(a.output_dir, f"epoch{epoch + 1}"))
            print(f"saved epoch {epoch + 1} copy at step {step}", flush=True)
        if step >= total:
            break

    save_model(model, tok, cfg, a.output_dir)
    print(f"saved {a.output_dir} after {step} steps, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
