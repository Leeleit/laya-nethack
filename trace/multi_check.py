"""Checks of finetune_fast's multi-question mode (build_multi_item / collate_multi / multi_logits) on one GPU:
  1. a row with ONE question and the type embedding on every token (as DecisionModel.forward adds it) must give the
     single-question sequence token for token and the same logits as model(...);
  2. rows with all their questions: every marker sits on [MASK], each block starts with the question's text, the
     state is cut as the longest single question would cut it; the loss reaches the encoder and type_emb;
  3. cost: padded tokens and eval time, one sequence per question vs one per state (same batches of states).
  python multi_check.py <laya checkpoint> <jsonl with state/questions/gold> [rows]
"""
import json
import sys
import time

import torch

from finetune_fast import (build_multi_item, build_training_item, collate, collate_multi, load_model, logits_of,
                           multi_logits, pad_to)

DEV = torch.device("cuda", 0)


def batched(xs, n):
    return [xs[j:j + n] for j in range(0, len(xs), n)]


def main():
    model_dir, data = sys.argv[1], sys.argv[2]
    n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    cfg, tok, model = load_model(model_dir, DEV)
    cfg["max_len"] = 1152
    model.eval()
    rows = [json.loads(line) for line in open(data, encoding="utf-8")][:n_rows]

    # 1. one question per row == the single-question path
    worst, same, n = 0.0, 0, 0
    for chunk in batched(rows, 8):
        singles, multis = [], []
        for r in chunk:
            qid = next(q for q in r["questions"] if q in r["gold"])
            s = build_training_item(tok, cfg, r["state"], r["questions"][qid], r["gold"][qid])
            m = build_multi_item(tok, cfg, r["state"], {qid: r["questions"][qid]}, r["gold"])
            assert s and m and list(s["ids"]) == list(m["ids"]), f"tokens differ: {r['id']} {qid}"
            assert s["markers"] == m["markers"][0] and s["target"] == m["targets"][0], r["id"]
            singles.append(s)
            multis.append(m)
        bs = collate(singles, tok.pad_token_id)
        bm = collate_multi(multis, tok.pad_token_id)
        bm["tok_type"] = torch.where(bm["attention_mask"].bool(), bm["qtype"][:, None], -1)  # type on every token
        with torch.no_grad():
            zs = logits_of(model, bs, DEV)
            zm = logits_of(model, bm, DEV)
        k = zs.shape[1]
        assert zm.shape == (len(chunk), k), zm.shape
        worst = max(worst, float((zs - zm).abs()[bs["marker_mask"].to(DEV)].max()))
        same += int((zs.argmax(-1) == zm.argmax(-1)).sum())
        n += len(chunk)
    print(f"1. one question, type on every token: tokens identical, max |logit diff| vs model(...) {worst:.2e}, "
          f"argmax same {same}/{n}", flush=True)

    # 2. all questions of a row
    multis = [build_multi_item(tok, cfg, r["state"], r["questions"], r["gold"]) for r in rows]
    q_total = 0
    for r, m in zip(rows, multis):
        want = [q for q in r["questions"] if q in r["gold"]]
        assert m["qids"] == want, (r["id"], m["qids"], want)
        state_single = None
        for (s, e, t), mk, qid in zip(m["blocks"], m["markers"], m["qids"]):
            assert all(m["ids"][p] == tok.mask_token_id for p in mk), r["id"]
            single = build_training_item(tok, cfg, r["state"], r["questions"][qid], r["gold"][qid])
            prefix = single["ids"].index(tok.sep_token_id, single["markers"][-1]) + 1  # [CLS] question [SEP] opts [SEP]
            assert list(m["ids"][s:e]) == list(single["ids"][1:prefix]), (r["id"], qid)  # the block, token for token
            assert [p - s for p in mk] == [p - 1 for p in single["markers"]], (r["id"], qid)
            st = single["ids"][prefix:-1]
            state_single = st if state_single is None or len(st) < len(state_single) else state_single
            q_total += 1
        # the state: what the single question with the longest prefix keeps
        assert m["ids"][m["blocks"][-1][1]:-1] == state_single, r["id"]
    model.train()
    b = collate_multi(multis[:2], tok.pad_token_id)
    b["pad_id"] = tok.pad_token_id
    b = pad_to(b, 64)
    z = logits_of(model, b, DEV)
    target = b["target"].to(DEV)
    loss = -(target * torch.log_softmax(z, -1)).sum(-1).mean()
    loss.backward()
    g_enc = model.encoder.layers[0].attn.Wqkv.weight.grad
    g_type = model.type_emb.weight.grad
    print(f"2. {len(rows)} rows, {q_total} questions: markers on [MASK], blocks = the questions' text, state as the "
          f"longest single question cuts it; loss {float(loss):.3f}, grad |encoder L0| {float(g_enc.norm()):.3e}, "
          f"|type_emb| rows used {[(round(float(g_type[t].norm()), 4)) for t in range(3)]}", flush=True)
    model.zero_grad(set_to_none=True)
    model.eval()

    # 3. cost, the same 8-state batches
    singles = [build_training_item(tok, cfg, r["state"], r["questions"][q], r["gold"][q])
               for r in rows for q in r["questions"] if q in r["gold"]]
    order = sorted(range(len(rows)), key=lambda i: len(multis[i]["ids"]))
    tok_s = tok_m = 0
    t_s = t_m = 0.0
    with torch.no_grad():
        for c in batched(order, 8):
            ss = [build_training_item(tok, cfg, rows[i]["state"], rows[i]["questions"][q], rows[i]["gold"][q])
                  for i in c for q in multis[i]["qids"]]
            for part in batched(ss, 8):
                bs = pad_to(collate(part, tok.pad_token_id) | {"pad_id": tok.pad_token_id}, 64)
                tok_s += bs["input_ids"].numel()
                torch.cuda.synchronize(); t = time.time()
                logits_of(model, bs, DEV)
                torch.cuda.synchronize(); t_s += time.time() - t
            bm = pad_to(collate_multi([multis[i] for i in c], tok.pad_token_id) | {"pad_id": tok.pad_token_id}, 64)
            tok_m += bm["input_ids"].numel()
            torch.cuda.synchronize(); t = time.time()
            logits_of(model, bm, DEV)
            torch.cuda.synchronize(); t_m += time.time() - t
    print(f"3. {len(singles)} questions of {len(rows)} states: padded tokens single {tok_s} / multi {tok_m} "
          f"(x{tok_s / tok_m:.2f}); eval time single {t_s:.1f}s / multi {t_m:.1f}s (x{t_s / t_m:.2f})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
