"""Padding-free encoder for finetune_fast.py / predict.py: the sequences of a batch are packed into one
row, so no padding token goes through the encoder (47 % of A's step tokens were padding: pad_waste.py).
Attention: torch 2.14's varlen_attn (FlashAttention with cu_seqlens, fwd + bwd, fp16 / bf16). ModernBERT's local
layers keep their window: the sdpa mask allows |q - k| <= sliding_window (masking_utils'
sliding_window_bidirectional_overlay); the attention module holds w = sliding_window + 1 and flash takes (w - 1, w - 1).
The packed row is padded to the next multiple of BUCKET by one extra segment (1..BUCKET tokens), so torch.compile
sees few shapes; max_length is a constant for the same reason.
  enable(model, max_len) -> model.encoder's forward takes and returns the padded tensors as before (state_dict keys
  unchanged); compile the encoder layers, not model.forward (the packing indices are data-dependent).
"""
import types

import torch
from torch.nn.attention.varlen import varlen_attn
from transformers import AttentionInterface

BUCKET = 256


def _varlen(module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None,
            cu_seq_lens_q=None, max_length_q=None, **kwargs):
    """(1, H, T, D) packed q / k / v -> (1, T, H, D), as transformers' sdpa_attention_forward returns"""
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else value.dtype
    q, k, v = (t[0].transpose(0, 1).to(dt) for t in (query, key, value))  # rotary may have promoted q, k to fp32
    w = (sliding_window - 1, sliding_window - 1) if sliding_window else (-1, -1)
    out = varlen_attn(q, k, v, cu_seq_lens_q, cu_seq_lens_q, max_length_q, max_length_q, scale=scaling, window_size=w)
    return out.unsqueeze(0), None


AttentionInterface.register("laya_varlen", _varlen)


def _forward(self, input_ids=None, attention_mask=None, **kw):
    b, l = input_ids.shape
    m = attention_mask.bool()
    flat = m.flatten().nonzero().squeeze(1)
    n = flat.numel()
    pad = (n // BUCKET + 1) * BUCKET - n  # >= 1: the last segment is never empty
    dev = input_ids.device
    ids = torch.cat([input_ids.flatten()[flat], input_ids.new_full((pad,), self.config.pad_token_id)])
    pos = torch.cat([(m.cumsum(1) - 1).flatten()[flat], torch.arange(pad, device=dev)])
    cu = torch.zeros(b + 2, dtype=torch.int32, device=dev)
    cu[1:b + 1] = m.sum(1).cumsum(0)
    cu[b + 1] = n + pad
    h = self._padded_forward(input_ids=ids[None], position_ids=pos[None],
                             attention_mask={"full_attention": None, "sliding_attention": None},
                             cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=self._max_len,
                             max_length_k=self._max_len).last_hidden_state[0]
    out = h.new_zeros(b * l, h.size(-1))
    out[flat] = h[:n]
    return types.SimpleNamespace(last_hidden_state=out.view(b, l, -1))


def enable(model, max_len, compile_mode=None):
    """model: laya's DecisionModel; max_len: >= the longest sequence and >= BUCKET. compile_mode: torch.compile the
    encoder's own forward on the packed row, dynamic in its length (static shapes would compile once per bucket and
    layer type, ~70 graphs)"""
    enc = model.encoder
    enc.config._attn_implementation = "laya_varlen"
    enc._padded_forward = enc.forward
    if compile_mode:
        enc._padded_forward = torch.compile(enc._padded_forward, mode=compile_mode, dynamic=True)
    enc._max_len = max(int(max_len), BUCKET)
    enc.forward = types.MethodType(_forward, enc)
    return model
