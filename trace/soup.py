"""Uniform weight soup (Wortsman et al. 2022) of checkpoints fine-tuned from the same weights: every float tensor is
the mean over the checkpoints (summed in fp32, saved in the original dtype); config, encoder config and tokenizer are
taken from the first one and must match in the others.
  python soup.py <out dir> <ckpt> <ckpt> [...]
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    out, ckpts = Path(sys.argv[1]), [Path(c) for c in sys.argv[2:]]
    assert len(ckpts) >= 2 and not out.exists()
    cfgs = [json.loads((c / "rl_agent_config.json").read_text()) for c in ckpts]
    for c, cfg in zip(ckpts[1:], cfgs[1:]):
        assert cfg == cfgs[0], f"{c}: rl_agent_config differs"
        assert (c / "encoder/config.json").read_bytes() == (ckpts[0] / "encoder/config.json").read_bytes()
    files = [safe_open(c / "model.safetensors", "pt") for c in ckpts]
    keys = sorted(files[0].keys())
    assert all(sorted(f.keys()) == keys for f in files[1:])
    soup = {}
    for k in keys:
        ts = [f.get_tensor(k) for f in files]
        assert all(t.shape == ts[0].shape and t.dtype == ts[0].dtype for t in ts)
        if ts[0].is_floating_point():
            soup[k] = (sum(t.float() for t in ts) / len(ts)).to(ts[0].dtype)
        else:
            assert all(torch.equal(t, ts[0]) for t in ts[1:]), k
            soup[k] = ts[0]
    out.mkdir(parents=True)
    shutil.copytree(ckpts[0] / "encoder", out / "encoder")
    shutil.copytree(ckpts[0] / "tokenizer", out / "tokenizer")
    cfg = dict(cfgs[0], soup_of=[c.name for c in ckpts])
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    save_file(soup, str(out / "model.safetensors"))
    print(f"{out}: {len(keys)} tensors, mean of {', '.join(c.name for c in ckpts)}")


if __name__ == "__main__":
    main()
