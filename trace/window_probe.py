"""What the model sees: one state of a dataset verbatim, and the tokens per section of the state text over a sample
(sections start at lines like "Inventory (as last shown): ..." / "Map around the hero (@):"), with the share of
states over a few windows. The tokenizer is the checkpoint's.
  python window_probe.py <ckpt> <cases.jsonl> [--n 2000] [--show 1]"""
import argparse
import json
import random
import re

import numpy as np
from transformers import AutoTokenizer

HEAD = re.compile(r"^([A-Z][A-Za-z ()@/'-]{1,40}?)(:|\.)( |$)")


def sections(state):
    out, name = {}, "start"
    for line in state.split("\n"):
        m = HEAD.match(line)
        if m and not line.startswith(" "):
            name = m.group(1).split(" (")[0]
        out[name] = out.get(name, "") + line + "\n"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("cases")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--show", type=int, default=1)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(f"{a.ckpt}/tokenizer")
    lines = open(a.cases, encoding="utf-8").read().splitlines()
    rows = [json.loads(x) for x in random.Random(0).sample(lines, min(a.n, len(lines)))]
    for r in rows[:a.show]:
        print("=" * 30, r["id"], "| questions:", json.dumps(r["questions"])[:400])
        print(r["state"])
    total, per = [], {}
    for r in rows:
        total.append(len(tok(r["state"], add_special_tokens=False)["input_ids"]))
        for k, v in sections(r["state"]).items():
            per.setdefault(k, []).append(len(tok(v, add_special_tokens=False)["input_ids"]))
    t = np.array(total)
    print(f"\n{len(rows)} states: tokens median {np.median(t):.0f}, mean {t.mean():.0f}, p90 {np.percentile(t, 90):.0f}, "
          f"p99 {np.percentile(t, 99):.0f}, max {t.max()}")
    print("  over: " + ", ".join(f"{w} {100 * (t > w).mean():.1f} %" for w in (768, 1024, 1152, 1536, 2048)))
    print("  section (share of states with it): mean / median / p90 tokens")
    for k, v in sorted(per.items(), key=lambda kv: -np.mean(kv[1])):
        v = np.array(v)
        print(f"    {k[:34]:34s} ({100 * len(v) / len(rows):3.0f} %)  {v.mean():6.0f} {np.median(v):6.0f} "
              f"{np.percentile(v, 90):6.0f}")


if __name__ == "__main__":
    main()
