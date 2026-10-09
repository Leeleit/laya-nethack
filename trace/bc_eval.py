"""How often laya's answer is the bot's on held-out games (bc_build.py test split, predict.py logits).

Test rows were kept with the same per-command shares as train (bc_vocab.KEEP), so each row is weighted by
1 / KEEP[its command] -- accuracy at the natural rate of commands -- and the command probabilities are multiplied
by UNDO = 1 / KEEP, as laya_play.py does. Per question: top-1 / top-3 agreement and log loss; for `command` also the recall
of each command and what it is mistaken for, and the copycat check: agreement where the bot repeats the hero's
previous command and where it changes it, and how often laya repeats it there.
  python bc_eval.py <test.jsonl> <preds.parquet> [--out results/bc_eval.txt]
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bc_vocab import KEEP, UNDO

LAST = re.compile(r"Last commands: (.*)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("preds")
    ap.add_argument("--out", default="results/bc_eval.txt")
    a = ap.parse_args()
    rows = {}
    for line in open(a.data, encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r
    p = pd.read_parquet(a.preds)
    stats = collections.defaultdict(lambda: collections.defaultdict(float))
    confusion = collections.defaultdict(collections.Counter)
    for rid, q, z in zip(p.id, p.question, p.logits):
        r = rows[rid]
        spec, gold = r["questions"][q], r["gold"][q]["label"]
        opts = ["false", "true"] if spec["type"] == "noul" else list(spec["criteria"])
        pr = np.exp(np.asarray(z) - max(z))
        if q == "command":
            pr *= [UNDO[o] for o in opts]
        pr /= pr.sum()
        w = 1 / KEEP[r["gold"]["command"]["label"]]
        g = opts.index(gold)
        order = np.argsort(-pr)
        s = stats[q]
        s["w"] += w
        s["n"] += 1
        s["top1"] += w * (order[0] == g)
        s["top3"] += w * (g in order[:3])
        s["nll"] += -w * np.log(max(pr[g], 1e-12))
        if q == "command":
            stats["command:" + gold]["w"] += w
            stats["command:" + gold]["top1"] += w * (order[0] == g)
            confusion[gold][opts[order[0]]] += w
            m = LAST.search(r["state"])
            prev = m.group(1).split(", ")[-1] if m else "none"
            s = stats["copy:" + ("repeat" if gold == prev else "change")]
            s["w"] += w
            s["n"] += 1
            s["top1"] += w * (order[0] == g)
            s["copied"] += w * (opts[order[0]] == prev)
    lines = ["question: rows, agreement with the bot top-1 / top-3, log loss (natural command rates)"]
    for q in ("command", "item_cmd", "item", "direction", "eat_here"):
        s = stats[q]
        if s["w"]:
            lines.append(f"  {q:10s} {int(s['n']):7d}  {s['top1'] / s['w']:.3f} / {s['top3'] / s['w']:.3f}  {s['nll'] / s['w']:.3f}")
    lines.append("command: share of turns, recall, most common mistakes")
    tot = stats["command"]["w"]
    for c in sorted((k[8:] for k in stats if k.startswith("command:")), key=lambda c: -stats["command:" + c]["w"]):
        s = stats["command:" + c]
        wrong = [(o, v) for o, v in confusion[c].most_common() if o != c][:3]
        lines.append(f"  {c:10s} {s['w'] / tot:7.2%}  {s['top1'] / s['w']:.3f}  "
                     + ", ".join(f"{o} {v / s['w']:.0%}" for o, v in wrong))
    lines.append("command vs the hero's previous command (copycat): share of turns, top-1, laya repeats the previous")
    for k in ("repeat", "change"):
        s = stats["copy:" + k]
        if s["w"]:
            lines.append(f"  bot {k:7s} {s['w'] / tot:7.2%}  {s['top1'] / s['w']:.3f}  {s['copied'] / s['w']:.3f}")
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
