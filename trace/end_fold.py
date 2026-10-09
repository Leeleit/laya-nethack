"""Two v6-family models whose end questions name different monster classes (T14 30 classes, head 512;
T15 all 55, head 1024), on the same held-out states: the wider one's classes the narrower lacks fold into its
"another monster", both scored on the narrower's labels; ttd, depth, ascends on their shared targets. Paired bootstrap
over players (fold_eval.boot / ci); "gain" = A's loss minus B's (positive: B better).
  python end_fold.py --a human_data14d --pa preds/a.parquet --b human_data15d --pb preds/b.parquet [--split val]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fold_eval import boot, ci, ll, softmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--pa", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pb", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ra = {r["id"]: r for r in map(json.loads, open(Path(a.a, f"{a.split}.jsonl"), encoding="utf-8"))}
    rb = {r["id"]: r for r in map(json.loads, open(Path(a.b, f"{a.split}.jsonl"), encoding="utf-8"))}
    pa = pd.read_parquet(a.pa).set_index(["id", "question"]).logits
    pb = pd.read_parquet(a.pb).set_index(["id", "question"]).logits
    ids = [i for i in ra if i in rb]
    rng = np.random.default_rng(0)
    out = [f"{len(ids)} states of {a.split} in both; A {a.pa}, B {a.pb}"]
    for q in ("end", "ttd", "depth", "ascends"):
        sel = [i for i in ids if q in ra[i]["gold"] and q in rb[i]["gold"]]
        ka = list(ra[sel[0]]["questions"][q]["criteria"])
        kb = list(rb[sel[0]]["questions"][q]["criteria"])
        if not isinstance(ra[sel[0]]["questions"][q]["criteria"], dict):
            ka, kb = [str(k) for k in range(len(ka))], [str(k) for k in range(len(kb))]
        y = np.array([ka.index(ra[i]["gold"][q]["label"]) for i in sel])
        p_a = softmax(np.stack([pa[(i, q)] for i in sel]))
        p_b = softmax(np.stack([pb[(i, q)] for i in sel]))
        if kb != ka:  # B names more classes: each of B's options goes to A's same key, else to "monster"
            to = [ka.index(k) if k in ka else ka.index("monster") for k in kb]
            p_b = np.stack([p_b[:, [j for j, t in enumerate(to) if t == c]].sum(1) for c in range(len(ka))], 1)
        b = boot(np.array([i.split("@")[0].lower() for i in sel]), rng)
        la, lb = ll(p_a, y), ll(p_b, y)
        out.append(f"{q:7s} {len(sel)} rows, {len(ka)} options{' (B folded from ' + str(len(kb)) + ')' if kb != ka else ''}"
                   f": A {ci(la, b)}  B {ci(lb, b)}  gain B over A {ci(la - lb, b)}; accuracy A "
                   f"{(p_a.argmax(1) == y).mean():.3f} B {(p_b.argmax(1) == y).mean():.3f}")
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
