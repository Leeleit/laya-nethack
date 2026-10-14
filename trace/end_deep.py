"""Where a question (end by default) wins or loses by Dlvl band: per true class, its rows' loss for each model (with T fitted on the
other half of the players, as curve_eval.py) and its share of the band's difference to the reference model, with a
95 % bootstrap over players; each class's predicted share against the observed one; the end labels of the same bands in
a training set, Val-Dwa against the other roles.
  python end_deep.py human_dwa19h dwa=c17v mix=m17 staged=s17 others=o17 --train human_mix17 [--out ../results/x.txt]
  python end_deep.py human_dwa20t rate=c17r gbm=gbm-c17r --question depth  (score: classes are the relative bins)
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from curve_eval import BANDS, boot_idx, ci, crossfit, labels, load_rows, softmax

TOP = 12


def band_of(dlvl):
    return next((f"{lo}-{hi}" for lo, hi in BANDS if lo <= dlvl <= hi), "?")


def train_shares(path, q):
    cnt = collections.defaultdict(collections.Counter)  # (role, band) -> end label
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if q not in r["gold"]:
                continue
            dl = re.search(r"\bDlvl (\d+)", r["state"])
            role = "Val-Dwa" if r["state"].startswith("Val-Dwa-") else "other roles"
            cnt[(role, band_of(int(dl.group(1)) if dl else -1))][r["gold"][q]["label"]] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("models", nargs="+", help="label=model; the first is the reference")
    ap.add_argument("--train", default="")
    ap.add_argument("--question", default="end")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = load_rows(a.data)
    q = a.question
    sel, y, k = labels(rows, q)
    crit = rows[sel[0]]["q"][q]["criteria"]
    keys = list(crit) if isinstance(crit, dict) else [str(c) for c in range(len(crit))]
    # a score question's option texts depend on the row (absolute Dlvl); its classes are the relative bins
    crit = crit if isinstance(crit, dict) else {c: f"bin {c}" for c in keys}
    pl = np.array([rows[i]["player"] for i in sel])
    band = np.array([band_of(rows[i]["dlvl"]) for i in sel])
    specs = [m.split("=", 1) for m in a.models]
    loss, prob = {}, {}
    for lab, m in specs:
        df = pd.concat([pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{s}.parquet") for s in ("val", "test")])
        lg = df[df.question == q].set_index("id").logits
        z = np.stack([np.asarray(lg[rows[i]["id"]], float) for i in sel])
        loss[lab], t = crossfit(z, y, pl)
        prob[lab] = softmax(z / t)
    ref = specs[0][0]
    out = [f"{q}: {len(y)} held-out rows, {len(set(pl))} players; reference {ref}; models "
           + ", ".join(f"{lab}={m}" for lab, m in specs)]
    tr = train_shares(Path(a.train, "train.jsonl"), q) if a.train else {}
    for b in [f"{lo}-{hi}" for lo, hi in BANDS] + ["21-29"]:
        m = (band == b) if b != "21-29" else np.isin(band, ["21-24", "25-29"])
        if not m.sum():
            continue
        yb, bb = y[m], boot_idx(pl[m])
        out.append(f"\n== Dlvl {b}: {m.sum()} rows, {len(set(pl[m]))} players; mean loss "
                   + "  ".join(f"{lab} {loss[lab][m].mean():.4f}" for lab, _ in specs))
        obs = np.bincount(yb, minlength=k) / len(yb)
        for lab, _ in specs[1:]:
            d = loss[lab][m] - loss[ref][m]
            out.append(f"  {lab} - {ref}: {ci(d, bb)}; by true class (share of the difference, its rows' loss "
                       f"{ref} / {lab}):")
            contrib = [(abs((d * (yb == c)).sum()), c) for c in range(k) if (yb == c).any()]
            for _, c in sorted(contrib, reverse=True)[:TOP]:
                rc = yb == c
                out.append(f"    {keys[c]!s:3s} n {rc.sum():4d}  {ci(d * rc, bb)}  "
                           f"{loss[ref][m][rc].mean():.3f} / {loss[lab][m][rc].mean():.3f}  {crit[keys[c]][:44]}")
        out.append("  share of each class: observed | predicted " + " | ".join(lab for lab, _ in specs)
                   + (" | train Val-Dwa | train other roles" if tr else ""))
        trb = {}
        for role in ("Val-Dwa", "other roles"):
            c = sum((tr.get((role, x), collections.Counter()) for x in (["21-24", "25-29"] if b == "21-29" else [b])),
                    collections.Counter())
            trb[role] = (c, max(sum(c.values()), 1))
        order = np.argsort(-obs)
        for c in order[:TOP + 6]:
            line = f"    {keys[c]!s:3s} {obs[c]:6.3f} | " + " | ".join(f"{prob[lab][m][:, c].mean():.3f}" for lab, _ in specs)
            if tr:
                line += " | " + " | ".join(f"{trb[r][0][keys[c]] / trb[r][1]:.3f}" for r in ("Val-Dwa", "other roles"))
            out.append(line + f"  {crit[keys[c]][:40]}")
        if tr:
            out.append("    train rows: " + ", ".join(f"{r} {trb[r][1]}" for r in ("Val-Dwa", "other roles")))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
