"""v5 questions (fate, ttd) against v6 (ttd, depth, end, ascends) on the same held-out states: v6's finer answers are
folded into v5's classes, so both models are scored on the same targets (human_q6: the bounds 1000 / 5000 turns and
1 / 5 levels are in v6's bins on purpose).
  ttd      -- rows with a turn counter, the game not ascended: v6 bins <= 1000 / <= 5000 / later  vs  v5 ttd
  fate 0-2 -- rows of games not ascended: v6 depth bins <= 1 / <= 5 / more levels  vs  v5 fate 0-2 renormalised
  ascends  -- every row: P(ascends) of v6  vs  P(fate = ascends) of v5 (binary log loss, AUC)
Plus v6's own questions (log loss, accuracy, RPS of the ordinal ones; end against the train prior). Paired bootstrap
over players (200 draws), 95 % intervals; "gain" = v5's loss minus v6's (positive: v6 better).
  python fold_eval.py --v5 human_data5d --v6 human_data6d --split val --p5 preds/a.parquet --p6 preds/e.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def rps(p, y):
    k = p.shape[1]
    return ((p.cumsum(1) - (np.arange(k) >= y[:, None]))[:, :-1] ** 2).sum(1) / (k - 1)


def ll(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)) if 0 < n1 < len(y) else float("nan")


def boot(players, rng, n=200):
    codes, uniq = pd.factorize(players)
    idx = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    return [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(n)]


def ci(x, b):
    lo, hi = np.percentile([x[t].mean() for t in b], [2.5, 97.5])
    return f"{x.mean():.4f} [{lo:.4f}, {hi:.4f}]"


def fold(p, groups):
    return np.stack([p[:, g].sum(1) for g in groups], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True)
    ap.add_argument("--v6", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--p5", required=True)
    ap.add_argument("--p6", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    meta = json.loads(Path(a.v6, "meta.json").read_text())["v6"]
    tb, db = meta["ttd_bounds"], meta["depth_gain_bounds"]
    r5 = {r["id"]: r for r in map(json.loads, open(Path(a.v5, f"{a.split}.jsonl"), encoding="utf-8"))}
    r6 = {r["id"]: r for r in map(json.loads, open(Path(a.v6, f"{a.split}.jsonl"), encoding="utf-8"))}
    p5 = pd.read_parquet(a.p5).set_index(["id", "question"]).logits
    p6 = pd.read_parquet(a.p6).set_index(["id", "question"]).logits
    ids = [i for i in r6 if i in r5]
    rng = np.random.default_rng(0)
    out = [f"{len(ids)} states of {a.split} in both; v5 {a.p5}, v6 {a.p6}"]

    def section(name, sel, y, pa, pb, extra=None):
        sel = np.asarray(sel)
        pl = np.array([i.split("@")[0].lower() for i in sel])
        b = boot(pl, rng)
        la, lb = ll(pa, y), ll(pb, y)
        out.append(f"\n== {name}: {len(y)} rows, labels {np.bincount(y, minlength=pa.shape[1]).tolist()}")
        for tag, p, l_ in (("v5", pa, la), ("v6", pb, lb)):
            out.append(f"  {tag} log loss {ci(l_, b)}  accuracy {(p.argmax(1) == y).mean():.3f}  RPS {ci(rps(p, y), b)}")
        out.append(f"  gain v6 over v5: log loss {ci(la - lb, b)}, RPS {ci(rps(pa, y) - rps(pb, y), b)}")
        if extra:
            out.append(extra)

    # ttd: rows with a turn counter, not ascended
    t_lo = [i for i, x in enumerate(tb) if x <= 1000]
    t_mid = [i for i, x in enumerate(tb) if 1000 < x <= 5000]
    t_hi = sorted(set(range(len(tb) + 1)) - set(t_lo) - set(t_mid))
    sel = [i for i in ids if "ttd" in r5[i]["gold"] and "ttd" in r6[i]["gold"] and r6[i]["gold"]["ascends"]["label"] == "false"]
    y = np.array([int(r5[i]["gold"]["ttd"]["label"]) for i in sel])
    pa = softmax(np.stack([p5[(i, "ttd")] for i in sel]))
    pb = fold(softmax(np.stack([p6[(i, "ttd")] for i in sel])), [t_lo, t_mid, t_hi])
    section("ttd (v5 classes)", sel, y, pa, pb)

    # fate 0-2: games not ascended
    d_lo = [i for i, x in enumerate(db) if x <= 1]
    d_mid = [i for i, x in enumerate(db) if 1 < x <= 5]
    d_hi = sorted(set(range(len(db) + 1)) - set(d_lo) - set(d_mid))
    sel = [i for i in ids if r6[i]["gold"]["ascends"]["label"] == "false"]
    y = np.array([int(r5[i]["gold"]["fate"]["label"]) for i in sel])
    pa = softmax(np.stack([p5[(i, "fate")] for i in sel]))[:, :3]
    pa = pa / pa.sum(1, keepdims=True)
    pb = fold(softmax(np.stack([p6[(i, "depth")] for i in sel])), [d_lo, d_mid, d_hi])
    section("fate 0-2 (games not ascended; v5 renormalised, v6 depth folded)", sel, y, pa, pb)

    # ascends
    y = np.array([r6[i]["gold"]["ascends"]["label"] == "true" for i in ids]).astype(int)
    qa = softmax(np.stack([p5[(i, "fate")] for i in ids]))[:, 3]
    qb = softmax(np.stack([p6[(i, "ascends")] for i in ids]))[:, 1]  # noul: [false, true]
    pa, pb = np.stack([1 - qa, qa], 1), np.stack([1 - qb, qb], 1)
    section("ascends (binary)", ids, y, pa, pb, f"  AUC v5 {auc(qa, y):.3f}  v6 {auc(qb, y):.3f}")

    # v6's own questions
    out.append("\n== v6 questions on their own targets (hard label; RPS for the ordinal ones)")
    for q in ("ttd", "depth", "end"):
        sel = [i for i in ids if q in r6[i]["gold"]]
        keys = list(r6[sel[0]]["questions"][q]["criteria"])
        keys = keys if isinstance(r6[sel[0]]["questions"][q]["criteria"], dict) else [str(k) for k in range(len(keys))]
        y = np.array([keys.index(r6[i]["gold"][q]["label"]) for i in sel])
        p = softmax(np.stack([p6[(i, q)] for i in sel]))
        b = boot(np.array([i.split("@")[0].lower() for i in sel]), rng)
        prior = np.bincount(y, minlength=len(keys)) / len(y)  # held-out prior: an optimistic floor
        line = (f"  {q:6s} {len(sel)} rows, {len(keys)} options: log loss {ci(ll(p, y), b)} (held-out prior "
                f"{ll(np.tile(prior, (len(y), 1)), y).mean():.4f}), accuracy {(p.argmax(1) == y).mean():.3f}")
        if q != "end":
            line += f", RPS {ci(rps(p, y), b)}"
        out.append(line)
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
