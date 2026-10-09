"""Tabular baselines for the humans' questions (human_build.py: fate, ttd) -- the yardstick a laya model on the same
held-out players has to beat. Features parsed from the state text: HP, max HP, Dlvl, deepest, XL, AC, turn, game
version, alignment / gender, inventory size and the item kinds that decide a game (wands, rings, amulets, reflection,
magic resistance, ...). Models: the train prior, multinomial logit, gradient boosting (sklearn). With --laya, also a
laya model's raw logits (predict.py) on the same rows. Log loss, accuracy and RPS (ranked probability score: both
questions are ordered) with a bootstrap over players; each laya model after the first also against the first.
  python human_baseline.py --data human_data [--laya preds/nh-v1-human-test.parquet ...] --out results/human_baseline
-> <out>.txt, <out>.parquet (test rows: id, question, label, each model's probabilities)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HEAD = re.compile(r"HP (-?\d+)/(\d+)\. Dlvl (\d+) \(deepest so far (\d+)\), XL (\d+), AC (-?\d+), turn (\w+)")
ITEMS = {"wand": r"\bwand\b", "ring": r"\bring\b", "amulet": r"\bamulet\b", "potion": r"\bpotions?\b",
         "scroll": r"\bscrolls?\b", "spellbook": r"\bspellbook\b", "bag": r"\b(bag|sack)\b", "worn": r"being worn",
         "blessed": r"\bblessed\b", "cursed": r"\buncursed\b", "unihorn": r"unicorn horn",
         "reflection": r"reflection|silver dragon scale", "mr": r"magic resistance|gray dragon scale",
         "lifesave": r"life saving", "speed": r"speed boots", "excalibur": r"Excalibur", "dragon": r"dragon scale",
         "mithril": r"mithril", "marker": r"magic marker", "digging": r"pick-axe|mattock|wand of digging"}
ITEM_RE = {k: re.compile(v, re.I) for k, v in ITEMS.items()}
GOLD = re.compile(r"\$:(\d+)")


def features(state):
    m = HEAD.search(state)
    hp, maxhp, dlvl, deepest, xl, ac, turn = m.groups()
    inv = state.split("Inventory (as last shown): ", 1)[1].split("\nLevel map:", 1)[0]
    items = [] if inv == "unknown" else inv.split("; ")
    g = GOLD.search(state)
    f = dict(hp=int(hp), maxhp=int(maxhp), hp_frac=int(hp) / max(int(maxhp), 1), dlvl=int(dlvl), deepest=int(deepest),
             xl=int(xl), ac=int(ac), turn=int(turn) if turn.isdigit() else -1, turn_known=turn.isdigit(),
             v34=" 3.4" in state[:120], chaotic="-Cha-" in state[:60], male="-Mal." in state[:60],
             inv_known=inv != "unknown", n_items=len(items), gold=int(g.group(1)) if g else 0)
    f.update({k: sum(bool(r.search(x)) for x in items) for k, r in ITEM_RE.items()})
    return f


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        f = features(r["state"])
        for q, g in r["gold"].items():
            # "Character: Val-Dwa-Law-Fem." -> Dwa
            rows.append(dict(f, id=r["id"], question=q, label=int(g["label"]), race=r["state"].split("-")[1]))
    return pd.DataFrame(rows)


def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def draws(players, rng, n=200):
    """bootstrap resamples of the test rows by player (shared by all models: paired differences)"""
    codes, uniq = pd.factorize(players)
    idx = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    return [np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(n)]


def ci(x, boot):
    return np.percentile([x[t].mean() for t in boot], [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="human_data")
    ap.add_argument("--laya", nargs="*", default=[])
    ap.add_argument("--out", default="../results/human_baseline")
    ap.add_argument("--split", default="test", help="held-out file of --data to score (val: choosing between models)")
    ap.add_argument("--race", default="", help="score only the held-out rows of this race (Dwa, Hum, ...)")
    a = ap.parse_args()
    tr, te = load(Path(a.data, "train.jsonl")), load(Path(a.data, f"{a.split}.jsonl"))
    if a.race:
        te = te[te.race == a.race]
    cols = [c for c in tr.columns if c not in ("id", "question", "label", "race")]
    rng = np.random.default_rng(0)
    out = [f"human questions: train {tr.id.nunique()} states, {a.split} {a.race} {te.id.nunique()} states "
           f"({te.id.str.split('@').str[0].nunique()} players); features: {', '.join(cols)}"]
    pred = []
    for q in sorted(te.question.unique()):
        a_tr, a_te = tr[tr.question == q], te[te.question == q].copy()
        k = a_tr.label.max() + 1
        X, Xt, y, yt = a_tr[cols].astype(float), a_te[cols].astype(float), a_tr.label.values, a_te.label.values
        probs = {"prior": np.tile(np.bincount(y, minlength=k) / len(y), (len(yt), 1))}
        probs["logit"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y).predict_proba(Xt)
        probs["gbm"] = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True,
                                                      random_state=0).fit(X, y).predict_proba(Xt)
        for path in a.laya:
            lp = pd.read_parquet(path)
            lp = lp[lp.question == q].set_index("id").logits
            probs[Path(path).stem] = softmax(np.stack(lp.reindex(a_te.id).values))
        boot = draws(a_te.id.str.split("@").str[0].values, rng)
        out += ["", f"== {q}: {len(a_te)} test rows, labels {np.bincount(yt, minlength=k).tolist()} "
                    f"(gains vs gbm / vs the first laya model: positive = better)"]
        keep = a_te[["id", "question", "label"]].copy()
        lls, rpss = {}, {}
        first = Path(a.laya[0]).stem if a.laya else None
        for name, p in probs.items():
            ll = lls[name] = -np.log(np.clip(p[np.arange(len(yt)), yt], 1e-12, 1))
            acc = (p.argmax(1) == yt).astype(float)
            rps = rpss[name] = ((p.cumsum(1) - (np.arange(k) >= yt[:, None]))[:, :-1] ** 2).sum(1) / (k - 1)
            (llo, lhi), (alo, ahi), (rlo, rhi) = ci(ll, boot), ci(acc, boot), ci(rps, boot)
            d = ""
            if name not in ("prior", "logit", "gbm"):
                g = lls["gbm"] - ll
                lo, hi = ci(g, boot)
                d = f"  vs gbm {g.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
                if name != first:
                    g, r = lls[first] - ll, rpss[first] - rps
                    (lo, hi), (rlo2, rhi2) = ci(g, boot), ci(r, boot)
                    d += (f"  vs {first}: log loss {g.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], "
                          f"RPS {r.mean():+.4f} [{rlo2:+.4f}, {rhi2:+.4f}]")
            out.append(f"  {name:22s} log loss {ll.mean():.4f} [{llo:.4f}, {lhi:.4f}]  accuracy {acc.mean():.3f} "
                       f"[{alo:.3f}, {ahi:.3f}]  RPS {rps.mean():.4f} [{rlo:.4f}, {rhi:.4f}]{d}")
            for i in range(k):
                keep[f"{name}_{i}"] = p[:, i]
        pred.append(keep)
    text = "\n".join(out) + "\n"
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".txt").write_text(text, encoding="utf-8")
    pd.concat(pred).to_parquet(a.out + ".parquet", index=False)


if __name__ == "__main__":
    main()
