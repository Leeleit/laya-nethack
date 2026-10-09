"""Tabular yardstick for the learning curve (the curve's shares): a gradient-boosted model per question
(ttd, depth, end, ascends) on features of the terse state (HP now / 10 / 30 turns ago, Dlvl, deepest, XL, AC, turn,
stats, gold, kinds of items, what is around) and the player's strength in the question (experience / result levels,
no history), trained on each share of train; predictions on val + test in predict.py's format (id, question,
logits = log p over the question's options in their order) -> curve_eval.py takes them as models.
  python human_tab.py human_data16d all=train quarter=train_g4 ... [--workers 7 --threads 4]
"""
import argparse
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HEAD = re.compile(r"HP (-?\d+)/(\d+) \(10 turns ago (-?\d+)/\d+, 30 ago (-?\d+)/\d+\) Dlvl (\d+) deepest (\d+) "
                  r"XL (\d+) AC (-?\d+) turn (\d+)")
HEAD0 = re.compile(r"HP (-?\d+)/(\d+).*?Dlvl (\d+) deepest (\d+) XL (\d+) AC (-?\d+)(?: turn (\d+))?")
STATS = re.compile(r"St:(\d+)(?:/(\d+|\*\*))? Dx:(\d+) Co:(\d+) In:(\d+) Wi:(\d+) Ch:(\d+)")
GOLD = re.compile(r"\$:(\d+)")
SKILL = re.compile(r"experience (\d)/\d, result (\d)/\d")
ITEMS = {"wand": r"\bwand\b", "ring": r"\bring\b", "amulet": r"\bamulet\b", "potion": r"\bpotions?\b",
         "scroll": r"\bscrolls?\b", "spellbook": r"\bspellbook\b", "bag": r"\b(bag|sack)\b", "worn": r"\(worn\)",
         "blessed": r"\bblessed\b", "unihorn": r"unicorn horn", "reflection": r"reflection|silver dragon scale",
         "mr": r"magic resistance|gray dragon scale", "lifesave": r"life saving", "speed": r"speed boots",
         "excalibur": r"Excalibur", "dragon": r"dragon scale", "mithril": r"mithril", "marker": r"magic marker",
         "digging": r"pick-axe|mattock|wand of digging", "levitation": r"levitation", "lizard": r"lizard corpse"}
ITEM_RE = {k: re.compile(v, re.I) for k, v in ITEMS.items()}
AROUND = ("up stairs", "down stairs", "fountain", "altar", "throne", "trap", "water", "lava", "Monsters")


def features(state, ins):
    f = {}
    m = HEAD.search(state)
    if m:
        hp, mx, hp10, hp30, dl, dp, xl, ac, t = (int(x) for x in m.groups())
    else:
        m = HEAD0.search(state)
        hp, mx, dl, dp, xl, ac = (int(x) for x in m.groups()[:6])
        t = int(m.group(7)) if m.group(7) else -1
        hp10 = hp30 = hp
    f.update(hp=hp, maxhp=mx, hp_frac=hp / max(mx, 1), hp10=hp10 / max(mx, 1), hp30=hp30 / max(mx, 1), dlvl=dl,
             deepest=dp, xl=xl, ac=ac, turn=t, v34=" 3.4" in state[:40], chaotic="-Cha-" in state[:40],
             male="-Mal " in state[:40] or state[:40].split()[0].endswith("-Mal"))
    s = STATS.search(state)
    if s:
        f.update(st=int(s.group(1)) + (0.5 if s.group(2) else 0), dx=int(s.group(3)), co=int(s.group(4)),
                 in_=int(s.group(5)), wi=int(s.group(6)), ch=int(s.group(7)))
    g = GOLD.search(state)
    f["gold"] = int(g.group(1)) if g else 0
    inv = state.split("Inventory: ", 1)[1] if "Inventory: " in state else ""
    items = inv.split("; ") if inv else []
    f["n_items"] = len(items)
    f.update({k: sum(bool(r.search(x)) for x in items) for k, r in ITEM_RE.items()})
    around = next((ln for ln in state.split("\n") if ln.startswith("Around:")), "")
    f.update({f"around_{w.split()[0]}{'_' + w.split()[1] if ' ' in w else ''}": w in around for w in AROUND})
    k = SKILL.search(ins)
    f.update(exp=int(k.group(1)) if k else 0, res=int(k.group(2)) if k else 0, no_hist=k is None)
    return f


def label_of(r, q):
    crit = r["questions"][q]["criteria"]
    keys = list(crit) if isinstance(crit, dict) else [str(i) for i in range(len(crit))]
    return keys.index(r["gold"][q]["label"]), len(keys)


def load(path):
    feats, labs = {}, {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        ins = next(iter(r["questions"].values()))["instructions"]
        feats[r["id"]] = features(r["state"], ins)
        labs[r["id"]] = {q: label_of(r, q) for q in r["gold"]}
    return pd.DataFrame.from_dict(feats, orient="index").astype(float), labs


def fit_one(job):
    name, q, ids_tr, threads = job
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = [i for i in ids_tr if q in LABS[i]]
    y = np.array([LABS[i][q][0] for i in tr])
    k = LABS[tr[0]][q][1]
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True, random_state=0)
    m.fit(X.loc[tr].to_numpy(), y)
    out = []
    for split, (Xh, labs) in HELD.items():
        hid = [i for i in Xh.index if q in labs[i]]
        p = np.full((len(hid), k), 1e-9)
        p[:, m.classes_] = np.clip(m.predict_proba(Xh.loc[hid].to_numpy()), 1e-9, 1)
        p /= p.sum(1, keepdims=True)
        out += [(split, i, q, np.log(row).tolist()) for i, row in zip(hid, p)]
    return name, out


def main():
    global X, LABS, HELD
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("shares", nargs="+", help="label=train file stem (train, train_g4, ...)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prefix", default="gbm16")
    a = ap.parse_args()
    X, LABS = load(Path(a.data, "train.jsonl"))
    HELD = {s: load(Path(a.data, f"{s}.jsonl")) for s in ("val", "test")}
    for split, (Xh, _) in HELD.items():
        HELD[split] = (Xh.reindex(columns=X.columns), HELD[split][1])
    jobs = []
    for spec in a.shares:
        lab, stem = spec.split("=", 1)
        ids = [json.loads(x)["id"] for x in open(Path(a.data, f"{stem}.jsonl"), encoding="utf-8")]
        name = a.prefix + ("" if stem == "train" else stem.replace("train_", ""))
        if all(Path(f"preds/{name}-{Path(a.data).name}-{s}.parquet").exists() for s in ("val", "test")):
            print(f"{name}: done before")
            continue
        jobs += [(name, q, ids, a.threads) for q in ("ttd", "depth", "end", "ascends")]
    res = {}
    with Pool(a.workers) as pool:
        for name, out in pool.imap_unordered(fit_one, jobs):
            res.setdefault(name, []).extend(out)
            print(f"{name}: {len(res[name])} predictions so far", flush=True)
    for name, out in res.items():
        df = pd.DataFrame(out, columns=["split", "id", "question", "logits"])
        for s in ("val", "test"):
            df[df.split == s].drop(columns="split").to_parquet(f"preds/{name}-{Path(a.data).name}-{s}.parquet")
        print(f"{name}: written")


if __name__ == "__main__":
    main()
