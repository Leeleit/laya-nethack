"""The GBM reference of game_probe.py: human_tab.py's model per question (hard labels), cross-fitted by players over the
train set (5 folds by crc32 of the player); a row of a train game the teacher did not see (--unseen) and a held-out row
(--held val / test) are both predicted by the fold model that left out their player's fold; --train-oof: the train rows
too (out of fold).
  python gbm_cf.py human_dwa24 --unseen human_mem29/unseen.jsonl --held human_dwa20t --name gbmcf [--workers 5 --threads 3]
-> preds/<name>-human_mem29-unseen.parquet, preds/<name>-human_dwa20t-{val,test}.parquet (predict.py's format)
  python gbm_cf.py human_dwa24 --train-oof --name gbmcf -> preds/gbmcf-human_dwa24-train.parquet"""
import argparse
import os
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from human_tab import load

K = 5


def fold(i):
    return zlib.crc32(i.split("@")[0].lower().encode()) % K


def fit_one(job):
    q, f, threads, seed = job
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = [i for i in X.index if q in LABS[i] and FOLD[i] != f]
    k = LABS[tr[0]][q][1]
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True, random_state=seed)
    m.fit(X.loc[tr].to_numpy(), np.array([LABS[i][q][0] for i in tr]))
    out = []
    for target, (Xh, labs) in HELD.items():
        hid = [i for i in Xh.index if q in labs[i] and fold(i) == f]
        if not hid:
            continue
        p = np.full((len(hid), k), 1e-9)
        p[:, m.classes_] = np.clip(m.predict_proba(Xh.loc[hid].to_numpy()), 1e-9, 1)
        p /= p.sum(1, keepdims=True)
        out += [(target, i, q, np.log(row).tolist()) for i, row in zip(hid, p)]
    return q, f, m.n_iter_, out


def main():
    global X, LABS, FOLD, HELD
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--unseen", default="")
    ap.add_argument("--held", default="")
    ap.add_argument("--train-oof", action="store_true")
    ap.add_argument("--name", default="gbmcf")
    ap.add_argument("--questions", nargs="+", default=["ttd", "depth", "end", "ascends"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    X, LABS = load(Path(a.data, "train.jsonl"))
    FOLD = {i: fold(i) for i in X.index}
    targets = {}
    if a.unseen:
        targets[f"{Path(a.unseen).parent.name}-{Path(a.unseen).stem}"] = Path(a.unseen)
    if a.held:
        targets.update({f"{Path(a.held).name}-{s}": Path(a.held, f"{s}.jsonl") for s in ("val", "test")})
    assert targets or a.train_oof, "nothing to predict"
    HELD = {}
    for t, path in targets.items():
        Xh, labs = load(path)
        HELD[t] = (Xh.reindex(columns=X.columns), labs)
    if a.train_oof:  # fold f's rows are held out of the fold-f model
        targets[f"{Path(a.data).name}-train"] = None
        HELD[f"{Path(a.data).name}-train"] = (X, LABS)
    trp = {i.split("@")[0].lower() for i in X.index}
    for t, (Xh, _) in HELD.items():
        n = sum(i.split("@")[0].lower() in trp for i in Xh.index)
        print(f"{t}: {len(Xh)} rows, of train players {n}", flush=True)
    print("train rows per fold: " + " ".join(str(sum(v == f for v in FOLD.values())) for f in range(K)), flush=True)
    jobs = [(q, f, a.threads, a.seed) for q in a.questions for f in range(K)]
    res = []
    with Pool(a.workers) as pool:
        for q, f, it, out in pool.imap_unordered(fit_one, jobs):
            res += out
            print(f"{q} fold {f}: {it} iterations, {len(out)} predictions", flush=True)
    df = pd.DataFrame(res, columns=["target", "id", "question", "logits"])
    for t in targets:
        sub = df[df.target == t].drop(columns="target")
        sub.to_parquet(f"preds/{a.name}-{t}.parquet")
        print(f"preds/{a.name}-{t}.parquet: {len(sub)} predictions")


if __name__ == "__main__":
    main()
