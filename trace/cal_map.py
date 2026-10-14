"""Does a student reproduce the mean of its train targets? Per Dlvl band and question, the models' mean probability per
option against the mean target (gold.probabilities) and the observed share of the labels, on (a part of) its train rows;
a student whose mean falls short of its targets' mean where the teacher already fell short of the old target
compounds the teacher's miss.
  cut:  every n-th train row by crc32 of the id -> a smaller file to predict
    python cal_map.py cut human_dwa29d/train.jsonl human_dwa29d/train_g5.jsonl --every 5
  read: python cal_map.py read human_dwa29d/train_g5.jsonl --models c17d --tag human_dwa29d-train_g5 \
            --out ../results/cal_map29.txt
predictions: preds/<model>-<tag>.parquet (predict.py)"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np

import leak_total
from curve_eval import QS
from leak_total import NAMES, load, probs


def cut(a):
    n = k = 0
    with open(a.src, encoding="utf-8") as f, open(a.dst, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            n += 1
            if zlib.crc32(json.loads(line)["id"].encode()) % a.every == 0:
                g.write(line)
                k += 1
    print(f"{a.dst}: {k} of {n} rows")


def read(a):
    data, keys, _ = load([a.data], targets=True)
    texts = leak_total.TEXTS
    out = [f"{a.data}; models {' '.join(a.models)}; mean probability per option: model - mean target (x1000), observed "
           f"share - mean target next to the largest"]
    ev = {}
    for q in QS:
        d = data[q]
        ps = {m: probs([f"preds/{m}-{a.tag}.parquet"], q, d["ids"]) for m in a.models}
        k_ = d["old"].shape[1]
        if q == "ttd":
            hi = next(i for i, t in enumerate(texts[q]) if t.split("-")[-1].split()[-1] == "1000") + 1
            evf = lambda p: p[:, :hi].sum(1)
        elif q == "depth":
            evf = lambda p: p[:, :1].sum(1)
        elif q == "ascends":
            j = keys[q].index("true")
            evf = lambda p: p[:, j]
        else:
            evf = None
        for b in range(6):
            k = d["bands"] == b
            if not k.any():
                continue
            gold = d["old"][k].mean(0)
            obs = np.bincount(d["y"][k], minlength=k_) / k.sum()
            for m, p in ps.items():
                dlt = p[k].mean(0) - gold
                jj = int(np.argmax(np.abs(dlt)))
                out.append(f"  {q + ' ' + NAMES[b]:<16}{m:<10}rows {k.sum():6d}  max |d| {dlt[jj]:+.4f} at "
                           f"{texts[q][jj][:24]!r} (observed - target {obs[jj] - gold[jj]:+.4f}); "
                           f"sum |d| {np.abs(dlt).sum():.3f}")
                if len(dlt) <= 21:
                    out.append("    " + " ".join(f"{1000 * x:+.0f}" for x in dlt))
            if evf is not None:
                yy = np.eye(k_)[d["y"][k]]
                ev[q, b] = (evf(yy).mean(), evf(d["old"][k]).mean(), {m: evf(p[k]).mean() for m, p in ps.items()})
    out.append("\nmean event probability: observed / mean target / models")
    for q, name in (("ttd", "P(ttd <= 1000)"), ("depth", "P(no deeper)"), ("ascends", "P(ascended)")):
        for b in range(6):
            if (q, b) in ev:
                o, g, ms = ev[q, b]
                out.append(f"  {name + ' ' + NAMES[b]:<22}{o:.3f} / {g:.3f} / " +
                           " ".join(f"{m} {v:.3f}" for m, v in ms.items()))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--every", type=int, default=5)
    r = sub.add_parser("read")
    r.add_argument("data")
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--out", default="")
    a = ap.parse_args()
    (cut if a.cmd == "cut" else read)(a)


if __name__ == "__main__":
    main()
