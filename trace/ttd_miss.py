"""Where does a model's mean P(ttd <= 1000) fall short of the observed share (CPU, from ready predictions)? Per deep band,
rows split by what the state shows (HP share, HP lost in the last 10 turns, on the deepest level or back up, turn
tercile within the band, the game's end) and by the reference's own P(ttd <= 1000) quintile (does the model miss the
rows the reference finds risky, or all rows alike?): rows, observed share, mean old target (train only), each model.
  python ttd_miss.py human_dwa24/train.jsonl --tag human_dwa24-train --models c17e c17es1 c17es2 c17es3 gbmcf \
      --ref gbmcf --out ../results/ttd_miss24.txt
  python ttd_miss.py human_dwa20t/val.jsonl human_dwa20t/test.jsonl --tag human_dwa20t-val human_dwa20t-test \
      --models c17e c17es1 c17es2 c17es3 gbmcf --ref gbmcf --drop-top 1 --out ../results/ttd_miss20t.txt
predictions: preds/<model>-<tag>.parquet (all tags of a model are read together)"""
import argparse
import bisect
import collections
import json
import re
from pathlib import Path

import numpy as np

from leak_total import BANDS, NAMES, probs

HP = re.compile(r"\bHP (-?\d+)/(\d+)(?: \(10 turns ago (-?\d+)/\d+)?")
DL = re.compile(r"\bDlvl (\d+) deepest (\d+)")
TURN = re.compile(r"\bturn (\d+)")


def rows_of(paths):
    out = collections.defaultdict(list)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if "ttd" not in r["gold"]:
                continue
            s = r["state"]
            hp, dl, tu = HP.search(s), DL.search(s), TURN.search(s)
            if not (hp and dl):
                continue
            crit = r["questions"]["ttd"]["criteria"]
            hi = next(i for i, t in enumerate(crit) if t.split("-")[-1].split()[-1] == "1000") + 1
            out["id"].append(r["id"])
            out["player"].append(r["id"].split("@")[0].lower())
            out["band"].append(bisect.bisect_left(BANDS, int(dl.group(1))))
            out["dlvl"].append(int(dl.group(1)))
            out["deepest"].append(int(dl.group(1)) >= int(dl.group(2)))
            cur, mx = int(hp.group(1)), max(int(hp.group(2)), 1)
            out["hp"].append(cur / mx)
            out["lost"].append(hp.group(3) is not None and int(hp.group(3)) > cur)
            out["turn"].append(int(tu.group(1)) if tu else -1)
            out["short"].append(int(r["gold"]["ttd"]["label"]) < hi)
            pr = r["gold"]["ttd"]["probabilities"]
            out["target"].append(sum(pr.get(str(k), pr.get(c, 0.0)) for k, c in enumerate(crit[:hi])) / sum(pr.values()))
            out["ascend"].append(r["gold"].get("end", {}).get("label") == "ascend")
            out["hi"].append(hi)
    return {k: np.array(v) if k not in ("id", "player") else v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--tag", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ref", default="gbmcf")
    ap.add_argument("--drop-top", type=int, default=0)
    ap.add_argument("--bands", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = rows_of(a.data)
    keep = np.ones(len(d["id"]), bool)
    dropped = []
    if a.drop_top:  # as deep_checks.py: the players with the most ascended rows
        cnt = collections.Counter(p for p, v in zip(d["player"], d["ascend"]) if v)
        dropped = [p for p, _ in cnt.most_common(a.drop_top)]
        keep = np.array([p not in dropped for p in d["player"]])
    hi = d["hi"][0]
    assert (d["hi"] == hi).all()
    pshort = {}
    for m in a.models:
        p = probs([f"preds/{m}-{t}.parquet" for t in a.tag], "ttd", d["id"])
        pshort[m] = p[:, :hi].sum(1)
    out = [f"{' '.join(a.data)}; mean P(ttd <= 1000): observed / mean old target / models"
           + (f"; left out the top player(s)" if dropped else "")]
    for b in a.bands:
        kb = keep & (d["band"] == b)
        if not kb.any():
            continue
        tu = d["turn"][kb]
        t1, t2 = np.quantile(tu[tu >= 0], [1 / 3, 2 / 3]) if (tu >= 0).any() else (0, 0)
        rq = pshort[a.ref][kb]
        qs = np.quantile(rq, [0.2, 0.4, 0.6, 0.8])
        groups = [("all", np.ones(kb.sum(), bool)),
                  ("HP < 1/2", d["hp"][kb] < 0.5), ("HP >= 1/2", d["hp"][kb] >= 0.5),
                  ("lost HP in 10 turns", d["lost"][kb]), ("no HP lost", ~d["lost"][kb]),
                  ("on the deepest level", d["deepest"][kb]), ("back above the deepest", ~d["deepest"][kb]),
                  (f"turn <= {t1:.0f}", (tu >= 0) & (tu <= t1)), (f"turn {t1:.0f}-{t2:.0f}", (tu > t1) & (tu <= t2)),
                  (f"turn > {t2:.0f}", tu > t2),
                  ("game ascended", d["ascend"][kb]), ("game not ascended", ~d["ascend"][kb])]
        groups += [(f"{a.ref} quintile {i + 1}", np.digitize(rq, qs) == i) for i in range(5)]
        out.append(f"\n==== Dlvl {NAMES[b]}: {kb.sum()} rows")
        out.append(f"  {'':<24}{'rows':>6}  {'obs':>6}  {'target':>6}  " + "  ".join(f"{m:>8}" for m in a.models))
        for name, g in groups:
            if not g.any():
                continue
            obs = d["short"][kb][g].mean()
            tgt = d["target"][kb][g].mean()
            out.append(f"  {name:<24}{g.sum():6d}  {obs:6.3f}  {tgt:6.3f}  "
                       + "  ".join(f"{pshort[m][kb][g].mean():8.3f}" for m in a.models))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
