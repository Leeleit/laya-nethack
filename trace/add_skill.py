"""A dataset with the player's strength in every question: "A human plays. " becomes
"A human (experience E/5, result R/5) plays. " or "A human (no earlier games) plays. "; rows, states and labels stay.
experience = n, the account's earlier real NAO games (> --min-turns turns, any role; turns_dist.py: 67 % of NAO games
end on turn 1, the dip runs to ~50); result = (k + a) / (n + a + b), k earlier ascensions, a, b a beta-binomial prior
fitted on train rows. Both cut into --bins equal-share bins of train rows with n >= 1 (skill_bins.py; the bins
are this set's own). Report per split and bin: share of rows,
players, this game's ascended share (Wilson 95 %) -> <out>/skill.txt; edges and prior -> <out>/meta.json "skill".
  python add_skill.py human_data15d human_data16d [--min-turns 50] [--bins 5]
"""
import argparse
import bisect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from skill_bins import GAMES, fit_prior, wilson

OLD = "A human plays. "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--min-turns", type=int, default=50)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--skill-from", default="", help="take the prior and bin edges from this dataset's meta.json")
    a = ap.parse_args()
    d = pd.read_parquet(GAMES, columns=["source", "name", "death", "turns", "endtime"])
    d = d[(d.source == "nao") & d.endtime.notna() & (d.turns > a.min_turns)]
    ends = {nm: np.sort(g.endtime.to_numpy()) for nm, g in d.groupby(d.name.str.lower())}
    asc = {nm: np.sort(g.endtime.to_numpy()) for nm, g in d[d.death == "ascended"].groupby(d.name.str.lower())}
    rows, nk = {}, {}
    for split in ("train", "val", "test"):
        rows[split] = [json.loads(x) for x in open(Path(a.src, f"{split}.jsonl"), encoding="utf-8")]
        n, k = [], []
        for r in rows[split]:
            name, start = r["id"].split("#", 1)[0].split("@", 1)
            n.append(bisect.bisect_left(ends.get(name.lower(), []), float(start)))
            k.append(bisect.bisect_left(asc.get(name.lower(), []), float(start)))
        nk[split] = (np.array(n), np.array(k))
    if a.skill_from:  # another set's prior and bins
        sk = json.loads(Path(a.skill_from, "meta.json").read_text())["skill"]
        assert sk["min_turns"] == a.min_turns and sk["bins"] == a.bins, sk
        (pa, pb), e_exp, e_res = sk["prior"], sk["experience_edges"], sk["result_edges"]
    else:
        pa, pb = fit_prior(*nk["train"])
        n, k = nk["train"]
        q = np.linspace(0, 100, a.bins + 1)[1:-1]
        e_exp = np.percentile(n[n >= 1], q).tolist()
        e_res = np.percentile(((k + pa) / (n + pa + pb))[n >= 1], q).tolist()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rep = [f"min turns {a.min_turns}; prior a {pa:.4f} b {pb:.2f}; experience edges {e_exp}; result edges {e_res}"]
    for split, rs in rows.items():
        n, k = nk[split]
        lvl = []
        with open(Path(a.out, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r, ni, ki in zip(rs, n, k):
                if ni == 0:
                    who, e, s = "A human (no earlier games) plays. ", 0, 0
                else:
                    e = bisect.bisect_right(e_exp, ni) + 1
                    s = bisect.bisect_right(e_res, (ki + pa) / (ni + pa + pb)) + 1
                    who = f"A human (experience {e}/{a.bins}, result {s}/{a.bins}) plays. "
                lvl.append((e, s))
                for qd in r["questions"].values():
                    assert qd["instructions"].startswith(OLD), qd["instructions"][:40]
                    qd["instructions"] = who + qd["instructions"][len(OLD):]
                f.write(json.dumps(r) + "\n")
        lvl = np.array(lvl)
        y = np.array([r["gold"]["ascends"]["label"] == "true" for r in rs])
        pl = np.array([r["id"].split("@")[0].lower() for r in rs])
        rep.append(f"== {split}: {len(rs)} rows")
        groups = [("no history", lvl[:, 0] == 0)] + [(f"experience {v}", lvl[:, 0] == v) for v in range(1, a.bins + 1)] \
            + [(f"result {v}", lvl[:, 1] == v) for v in range(1, a.bins + 1)]
        for name, sel in groups:
            if sel.sum():
                lo, hi = wilson(y[sel].sum(), sel.sum())
                rep.append(f"  {name:12s} rows {sel.mean():6.1%}  players {len(set(pl[sel])):5d}  "
                           f"ascended {y[sel].mean():6.2%} [{lo:.2%}, {hi:.2%}]")
    meta = json.loads(Path(a.src, "meta.json").read_text())
    meta["skill"] = {"src": a.skill_from or a.src, "min_turns": a.min_turns, "bins": a.bins, "prior": [pa, pb],
                     "experience_edges": e_exp, "result_edges": e_res}
    Path(a.out, "meta.json").write_text(json.dumps(meta, indent=1))
    Path(a.out, "skill.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    for extra in Path(a.src).glob("*.log"):
        shutil.copy(extra, Path(a.out, extra.name))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
