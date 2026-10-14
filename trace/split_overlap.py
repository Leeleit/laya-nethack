"""Overlap of a train set with its held-out set: players (case-folded), games (id before '#'), identical state texts,
and identical state texts once the turn counter and the bottom status line numbers are masked -- per Dlvl band of the
held row. A held-out row whose state also occurs in train is predicted from memory, not from generalisation.
  python split_overlap.py human_dwa30/train.jsonl human_dwa30t [--out ../results/split_overlap30.txt]
"""
import argparse
import bisect
import collections
import hashlib
import json
import re
from pathlib import Path

from deep_checks import BANDS, NAMES

NUM = re.compile(r"\d+")


def keys(path):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        s = r["state"]
        dl = re.search(r"\bDlvl (\d+)", s)
        yield (r["id"], r["id"].split("@")[0].lower(), r["id"].split("#")[0].lower(),
               hashlib.md5(s.encode()).hexdigest(), hashlib.md5(NUM.sub("#", s).encode()).hexdigest(),
               int(dl.group(1)) if dl else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("held")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tr = list(keys(a.train))
    tp, tg, ts, tn = ({x[i] for x in tr} for i in (1, 2, 3, 4))
    out = [f"{a.train}: {len(tr)} rows, {len(tp)} players, {len(tg)} games"]
    for split in ("val", "test"):
        he = list(keys(Path(a.held, f"{split}.jsonl")))
        out.append(f"\n{a.held}/{split}: {len(he)} rows, {len({x[1] for x in he})} players, {len({x[2] for x in he})} games;"
                   f" players in train {len({x[1] for x in he} & tp)}, games in train {len({x[2] for x in he} & tg)}")
        c = collections.defaultdict(lambda: [0, 0, 0])
        for x in he:
            b = NAMES[bisect.bisect_left(BANDS, x[5])]
            c[b][0] += 1
            c[b][1] += x[3] in ts
            c[b][2] += x[4] in tn
        for b in NAMES:
            n, s, m = c[b]
            if n:
                out.append(f"  {b:6s} rows {n:6d}  same state in train {s:5d} ({s / n:.1%})  same with numbers masked {m:5d}"
                           f" ({m / n:.1%})")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
