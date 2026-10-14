"""Is the deep test's ascension share explained by the player's strength in the question? Per Dlvl band: the observed
share of ascended rows in val / test against what train's share per strength cell (experience x result, or no earlier
games) predicts for the same cells; the gap is what strength does not carry (who the held-out players are).
  python asc_shift.py human_dwa20/train.jsonl human_dwa20t [--drop-top 1] [--out ../results/asc_shift20.txt]
"""
import argparse
import bisect
import collections
import json
import re
from pathlib import Path

BANDS = (4, 12, 20, 24, 29)
NAMES = ("1-4", "5-12", "13-20", "21-24", "25-29", "30+")
SKILL = re.compile(r"experience (\d)/\d, result (\d)/\d")


def rows(path):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "ascends" not in r["gold"]:
            continue
        ins = r["questions"]["ascends"]["instructions"]
        m = SKILL.search(ins)
        cell = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        dl = int(r["state"].split("Dlvl ", 1)[1].split()[0])
        yield bisect.bisect_left(BANDS, dl), cell, r["gold"]["ascends"]["label"] == "true", r["id"].split("@")[0].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("held")
    ap.add_argument("--drop-top", type=int, default=0, help="leave out the N held-out players with the most ascended rows")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tr = collections.defaultdict(lambda: [0, 0])
    for b, c, y, _ in rows(a.train):
        tr[b, c][0] += 1
        tr[b, c][1] += y
    band_tr = collections.defaultdict(lambda: [0, 0])
    for (b, _), (n, k) in tr.items():
        band_tr[b][0] += n
        band_tr[b][1] += k
    out = [f"{a.train} -> {a.held}: ascended share per band; expected = train's share in the row's strength cell "
           "(experience/result levels, (0,0) no earlier games; a cell empty in train takes the band's share)",
           "split band    rows  observed  expected  train-band  | rows by experience 0..5 ; by result 0..5"]
    hr = {s: list(rows(Path(a.held, f"{s}.jsonl"))) for s in ("val", "test")}
    top = collections.Counter(p for v in hr.values() for *_, y, p in v if y).most_common(a.drop_top)
    if top:
        out.insert(1, f"left out: {len(top)} held-out player(s) with the most ascended rows ({sum(n for _, n in top)})")
    top = {p for p, _ in top}
    for s in ("val", "test"):
        held = collections.defaultdict(list)
        for b, c, y, p in hr[s]:
            if p not in top:
                held[b].append((c, y))
        for b, name in enumerate(NAMES):
            v = held.get(b, [])
            if not v:
                continue
            nb, kb = band_tr[b]
            exp = [tr[b, c][1] / tr[b, c][0] if tr[b, c][0] else kb / nb for c, _ in v]
            ex = collections.Counter(c[0] for c, _ in v)
            rs = collections.Counter(c[1] for c, _ in v)
            out.append(f"{s:5s} {name:6s} {len(v):6d}  {sum(y for _, y in v) / len(v):.3f}     {sum(exp) / len(v):.3f}"
                       f"     {kb / nb:.3f}     | " + " ".join(f"{ex[i] / len(v):.2f}" for i in range(6)) + " ; "
                       + " ".join(f"{rs[i] / len(v):.2f}" for i in range(6)))
    for b, name in enumerate(NAMES):
        n = band_tr[b][0]
        ex = collections.Counter()
        rs = collections.Counter()
        for (bb, c), (m, _) in tr.items():
            if bb == b:
                ex[c[0]] += m
                rs[c[1]] += m
        out.append(f"train {name:6s} {n:6d}  {band_tr[b][1] / n:.3f}                         | "
                   + " ".join(f"{ex[i] / n:.2f}" for i in range(6)) + " ; " + " ".join(f"{rs[i] / n:.2f}" for i in range(6)))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
