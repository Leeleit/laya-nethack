"""Shares of the player's strength in the questions (experience E/5, result R/5, or no earlier games) per Dlvl band,
train against held-out (val + test), with TV distance.
  python skill_share.py human_dwa30/train.jsonl human_dwa30t [--out ../results/skill_share30.txt]
"""
import argparse
import bisect
import collections
import json
import re
from pathlib import Path

from deep_checks import BANDS, NAMES

PAT = re.compile(r"experience (\d)/\d, result (\d)/\d")


def count(paths):
    c = collections.defaultdict(collections.Counter)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            dl = re.search(r"\bDlvl (\d+)", r["state"])
            m = PAT.search(next(iter(r["questions"].values()))["instructions"])
            c[NAMES[bisect.bisect_left(BANDS, int(dl.group(1)) if dl else -1)]][m.groups() if m else ("-", "-")] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("held")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tr = count([a.train])
    he = count([Path(a.held, f"{s}.jsonl") for s in ("val", "test")])
    out = [f"{a.train} against {a.held}: shares of experience (E) and result (R), '-' no earlier games"]
    for b in NAMES:
        nt, nh = sum(tr[b].values()), sum(he[b].values())
        if not nt or not nh:
            continue
        line = [f"  {b:6s}"]
        for i, name in ((0, "E"), (1, "R")):
            ct, ch = collections.Counter(), collections.Counter()
            for k, v in tr[b].items():
                ct[k[i]] += v
            for k, v in he[b].items():
                ch[k[i]] += v
            ks = sorted(set(ct) | set(ch))
            tv = 0.5 * sum(abs(ct[k] / nt - ch[k] / nh) for k in ks)
            line.append(f"{name} TV {tv:.3f} " + " ".join(f"{k}:{ct[k] / nt:.2f}|{ch[k] / nh:.2f}" for k in ks))
        out.append("  ".join(line))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
