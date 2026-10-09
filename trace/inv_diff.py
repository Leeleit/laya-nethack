"""Inventory text of the same states in two human datasets (e.g. before / after clean_inv): mean length and the states
that shrank most, to see what the cleaning removed.
  python inv_diff.py <old.jsonl> <new.jsonl> [--n 3]"""
import argparse
import json
import re

INV = re.compile(r"\nInventory[^:]*: (.*)\n")


def invs(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = INV.search(r["state"])
        out[r["id"]] = m.group(1) if m else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    old, new = invs(a.old), invs(a.new)
    ids = [i for i in new if i in old]
    lo, ln = sum(len(old[i]) for i in ids), sum(len(new[i]) for i in ids)
    print(f"{len(ids)} common states: inventory chars {lo / len(ids):.0f} -> {ln / len(ids):.0f}; "
          f"items {sum(old[i].count('; ') + 1 for i in ids) / len(ids):.1f} -> "
          f"{sum(new[i].count('; ') + 1 for i in ids) / len(ids):.1f}")
    for i in sorted(ids, key=lambda i: len(new[i]) - len(old[i]))[:a.n]:
        print(f"==== {i}\nOLD: {old[i][:1500]}\nNEW: {new[i][:600]}")


if __name__ == "__main__":
    main()
