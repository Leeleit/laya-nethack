"""Checks of a fresh player split (human_build.py --resplit) against the old sets of the same recipe:
  - no player of the new held-out set (val + test) has a row in any new train file;
  - every new held-out player was a train player of the old split (crc32 of the name % 20 not 0 or 1);
  - ids unique within each file;
  - a row in both the old and the new train is byte-identical (the recipe did not change, only the split);
  - the options of ttd, end and ascends are the same as in the old set (depth names absolute levels and the question
    text carries the player's experience, both vary by row; overlapping rows are compared whole above).
  python split_check.py --new-train human_dwa30/train.jsonl human_oth30n/train.jsonl \
      --new-held human_dwa30t/val.jsonl human_dwa30t/test.jsonl \
      --old-train human_dwa24/train.jsonl --questions-from human_dwa24/train.jsonl
exit 1 on any failure"""
import argparse
import json
import zlib

FIXED = ("ttd", "end", "ascends")


def rows(path):
    for line in open(path, encoding="utf-8"):
        yield line, json.loads(line)


def player(i):
    return i.split("@")[0].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-train", nargs="+", required=True)
    ap.add_argument("--new-held", nargs="+", required=True)
    ap.add_argument("--old-train", nargs="+", default=[])
    ap.add_argument("--questions-from", default="")
    a = ap.parse_args()
    bad = 0
    held, held_rows = set(), 0
    for p in a.new_held:
        ids = set()
        for _, r in rows(p):
            held.add(player(r["id"]))
            held_rows += 1
            if r["id"] in ids:
                print(f"FAIL duplicate id {r['id']} in {p}")
                bad += 1
            ids.add(r["id"])
    old_held = [x for x in held if zlib.crc32(x.encode()) % 20 in (0, 1)]
    print(f"held-out: {len(held)} players, {held_rows} rows; of the old val/test players: {len(old_held)}")
    bad += len(old_held) > 0
    old = {}
    for p in a.old_train:
        for line, r in rows(p):
            old[r["id"]] = line
    qs = {}
    if a.questions_from:
        for _, r in rows(a.questions_from):
            for q in FIXED:
                if q in r["questions"]:
                    qs.setdefault(q, json.dumps(r["questions"][q]["criteria"]))
    for p in a.new_train:
        ids, leak, same, diff, qbad, n = set(), 0, 0, 0, 0, 0
        for line, r in rows(p):
            n += 1
            if player(r["id"]) in held:
                leak += 1
            if r["id"] in ids:
                print(f"FAIL duplicate id {r['id']} in {p}")
                bad += 1
            ids.add(r["id"])
            if r["id"] in old:
                if old[r["id"]] == line:
                    same += 1
                else:
                    diff += 1
            for q in FIXED:
                if q in qs and q in r["questions"] and json.dumps(r["questions"][q]["criteria"]) != qs[q]:
                    qbad += 1
        print(f"{p}: {n} rows; rows of held-out players {leak}; in the old train {same + diff} "
              f"(identical {same}, different {diff}); questions unlike the old set {qbad}")
        bad += leak + diff + qbad
    print("OK" if not bad else f"FAILED ({bad})")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
