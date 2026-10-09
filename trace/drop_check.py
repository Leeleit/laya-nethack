"""A dataset rebuilt without some players (human_build.py --exclude-players) against the original:
per split, the new rows must be exactly the old ones minus those players' (same ids, same states); the questions'
labels may move where v6 bins come from train (reported: share of rows whose gold labels differ, per question).
  python drop_check.py human_data14d human_data15d nao/bot_players.txt
"""
import collections
import json
import sys
from pathlib import Path


def main():
    old, new, players = sys.argv[1:4]
    bots = {ln.split()[0].lower() for ln in open(players, encoding="utf-8") if ln.strip() and not ln.startswith("#")}
    ok = True
    for split in ("train", "val", "test"):
        a = {r["id"]: r for r in map(json.loads, open(Path(old, f"{split}.jsonl"), encoding="utf-8"))}
        b = {r["id"]: r for r in map(json.loads, open(Path(new, f"{split}.jsonl"), encoding="utf-8"))}
        kept = {i for i in a if i.split("@")[0].lower() not in bots}
        same_ids = kept == set(b)
        same_state = all(a[i]["state"] == b[i]["state"] for i in kept & set(b))
        moved = collections.Counter()
        for i in kept & set(b):
            for q in b[i]["gold"]:
                if a[i]["gold"].get(q, {}).get("label") != b[i]["gold"][q]["label"]:
                    moved[q] += 1
        n = max(1, len(kept & set(b)))
        ok &= same_ids and same_state
        print(f"{split}: old {len(a)}, bots' rows {len(a) - len(kept)}, new {len(b)}; ids = old minus bots: {same_ids}; "
              f"states equal: {same_state}; labels moved: " + ", ".join(f"{q} {k / n:.2%}" for q, k in sorted(moved.items())))
    print("OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
