"""Which train rows of a set came through unchanged into a bigger one, and whether every new row is from the given
games (json list with name/start).
  python rows_same.py human_dwa20n/train.jsonl human_dwa24/train.jsonl nao/val_dwa_early.json
"""
import json
import sys


def main(old, new, games):
    keep = {f"{g['name']}@{g['start']}" for g in json.load(open(games, encoding="utf-8"))}
    a = {json.loads(x)["id"]: x for x in open(old, encoding="utf-8")}
    b = {json.loads(x)["id"]: x for x in open(new, encoding="utf-8")}
    same = sum(a[i] == b.get(i) for i in a)
    extra = [i for i in b if i not in a]
    inside = sum(i.split("#")[0] in keep for i in extra)
    print(f"{old}: {len(a)} rows, {new}: {len(b)}; identical {same}, missing {sum(i not in b for i in a)}; "
          f"new rows {len(extra)}, of the given games {inside}")
    return same == len(a) and inside == len(extra)


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:4]) else 1)
