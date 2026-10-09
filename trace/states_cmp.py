"""Two nao_corridor.py states files over the same games (e.g. the old and the fixed extractor): states per game, the
games that lost all their states, and inventories side by side for a few states at the same turn.
  python states_cmp.py <old.jsonl> <new.jsonl> [--show 3]"""
import argparse
import json


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        try:
            g = json.loads(line)
        except ValueError:
            continue
        out[(g["name"], g["start"])] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    keys = [k for k in new if k in old]
    so, sn = sum(len(old[k]["states"]) for k in keys), sum(len(new[k]["states"]) for k in keys)
    lost = [k for k in keys if old[k]["states"] and not new[k]["states"]]
    print(f"{len(keys)} common games: states {so} -> {sn}; games with states before and none now: {len(lost)}")
    print(f"  items per state {sum(len(s['inv']) for k in keys for s in old[k]['states']) / max(so, 1):.1f} -> "
          f"{sum(len(s['inv']) for k in keys for s in new[k]['states']) / max(sn, 1):.1f}")
    shown = 0
    for k in keys:
        by_turn = {s["turn"]: s for s in old[k]["states"]}
        for s in new[k]["states"]:
            o = by_turn.get(s["turn"])
            if o and o["inv"] != s["inv"] and s["turn"] > 3000 and shown < a.show:
                shown += 1
                print(f"==== {k[0]} turn {s['turn']}\nOLD ({len(o['inv'])}): {'; '.join(o['inv'])[:900]}\n"
                      f"NEW ({len(s['inv'])}): {'; '.join(s['inv'])[:900]}")


if __name__ == "__main__":
    main()
