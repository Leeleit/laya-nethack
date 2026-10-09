"""The games the maxlvl >= 4 threshold cut (maxlvl_cut.py: Val-Dwa 15 001,
other roles 385 683 games): from `nao_corridor.py list --role any --race any --outcome any --max-maxlvl 3`,
the same filters as maxlvl_cut.py -- more than 50 turns, no bot accounts -- then Val-Dwa, and every role but Val with
at most 300 000 turns shuffled with seed 0 as other_any4_cut.json (whatever part gets extracted is a random sample).
  python make_early.py nao/all_early.json --bots nao/bot_players_all.txt \
      --dwa nao/val_dwa_early.json --other nao/other_early.json
"""
import argparse
import collections
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games")
    ap.add_argument("--bots", required=True)
    ap.add_argument("--dwa", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--min-turns", type=int, default=50)
    a = ap.parse_args()
    bots = {ln.split()[0].lower() for ln in open(a.bots, encoding="utf-8") if ln.strip() and not ln.startswith("#")}
    g = json.load(open(a.games, encoding="utf-8"))
    drop = collections.Counter()
    keep = []
    for x in g:
        if x["maxlvl"] >= 4:
            drop["maxlvl >= 4"] += 1
        elif x["turns"] <= a.min_turns:
            drop[f"turns <= {a.min_turns}"] += 1
        elif x["name"].lower() in bots:
            drop["bot accounts"] += 1
        else:
            keep.append(x)
    dwa = [x for x in keep if x["character"].startswith("Val-Dwa-")]
    other = [x for x in keep if not x["character"].startswith("Val-") and x["turns"] <= 300000]
    random.Random(0).shuffle(other)
    json.dump(dwa, open(a.dwa, "w", encoding="utf-8"))
    json.dump(other, open(a.other, "w", encoding="utf-8"))
    print(f"{len(g)} games listed; dropped {dict(drop)}")
    print(f"Val-Dwa {len(dwa)} games, {sum(x['bytes'] for x in dwa) / 1e9:.2f} GB bz2; other roles {len(other)}, "
          f"{sum(x['bytes'] for x in other) / 1e9:.2f} GB; Val-Hum and other Valkyries left out "
          f"{sum(x['character'].startswith('Val-') and not x['character'].startswith('Val-Dwa-') for x in keep)}")
    print("deathlev:", {k: collections.Counter(x["deathlev"] for x in v).most_common()
                        for k, v in (("Val-Dwa", dwa), ("other", other))})


if __name__ == "__main__":
    main()
