"""Does an inventory letter name the same item across games? For each letter of the inventories in a dataset's state
texts: how many states have it, its most common item (count, BUC, enchantment and charges dropped) and that item's
share, and the most common item class and its share. If letters were fixed, "$Afqr" could stand for the inventory.
  python inv_letters.py <cases.jsonl> [--n 20000]"""
import argparse
import collections
import json
import random
import re

CLASSES = [("potion", r"\bpotions?\b"), ("scroll", r"\bscrolls?\b"), ("wand", r"\bwand\b"), ("ring", r"\bring\b"),
           ("amulet", r"\bamulet\b"), ("spellbook", r"\bspellbook\b|\bbook\b"),
           ("food", r"ration|corpse|egg|apple|carrot|pie|meat|lembas|cram|tin\b|fortune cookie|banana|orange\b|melon"),
           ("gem", r"\bgems?\b|\bstones?\b|glass"),
           ("armor", r"mail|armor|shield|helm|cloak|boots|shoes|gloves|gauntlets|cape|robe|hat|shirt|apron|cap\b"),
           ("weapon", r"sword|dagger|axe|bow|arrows?|spear|mace|hammer|knife|club|whip|lance|darts?|polearm|flail|"
                      r"crossbow|bolts?|scimitar|saber|katana|trident|javelin|sling|boomerang|shuriken")]
CLS = [(k, re.compile(v, re.I)) for k, v in CLASSES]
NORM = re.compile(r"^(an?|the|\d+)\s+|\b(blessed|uncursed|cursed)\s+|[+-]\d+\s+|\s*\(.*?\)", re.I)


def norm(item):
    prev = None
    while prev != item:
        prev, item = item, NORM.sub("", item).strip()
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    lines = open(a.cases, encoding="utf-8").read().splitlines()
    items, classes, n = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), 0
    for line in random.Random(0).sample(lines, min(a.n, len(lines))):
        st = json.loads(line)["state"]
        m = re.search(r"\nInventory[^:]*: (.*)\n", st)
        if not m or m.group(1) == "unknown":
            continue
        n += 1
        for it in m.group(1).split("; "):
            if len(it) < 5 or it[1:4] != " - ":
                continue
            letter, what = it[0], norm(it[4:])
            items[letter][what] += 1
            classes[letter][next((k for k, r in CLS if r.search(what)), "other")] += 1
    print(f"{n} inventories; per letter: states with it, top item (share), top class (share)")
    for letter in sorted(items, key=lambda k: -sum(items[k].values())):
        c, k = items[letter], classes[letter]
        tot = sum(c.values())
        (it, ni), (cl, nc) = c.most_common(1)[0], k.most_common(1)[0]
        print(f"  {letter}  {100 * tot / n:5.1f} %  {it[:38]:38s} {100 * ni / tot:5.1f} %   {cl:9s} {100 * nc / tot:5.1f} %"
              f"   distinct items {len(c)}")


if __name__ == "__main__":
    main()
