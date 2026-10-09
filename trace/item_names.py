"""Is an inventory item's text a NetHack object name? The grammar of objnam.c (doname / xname) over the names NLE
gives (nh_names.json from nh_names.py): count or article, BUC and the other prefixes, enchantment, the class-wise name
forms ("potion of X", "X potion", "scroll labeled X", "X corpse", "pair of X"...), "(...)" and "named / called X"
suffixes. Custom fruit names (the player's fruit option: "Taco Bell") cannot be known and come out unknown.
  known("an uncursed +3 small shield (being worn)") -> True;  known("an uncursed +3 smal") -> False"""
import json
import re
from pathlib import Path

ARTIFACTS = ["Excalibur", "Stormbringer", "Mjollnir", "Cleaver", "Grimtooth", "Orcrist", "Sting", "Magicbane",
             "Frost Brand", "Fire Brand", "Dragonbane", "Demonbane", "Werebane", "Grayswandir", "Giantslayer",
             "Ogresmasher", "Trollsbane", "Vorpal Blade", "Snickersnee", "Sunsword", "Orb of Detection",
             "Heart of Ahriman", "Sceptre of Might", "Staff of Aesculapius", "Magic Mirror of Merlin",
             "Eyes of the Overworld", "Mitre of Holiness", "Longbow of Diana", "Master Key of Thievery",
             "Tsurugi of Muramasa", "Platinum Yendorian Express Card", "Orb of Fate", "Eye of the Aethiopica",
             "Bell of Opening", "Candelabrum of Invocation", "Book of the Dead", "Amulet of Yendor"]
PREFIX = re.compile(r"^(an?|the|\d+|uncursed|blessed|cursed|unholy|holy|greased|poisoned|empty|lit|partly eaten|"
                    r"partly used|diluted|moist|wet|trapped|locked|unlocked|unlockable|broken|historic|unpaid|"
                    r"(very |thoroughly )?(rusty|burnt|corroded|rotted)|rustproof|fireproof|corrodeproof|rotproof|"
                    r"erodeproof|[+-]\d+)\s+", re.I)
SUFFIX = re.compile(r"\s*\([^()]*\)$|\s+(named|called)\s.*$|\s+containing \d+ items?$")
PLURAL = [(r"ves$", "f"), (r"ves$", "fe"), (r"ies$", "y"), (r"es$", ""), (r"s$", ""), (r"(?<=e)eth$", "ooth"),
          (r"ice$", "ouse"), (r"i$", "us"), (r"ae$", "a"), (r"men$", "man"), (r"ina$", "en")]


def _names():
    d = json.loads((Path(__file__).parent / "nh_names.json").read_text())
    k = {a.lower() for a in ARTIFACTS} | {"gold piece", "slime mold"}
    forms = {"potion": ["potion of {n}", "{d} potion"], "scroll": ["scroll of {n}", "scroll labeled {d}", "{d} scroll"],
             "spellbook": ["spellbook of {n}", "{d} spellbook"], "wand": ["wand of {n}", "{d} wand"],
             "ring": ["ring of {n}", "{d} ring"], "amulet": ["{n}", "{d} amulet"],
             "gem": ["{n}", "{n} stone", "{d} gem", "{d} stone"]}
    for o in d["objects"]:
        n, de = (o["name"] or "").lower(), (o["descr"] or "").lower()
        k.add(o["cls"] or "")
        for f in forms.get(o["cls"], ["{n}", "{d}"]):
            if ("{n}" in f and n) or ("{d}" in f and de):
                k.add(f.format(n=n, d=de))
    k |= {"potion of holy water", "potion of unholy water", "novel", "paperback book", "amulet", "gem", "stone",
          "cheap plastic imitation of the amulet of yendor", "tin of spinach", "candy bar",
          "cloak", "boots", "gloves", "helmet", "shield", "armor",  # an unidentified piece "called X"
          "scroll of mail", "stamped scroll", "leather spellbook",  # NLE has no mail; 3.4.3's "leather"
          "shito", "tanto", "wakizashi", "ninja-to", "nunchaku", "naginata", "osaku", "koto", "kabuto", "yugake",
          "gunyoki", "potion of sake"}  # a Samurai's Japanese names (objnam.c Japanese_items)
    mons = {m.lower() for m in d["monsters"]}
    return k, mons


KNOWN, MONSTERS = _names()
MON_FORM = re.compile(r"^tin of (.+?)(?: meat)?$|^(?!tin of )(.+?) (corpse|egg|meat)$|"
                      r"^(statue|figurine) of (?:an? |the )?(.+)$")


def singulars(s):
    """s and its forms with one word made singular (the first word: "potions of X", the last: "darts")."""
    out = {s}
    w = s.split(" ")
    for i in {0, len(w) - 1, w.index("of") - 1 if "of" in w[1:] else 0}:
        for pat, rep in PLURAL:
            if re.search(pat, w[i]):
                out.add(" ".join(w[:i] + [re.sub(pat, rep, w[i])] + w[i + 1:]))
    return out


def core(text):
    """The item's name without count, prefixes and suffixes, lower case."""
    prev = None
    s = text.strip()
    while prev != s:
        prev = s
        s = SUFFIX.sub("", PREFIX.sub("", s)).strip()
    s = re.sub(r"^(pairs?|sets?) of ", "", s, flags=re.I)
    return s.lower()


def known(text):
    s = core(text)
    for c in singulars(s):
        if c in KNOWN or re.match(r"((very )?(small|medium|large) )?glob of ", c) or c.endswith("'s corpse"):
            return True
        m = MON_FORM.match(c)
        if m and (m.group(1) or m.group(2) or m.group(5) or "") in MONSTERS | {"spinach", "empty"}:
            return True
        if m and m.group(2) and any(x in MONSTERS for x in singulars(m.group(2))):
            return True
    return False
