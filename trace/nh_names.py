"""NetHack's object names and appearances (by class) and monster names from NLE, into nh_names.json, for the
unknown_item check of data_checks.py.   python nh_names.py   (needs nle)"""
import json

import nle.nethack as nh

CLASSES = {nh.WEAPON_CLASS: "weapon", nh.ARMOR_CLASS: "armor", nh.RING_CLASS: "ring", nh.AMULET_CLASS: "amulet",
           nh.TOOL_CLASS: "tool", nh.FOOD_CLASS: "food", nh.POTION_CLASS: "potion", nh.SCROLL_CLASS: "scroll",
           nh.SPBOOK_CLASS: "spellbook", nh.WAND_CLASS: "wand", nh.COIN_CLASS: "coin", nh.GEM_CLASS: "gem",
           nh.ROCK_CLASS: "rock", nh.BALL_CLASS: "ball", nh.CHAIN_CLASS: "chain", nh.VENOM_CLASS: "venom"}
objects = []
for i in range(nh.NUM_OBJECTS):
    o = nh.objclass(i)
    name, descr = nh.OBJ_NAME(o), nh.OBJ_DESCR(o)
    cls = CLASSES.get(ord(o.oc_class) if isinstance(o.oc_class, str) else o.oc_class)
    if name or descr:
        objects.append(dict(name=name, descr=descr, cls=cls))
monsters = [nh.permonst(i).mname for i in range(nh.NUMMONS)]
json.dump(dict(objects=objects, monsters=monsters), open("nh_names.json", "w"), indent=0)
print(len(objects), "objects,", len(monsters), "monsters;", sum(o["cls"] is None for o in objects), "without a class")
