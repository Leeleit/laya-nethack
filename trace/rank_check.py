"""States of another game in a human game's states (an account's games share ttyrec files): the rank
title on the status line ("<name> the Hatamoto") must be one of the xlogfile role's ranks (role.c 3.4.3 / 3.6).
A player running two games at once interleaves their frames inside one game's [start, end] window, which the time
filter cannot split: a game with another role's title is listed in --bad-out, for human_build.py --exclude (the whole
game: its states carry the inventory and deepest level over from the frames before). A title that is no role's rank
is a polymorphed hero's monster name -- counted, not excluded.
  python rank_check.py <states.jsonl> [--bad-out bad_games.txt]   (nao_corridor states)"""
import argparse
import collections
import json
import re

RANKS = {  # role.c: the rank titles of each role, male / female
    "Arc": "Digger Field Worker Investigator Exhumer Excavator Spelunker Speleologist Collector Curator",
    "Bar": "Plunderer Plunderess Pillager Bandit Brigand Raider Reaver Slayer Chieftain Chieftainess Conqueror "
           "Conqueress",
    "Cav": "Troglodyte Aborigine Wanderer Vagrant Wayfarer Roamer Nomad Rover Pioneer",
    "Hea": "Rhizotomist Empiric Embalmer Dresser Medicus ossium Medica ossium Herbalist Magister Magistra Physician "
           "Chirurgeon",
    "Kni": "Gallant Esquire Bachelor Sergeant Knight Banneret Chevalier Chevaliere Seignieur Dame Paladin",
    "Mon": "Candidate Novice Initiate Student of Stones Student of Waters Student of Metals Student of Winds "
           "Student of Fire Master",
    "Pri": "Aspirant Acolyte Adept Priest Priestess Curate Canon Canoness Lama Patriarch Matriarch High Priest "
           "High Priestess",
    "Rog": "Footpad Cutpurse Rogue Pilferer Robber Burglar Filcher Magsman Magswoman Thief",
    "Ran": "Tenderfoot Lookout Trailblazer Reconnoiterer Reconnoiteress Scout Arbalester Archer Sharpshooter Marksman "
           "Markswoman",
    "Sam": "Hatamoto Ronin Ninja Kunoichi Joshu Ryoshu Kokushu Daimyo Kuge Shogun",
    "Tou": "Rambler Sightseer Excursionist Peregrinator Peregrinatrix Traveler Journeyer Voyager Explorer Adventurer",
    "Val": "Stripling Skirmisher Fighter Man-at-arms Woman-at-arms Warrior Swashbuckler Hero Heroine Champion Lord "
           "Lady",
    "Wiz": "Evoker Conjurer Thaumaturge Magician Enchanter Enchantress Sorcerer Sorceress Necromancer Wizard Mage",
}
TITLE = re.compile(r"the ([A-Z][\w-]*(?: [a-z]+ [A-Z]\w+| ossium)?)")

ap = argparse.ArgumentParser()
ap.add_argument("states")
ap.add_argument("--bad-out", help="append the games with another role's rank title here, one name@start a line")
a = ap.parse_args()
n = bad = 0
games, example, foreign, n_games = collections.Counter(), {}, collections.Counter(), 0
for line in open(a.states, encoding="utf-8"):
    g = json.loads(line)
    ranks = RANKS.get(g["character"][:3], "")
    others = [v for k, v in RANKS.items() if k != g["character"][:3]]
    n_games += 1
    for s in g["states"]:
        n += 1
        m = TITLE.search(s["screen"][22].split("St:")[0])
        if m and m.group(1) not in ranks:  # a polymorphed hero shows a monster name: also not the game's rank
            bad += 1
            games[f"{g['name']}@{g['start']}"] += 1
            example.setdefault(f"{g['name']}@{g['start']}", s["screen"][22][:60])
            if any(m.group(1) in o for o in others):
                foreign[f"{g['name']}@{g['start']}"] += 1
print(f"{n} states, {bad} with a rank title not of the role ({len(games)} games)")
print(f"another role's title: {sum(foreign.values())} states, {len(foreign)} of {n_games} games "
      f"({100 * len(foreign) / max(n_games, 1):.2f} %){' -> ' + a.bad_out if a.bad_out else ''}")
if a.bad_out:
    with open(a.bad_out, "a", encoding="utf-8") as f:
        f.writelines(k + "\n" for k in sorted(foreign))
for k, v in games.most_common(30):
    print(f"  {v:4d} {k}  {example[k]}")
