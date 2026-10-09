"""Shares of the state-text defects found by reading random states (review_states.py), over a dataset:
  msg_in_inv   a game message glued into an inventory item ("q - 64 rocks.  You rebalance your load")
  glued_items  several items in one ("n - an egg.  o - a yellow gem")
  cut_item     an item cut off ("an uncurs", "a +1 long")
  dup_letter   one letter twice in an inventory (pickup messages stacked on an older listing)
  class_sym    items with the menu's class symbol ("a - ) a long sword"), as some players' option shows
  dec_walls    DECgraphics letters left in a map (x / q walls: "|....<..x", "lqqqk")
  bot_ui_msg   the bot's own interface lines in the recent messages ("# terrain", "Showing known terrain")
  status_bad   a status line that does not start with the attributes; map_text  words in the level map (a window over
               it); odd_syms  corridors drawn as | (||||); no_hero  no @ on a human level map; more_prompt  "--More--"; hp_bad  HP over max or max <= 0;
               deepest_bad  deepest level above the current one; score  S: in the status; gold_in_inv  a "$ - " item
               (stale: the status has gold); player_name  the player's name in the message; door7  a 7 on the map (the
               closed door some players draw as 7); non_ascii  bytes >= 0x80 on the map (UTF-8 walls); align_mismatch  the Character line's
               alignment is not the status line's; unknown_item  an item that is no NetHack object name
               (item_names.known: a cut item, or a custom fruit name)
  python data_checks.py <cases.jsonl> [--n 20000] [--show unknown_item]   (--show: what the check found, per state)"""
import argparse
import json
import random
import re

from item_names import known
from nao_corridor import dec_walls, map_text

INV = re.compile(r"\nInventory[^:]*: (.*)(?:\n|$)")  # the last line since the level map went
MSG = re.compile(r"\.  [A-Z]|\b(You|Your|It|The|There) [a-z]")
GLUED = re.compile(r"\.  [a-zA-Z$] - ")
CUT = re.compile(r"(\b(an?|the|\d+) (unc|uncurs|uncurse|blessed|cursed|uncursed|\+\d+)|\b(unc|uncurs|uncurse|dagge|lon))$")
SYM = re.compile(r"^[a-zA-Z$] - [^\w\s] ")
ITEMS_SPLIT = re.compile(r"; (?=[a-zA-Z$] - )")  # "(alternate weapon; not wielded)" has one too
UI = re.compile(r"# terrain|Showing known terrain|^# t\b| \| # \|")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--show")
    a = ap.parse_args()
    lines = open(a.cases, encoding="utf-8").read().splitlines()
    rows = random.Random(0).sample(lines, min(a.n, len(lines)))
    hits = dict(msg_in_inv=0, glued_items=0, cut_item=0, dup_letter=0, class_sym=0, dec_walls=0, bot_ui_msg=0,
                status_bad=0, map_text=0, odd_syms=0, no_hero=0, more_prompt=0, hp_bad=0, deepest_bad=0, score=0,
                gold_in_inv=0, player_name=0, door7=0, non_ascii=0, align_mismatch=0, unknown_item=0)
    example = {}
    for line in rows:
        row = json.loads(line)
        st = row["state"]
        found = set()
        # --terse rows (human_build.py): "<character> <version>", "HP a/b Dlvl d deepest d ...", the status
        # (no alignment in it), the bot's "Bot: ..." line (build_v2.py --terse), the messages if any -- no labels
        lines = st.split("\n")
        terse = len(lines) > 2 and lines[1].startswith("HP ") and " deepest " in lines[1]
        if terse:
            status_s = lines[2]
            rest = [l for l in lines[3:] if not l.startswith("Bot: ")]
            msg_s = rest[0] if rest and not rest[0].startswith(("Around:", "Inventory")) else None
            ch = re.match(r"\w+-\w+-(\w+)-", st)
        else:
            m = re.search(r"\nStatus: (.*)\n", st)
            status_s = m and m.group(1)
            m = re.search(r"\nMessage: (.*)\n", st)
            msg_s = m and m.group(1)
            ch = re.match(r"Character: \w+-\w+-(\w+)-", st)
        if status_s and not status_s.startswith("St:"):
            found.add("status_bad")
        if status_s and re.search(r"\bS:", status_s):
            found.add("score")
        al = status_s and re.search(r"\b(Lawful|Neutral|Chaotic)\b", status_s)
        if ch and al and ch.group(1).lower() != al.group(1)[:3].lower():
            found.add("align_mismatch")
        name = row.get("id", "").split("@")[0]
        if name and msg_s and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", msg_s, re.I):
            found.add("player_name")
        # the map checked: the whole level map in rows of the older format, else the map around the hero
        level = (st.split("\nLevel map:\n", 1)[-1] if "\nLevel map:\n" in st else
                 st.split("\nMap around the hero (@):\n", 1)[-1].split("\nInventory", 1)[0]
                 if "\nMap around the hero (@):\n" in st else "")
        if any(map_text(r) for r in level.split("\n")):  # monsters side by side (dZeFFF, aaaaa) are no text
            found.add("map_text")
        if "||||" in level:
            found.add("odd_syms")
        if "7" in level:
            found.add("door7")
        if max(level, default=" ") >= "\x80":
            found.add("non_ascii")
        if level and "@" not in level and "Bot behaviour" not in st:
            found.add("no_hero")
        if "--More--" in st:
            found.add("more_prompt")
        hm = re.search(r"\nHP (-?\d+)/(-?\d+)[.; ]", st)
        if hm and (int(hm.group(2)) <= 0 or int(hm.group(1)) > int(hm.group(2))):
            found.add("hp_bad")
        dm = re.search(r"Dlvl (\d+) (?:\(deepest so far|deepest) (\d+)", st)
        if dm and int(dm.group(2)) < int(dm.group(1)):
            found.add("deepest_bad")
        m = INV.search(st)
        if m and m.group(1) != "unknown":
            items = ITEMS_SPLIT.split(m.group(1))
            for it in items:
                body = it[4:]
                if MSG.search(body):
                    found.add("msg_in_inv")
                if GLUED.search(it):
                    found.add("glued_items")
                if CUT.search(body):
                    found.add("cut_item")
                if SYM.match(it):
                    found.add("class_sym")
            unknown = [it for it in items if not known(it[4:])]
            if unknown:
                found.add("unknown_item")
            if any(it.startswith("$ - ") for it in items) and "Bot behaviour" not in st:  # the bot's is live
                found.add("gold_in_inv")
            letters = [it[0] for it in items if it[1:4] == " - "]
            if len(letters) != len(set(letters)):
                found.add("dup_letter")
        crop = st.split("\nMap around the hero", 1)[-1].split("\nInventory", 1)[0].split("\n")[1:]
        if dec_walls(crop) or dec_walls(level.split("\n")):  # the extractor's rule: a grid bug x is no wall
            found.add("dec_walls")
        rm = re.search(r"\nRecent messages: (.*)\n", st)
        if rm and UI.search(rm.group(1)):
            found.add("bot_ui_msg")
        if a.show in found:
            if a.show == "unknown_item":
                print("\n".join("UNKNOWN\t" + it[4:] for it in unknown))
            elif a.show == "map_text":
                print("MAP:\n" + level)
            else:
                print(st)
        for k in found:
            hits[k] += 1
            example.setdefault(k, st[:0] + (m.group(1)[:160] if m and k != "dec_walls" and k != "bot_ui_msg" else
                                            (rm.group(1)[:160] if k == "bot_ui_msg" else "")))
    print(f"{len(rows)} states of {a.cases}")
    for k, v in hits.items():
        print(f"  {k:12s} {100 * v / len(rows):5.1f} %   {example.get(k, '')[:110]}")


if __name__ == "__main__":
    main()
