"""What the message logs hold around a message (prep_survey.py checks): for the first games whose log
has a match of PATTERN, the matching event and the K events after it.
  python msg_after.py nao/val_dwa_states3e.jsonl.events.jsonl "What do you want to wear" --games 5 --after 4
"""
import argparse
import json
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("events")
    p.add_argument("pattern")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--after", type=int, default=4)
    p.add_argument("--per-game", type=int, default=2)
    a = p.parse_args()
    rx = re.compile(a.pattern)
    shown = 0
    for line in open(a.events, encoding="utf-8"):
        if not rx.search(line):
            continue
        g = json.loads(line)
        ev = g["events"]
        hits = [i for i, e in enumerate(ev) if rx.search(e[-1])][:a.per_game]
        print(f"--- {g['name']}@{g['start']}")
        for i in hits:
            for e in ev[max(i - 1, 0):i + 1 + a.after]:
                print(f"   {e[0]:6d} Dlvl {e[1]:2d} AC {e[4]:3d}  {e[-1][:150]}")
            print()
        shown += 1
        if shown >= a.games:
            break


if __name__ == "__main__":
    main()
