"""Where Excalibur's blessing goes missing in the logs (prep_survey.py: the message in 28 % of the
ascended Val-Dwa games of a probe, while 92 % of the ascended carry Excalibur by Dlvl 25): over the ascended games,
how many have the blessing message, how many name Excalibur anywhere, and for those that name it without the
message -- the events around the first "Excalibur" and around each "# dip".
  python excal_check.py nao/val_dwa_states3e.jsonl --games 300 --show 4
"""
import argparse
import json
import re

BLESS = re.compile(r"hand reaches up to bless the sword")
NAMED = re.compile(r"Excalibur")
KEY = re.compile(r'"name": "([^"]*)", "start": (\d+)')


def show(ev, i, before=3, after=3):
    for e in ev[max(i - before, 0):i + after + 1]:
        print(f"   {e[0]:6d} Dlvl {e[1]:2d} XL {e[5]:2d}  {e[-1][:150]}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("states")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--show", type=int, default=4)
    a = p.parse_args()
    asc = set()
    for line in open(a.states, encoding="utf-8"):
        g = json.loads(line)
        if "ascended" in g["death"]:
            asc.add((g["name"], g["start"]))
    n = blessed = named = shown = 0
    for line in open(a.states + ".events.jsonl", encoding="utf-8"):
        m = KEY.search(line[:300])
        if not m or (m.group(1), int(m.group(2))) not in asc:
            continue
        ev = json.loads(line)["events"]
        n += 1
        b = any(BLESS.search(e[-1]) for e in ev)
        k = next((i for i, e in enumerate(ev) if NAMED.search(e[-1])), None)
        blessed += b
        named += k is not None
        if k is not None and not b and shown < a.show:
            shown += 1
            print(f"--- {m.group(1)}@{m.group(2)}: first 'Excalibur'")
            show(ev, k)
            for i in [i for i, e in enumerate(ev) if e[-1].startswith("# dip")][:3]:
                print("  # dip:")
                show(ev, i, 1, 4)
        if n >= a.games:
            break
    print(f"ascended games {n}: blessing message {blessed} ({100 * blessed / n:.0f} %), "
          f"'Excalibur' anywhere {named} ({100 * named / n:.0f} %)")


if __name__ == "__main__":
    main()
