"""The games of a nao_corridor games list that have no message log yet (a second pass with --events
over the games extracted before it -- Val-Dwa and the first ~319k other-role games): the list minus the (name, start)
of every line of the given events files.
  python make_noev.py nao/other_any4_cut.json nao/other_noev.json --events nao/other_states3.jsonl.events.jsonl
"""
import argparse
import json
import re

KEY = re.compile(r'^\{"name": ("(?:[^"\\]|\\.)*"), "start": (\d+)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("games")
    p.add_argument("out")
    p.add_argument("--events", nargs="*", default=[])
    a = p.parse_args()
    have = set()
    for path in a.events:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = KEY.match(line)
                if m:
                    have.add((json.loads(m.group(1)), int(m.group(2))))
    games = json.load(open(a.games))
    left = [g for g in games if (g["name"], g["start"]) not in have]
    json.dump(left, open(a.out, "w"))
    print(f"{a.games}: {len(games)} games, {len(have)} with a log -> {len(left)} without -> {a.out}")


if __name__ == "__main__":
    main()
