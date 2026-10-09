"""Examples for junk_scan.py's categories: one game's events around each turn that goes back, or its states
whose status line names someone else.
  python junk_show.py back <events.jsonl> name@start [--n 3]
  python junk_show.py name <states.jsonl> name@start [--n 3]
"""
import argparse
import json


def find(path, key):
    name, start = key.rsplit("@", 1)
    for line in open(path, encoding="utf-8"):
        if line.startswith('{"name": ' + json.dumps(name) + ', ') or f'"name": {json.dumps(name)}' in line[:400]:
            g = json.loads(line)
            if g["name"] == name and str(g["start"]) == start:
                return g
    raise SystemExit(f"{key} not found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["back", "name", "odd"])
    p.add_argument("path")
    p.add_argument("key")
    p.add_argument("--n", type=int, default=3)
    a = p.parse_args()
    g = find(a.path, a.key)
    if a.mode == "odd":
        import re
        odd = re.compile(r"[\x00-\x1f\x7f-\xff]")
        pre = [e for e in g["events"] if e[0] < 0]
        print(f"{a.key}: {len(g['events'])} events, {len(pre)} before the first status")
        for e in pre[:a.n * 4]:
            print(f"  pre {e}")
        for e in [e for e in g["events"] if odd.search(e[-1])][:a.n]:
            print(f"  odd {e[:6]} {e[-1]!r}")
        return
    if a.mode == "back":
        ev, last, shown = g["events"], -1, 0
        print(f"{a.key}: {len(ev)} events")
        for i, e in enumerate(ev):
            if 0 <= e[0] < last and shown < a.n:
                shown += 1
                print(f"--- back at event {i} (turn {last} -> {e[0]})")
                for j in range(max(0, i - 4), min(len(ev), i + 3)):
                    print(f"  {j:6d} {ev[j]}")
            last = max(last, e[0])
    else:
        name10, shown = g["name"][:10].lower(), 0
        print(f"{a.key}: {g.get('character')} {g.get('version')} turns {g.get('turns')}, {len(g['states'])} states")
        for s in g["states"]:
            if not s["screen"][22].lstrip("[").lower().startswith(name10) and shown < a.n:
                shown += 1
                print(f"--- turn {s['turn']} dlvl {s['dlvl']} kind {s['kind']}")
                print("  " + s["screen"][22] + "\n  " + s["screen"][23])


if __name__ == "__main__":
    main()
