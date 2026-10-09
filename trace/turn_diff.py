"""How often the half-drawn "T:" spoilt the states' turns: the same games extracted before and after
nao_corridor.py's "turns never go back" -- states paired in order per game; a pair whose turn differs is counted, with
the difference (old - new) and the ttd it moves (turns left = game turns - turn).
  python turn_diff.py nao/val_dwa_states3.jsonl nao/val_dwa_states3e.jsonl
"""
import collections
import json
import statistics
import sys


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        g = json.loads(line)
        out[(g["name"], g["start"])] = g
    return out


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    common = old.keys() & new.keys()
    n_states = diff = games_diff = other_count = 0
    deltas, kinds, rel, worst = [], collections.Counter(), [], []
    for k in common:
        a, b = old[k]["states"], new[k]["states"]
        if len(a) != len(b):
            other_count += 1
        hit = False
        for s, t in zip(a, b):
            n_states += 1
            if s["turn"] != t["turn"]:
                diff += 1
                hit = True
                deltas.append(s["turn"] - t["turn"])
                worst.append((s["turn"] - t["turn"], f"{k[0]}@{k[1]}", s["turn"], t["turn"], old[k]["turns"]))
                kinds[s["kind"]] += 1
                left = old[k]["turns"] - t["turn"]
                if left > 0:
                    rel.append((t["turn"] - s["turn"]) / left)
        games_diff += hit
    print(f"{len(common)} games in both ({len(old)} old, {len(new)} new); {other_count} with another state count")
    print(f"states paired {n_states}, turn differs {diff} ({100 * diff / max(n_states, 1):.3f} %) "
          f"in {games_diff} games ({100 * games_diff / max(len(common), 1):.2f} %); kinds {dict(kinds)}")
    if deltas:
        d = sorted(deltas)
        print(f"old - new turn: median {statistics.median(d)}, p10 {d[len(d) // 10]}, p90 {d[9 * len(d) // 10]}, "
              f"min {d[0]}, max {d[-1]}; old turn = -1: {sum(1 for k in common for s in old[k]['states'] if s['turn'] < 0)}"
              f" states, new: {sum(1 for k in common for s in new[k]['states'] if s['turn'] < 0)}")
    past = [w for w in worst if w[3] > w[4] + 50]
    print(f"new turn past the game's end: {len(past)} states in {len({w[1] for w in past})} games")
    for w in sorted(worst)[:8] + sorted(worst)[-4:]:
        print(f"  old - new {w[0]:9d}  {w[1]}  old {w[2]} new {w[3]} game turns {w[4]}")
    if rel:
        r = sorted(rel)
        print(f"turns left overstated by (new - old) / left: median {statistics.median(r):.3f}, p90 {r[9 * len(r) // 10]:.3f}")


if __name__ == "__main__":
    main()
