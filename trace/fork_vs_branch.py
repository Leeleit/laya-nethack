"""forkrun.py rep 0 against the old branch method (collect.py --branch passes) at the same points.

rep 0 of an option replays nothing and reseeds nothing, so it must give the old pass's outcome exactly:
the same death (turn, cause) or survival past fire + h.
  python fork_vs_branch.py --fork F --branches PREFIX [--point 0] [--options pray,elbereth]
"""
import argparse
import json
from pathlib import Path

H = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True)
    ap.add_argument("--branches", required=True, help="<prefix>-<option>-p<i>/<run>")
    ap.add_argument("--run", default="val-dwa-law-fem-100000-1000")
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--options", default="pray,elbereth")
    a = ap.parse_args()
    fork = Path(a.fork)
    for opt in a.options.split(","):
        d = Path(f"{a.branches}-{opt}-p{a.point}/{a.run}")
        res = {}
        for line in (d / "results.partial.jsonl").read_text().splitlines():
            r = json.loads(line)
            res[r["trajectory_id"]] = r
        logs = {}
        for lg in map(json.loads, (d / "branch_log.jsonl").read_text().splitlines()):
            logs[lg.get("seed")] = lg
        same = differ = missing = 0
        examples = []
        for seed, lg in logs.items():
            if seed is None or not lg.get("done") or seed not in res:
                continue
            f = fork / "games" / str(seed) / f"p{a.point}-{opt}-r0.json"
            if not f.exists():
                missing += 1
                continue
            c = json.loads(f.read_text())
            r = res[seed]
            old = ("died", r["turns"], r["cause_of_death"]) if r["cause_of_death"] and r["turns"] <= lg["turn"] + H \
                else ("alive",)
            new = ("died", c["end_turn"], c["cause"]) if c["status"] == "died" else (c["status"],)
            if old == new and lg["turn"] == c["fire_turn"]:
                same += 1
            else:
                differ += 1
                if len(examples) < 5:
                    examples.append((seed, lg["turn"], old, c["fire_turn"], new))
        print(f"{opt} p{a.point}: identical {same}, different {differ}, not in fork yet {missing}")
        for e in examples:
            print("   ", e)


if __name__ == "__main__":
    main()
