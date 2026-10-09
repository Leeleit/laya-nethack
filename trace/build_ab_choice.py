"""Choice data from a paired flag A/B (ab_points.py with --endgame): at a seed's branch point the bot either follows
the flag (the arm's game) or not (the base's game), and both games were played to the end from that same state. One
row per point, two questions on its state (risk-v2's text) -- one per option, the label the milestone that
option's game reached:
  "If the bot <arm text>, how far does the game get?"  /  "If the bot <base text>, ..."
One game per option: a label is one draw of a noisy outcome; models are judged on held-out seeds, the options compared
on the same points. Split by seed (crc32, as build_v2): test crc%5==0, val (crc//5)%10==0.
  python build_ab_choice.py --name soko --arm "goes up to Sokoban now and solves it before diving" \
      --base "keeps diving as it does now, skipping Sokoban" --out ab_choice_soko results/ab_points_soko*.jsonl
-> <out>/{train,val,test}.jsonl, <out>/outcomes.parquet; prints the per-option milestone rates
"""
import argparse
import json
import zlib
from pathlib import Path

import pandas as pd

GOAL = ["dies before Medusa's level", "reaches Medusa's level but not the castle", "reaches the castle but not below it",
        "gets into Gehennom"]


def goal(o):
    return 3 if o["gehennom"] else 2 if o["reached_castle"] else 1 if o["reached_medusa"] else 0


def split(seed):
    c = zlib.crc32(str(seed).encode())
    return "test" if c % 5 == 0 else "val" if (c // 5) % 10 == 0 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points", nargs="+")
    ap.add_argument("--name", required=True, help="the flag: question keys <name>_yes / <name>_no")
    ap.add_argument("--arm", required=True, help="what the bot does under the flag")
    ap.add_argument("--base", required=True, help="what it does without it")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    q = {f"{a.name}_{k}": {"type": "score", "criteria": GOAL,
                           "instructions": f"If the bot {text}, how far does the game get?"}
         for k, text in (("yes", a.arm), ("no", a.base))}
    rows, files = [], {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for path in a.points:
        for p in map(json.loads, open(path, encoding="utf-8")):
            s = split(p["seed"])
            g = {"yes": goal(p["arm"]), "no": goal(p["base"])}
            rows.append(dict(source=path, seed=p["seed"], turn=p["turn"], split=s, yes=g["yes"], no=g["no"],
                             yes_progress=p["arm"]["progress"], no_progress=p["base"]["progress"]))
            f = files.setdefault(s, open(Path(a.out, f"{s}.jsonl"), "w", encoding="utf-8", newline="\n"))
            f.write(json.dumps({"id": f"{a.name}#{p['seed']}", "state": p["state"], "questions": q,
                                "gold": {f"{a.name}_{k}": {"probabilities": {str(i): float(i == v) for i in range(4)},
                                                           "label": str(v)} for k, v in g.items()}}) + "\n")
    for f in files.values():
        f.close()
    df = pd.DataFrame(rows)
    df.to_parquet(Path(a.out, "outcomes.parquet"), index=False)
    print(f"{len(df)} points, {df.split.value_counts().to_dict()}")
    for k in ("yes", "no"):
        print(f"{a.name}_{k}: " + ", ".join(f"{GOAL[i]} {(df[k] == i).mean():.1%}" for i in range(4)) +
              f"; progress {df[k + '_progress'].mean():.3f}")
    print(f"milestone: flag better {(df.yes > df.no).mean():.1%}, worse {(df.yes < df.no).mean():.1%}, "
          f"same {(df.yes == df.no).mean():.1%}")


if __name__ == "__main__":
    main()
