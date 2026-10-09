"""Random states to check by eye, each twice: the raw screen as captured (human: the nao_corridor.py states screen,
24 rows, and its inventory as extracted; bot: a trace record's "scr" and its inventory) and then the text the model
gets (human_build.state_text / build_v2.state_text).
  python review_raw.py --states nao/probe/hum2000.jsonl --traces /data/.../runs/traces2 --human 50 --bot 50 --out r.txt"""
import argparse
import glob
import json
import random
import subprocess

import build_v2
import human_build


def raw(rows):
    return "\n".join(f"{i:2d}|{r.rstrip()}" for i, r in enumerate(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--human", type=int, default=50)
    ap.add_argument("--bot", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    games = [json.loads(line) for p in a.states for line in open(p, encoding="utf-8")]
    picks = rng.sample([(gi, si) for gi, g in enumerate(games) for si in range(len(g["states"]))], a.human)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for i, (gi, si) in enumerate(picks):
            g, s = games[gi], games[gi]["states"][si]
            f.write(f"######## H{i} {g['name']}@{g['start']} turn {s['turn']} ({s['kind']}), cursor x={s['x']} "
                    f"y={s['y']}\n---- raw screen\n{raw(s['screen'])}\n---- raw inventory: {' | '.join(s['inv'])}\n"
                    f"---- model text\n{human_build.state_text(g, s)}\n")
        paths = sorted(glob.glob(f"{a.traces}/*/trace/*.jsonl.zst"))
        for i, path in enumerate(rng.sample(paths, a.bot)):
            lines = subprocess.run(["zstdcat", path], capture_output=True, check=True).stdout.splitlines()
            identity = path.split("/")[-3].rsplit("-", 2)[0]
            recs = [(idx, rec, back, list(msgs), list(inv)) for idx, rec, back, msgs, inv in build_v2.walk(lines[1:])]
            idx, rec, back, msgs, inv = recs[rng.randrange(len(recs))]
            f.write(f"######## B{i} {path.split('/')[-3]}/{path.split('/')[-1]} record {idx}, hero x={rec['b'][0]} "
                    f"y={rec['b'][1] + 1}\n---- raw screen\n{raw(rec['scr'])}\n---- raw inventory: {' | '.join(inv)}\n"
                    f"---- model text\n{build_v2.state_text(identity, rec['s'], rec, back, msgs, inv)}\n")


if __name__ == "__main__":
    main()
