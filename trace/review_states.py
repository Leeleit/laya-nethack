"""Random state texts to read by eye (data errors like a game message inside the inventory): N human states of a
human_build.py dataset and N bot states rendered with build_v2.state_text from random records of random traces.
  python review_states.py --human human_data3/test.jsonl --traces /data/.../runs/traces2 --n 50 --out review.txt"""
import argparse
import glob
import json
import random
import subprocess

from build_v2 import state_text, walk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lines = open(a.human, encoding="utf-8").read().splitlines()
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for i, line in enumerate(rng.sample(lines, a.n)):
            r = json.loads(line)
            f.write(f"######## H{i} {r['id']} gold={json.dumps({q: g['label'] for q, g in r['gold'].items()})}\n")
            f.write(r["state"] + "\n")
        paths = sorted(glob.glob(f"{a.traces}/*/trace/*.jsonl.zst"))
        for i, path in enumerate(rng.sample(paths, a.n)):
            raw = subprocess.run(["zstdcat", path], capture_output=True, check=True).stdout.splitlines()
            identity = path.split("/")[-3].rsplit("-", 2)[0]
            recs = [(idx, rec, back, list(msgs), inv) for idx, rec, back, msgs, inv in walk(raw[1:])]  # msgs: one deque
            idx, rec, back, msgs, inv = recs[rng.randrange(len(recs))]
            f.write(f"######## B{i} {path.split('/')[-3]}/{path.split('/')[-1]} record {idx}\n")
            f.write(state_text(identity, rec["s"], rec, back, msgs, inv) + "\n")


if __name__ == "__main__":
    main()
