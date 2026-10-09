"""The most frequent message lines ("m") of random bot traces -- to find the bot's own interface lines that should not
count as game messages in a state's "Recent messages".
  python msg_freq.py <runs dir> [--traces 60] [--top 60]"""
import argparse
import collections
import glob
import json
import random
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--traces", type=int, default=60)
    ap.add_argument("--top", type=int, default=60)
    a = ap.parse_args()
    paths = random.Random(0).sample(sorted(glob.glob(f"{a.root}/*/trace/*.jsonl.zst")), a.traces)
    c, n = collections.Counter(), 0
    for p in paths:
        for line in subprocess.run(["zstdcat", p], capture_output=True, check=True).stdout.splitlines()[1:]:
            m = json.loads(line).get("m")
            if m:
                n += 1
                c[m[:90]] += 1
    print(f"{n} messages in {len(paths)} traces")
    for m, k in c.most_common(a.top):
        print(f"{100 * k / n:6.2f} %  {m}")


if __name__ == "__main__":
    main()
