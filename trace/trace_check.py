"""Find trace files whose lines do not all parse (e.g. cut off when the disk filled up).
  python trace_check.py <trace dir> [--move DIR]   -> prints the bad files; --move puts them aside"""
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path


def bad(path):
    try:
        out = subprocess.run(["zstd", "-dcq", str(path)], capture_output=True, check=True).stdout
        for line in out.splitlines():
            json.loads(line)
    except (subprocess.CalledProcessError, ValueError) as e:
        return str(path), type(e).__name__
    return None


if __name__ == "__main__":
    files = sorted(Path(sys.argv[1]).glob("*.jsonl.zst"))
    with Pool(16) as pool:
        found = [x for x in pool.map(bad, files, chunksize=8) if x]
    print(f"{len(files)} traces, {len(found)} bad")
    for p, e in found:
        print(" ", p, e)
        if "--move" in sys.argv:
            d = Path(sys.argv[sys.argv.index("--move") + 1])
            d.mkdir(parents=True, exist_ok=True)
            shutil.move(p, d / Path(p).name)
