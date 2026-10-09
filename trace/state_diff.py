"""Rows of two datasets with the same id but different states (drop_check found some 14d -> 15d): how many,
and the first differing lines of a few.
  python state_diff.py human_data14d human_data15d train [--show 3]
"""
import difflib
import json
import sys
from pathlib import Path

old, new, split = sys.argv[1:4]
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 3
a = {r["id"]: r["state"] for r in map(json.loads, open(Path(old, f"{split}.jsonl"), encoding="utf-8"))}
b = {r["id"]: r["state"] for r in map(json.loads, open(Path(new, f"{split}.jsonl"), encoding="utf-8"))}
diff = [i for i in b if i in a and a[i] != b[i]]
print(f"{split}: {len(diff)} of {len(b)} shared rows differ")
for i in diff[:show]:
    print("==", i)
    for line in difflib.unified_diff(a[i].splitlines(), b[i].splitlines(), lineterm="", n=0):
        print("  ", line[:220])
