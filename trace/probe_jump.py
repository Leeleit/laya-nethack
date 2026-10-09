"""probe: level changes of a bot trace with the messages around them; python probe_jump.py <trace.jsonl.zst>"""
import json
import subprocess
import sys

from bot_profile import DEPTH, DNUM, TIME

lines = subprocess.run(["zstd", "-dcq", sys.argv[1]], capture_output=True, check=True).stdout.splitlines()[1:]
prev, b, msgs, strat = None, None, [], []
for line in lines:
    r = json.loads(line)
    b = r.get("b", b)
    if r.get("m"):
        msgs = (msgs + [r["m"][:90]])[-4:]
    strat = r.get("s", strat)
    if b is None:
        continue
    key = (b[DNUM], b[24], b[DEPTH])
    if key != prev:
        print(b[TIME], prev, "->", key, "|", " / ".join(msgs), "|", strat[-1:] if strat else "")
        prev = key
