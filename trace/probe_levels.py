"""probe: per level of a bot trace (depth >= 20) -- water squares seen, the bot's strategies (all stack names) and
telling messages; python probe_levels.py <trace.jsonl.zst> [...]"""
import json
import subprocess
import sys
from collections import Counter

from bot_profile import DEPTH, DNUM, TIME

for path in sys.argv[1:]:
    lines = subprocess.run(["zstd", "-dcq", path], capture_output=True, check=True).stdout.splitlines()[1:]
    lv, b = {}, None
    for line in lines:
        r = json.loads(line)
        b = r.get("b", b)
        if b is None or b[DEPTH] < 20:
            continue
        key = (b[DNUM], b[24], b[DEPTH])
        v = lv.setdefault(key, dict(t0=b[TIME], water=0, strat=Counter(), msg=[]))
        v["t1"] = b[TIME]
        for s in r.get("s", []):
            v["strat"][".".join(p for p in s.split(".") if not p.startswith("<"))[-45:]] += 1
        if "scr" in r:
            v["water"] = max(v["water"], sum(row.count("}") for row in r["scr"][1:22]))
        m = r.get("m", "")
        if any(w in m.lower() for w in ("dig", "drawbridge", "medusa", "water", "trap door", "fall")):
            v["msg"].append(m[:70])
    print("==", path[-22:])
    for k, v in lv.items():
        print(k, f"t {v['t0']}-{v['t1']} water {v['water']}")
        print("   ", "; ".join(f"{s} {n}" for s, n in v["strat"].most_common(12)))
        print("   ", " | ".join(dict.fromkeys(v["msg"]))[:400])
