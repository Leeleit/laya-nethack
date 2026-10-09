"""What the games list knows about a game's outcome (xlogfile fields), for choosing the humans' questions:
fields, how games end (death reasons grouped), turns and max depth quantiles.
  python outcome_fields.py nao/val_dwa_any4.json
"""
import collections
import json
import re
import sys

import numpy as np

games = json.load(open(sys.argv[1]))
print(f"{len(games)} games; fields: {sorted(games[0])}")
print("example:", {k: v for k, v in games[0].items() if k not in ("files",)})
kind = collections.Counter()
for g in games:
    d = g["death"]
    k = ("ascended" if d == "ascended" else "quit" if d.startswith("quit") else "escaped" if d.startswith("escaped")
         else "starved" if "starv" in d else "drowned" if "drown" in d else "petrified" if "petrif" in d or "stone" in d
         else "poison" if "poison" in d else "killed" if d.startswith("killed by") else re.sub(r"\W.*", "", d))
    kind[k] += 1
print("ends:", kind.most_common(20))
for f in ("turns", "maxlvl"):
    x = np.array([g[f] for g in games if isinstance(g.get(f), int)])
    print(f, "quantiles 10/25/50/75/90/99:", np.percentile(x, [10, 25, 50, 75, 90, 99]).round().tolist())
extra = [k for k in games[0] if k in ("achieve", "conduct", "realtime", "points", "hp", "maxhp")]
print("extra fields present:", extra)
