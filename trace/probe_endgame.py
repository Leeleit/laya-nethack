"""probe: bot_endgame.game() on the first N val1 games (python probe_endgame.py [N])"""
import glob
import sys

import bot_endgame as e

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
for f in sorted(glob.glob("../runs/val1/val-dwa-law-fem-100000-1000/trace/*.zst"))[:N]:
    _, _, _, d = e.game(f)
    print(f[-20:], {k: d[k] for k in ["section", "depth", "medusa", "castle", "strategy",
                                      "t_medusa", "t_mazes", "t_castle"]})
