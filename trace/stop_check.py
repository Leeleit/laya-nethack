"""Check of `nao_corridor.py states` soft stop (Linux: fork, so the patched game_states reaches the workers).
200 fake games (0.05 s each), 4 workers; <out>.stop appears after 20 written -> exit 3, every started game
written, none twice; the rerun finishes the rest -> exit 0, all 200 once.
  python stop_check.py <scratch dir>
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

if len(sys.argv) > 2:  # the child: run states() with a fake game_states
    import nao_corridor

    def fake(g):
        time.sleep(0.05)
        return dict(name=g["name"], start=g["start"], states=[1])
    nao_corridor.game_states = fake
    sys.argv = ["nao_corridor.py", "states", "--games", sys.argv[1], "--out", sys.argv[2], "--workers", "4"]
    nao_corridor.main()
    sys.exit(0)

d = Path(sys.argv[1])
d.mkdir(parents=True, exist_ok=True)
games, out = d / "games.json", d / "out.jsonl"
games.write_text(json.dumps([dict(name=f"p{i}", start=i) for i in range(200)]))
out.unlink(missing_ok=True)
stop = Path(f"{out}.stop")


def lines():
    return [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []


def touch_after(k):
    while len(lines()) < k:
        time.sleep(0.01)
    stop.touch()


t = threading.Thread(target=touch_after, args=(20,))
t.start()
r1 = subprocess.run([sys.executable, __file__, str(games), str(out)], capture_output=True, text=True)
t.join()
g1 = lines()
print("run 1: exit", r1.returncode, "| written", len(g1), "| stop file left", stop.exists())
print(r1.stdout.strip())
r2 = subprocess.run([sys.executable, __file__, str(games), str(out)], capture_output=True, text=True)
g2 = lines()
keys = [(g["name"], g["start"]) for g in g2]
print("run 2: exit", r2.returncode, "| written", len(g2), "| distinct", len(set(keys)))
print(r2.stdout.strip())
ok = (r1.returncode == 3 and 20 <= len(g1) < 200 and not stop.exists()
      and r2.returncode == 0 and len(keys) == 200 == len(set(keys)))
print("OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
