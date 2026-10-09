"""Who is running in a forkrun.py tree: games (parents), templates (frozen snapshots), children (branches).
Samples /proc every second; for each kind: processes and how many of them are on CPU (state R).
  python fork_top.py [samples=10]        (inside the container that runs forkrun.py, or on its host)
"""
import os
import sys
import time


def sample():
    ppid, state = {}, {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read()
            stat = open(f"/proc/{pid}/stat").read()
        except OSError:
            continue
        if b"forkrun.py" not in cmd or cmd.startswith(b"sh\0"):
            continue
        ppid[int(pid)] = int(stat[stat.rindex(")") + 2:].split()[1])
        # a game runs the bot in its agent thread: on CPU = any thread in state R
        run = False
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                t = open(f"/proc/{pid}/task/{tid}/stat").read()
            except OSError:
                continue
            run |= t[t.rindex(")") + 2] == "R"
        state[int(pid)] = "R" if run else "S"
    count = {}
    for pid in state:
        depth, q = 0, pid
        for _ in range(8):  # launcher -> game -> template -> child
            if q not in ppid:
                break
            q, depth = ppid[q], depth + 1
        kind = {1: "launcher", 2: "games", 3: "templates"}.get(depth, "children")
        n, r = count.get(kind, (0, 0))
        count[kind] = (n + 1, r + (state[pid] == "R"))
    return count


for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 10):
    c = sample()
    print("  ".join(f"{k} {c.get(k, (0, 0))[0]} (R {c.get(k, (0, 0))[1]})" for k in ("games", "templates", "children")),
          flush=True)
    time.sleep(1)
