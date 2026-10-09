"""CPU of every thread of the forkrun.py processes over an interval: who burns the CPU inside a game process.
  python thread_top.py [seconds=5]        (inside the container that runs forkrun.py)
"""
import os
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def threads():
    out = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if b"forkrun.py" not in open(f"/proc/{pid}/cmdline", "rb").read():
                continue
            for tid in os.listdir(f"/proc/{pid}/task"):
                s = open(f"/proc/{pid}/task/{tid}/stat").read()
                f = s[s.rindex(")") + 2:].split()
                out[pid, tid] = (s[s.index("(") + 1:s.rindex(")")], int(f[11]), int(f[12]), int(f[36]))
        except OSError:
            continue
    return out


dt = float(sys.argv[1]) if len(sys.argv) > 1 else 5
a = threads()
time.sleep(dt)
b = threads()
for k in sorted(b, key=lambda k: -(b[k][1] + b[k][2] - a.get(k, b[k])[1] - a.get(k, b[k])[2])):
    name, u, s, cpu = b[k]
    u0, s0 = a.get(k, b[k])[1:3]
    print(f"pid {k[0]:>6} tid {k[1]:>6} {name:16s} cpu {cpu:>2}  user {100 * (u - u0) / HZ / dt:5.1f}%  "
          f"sys {100 * (s - s0) / HZ / dt:5.1f}%")
