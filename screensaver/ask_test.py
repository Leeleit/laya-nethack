"""Check laya_serve answers the screensaver's question: risk-v2 on a few states of one trace, with timings.
  python ask_test.py <trace> [address]"""
import sys
import time

import saver

path = sys.argv[1]
addr = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:6000"
laya = saver.Laya(addr)
identity, lines = saver.read(path)
states = [(r["b"], saver.state_text(identity, r["s"], r, back, list(msgs), inv))
          for _, r, back, msgs, inv in saver.walk(lines)]
for i in (len(states) // 4, len(states) // 2, len(states) - 40, len(states) - 5, len(states) - 1):
    b, st = states[i]
    n0 = laya.asked
    t = time.time()
    laya.ask(st)
    while laya.asked == n0 and time.time() - t < 120:
        time.sleep(0.01)
    p = laya.probs
    print(f"frame {i}/{len(states)} Dlvl {b[saver.DEPTH]} HP {b[saver.HP]}/{b[saver.MAXHP]}: {laya.status} "
          f"{time.time() - t:.2f} s  " + (" ".join(f"{x:.3f}" for x in p) if p else "no answer"))
