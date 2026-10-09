"""Check the screensaver without a display: render frames of one trace into PNGs.
  python shot.py <trace> <out prefix> [frame indices...] [--laya 127.0.0.1:6000 --name risk-v2-l20]
(SDL dummy video driver.) Without --laya the panel shows made-up numbers (a layout check); with it, every frame asks
laya_serve.py, as the screensaver does."""
import argparse
import os
from multiprocessing.connection import Client
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

import saver  # noqa: E402


class A:
    window = True
    laya = ""
    fps = 8
    traces = []
    human = False
    name = "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("prefix")
    ap.add_argument("frames", nargs="*", type=int)
    ap.add_argument("--laya", default="", help="laya_serve.py address; '' = made-up numbers")
    ap.add_argument("--name", default="risk-v2-l20")
    a = ap.parse_args()
    want = set(a.frames) or {0, 50, 500}
    s = saver.Saver(A)
    identity, lines = saver.read(a.trace)
    g = dict(seed=Path(a.trace).name.split(".")[0], run=Path(a.trace).parent.parent.name, deepest=0, gehennom=False,
             last_depth=0)
    s.session["games"] = 1
    conn = None
    if a.laya:
        from bc_vocab import address
        conn = Client(address(a.laya), authkey=b"laya")
        s.laya.status = a.name
    else:
        s.laya.probs, s.laya.status = [0.02, 0.05, 0.1, 0.2, 0.63], "made-up numbers"
    for i, (_, r, back, msgs, inv) in enumerate(saver.walk(lines)):
        b = r["b"]
        g["deepest"] = max(g["deepest"], b[saver.DEPTH] if b[saver.DNUM] != 2 else 0)
        g["gehennom"] = g["gehennom"] or b[saver.DNUM] == 1
        if i not in want:
            continue
        if conn:
            conn.send([(saver.state_text(identity, r["s"], r, back, list(msgs), inv), saver.QUESTION)])
            s.laya.probs = saver.Laya.softmax(conn.recv()[0])
        s.screen.fill(saver.BG)
        s.draw_screen(r)
        s.draw_panel(g, r)
        pygame.image.save(s.screen, f"{a.prefix}{i}.png")
        if i == max(want):
            break


if __name__ == "__main__":
    main()
