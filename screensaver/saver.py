"""NetHack screensaver v0: the bot's games at a watchable pace, with the bot's and
laya's numbers beside the screen.

v0 replays recorded games (laya_trace traces of the arena: nethack/runs/*/<run>/trace, nethackers/runs/...) --
each a real game the bot played; a live bot + model game comes when the model takes decisions.
Every game-turn screen is a frame; calm stretches (no hostile in view, HP >= 80 %, no new message) fast-forward.
The panel: the hero (Dlvl, XL, AC, HP, turn), the bot (its strategy stack), laya (risk-v2: when will the hero die,
asked of laya_serve.py on the state text the model was trained on -- build_v2.state_text), the milestones and the
BALROG progress of the game, and the session.

  python saver.py [--traces DIR ...] [--laya 127.0.0.1:6000] [--fps 8] [--window]
  (laya: python trace/laya_serve.py ckpt/risk-v2-l20 --listen 127.0.0.1:6000; without it the panel says offline)
Keys: Esc / q quit, space pause, right arrow next game, +/- speed.
"""
import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client
from pathlib import Path

import pygame

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace"))
sys.path.insert(0, str(HERE.parent / "arena-src"))
from build_laya_data import short  # noqa: E402
from build_v2 import QUESTION, state_text, walk  # noqa: E402
from human_build import Q_FATE, bot_state  # noqa: E402
from bot_profile import AC, DEPTH, DNUM, HP, MAXHP, TIME, XL  # noqa: E402
from nethackers.arena.progress import ACHIEVEMENTS  # noqa: E402

TRACES = [str(HERE.parent / "runs" / "*" / "*" / "trace"), str(HERE.parent / "nethackers" / "runs" / "*" / "*" / "trace")]
PANEL_COLS = 44
BG = (8, 8, 12)
FG = (200, 200, 200)
DIM = (90, 90, 100)
COLORS = {"}": (60, 110, 255), "{": (60, 110, 255), "<": (250, 220, 60), ">": (250, 220, 60), "$": (250, 210, 40),
          "!": (220, 80, 220), "?": (235, 235, 235), "/": (60, 210, 230), "=": (60, 210, 230), "[": (60, 210, 230),
          ")": (60, 210, 230), "(": (190, 150, 90), "%": (190, 120, 60), "*": (230, 230, 120), '"': (60, 210, 230),
          "+": (170, 110, 50), "#": (130, 130, 130), ".": (75, 75, 85), "-": (150, 150, 160), "|": (150, 150, 160),
          "`": (150, 150, 150), "0": (150, 150, 150), "^": (230, 60, 200)}
HERO = (255, 255, 120)
HOSTILE = (240, 70, 60)
PET = (80, 220, 90)
PEACEFUL = (110, 160, 255)
BUCKETS = ["<= 2", "3-7", "8-30", "31-100", "> 100"]
FATES = ["dies here / 1 deeper", "2-5 levels deeper", "6+ levels deeper", "ascends"]
ROWS = 31          # screen rows the font is sized for (the panel is the taller part)
MILESTONES = [("Dlvl 10", lambda d, g: d >= 10), ("Medusa zone (Dlvl 21+)", lambda d, g: d >= 21),
              ("castle zone (Dlvl 25+)", lambda d, g: d >= 25), ("Gehennom", lambda d, g: g)]


def games(patterns):
    out = []
    for pat in patterns:
        for d in glob.glob(pat):
            out += sorted(glob.glob(os.path.join(d, "*.jsonl.zst")) + glob.glob(os.path.join(d, "*.jsonl")))
    return out


def result_of(path):
    """the arena's result for a trace <seed>.jsonl[.zst] (evidence.json beside trace/), or {}"""
    ev = Path(path).parent.parent / "evidence.json"
    seed = Path(path).name.split(".")[0]
    try:
        return next((r for r in json.load(open(ev))["results"] if str(r["trajectory_id"]) == seed), {})
    except (OSError, ValueError, KeyError):
        return {}


def read(path):
    if path.endswith(".zst"):
        data = subprocess.run(["zstd", "-dcq", path], capture_output=True, check=True).stdout
    else:
        data = open(path, "rb").read()
    lines = data.splitlines()
    head = json.loads(lines[0])
    return head.get("identity", "unknown"), lines[1:]


def progress(depth, gehennom):
    """the BALROG progress of a game's deepest Dlvl (the arena's table)"""
    return max(ACHIEVEMENTS.get(f"Dlvl:{depth}", 0.0), ACHIEVEMENTS.get("Gehennom", 0.0) if gehennom else 0.0)


class Laya:
    """laya's answers on the current state, asked in a background thread (the frame never waits for the GPU):
    risk-v2 on the bot's state text, and with --human the unified model's human fate on the humans' layout."""

    def __init__(self, addr, name):
        self.addr, self.name, self.conn, self.want, self.status = addr, name, None, None, "offline"
        self.probs, self.fate = None, None
        self.lock = threading.Lock()
        self.asked = 0
        if addr:
            threading.Thread(target=self.loop, daemon=True).start()

    def ask(self, state, human_state=None):
        with self.lock:
            self.want = (state, human_state)

    @staticmethod
    def softmax(z):
        m = max(z)
        e = [math.exp(v - m) for v in z]
        return [v / sum(e) for v in e]

    def loop(self):
        from bc_vocab import address
        while True:
            with self.lock:
                want, self.want = self.want, None
            if want is None:
                time.sleep(0.05)
                continue
            state, human_state = want
            try:
                if self.conn is None:
                    self.conn = Client(address(self.addr), authkey=b"laya")
                self.conn.send([(state, QUESTION)] + ([(human_state, Q_FATE)] if human_state else []))
                z = self.conn.recv()
                self.probs = self.softmax(z[0])
                self.fate = self.softmax(z[1]) if human_state else None
                self.status = self.name
                self.asked += 1
            except (OSError, EOFError, ConnectionError):
                self.conn, self.probs, self.fate, self.status = None, None, None, "offline"
                time.sleep(10)


class Saver:
    def __init__(self, a):
        pygame.init()
        flags = 0 if a.window else pygame.FULLSCREEN
        size = (1600, 900) if a.window else (0, 0)
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption("NetHack: pf_base + laya")
        pygame.mouse.set_visible(bool(a.window))
        w, h = self.screen.get_size()
        cols = 80 + PANEL_COLS
        size = 8
        while True:  # the largest monospace font whose 124 x ROWS cells fit
            f = pygame.font.SysFont("consolas,dejavusansmono,couriernew", size + 1)
            if f.size("M" * cols)[0] > w * 0.97 or f.get_linesize() * ROWS > h * 0.95:
                break
            size += 1
        self.font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", size)
        self.bold = pygame.font.SysFont("consolas,dejavusansmono,couriernew", size, bold=True)
        self.cw, self.ch = self.font.size("M")[0], self.font.get_linesize()
        self.x0 = (w - self.cw * cols) // 2
        self.y0 = (h - self.ch * ROWS) // 2
        self.glyphs = {}
        self.laya = Laya(a.laya, a.name)
        self.human = a.human
        self.fps = a.fps
        self.files = games(a.traces)
        random.shuffle(self.files)
        self.session = dict(games=0, best=0, ends=[])
        self.paused = False

    def glyph(self, ch, color, bold=False):
        key = (ch, color, bold)
        if key not in self.glyphs:
            self.glyphs[key] = (self.bold if bold else self.font).render(ch, True, color)
        return self.glyphs[key]

    def text(self, col, row, s, color=FG, bold=False):
        self.screen.blit((self.bold if bold else self.font).render(s, True, color),
                         (self.x0 + col * self.cw, self.y0 + row * self.ch))

    def draw_screen(self, r):
        b, scr = r["b"], r["scr"]
        hx, hy = b[0], b[1] + 1
        mons = {(hx + dx, hy + dy): kind for _, dx, dy, kind in r.get("mon", [])}
        for y, row in enumerate(scr[:24]):
            for x, ch in enumerate(row[:80]):
                if ch == " ":
                    continue
                if y in (0, 22, 23):
                    color = FG if y else (235, 235, 235)
                elif (x, y) == (hx, hy):
                    color = HERO
                elif (x, y) in mons:
                    color = {"pet": PET, "peaceful": PEACEFUL}.get(mons[(x, y)], HOSTILE)
                elif ch.isalpha() or ch in "@&;:'":
                    color = HOSTILE
                else:
                    color = COLORS.get(ch, FG)
                self.screen.blit(self.glyph(ch, color, (x, y) == (hx, hy)),
                                 (self.x0 + x * self.cw, self.y0 + (y + 1) * self.ch))

    def bar(self, col, row, width, frac, color):
        x, y = self.x0 + col * self.cw, self.y0 + row * self.ch + self.ch // 5
        pygame.draw.rect(self.screen, (40, 40, 50), (x, y, width * self.cw, self.ch * 3 // 5))
        pygame.draw.rect(self.screen, color, (x, y, int(width * self.cw * max(0.0, min(1.0, frac))), self.ch * 3 // 5))

    def draw_panel(self, g, r):
        c = 82
        b = r["b"]
        self.text(c, 0, "NetHack  pf_base bot + laya", HERO, True)
        self.text(c, 1, f"game {self.session['games']}  seed {g['seed']}  {g['run'][:22]}", DIM)
        self.text(c, 3, "HERO", HERO, True)
        self.text(c, 4, f"Dlvl {b[DEPTH]:<3} deepest {g['deepest']:<3} XL {b[XL]:<3} AC {b[AC]}")
        hp = b[HP] / max(b[MAXHP], 1)
        self.text(c, 5, f"HP {b[HP]:>3}/{b[MAXHP]:<3}")
        self.bar(c + 12, 5, 28, hp, (70, 200, 80) if hp > 0.5 else (230, 180, 40) if hp > 0.25 else (230, 60, 50))
        self.text(c, 6, f"turn {b[TIME]}")
        self.text(c, 8, "BOT  strategy stack", HERO, True)
        stack = [short(s) for s in r.get("s", [])][-4:] or ["explore / descend"]
        for i, s in enumerate(stack):
            self.text(c, 9 + i, ("> " if i else "") + s[-40:], FG if i == len(stack) - 1 else DIM)
        self.text(c, 13, "LAYA  when will the hero die?", HERO, True)
        self.text(c, 14, f"{self.laya.status}", DIM)
        p = self.laya.probs
        for i, name in enumerate(BUCKETS):
            self.text(c, 15 + i, f"{name:>6} turns")
            if p:
                self.bar(c + 13, 15 + i, 22, p[i], (230, 70, 60) if i < 2 else (230, 180, 40) if i < 4 else (70, 200, 80))
                self.text(c + 36, 15 + i, f"{p[i]:5.1%}")
        if self.human:
            self.text(c, 21, "HUMANS  where would a human's game end?", HERO, True)
            f = self.laya.fate
            for i, name in enumerate(FATES):
                self.text(c, 22 + i, f"{name[:20]:>20}")
                if f:
                    self.bar(c + 21, 22 + i, 15, f[i], (230, 70, 60) if i == 0 else (230, 180, 40) if i < 2 else
                             (70, 200, 80))
                    self.text(c + 37, 22 + i, f"{f[i]:5.1%}")
        top = 27 if self.human else 21
        self.text(c, top, "MILESTONES", HERO, True)
        for i, (n, f) in enumerate(MILESTONES):
            ok = f(g["deepest"], g["gehennom"])
            self.text(c + (i % 2) * 20, top + 1 + i // 2, ("+ " if ok else "- ") + n.split(" (")[0],
                      (70, 200, 80) if ok else DIM)
        prog = progress(g["deepest"], g["gehennom"])
        self.text(c, top + 3, f"progress {prog:.3f}")
        self.bar(c + 15, top + 3, 25, prog, (110, 160, 255))
        s = self.session
        self.text(c, top + 4, f"session: {s['games']} game{'s' * (s['games'] != 1)}" +
                  (f", best Dlvl {s['best']}, last: {s['ends'][-1][:24]}" if s["ends"] else ""), DIM)

    def end_card(self, g, res):
        cause = res.get("cause_of_death") or res.get("end_status") or "game over"
        self.session["ends"].append(cause)
        self.session["best"] = max(self.session["best"], g["deepest"])
        msg = f"{cause}  on Dlvl {g['last_depth']}  (deepest {g['deepest']}, progress {progress(g['deepest'], g['gehennom']):.3f})"
        w, h = self.screen.get_size()
        box = pygame.Surface((w, self.ch * 3), pygame.SRCALPHA)
        box.fill((0, 0, 0, 200))
        self.screen.blit(box, (0, h // 2 - self.ch))
        t = self.bold.render(msg, True, (240, 90, 80))
        self.screen.blit(t, ((w - t.get_width()) // 2, h // 2))
        pygame.display.flip()
        self.wait(6.0)

    def wait(self, sec):
        end = time.time() + sec
        while time.time() < end:
            if self.events():
                return True
            time.sleep(0.03)
        return False

    def events(self):
        """True: skip to the next game"""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    raise SystemExit
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if e.key == pygame.K_RIGHT:
                    return True
                if e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.fps = min(60, self.fps * 1.5)
                if e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.fps = max(1, self.fps / 1.5)
        return False

    def play(self, path):
        identity, lines = read(path)
        res = result_of(path)
        g = dict(seed=Path(path).name.split(".")[0], run=Path(path).parent.parent.name, deepest=0, gehennom=False,
                 last_depth=0)
        self.session["games"] += 1
        last_msg, frame_t = None, 0.0
        for _, r, back, msgs, inv in walk(lines):
            b = r["b"]
            g["last_depth"] = b[DEPTH]
            calm = (b[HP] >= 0.8 * b[MAXHP] and not any(m[3] not in ("pet", "peaceful") for m in r.get("mon", []))
                    and (msgs[-1] if msgs else None) == last_msg)
            last_msg = msgs[-1] if msgs else None
            if calm and frame_t > 0 and time.time() - frame_t < 1.0:
                continue  # fast-forward: a calm screen is shown at most once a second
            g["deepest"] = max(g["deepest"], b[DEPTH] if b[DNUM] != 2 else 0)
            g["gehennom"] = g["gehennom"] or b[DNUM] == 1
            self.laya.ask(state_text(identity, r["s"], r, back, list(msgs), inv),
                          bot_state(identity, r, inv, g["deepest"]) if self.human else None)
            self.screen.fill(BG)
            self.draw_screen(r)
            self.draw_panel(g, r)
            pygame.display.flip()
            frame_t = time.time()
            if self.wait(1.0 / self.fps):
                return
            while self.paused:
                if self.wait(0.1):
                    return
        self.end_card(g, res)

    def run(self):
        while True:
            for path in self.files:
                self.play(path)
            random.shuffle(self.files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", default=TRACES, help="trace dirs (globs)")
    ap.add_argument("--laya", default="127.0.0.1:6000", help="laya_serve.py address ('' = no model)")
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--human", action="store_true", help="also the human fate scale (a unified nh-v* model)")
    ap.add_argument("--name", default="risk-v2-l20", help="the served model's name on the panel")
    a = ap.parse_args()
    s = Saver(a)
    if not s.files:
        raise SystemExit(f"no traces in {a.traces}")
    s.run()


if __name__ == "__main__":
    main()
