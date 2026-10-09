"""laya plays NetHack by itself (behaviour cloning v1, bc_vocab.py, docs/behaviour-cloning.md): every decision is one of laya's
typed questions, answered by laya_serve.py; a thin interface layer presses only what carries no decision.

laya decides: at the top level the command (bc_vocab.COMMANDS; probabilities times UNDO = 1 / KEEP, which undoes
the training resampling; commands that just changed nothing are masked), for "item" the item command, at "What do you want to <verb>? [...]" the item, at
"In what direction?" the direction, at "...; eat it?" yes or no. The argument questions read the state of the
command's decision, as in training.
Interface (fixed): --More-- (space); "Are you sure you want to pray?" y; "add to the current engraving?" n;
"Which ring-finger" r; Elbereth's write-with (fingers) and text; a wish (WISH); pickup menus: everything but JUNK;
any other prompt or menu: Esc.

The arena's environment, seeds and character (as forkrun.py). Per game out/results/<seed>.json, written when it
ends (a rerun skips those seeds), and out/trace/<seed>.jsonl: laya's decisions (step, turn, question, answer,
its probability).
  python laya_play.py --server ADDRESS --out DIR --seeds 200000:200010 [--parallel 8] [--greedy]
(a container of the laya-frozen image: NLE and the arena; the model runs in laya_serve.py; ADDRESS:
bc_vocab.address -- a Unix socket in a shared directory, or host.docker.internal:6000 from a container with a
network)
"""
import argparse
import collections
import json
import multiprocessing
import os
import re
import sys
import time
from bisect import bisect_right
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
from nethackers.arena.environment import make_environment
from nethackers.arena.seeds import trajectory_spec
from nethackers.harness.seeds import validation_spec

import bc_vocab as V

sys.path.insert(0, str(Path(__file__).parent / "patched"))
from laya_trace import _monsters  # noqa: E402  (the "mon" records the training states were built from)

IDENTITY = "val-dwa-law-fem"
INDEX = {}
for _i, _c in enumerate(V.ACTION_KEYS):
    INDEX.setdefault(_c, _i)
KEY = {**V.DIR_OPTS, "search": "s", "pickup": ",", "throw": "t", "apply": "a", "eat": "e", "elbereth": "E",
       "pray": chr(0x80 | ord("p")), "down": ">", "up": "<", "kick": chr(4)}
ITEM_KEY = {v: k for k, v in V.ITEM_CMDS.items() if k != "D"}
WISH = "blessed +2 gray dragon scale mail"
JUNK = re.compile(r"boulder|statue|large box|chest|ice box|corpse|iron ball|iron chain|rock|gray stone|"
                  r"heavy iron|anvil|figurine", re.I)
MENU_END = re.compile(r"\((end|\d+ of \d+)\)\s*$")
MENU_ITEM = re.compile(r"^([a-zA-Z]) - (.+)$")
ESC, ENTER, SPACE = 27, 13, 32


class Game:
    def __init__(self, conn, rng, greedy, trace):
        self.conn, self.rng, self.greedy, self.trace = conn, rng, greedy, trace
        self.msgs, self.last = collections.deque(maxlen=6), collections.deque(maxlen=6)
        self.hp_turns, self.hp_vals = [], []
        self.ctx, self.queue = None, []
        self.counts = collections.Counter()
        self.step = self.turn = 0
        self.msg, self.prev_b, self.blocked = "", None, set()

    def ask(self, name, state, q, prior=None):
        """laya's answer to one question: sampled from its distribution (argmax with --greedy)."""
        self.conn.send([(state, q)])
        z = np.array(self.conn.recv()[0], dtype=np.float64)
        opts = [False, True] if q["type"] == "noul" else list(q["criteria"])
        p = np.exp(z - z.max())
        if prior:
            masked = p * [prior[o] for o in opts]
            p = masked if masked.sum() > 0 else p  # everything blocked: let it choose freely
        p /= p.sum()
        i = int(p.argmax()) if self.greedy else int(self.rng.choice(len(p), p=p))
        self.trace.write(json.dumps({"s": self.step, "t": self.turn, "q": name, "a": opts[i], "p": round(float(p[i]), 3),
                                     "m": self.msg}) + "\n")
        return opts[i]

    def key(self, obs):
        """The next key for this observation."""
        msg = self.msg = bytes(obs["message"]).split(b"\0")[0].decode("latin-1").strip()
        if V.message_ok(msg) and (not self.msgs or self.msgs[-1] != msg[:160]):
            self.msgs.append(msg[:160])
        tty = [bytes(r).decode("latin-1").rstrip() for r in obs["tty_chars"]]
        yn, getlin, more = (int(v) for v in obs["misc"])
        if self.queue:
            return self.queue.pop(0)
        if more or any("--More--" in r for r in tty):
            return SPACE
        cmd = self.ctx["cmd"] if self.ctx else None
        menu = next((r for r in tty if MENU_END.search(r)), None)
        if menu is not None and not yn and not getlin:
            if cmd == "pickup":
                x0 = len(menu) - len(menu.lstrip())
                pick = [m.group(1) for r in tty if (m := MENU_ITEM.match(r[x0:].strip())) and not JUNK.search(m.group(2))]
                self.queue = pick + [ENTER]
                self.ctx = dict(self.ctx, cmd="pickup-menu")
                return self.queue.pop(0)
            return ESC
        if getlin:
            p = msg.lower()
            text = "Elbereth" if "write in" in p or "engrave" in p else WISH if "wish" in p else None
            if text is None:
                return ESC
            self.queue = list(text) + [ENTER]
            return self.queue.pop(0)
        if yn:
            return self.answer(msg, obs)
        # the top level: a new command
        prev, self.ctx = self.ctx, None
        b = [int(v) for v in obs["blstats"]]
        # the last command changed nothing ("You don't have anything to eat.", a wall, Esc on its prompt): not part
        # of the hero's history (the training data only has commands that took effect) and masked until the state
        # changes -- else the same state brings the same choice again, in a loop
        if prev is not None and b == self.prev_b:
            self.blocked.add(prev["cmd"])
            if self.last and self.last[-1] == prev["cmd"]:
                self.last.pop()
        elif b != self.prev_b:
            self.blocked = set()
        self.prev_b = b
        self.turn = b[V.TIME]
        if not self.hp_turns or self.hp_turns[-1] != b[V.TIME]:
            self.hp_turns.append(b[V.TIME])
            self.hp_vals.append((b[V.HP], b[V.MAXHP]))
        back = []
        for d in (10, 30):
            i = bisect_right(self.hp_turns, b[V.TIME] - d)
            back.append(self.hp_vals[i - 1] if i else None)
        inv = [f"{chr(int(c))} - {s}" for c, s in zip(obs["inv_letters"], (
            bytes(r).split(b"\0")[0].decode("latin-1").strip() for r in obs["inv_strs"])) if s]
        state = V.state_text(IDENTITY, b, tty, _monsters(obs["glyphs"], b[0], b[1]), back, list(self.msgs), inv,
                             list(self.last))
        cmd = self.ask("command", state, V.Q_COMMAND, {c: 0.0 if c in self.blocked else u for c, u in V.UNDO.items()})
        self.counts[cmd] += 1
        self.last.append(cmd)
        self.ctx = {"cmd": cmd, "state": state, "inv": inv, "item": None}
        if cmd == "search5":
            self.queue = ["s"]
            return "5"
        if cmd == "item":
            sub = self.ask("item_cmd", state, V.Q_ITEM_CMD)
            self.ctx["sub"] = sub
            return ITEM_KEY[sub]
        return KEY[cmd]

    def answer(self, prompt, obs):
        """A yes/no, item or direction prompt (NetHack asks all three through yn_function)."""
        p = prompt.lower()
        if "stop eating?" in p:  # "having a hard time getting all of it down": Esc means go on, and choke
            return "y"
        ctx = self.ctx
        if ctx is None:
            return ESC
        if "pray?" in p:
            return "y"
        if "add to the current engraving" in p:
            return "n"
        if "ring-finger" in p:
            return "r"
        if "eat it?" in p:
            return "y" if self.ask("eat_here", ctx["state"], V.q_eat_here(prompt)) else "n"
        if p.startswith("what do you want to") and "[" in p:
            if ctx["cmd"] == "elbereth":
                return "-"
            keys = [k for k in V.allowed(prompt) if k not in "?*"]
            if not keys:
                return ESC
            q = V.q_item(prompt, keys, ctx["inv"])
            k = self.ask("item", ctx["state"], q)
            ctx["item"] = q["criteria"].get(k) or "it"
            return k
        if "direction" in p:
            verb = V.VERB.get(ctx.get("sub", ctx["cmd"]), "acts")
            verb += f" {ctx['item']}" if ctx["item"] and verb != "kicks" else ""
            return V.DIR_OPTS[self.ask("direction", ctx["state"], V.q_dir(verb, V.allowed(prompt) or None))]
        return ESC


def play(seed, a):
    out = Path(a.out)
    spec = trajectory_spec("public", "local", seed)
    vs = validation_spec(IDENTITY, n=1, start=seed)
    char = vs.batch[0][1]
    env = make_environment(vs.max_steps, vs.no_progress_timeout, None if char == "-" else char)
    t0 = time.monotonic()
    with Client(V.address(a.server), authkey=b"laya") as conn, open(out / "trace" / f"{seed}.jsonl", "w") as tr:
        g = Game(conn, np.random.default_rng(seed), a.greedy, tr)
        obs = env.reset(spec)
        done = False
        while not done and g.step < (a.max_steps or vs.max_steps):
            k = g.key(obs)
            obs, _r, term, trunc = env.step(INDEX[ord(k) if isinstance(k, str) else k])
            g.step += 1
            done = term or trunc
    m = env.metrics()
    res = {"seed": seed, "steps": g.step, "turns": m.turns, "max_depth": m.max_depth, "end_status": m.end_status,
           "cause": m.cause_of_death, "progress": m.progress, "milestone": m.milestone,
           "decisions": sum(g.counts.values()), "commands": dict(g.counts.most_common()),
           "seconds": round(time.monotonic() - t0)}
    tmp = out / "results" / f"{seed}.json.tmp"
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out / "results" / f"{seed}.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="host.docker.internal:6000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", required=True, help="A:B, seeds A..B-1")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling laya's distribution")
    ap.add_argument("--max-steps", type=int, default=0, help="end a game after this many steps (probes)")
    a = ap.parse_args()
    for d in ("results", "trace"):
        Path(a.out, d).mkdir(parents=True, exist_ok=True)
    lo, hi = map(int, a.seeds.split(":"))
    seeds = [s for s in range(lo, hi) if not Path(a.out, "results", f"{s}.json").exists()]
    with multiprocessing.get_context("fork").Pool(a.parallel, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_play, [(s, a) for s in seeds]):
            print(f"{r['seed']}: turns {r['turns']}, depth {r['max_depth']}, {r['end_status']} ({r['cause']}), "
                  f"{r['decisions']} decisions, {r['seconds']} s", flush=True)


def _play(job):
    return play(*job)


if __name__ == "__main__":
    main()
