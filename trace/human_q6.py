"""The humans' questions, v6: explicit, finer and with the cause of the end, replacing fate / ttd (v5),
which were three or four classes with hand-picked bounds (fate 1.6 bits a row) and no cause.
  ttd     -- how many more turns until the game ends (death or ascension): K bins of equal mass in log(turns left)
             on the train rows, 1000 and 5000 forced in as bounds (so the answer folds into v5's ttd exactly)
  depth   -- the deepest Dlvl the game reaches: bins of the levels it adds to the deepest so far (equal mass,
             1 and 5 forced in: folds into v5's fate), written out per state as absolute Dlvl ("Dlvl 17-20")
  end     -- what ends the game: the commonest monster classes (letter, NetHack's own class name, an example),
             another monster, a wand / ray / spell, poison, petrification, starvation / choking, drowning, a trap,
             something else, ascension
  ascends -- yes / no
ttd and depth targets are soft: a Gaussian over the bins around the true value, sigma 0.75 of the true bin's width
(HL-Gauss, 2403.03950; equal-mass bins and smoothing scaled by bin width: 2603.07448, 2606.04564).
Bin bounds and categories come from the train rows only and are written to meta.json.
"""
import collections
import math
import re

import numpy as np

INS = "A human plays this game. "
FIXED_ENDS = {  # key: text, in the order the options are shown
    "ray": "a wand, ray or spell", "poison": "poison", "stone": "turned to stone",
    "food": "starvation or choking", "water": "drowning", "trap": "a trap or a fall",
    "monster": "another monster", "other": "something else", "ascend": "ascension"}
TRAP = re.compile(r"\b(fall|fell|land mine|trap|rolling boulder|falling rock|pit|spikes)\b")
RAY = re.compile(r"\b(wand|bolt of|death ray|magic missile|blast of|ray of|exploding|psi bolt|spell|touch of death)\b")
# killers that are no monster name as written (end_unmapped.py: 5.5 % of rows fell to "another monster")
GODS = r"(Tyr|Odin|Loki|Moloch)"
NOT_MONSTER = re.compile(rf"^(wrath of {GODS}|acidic|boiling potion|contaminated potion|potion of acid|burning scroll|"
                         r"scroll of genocide|scroll of earth|genocidal confusion|strangulation|touching Excalibur|"
                         r"gas cloud)$")
TRAP_MISSILE = re.compile(r"^(arrow|dart)$")  # an arrow / dart trap; a monster's missile has its kind ("orcish arrow")
_MON = None


def monsters():
    """name -> (class letter, NetHack's class name) from nle's monster table (3.6; the 3.4.3 names are the same)."""
    global _MON
    if _MON is None:
        from nle import nethack
        _MON = {}
        for i in range(nethack.NUMMONS):
            p = nethack.permonst(i)
            c = nethack.class_sym.from_mlet(p.mlet)
            _MON.setdefault(p.mname.lower(), (c.sym, c.explain))
    return _MON


def norm(death):
    d = death.split(", while ")[0]
    d = re.sub(r" (called|named) .*", "", d)
    d = re.sub(r"\b(an?|the) ", "", d)
    return re.sub(r"\b(invisible|hallucinogen-distorted) ", "", d).strip()


def end_kind(death):
    """xlogfile death -> a class letter ('a', '@', ...) or one of FIXED_ENDS' keys (before grouping)."""
    d = norm(death)
    if d == "ascended":
        return "ascend"
    if d.startswith(("petrified", "turned to stone")):
        return "stone"
    if d.startswith("poisoned"):
        return "poison"
    if d.startswith(("died of starvation", "choked")):
        return "food"
    if d.startswith("drowned"):
        return "water"
    if TRAP.search(d):
        return "trap"
    if RAY.search(d):
        return "ray"
    m = re.match(r"killed by (.+?)(?:'s explosion| corpse)?$", d)
    if m:
        name = killer_name(m.group(1))
        if name == "poisonous":  # a poisonous corpse
            return "poison"
        if NOT_MONSTER.match(name):
            return "other"
        if TRAP_MISSILE.match(name):
            return "trap"
        hit = monsters().get(name.lower())
        return hit[0] if hit else "monster"
    return "other"


def killer_name(name):
    """the monster behind xlogfile's wording: 'priestess of Odin' -> aligned priest, 'Mr. Izchak, shopkeeper' ->
    shopkeeper, 'couatl of Tyr' -> couatl, 'chameleon imitating minotaur' -> chameleon"""
    if name == "brainlessness":
        return "mind flayer"
    if re.match(rf"^(high )?priest(ess)? of {GODS}$", name):
        return "aligned priest"
    if name.endswith(", shopkeeper"):
        return "shopkeeper"
    if name.startswith("wrath of "):
        return name
    name = re.sub(rf" of {GODS}$| \(with Amulet\)$| in bat form$", "", name)
    return re.sub(r" imitating .*", "", name)


def nice(x):
    """two significant digits: 4873 -> 4900, 312 -> 310, 57 -> 57"""
    if x < 100:
        return int(round(x))
    p = 10 ** (int(math.log10(x)) - 1)
    return int(round(x / p) * p)


def upper_bounds(values, k, forced, transform=None, rounder=round):
    """bounds b_0 < b_1 < ... (bin i: b_{i-1} < v <= b_i, the last bin open) of ~k equal-mass bins, plus `forced`."""
    v = np.asarray(values, dtype=float)
    q = np.quantile(transform(v) if transform else v, np.arange(1, k) / k)
    if transform is np.log1p:
        q = np.expm1(q)
    # a quantile bound within 15 % of a forced one would leave a sliver of a bin (950 next to 1000)
    near = lambda x: any(f > 0 and x > 0 and abs(math.log(x / f)) < 0.15 for f in forced)
    return sorted(x for x in {rounder(x) for x in q if not near(rounder(x))} | set(forced) if x >= 0)


def soft(lo_edges, hi_edges, x, sigma):
    """mass of N(x, sigma) on each bin [lo, hi) (outer bins open)"""
    cdf = lambda z: 0.5 * (1 + math.erf(z / (sigma * math.sqrt(2))))
    p = [cdf(hi - x) - cdf(lo - x) for lo, hi in zip(lo_edges, hi_edges)]
    s = sum(p)
    return [v / s for v in p]


def gold_soft(p, label):
    return {"probabilities": {str(i): float(v) for i, v in enumerate(p)}, "label": str(label)}


class Builder:
    def __init__(self, train_outcomes, k_ttd=16, k_depth=12, n_classes=12, terse=False):
        self.terse = terse
        lefts = [o["left"] for o in train_outcomes if o["left"] is not None]
        # forced: 1000 / 5000 turns and 1 / 5 levels fold into v5's ttd / fate; the tails (ascensions take 40-60k
        # turns and go 30-50 levels down) get bounds of their own, equal mass would lump them into one bin
        self.ttd = upper_bounds(lefts, k_ttd, {1000, 5000, 20000, 40000}, np.log1p, nice)
        self.depth = upper_bounds([o["gain"] for o in train_outcomes], k_depth, {0, 1, 5, 14, 20, 30},
                                  rounder=lambda x: int(x))
        kinds = collections.Counter(o["end"] for o in train_outcomes)
        letters = [k for k, _ in kinds.most_common() if k not in FIXED_ENDS][:n_classes]
        self.classes = letters
        ex = collections.Counter()  # the commonest monster of each kept class, for its text
        for o in train_outcomes:
            if o["end"] in letters:
                ex[(o["end"], o["killer"])] += 1
        mon = monsters()
        explain = {sym: e for sym, e in mon.values()}
        best = {}
        for (c, name), n in ex.most_common():
            best.setdefault(c, name)
        self.end = {c: f"{explain.get(c, 'monster')}, e.g. {best.get(c, '?')}" for c in letters}
        self.end.update(FIXED_ENDS)
        self.shares = {k: kinds[k] / len(train_outcomes) for k in kinds}

    @classmethod
    def from_meta(cls, m, train_outcomes, terse=False):
        """another set's bins and end classes (so that the same question means the same
        options in both); only the shares are of this set."""
        b = cls.__new__(cls)
        b.terse = terse
        b.ttd, b.depth, b.end = list(m["ttd_bounds"]), list(m["depth_gain_bounds"]), dict(m["end"])
        b.classes = [k for k in b.end if k not in FIXED_ENDS]
        kinds = collections.Counter(o["end"] for o in train_outcomes)
        b.shares = {k: kinds[k] / len(train_outcomes) for k in kinds}
        return b

    def meta(self):
        return {"ttd_bounds": self.ttd, "depth_gain_bounds": self.depth, "end": self.end,
                "end_train_share": {k: round(v, 4) for k, v in sorted(self.shares.items(), key=lambda t: -t[1])}}

    def end_key(self, kind):
        return kind if kind in self.end else "monster"

    def questions(self, o):
        """terse (human_build.py --terse): the same questions and bins in fewer tokens
        (ttd 140 -> ~115, depth 91 -> ~55, ascends 25 -> 14; end's classes unchanged -- tok_alt.py)."""
        qs, gd = {}, {}
        ins = "A human plays. " if self.terse else INS
        if o["left"] is not None:  # ttd: bins in log(1 + turns left)
            b = self.ttd
            u = "" if self.terse else " turns"  # terse: the question says "turns" once
            texts = ([f"at most {b[0]}{u}"] + [f"{b[i - 1] + 1}-{b[i]}{u}" for i in range(1, len(b))]
                     + [f"over {b[-1]}" if self.terse else f"more than {b[-1]} turns"])
            lo = [-math.inf] + [math.log1p(x) for x in b]
            hi = [math.log1p(x) for x in b] + [math.inf]
            k = next((i for i, x in enumerate(b) if o["left"] <= x), len(b))
            w = (hi[k] - lo[k]) if 0 < k < len(b) else (hi[1] - lo[1] if k == 0 else hi[-2] - lo[-2])
            qs["ttd"] = {"type": "score", "criteria": texts,
                         "instructions": ins + ("Turns until the game ends?" if self.terse else
                                                "How many more turns until the game ends (the hero dies or ascends)?")}
            gd["ttd"] = gold_soft(soft(lo, hi, math.log1p(o["left"]), 0.75 * w), k)
        b, d = self.depth, o["deepest"]  # depth: bins of added levels, written as absolute Dlvl
        ranges = [(0 if i == 0 else b[i - 1] + 1, b[i]) for i in range(len(b))]
        texts = []
        dl = "" if self.terse else "Dlvl "  # terse: the question says "Dlvl" once
        for lo_g, hi_g in ranges:
            if hi_g == 0:
                texts.append("no deeper" if self.terse else f"Dlvl {d}, no deeper")
            elif lo_g == hi_g:
                texts.append(f"{dl}{d + lo_g}")
            else:
                texts.append(f"{dl}{d + lo_g}-{d + hi_g}")
        texts.append(f"{d + b[-1] + 1}+" if self.terse else f"Dlvl {d + b[-1] + 1} or deeper")
        lo = [-math.inf] + [x + 0.5 for x in b]
        hi = [x + 0.5 for x in b] + [math.inf]
        k = next((i for i, x in enumerate(b) if o["gain"] <= x), len(b))
        w = (hi[k] - lo[k]) if 0 < k < len(b) else 1.0
        qs["depth"] = {"type": "score", "criteria": texts,
                       "instructions": ins + ("Deepest Dlvl this game will reach?" if self.terse else
                                              "What is the deepest dungeon level this game will reach?")}
        gd["depth"] = gold_soft(soft(lo, hi, o["gain"], 0.75 * w), k)
        key = self.end_key(o["end"])
        qs["end"] = {"type": "choice", "criteria": dict(self.end), "instructions": ins + "What will end this game?"}
        gd["end"] = {"probabilities": {c: float(c == key) for c in self.end}, "label": key}
        yes_no = {"false": "no", "true": "yes"} if self.terse else {"false": "no, the hero dies",
                                                                   "true": "yes, the hero ascends"}
        qs["ascends"] = {"type": "noul", "criteria": yes_no, "instructions": ins + "Will the hero ascend?"}
        gd["ascends"] = {"probabilities": {"false": float(not o["asc"]), "true": float(o["asc"])},
                         "label": "true" if o["asc"] else "false"}
        return qs, gd


def outcome(g, s):
    """what v6 needs from a game and one of its states"""
    left = g["turns"] - s["turn"] if s["turn"] >= 0 else None
    d = norm(g["death"])
    m = re.match(r"killed by (.+?)(?:'s explosion| corpse)?$", d)
    return {"left": max(0, left) if left is not None else None, "gain": max(0, g["maxlvl"] - s["deepest"]),
            "deepest": s["deepest"], "asc": g["death"] == "ascended", "end": end_kind(g["death"]),
            "killer": killer_name(m.group(1)) if m else d}
