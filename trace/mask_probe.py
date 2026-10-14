"""Masking probe R1 for the recognition probe (fam_probe.py): does c17a know a game by its per-game strings? Every row
of the four groups gets (1) a random permutation of the unidentified appearances within their class -- potions,
scrolls, rings, wands, amulets, spellbooks and the four randomized armor groups -- one permutation per class for the
whole row (inventory and messages alike), seeded by crc32 of the row's id; the fixed appearances (clear potion,
unlabeled scroll, plain / papyrus / paperback spellbook) stay; (2) every "called X" / "named X" (artifact names
excepted) replaced by a name of another game of the four groups (the item names and the message names drawn from
their own pools), the same X by the same name within the row, with the bare occurrences of a capitalized message name
(a pet: "You swap places with Rex."). Then predict on the masked groups and fam_probe.py robust over them.
  python mask_probe.py build human_fam23 --out human_fam23m [--report ../results/mask_build23.txt]
  python mask_probe.py uniq human_fam23 --train c17a=human_dwa21/train.jsonl c17r=human_dwa20/train.jsonl \
      [--out ../results/mask_uniq23.txt]
  python mask_probe.py fp human_fam23 --train human_dwa21/train.jsonl [--drop-top 1] [--out ...]
  python mask_probe.py parts human_fam23 --train human_dwa21/train.jsonl --states nao/val_dwa_states3.jsonl \
      [--drop-top 1] [--out ...]
  python mask_probe.py ident human_fam23 --out human_fam23i [--n 300]   # rows as they are, for the pipeline check
  python mask_probe.py same human_fam23i human_fam23 c17a c17r          # their logits against the existing preds
"""
import argparse
import json
import random
import re
import zlib
from collections import Counter
from pathlib import Path

from item_names import ARTIFACTS, core, known

GROUPS = ("fam13", "new13", "fam12", "new12")
NEW = ("new13", "new12")
OBJ = json.loads((Path(__file__).parent / "nh_names.json").read_text())["objects"]
DESCR = {o["name"]: o["descr"] for o in OBJ if o["name"] and o["descr"]}
ARMOR = {"helmet": ["helmet", "helm of brilliance", "helm of opposite alignment", "helm of telepathy"],
         "gloves": ["leather gloves", "gauntlets of fumbling", "gauntlets of power", "gauntlets of dexterity"],
         "cloak": ["cloak of protection", "cloak of invisibility", "cloak of magic resistance", "cloak of displacement"],
         "boots": ["speed boots", "water walking boots", "jumping boots", "elven boots", "kicking boots",
                   "fumble boots", "levitation boots"]}
FIXED = {"clear", "unlabeled", "plain", "papyrus", "paperback", "Amulet of Yendor"}
ART = {x.lower() for x in ARTIFACTS}


def _alt(xs):
    return "|".join(re.escape(x) for x in sorted(xs, key=len, reverse=True))


def _classes():
    out = {}
    for cls in ("potion", "ring", "wand", "amulet", "spellbook"):
        ds = sorted({o["descr"] for o in OBJ if o["cls"] == cls and o["descr"] and o["descr"] not in FIXED})
        out[cls] = (re.compile(rf"(?<![\w-])({_alt(ds)})(?= {cls}s?\b)"), ds)
    ds = sorted({o["descr"] for o in OBJ if o["cls"] == "scroll" and o["descr"] and o["descr"] not in FIXED})
    out["scroll"] = (re.compile(rf"(?<=labeled )({_alt(ds)})(?![\w])"), ds)
    for g, names in ARMOR.items():
        ds = sorted(DESCR[n] for n in names)
        out[g] = (re.compile(rf"(?<![\w-])({_alt(ds)})(?![\w-])"), ds)
    return out


CLASSES = _classes()
# a / an before an appearance follows the word now standing there; otherwise "an" would tell that the original one
# began with a vowel. objnam.c just_an(): "a" before these despite the vowel.
ARTICLE = re.compile(rf"(?<![\w-])([Aa])n? (?=({_alt({d for _, ds in CLASSES.values() for d in ds})})(?![\w-]))")
A_EXCEPT = ("one", "eu", "uke", "unicorn", "uranium", "useful")


def article(m):
    w = m.group(2).lower()
    return m.group(1) + ("n " if w[0] in "aeiou" and not w.startswith(A_EXCEPT) else " ")


NAME =re.compile(r"(?:\b(\w+) )?\b(called|named) (.+?)(?= \(|;| \||\n|$|[.!?,](?= |\n|$))")
PET = [re.compile(p) for p in (r"You swap places with (?!your )([A-Z][\w' -]*?)[.!]",
                                r"You stop\. (?!Your )([A-Z][\w' -]*?) is in your way",
                                r"You displace (?!your )([A-Z][\w' -]*?)[.!]")]


def game(i):
    return i.split("#")[0]


def appearances(text):
    return {(c, m.group(1)) for c, (rx, _) in CLASSES.items() for m in rx.finditer(text)}


def names(state):
    """(pool, X, key) of every called / named X but the artifacts; pool "inv" -- the inventory line, "msg" -- the rest;
    key -- "called potion", "named dagger": a replacement comes from names of the same kind."""
    out = []
    for line in state.split("\n"):
        pool = "inv" if line.startswith("Inventory:") else "msg"
        out += [(pool, m.group(3), f"{m.group(2)} {re.sub(r's$', '', m.group(1) or '')}")
                for m in NAME.finditer(line) if m.group(3).lower() not in ART]
    return out


def pet_names(state):
    """Capitalized names a pet goes by in the messages: "tame kitten called Rex", "You swap places with Fido."."""
    ns = {x for pool, x, _ in names(state) if pool == "msg" and x[:1].isupper()}
    ns |= {m.group(1) for rx in PET for m in rx.finditer(state)}
    return {x for x in ns if len(x) >= 3 and x.lower() not in ART and x not in ("You", "The", "Your", "Its")}


def mask(r, pools, identity=False):
    """The masked state, the number of appearances permuted and the names replaced (old -> new). pools["pets"][game]:
    the pet names of the game seen in any row of the four groups -- a row that only says "Fido bites the newt" gets them
    replaced too."""
    s, seed = r["state"], zlib.crc32(r["id"].encode())
    rng = random.Random(seed)
    napp = 0
    for c, (rx, ds) in CLASSES.items():
        p = ds[:]
        if not identity:
            random.Random(f"{seed}:{c}").shuffle(p)
        mp = dict(zip(ds, p))
        s, n = rx.subn(lambda m: mp[m.group(1)], s)
        napp += n
    s = ARTICLE.sub(article, s)
    rep, bare = {}, []

    def draw(cands, x):
        mine = game(r["id"])
        if not any(g != mine and y != x for g, y in cands):
            cands = pools["inv"]
        for _ in range(1000):
            g, y = rng.choice(cands)
            if g != mine and y != x:
                break
        return x if identity else y

    pets = pools["pets"].get(game(r["id"]), set())
    for _, x, key in names(s):
        if x not in rep:
            rep[x] = draw(pools["msg"] if x in pets else pools["key"].get(key, pools["inv"]), x)
    for x in sorted(pets):
        if re.search(rf"(?<![\w]){re.escape(x)}(?![\w])", s):
            rep.setdefault(x, draw(pools["msg"], x))
            bare.append(x)
    if rep:
        s = NAME.sub(lambda m: (f"{m.group(1)} " if m.group(1) else "") + f"{m.group(2)} "
                     f"{rep.get(m.group(3), m.group(3))}", s)
        for x in bare:  # a pet's name outside "called" (its moves in the messages)
            s = re.sub(rf"(?<![\w]){re.escape(x)}(?![\w])", lambda m: rep[x], s)
    return s, napp, rep


def load(d):
    return {g: [json.loads(x) for x in open(Path(d, f"{g}.jsonl"), encoding="utf-8")] for g in GROUPS}


def make_pools(rows):
    """inv: every called / named X that is not a pet's name; key: the same by kind ("called wand"); msg: pet names.
    Replacements come only from the new games, which no model has seen: a name from a seen game would plant another
    train game's fingerprint. pets (which names of a game are its pets) is looked up in all four groups."""
    pools = {"inv": [], "key": {}, "msg": [], "pets": {}}
    for rs in rows.values():
        for r in rs:
            pools["pets"].setdefault(game(r["id"]), set()).update(pet_names(r["state"]))
    new = set()
    for grp in NEW:
        for r in rows[grp]:
            g = game(r["id"])
            new.add(g)
            for _, x, key in names(r["state"]):
                if x not in pools["pets"][g]:
                    pools["inv"].append((g, x))
                    pools["key"].setdefault(key, []).append((g, x))
    pools["msg"] = [(g, x) for g, ns in pools["pets"].items() if g in new for x in sorted(ns)]
    return pools


def inv_items(state):
    line = next((x for x in state.split("\n") if x.startswith("Inventory:")), "")
    return [re.sub(r"^[\w$#] - ", "", it) for it in re.split(r"; (?=[\w$#] - )", line[len("Inventory:"):].strip()) if it]


def build(a):
    rows = load(a.data)
    pools = make_pools(rows)
    out = [f"{a.data} -> {a.out}: appearances permuted within class per row, called / named replaced from the pools "
           f"(item names {len(pools['inv'])}, {len({x for _, x in pools['inv']})} distinct; message names "
           f"{len(pools['msg'])}, {len({x for _, x in pools['msg']})} distinct)"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    unknown, pets, labels = Counter(), Counter(), Counter()
    for g in GROUPS:
        st = Counter()
        lines = []
        for r in rows[g]:
            same, _, _ = mask(r, pools, identity=True)
            assert same == r["state"], r["id"]  # the replacing machinery itself changes nothing
            s, napp, rep = mask(r, pools)
            st["rows"] += 1
            st["app"] += napp > 0
            st["napp"] += len(appearances(r["state"]))
            st["name"] += bool(rep)
            st["changed"] += s != r["state"]
            items = inv_items(r["state"])
            bad = [core(it) for it in items if not known(it)]
            st["unknown"] += bool(bad)
            unknown.update(bad)
            for rx in PET:
                for m in rx.finditer(r["state"]):
                    if m.group(1) not in rep:
                        st["pet_left"] += 1
                        pets[m.group(1)] += 1
            for m in re.finditer(r"labeled ([A-Z][A-Z0-9 ]+?)(?=[^A-Z0-9 ]|$)", s):
                if not CLASSES["scroll"][0].match(s, m.start(1)):
                    labels[m.group(1)] += 1
            lines.append(json.dumps({**r, "state": s}, ensure_ascii=False) + "\n")
        Path(a.out, f"{g}.jsonl").write_text("".join(lines), encoding="utf-8")
        n = st["rows"]
        out.append(f"  {g}: {n} rows; changed {st['changed'] / n:.3f}; with a permuted appearance {st['app'] / n:.3f} "
                   f"(appearances per row {st['napp'] / n:.1f}); with a replaced name {st['name'] / n:.3f}; "
                   f"with an item item_names does not know {st['unknown'] / n:.3f}; pet-like names left {st['pet_left']}")
    out.append("\n== not caught: inventory items item_names does not know (player's fruit and the like), top 40")
    out += [f"  {c:6d}  {x}" for x, c in unknown.most_common(40)]
    out.append("\n== not caught: bare capitalized names in pet messages after the mask, top 20")
    out += [f"  {c:6d}  {x}" for x, c in pets.most_common(20)]
    out.append("\n== not caught: scroll labels not in the table, top 20")
    out += [f"  {c:6d}  {x}" for x, c in labels.most_common(20)]
    text = "\n".join(out) + "\n"
    print(text)
    if a.report:
        Path(a.report).write_text(text, encoding="utf-8")


def uniq(a):
    """Before any model: in how many rows of a group is there a set of >= 2 appearances that also stands in the 13+
    rows (or any rows) of the same game in a model's train and in no row of another game there. For each own train row
    R the witness is the row's set cut to R; a subset unique to the game exists iff one such cut is."""
    rows = load(a.data)
    tok = {}

    def ids(text):
        return sorted(tok.setdefault(t, len(tok)) for t in appearances(text))

    sets = {g: [(game(r["id"]), ids(r["state"])) for r in rs] for g, rs in rows.items()}
    out = [f"{a.data}: rows with a set of >= 2 unidentified appearances found in the same game's train rows and in no "
           f"row of another train game (13+: the same game's rows of Dlvl 13+; any: all its rows)"]
    import numpy as np
    for spec in a.train:
        lab, path = spec.split("=", 1)
        tok_rows, game_rows, own13, own = {}, {}, {}, {}
        n = 0
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            gm = game(r["id"])
            dl = re.search(r"\bDlvl (\d+)", r["state"])
            s = ids(r["state"])
            for t in s:
                tok_rows.setdefault(t, []).append(n)
            game_rows.setdefault(gm, []).append(n)
            own.setdefault(gm, []).append(s)
            if dl and int(dl.group(1)) >= 13:
                own13.setdefault(gm, []).append(s)
            n += 1

        def bitset(ix):  # rows as the bits of one int: the AND of a few of them is the rows holding them all
            v = np.zeros(n, np.uint8)
            v[ix] = 1
            return int.from_bytes(np.packbits(v, bitorder="little").tobytes(), "little")

        bits = {t: bitset(ix) for t, ix in tok_rows.items()}
        out.append(f"\n  {lab} ({path}): {n} rows, {len(game_rows)} games, {len(tok)} appearance tokens")
        for g in GROUPS:
            st = Counter()
            for gm, s in sets[g]:
                st["rows"] += 1
                st["two"] += len(s) >= 2
                mine = bitset(game_rows[gm]) if gm in game_rows else 0
                for kind, src in (("13", own13), ("any", own)):
                    cuts = {tuple(sorted(set(s) & set(o))) for o in src.get(gm, [])}
                    cuts = [c for c in cuts if len(c) >= 2]
                    st["shared" + kind] += bool(cuts)
                    for c in cuts:
                        b = ~mine
                        for t in c:
                            b &= bits[t]
                            if not b:
                                break
                        if not b:
                            st["unique" + kind] += 1
                            break
            k = st["rows"]
            out.append(f"    {g}: {k} rows; >= 2 appearances {st['two'] / k:.3f}; shared with own 13+ rows "
                       f"{st['shared13'] / k:.3f}, unique {st['unique13'] / k:.3f}; shared with own rows "
                       f"{st['sharedany'] / k:.3f}, unique {st['uniqueany'] / k:.3f}")
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


TURN = re.compile(r"\bturn (\d+)")


def own13_index(train):
    """kind(row): "unique" -- the row holds a set of >= 2 appearances found in the game's own 13+ rows of train and in
    no row of another game there (as uniq); "shared" -- such a set but not unique; "none"."""
    import numpy as np
    tok = {}

    def ids(text):
        return sorted(tok.setdefault(t, len(tok)) for t in appearances(text))

    n, tok_rows, game_rows, own13 = 0, {}, {}, {}
    for line in open(train, encoding="utf-8"):
        r = json.loads(line)
        gm = game(r["id"])
        s = ids(r["state"])
        for t in s:
            tok_rows.setdefault(t, []).append(n)
        game_rows.setdefault(gm, []).append(n)
        dl = re.search(r"\bDlvl (\d+)", r["state"])
        if dl and int(dl.group(1)) >= 13:
            own13.setdefault(gm, []).append(s)
        n += 1

    def bitset(ix):
        v = np.zeros(n, np.uint8)
        v[ix] = 1
        return int.from_bytes(np.packbits(v, bitorder="little").tobytes(), "little")

    bits = {t: bitset(ix) for t, ix in tok_rows.items()}

    def kind(r):
        gm, s = game(r["id"]), set(ids(r["state"]))
        cuts = [c for c in {tuple(sorted(s & set(o))) for o in own13.get(gm, [])} if len(c) >= 2]
        mine = bitset(game_rows[gm]) if gm in game_rows else 0
        for c in cuts:
            b = ~mine
            for t in c:
                b &= bits[t]
                if not b:
                    break
            if not b:
                return "unique"
        return "shared" if cuts else "none"

    return kind


GAME_HEAD = re.compile(r'^\{"name": "([^"]*)".*?"start": (\d+),')
STATE_HEAD = re.compile(r'\{"dlvl": (\d+), "hp": -?\d+, "maxhp": -?\d+, "ac": -?\d+, "xl": -?\d+, "turn": (-?\d+)')


def first13_turns(states, games):
    """The turn of each game's first state on Dlvl 13 or deeper in the games file (an arrival: the earliest 13+ row
    c17a had of a train game)."""
    out = {}
    for line in open(states, encoding="utf-8"):
        h = GAME_HEAD.match(line[:400])
        if not h or f"{h.group(1)}@{h.group(2)}" not in games:
            continue
        for m in STATE_HEAD.finditer(line):
            if int(m.group(1)) >= 13 and int(m.group(2)) >= 0:
                out[f"{h.group(1)}@{h.group(2)}"] = int(m.group(2))
                break
    return out


def fp(a):
    """On the untouched preds: fam13 rows split by whether they hold a set of >= 2 appearances shared with the game's
    own 13+ rows in --train and in no row of another game there (as uniq). Difference of differences
    [(with - without) of a model] - [(with - without) of the reference], one bootstrap of players; the reference model
    sees the same rows, so the composition of the two parts (later, richer states) cancels."""
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize_scalar
    from curve_eval import QS, labels, nll
    rows = load(a.data)
    kind = own13_index(a.train)
    rs = [{"id": r["id"], "player": r["id"].split("@")[0].lower(), "q": r["questions"], "gold": r["gold"],
           "fp": kind(r) == "unique"} for r in rows["fam13"]]
    new = [{"id": r["id"], "q": r["questions"], "gold": r["gold"]} for r in rows["new13"]]
    top = pd.Series([r["player"] for r in rs]).value_counts().index[: a.drop_top]
    rs = [r for r in rs if r["player"] not in set(top)]
    allp = sorted({r["player"] for r in rs})
    pidx = {p: i for i, p in enumerate(allp)}
    rng = np.random.default_rng(0)
    ws = [np.ones(len(allp))] + [np.bincount(rng.integers(0, len(allp), len(allp)), minlength=len(allp))
                                 for _ in range(1000)]
    name = Path(a.data).name
    preds = {(m, g): pd.read_parquet(f"preds/{m}-{name}-{g}.parquet") for m in a.models for g in ("fam13", "new13")}
    f1 = [r for r in rs if r["fp"]]
    out = [f"{name} fam13 by a unique fingerprint shared with the game's own 13+ rows in {a.train}; models "
           f"{', '.join(a.models)}; left out {list(top)}; (with - without) of a model minus that of {a.models[0]}, "
           f"one bootstrap of players (1000)",
           f"  with: {len(f1)} rows, {len({game(r['id']) for r in f1})} games, {len({r['player'] for r in f1})} players;"
           f" without: {len(rs) - len(f1)} rows, {len({game(r['id']) for r in rs if not r['fp']})} games, "
           f"{len({r['player'] for r in rs if not r['fp']})} players"]
    did = {}
    for q in QS:
        sel, y, _ = labels(rs, q)
        code = np.array([pidx[rs[i]["player"]] for i in sel])
        flag = np.array([rs[i]["fp"] for i in sel])
        nsel, ny, _ = labels(new, q)
        for m in a.models:
            lg = preds[m, "fam13"][preds[m, "fam13"].question == q].set_index("id").logits
            z = np.stack([np.asarray(lg[rs[i]["id"]], float) for i in sel])
            lgn = preds[m, "new13"][preds[m, "new13"].question == q].set_index("id").logits
            zn = np.stack([np.asarray(lgn[new[i]["id"]], float) for i in nsel])
            t = minimize_scalar(lambda t: nll(zn, ny, t).mean(), bounds=(0.3, 5), method="bounded").x
            for kind, loss in (("raw", nll(z, y)), ("Tnew", nll(z, y, t))):
                s1 = np.bincount(code[flag], weights=loss[flag], minlength=len(allp))
                c1 = np.bincount(code[flag], minlength=len(allp))
                s0 = np.bincount(code[~flag], weights=loss[~flag], minlength=len(allp))
                c0 = np.bincount(code[~flag], minlength=len(allp))
                did[q, kind, m] = np.array([[(w * s1).sum() / (w * c1).sum(), (w * s0).sum() / (w * c0).sum()]
                                            for w in ws])
    for qs, label in ((QS, "mean of the four"), (("end", "ascends"), "end + ascends")) + tuple(((q,), q) for q in QS):
        out.append(f"\n==== {label}")
        for kind in ("raw", "Tnew"):
            v = {m: np.mean([did[q, kind, m] for q in qs], axis=0) for m in a.models}
            line = f"  {kind:4s} " + "  ".join(f"{m} with {v[m][0, 0]:.4f} without {v[m][0, 1]:.4f}" for m in a.models)
            ref = v[a.models[0]][:, 0] - v[a.models[0]][:, 1]
            for m in a.models[1:]:
                d = v[m][:, 0] - v[m][:, 1] - ref
                lo, hi = np.percentile(d[1:], [2.5, 97.5])
                line += f"  | {m} - {a.models[0]}: {d[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            out.append(line)
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


def parts(a):
    """Where in fam13 the gain of a model over the reference sits, on the untouched preds. new13 is the zero (games no
    model saw): recognition of a part = (model - ref on new13) - (model - ref on the part). fam13 split three ways: by
    the appearance sets shared with the game's own 13+ rows in --train (unique to the game / shared but not unique /
    none of >= 2); by richness (distinct unidentified appearances >= the median of fam13 + new13), with the quadruple
    difference [(fam13 rich - poor) - (new13 rich - poor)] of model - ref; by turns from the row to the game's first
    state on Dlvl 13+ in --states (after: the row comes later), new13 split the same way being the zero of each bin
    (c17a may be better near a descent to 13 in any game). One bootstrap of players over both groups; raw and Tnew
    (T fitted on the whole new13 per model and question)."""
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize_scalar
    from curve_eval import QS, labels, nll
    rows = load(a.data)
    kind = own13_index(a.train)
    first13 = first13_turns(a.states, {game(r["id"]) for g in ("fam13", "new13") for r in rows[g]})
    edges = a.bins

    def gap(r):
        tn, gm = TURN.search(r["state"]), game(r["id"])
        return first13[gm] - int(tn.group(1)) if tn and gm in first13 else None

    def to13(r):
        x = gap(r)
        if x is None:
            return "no turn"
        if x <= 0:
            return "after"
        lo = max([0] + [e for e in edges if e < x])
        hi = min([e for e in edges if e >= x], default=None)
        return f"{lo}-{hi}" if hi else f">{lo}"

    rs = {g: [{"id": r["id"], "player": r["id"].split("@")[0].lower(), "q": r["questions"], "gold": r["gold"],
               "kind": kind(r) if g == "fam13" else "", "napp": len(appearances(r["state"])), "to13": to13(r)}
              for r in rows[g]] for g in ("fam13", "new13")}
    top = pd.Series([r["player"] for v in rs.values() for r in v]).value_counts().index[: a.drop_top]
    rs = {g: [r for r in v if r["player"] not in set(top)] for g, v in rs.items()}
    med = float(np.median([r["napp"] for v in rs.values() for r in v]))
    bins = [f"{lo}-{hi}" for lo, hi in zip([0] + edges, edges)] + [f">{edges[-1]}", "after", "no turn"]
    defs = ([("fam13", "fam13", lambda r: True)]
            + [(f"fam13 {k}", "fam13", lambda r, k=k: r["kind"] == k) for k in ("unique", "shared", "none")]
            + [("fam13 rich", "fam13", lambda r: r["napp"] >= med), ("fam13 poor", "fam13", lambda r: r["napp"] < med),
               ("new13", "new13", lambda r: True), ("new13 rich", "new13", lambda r: r["napp"] >= med),
               ("new13 poor", "new13", lambda r: r["napp"] < med)]
            + [(f"{g} to13 {b}", g, lambda r, b=b: r["to13"] == b) for b in bins for g in ("fam13", "new13")])
    allp = sorted({r["player"] for v in rs.values() for r in v})
    pidx = {p: i for i, p in enumerate(allp)}
    rng = np.random.default_rng(0)
    ws = np.array([np.ones(len(allp))] + [np.bincount(rng.integers(0, len(allp), len(allp)), minlength=len(allp))
                                          for _ in range(1000)])
    name = Path(a.data).name
    preds = {(m, g): pd.read_parquet(f"preds/{m}-{name}-{g}.parquet") for m in a.models for g in ("fam13", "new13")}
    ref, other = a.models[0], a.models[1:]
    out = [f"{name}: where the gain of {', '.join(other)} over {ref} sits in fam13; train {a.train}; left out "
           f"{list(top)}; median of distinct appearances {med:g}; one bootstrap of players (1000) over fam13 + new13"]
    for p, g, f in defs:
        sub = [r for r in rs[g] if f(r)]
        out.append(f"  {p:22s} {len(sub):5d} rows, {len({game(r['id']) for r in sub}):4d} games, "
                   f"{len({r['player'] for r in sub}):4d} players")
    for i, g in enumerate(("fam13", "new13")):
        gaps = [x for x in map(gap, rows[g]) if x is not None]
        out.insert(1 + i, f"  turns from a {g} row to the game's first state on Dlvl 13+, quantiles 0.1 / 0.25 / 0.5 / "
                   "0.75 / 0.9: " + " / ".join(f"{np.quantile(gaps, x):.0f}" for x in (0.1, 0.25, 0.5, 0.75, 0.9)))
    mu = {}
    for q in QS:
        z, y, code, part = {}, {}, {}, {}
        for g in ("fam13", "new13"):
            sel, y[g], _ = labels(rs[g], q)
            code[g] = np.array([pidx[rs[g][i]["player"]] for i in sel])
            for m in a.models:
                lg = preds[m, g][preds[m, g].question == q].set_index("id").logits
                z[g, m] = np.stack([np.asarray(lg[rs[g][i]["id"]], float) for i in sel])
            for p, gg, f in defs:
                if gg == g:
                    part[p] = np.array([f(rs[g][i]) for i in sel])
        for m in a.models:
            t = minimize_scalar(lambda t: nll(z["new13", m], y["new13"], t).mean(), bounds=(0.3, 5),
                                method="bounded").x
            for k, tt in (("raw", 1.0), ("Tnew", t)):
                for p, g, _ in defs:
                    sel = part[p]
                    loss = nll(z[g, m][sel], y[g][sel], tt)
                    s = np.bincount(code[g][sel], weights=loss, minlength=len(allp))
                    c = np.bincount(code[g][sel], minlength=len(allp))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mu[q, k, m, p] = (ws @ s) / (ws @ c)

    def ci(v):
        lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
        return f"{v[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]"

    for qs, label in ((QS, "mean of the four"), (("end", "ascends"), "end + ascends")) + tuple(((q,), q) for q in QS):
        out.append(f"\n==== {label}")
        for k in ("raw", "Tnew"):
            out.append(f"  {k}")
            lv = {(m, p): np.mean([mu[q, k, m, p] for q in qs], axis=0) for m in a.models for p, _, _ in defs}
            for m in other:
                gain = {p: lv[m, p] - lv[ref, p] for p, _, _ in defs}
                for p, _, _ in defs:
                    same = p.replace("fam13", "new13")
                    out.append(f"    {p:22s} " + "  ".join(f"{x} {lv[x, p][0]:.4f}" for x in a.models)
                               + f"  | {m} - {ref} {ci(gain[p])}"
                               + ("" if p.startswith("new13") else f"  recognition {ci(gain['new13'] - gain[p])}")
                               + (f"  against {same}: {ci(gain[same] - gain[p])}" if " to13 " in p
                                  and p.startswith("fam13") else ""))
                out.append(f"    unique - none of {m} - {ref}: {ci(gain['fam13 unique'] - gain['fam13 none'])}")
                out.append(f"    [(fam13 rich - poor) - (new13 rich - poor)] of {m} - {ref}: " + ci(
                    (gain["fam13 rich"] - gain["fam13 poor"]) - (gain["new13 rich"] - gain["new13 poor"])))
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


def ident(a):
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for g in GROUPS:
        lines = open(Path(a.data, f"{g}.jsonl"), encoding="utf-8").readlines()
        keep = sorted(lines, key=lambda x: zlib.crc32(json.loads(x)["id"].encode()))[: a.n]
        Path(a.out, f"{g}.jsonl").write_text("".join(keep), encoding="utf-8")
        print(g, len(keep))


def same(a):
    import numpy as np
    import pandas as pd
    for m in a.models:
        for g in GROUPS:
            x = pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{g}.parquet").set_index(["id", "question"]).logits
            y = pd.read_parquet(f"preds/{m}-{Path(a.ref).name}-{g}.parquet").set_index(["id", "question"]).logits
            d = max(float(np.abs(np.asarray(x[k], float) - np.asarray(y[k], float)).max()) for k in x.index)
            print(f"{m} {g}: {len(x)} rows x questions, max |logit difference| {d:.2e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("data")
    b.add_argument("--out", required=True)
    b.add_argument("--report", default="")
    u = sub.add_parser("uniq")
    u.add_argument("data")
    u.add_argument("--train", nargs="+", required=True, help="label=path")
    u.add_argument("--out", default="")
    f = sub.add_parser("fp")
    f.add_argument("data")
    f.add_argument("--train", required=True)
    f.add_argument("--models", nargs="+", default=["c17r", "c17a"])
    f.add_argument("--drop-top", type=int, default=0)
    f.add_argument("--out", default="")
    pp = sub.add_parser("parts")
    pp.add_argument("data")
    pp.add_argument("--train", required=True)
    pp.add_argument("--models", nargs="+", default=["c17r", "c17a"])
    pp.add_argument("--states", required=True, help="the games file the set was built from (first 13+ arrival)")
    pp.add_argument("--bins", nargs="+", type=int, default=[500, 1500, 4000], help="turns to the first 13+ state")
    pp.add_argument("--drop-top", type=int, default=0)
    pp.add_argument("--out", default="")
    i = sub.add_parser("ident")
    i.add_argument("data")
    i.add_argument("--out", required=True)
    i.add_argument("--n", type=int, default=300)
    s = sub.add_parser("same")
    s.add_argument("data")
    s.add_argument("ref")
    s.add_argument("models", nargs="+")
    a = ap.parse_args()
    {"build": build, "uniq": uniq, "fp": fp, "parts": parts, "ident": ident, "same": same}[a.cmd](a)


if __name__ == "__main__":
    main()
