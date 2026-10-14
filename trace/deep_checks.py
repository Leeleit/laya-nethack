"""Checks of the per-band comparison of curve_eval.py.
preds: per Dlvl band and question -- raw loss, loss with T fitted on all rows (as curve_eval), loss with T fitted inside
the band (both cross-fitted over players), differences to the reference with a 95 % bootstrap over the band's players,
val and test apart; end on the rows of games that did not ascend, and the cause given death (the ascension option
dropped, the rest renormalised, T fitted on those rows); the top players' share of the band's ascended rows.
select: what --per-band (one random state of each band a game was in: a state's weight is 1 / the game's states in
that band, a number set by the game's future) does to the labels, against all states and the first state of the band
(set by the past only), on the same games; and on one set of states -- the first of each band -- weighted 1 / n as
--per-band weighs them against unweighted.
  python deep_checks.py preds human_dwa19h dwa=c17v mix=m17 staged=s17 others=o17 [--drop-top N] [--out ../results/x.txt]
      [--expect-from human_dwa20/train.jsonl]   (P(ascend) - E, E: train share by band x experience x result)
  python deep_checks.py select nao/val_dwa_states3e.jsonl [--exclude-players nao/bot_players_all.txt] [--out ...]
"""
import argparse
import bisect
import collections
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

BANDS = (4, 12, 20, 24, 29)  # human_build --per-band
NAMES = ("1-4", "5-12", "13-20", "21-24", "25-29", "30+")


EVENTS = {"ascends": ("ascends", "true"), "nodeeper": ("depth", "0")}  # question, the gold label of the event


def expectation(path, event="ascends", shrink=20, n=1000, seed=1):
    """E = train's share of the event (ascended; depth's hard label "no deeper") in the cell band x experience x result,
    shrunk to band x result and then to the band (pseudo-count `shrink`): what strength in the question predicts. Also
    n bootstrap draws of E over train players."""
    import re
    import pandas as pd
    q, hit = EVENTS[event]
    cells, players, ys = [], [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if q not in r["gold"]:
            continue
        m = re.search(r"experience (\d)/\d, result (\d)/\d", r["questions"][q]["instructions"])
        b = bisect.bisect_left(BANDS, int(re.search(r"\bDlvl (\d+)", r["state"]).group(1)))
        cells.append(b * 36 + (int(m.group(1)) * 6 + int(m.group(2)) if m else 0))
        players.append(r["id"].split("@")[0].lower())
        ys.append(r["gold"][q]["label"] == hit)
    codes, uniq = pd.factorize(np.array(players))
    cells, ys = np.array(cells), np.array(ys, float)
    cnt, asc = np.zeros((len(uniq), 216)), np.zeros((len(uniq), 216))
    np.add.at(cnt, (codes, cells), 1)
    np.add.at(asc, (codes, cells), ys)

    def shrunk(nc, kc):  # nc, kc: (..., 216)
        n3, k3 = nc.reshape(-1, 6, 6, 6), kc.reshape(-1, 6, 6, 6)  # band, experience, result
        n2, k2 = n3.sum(2, keepdims=True), k3.sum(2, keepdims=True)
        n1, k1 = n2.sum(3, keepdims=True), k2.sum(3, keepdims=True)
        e2 = (k2 + shrink * k1 / np.maximum(n1, 1)) / (n2 + shrink)
        return ((k3 + shrink * e2) / (n3 + shrink)).reshape(nc.shape)

    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(n)]).astype(float)
    return shrunk(cnt.sum(0), asc.sum(0)), shrunk(w @ cnt, w @ asc)


def minus_e(p, cells, e_point, e_draws, b):
    """mean(p - E) with a 95 % interval from test players and train players drawn together."""
    codes, ws, cnt = b
    sp = np.bincount(codes, weights=p, minlength=len(cnt))
    v = []
    for w, ed in zip(ws, e_draws):
        se = np.bincount(codes, weights=ed[cells], minlength=len(cnt))
        v.append((w * (sp - se)).sum() / (w * cnt).sum())
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"{(p - e_point[cells]).mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def preds(a):
    import pandas as pd
    from curve_eval import boot_idx, ci, crossfit, labels, load_rows, nll, softmax
    rows = load_rows(a.data)
    exp = {ev: expectation(a.expect_from, ev) for ev in EVENTS} if a.expect_from else {}
    head = []
    if a.drop_top:
        c = collections.Counter(r["player"] for r in rows if r["asc"])
        top = {p for p, _ in c.most_common(a.drop_top)}
        gone = [r for r in rows if r["player"] in top]
        rows = [r for r in rows if r["player"] not in top]
        head.append(f"left out: the {a.drop_top} player(s) with the most ascended rows -- {len(gone)} rows, "
                    f"{sum(r['asc'] for r in gone)} ascended, splits {dict(collections.Counter(r['split'] for r in gone))}")
    for path, keep in ((a.keep_games, True), (a.drop_games, False)):  # after --drop-top: the same player left out
        if path:
            games = {f"{g['name']}@{g['start']}" for g in json.load(open(path, encoding="utf-8"))}
            rows = [r for r in rows if (r["id"].split("#")[0] in games) == keep]
            head.append(f"{'only' if keep else 'without'} the games of {path}: {len(rows)} rows")
    if a.deepest:  # cut by the present (deepest so far), not by how the game ended
        lo, hi = map(int, a.deepest.split("-"))
        rows = [r for r in rows if lo <= r["deepest"] <= hi]
        head.append(f"deepest so far {lo}-{hi}: {len(rows)} rows")
    specs =[m.split("=", 1) for m in a.models]
    ref = specs[0][0]
    pairs = [(lab, ref) for lab, _ in specs[1:]] + ([("staged", "mix")] if {"staged", "mix"} <= {l for l, _ in specs} else [])
    dfs = {lab: pd.concat([pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{s}.parquet") for s in ("val", "test")])
           for lab, m in specs}
    out = [f"{a.data}: {len(rows)} rows; models " + ", ".join(f"{l}={m}" for l, m in specs)] + head
    for q in ("ttd", "depth", "end", "ascends"):
        sel, y, k = labels(rows, q)
        pl = np.array([rows[i]["player"] for i in sel])
        band = np.array([bisect.bisect_left(BANDS, rows[i]["dlvl"]) for i in sel])
        split = np.array([rows[i]["split"] for i in sel])
        asc = np.array([rows[i]["asc"] for i in sel])
        z = {}
        for lab, _ in specs:
            lg = dfs[lab][dfs[lab].question == q].set_index("id").logits
            z[lab] = np.stack([np.asarray(lg[rows[i]["id"]], float) for i in sel])
        glob = {lab: crossfit(z[lab], y, pl) for lab in z}

        def e_line(name, probs, m, event="ascends", observed=None):  # --expect-from: P - E per model, all / val / test
            e_point, e_draws = exp[event]
            cells = np.array([bisect.bisect_left(BANDS, rows[i]["dlvl"]) * 36 + rows[i]["exp"] * 6 + rows[i]["res"]
                              for i in sel[m]])
            s, parts = split[m], []
            for lab, p in probs.items():
                seg = f"{lab} {minus_e(p, cells, e_point, e_draws, boot_idx(pl[m]))}"
                for sp in ("val", "test"):
                    ms = s == sp
                    seg += f" {sp} {minus_e(p[ms], cells[ms], e_point, e_draws, boot_idx(pl[m][ms]))}"
                parts.append(seg)
            obs = asc[m].mean() if observed is None else observed
            return (f"    {name} - E (E {e_point[cells].mean():.3f}, observed {obs:.3f}; val / test after "
                    "each): " + "; ".join(parts))
        out.append(f"\n==== {q}: {len(y)} rows; T on all rows " + "  ".join(f"{l} {glob[l][1]:.2f}" for l in z))
        for b, name in enumerate(NAMES):
            m = band == b
            if m.sum() < 20:
                continue
            loc = {lab: crossfit(z[lab][m], y[m], pl[m]) for lab in z}
            raw = {lab: nll(z[lab][m], y[m]) for lab in z}
            bb = boot_idx(pl[m])
            out.append(f"  Dlvl {name}: {m.sum()} rows, {len(set(pl[m]))} players (the top one "
                       f"{collections.Counter(pl[m]).most_common(1)[0][1] / m.sum():.2f} of rows), ascended {asc[m].mean():.2f}"
                       f" (val {asc[m & (split == 'val')].mean():.2f}, test {asc[m & (split == 'test')].mean():.2f})")
            for lab in z:
                out.append(f"    {lab:7s} raw {raw[lab].mean():.4f}  T-all {glob[lab][0][m].mean():.4f}"
                           f"  T-band {loc[lab][0].mean():.4f} (T {loc[lab][1]:.2f})")
            for x, r in pairs:
                line = (f"    {x} - {r}: raw {ci(raw[x] - raw[r], bb)}  T-all {ci(glob[x][0][m] - glob[r][0][m], bb)}"
                        f"  T-band {ci(loc[x][0] - loc[r][0], bb)}")
                for s in ("val", "test"):
                    ms = split[m] == s
                    d = (loc[x][0] - loc[r][0])[ms]
                    line += f"  | {s} {ci(d, boot_idx(pl[m][ms]))}"
                out.append(line)
            if q in ("ttd", "depth"):  # calibration in the large at the forced 1000-turn bound / no deeper
                keys = list(rows[sel[0]]["q"][q]["criteria"])
                # ttd's bins differ between sets, 1000 is a forced bound in each ("761-1000", "at most 1000"...)
                hi = next(i for i, k in enumerate(keys) if k.split("-")[-1].split()[-1] == "1000") + 1 if q == "ttd" else 1
                out.append(f"    P({'ttd <= 1000' if q == 'ttd' else 'no deeper'}) mean: observed {(y[m] < hi).mean():.3f}  "
                           + "  ".join(f"{l} {softmax(z[l][m] / loc[l][1])[:, :hi].sum(1).mean():.3f}" for l in z))
                if q == "depth" and exp:
                    out.append(e_line("P(no deeper)", {l: softmax(z[l][m] / loc[l][1])[:, 0] for l in z}, m,
                                      "nodeeper", (y[m] == 0).mean()))
            if q == "end":
                keys = list(rows[sel[0]]["q"]["end"]["criteria"])
                ia = keys.index("ascend")
                nd = m & ~asc
                yd = y[nd]
                keep = [c for c in range(k) if c != ia]
                yc = np.searchsorted(keep, yd)
                bd = boot_idx(pl[nd])
                cond = {lab: crossfit(z[lab][nd][:, keep], yc, pl[nd]) for lab in z}
                out.append(f"    end, games that did not ascend ({nd.sum()} rows): T-band "
                           + "  ".join(f"{l} {loc[l][0][~asc[m]].mean():.4f}" for l in z)
                           + "; cause given death " + "  ".join(f"{l} {cond[l][0].mean():.4f}" for l in z))
                for x, r in pairs:
                    out.append(f"      {x} - {r}: T-band {ci(loc[x][0][~asc[m]] - loc[r][0][~asc[m]], bd)}"
                               f"  cause given death {ci(cond[x][0] - cond[r][0], bd)}")
                p = {lab: softmax(z[lab][m] / loc[lab][1])[:, ia].mean() for lab in z}
                out.append(f"    P(ascend) mean: observed {asc[m].mean():.3f}  " + "  ".join(f"{l} {p[l]:.3f}" for l in z))
                if exp:
                    out.append(e_line("P(ascend)", {l: softmax(z[l][m] / loc[l][1])[:, ia] for l in z}, m))
            if q == "ascends":
                it = list(rows[sel[0]]["q"]["ascends"]["criteria"]).index("true")
                out.append(f"    P(yes) mean: observed {asc[m].mean():.3f}  " + "  ".join(
                    f"{l} {softmax(z[l][m] / loc[l][1])[:, it].mean():.3f}" for l in z))
                if exp:
                    out.append(e_line("P(yes)", {l: softmax(z[l][m] / loc[l][1])[:, it] for l in z}, m))
                c = collections.Counter(pl[m & asc])
                top = c.most_common(5)
                out.append(f"    ascended rows {sum(c.values())}, players {len(c)}; top 5 "
                           + ", ".join(f"{n}" for _, n in top) + f" = {sum(n for _, n in top) / max(sum(c.values()), 1):.2f}")
    return out


SKIP = set()


def game(line):
    try:
        g = json.loads(line)
    except ValueError:
        return None
    if not g.get("states") or g["name"].lower() in SKIP:
        return None
    st = [(s["dlvl"], s["deepest"], s["turn"]) for s in g["states"]]
    return g["death"] == "ascended", g["maxlvl"], g["turns"], st


def _init(skip):
    SKIP.update(skip)


def select(a):
    skip = set()
    for p in a.exclude_players:
        skip |= {x.split()[0].lower() for x in open(p, encoding="utf-8") if x.strip() and not x.startswith("#")}
    acc = {w: collections.defaultdict(lambda: np.zeros(5)) for w in ("all", "perband", "first", "first/n")}
    games = 0
    lines = (x for p in a.states for x in open(p, encoding="utf-8"))
    with Pool(a.workers, _init, (skip,)) as pool:
        for g in pool.imap(game, lines, chunksize=64):
            if g is None:
                continue
            games += 1
            asc, maxlvl, turns, st = g
            by = collections.defaultdict(list)
            for s in st:
                by[bisect.bisect_left(BANDS, s[0])].append(s)
            for b, v in by.items():
                n = len(v)
                for j, (dl, deepest, turn) in enumerate(v):
                    left = turns - turn if turn >= 0 else None
                    die1k = float(not asc and left is not None and left <= 1000)
                    lab = (asc, maxlvl - deepest, math.log1p(left) if left is not None and left >= 0 else 0.0, die1k)
                    for w, wt in (("all", 1.0), ("perband", 1 / n), ("first", float(j == 0)), ("first/n", (j == 0) / n)):
                        if wt:
                            acc[w][b] += wt * np.array([1.0, *lab])
    out = [f"{' '.join(a.states)}: {games} games (bots left out: {len(skip)} names); per band, label means of the states each "
           "selection takes (weight: all 1, per-band 1/n of the game's states in the band, first = the band's first "
           "state, first/n = the first states weighted 1/n as per-band weighs them)",
           "band   selection   weight sum   ascended  depth gain  log(1+turns left)  dies within 1000"]
    for b, name in enumerate(NAMES):
        for w in ("all", "perband", "first", "first/n"):
            s = acc[w][b]
            if s[0]:
                out.append(f"{name:6s} {w:10s} {s[0]:11.0f}   {s[1] / s[0]:.4f}    {s[2] / s[0]:7.3f}   "
                           f"{s[3] / s[0]:10.3f}          {s[4] / s[0]:.4f}")
    out.append("games / states by band (human_build --band-rates for as many rows as --per-band): "
               + " ".join(f"{acc['first'][b][0] / max(acc['all'][b][0], 1):.4f}" for b in range(len(NAMES))))
    return out


def cover(a):
    """Per band of each training file: rows, distinct games, distinct games that did not ascend, ascended rows."""
    out = ["file  band  rows  games  games-not-ascended  ascended-rows"]
    for path in a.files:
        cnt = collections.defaultdict(lambda: [0, set(), set(), 0])
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            dl = int(r["state"].split("Dlvl ", 1)[1].split()[0]) if "Dlvl " in r["state"] else -1
            c = cnt[bisect.bisect_left(BANDS, dl)]
            g = r["id"].rsplit("#", 1)[0]
            asc = r["gold"]["ascends"]["label"] == "true"
            c[0] += 1
            c[1].add(g)
            if not asc:
                c[2].add(g)
            c[3] += asc
        for b, name in enumerate(NAMES):
            n, gs, gd, na = cnt[b]
            out.append(f"{path}  {name:6s} {n:7d} {len(gs):7d} {len(gd):7d} {na:7d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cover")
    c.add_argument("files", nargs="+")
    c.add_argument("--out", default="")
    p = sub.add_parser("preds")
    p.add_argument("data")
    p.add_argument("models", nargs="+", help="label=model; the first is the reference")
    p.add_argument("--drop-top", type=int, default=0, help="leave out the N players with the most ascended rows")
    p.add_argument("--expect-from", default="",
                   help="train jsonl picked as the test is (fixed rates): print P(ascend) - E, E = that train's share "
                        "in the row's band x experience x result cell")
    p.add_argument("--keep-games", default="", help="games json (name, start): only their rows")
    p.add_argument("--drop-games", default="", help="games json (name, start): without their rows")
    p.add_argument("--deepest", default="", help="lo-hi: only rows whose deepest Dlvl so far is in it")
    p.add_argument("--out", default="")
    s = sub.add_parser("select")
    s.add_argument("states", nargs="+")
    s.add_argument("--exclude-players", nargs="*", default=[])
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--out", default="")
    a = ap.parse_args()
    text = "\n".join({"preds": preds, "select": select, "cover": cover}[a.cmd](a))
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
