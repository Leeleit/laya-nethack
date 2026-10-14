"""Does a teacher know its own train rows? Raw log loss of the teacher (the mean softmax of its seeds) on its train rows
in Dlvl 21-24 / 25-29 against the unseen states of the same games in the same bands; a teacher that leaks the label of
a game it has seen would give its students that label, not what the state says.

  build: rows of a set with every state (e.g. human_dwa21: all states from Dlvl 13) in the bands, of the train games of
         the teacher's set, not among its train rows -> unseen.jsonl; the teacher's train rows in the bands -> seen.jsonl
         (both small enough to predict before the whole train file); the shared rows are checked to carry the same
         state, questions and labels.
    python mem_gate.py build --all human_dwa21/train.jsonl --seen human_dwa24/train.jsonl --out human_mem29
  read:  G = mean over the 8 cells (ttd, depth, end, ascends x 21-24, 25-29) of the mean raw log loss; delta = G(seen)
         - G(unseen); in each band only the games with a seen row there; bootstrap over players (both groups at once).
    python mem_gate.py read --seen human_mem29/seen.jsonl --unseen human_mem29/unseen.jsonl \
        --models c17e c17es1 c17es2 c17es3 --out ../results/mem29.txt
  pair:  the same delta within each game (unseen rows weighted to the game's seen rows), the same arguments as read;
         build and pair take --bands (0: 1-4 ... 5: 30+; 21-29 by default), build several --all files and --cap k (at
         most k unseen rows per game and band, by crc32 of the id: within a game the rows stay exchangeable).
    python mem_gate.py build --all human_dwa23/train.jsonl human_dwa21/train.jsonl --seen human_dwa24/train.jsonl \
        --bands 1 2 5 --cap 4 --out human_mem30
  predictions: preds/<model>-human_mem29-seen.parquet, preds/<model>-human_mem29-unseen.parquet (predict.py)
"""
import argparse
import bisect
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from curve_eval import QS, labels, softmax

BANDS = (4, 12, 20, 24, 29)  # deep_checks.BANDS
DEEP = {3: "21-24", 4: "25-29"}
NAMES = {0: "1-4", 1: "5-12", 2: "13-20", 3: "21-24", 4: "25-29", 5: "30+"}


def game(i):
    return i.split("#")[0]


def band(state):
    m = re.search(r"\bDlvl (\d+)", state)
    return bisect.bisect_left(BANDS, int(m.group(1))) if m else -1


def build(a):
    bands = set(a.bands)
    seen, seen_lines = {}, []
    for line in open(a.seen, encoding="utf-8"):
        r = json.loads(line)
        seen[r["id"]] = r
        if band(r["state"]) in bands:
            seen_lines.append(line)
    games = {game(i) for i in seen}
    out, shared, differ, n_band, done = {}, 0, 0, {b: [0, 0, 0] for b in bands}, set()
    for path in a.all:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            b = band(r["state"])
            if b not in bands or game(r["id"]) not in games or r["id"] in done:  # a row in two --all files: once
                continue
            done.add(r["id"])
            s = seen.get(r["id"])
            if s is None:
                out.setdefault((game(r["id"]), b), []).append((zlib.crc32(r["id"].encode()), line))
                n_band[b][1] += 1
                continue
            shared += 1
            n_band[b][0] += 1
            same = s["state"] == r["state"] and s["questions"] == r["questions"] and \
                {q: g["label"] for q, g in s["gold"].items()} == {q: g["label"] for q, g in r["gold"].items()}
            differ += not same
    kept = []
    for (_, b), rs in out.items():  # --cap: the unseen rows of a game in a band with the smallest crc32 of their id
        rs = sorted(rs)[:a.cap] if a.cap else rs
        n_band[b][2] += len(rs)
        kept += [line for _, line in rs]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "unseen.jsonl").write_text("".join(kept), encoding="utf-8")
    Path(a.out, "seen.jsonl").write_text("".join(seen_lines), encoding="utf-8")
    print(f"seen rows in bands {sorted(bands)}: {len(seen_lines)}, found in {' '.join(a.all)}: {shared}, of them differing "
          f"in state, questions or labels: {differ}")
    for b, (s, u, k) in sorted(n_band.items()):
        print(f"  {NAMES[b]}: seen {s}, unseen {u}, kept {k}")
    print(f"-> {a.out}/seen.jsonl, {a.out}/unseen.jsonl")
    if differ:  # other bins or questions in an --all file: its unseen rows are not comparable with the seen ones
        raise SystemExit(1)


def rows_of(path, bands=DEEP):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        b = band(r["state"])
        if b in bands:
            rows.append({"id": r["id"], "player": r["id"].split("@")[0].lower(), "band": b, "q": r["questions"],
                         "gold": r["gold"]})
    return rows


def losses(rows, dfs, q):
    """-> (row indices with q, raw log loss of the mean softmax of dfs)."""
    sel, y, _ = labels(rows, q)
    ps = []
    for df in dfs:
        lg = df[df.question == q].set_index("id").logits
        ps.append(softmax(np.stack([np.asarray(lg[rows[i]["id"]], float) for i in sel])))
    p = np.mean(ps, 0)
    return sel, -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def read(a):
    seen, unseen = rows_of(a.seen), rows_of(a.unseen)
    seen_games = {b: {game(r["id"]) for r in seen if r["band"] == b} for b in DEEP}
    groups = {"seen": seen, "unseen same games": [r for r in unseen if game(r["id"]) in seen_games[r["band"]]],
              "unseen all": unseen}
    players = sorted({r["player"] for rs in groups.values() for r in rs})
    pidx = {p: i for i, p in enumerate(players)}
    rng = np.random.default_rng(0)
    ws = np.array([np.ones(len(players))] + [np.bincount(rng.integers(0, len(players), len(players)),
                                                         minlength=len(players)) for _ in range(1000)])
    sname, uname = (f"{Path(p).parent.name}-{Path(p).stem}" for p in (a.seen, a.unseen))
    pr = {m: (pd.read_parquet(f"preds/{m}-{sname}.parquet"), pd.read_parquet(f"preds/{m}-{uname}.parquet"))
          for m in a.models}
    teachers = [("mean of " + " ".join(a.models), a.models)] + [(m, [m]) for m in a.models]
    out = [f"seen {a.seen}: {len(seen)} rows in 21-29; unseen {a.unseen}: {len(unseen)}, of games with a seen row in the "
           f"band {len(groups['unseen same games'])}; players {len(players)}"]
    for g, rs in groups.items():
        out.append(f"  {g}: " + ", ".join(f"{DEEP[b]} {sum(r['band'] == b for r in rs)} rows / "
                                         f"{len({game(r['id']) for r in rs if r['band'] == b})} games" for b in DEEP))
    out.append("raw log loss, the mean over each group's rows; delta = seen - unseen [95 % bootstrap over players]; "
               "delta < -0.05 on G of the teacher -> it leaks labels")
    Gs = {}
    for tname, ms in teachers:
        out.append(f"\n==== teacher: {tname}")
        cell = {}  # (q, band, group) -> means under every weight
        for q in QS:
            for g, rs in groups.items():
                dfs = [pr[m][0 if g == "seen" else 1] for m in ms]
                sel, loss = losses(rs, dfs, q)
                for b in DEEP:
                    k = np.array([rs[i]["band"] == b for i in sel])
                    codes = np.array([pidx[rs[i]["player"]] for i in sel[k]])
                    s = np.bincount(codes, weights=loss[k], minlength=len(players))
                    c = np.bincount(codes, minlength=len(players))
                    cell[q, b, g] = (ws * s).sum(1) / (ws * c).sum(1)
        out.append(f"  {'cell':<16}{'seen':>8}{'unseen':>9}{'delta (same games)':>34}{'delta (all unseen)':>34}")

        def fmt(d):
            lo, hi = np.percentile(d[1:], [2.5, 97.5])
            return f"{d[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        for q in QS:
            for b in DEEP:
                s, u, ua = cell[q, b, "seen"], cell[q, b, "unseen same games"], cell[q, b, "unseen all"]
                out.append(f"  {q + ' ' + DEEP[b]:<16}{s[0]:8.4f}{u[0]:9.4f}{fmt(s - u):>34}{fmt(s - ua):>34}")
        G = {g: np.mean([cell[q, b, g] for q in QS for b in DEEP], 0) for g in groups}
        d, da = G["seen"] - G["unseen same games"], G["seen"] - G["unseen all"]
        out.append(f"  {'G':<16}{G['seen'][0]:8.4f}{G['unseen same games'][0]:9.4f}{fmt(d):>34}{fmt(da):>34}")
        Gs[tname] = G
        if tname.startswith("mean of"):
            lo, hi = np.percentile(d[1:], [2.5, 97.5])
            if lo > 0:  # worse on its own train rows than on unseen rows of the same games: a plumbing error
                verdict = "review (the 95 % interval above 0)"
            elif d[0] < -0.05:
                verdict = "leaks (below -0.05)"
            elif d[0] < -0.024:
                verdict = "passes, intermediate (from -0.05 to -0.024: the leak is comparable with the signal)"
            else:
                verdict = "passes (not below -0.024)"
            out.append(f"  gate: delta_mem (same games) {d[0]:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {verdict}")
            ea = np.mean([cell[q, b, "seen"] - cell[q, b, "unseen same games"] for q in ("end", "ascends")
                          for b in DEEP], 0)
            out.append(f"  end + ascends delta (same games) {fmt(ea)}" +
                       ("  -> leak in the game labels (below -0.05)" if ea[0] < -0.05 else ""))
    out.append("\n==== ensemble gain E = G(mean of the seeds' probabilities) - mean of the seeds' G")
    E = {}
    for g in ("seen", "unseen same games"):
        E[g] = Gs[teachers[0][0]][g][0] - np.mean([Gs[m][g][0] for m in a.models])
        out.append(f"  {g}: {E[g]:+.4f}")
    r = E["seen"] / E["unseen same games"] if E["unseen same games"] < 0 else float("nan")
    out.append(f"  E seen / E unseen {r:.2f}" + ("  -> below 0.5: the teacher has little to pass on" if r < 0.5 else ""))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


def pair(a):
    """delta within each game: unseen rows weighted so that every game carries as much as its seen rows (a game with
    one state in the band is seen or nothing, so the plain mean over unseen rows leans on long stays in the band)."""
    bands = sorted(a.bands)
    seen, unseen = rows_of(a.seen, bands), rows_of(a.unseen, bands)
    games = sorted({(game(r["id"]), r["band"]) for r in seen} & {(game(r["id"]), r["band"]) for r in unseen})
    gidx = {g: i for i, g in enumerate(games)}
    players = sorted({g[0].split("@")[0].lower() for g in games})
    pidx = {p: i for i, p in enumerate(players)}
    gpl = np.array([pidx[g[0].split("@")[0].lower()] for g in games])
    gband = np.array([g[1] for g in games])
    rng = np.random.default_rng(0)
    ws = np.array([np.ones(len(players))] + [np.bincount(rng.integers(0, len(players), len(players)),
                                                         minlength=len(players)) for _ in range(1000)])
    sname, uname = (f"{Path(p).parent.name}-{Path(p).stem}" for p in (a.seen, a.unseen))
    pr = {m: (pd.read_parquet(f"preds/{m}-{sname}.parquet"), pd.read_parquet(f"preds/{m}-{uname}.parquet"))
          for m in a.models}
    teachers = [("mean of " + " ".join(a.models), a.models)] + [(m, [m]) for m in a.models]
    out = [f"games with seen and unseen rows in the band: " + ", ".join(
        f"{NAMES[b]} {(gband == b).sum()}" for b in bands) + f"; players {len(players)}",
        "raw log loss; per game d = mean(seen) - mean(unseen), the mean of d weighted by the game's seen rows "
        "[95 % bootstrap over players]"]

    def fmt(d):
        lo, hi = np.percentile(d[1:], [2.5, 97.5])
        return f"{d[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    for tname, ms in teachers:
        out.append(f"\n==== teacher: {tname}")
        cells = []
        for q in QS:
            sums = {}
            for g, rs in (("seen", seen), ("unseen", unseen)):
                sel, loss = losses(rs, [pr[m][0 if g == "seen" else 1] for m in ms], q)
                s, c = np.zeros(len(games)), np.zeros(len(games))
                for i, l in zip(sel, loss):
                    k = gidx.get((game(rs[i]["id"]), rs[i]["band"]))
                    if k is not None:
                        s[k] += l
                        c[k] += 1
                sums[g] = (s, c)
            (ss, cs), (su, cu) = sums["seen"], sums["unseen"]
            ok = (cs > 0) & (cu > 0)
            d = np.where(ok, ss / np.maximum(cs, 1) - su / np.maximum(cu, 1), 0.0)
            for b in bands:
                k = ok & (gband == b)
                num = np.bincount(gpl[k], weights=cs[k] * d[k], minlength=len(players))
                den = np.bincount(gpl[k], weights=cs[k], minlength=len(players))
                v = (ws * num).sum(1) / (ws * den).sum(1)
                cells.append((b, v))
                out.append(f"  {q + ' ' + NAMES[b]:<16}{fmt(v):>30}")
        for b in bands:
            out.append(f"  {'G ' + NAMES[b]:<16}{fmt(np.mean([v for bb, v in cells if bb == b], 0)):>30}")
        if set(DEEP) <= set(bands):
            out.append(f"  {'G 21-29':<16}{fmt(np.mean([v for bb, v in cells if bb in DEEP], 0)):>30}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--seen", required=True)
    p.add_argument("--unseen", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--bands", nargs="+", type=int, default=sorted(DEEP), help="0: 1-4 ... 5: 30+")
    b = sub.add_parser("build")
    b.add_argument("--all", nargs="+", required=True)
    b.add_argument("--seen", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--bands", nargs="+", type=int, default=sorted(DEEP), help="0: 1-4 ... 5: 30+")
    b.add_argument("--cap", type=int, default=0, help="at most this many unseen rows per game and band (0: all)")
    r = sub.add_parser("read")
    r.add_argument("--seen", required=True)
    r.add_argument("--unseen", required=True)
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--out", default="")
    a = ap.parse_args()
    {"build": build, "read": read, "pair": pair}[a.cmd](a)


if __name__ == "__main__":
    main()
