"""Does a model recognise the games it was trained on? States of Dlvl 5-12 no model saw, of train games the model had
other states of (fam) and of held-out players' games (new), each split by whether the game went to Dlvl 13 or deeper
(maxlvl): fam13 / new13, fam12 / new12. A model that knows a game by its fixed traits (named items, pets, fruit)
predicts the game's labels (end, ascends) on its unseen states better than on a new game's: the gap new13 - fam13
grows against a model that saw one state per band. 12 is the control (c17a and c17r saw the same states of those games).
  python fam_probe.py build human_dwa23 --seen human_dwa21/train.jsonl --states nao/val_dwa_states3.jsonl \
      --out human_fam23 [--per-group 5000]
  python predict.py ckpt/<model> human_fam23/<group>.jsonl preds/<model>-human_fam23-<group>.parquet  (each group)
  python fam_probe.py eval human_fam23 c17r c17a [--out ../results/fam_probe23.txt]
  python fam_probe.py robust human_fam23 c17r c17a [--per-game 5] [--drop-top 1] [--out ...]
  python fam_probe.py delta human_fam23 human_fam23m c17r c17a [--per-game 5] [--drop-top 1] [--out ...]
  python fam_probe.py player human_fam23 c17r c17a --states nao/val_dwa_states3.jsonl [--drop-top 1] [--out ...]
"""
import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from curve_eval import QS, boot_idx, crossfit, labels, nll

GROUPS = ("fam13", "new13", "fam12", "new12")
HEAD = re.compile(r'^\{"name": "([^"]*)".*?"start": (\d+), "end": \d+, "maxlvl": (\d+)')


def game(i):
    return i.split("#")[0]


def build(a):
    seen = {json.loads(line)["id"] for line in open(a.seen, encoding="utf-8")}
    games = {game(i) for i in seen}
    deep = set()
    for line in open(a.states, encoding="utf-8"):
        m = HEAD.match(line[:400])
        if m and int(m.group(3)) >= 13:
            deep.add(f"{m.group(1)}@{m.group(2)}")
    out = {g: [] for g in GROUPS}
    for split in ("train", "val", "test"):
        for line in open(Path(a.probe, f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            d = "13" if game(r["id"]) in deep else "12"
            if split != "train":
                out["new" + d].append(line)
            elif r["id"] not in seen and game(r["id"]) in games:
                out["fam" + d].append(line)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for g, lines in out.items():
        keep = sorted(lines, key=lambda x: zlib.crc32(json.loads(x)["id"].encode()))[: a.per_group]
        Path(a.out, f"{g}.jsonl").write_text("".join(keep), encoding="utf-8")
        print(f"{g}: {len(lines)} rows, kept {len(keep)}, games {len({game(json.loads(x)['id']) for x in keep})}")


def evaluate(a):
    loss, players = {}, {}
    for g in GROUPS:
        rows = []
        for line in open(Path(a.data, f"{g}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            rows.append({"id": r["id"], "player": r["id"].split("@")[0].lower(), "q": r["questions"], "gold": r["gold"]})
        for q in QS:
            sel, y, _ = labels(rows, q)
            players[g, q] = np.array([rows[i]["player"] for i in sel])
            for m in a.models:
                df = pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{g}.parquet")
                lg = df[df.question == q].set_index("id").logits
                z = np.stack([np.asarray(lg[rows[i]["id"]], float) for i in sel])
                loss[g, q, m, "raw"] = nll(z, y)
                loss[g, q, m, "T"], _ = crossfit(z, y, players[g, q])
    out = [f"{Path(a.data).name}: models {', '.join(a.models)}; log loss, T fitted within each group (crossfit by "
           f"player); gap = new - fam at the same maxlvl; 95 % bootstrap over players of both groups"]
    ref, other = a.models[0], a.models[1:]
    for q in QS:
        out.append(f"\n==== {q}")
        for g in GROUPS:
            out.append(f"  {g:6s} {len(players[g, q]):5d} rows, {len(set(players[g, q])):4d} players: " + "  ".join(
                f"{m} raw {loss[g, q, m, 'raw'].mean():.4f} T {loss[g, q, m, 'T'].mean():.4f}" for m in a.models))
        for d in ("13", "12"):
            g, n = "fam" + d, "new" + d
            bn, bg = boot_idx(players[n, q]), boot_idx(players[g, q], seed=1)

            def wmean(b, x, w):
                codes, _, cnt = b
                return (w * np.bincount(codes, weights=x, minlength=len(cnt))).sum() / (w * cnt).sum()

            for kind in ("raw", "T"):
                gaps = {m: loss[n, q, m, kind].mean() - loss[g, q, m, kind].mean() for m in a.models}
                line = f"  gap {n} - {g} ({kind}): " + "  ".join(f"{m} {gaps[m]:+.4f}" for m in a.models)
                for m in other:
                    v = [(wmean(bn, loss[n, q, m, kind], wn) - wmean(bg, loss[g, q, m, kind], wg))
                         - (wmean(bn, loss[n, q, ref, kind], wn) - wmean(bg, loss[g, q, ref, kind], wg))
                         for wn, wg in zip(bn[1], bg[1])]
                    lo, hi = np.percentile(v, [2.5, 97.5])
                    line += f"  | {m} - {ref}: {gaps[m] - gaps[ref]:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                out.append(line)
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


KINDS = ("raw", "Tnew")
SETS = ((QS, "mean of the four"), (("end", "ascends"), "end + ascends")) + tuple(((q,), q) for q in QS)


def load_rows(data, per_game, drop_top):
    rows = {}
    for g in GROUPS:
        per = {}
        for line in open(Path(data, f"{g}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            per.setdefault(game(r["id"]), []).append(r)
        rows[g] = [{"id": r["id"], "player": r["id"].split("@")[0].lower(), "q": r["questions"], "gold": r["gold"]}
                   for rs in per.values()
                   for r in sorted(rs, key=lambda r: zlib.crc32(r["id"].encode()))[: per_game or None]]
    top = pd.Series([r["player"] for g in ("fam13", "new13") for r in rows[g]]).value_counts().index[: drop_top]
    return {g: [r for r in rs if r["player"] not in set(top)] for g, rs in rows.items()}, list(top)


def boot_players(rows):
    """Weights of players: the first all ones, then 1000 bootstrap draws."""
    allp = sorted({r["player"] for rs in rows.values() for r in rs})
    rng = np.random.default_rng(0)
    ws = [np.ones(len(allp))] + [np.bincount(rng.integers(0, len(allp), len(allp)), minlength=len(allp))
                                 for _ in range(1000)]
    return {p: i for i, p in enumerate(allp)}, ws


def group_means(name, models, rows, pidx, ws):
    """mu[q, kind, model, group]: the group's mean log loss under every weight of ws. raw: as is; Tnew: T fitted on the
    new group of the same maxlvl and applied to fam too (a T fitted on fam would take off part of the overconfidence on
    known games)."""
    from scipy.optimize import minimize_scalar
    preds = {(m, g): pd.read_parquet(f"preds/{m}-{name}-{g}.parquet") for m in models for g in GROUPS}
    n, mu = len(next(iter(ws))), {}
    for q in QS:
        loss, codes = {}, {}
        for g in GROUPS:
            sel, y, _ = labels(rows[g], q)
            codes[g] = np.array([pidx[rows[g][i]["player"]] for i in sel])
            for m in models:
                lg = preds[m, g][preds[m, g].question == q].set_index("id").logits
                loss[g, m, "z"] = np.stack([np.asarray(lg[rows[g][i]["id"]], float) for i in sel])
                loss[g, m, "y"] = y
        for m in models:
            for d in ("13", "12"):
                zn, yn = loss["new" + d, m, "z"], loss["new" + d, m, "y"]
                t = minimize_scalar(lambda t: nll(zn, yn, t).mean(), bounds=(0.3, 5), method="bounded").x
                for g in ("fam" + d, "new" + d):
                    loss[g, m, "raw"] = nll(loss[g, m, "z"], loss[g, m, "y"])
                    loss[g, m, "Tnew"] = nll(loss[g, m, "z"], loss[g, m, "y"], t)
        cnts = {g: np.bincount(codes[g], minlength=n) for g in GROUPS}
        for (g, m, kind), v in loss.items():
            if kind in KINDS:
                s = np.bincount(codes[g], weights=v, minlength=n)
                mu[q, kind, m, g] = np.array([(w * s).sum() / (w * cnts[g]).sum() for w in ws])
    return mu


def triple(mu, q, kind, m):
    return (mu[q, kind, m, "new13"] - mu[q, kind, m, "fam13"]) - (mu[q, kind, m, "new12"] - mu[q, kind, m, "fam12"])


def robust(a):
    """The triple difference [(new13 - fam13) - (new12 - fam12)] of each model against the reference, one bootstrap of
    players over all four groups (a player of train can have games in fam13 and fam12), paired by model. raw: log loss
    as is; Tnew: see group_means. --per-game k: at most k rows of a game per group (new13 holds long games of few
    players); --drop-top n: without the n players with the most rows in fam13 + new13."""
    rows, top = load_rows(a.data, a.per_game, a.drop_top)
    out = [f"{Path(a.data).name}: models {', '.join(a.models)}; triple difference [(new13 - fam13) - (new12 - fam12)], "
           f"model - {a.models[0]}; per game {a.per_game or 'all'}, left out {top}; one bootstrap of players over "
           f"the four groups (1000), paired by model"]
    out.append("  rows: " + "  ".join(f"{g} {len(rows[g])} ({len({game(r['id']) for r in rows[g]})} games, "
                                      f"{len({r['player'] for r in rows[g]})} players)" for g in GROUPS))
    pidx, ws = boot_players(rows)
    mu = group_means(Path(a.data).name, a.models, rows, pidx, ws)
    tri = {(q, kind, m): triple(mu, q, kind, m) for q in QS for kind in KINDS for m in a.models}
    for qs, name in SETS:
        out.append(f"\n==== {name}")
        for kind in ("raw", "Tnew"):
            v = {m: np.mean([tri[q, kind, m] for q in qs], axis=0) for m in a.models}
            line = f"  {kind:4s} " + "  ".join(f"{m} {v[m][0]:+.4f}" for m in a.models)
            for m in a.models[1:]:
                dv = v[m] - v[a.models[0]]
                lo, hi = np.percentile(dv[1:], [2.5, 97.5])
                line += f"  | {m} - {a.models[0]}: {dv[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            out.append(line)
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


def delta(a):
    """The triple difference on the original set minus that on its masked copy (the same ids and labels, only the
    states changed), paired: one bootstrap of players for both. If the mask takes away what a model knows a game by,
    c17a - c17r shrinks on the masked copy. Then masked - original of each group's mean log loss: a mask that only hid
    such traits would leave the new groups (games no model saw) where they were."""
    rows, top = load_rows(a.data, a.per_game, a.drop_top)
    for g in GROUPS:
        masked = {json.loads(x)["id"]: json.loads(x)["gold"] for x in open(Path(a.masked, f"{g}.jsonl"), encoding="utf-8")}
        assert all(masked.get(r["id"]) == r["gold"] for r in rows[g]), f"{a.masked}/{g}: other rows or labels"
    pidx, ws = boot_players(rows)
    mo = group_means(Path(a.data).name, a.models, rows, pidx, ws)
    mm = group_means(Path(a.masked).name, a.models, rows, pidx, ws)
    ref, other = a.models[0], a.models[1:]
    out = [f"{Path(a.data).name} against its masked copy {Path(a.masked).name}: models {', '.join(a.models)}; per game "
           f"{a.per_game or 'all'}, left out {top}; one bootstrap of players (1000) for both sets and all models"]
    out.append("  rows: " + "  ".join(f"{g} {len(rows[g])} ({len({r['player'] for r in rows[g]})} players)"
                                      for g in GROUPS))

    def ci(v):
        lo, hi = np.percentile(v[1:], [2.5, 97.5])
        return f"{v[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]"

    for qs, name in SETS:
        out.append(f"\n==== {name}")
        for kind in KINDS:
            to = {m: np.mean([triple(mo, q, kind, m) for q in qs], axis=0) for m in a.models}
            tm = {m: np.mean([triple(mm, q, kind, m) for q in qs], axis=0) for m in a.models}
            out.append(f"  {kind:4s} TD original " + "  ".join(f"{m} {to[m][0]:+.4f}" for m in a.models)
                       + "  masked " + "  ".join(f"{m} {tm[m][0]:+.4f}" for m in a.models))
            for m in other:
                do, dm = to[m] - to[ref], tm[m] - tm[ref]
                out.append(f"       {m} - {ref}: original {ci(do)}  masked {ci(dm)}  original - masked {ci(do - dm)}")
            for m in a.models:
                out.append(f"       masked - original, {m}: " + "  ".join(
                    f"{g} {ci(np.mean([mm[q, kind, m, g] - mo[q, kind, m, g] for q in qs], axis=0))}"
                    for g in GROUPS))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


def player(a):
    """Game or player? fam12 / new12 split by whether the player has a game that went to Dlvl 13+ (P) or not (O). c17a
    saw no more states of the fam12 games than c17r did, only more of the same players' deep games: if it knows players
    (fruit is a player's option, pets' names travel between games), it gains on fam12P against fam12O. The new12 split
    takes off what P / O differ in otherwise (stronger players): [(fam12P - fam12O) - (new12P - new12O)], model - ref,
    one bootstrap of players; raw log loss and Tnew (T fitted on the whole new12, applied to all four parts)."""
    from scipy.optimize import minimize_scalar
    deep = set()
    for line in open(a.states, encoding="utf-8"):
        m = HEAD.match(line[:400])
        if m and int(m.group(3)) >= 13:
            deep.add(m.group(1).lower())
    load = {g: [json.loads(x) for x in open(Path(a.data, f"{g}.jsonl"), encoding="utf-8")] for g in GROUPS}
    top = pd.Series([r["id"].split("@")[0].lower() for g in ("fam13", "new13") for r in load[g]]).value_counts()
    top = set(top.index[: a.drop_top])
    parts, src = {}, {}
    for g in ("fam12", "new12"):
        for r in load[g]:
            p = r["id"].split("@")[0].lower()
            if p in top:
                continue
            k = g + ("P" if p in deep else "O")
            parts.setdefault(k, []).append({"id": r["id"], "player": p, "q": r["questions"], "gold": r["gold"]})
            src[k] = g
    names = ("fam12P", "fam12O", "new12P", "new12O")
    preds = {(m, g): pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{g}.parquet") for m in a.models
             for g in ("fam12", "new12")}
    out = [f"{Path(a.data).name}: models {', '.join(a.models)}; [(fam12P - fam12O) - (new12P - new12O)], P = the player "
           f"has a game to Dlvl 13+ ({len(deep)} players in {Path(a.states).name}); left out {sorted(top)}; one bootstrap "
           f"of players (1000), paired by model"]
    out.append("  rows: " + "  ".join(f"{k} {len(parts[k])} ({len({game(r['id']) for r in parts[k]})} games, "
                                      f"{len({r['player'] for r in parts[k]})} players)" for k in names))
    allp = sorted({r["player"] for rs in parts.values() for r in rs})
    pidx = {p: i for i, p in enumerate(allp)}
    rng = np.random.default_rng(0)
    ws = [np.ones(len(allp))] + [np.bincount(rng.integers(0, len(allp), len(allp)), minlength=len(allp))
                                 for _ in range(1000)]
    quad, dbl = {}, {}
    for q in QS:
        z, y, codes = {}, {}, {}
        for k in names:
            sel, y[k], _ = labels(parts[k], q)
            codes[k] = np.array([pidx[parts[k][i]["player"]] for i in sel])
            for m in a.models:
                pr = preds[m, src[k]]
                lg = pr[pr.question == q].set_index("id").logits
                z[k, m] = np.stack([np.asarray(lg[parts[k][i]["id"]], float) for i in sel])
        cnts = {k: np.bincount(codes[k], minlength=len(allp)) for k in names}
        for m in a.models:
            zn = np.concatenate([z["new12P", m], z["new12O", m]])
            yn = np.concatenate([y["new12P"], y["new12O"]])
            t = minimize_scalar(lambda t: nll(zn, yn, t).mean(), bounds=(0.3, 5), method="bounded").x
            for kind, tt in (("raw", 1.0), ("Tnew", t)):
                s = {k: np.bincount(codes[k], weights=nll(z[k, m], y[k], tt), minlength=len(allp)) for k in names}
                mu = [{k: (w * s[k]).sum() / (w * cnts[k]).sum() for k in names} for w in ws]
                quad[q, kind, m] = np.array([(u["fam12P"] - u["fam12O"]) - (u["new12P"] - u["new12O"]) for u in mu])
                dbl[q, kind, m] = np.array([u["fam12P"] - u["fam12O"] for u in mu])
    for qs, name in ((QS, "mean of the four"), (("end", "ascends"), "end + ascends")) + tuple(((q,), q) for q in QS):
        out.append(f"\n==== {name}")
        for what, d in (("quad", quad), ("fam12P - fam12O", dbl)):
            for kind in ("raw", "Tnew"):
                v = {m: np.mean([d[q, kind, m] for q in qs], axis=0) for m in a.models}
                line = f"  {what} {kind:4s} " + "  ".join(f"{m} {v[m][0]:+.4f}" for m in a.models)
                for m in a.models[1:]:
                    dv = v[m] - v[a.models[0]]
                    lo, hi = np.percentile(dv[1:], [2.5, 97.5])
                    line += f"  | {m} - {a.models[0]}: {dv[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                out.append(line)
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("player")
    p.add_argument("data")
    p.add_argument("models", nargs="+")
    p.add_argument("--states", required=True, help="the games file (maxlvl per game)")
    p.add_argument("--drop-top", type=int, default=0)
    p.add_argument("--out", default="")
    r = sub.add_parser("robust")
    r.add_argument("data")
    r.add_argument("models", nargs="+")
    r.add_argument("--per-game", type=int, default=0)
    r.add_argument("--drop-top", type=int, default=0)
    r.add_argument("--out", default="")
    d = sub.add_parser("delta")
    d.add_argument("data")
    d.add_argument("masked")
    d.add_argument("models", nargs="+")
    d.add_argument("--per-game", type=int, default=0)
    d.add_argument("--drop-top", type=int, default=0)
    d.add_argument("--out", default="")
    b = sub.add_parser("build")
    b.add_argument("probe")
    b.add_argument("--seen", required=True)
    b.add_argument("--states", required=True, help="the games file the set was built from (maxlvl per game)")
    b.add_argument("--out", required=True)
    b.add_argument("--per-group", type=int, default=6000)
    e = sub.add_parser("eval")
    e.add_argument("data")
    e.add_argument("models", nargs="+")
    e.add_argument("--out", default="")
    a = ap.parse_args()
    {"build": build, "eval": evaluate, "robust": robust, "delta": delta, "player": player}[a.cmd](a)


if __name__ == "__main__":
    main()
