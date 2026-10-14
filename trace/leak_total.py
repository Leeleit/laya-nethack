"""How much of a teacher's output on its own train rows is the label (CPU, from ready predictions), over all bands.
  Delta_total = [teacher(train) - teacher(test)] - [ref(train, out of fold) - ref(test)]
raw log loss per (question, band), the mean over each set's rows, bootstrap over players (train and test players are
disjoint). Read as a mix q = (1 - beta) p + beta e_y: beta = p (e^-Delta - 1) / (1 - p), p = exp(-(L_train - Delta)) (the
teacher's geometric-mean p_y on its train rows without the leak); beta_bar = mean over the 8 cells of G (21-29).
Next to it:
  beta in probabilities: delta = [mean p_y train - test]_teacher - [same]_ref, beta = delta / (1 - (mean p_y train -
          delta)) (exp(-L) is the geometric mean of p_y, which lowers beta most where p_y spreads, as ascends);
  pair    beta from the paired rows of mem_gate.py (seen against unseen rows of the same game, the game weighted by its seen
          rows): (q_seen - q_unseen) / (1 - q_unseen) of the mean p_y, 21-29 (a lower bound: unseen rows near seen ones
          are recognised too);
  p_y     the teacher's mean p_y, train against test; L1 between the teacher and the old target on train;
  E       ensemble gain (G of the mean probabilities - mean of the seeds' G) per band, train against test;
  tail    the share of rows with p_y > 0.5 / 0.9, train against test;
  gold    calibration in the large on train: the observed share of ascended / no deeper / ttd <= 1000 against the mean
          old target, teacher and distilled gold; per option, mean teacher - mean old target (band x bin or class).
  python leak_total.py --train human_dwa24 --test human_dwa20t --teacher c17e c17es1 c17es2 c17es3 --ref gbmcf \
      --distilled human_dwa29d --pair human_mem29 --drop-top 1 --out ../results/leak29.txt
predictions: preds/<m>-<train>-train.parquet, preds/<m>-<test>-{val,test}.parquet for the teacher's seeds and the ref
(gbm_cf.py --train-oof), preds/<m>-<pair>-{seen,unseen}.parquet for the seeds."""
import argparse
import bisect
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from curve_eval import QS, softmax
from distill_build import keys_of
from mem_gate import game, rows_of

BANDS = (4, 12, 20, 24, 29)
NAMES = ("1-4", "5-12", "13-20", "21-24", "25-29", "30+")
GB = (3, 4)
TEXTS = {}  # q -> option texts in logit order (list(criteria), as deep_checks.py reads them)


def load(paths, targets=False):
    """-> {q: dict(ids, players, bands, y[, old])}, option keys per q (logit order), labels' keys per q"""
    out = {q: collections.defaultdict(list) for q in QS}
    keys, lkeys = {}, {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            dl = re.search(r"\bDlvl (\d+)", r["state"])
            if not dl:
                continue
            b = bisect.bisect_left(BANDS, int(dl.group(1)))
            for q in QS:
                if q not in r["gold"]:
                    continue
                if q not in keys:
                    crit = r["questions"][q]["criteria"]
                    keys[q] = keys_of(r["questions"][q])
                    lkeys[q] = list(crit) if isinstance(crit, dict) else [str(k) for k in range(len(crit))]
                    TEXTS.setdefault(q, [str(c) for c in crit])
                d = out[q]
                d["ids"].append(r["id"])
                d["players"].append(r["id"].split("@")[0].lower())
                d["bands"].append(b)
                d["dlvl"].append(int(dl.group(1)))
                d["y"].append(lkeys[q].index(r["gold"][q]["label"]))
                if targets:
                    pr = r["gold"][q]["probabilities"]
                    d["old"].append([pr.get(k, 0.0) for k in keys[q]])
    res = {}
    for q, d in out.items():
        res[q] = {k: np.array(v) if k != "ids" and k != "players" else v for k, v in d.items()}
        if targets:
            o = res[q]["old"]
            res[q]["old"] = o / o.sum(1, keepdims=True)
    return res, keys, lkeys


def probs(paths, q, ids):
    df = pd.concat([pd.read_parquet(p) for p in paths])
    lg = df[df.question == q].set_index("id").logits
    lg = lg[~lg.index.duplicated()]
    got = lg.reindex(ids)
    assert got.notna().all(), f"{paths[0]} {q}: {int(got.isna().sum())} of {len(ids)} rows without a prediction"
    return softmax(np.stack([np.asarray(x, float) for x in got]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--teacher", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--distilled", default="")
    ap.add_argument("--pair", default="")
    ap.add_argument("--drop-top", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tn, sn = Path(a.train).name, Path(a.test).name
    sets = {"train": load([Path(a.train, "train.jsonl")], targets=True),
            "test": load([Path(a.test, f"{s}.jsonl") for s in ("val", "test")])}
    for q in QS:
        assert sets["train"][1][q] == sets["test"][1][q] == sets["train"][2][q], f"{q}: option keys differ"
    data = {s: v[0] for s, v in sets.items()}
    keys = sets["train"][1]
    head = [f"teacher = mean softmax of {' '.join(a.teacher)}; reference {a.ref} (train rows out of fold)"]
    if a.drop_top:  # as deep_checks.py: the held-out player(s) with the most ascended rows
        d = data["test"]["ascends"]
        c = collections.Counter(p for p, y in zip(d["players"], d["y"]) if keys["ascends"][y] == "true")
        top = {p for p, _ in c.most_common(a.drop_top)}
        for q in QS:
            d = data["test"][q]
            k = np.array([p not in top for p in d["players"]])
            data["test"][q] = {f: (np.asarray(v)[k] if not isinstance(v, list) else [x for x, kk in zip(v, k) if kk])
                               for f, v in d.items()}
        head.append(f"test without the {a.drop_top} player(s) with the most ascended rows")
    tp = {p for d in data["train"].values() for p in d["players"]}
    sp = {p for d in data["test"].values() for p in d["players"]}
    assert not tp & sp, "a test player among the train players"
    players = sorted(tp | sp)
    pidx = {p: i for i, p in enumerate(players)}
    rng = np.random.default_rng(0)
    ws = np.array([np.ones(len(players))] + [np.bincount(rng.integers(0, len(players), len(players)),
                                                         minlength=len(players)) for _ in range(1000)])
    files = {"train": lambda m: [f"preds/{m}-{tn}-train.parquet"],
             "test": lambda m: [f"preds/{m}-{sn}-{s}.parquet" for s in ("val", "test")]}
    dist = load([Path(a.distilled, "train.jsonl")], targets=True)[0] if a.distilled else None

    def cellmean(v, d, b):  # mean of v over the set's rows in band b, under every bootstrap weight
        k = d["bands"] == b
        codes = np.array([pidx[p] for p, kk in zip(d["players"], k) if kk])
        s = np.bincount(codes, weights=v[k], minlength=len(players))
        n = np.bincount(codes, minlength=len(players))
        return (ws * s).sum(1) / np.maximum((ws * n).sum(1), 1e-12)

    def fmt(v, f="+.4f"):
        lo, hi = np.percentile(v[1:], [2.5, 97.5])
        return f"{v[0]:{f}} [{lo:{f}}, {hi:{f}}]"

    L, PY, PYR, TAIL, L1, GOLD, SEEDL, CAL = {}, {}, {}, {}, {}, {}, {}, {}
    for q in QS:
        for s in data:
            d = data[s][q]
            n = len(d["y"])
            ps = [probs(files[s](m), q, d["ids"]) for m in a.teacher]
            pt = np.mean(ps, 0)
            pr = probs(files[s](a.ref), q, d["ids"])
            ar = np.arange(n)
            py = pt[ar, d["y"]]
            for b in range(6):
                if not (d["bands"] == b).any():
                    continue
                k = d["bands"] == b
                L[s, "teacher", q, b] = cellmean(-np.log(np.clip(py, 1e-12, 1)), d, b)
                L[s, "ref", q, b] = cellmean(-np.log(np.clip(pr[ar, d["y"]], 1e-12, 1)), d, b)
                for m, p in zip(a.teacher, ps):
                    SEEDL[s, m, q, b] = float(-np.log(np.clip(p[ar, d["y"]][k], 1e-12, 1)).mean())
                PY[s, q, b] = cellmean(py, d, b)
                PYR[s, q, b] = cellmean(pr[ar, d["y"]], d, b)
                TAIL[s, q, b] = ((py[k] > 0.5).mean(), (py[k] > 0.9).mean())
                if s == "train":
                    L1[q, b] = float(np.abs(pt[k] - d["old"][k]).sum(1).mean())
                    CAL[q, b] = (pt[k].mean(0), d["old"][k].mean(0),
                                 np.bincount(d["y"][k], minlength=pt.shape[1]) / k.sum())
                    if q in ("ttd", "depth", "ascends"):
                        hi = (next(i for i, kk in enumerate(TEXTS[q]) if kk.split("-")[-1].split()[-1] == "1000") + 1
                              if q == "ttd" else 1)
                        ev = (lambda p: p[:, :hi].sum(1)) if q != "ascends" else (lambda p: p[:, keys[q].index("true")])
                        yy = d["y"][k]
                        obs = (yy < hi).mean() if q != "ascends" else (yy == keys[q].index("true")).mean()
                        g = dist[q] if dist else None
                        if g is not None:
                            assert g["ids"] == d["ids"], "the distilled set's rows differ"
                        GOLD[q, b] = (obs, ev(d["old"][k]).mean(), ev(pt[k]).mean(),
                                      ev(g["old"][k]).mean() if g is not None else float("nan"))
            print(f"{q} {s}: {n} rows", flush=True)
    out = head + [f"rows train / test: " + ", ".join(f"{q} {len(data['train'][q]['y'])} / {len(data['test'][q]['y'])}"
                                                    for q in QS),
                  "rows per band train / test (ttd): " + ", ".join(
                      f"{NAMES[b]} {(data['train']['ttd']['bands'] == b).sum()} / {(data['test']['ttd']['bands'] == b).sum()}"
                      for b in range(6))]
    out.append("\n==== Delta_total = [teacher(train) - teacher(test)] - [ref(train) - ref(test)], raw log loss; "
               "beta = p (e^-Delta - 1) / (1 - p), p = exp(-(L_train - Delta))")
    out.append(f"  {'cell':<16}{'teacher tr / te':>18}{'ref tr / te':>18}{'Delta_total':>30}{'p':>7}{'beta':>28}")
    betas = {}
    for q in QS:
        for b in range(6):
            if ("test", "teacher", q, b) not in L or ("train", "teacher", q, b) not in L:
                continue
            tt, ts = L["train", "teacher", q, b], L["test", "teacher", q, b]
            rt, rs = L["train", "ref", q, b], L["test", "ref", q, b]
            dlt = (tt - ts) - (rt - rs)
            p = np.exp(-(tt - dlt))
            beta = p * (np.exp(-dlt) - 1) / (1 - p)
            betas[q, b] = beta
            out.append(f"  {q + ' ' + NAMES[b]:<16}{tt[0]:9.4f}/{ts[0]:<8.4f}{rt[0]:9.4f}/{rs[0]:<8.4f}{fmt(dlt):>30}"
                       f"{p[0]:7.3f}{fmt(beta, '+.3f'):>28}")
    bb = np.mean([betas[q, b] for q in QS for b in GB], 0)
    out.append(f"  beta_bar over the 8 cells of G (21-29): {fmt(bb, '+.3f')}"
               + ("  -> above 0.15" if bb[0] > 0.15 else "  -> not above 0.15"))
    for name, bs in (("1-20", (0, 1, 2)), ("all bands", range(6))):
        v = np.mean([betas[q, b] for q in QS for b in bs if (q, b) in betas], 0)
        out.append(f"  beta mean over {name}: {fmt(v, '+.3f')}")
    out.append("  per question, mean over 21-29: " + ", ".join(
        f"{q} {np.mean([betas[q, b][0] for b in GB]):+.3f}" for q in QS))
    # the same mix read in probabilities (log loss through exp(-L) is the geometric mean of p_y: Jensen lowers beta)
    out.append("\n==== beta in probabilities: delta = [mean p_y train - test]_teacher - [same]_ref, p_cf = mean p_y train "
               "- delta, beta = delta / (1 - p_cf)")
    pbetas = {}
    for q in QS:
        for b in range(6):
            if ("test", q, b) not in PY or ("train", q, b) not in PY:
                continue
            dl = (PY["train", q, b] - PY["test", q, b]) - (PYR["train", q, b] - PYR["test", q, b])
            pcf = PY["train", q, b] - dl
            beta = dl / (1 - pcf)
            pbetas[q, b] = beta
            out.append(f"  {q + ' ' + NAMES[b]:<16}p_y teacher {PY['train', q, b][0]:.4f}/{PY['test', q, b][0]:.4f}  "
                       f"ref {PYR['train', q, b][0]:.4f}/{PYR['test', q, b][0]:.4f}  delta {fmt(dl)}  "
                       f"p_cf {pcf[0]:.3f}  beta {fmt(beta, '+.3f')}")
    pb = np.mean([pbetas[q, b] for q in QS for b in GB], 0)
    out.append(f"  beta_bar over the 8 cells of G (21-29): {fmt(pb, '+.3f')}"
               + ("  -> above 0.15" if pb[0] > 0.15 else "  -> not above 0.15"))
    out.append("  per question, mean over 21-29: " + ", ".join(
        f"{q} {np.mean([pbetas[q, b][0] for b in GB]):+.3f}" for q in QS))

    if a.pair:  # the paired seen / unseen rows of mem_gate.py
        pn = Path(a.pair).name
        seen, unseen = rows_of(Path(a.pair, "seen.jsonl")), rows_of(Path(a.pair, "unseen.jsonl"))
        games = sorted({(game(r["id"]), r["band"]) for r in seen} & {(game(r["id"]), r["band"]) for r in unseen})
        gidx = {g: i for i, g in enumerate(games)}
        gpl = np.array([pidx.get(g[0].split("@")[0].lower(), -1) for g in games])
        assert (gpl >= 0).all(), "a paired game of a player outside train"
        gband = np.array([g[1] for g in games])
        out.append(f"\n==== pair: beta = (p_y seen - p_y unseen) / (1 - p_y unseen) of the paired means "
                   f"({(gband == 3).sum()} / {(gband == 4).sum()} games in 21-24 / 25-29)")
        pb = []
        for q in QS:
            sums = {}
            for g, rs in (("seen", seen), ("unseen", unseen)):
                ids = [r["id"] for r in rs if q in r["gold"]]
                rr = [r for r in rs if q in r["gold"]]
                pt = np.mean([probs([f"preds/{m}-{pn}-{g}.parquet"], q, ids) for m in a.teacher], 0)
                y = np.array([keys[q].index(r["gold"][q]["label"]) for r in rr])
                py = pt[np.arange(len(y)), y]
                s, c = np.zeros(len(games)), np.zeros(len(games))
                for r, v in zip(rr, py):
                    j = gidx.get((game(r["id"]), r["band"]))
                    if j is not None:
                        s[j] += v
                        c[j] += 1
                sums[g] = (s, c)
            (ss, cs), (su, cu) = sums["seen"], sums["unseen"]
            ok = (cs > 0) & (cu > 0)
            for b in GB:
                k = ok & (gband == b)
                w = cs[k]
                num_s = np.bincount(gpl[k], weights=w * ss[k] / cs[k], minlength=len(players))
                num_u = np.bincount(gpl[k], weights=w * su[k] / cu[k], minlength=len(players))
                den = np.bincount(gpl[k], weights=w, minlength=len(players))
                qs_, qu_ = (ws * num_s).sum(1) / (ws * den).sum(1), (ws * num_u).sum(1) / (ws * den).sum(1)
                beta = (qs_ - qu_) / (1 - qu_)
                pb.append(beta)
                out.append(f"  {q + ' ' + NAMES[b]:<16}p_y seen {qs_[0]:.4f}  unseen {qu_[0]:.4f}  beta {fmt(beta, '+.3f')}")
        out.append(f"  beta_bar (8 cells): {fmt(np.mean(pb, 0), '+.3f')}")

    out.append("\n==== p_y (teacher's mean probability of the label), train / test; L1(teacher, old target) on train; "
               "share of rows with p_y > 0.5 / > 0.9, train / test")
    for q in QS:
        for b in range(6):
            if ("test", q, b) not in PY:
                continue
            out.append(f"  {q + ' ' + NAMES[b]:<16}p_y {PY['train', q, b][0]:.3f} / {PY['test', q, b][0]:.3f}  "
                       f"L1 {L1[q, b]:.3f}  >0.5 {TAIL['train', q, b][0]:.3f} / {TAIL['test', q, b][0]:.3f}  "
                       f">0.9 {TAIL['train', q, b][1]:.3f} / {TAIL['test', q, b][1]:.3f}")
    out.append("\n==== ensemble gain E = G(mean probabilities) - mean of the seeds' G, per band (G: mean over the 4 "
               "questions), train / test")
    for b in range(6):
        if any(("test", "teacher", q, b) not in L for q in QS):
            continue
        e = {}
        for s in ("train", "test"):
            ge = np.mean([L[s, "teacher", q, b][0] for q in QS])
            gs = np.mean([np.mean([SEEDL[s, m, q, b] for m in a.teacher]) for q in QS])
            e[s] = ge - gs
        out.append(f"  {NAMES[b]:<8}E train {e['train']:+.4f}  test {e['test']:+.4f}  ratio "
                   f"{e['train'] / e['test'] if e['test'] < 0 else float('nan'):.2f}")
    out.append("\n==== calibration in the large on train: observed share / mean old target / teacher / distilled gold")
    for q, ev in (("ascends", "P(ascended)"), ("depth", "P(no deeper)"), ("ttd", "P(ttd <= 1000)")):
        for b in range(6):
            if (q, b) in GOLD:
                o, t0, t1, t2 = GOLD[q, b]
                out.append(f"  {ev + ' ' + NAMES[b]:<22}{o:.3f} / {t0:.3f} / {t1:.3f} / {t2:.3f}")
    out.append("\n==== calibration map on train: mean teacher - mean old target per option (x1000); observed share - old "
               "target next to the largest; options in logit order")
    for q in QS:
        for b in range(6):
            if (q, b) not in CAL:
                continue
            t, o, obs = CAL[q, b]
            dlt = t - o
            j = int(np.argmax(np.abs(dlt)))
            line = (f"  {q + ' ' + NAMES[b]:<16}max |d| {dlt[j]:+.4f} at {TEXTS[q][j][:28]!r} (observed - old "
                    f"{obs[j] - o[j]:+.4f}); sum |d| {np.abs(dlt).sum():.3f}")
            out.append(line)
            if len(dlt) <= 21:
                out.append("    " + " ".join(f"{1000 * x:+.0f}" for x in dlt))
            else:  # end: the five largest
                top = np.argsort(-np.abs(dlt))[:5]
                out.append("    " + "; ".join(f"{TEXTS[q][i][:24]} {1000 * dlt[i]:+.0f}" for i in top))
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
