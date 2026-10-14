"""The learning curve on one dataset: N models on the same held-out states (val + test
together; nothing is chosen on them). Per question (ttd, depth, end, ascends): log loss raw and
with a temperature fitted on the other half of the players (2-fold by player, 20 halvings, as c_probe.py), accuracy,
RPS for the ordered ones; differences to the reference model with a paired bootstrap over players; the noise pairs;
the decision rule fixed beforehand ("saturated" when |1/2 - all| with T < |1/4 - 1/4-B| on every question); slices
(strength in the question, ascended games, XL 8-14, Dlvl 5-10); L(N) = Linf + a N^-b over the nested shares.
  python curve_eval.py human_data16d all=c16:82188 half=c16g2:41263 third=c16g3:27619 quarter=c16g4:20693 \
      quarterB=c16g4b:20570 seed1=c16s1:82188 [--out ../results/curve_16d.txt]
  python curve_eval.py human_data16d --check     # random and perfect logits: the pipeline itself
"""
import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar

QS = ("ttd", "depth", "end", "ascends")
ORDERED = ("ttd", "depth")
BANDS = ((1, 4), (5, 12), (13, 20), (21, 24), (25, 29), (30, 99))  # human_build --per-band


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(z, y, t=1.0):
    p = softmax(z / t)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def rps(z, y):
    p, k = softmax(z), z.shape[1]
    return ((p.cumsum(1) - (np.arange(k) >= y[:, None]))[:, :-1] ** 2).sum(1) / (k - 1)


def crossfit(z, y, players, n=20):
    out, ts = np.zeros(len(y)), []
    for s in range(n):
        half = np.array([zlib.crc32(f"{s}:{x}".encode()) % 2 for x in players])
        for h in (0, 1):
            m = half != h
            t = minimize_scalar(lambda t: nll(z[m], y[m], t).mean(), bounds=(0.3, 5), method="bounded").x
            out[half == h] += nll(z[half == h], y[half == h], t)
            ts.append(t)
    return out / n, float(np.median(ts))


def boot_idx(players, n=1000, seed=0):
    codes, uniq = pd.factorize(players)
    rng = np.random.default_rng(seed)
    cnt = np.bincount(codes, minlength=len(uniq))
    return codes, [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(n)], cnt


def ci(x, b):
    codes, ws, cnt = b
    s = np.bincount(codes, weights=x, minlength=len(cnt))
    v = [(w * s).sum() / (w * cnt).sum() for w in ws]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"{x.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]" if x.mean() < 0 or lo < 0 else f"{x.mean():.4f} [{lo:.4f}, {hi:.4f}]"


def load_rows(data):
    rows = []
    for split in ("val", "test"):
        for line in open(Path(data, f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            ins = next(iter(r["questions"].values()))["instructions"]
            m = re.search(r"experience (\d)/\d, result (\d)/\d", ins)
            xl = re.search(r"\bXL (\d+)", r["state"])
            dl = re.search(r"\bDlvl (\d+)", r["state"])
            dp = re.search(r"\bdeepest (\d+)", r["state"])
            rows.append({"id": r["id"], "player": r["id"].split("@")[0].lower(), "split": split,
                         "q": r["questions"], "gold": r["gold"], "exp": int(m.group(1)) if m else 0,
                         "res": int(m.group(2)) if m else 0, "asc": r["gold"]["ascends"]["label"] == "true",
                         "xl": int(xl.group(1)) if xl else -1, "dlvl": int(dl.group(1)) if dl else -1,
                         "deepest": int(dp.group(1)) if dp else -1})
    return rows


def labels(rows, q):
    sel = [i for i, r in enumerate(rows) if q in r["gold"]]
    crit = rows[sel[0]]["q"][q]["criteria"]
    keys = list(crit) if isinstance(crit, dict) else [str(k) for k in range(len(crit))]
    return np.array(sel), np.array([keys.index(rows[i]["gold"][q]["label"]) for i in sel]), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("models", nargs="*", help="label=model:train_rows; the first is the reference")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = load_rows(a.data)
    players = np.array([r["player"] for r in rows])
    rng = np.random.default_rng(0)
    if a.check:
        specs = [("perfect", None, 1), ("random", None, 1), ("random2", None, 1)]
    else:
        specs = [(lab, *ms.split(":")) for lab, ms in (m.split("=", 1) for m in a.models)]
        specs = [(lab, m, int(n)) for lab, m, n in specs]
    logits = {}
    for lab, m, _ in specs:
        if m is None:
            continue
        df = pd.concat([pd.read_parquet(f"preds/{m}-{Path(a.data).name}-{s}.parquet") for s in ("val", "test")])
        logits[lab] = df.set_index(["id", "question"]).logits
    out = [f"{len(rows)} held-out states (val + test), {len(set(players))} players; models: "
           + ", ".join(f"{lab}={m}({n} train rows)" for lab, m, n in specs)]
    ref = specs[0][0]
    res = {}  # (label, q) -> raw loss, T loss, T
    for q in QS:
        sel, y, k = labels(rows, q)
        pl = players[sel]
        b = boot_idx(pl)
        out.append(f"\n== {q}: {len(y)} rows, {k} options")
        for lab, _, _ in specs:
            if lab == "perfect":
                z = np.eye(k)[y] * 20.0
            elif lab.startswith("random"):
                z = rng.normal(size=(len(y), k))
            else:
                z = np.stack([np.asarray(logits[lab][(rows[i]["id"], q)], float) for i in sel])
            raw = nll(z, y)
            tl, t = crossfit(z, y, pl)
            res[(lab, q)] = (raw, tl, t, z)
            extra = f"  RPS {ci(rps(z, y), b)}" if q in ORDERED else ""
            out.append(f"  {lab:9s} raw {ci(raw, b)}  with T {ci(tl, b)} (T {t:.2f})  acc {(z.argmax(1) == y).mean():.3f}"
                       + extra)
        for lab, _, _ in specs[1:]:
            out.append(f"  {lab:9s} minus {ref}: raw {ci(res[(lab, q)][0] - res[(ref, q)][0], b)}  "
                       f"with T {ci(res[(lab, q)][1] - res[(ref, q)][1], b)}")
        labs = {s[0] for s in specs}
        # stages 2-3: "halfcap" vs "half" (same rows, cap 20 per player), "halfx2" / "thirdx3e1" vs "half" /
        # "third" (the same games, more epochs)
        extra_pairs = [(lab, lab[:-3]) for lab in labs if lab.endswith("cap")] \
            + [(lab, lab.split("x", 1)[0]) for lab in labs if "x" in lab and lab != "max"]
        for x, y2 in (("quarter", "quarterB"), ("seed1", "all"), ("half", "all"), *sorted(extra_pairs)):
            if x in labs and y2 in labs:
                out.append(f"  {x} minus {y2}: with T {ci(res[(x, q)][1] - res[(y2, q)][1], b)}")
    labs = {s[0] for s in specs}
    if {"half", "all", "quarter", "quarterB"} <= labs:
        out.append("\n== decision rule (with T): |half - all| < |quarter - quarterB| on every question -> saturated")
        verdict = True
        for q in QS:
            d1 = abs(res[("half", q)][1].mean() - res[("all", q)][1].mean())
            d0 = abs(res[("quarter", q)][1].mean() - res[("quarterB", q)][1].mean())
            verdict &= d1 < d0
            out.append(f"  {q:8s} |half - all| {d1:.4f}  |quarter - quarterB| {d0:.4f}  {'<' if d1 < d0 else '>='}")
        out.append(f"  -> {'saturated' if verdict else 'not saturated (by this rule)'}")
    out.append("\n== slices, mean loss with T")
    slices = [("all rows", lambda r: True), ("no history", lambda r: r["exp"] == 0)] \
        + [(f"experience {v}", lambda r, v=v: r["exp"] == v) for v in range(1, 6)] \
        + [(f"result {v}", lambda r, v=v: r["res"] == v) for v in range(1, 6)] \
        + [("ascended games", lambda r: r["asc"]), ("XL 8-14", lambda r: 8 <= r["xl"] <= 14),
           ("Dlvl 5-10", lambda r: 5 <= r["dlvl"] <= 10)] \
        + [(f"Dlvl {lo}-{hi}", lambda r, lo=lo, hi=hi: lo <= r["dlvl"] <= hi)
           for lo, hi in ((1, 2), (3, 4), (5, 7), (8, 12), (13, 20), (21, 99), (21, 24), (25, 29), (30, 99))]
    for q in QS:
        sel, _, _ = labels(rows, q)
        out.append(f"  {q}: " + " | ".join(lab for lab, _, _ in specs))
        for name, f in slices:
            m = np.array([f(rows[i]) for i in sel])
            if m.sum():
                out.append(f"    {name:15s} n {m.sum():5d}  " + "  ".join(f"{res[(lab, q)][1][m].mean():.4f}"
                                                                      for lab, _, _ in specs))
    # the bands where the bot dies (deep_counts.py), with intervals -- the slices above have none. A game's
    # states share its fate (n_eff ~ games), so the bootstrap is over the band's players
    out.append(f"\n== Dlvl bands (the state's Dlvl): minus {ref}, with T, 95 % bootstrap over the band's players")
    for q in QS:
        sel, _, _ = labels(rows, q)
        pl = players[sel]
        out.append(f"  {q}")
        for lo, hi in BANDS:
            m = np.array([lo <= rows[i]["dlvl"] <= hi for i in sel])
            if m.sum() < 2:
                continue
            b = boot_idx(pl[m])
            base = res[(ref, q)][1][m]
            games = len({rows[i]["id"].split("#")[0] for i in sel[m]})
            out.append(f"    Dlvl {lo}-{hi}: rows {m.sum()} games {games} players {len(set(pl[m]))}  {ref} "
                       f"{base.mean():.4f}" + "".join(f" | {lab} {ci(res[(lab, q)][1][m] - base, b)}"
                                                       for lab, _, _ in specs[1:]))
    nested = [s for s in specs if s[0] in ("quarter", "third", "half", "all")]
    if len(nested) >= 4:
        out.append("\n== L(N) = Linf + a N^-b on the nested shares (with T; 4 points, 3 parameters: a rough guide)")
        n = np.array([s[2] for s in nested], float)
        for q in QS:
            L = np.array([res[(s[0], q)][1].mean() for s in nested])
            try:
                (li, aa, bb), _ = curve_fit(lambda x, li, aa, bb: li + aa * x ** -bb, n / n.max(), L,
                                            p0=[L.min() - 0.01, 0.01, 0.5], bounds=([0, 0, 0.01], [10, 10, 5]),
                                            maxfev=20000)
                pred = [li + aa * x ** -bb for x in (2.0, 4.0)]
                out.append(f"  {q:8s} points " + " ".join(f"{x:.4f}" for x in L) + f"  Linf {li:.4f} a {aa:.4f} b {bb:.2f}"
                           f"  at 2x data {pred[0]:.4f}, 4x {pred[1]:.4f}")
            except RuntimeError as e:
                out.append(f"  {q:8s} fit failed: {e}")
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
