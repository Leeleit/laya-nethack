"""How much of a change's effect on the gradient boosting shows up on laya: for each change measured on both, the pair
(GBM effect, laya effect) of every cell question x Dlvl band 13+ (T-band, from the saved result files), and per
question the slope of laya on GBM through the origin, with a bootstrap over the cells.
  python transfer.py [--res ../results] [--out ../results/transfer.txt]
"""
import argparse
import re
from pathlib import Path

import numpy as np

QS = ("ttd", "depth", "end", "ascends")
DEEP = ("13-20", "21-24", "25-29", "30+")


def pairs(path):
    """deep_checks.py preds output -> {(q, band, 'x - r'): T-band}"""
    out, q, band = {}, None, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if m := re.match(r"==== (\w+):", line):
            q = m.group(1)
        elif m := re.match(r"  Dlvl (\S+): ", line):
            band = m.group(1)
        elif m := re.match(r"    (\w+ - \w+): raw .* T-band ([-+]?[0-9.]+)", line):
            out[q, band, m.group(1)] = float(m.group(2))
    return out


def seed_avg(path):
    """seed_avg.py output -> {(q, band): D T-band}"""
    out, q = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if m := re.match(r"==== (\w+)", line):
            q = m.group(1)
        elif m := re.match(r"(\S+)\s+([-+]?[0-9.]+) \[", line):
            out[q, m.group(1)] = float(m.group(2))
    return out


def soft_cv(path):
    """soft_variants.py --cv output -> {(band, variant): variant - hard, T-band}; band names as in deep_checks"""
    out, band = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if m := re.match(r"Dlvl (\d+)-(\d+):", line):
            band = f"{m.group(1)}+" if m.group(2) == "99" else f"{m.group(1)}-{m.group(2)}"
        elif m := re.search(r"\| (\w+) - hard: raw .* T-band ([-+]?[0-9.]+)", line):
            out[band, m.group(1)] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="../results")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = Path(a.res)
    rows = []  # (change, q, band, gbm, laya)
    g25, av = pairs(r / "gbm25_drop1.txt"), seed_avg(r / "avg25_drop1.txt")
    g20, m20 = pairs(r / "gbm20_drop1.txt"), pairs(r / "mix20_drop1.txt")
    g24, e20 = pairs(r / "gbm24s_drop1.txt"), pairs(r / "early20_drop1.txt")
    cv, n20 = soft_cv(r / "soft_variants20cv.txt"), pairs(r / "narrow20_drop1.txt")
    g28, t28 = pairs(r / "gbm28_drop1.txt"), pairs(r / "c17tx_drop1.txt")
    for q in QS:
        for b in DEEP:
            # mix: mean of two GBM seeds each side, against laya's two-seed D
            g = (g25[q, b, "mixe - early"] + g25[q, b, "mixe1 - early"] - g25[q, b, "early1 - early"]) / 2
            rows.append(("mix (m20e vs c17e)", q, b, g, av[q, b]))
            rows.append(("other roles on rates (m17r vs c17r)", q, b, g20[q, b, "mixr - rate"], m20[q, b, "mixr - rate"]))
            g = (g24[q, b, "early - rate"] + g24[q, b, "early1 - rate"] - g24[q, b, "rate1 - rate"]) / 2
            rows.append(("early games (c17e vs c17n)", q, b, g, e20[q, b, "early - narrow"]))
            if q == "depth":  # the GBM side is a cross-validation on train, not the test
                rows.append(("depth target narrow vs orig (GBM: CV on train)", q, b,
                             cv[b, "narrow"] - cv[b, "orig"], n20[q, b, "narrow - rate"]))
            if q == "ttd":  # the other questions: the same targets, the GBM side exactly 0
                g = (g28[q, b, "flat - orig"] + g28[q, b, "flat1 - orig"] - g28[q, b, "orig1 - orig"]) / 2
                lay = (t28[q, b, "flat - orig"] + t28[q, b, "flat1 - orig"]) / 2 - sum(
                    t28[q, b, f"orig{k} - orig"] for k in (1, 2, 3)) / 4
                rows.append(("ttd target flat vs orig (c17t vs c17e)", q, b, g, lay))
    out = ["per cell (T-band, without the top player): GBM effect -> laya effect"]
    for ch in dict.fromkeys(x[0] for x in rows):
        out.append(f"\n  {ch}")
        for _, q, b, g, l in (x for x in rows if x[0] == ch):
            out.append(f"    {q:8s} {b:6s} GBM {g:+.4f}  laya {l:+.4f}")
    rng = np.random.default_rng(0)

    def slope(sub):
        g, l = np.array([x[3] for x in sub]), np.array([x[4] for x in sub])
        s = (g * l).sum() / (g * g).sum()
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(g), len(g))
            if (g[i] ** 2).sum() > 0:
                bs.append((g[i] * l[i]).sum() / (g[i] ** 2).sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return f"slope {s:+.2f} [{lo:+.2f}, {hi:+.2f}]  ({len(g)} cells; sign agrees in {(np.sign(g) == np.sign(l)).sum()})"

    out.append("\nslope of laya on GBM through the origin, bootstrap over cells")
    for q in QS:
        out.append(f"  {q:8s} {slope([x for x in rows if x[1] == q])}")
    out.append(f"  {'depth + end':8s} {slope([x for x in rows if x[1] in ('depth', 'end')])}")
    out.append(f"  {'all':8s} {slope(rows)}")
    data = [x for x in rows if not x[0].startswith(("depth target", "ttd target"))]
    out.append("\nthe same for the data changes only (mix, other roles on rates, early games)")
    for q in QS:
        out.append(f"  {q:8s} {slope([x for x in data if x[1] == q])}")
    out.append(f"  {'depth + end':8s} {slope([x for x in data if x[1] in ('depth', 'end')])}")
    out.append(f"  {'all':8s} {slope(data)}")
    out.append("\nthe mix changes only (mix, other roles on rates), cells 21-24 and 25-29")
    mix = [x for x in data if not x[0].startswith("early") and x[2] in ("21-24", "25-29")]
    for q in QS:
        out.append(f"  {q:8s} {slope([x for x in mix if x[1] == q])}")
    out.append(f"  {'all':8s} {slope(mix)}")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
