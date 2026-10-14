"""Reads a seed_g.py output and applies the acceptance rule. Exit code 0 = accept, 1 = reject, 2 = review (some guard
question or guarded cell counts worse; such a run is never accepted, whatever G says).
Default: G counts (T-band and raw, each with D < 0 in val and test). --transfer (a change carried over to another base,
one seed against several): D < 0 on T-band and on raw, in val and in test; significance is not asked.
The file must have been made with G over --g-questions, a guard line for every other question and, with --band-guard,
a band-guard line for every question of G; anything missing gives exit code 3 and no verdict line.
Harm: a guard or band-guard question whose verdict starts with "worse" (T-band, or raw confirmed by the global T), or a
cell of the --cells questions in --bands marked "counts (worse" (--cells with nothing: no cell guard).
  python accept_g.py ../results/c17t_drop1.txt --g-questions ttd --cells ttd --bands 1-4 5-12 13-20 30+ [--transfer]
  python accept_g.py ../results/c17d_drop1.txt --band-guard --cells"""
import argparse
import re
import sys
from pathlib import Path

QS = ("ttd", "depth", "end", "ascends")


def sections(lines):
    """{name: its lines}; the name is what follows '==== ' up to the colon or the parenthesis"""
    out, name = {"head": []}, "head"
    for s in lines:
        m = re.match(r"==== ([a-z ]+?)[:(]", s)
        if m:
            name = m.group(1).strip()
            out[name] = []
        else:
            out[name].append(s)
    return out


def verdicts(sec):
    """{question: verdict text} of the 'q counts: ...' lines of a guard section"""
    out = {}
    for s in sec:
        m = re.match(r"\s+(\w+)\s+counts: (.*)", s)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--g-questions", nargs="+", default=list(QS), choices=QS)
    ap.add_argument("--band-guard", action="store_true")
    ap.add_argument("--cells", nargs="*", default=["ttd"], help="the questions whose cells are guarded")
    ap.add_argument("--bands", nargs="+", default=["1-4", "5-12", "13-20", "30+"])
    ap.add_argument("--transfer", action="store_true")
    a = ap.parse_args()
    lines = Path(a.file).read_text(encoding="utf-8").splitlines()
    sec = sections(lines)
    gq = [q for q in QS if q in a.g_questions]
    m = re.search(r"G = mean T-band log loss over (.+?) x Dlvl 21-24, 25-29", lines[0])
    assert m and m.group(1).split(", ") == gq, f"G is not over {', '.join(gq)}: {lines[0]}"
    why = []
    if a.transfer:
        signs = {}
        ml = sec["main"]
        for k, s in enumerate(ml):
            m = re.match(r"\s+(T-band|raw)\s+G per seed", s)
            if m:
                d = {x: float(re.search(rf"{x} ([+-][0-9.]+)", ml[k + 1]).group(1)) for x in ("D", "val", "test")}
                signs[m.group(1)] = all(v < 0 for v in d.values())
                why.append(f"{m.group(1)}: D {d['D']:+.4f} val {d['val']:+.4f} test {d['test']:+.4f}")
        assert set(signs) == {"T-band", "raw"}, "no T-band / raw D lines"
        g_ok = all(signs.values())
        g_msg = "D < 0 on T-band and raw, in val and test" if g_ok else "D not below 0 everywhere"
    else:
        g = [s for s in sec["main"] if s.strip().startswith("G counts:")]
        assert len(g) == 1, "no G verdict line"
        g_ok = g[0].split(":", 1)[1].strip().startswith("yes")
        g_msg = g[0].strip()
        why.append(f"G: {g_msg}")
    harm = []
    guards = [("guard", [q for q in QS if q not in gq])] + ([("band guard", gq)] if a.band_guard else [])
    for name, want in guards:
        got = verdicts(sec.get(name, []))
        assert sorted(got) == sorted(want), f"{name}: lines for {sorted(got)}, expected {sorted(want)}"
        for q in want:
            why.append(f"{name} {q}: {got[q]}")
            if got[q].startswith("worse"):
                harm.append(f"{name} {q}")
    if a.cells:
        q, seen = None, set()
        for s in sec["cells"]:
            if re.match(r"^  \w+$", s):
                q = s.strip()
                continue
            f = s.split()
            if q in a.cells and f and f[0] in a.bands:
                seen.add((q, f[0]))
                if "counts (worse" in s:
                    harm.append(f"cell {q} {f[0]}")
                why.append(f"cell {q} {f[0]}: D {f[1]}" + ("  " + s.split("  ")[-1].strip()
                                                             if "counts" in s or "only" in s else ""))
        want = {(q, b) for q in a.cells for b in a.bands}
        assert seen == want, f"cells missing in the table: {sorted(want - seen)}"
    print("\n".join(why))
    if harm:
        print(f"verdict: review (harm: {', '.join(harm)}; main: {g_msg})")
        sys.exit(2)
    print(f"verdict: {'accept' if g_ok else 'reject'} ({g_msg})")
    sys.exit(0 if g_ok else 1)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, StopIteration) as e:  # a file that does not fit the rule: no verdict, not reject
        print(f"no verdict: {type(e).__name__} {e}")
        sys.exit(3)
