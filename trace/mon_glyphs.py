"""How well a map letter names a monster: monsters per class letter, and per (letter, colour) as tty
shows them (nle's monster table, 3.6).
  python mon_glyphs.py
"""
import collections

from nle import nethack

by_letter, by_glyph = collections.defaultdict(list), collections.defaultdict(list)
for i in range(nethack.NUMMONS):
    p = nethack.permonst(i)
    sym = nethack.class_sym.from_mlet(p.mlet).sym
    by_letter[sym].append(p.mname)
    by_glyph[(sym, p.mcolor)].append(p.mname)
n = nethack.NUMMONS
print(f"{n} monsters, {len(by_letter)} letters, {len(by_glyph)} (letter, colour) pairs")
print(f"unique by letter: {sum(len(v) == 1 for v in by_letter.values())} letters; "
      f"by (letter, colour): {sum(len(v) for v in by_glyph.values() if len(v) == 1)} of {n} monsters unique")
for k in ("a", "@", "d", "q", "f"):
    print(f"  {k}: {len(by_letter[k])} -- {', '.join(by_letter[k][:12])}")
print("shared (letter, colour):")
for (sym, c), v in sorted(by_glyph.items(), key=lambda t: -len(t[1]))[:15]:
    if len(v) > 1:
        print(f"  {sym} colour {c}: {', '.join(v)}")
