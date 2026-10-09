"""Figure of skill_dist.py's train arrays: experience log10(n + 1) with its normal fit; result k / n
for players with an ascension (k > 0; k = 0 is most rows and is given as a share in the title) with its normal fit.
  python skill_plot.py ../results/skill_dist_15d.npz ../results/skill_dist_15d.png
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Z = 1.959964
INK, MID, MARK, SURF, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb", "#e6e5e0"

d = np.load(sys.argv[1])
n, k = d["n"], d["k"]
exp = np.log10(n + 1.0)
pos = k > 0
res = k[pos] / n[pos]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURF)
panels = ((axes[0], exp, f"Experience: earlier NAO games of the account, log10(n + 1)\ntrain rows {len(n)}; "
           f"no earlier game {np.mean(n == 0):.1%}", "log10(n + 1)"),
          (axes[1], res, f"Result: earlier ascensions / earlier games, accounts with k > 0\nthat is {pos.mean():.1%} "
           f"of train rows; the other {1 - pos.mean():.1%} have k = 0", "k / n"))
for ax, x, title, xl in panels:
    m, s = x.mean(), x.std(ddof=1)
    ax.set_facecolor(SURF)
    ax.hist(x, bins=60, density=True, color=MARK, edgecolor=SURF, linewidth=0.6)
    g = np.linspace(x.min(), x.max(), 400)
    ax.plot(g, np.exp(-(g - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi)), color=INK, linewidth=2)
    ax.axvspan(m - Z * s, m + Z * s, color=MID, alpha=0.08, linewidth=0)
    ax.set_title(f"{title}\nmean {m:.3g}, std {s:.3g}; line = normal fit, band = mean ± 1.96 std", fontsize=9.5,
                 color=INK, loc="left")
    ax.set_xlabel(xl, color=MID)
    ax.set_ylabel("density", color=MID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=120, facecolor=SURF)
