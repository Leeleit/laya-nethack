"""The figures of the docs, drawn from the text results in results/ (the numbers are parsed, not typed in, except the
training times, which come from the training logs).
  python docs/figures/make_figures.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


def read(name):
    return (RES / name).read_text(encoding="utf-8")


def finish(fig, name, title, note=""):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 0.97))
    if note:
        fig.text(0.01, 0.005, note, ha="left", va="bottom", fontsize=8, color=MUTED)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(name)


def hbar_labels(ax, bars, fmt):
    for b in bars:
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt(b.get_width()), va="center", ha="left",
                fontsize=9, color=INK2)


def bot_vs_humans():
    rows = {}
    for m in re.finditer(r"^\s*(\d+) (ascended|died ≤ d\+2|bot)\s+(\d+)\s+(\d+) \[[^\]]*\]\s+(-?\d+) \[",
                         read("nao_compare_val_dwa.txt"), re.M):
        rows.setdefault(m.group(2), []).append((int(m.group(1)), int(m.group(4)), int(m.group(5))))
    series = [("ascended", "Humans who ascended", BLUE), ("died ≤ d+2", "Humans who died within 2 levels", ORANGE),
              ("bot", "The bot", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k, title in ((axes[0], 2, "Armour class on arrival (lower is better)"),
                         (axes[1], 1, "Experience level on arrival")):
        for key, label, color in series:
            r = sorted(rows[key])
            x, y = [v[0] for v in r], [v[k] for v in r]
            ax.plot(x, y, color=color, marker="o", markersize=4, label=label)
            ax.annotate(label.replace("Humans who ", "").replace("The ", ""), (x[-1], y[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.set_title(title)
        ax.set_xlabel("Dlvl, first arrival")
        ax.set_xlim(4, 33)
    axes[0].legend(loc="lower left", fontsize=8.5)
    finish(fig, "bot_vs_humans.png", "The bot reaches the castle with the armour and experience of a human on Dlvl 8",
           "Medians. Humans: Val-Dwa games on NAO (1539 ascended, 12 849 died on Dlvl 10+); the bot: 1000 games. "
           "results/nao_compare_val_dwa.txt")


def where_games_end():
    bands = re.findall(r"^(Dlvl [\d+-]+(?: \([^)]*\))?)\s+(\d+)\s+([\d.]+)%", read("bot_bottlenecks.txt"), re.M)
    end = read("bot_endgame_val1.txt")
    n = re.search(r"from Medusa down: (\d+) games", end).group(1)
    stages = [("Reached Medusa", r"reached Medusa ([\d.]+)%"), ("Passed Medusa", r"passed it ([\d.]+)%"),
              ("Reached the castle", r"reached the castle ([\d.]+)%"),
              ("Crossed the castle", r"crossed it into the Valley ([\d.]+)%")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    labels = [b[0] for b in bands][::-1]
    vals = [float(b[2]) for b in bands][::-1]
    bars = ax.barh(labels, vals, color=BLUE, height=0.6)
    hbar_labels(ax, bars, lambda v: f"{v:.1f} %")
    ax.set_title(f"Where {sum(int(b[1]) for b in bands)} games end")
    ax.set_xlabel("share of games, %")
    ax.set_xlim(0, 72)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    vals = [float(re.search(p, end).group(1)) for _, p in stages][::-1]
    bars = ax.barh([s[0] for s in stages][::-1], vals, color=BLUE, height=0.6)
    hbar_labels(ax, bars, lambda v: f"{v:.1f} %")
    ax.set_title(f"Milestones, {n} games")
    ax.set_xlabel("share of games, %")
    ax.set_xlim(0, 80)
    ax.grid(axis="y", visible=False)
    finish(fig, "bot_where_games_end.png", "Six games in ten end at the castle, and almost none get past it",
           "The bot playing Val-Dwa on fixed seeds. results/bot_bottlenecks.txt, results/bot_endgame_val1.txt")


def risk_ap():
    text = read("compare_v2.txt")
    hs, ap = [], {"logit": [], "gbm": [], "risk-v2": []}
    for block in re.split(r"^== ", text, flags=re.M)[1:]:
        m = re.match(r"h=(\d+) all: (\d+) states, (\d+) positives", block)
        if not m:
            continue
        hs.append((int(m.group(1)), int(m.group(3)) / int(m.group(2))))
        for k in ap:
            ap[k].append(float(re.search(rf"^\s+{re.escape(k)}\s+AUC .*? AP ([\d.]+)", block, re.M).group(1)))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.26
    names = [("logit", "Logistic regression", BLUE), ("gbm", "Gradient boosting", ORANGE),
             ("risk-v2", "laya (risk-v2)", AQUA)]
    for i, (k, label, color) in enumerate(names):
        xs = [j + (i - 1) * w for j in range(len(hs))]
        bars = ax.bar(xs, ap[k], w - 0.03, color=color, label=label)
        for b, v in zip(bars, ap[k]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(hs)), [f"dies within {h} turns\n(base rate {r:.1%})" for h, r in hs])
    ax.set_ylabel("average precision")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9, ncols=3)
    ax.set_ylim(0, max(max(v) for v in ap.values()) * 1.25)
    finish(fig, "risk_ap.png", "laya is best at spotting death in the next few turns; tables win at 100 turns",
           "Test: 477 bot games, 138 353 states. Tables: numeric features of the states; laya: the same states as text. "
           "results/compare_v2.txt")


def risk_alarm():
    rows = {m.group(1): [float(m.group(2)), float(m.group(3)), float(m.group(5))] + [int(m.group(4))]
            for m in re.finditer(r"^(hp_rule|laya)\s+alarm in last 7 turns: ([\d.]+)\s+last 30: ([\d.]+)\s+first alarm, "
                                 r"turns before death: median (\d+) \(deaths never alarmed: ([\d.]+)\)",
                                 read("alarm_v2.txt"), re.M)}
    budget = float(re.search(r"alarm budget on safe states \(HP rule\): ([\d.]+)", read("alarm_v2.txt")).group(1))
    cats = ["Warned in the last 7 turns", "Warned in the last 30 turns", "Never warned"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    h = 0.36
    for i, (k, label, color) in enumerate((("laya", "laya", BLUE), ("hp_rule", "The bot's HP rule", ORANGE))):
        ys = [j + (0.5 - i) * h for j in range(3)][::-1]
        bars = ax.barh(ys, [v * 100 for v in rows[k][:3]], h - 0.04, color=color, label=label)
        hbar_labels(ax, bars, lambda v: f"{v:.0f} %")
    ax.set_yticks(range(3)[::-1], cats)
    ax.set_xlabel("share of the bot's deaths, %")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    finish(fig, "risk_alarm.png",
           f"Same false-alarm rate, nearly twice the deaths foreseen (median {rows['laya'][3]} turns ahead vs "
           f"{rows['hp_rule'][3]})",
           f"Both alarms fire on {budget:.1%} of safe states (death > 100 turns away). 468 deaths in the test games. "
           "results/alarm_v2.txt")


def branches():
    text = read("fork_h300_stats.txt")
    block = text.split("horizon 300:")[1].split("bootstrap")[0]
    names = {"pray": "Pray", "elbereth": "Engrave Elbereth", "elbereth_rest": "Elbereth, then rest",
             "quaff_heal": "Quaff a healing potion", "stairs": "Take the stairs underfoot",
             "upstairs": "Walk to the up stairs", "dig_down": "Dig down"}
    rows = [(names[m.group(1)], int(m.group(2)), float(m.group(3)) * 100, float(m.group(4)) * 100)
            for m in re.finditer(r"^\s+(\w+)\s+(\d+) points: .*?effect ([+-][\d.]+) ± ([\d.]+)", block, re.M)]
    rows.sort(key=lambda r: r[2])
    oracle = re.search(r"oracle \(split reps\)\s+([\d.]+)\s+diff ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]",
                       read("fork_h300_eval_h300.txt"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ys = range(len(rows))
    ax.errorbar([r[2] for r in rows], ys, xerr=[r[3] for r in rows], fmt="o", color=BLUE, ecolor=BLUE,
                elinewidth=2, capsize=0, markersize=6)
    ax.axvline(0, color=AXIS, linewidth=1.2)
    ax.set_yticks(list(ys), [f"{r[0]}  ({r[1]} points)" for r in rows])
    ax.set_xlabel("survival after 300 turns, percentage points vs the bot's own move (95 % CI)")
    ax.grid(axis="y", visible=False)
    for y, r in zip(ys, rows):
        ax.text(r[2] + r[3] + 0.4, y, f"{r[2]:+.1f}", va="center", fontsize=8.5, color=INK2)
    finish(fig, "branches.png", "No emergency move beats the bot's own choice",
           f"Branches forked at the first risk alarm of each game, 8 replays per option. A per-point oracle "
           f"(chosen and scored on different replays): {float(oracle.group(2)) * 100:+.1f} pp "
           f"[{float(oracle.group(3)) * 100:+.1f}, {float(oracle.group(4)) * 100:+.1f}]. results/fork_h300_*.txt")


def sokoban():
    text = read("ab_soko_val2.txt")
    n = re.search(r"(\d+) paired seeds", text).group(1)
    stages = [("reached_medusa", "Reached Medusa"), ("passed_medusa", "Passed Medusa"),
              ("reached_castle", "Reached the castle"), ("crossed_castle", "Crossed the castle"),
              ("gehennom", "Entered Gehennom")]
    vals = {k: [float(x) for x in re.search(rf"^\s+{k}\s+([\d.]+)% ->\s+([\d.]+)%", text, re.M).groups()]
            for k, _ in stages}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.38
    for i, (label, color) in enumerate((("The bot as is", BLUE), ("Sokoban before diving", ORANGE))):
        xs = [j + (i - 0.5) * w for j in range(len(stages))]
        bars = ax.bar(xs, [vals[k][i] for k, _ in stages], w - 0.04, color=color, label=label)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}", ha="center",
                    va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(stages)), [s[1] for s in stages])
    ax.set_ylabel("share of games, %")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9)
    finish(fig, "sokoban.png", "Sending the bot to Sokoban first costs a third of its castle arrivals",
           f"{n} paired seeds; the bot's own (off by default) Sokoban routine switched on. results/ab_soko_val2.txt")


def training_time():
    # one epoch on the same 92 862 human states (two questions each), RTX 5090; from the training logs
    runs = [("Padded batches of 8, torch.compile", 2693), ("Packed sequences, batch 16", 1131),
            ("+ the map described in words", 877), ("+ shorter wording", 856)]
    fig, ax = plt.subplots(figsize=(8, 2.9))
    bars = ax.barh([r[0] for r in runs][::-1], [r[1] / 60 for r in runs][::-1], color=BLUE, height=0.6)
    hbar_labels(ax, bars, lambda v: f"{v:.0f} min")
    ax.set_xlabel("minutes per epoch")
    ax.set_xlim(0, 52)
    ax.grid(axis="y", visible=False)
    finish(fig, "training_time.png", "One epoch: 45 minutes down to 14, with no loss in quality",
           "laya, 20 encoder layers, fp16, RTX 5090. Quality checked on held-out players after each step "
           "(results/eff_*_val.txt).")


def repeats():
    text = read("curve_16d_steps.txt")
    blocks = dict(re.findall(r"^== (\w+): .*?\n(.*?)(?=^== |\Z)", text, re.M | re.S))
    titles = {"ttd": "Turns until the game ends", "depth": "Deepest level this game will reach",
              "end": "What ends the game", "ascends": "Will this game ascend"}
    shares = [("half", "Half of the games", BLUE), ("third", "A third", ORANGE), ("quarter", "A quarter", AQUA)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, q in zip(axes.flat, titles):
        loss = {m.group(1): float(m.group(2)) for m in
                re.finditer(r"^\s+(\w+)\s+raw [\d.]+ \[[^\]]*\]\s+with T ([\d.]+)", blocks[q], re.M)}
        ax.axhline(loss["all"], color=INK2, linewidth=1.2, linestyle="--")
        ax.text(4.25, loss["all"], "all games,\n1 epoch", va="bottom", fontsize=8, color=INK2)
        for key, label, color in shares:
            pts = [(1, loss[key])] + [(k, loss[f"{key}x{k}"]) for k in (2, 3, 4) if f"{key}x{k}" in loss]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker="o", markersize=5, label=label)
        ax.set_title(titles[q])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.8, 5.0)
        ax.set_xlabel("epochs over the same games")
        ax.set_ylabel("log loss, held-out players")
    axes[0, 0].legend(fontsize=8.5, loc="upper left")
    finish(fig, "repeats.png", "More epochs over fewer games do not replace new games",
           "Val-Dwa human games, 82 188 training rows in all; equal compute at the last point of each line. "
           "Temperature fitted on other players. results/curve_16d_steps.txt")


def mixing_bands():
    text = read("deep19.txt").split("== Dlvl bands")[1]
    models = [("mix", "Val-Dwa + other roles, mixed", BLUE), ("staged", "Other roles first, then Val-Dwa", ORANGE),
              ("others", "Other roles only", AQUA)]
    titles = {"ttd": "Turns until the game ends", "depth": "Deepest level this game will reach",
              "end": "What ends the game"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, q in zip(axes, titles):
        block = re.search(rf"^  {q}\n((?:    .*\n?)+)", text, re.M).group(1)
        bands = re.findall(r"^\s+Dlvl (\d+)-(\d+): rows \d+ games (\d+)(.*)$", block, re.M)
        for i, (key, label, color) in enumerate(models):
            xs, ys, lo, hi = [], [], [], []
            for j, (a, b, games, rest) in enumerate(bands):
                m = re.search(rf"\| {key} ([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\]", rest)
                v, l, h = (float(x) for x in m.groups())
                xs.append(j + (i - 1) * 0.24)
                ys.append(v)
                lo.append(v - l)
                hi.append(h - v)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=color, elinewidth=1.6, capsize=0, markersize=4.5,
                        label=label)
        ax.axhline(0, color=AXIS, linewidth=1.2)
        ax.set_xticks(range(len(bands)), [f"{a}-{b}" if int(b) < 90 else f"{a}+" for a, b, _, _ in bands])
        ax.set_xlabel("Dlvl of the state")
        ax.set_title(titles[q])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("log loss minus the Val-Dwa-only model")
    axes[0].legend(fontsize=8.5, loc="lower left")
    finish(fig, "mixing_bands.png", "Mixed with other roles, the model reads deep games better; other roles alone miss the cause of death",
           "Below zero = better than the model trained on Val-Dwa alone. Held-out Val-Dwa players, one state per "
           "game per band; 95 % bootstrap over players. results/deep19.txt")


if __name__ == "__main__":
    for f in (bot_vs_humans, where_games_end, risk_ap, risk_alarm, branches, sokoban, training_time, repeats,
              mixing_bands):
        f()
