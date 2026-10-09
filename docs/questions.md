# What the model is asked

laya reads one string: a question, the answer options, and the state, all as text. What goes into that string, and
how it is worded, decides both what the model can learn and how fast it trains. This page is about both.

## The state as text

`state_fmt.py` builds the same layout from a bot trace and from a human game, so one model can read both:

```
Val-Dwa-Law-Fem 3.4.3
HP 51/51 (10 turns ago 51/51, 30 ago 51/51) Dlvl 8 deepest 8 XL 6 AC 1 turn 1954
St:25 Dx:15 Co:16 In:7 Wi:10 Ch:10 $:547 Pw:3(3) Hungry
Galois bites the jackal.  The jackal misses Galois.  The jackal bites! | You kill the jackal!  The jackal bites!  Galois bites the jackal.
Around: up stairs 1. Monsters: humanoid 1; cat or other feline (pet) 2; unseen monster 2; vampire 2
Inventory: a - +1 long sword (wielded); b - +0 dagger; c - uncursed +3 small shield (worn); d - uncursed food ration
```

(A real state of a held-out game, as the model sees it. The question in front of it starts with the player's history,
for example "A human (experience 1/5, result 3/5) plays. Turns until the game ends?" followed by the answer options.)

The path to this layout, each step checked on held-out players before being kept:

| Change | Tokens per state |
|---|---|
| raw screen with an ASCII map, the level map and a window around the hero | 696 |
| no repeated numbers, no player name or rank title, maps trimmed | 650 |
| the level map dropped, only the area around the hero | 375 |
| **the map described in words** | 273 |
| short wording everywhere | ~185-215 |

**The map in words** (`state_fmt.surroundings`): instead of drawing the map, list the
notable features on the level (stairs, fountain, altar, throne, traps, water, lava) and the monsters in view, each
with the number of moves to reach it over the known map (a breadth-first search). A model trained on it was no worse
on any question than one trained on the drawn map (`results/eff_a16a8_val.txt`) and trained 1.3× faster. For human
games the drawn map turned out to add almost nothing anyway: removing it at inference changed the log loss by 0.001 to
0.002, within the noise (`map_variants.py`).

**A cut that did cost something.** The inventory is half of the state's tokens. A compact form (no item letters, no
articles, "(worn)" for "(being worn)") saved 45 tokens but made the "how far will this game go" question
significantly worse, mostly on games that ascended (`results/eff_a9s_val.txt`, `eff_a9_val.txt`). Most of the
saving, 30 of the 45 tokens, was the letters, so
the compact form now keeps the letters and drops only the articles and the long annotations. Whether that also removes
the loss has not been measured.

## The questions

All four are asked about every human state. The player's history comes first ("A human (experience 2/5, result 3/5)
plays."), see [human games](human-data.md).

| Question | Answers | Notes |
|---|---|---|
| Turns until the game ends? | 17-19 bins | equal shares of training games on a log scale, plus fixed edges at 1000, 5000, 20 000, 40 000 turns |
| Deepest Dlvl this game will reach? | 12 bins | bins of how many levels deeper, labelled with the level: "no deeper", "6", ..., "36+" |
| What ends the game? | 39 options | the 30 most common monster classes, other monster, ray or wand, poison, petrification, starvation or choking, drowning, trap, other, **ascension** |
| Does this game ascend? | yes / no | |

The ordered questions are trained on soft targets: a Gaussian over the neighbouring bins with a width of 0.75 bins
(the HL-Gauss recipe, 2403.03950). On a tabular stand-in, soft targets beat hard labels at this number of bins, and the
number of bins itself stops mattering beyond 20 to 30 (`bins_proxy.py`).

Two of the early choices were wrong and were fixed:

- **The first version had 3 to 4 hand-picked bins** and merged "ascends" with "lasts over 5000 turns". The current
  questions fold back into the old ones exactly, so the two could be compared on the same states (`fold_eval.py`):
  equal where they overlap, and the new ones carry far more information.
- **"Other monster" held 29 % of deaths.** Twelve classes were too few, and priests, shopkeepers, "touch of
  death", poisonous corpses and divine anger were counted as monsters too. With those fixed and 30 classes instead of
  12, "other monster" is 1.5 %.
  39 options need a question budget of 512 tokens instead of 256 (`--head-max-len 512`).

## laya against a table

The baseline: gradient boosting on numbers read from the same state (hit points, depth,
experience and armour, turn, which item kinds are carried, the player's history) (`human_tab.py`). On 4779 held-out
states of 664 players, log loss with a fitted temperature, laya minus the table:

| Question | laya | Gradient boosting | Difference |
|---|---|---|---|
| Turns until the end | 2.577 | 2.593 | laya better by 0.016 [0.007, 0.024] |
| What ends the game | 3.120 | 3.177 | laya better by 0.057 [0.043, 0.070] |
| Does it ascend | 0.066 | 0.068 | equal |
| How deep | 1.895 | 1.859 | **the table better** by 0.036 [0.026, 0.046] |

(`results/curve_16d_c16_vs_gbm.txt`.) The depth question is lost on Dlvl 1 to 4 and on Dlvl 21 and deeper; on Dlvl 8
to 20 laya is better. Why is not known yet.
