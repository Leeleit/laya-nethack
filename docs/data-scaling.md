# How much data, and which

Some 65 000 human dwarven Valkyrie games are available for training, and about seventeen times as many games of
other roles. The questions: is that enough, should the same games be seen more than once, which states of a game to
take, and do other roles help or get in the way? All models here use the same recipe (laya, 20 layers, one epoch
unless stated, packed batches of 16) and are scored on players never seen in training.

## How to measure

**Score by player, not by row.** The states of one game share its outcome, and a player's games share the player's
strength. Intervals are a bootstrap over players, and a game contributes about one independent observation however
many of its states are in the test (the design effect: with a correlation near 1 inside a game, the effective sample
size is the number of games).

**Score where it matters.** An average over all held-out states is dominated by the first levels, because every game
passes through Dlvl 1 to 4 and few reach Dlvl 25. The bot, on the other hand, dies at the castle in 60 % of its games.
The test takes every state of the held-out players from Dlvl 13 down and a fifth of the shallower ones, and reports
each depth band separately (1-4, 5-12, 13-20, 21-24, 25-29, 30+). The player with the most ascended states is left
out (one player held 1875 of the 25 076 test states, nearly all from ascending games).

The log loss is reported after a temperature is fitted inside each band on the other half of the players, so
differences in calibration do not hide differences in what the model knows. The raw loss is reported beside it.

**One number for a decision.** G is the mean of the four questions on Dlvl 21-24 and 25-29, the bands where the bot
dies. A change is accepted when G improves by more than twice its standard error, the val and test halves and the raw
loss agree in sign, and no other band gets significantly worse.

**Count the training noise.** A bootstrap over test players does not see how much a second training run would
differ. The standard error therefore adds the spread over training seeds (at least three per side) to the bootstrap
(`seed_g.py`, `accept_g.py`). The spread is easy to underestimate: four seeds of the Valkyrie model gave G with an SD of
0.0008, because the cells of G moved in opposite directions and cancelled; with every game weighed the
same, the spread is 0.0075 (`seed_games.py`, `results/seed_games24_drop1.txt`).

**Check on a second split.** Many decisions were read off one test set, so the effects accepted there are biased
upwards. A fresh split holds out another 15 % of the training players (`human_build.py --resplit`; 475, 397 and 228
held-out games reach Dlvl 21-24, 25-29 and 30+, `results/cover30t.txt`), and the main results were trained again on it.

## More games

Training on a quarter, a third, a half and all of the games (`subset_games.py`, `curve_eval.py`,
`results/curve_16d.txt`, 82 188 rows in all):

| Question | 1/4 | 1/3 | 1/2 | all | half minus all |
|---|---|---|---|---|---|
| Turns until the end | 2.601 | 2.598 | 2.586 | 2.577 | +0.010 [0.005, 0.015] |
| How deep | 1.911 | 1.913 | 1.896 | 1.895 | +0.002 (noise) |
| What ends the game | 3.141 | 3.141 | 3.122 | 3.120 | +0.002 (noise) |
| Ascends | 0.0700 | 0.0685 | 0.0649 | 0.0664 | −0.0015 (noise) |

The turns-left question still improves with data, about 0.01 per doubling. The others stop improving between half and
all. Two caveats: two different random quarters differ by as much as 0.013 on depth and 0.018 on the cause of death,
and a second training seed moves the ascension question by 0.004, more than the half-to-all difference.

## The same games again

If new games stop helping, maybe more passes over the same games would. They do not:

![More epochs over fewer games](figures/repeats.png)

(`finetune_fast.py --save-epochs`, `results/curve_16d_steps.txt`.) Every extra epoch over the same games makes every
question worse, and the fitted temperature grows (1.2 after two epochs, 3.2 after four on a quarter): the model starts
to memorise the training games. Half the games for two epochs, the same compute as all games once, is worse than all
games once. A checkpoint taken after the first epoch of a multi-epoch run is also worse than a one-epoch run (its
learning rate has not decayed yet at that point).

**Capping games per player** (at most 20 games from any one player, the same number of rows): no consistent effect
(`results/curve_16d_cap.txt`).

## Which states of a game

**Every state of the deep games is too many.** A model trained on all states from Dlvl 13 down instead of a sample
was much worse on held-out players: cause of death on Dlvl 13-29 by +0.46 to +0.57, depth by +0.22 to +0.35
(`results/all20_drop1.txt`). It had learned the games, not the situations. On states it had never seen, taken from
the shallower levels of games it had trained on, it was better than on games it had not seen, by 0.22 more than the
sampled model (`fam_probe.py`, `results/fam_robust23_drop1.txt`). The channel is near-duplicates: states a few hundred
turns apart in one game share the inventory and the outcome. Masking item appearances and pet names removes about
a third of it (0.217 to 0.140, `mask_probe.py`, `results/mask_delta23_drop1.txt`); what protects is taking few
states per game.

**One random state per band per game shifts the answers.** Taking one state per game in each depth band seems fair,
but it weighs each state by one over the number of the game's states in that band, and that number depends on the
future: a game that dies soon after arriving has few states there. On Dlvl 25-29, the share of states whose game
ends within 1000 turns is 0.093 over all states and 0.204 under that selection (`results/deep_select.txt`). Models
trained that way underestimated the chance to ascend. Each state is now kept on its own, by a hash of its id, at a
fixed rate per band (`human_build.py --band-rates`): on Dlvl 25-29, depth improved by 0.11
(`results/all20_drop1.txt`).

**Games that ended early.** Adding the games that never reached Dlvl 4 (see [human games](human-data.md)) fixed the
start: on Dlvl 1-4, depth −0.104 and turns-left −0.034, the predicted chance of going no deeper 0.113 against an
observed 0.108 (0.056 before), with no significant loss deeper (`results/early20_drop1.txt`).

## Other roles

The first comparison (models trained on one state per band per game, `results/deep19.txt`): other roles alone are
worse on the cause of death at almost every depth (+0.05 to +0.10), because different roles die to different things;
mixing them into the Valkyrie games and training on other roles first, then on Valkyries, came out the same at the
same compute (`results/mix17_staged_vs_mix.txt`).

![Other roles by depth band, first comparison](figures/mixing_bands.png)

With the current selection, Valkyrie games plus 2.16 times as many rows of other roles against Valkyrie games alone,
four seeds each: G −0.028 [−0.048, −0.004], passing the rule by 0.001 (`results/g29_drop1.txt`); the largest part,
about 30 %, is the turns-left question on Dlvl 25-29. On the fresh split, three seeds each, the difference is −0.002
[−0.021, +0.016], with val and test of opposite signs (`results/split30_drop1.txt`). The benefit of mixing is not
confirmed, and no harm is shown.

## Seeds, ensembles and distillation

**Averaging seeds helps; averaging weights does not.** The mean probability of k models trained with different seeds
improves G by about 0.027 × (1 − 1/k): for the mixed model 0.013 with two, 0.018 with three, 0.020 with four
(`results/g29_drop1.txt`); four Valkyrie models together reach 1.527 against 1.552 for one. Averaging the weights
of two seeds instead is worse than either of them, by 0.04 to 0.06 on turns-left at Dlvl 21-29
(`results/soupgain25_drop1.txt`).

**One model can keep most of it.** A model trained to match the ensemble (the target: 0.8 of the ensemble's
probabilities plus 0.2 of the original soft target, `distill_build.py`) beats the mean of its four teachers by 0.021
on G for the Valkyrie model and 0.022 for the mixed one, passing the rule, and is no worse outside Dlvl 21-29
(`results/c17d_drop1.txt`, `results/m20d_drop1.txt`). The distilled mixed model has G 1.501, below the ensemble it
learned from (1.503). The teacher has seen the labels of its own training states, so a little of each label leaks into
the target: its share in the teacher's output on its own training states, against a reference that never saw them,
is 0.058 averaged over the cells of G and 0.15 on the ascension question at Dlvl 21-29 (`leak_total.py`,
`results/leak29.txt`).

**On the fresh split** the student is 0.025 better than the mean of its teachers on G, in val and test alike, and keeps
88 % of the ensemble's gain (`results/c30d0_drop1.txt`, `results/c30d0_ensemble_drop1.txt`). It does not pass,
because the SD of G over the four teachers is 0.026: one of them is worse in every cell of G at once. Without it the
SD over the other three is 0.008. Against the best single teacher, the student is better by only 0.005. What made
that one seed worse is not found: the batch order, overlap between training and test players, label shares,
truncation and player strength were checked and ruled out (`seed_var.py`, `fit_held.py`, `seed_games.py`, `order_probe.py`; `results/seed_var*`,
`seed_games30_drop1.txt`, `order_probe30.txt`).

## Calibration by depth

The models underestimate an early death on Dlvl 25-29. Over the training states, the observed share of games that end
within 1000 turns peaks at Dlvl 25 (0.125), the models' at Dlvl 22-23 (`results/ttd_curve24.txt`). Shifting each
band's predicted shares to the observed shares of the training states (`band_recal.py --obs`) improves turns-left on
Dlvl 25-29 by 0.065 [0.034, 0.102] on the Valkyrie models, with no band worse (`results/band_recal24obs_drop1.txt`).
On the fresh split the same correction gives 0.034 there, but the cause of death and the ascension question on Dlvl
30+ get worse (+0.030 and +0.005, `results/band_recal_obs_c30_drop1.txt`).
