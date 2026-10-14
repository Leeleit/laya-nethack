# laya × NetHack

Teaching a small decision model to help a NetHack bot where the bot is weak.

![The screensaver: a recorded bot game with the risk model's answer beside it](docs/figures/screensaver_minotaur.png)

*Dlvl 28, the castle level. The hero has 60 of 67 hit points, so a rule based on hit points sees no danger. A minotaur
(H) and an arch-lich (L) stand next to her. The risk model gives 96.5 % to death within two turns. The hero was killed
by the minotaur on that turn.*

## The idea

The bot used here (`pf_base` from
[daglar-dragomirov/nethacker](https://github.com/daglar-dragomirov/nethacker), a descendant of
[AutoAscend](https://github.com/maciej-sypetkowski/autoascend)) reaches the Medusa level in two games out of three and
the castle in four out of ten, and almost never gets past it. It moves, fights and digs very well. What it lacks is
judgement: when it is about to die, which risks are worth taking, what a strong human would do in its place.

[laya](https://github.com/NandhaKishorM/laya) is a 421M-parameter decision model (a ModernBERT-large encoder with a
small head) that reads a situation as text and answers typed questions: yes/no, pick one of N, or an ordered scale. It
answers in one forward pass and gives a probability for every option. That fits the role of a fast "System 1" next to
a hand-written bot.

So the bot and the model work as a pair:

- **the bot** plays the game: movement, combat, digging, everything it already does well;
- **the model** takes over the decisions the bot has no good rule for, one mechanic at a time;
- the bot is also the teacher (its games are training data) and the yardstick (every change is an A/B test on fixed
  seeds).

The end product is a screensaver: a live game on one side and the numbers of both players on the other. The ideal is
a pair that ascends; an acceptable result is one that reaches the Amulet. All the work is on one character, the dwarven
Valkyrie: among players of similar experience, the role and race that ascends most often.

## What has been found so far

**The bot is short of armour, not of experience.** On every level from 5 to 28 the bot arrives with armour class 0
and experience level 7. Humans who go on to ascend reach Dlvl 25 with a median armour class of −13; by then 81 % of
them carry reflection, 72 % levitation and 51 % identified magic resistance (the bot: 0, 2 and 0 %).
Experience level barely separates humans who ascend from humans who die; armour class and resistances do.
[More](docs/bot.md)

![Armour class and experience level on arrival, bot vs humans](docs/figures/bot_vs_humans.png)

**The model sees death coming.** Trained on the bot's own games, laya ranks the deaths of the next 2 to 7 turns better
than logistic regression or gradient boosting on hand-picked features. At the same false-alarm rate as the bot's
hit-point rule, it warns before 71 % of deaths (the rule: 37 %), with a median of 27 turns of notice instead of 7.
[More](docs/risk.md)

![Alarm before death: laya vs the bot's hit-point rule](docs/figures/risk_alarm.png)

**But there is nothing to do with the warning in the moment.** Forking the game at the first alarm and trying every
emergency move the bot knows (pray, Elbereth, quaff, flee by stairs, dig down) with 8 replays each, none beats what
the bot does anyway. What would change the outcome is what the hero carries by then. [More](docs/branches.md)

**Cloning the bot was a dead end.** A model that imitates the bot's every keypress learns movement and never learns to
go downstairs. The model is now taught what the bot cannot do itself. [More](docs/behaviour-cloning.md)

**Human games are the reference.** From 227 GB of terminal recordings of public-server games (NLD-NAO) the project
extracts the state of a game at chosen moments and asks the model how a human's game will end: how many turns are left,
how deep it will go, what will kill the hero, whether it will ascend. Getting this data clean took more work than any
model. [Data](docs/human-data.md) · [Questions](docs/questions.md)

**Training data: new games, few states per game.** One epoch over the Valkyrie games was the best schedule tried;
several epochs over fewer games, or every state of the deep games, make the model memorise games instead of
situations. Which states are taken matters too: one per depth band per game looks fair but skews the answers toward
early deaths. Mixing in games of other roles looked better on the deepest levels on one test set and was not
confirmed on a second split of players; other roles without Valkyrie games get the cause of death wrong. Averaging
several training runs helps, and one model trained to match the average keeps most of the gain.
[More](docs/data-scaling.md)

**Training got 3× faster with no loss in quality.** Packed sequences instead of padded batches, a map described in
words instead of drawn in ASCII, shorter wording. [More](docs/training.md)

## Documents

| | |
|---|---|
| [The bot and where it dies](docs/bot.md) | profile of the bot, milestones, causes of death, humans vs bot, attempts to make it stronger |
| [Predicting death](docs/risk.md) | the risk model, comparison with tabular baselines, the alarm, calibration |
| [Emergency moves](docs/branches.md) | replaying games from a seed, forking a running game, why no move helps |
| [Behaviour cloning](docs/behaviour-cloning.md) | imitating the bot keypress by keypress, and why it was dropped |
| [Human games](docs/human-data.md) | extracting game states from terminal recordings, pitfalls, bots among the "humans" |
| [What the model is asked](docs/questions.md) | the state as text, the questions, the token budget |
| [How much data, and which](docs/data-scaling.md) | learning curves, repeats, which states, other roles, ensembles, evaluating where it matters |
| [Training speed](docs/training.md) | from 45 to 14 minutes an epoch |

## Repository layout

| Path | What |
|---|---|
| `nethackers/collect.py` | plays the bot on fixed seeds in the arena's sandbox and keeps a trace of every step; resumable |
| `trace/` | all the code: collection, datasets, training, evaluation (flat on purpose: the scripts import each other) |
| `trace/image/` | Dockerfiles for replaying the bot deterministically and the patch that makes the arena resumable |
| `trace/patched/` | the bot-side hooks: per-step traces (`laya_trace.py`) and decision branches (`laya_branch.py`) |
| `screensaver/` | the screensaver (pygame) and a headless renderer for stills |
| `stats/` | statistics over 2.28 million public-server games (xlogfile) |
| `results/` | the text output of every measurement quoted in the docs |
| `docs/` | the write-ups; `docs/figures/make_figures.py` redraws every chart from `results/` |

Main entry points in `trace/`:

| Stage | Scripts |
|---|---|
| Bot games | `../nethackers/collect.py`, `analyze_traces.py`, `bot_profile.py`, `bot_bottlenecks.py`, `bot_endgame.py`, `bot_ab.py` |
| Risk model | `build_v2.py`, `finetune_fast.py`, `predict.py`, `compare_v2.py`, `alarm_v2.py` |
| Branches | `plan_branches.py`, `forkrun.py`, `build_fork.py`, `eval_fork.py`, `rspi_sim.py` |
| Human data | `nao_corridor.py` (extraction), `human_build.py` (datasets), `state_fmt.py` (state text), `human_q6.py` (questions), `data_checks.py`, `junk_scan.py`, `rank_check.py`, `bot_list.py` |
| Model on humans | `finetune_fast.py`, `packed.py`, `predict.py`, `curve_eval.py`, `human_tab.py` (tabular baseline) |
| Evaluation | `deep_checks.py` (by band), `seed_g.py` and `accept_g.py` (the rule over seeds), `ens_preds.py`, `distill_build.py`, `band_recal.py`, `seed_games.py` |
| Screensaver | `laya_serve.py`, `../screensaver/saver.py` |

## Running it

This is a research log, not a package; the scripts are as they were used. They expect:

- the laya repository and its checkpoint (`laya-typed-decisions`) for the model code and the starting weights;
- the bot (`pf_base`) and the NetHackers arena for playing games on fixed seeds; NLE 1.x underneath;
- NLD-NAO (ttyrecs and xlogfiles) for the human games;
- PyTorch 2.14 with a CUDA GPU for training. Everything here was trained on one RTX 5090; an RTX 3060 Ti was enough
  for inference and the screensaver.

Most scripts give their command line in the docstring. Default paths (`/data/laya/...`) are those of the machine the
work ran on. The shell scripts that chained the steps into experiments are not included: they were tied to that
machine (service units, CPU pinning).

Not included: datasets, checkpoints, predictions, game traces and logs (hundreds of gigabytes in total), and the
third-party code listed above.

## Credits

- [laya](https://github.com/NandhaKishorM/laya): the decision model and its fine-tuning code.
- [NetHack Learning Environment](https://github.com/heiner/nle) and the NetHack Learning Dataset
  ([Dungeons and Data](https://arxiv.org/abs/2211.00539): NLD-NAO, NLD-AA).
- [AutoAscend](https://github.com/maciej-sypetkowski/autoascend) and the `pf_base` bot of
  [daglar-dragomirov/nethacker](https://github.com/daglar-dragomirov/nethacker), used here as is, with two hooks.
- [alt.org (NAO)](https://alt.org/nethack/) and [Hardfought](https://www.hardfought.org/) players, whose games are the
  human reference.

## License

[GPL-3.0](LICENSE).
