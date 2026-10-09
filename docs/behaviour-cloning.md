# Behaviour cloning

Before settling on "the model takes over what the bot cannot do", the opposite was tried: have laya play the whole
game by imitating the bot. If the model could reproduce the bot, it could later be improved past it. It could not.

## Setup

Each step of a bot trace becomes a question: given the state as text and the last few commands, which command next?
The bot's keypresses are grouped into 20 commands (the eight moves, search, search 5 turns, fight, pick up, eat, pray,
throw, apply, Elbereth, go down and so on), and the arguments are separate questions (which item, which direction,
eat what is here). Free queries the bot makes for its own information (look here, list monsters, show terrain,
discoveries, `#enhance`, a test Elbereth followed by reading it) are not decisions and are left out. Moves are thinned
in training (kept at 15 %), rare commands kept in full.

`bc_vocab.py`, `bc_build.py`, `laya_play.py` (the player), `laya_serve.py` (the model server).

## Getting the labels right

- **The action field is an index, not a key.** The trace stores the index into NLE's list of 121 actions. The first
  version read it as an ASCII code; every label was garbage.
- **The message tells what the keypress was.** A keypress made while the game asks "What do you want to throw?" or
  "In what direction?" is an argument, not a command.
- **Not every keypress is what it looks like.** An audit against the next step's status and messages
  (`bc_audit.py`, 100 games): moves went where intended in 94.7 % of cases (5.2 % stayed in place: an attack, a door, a
  pet), searches 99.9 %, throws 98.9 %, going down 99 %. But about **35 % of "pick up" were just looking at the menu**
  and closing it, and **half of "apply" were peeking into a bag**. Now a pick-up counts only if something was picked
  up, apply only with a direction (digging), and "You can't..." is not a command. After the fix, 99.3 % of pick-ups
  really pick something up.

## A sign error

Commands were kept with probability `KEEP[c]` in training, so the model learns `p(c | s) · KEEP[c]`, and the bot's
frequencies are recovered by **dividing** by `KEEP`. The evaluation and the player multiplied instead. Rare commands
(kept in full) against moves (kept at 15 %) ended up overweighted by (1 / 0.15)² ≈ 44 times: Elbereth was 40 % of
commands, "eat" topped the list on turn 9, the Valkyrie threw her own sword. The fix was one line.

## Results

On 3000 held-out states, as training went on:

| Share of the first epoch | Top-1 | Top-3 | Log loss | Diagonal moves |
|---|---|---|---|---|
| 6 % | 0.580 | 0.787 | 1.278 | 0.14-0.16 |
| 18 % | 0.615 | 0.817 | 1.177 | 0.16-0.22 |
| 36 % | 0.623 | 0.837 | 1.118 | 0.18-0.29 |

Items and directions were nearly perfect (0.98 and 0.996); throws, applies and searches were well predicted; diagonal
moves stayed poor, confused with the neighbouring straight ones: reading geometry from a text map is hard.

In play (8 games, up to 20 000 steps, at 36 % of the epoch): **every game stayed on Dlvl 1**. Three lived to the step
limit, the others died to Tyr's anger (praying too often), a homunculus, a coyote, a kobold and hunger. The model never
went downstairs: the bot reaches stairs with the travel command `_`, which is not in the command list, so in the
training data going down almost never follows a visible decision.

Playing also exposed loops. An impossible command ("You don't have anything to eat.") changes nothing, so the model is
asked the same question again and answers the same; and "Last commands: eat, eat, eat" in the input reinforces it (the
copycat problem of imitation learning). Fix: a command that left the status unchanged is masked until something
changes and is not added to the history.

## Why it was dropped

Imitation can at best reproduce the bot, and the bot already plays its own game well. Its weaknesses are not in the
keypresses but in the decisions it has no rule for. Errors also compound in a cloned policy (each mistake takes the
game to states the teacher never visited), so early checkpoints are a test of the player, not of the model's strength.
So the model was given the parts where the bot is weak (risk, how humans' games turn out, choices at forks) and the
bot keeps the mechanics.
