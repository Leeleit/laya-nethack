# 2.28 million public-server games

Statistics over the game logs (xlogfile) of the NAO server (2001-2020, from NLD-NAO) and Hardfought (3.6 and 3.7):
6 853 967 records in all.

- `parse_xlog.py` → `games.parquet`;
- `explore.py`: data hygiene;
- `analyze.py` → `results.json`;
- `roles_table.py` → `roles_table.txt`: ascension rate by role and race, split by the player's experience.

## Sample

2 278 092 games kept (3.4.3: 1.50 million, 3.6: 0.78 million), with 29 786 ascensions by 63 460 players. Dropped:
start-scumming (quit or escaped by turn 100, 4.3 million), wizard and explore mode, `nh361dev`, versions 3.3 and 3.7,
ascensions shorter than 1000 turns.

## Method

The player's skill is the main confounder, so every game gets the player's history before it, with no look-ahead:
how many games and ascensions this account (server + name) had already. Rates have Wilson intervals; models are
logistic regressions with standard errors clustered by player; tables show odds ratios with 95 % intervals.

## Findings

**1. Skill decides almost everything.** The chance to ascend: 0.28 % with no earlier ascensions, 1.8 % with 1-4,
4.4 % with 5-19, 10.7 % with 20 or more. Adjusted for role, race and the rest, the odds ratio of "20+" against "0" is
262. Any raw correlation that ignores this is mostly a correlation with skill.

**2. Where games end.** The chance of going deeper, having reached Dlvl d:

| d | 1 | 3 | 5 | 6 | 8 | 10 | 12 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all players | .80 | .83 | .75 | .73 | **.71** | .78 | .80 | .96 | .96 | .92 | .98 |
| 20+ earlier ascensions | .92 | .90 | .83 | .82 | .81 | .87 | .91 | .98 | .98 | .96 | .99 |

- The deadliest stretch is Dlvl 5-12: the Gnomish Mines, soldier ants, gnomes and dwarves.
- The next drop is at Medusa and the castle (~25). After that the game is nearly won: P(ascend | reached Dlvl 30) is
  75 %, 90 % for experienced players.
- Of 1.9 million deaths, 93 % happen on Dlvl 1-9 and 31 % in the Gnomish Mines.

**3. What kills.** Dlvl 1-4: jackals, rats, mimics, foxes. Dlvl 5-9: soldier ants (6 %), dwarves, gnome lords. Dlvl
10-14: leocrotta, trolls. Dlvl 15-24: death rays, rock trolls, minotaurs, drowning by eels, petrification. At the
castle: minotaurs (16 %), eels. Gehennom: cockatrice corpses (9 %), Juiblex, Demogorgon, touch of death. The Planes:
Pestilence, Famine, Death.

**4. Role.** Adjusted for skill, against the Archeologist:

| Role | Ascension, OR [95 % CI] | Reaching Dlvl 10, OR |
|---|---|---|
| Barbarian | 2.58 [2.19-3.03] | 2.57 |
| Valkyrie | 2.46 [2.12-2.86] | 2.80 |
| Samurai | 2.22 | |
| Caveman | 1.96 | |
| Ranger | 1.87 | |
| Monk | | 2.81 |
| Wizard | 1.15 [0.99-1.34], the weakest | 0.76 |

The raw ascension rate is 0.7 % for Wizards and 3.5 % for Cavemen.

By role and race (`roles_table.txt`), the dwarven Valkyrie ascends most often in three of the four experience
groups: 0.60 % (no earlier ascensions), 5.79 % (1-4), 9.87 % (5-19), 27.95 % (20+).

**5. Sokoban and Mines' End.** Among games that already killed Medusa (one checkpoint), adjusted for skill and role:

| Set | Sokoban prize, OR | Mines' End luckstone, OR |
|---|---|---|
| 3.4 + 3.6 | 1.19 [1.09-1.30] | 1.48 [1.34-1.62] |
| 3.6 only | 1.41 [1.20-1.65] | 1.38 [1.22-1.56] |

Raw: 61 % against 55 %, and 62 % against 52 %.

## Limits

- **Correlation, not cause.** "Solved Sokoban" may mark a careful style of play rather than cause success.
- **A collider.** Conditioning on "killed Medusa" selects players: strong roles (Valkyrie, Barbarian, Monk) carry weak
  players to Medusa too, so in a model conditioned on Medusa those roles look worse.
- **No order of events or pace in the xlogfile.** It cannot tell when Sokoban was solved, how fast the descent was, or
  how often the player prayed. That needs dumplogs or the ttyrecs of NLD-NAO, which is what [the human games
  extraction](../docs/human-data.md) does.
- **The experience measure is not monotonic.** Accounts with 1000+ games ascend less often, probably grinders or bots.
  It is a control variable in the models.
