"""Would a laya risk alarm fire earlier than the bot's HP rule before a death, at the same false-alarm budget?

Test games of v2, the l20 checkpoint calibrated with its BCTS fit (results/compare_layers.json).
  HP rule  -- the bot's emergency trigger (agent.py emergency_strategy: quaff at HP < max/3 or HP < 8);
  laya     -- P(death within 7 turns) > tau, tau set so both raise the same share of alarms on safe states
              (death more than 100 turns away).
For every death: was there an alarm in the last 7 / 30 turns, and how many turns before death was the first one
(within the last 100 turns)?  -> results/alarm_v2.txt
"""
import json

import numpy as np
import pandas as pd

from compare_v2 import bucket_probs

D = "/data/laya/nethack/trace/"
f = pd.read_parquet(D + "v2_data/features.parquet")
te = f[f.split == "test"].reset_index(drop=True)
te["game"] = te.id.str.split("#").str[0]
cal = json.load(open(D + "results/compare_layers.json"))["calibration"]["risk-v2-l20"]
z = np.stack(te[["id"]].merge(pd.read_parquet(D + "preds/risk-v2-l20-test.parquet"), on="id")["logits"].to_numpy())
te["p7"] = bucket_probs(z, cal["T"], np.array(cal["bias"]))[:, :2].sum(1)
hp = te.hp_frac * te.maxhp
te["hp_rule"] = (te.hp_frac < 1 / 3) | (hp < 8)

safe = te.ttd > 100
budget = te.hp_rule[safe].mean()
tau = float(np.quantile(te.p7[safe], 1 - budget))
te["laya"] = te.p7 > tau
deaths = te[te.ttd <= 100].groupby("game")
out = [f"test: {te.game.nunique()} games, {len(te)} states; deaths with states in their last 100 turns: {deaths.ngroups}",
       f"alarm budget on safe states (HP rule): {budget:.4f} -> laya tau P(die<=7) > {tau:.4f}"]
for name in ("hp_rule", "laya"):
    rec7 = deaths.apply(lambda g: bool(g[g.ttd <= 7][name].any())).mean()
    rec30 = deaths.apply(lambda g: bool(g[g.ttd <= 30][name].any())).mean()
    first = deaths.apply(lambda g: g.ttd[g[name]].max() if g[name].any() else np.nan)
    out.append(f"{name:8s} alarm in last 7 turns: {rec7:.3f}  last 30: {rec30:.3f}  "
               f"first alarm, turns before death: median {np.nanmedian(first):.0f} (deaths never alarmed: {first.isna().mean():.3f})")
only = deaths.apply(lambda g: bool(g[g.ttd <= 30].laya.any() and not g[g.ttd <= 30].hp_rule.any())).mean()
both_first = deaths.apply(lambda g: (g.ttd[g.laya].max() if g.laya.any() else -1) - (g.ttd[g.hp_rule].max() if g.hp_rule.any() else -1))
out.append(f"deaths where laya alarmed in the last 30 turns and the HP rule did not: {only:.3f}")
out.append(f"laya first alarm earlier than HP rule's: {(both_first > 0).mean():.3f}, later: {(both_first < 0).mean():.3f}, "
           f"median lead (turns, both fired): {np.median(both_first[(both_first != 0)]) if (both_first != 0).any() else 0:.0f}")
print("\n".join(out))
open(D + "results/alarm_v2.txt", "w").write("\n".join(out) + "\n")
