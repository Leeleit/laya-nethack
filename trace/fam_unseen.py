"""fam_probe's sets: no fam state in the train of c17r / c17a, no new game in either train, the same questions as the
training set.
  python fam_unseen.py human_fam23 human_dwa20/train.jsonl human_dwa21/train.jsonl
"""
import json
import re
import sys
from pathlib import Path

probe, trains = sys.argv[1], sys.argv[2:]
ids = {g: [json.loads(x)["id"] for x in open(Path(probe, f"{g}.jsonl"), encoding="utf-8")]
       for g in ("fam13", "fam12", "new13", "new12")}
for s in trains:
    seen = {json.loads(x)["id"] for x in open(s, encoding="utf-8")}
    games = {i.split("#")[0] for i in seen}
    print(s, "  ".join(f"{g}: ids seen {sum(i in seen for i in v)}, games in train {sum(i.split('#')[0] in games for i in v)}"
                       for g, v in ids.items()))
    a = json.loads(open(Path(probe, "fam13.jsonl"), encoding="utf-8").readline())["questions"]
    b = json.loads(open(s, encoding="utf-8").readline())["questions"]
    bare = lambda s: re.sub(r"\([^)]*\)", "", s)  # the player's skill differs by row
    print("  questions equal:", {k: a[k]["criteria"] == b[k]["criteria"]
                                 and bare(a[k]["instructions"]) == bare(b[k]["instructions"]) for k in a})
