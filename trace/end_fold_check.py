"""end_fold.py on random logits: A = B -> every gain 0; B with end widened by fake classes -> runs.
  python end_fold_check.py human_data13d
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = sys.argv[1]
rows = [json.loads(line) for line in open(Path(d, "val.jsonl"), encoding="utf-8")]
rng = np.random.default_rng(0)
recs = [{"id": r["id"], "question": q, "logits": rng.normal(size=len(r["questions"][q]["criteria"])).tolist()}
        for r in rows for q in r["questions"]]
Path("ckpt/efc").mkdir(exist_ok=True)
pd.DataFrame(recs).to_parquet("ckpt/efc/a.parquet")
subprocess.run([sys.executable, "end_fold.py", "--a", d, "--pa", "ckpt/efc/a.parquet", "--b", d, "--pb",
                "ckpt/efc/a.parquet"], check=True)
# B: end with two extra classes in front of "monster"
Path("ckpt/efc/b").mkdir(exist_ok=True)
with open("ckpt/efc/b/val.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        crit = dict(r["questions"]["end"]["criteria"])
        crit = {"Q1": "fake one", "Q2": "fake two", **crit}
        r["questions"]["end"]["criteria"] = crit
        f.write(json.dumps(r) + "\n")
recs_b = [dict(x, logits=([-9.0, -9.0] + x["logits"]) if x["question"] == "end" else x["logits"]) for x in recs]
pd.DataFrame(recs_b).to_parquet("ckpt/efc/b.parquet")
subprocess.run([sys.executable, "end_fold.py", "--a", d, "--pa", "ckpt/efc/a.parquet", "--b", "ckpt/efc/b", "--pb",
                "ckpt/efc/b.parquet"], check=True)
