"""Answer the held-out risk states with a laya checkpoint.

  python eval_laya.py <checkpoint dir> <out.parquet>   -> columns id, p (laya's raw P(true))
"""
import json
import sys
import time

import laya
import pandas as pd

DATA = "/data/laya/nethack/trace/laya_data/test.jsonl"


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    rows = [json.loads(x) for x in open(DATA)]
    questions = rows[0]["questions"]
    agent = laya.load(ckpt, device="cuda")
    t0 = time.time()
    res = agent.predict_batch([r["state"] for r in rows], questions, batch_size=128, sort_by_length=True)
    dt = time.time() - t0
    pd.DataFrame({"id": [r["id"] for r in rows], "p": [x["answers"]["die_soon"]["noul"] for x in res]}).to_parquet(out)
    print(f"{len(rows)} states in {dt:.0f}s ({1000 * dt / len(rows):.2f} ms/state) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
