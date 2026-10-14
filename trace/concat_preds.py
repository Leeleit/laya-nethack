"""One predictions file from several (the same model on parts of a data set), in the given order, with a check that no
(id, question) repeats -- distill_build.py takes one file per teacher seed.
  python concat_preds.py preds/m20e-human_mix25-train.parquet preds/m20e-human_dwa24-train.parquet \
      preds/m20e-human_oth24n-train.parquet"""
import sys

import pandas as pd


def main():
    out, parts = sys.argv[1], sys.argv[2:]
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    assert not df.duplicated(["id", "question"]).any(), "an (id, question) in two parts"
    df.to_parquet(out)
    print(f"{len(df)} predictions -> {out}")


if __name__ == "__main__":
    main()
