"""Parse NAO (NLD-NAO) and Hardfought xlogfiles into one parquet table, one row per game."""
import glob, os, sys

import pandas as pd

DATA = "/data/laya/nethack/data"
INT = ["points", "deathdnum", "deathlev", "maxlvl", "hp", "maxhp", "deaths", "deathdate", "birthdate",
       "turns", "realtime", "starttime", "endtime"]
HEX = ["conduct", "achieve", "flags"]


def parse_line(line):
    line = line.rstrip("\n")
    sep = "\t" if "\t" in line else ":"
    rec, key = {}, None
    for tok in line.split(sep):
        k, eq, v = tok.partition("=")
        if eq and k.isidentifier():
            key = k
            rec[k] = v
        elif key:  # a separator inside a value (death reasons in ':'-separated 3.4.3 lines)
            rec[key] += sep + tok
    return rec


def load(path, source):
    with open(path, encoding="latin-1") as f:
        df = pd.DataFrame.from_records(parse_line(l) for l in f if l.strip())
    df["source"] = source
    return df


def main():
    files = [(p, "nao") for p in sorted(glob.glob(f"{DATA}/nao/xlogfile.*")) if not p.endswith("blacklist.txt")]
    files += [(p, "hardfought-" + os.path.basename(p).split(".")[0]) for p in sorted(glob.glob(f"{DATA}/hf-*.xlog"))]
    frames = []
    for p, src in files:
        df = load(p, src)
        df["file"] = os.path.basename(p)
        print(f"{os.path.basename(p):22s} {len(df):9d} rows", flush=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in INT:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    for c in HEX:
        df[c] = df.get(c).map(lambda v: int(v, 16) if isinstance(v, str) and v.startswith("0x") else pd.NA).astype("Int64")
    keep = ["version", "source", "file", "name", "role", "race", "gender", "align", "death"] + INT + HEX
    df = df[keep]
    out = f"{DATA}/games.parquet"
    df.to_parquet(out, index=False)
    print(f"total {len(df)} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
