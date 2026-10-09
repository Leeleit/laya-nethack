"""A share of a dataset's games, every row of a kept game (the learning curve). A game
is kept when crc32 of its id's "name@start" falls in buckets [offset, offset + share) of 1000, so with the same offset
a smaller share is a subset of a larger one, and offset 250 with share 0.25 is a quarter disjoint from offset 0's
(the noise of the curve).
  python subset_games.py <in.jsonl> <out.jsonl> <share> [<offset buckets>]
"""
import sys
import zlib

share = float(sys.argv[3])
offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0
kept = rows = 0
with open(sys.argv[1], encoding="utf-8") as f, open(sys.argv[2], "w", encoding="utf-8", newline="\n") as out:
    for line in f:
        rows += 1
        game = line.split('"id": "', 1)[1].split("#", 1)[0]
        if offset <= zlib.crc32(game.encode()) % 1000 < offset + share * 1000:
            out.write(line)
            kept += 1
print(f"{kept} of {rows} rows")
