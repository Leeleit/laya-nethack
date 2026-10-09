"""One role's rows of a dataset: the rows whose state starts with the prefix ("Val-": the
--terse state's first line is the character, "Val-Dwa-Law-Fem 3.4.3") -> another dataset dir, every split; meta.json (with
this subset's row counts) and skill.txt copied (the questions, bins and labels stay the source's -- models trained on either answer the same ones).
  python subset_role.py human_mix17 human_mix17v Val-
  python subset_role.py human_mix17 human_mix17o '!Val-'     # "!": every row but these (the other roles)
"""
import json
import shutil
import sys
from pathlib import Path


def main():
    src, dst, prefix = sys.argv[1:4]
    Path(dst).mkdir(parents=True, exist_ok=True)
    kept = {}
    for split in ("train", "val", "test"):
        n = k = 0
        with open(Path(src, f"{split}.jsonl"), encoding="utf-8") as f, \
                open(Path(dst, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as o:
            for line in f:
                n += 1
                if json.loads(line)["state"].startswith(prefix.lstrip("!")) != prefix.startswith("!"):
                    o.write(line)
                    k += 1
        print(f"{split}: {n} rows, {k} {prefix}")
        kept[split] = k
        assert k, f"no {prefix} rows in {split}"
    if Path(src, "skill.txt").exists():
        shutil.copy(Path(src, "skill.txt"), Path(dst, "skill.txt"))
    meta = json.loads(Path(src, "meta.json").read_text())  # the source's, with this subset's row counts
    meta["rows"], meta["subset_of"] = kept, f"{src} {prefix}"
    Path(dst, "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
