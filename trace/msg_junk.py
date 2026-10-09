"""Which messages of a --terse human dataset look like interface leftovers: every message of the message
line (" | "-separated) with digits masked, counted; prints the rare-word / short / prompt-like ones with their share.
  python msg_junk.py <data.jsonl> [n states]
"""
import collections
import json
import re
import sys

n = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
cnt = collections.Counter()
states = with_msg = 0
seen = set()
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    st = r["state"]
    if st in seen:
        continue
    seen.add(st)
    states += 1
    lines = st.split("\n")
    msg = next((l for l in lines[2:] if not l.startswith(("Around:", "Inventory:", "Bot:", "St:"))), None)
    if msg is None or msg.startswith(("HP ",)):
        continue
    with_msg += 1
    for m in msg.split(" | "):
        cnt[re.sub(r"\d+", "N", m.strip())] += 1
    if states >= n:
        break
total = sum(cnt.values())
print(f"{states} states, {with_msg} with messages, {total} messages, {len(cnt)} distinct")
SUSPECT = re.compile(r"^(\S{1,3}|N|.*\?$|Call .*|\[.*\]|Specify.*|Talk.*|Pick.*|Where.*|What.*|Unknown.*|In what.*"
                     r"|Done\.|Status.*|Really.*|.*:$|.*--More--.*)$")
sus = [(m, c) for m, c in cnt.most_common() if SUSPECT.match(m)]
print(f"suspect: {sum(c for _, c in sus)} messages ({100 * sum(c for _, c in sus) / total:.2f} %)")
for m, c in sus[:80]:
    print(f"{c:6d}  {m[:110]}")
print("---- top 40 overall")
for m, c in cnt.most_common(40):
    print(f"{c:6d}  {m[:110]}")
