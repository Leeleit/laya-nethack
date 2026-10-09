"""Build-time patch of the arena inside our laya-frozen image (the pinned arena image is untouched).

1. run.py: every finished episode's full result is appended to /out/results.partial.jsonl at once, so an
   interrupted collection keeps its finished games (the arena writes results.json only after the whole batch).
2. trajectory.py: the trajectory id (= seed) goes into the environment before the bot process is spawned, so
   leader-traced/laya_trace.py names the trace <seed>.jsonl instead of <pid>-<episode>.jsonl.
Game code and seeding are unchanged: runs stay identical to the plain image.
"""
import importlib.util
import py_compile

import nethackers.arena.run as run
import nethackers.arena.trajectory as trajectory


def patch(module, pairs):
    path = module.__file__
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    open(path, "w").write(s)


NL = chr(10)
patch(run, [("            results[i] = result" + NL,
             "            results[i] = result" + NL
             + "            if os.path.isdir('/out'):  # laya: keep each finished episode at once" + NL
             + "                with open('/out/results.partial.jsonl', 'a') as fh:" + NL
             + "                    fh.write(json.dumps(result.to_dict()) + chr(10))" + NL)])
patch(trajectory, [("    started = time.monotonic()" + NL,
                    "    started = time.monotonic()" + NL
                    + "    __import__('os').environ['LAYA_TRAJECTORY_ID'] = str(spec.trajectory_id)  # laya: trace name" + NL)])
for m in (run, trajectory):  # compile now: a broken patch fails the build, and no stale .pyc is left behind
    py_compile.compile(m.__file__, cfile=importlib.util.cache_from_source(m.__file__), doraise=True)
