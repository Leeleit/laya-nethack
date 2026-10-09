"""Run the traced bot on fresh validation seeds and keep its decision traces.

Same image, sandbox and seed derivation as `nethackers eval` (this calls its eval_batch); the
only difference is that the box's /out mount is a persistent directory with a trace/ subdir,
which switches leader-traced/laya_trace.py on. Nothing leaves this machine.

Interruptible: in the laya-frozen image (image/patch_arena.py) every finished game is appended to
results.partial.jsonl at once and its trace is named <seed>.jsonl. Rerunning the same command resumes:
finished seeds are kept, the traces of unfinished ones are deleted and those seeds are played again
(games are deterministic). results.json / evidence.json are written when every seed has a result.

  python collect.py <identity> <n> [--start 1000] [--parallel 16] [--root runs/traces] [--image laya-frozen]
  -> <root>/<identity>-<start>-<n>/{evidence.json, results.json, results.partial.jsonl, trace/*.jsonl}
"""
import argparse
import contextlib
import dataclasses
import datetime
import json
import sys
from pathlib import Path

from nethackers.eval import runner
from nethackers.harness.sandbox_preflight import resolve_image
from nethackers.harness.seeds import validation_spec

HERE = Path(__file__).resolve().parent  # beside the arena's source (PYTHONPATH=../arena-src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("identity")
    ap.add_argument("n", type=int)
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--parallel", type=int, default=16, help="games at once; 16: the throughput knee on 32 vCPUs")
    ap.add_argument("--solution", default=str(HERE / "leader-traced"))
    ap.add_argument("--root", default="runs/traces", help="relative to the nethackers dir")
    ap.add_argument("--branch", default=None,
                    help="branch.json for leader-traced/laya_branch.py (plan_branches.py): forced decisions per game")
    ap.add_argument("--image", default="laya-frozen",
                    help="laya-frozen = the arena image with the wall clock frozen by libfaketime (NetHack reads the "
                         "real hour in play) and per-game results; 'arena' = the pinned image (no resume)")
    a = ap.parse_args()

    out = HERE / a.root / f"{a.identity}-{a.start}-{a.n}"
    if (out / "evidence.json").exists():
        print(out, "already complete")
        return 0
    (out / "trace").mkdir(parents=True, exist_ok=True)
    for p in (out, out / "trace"):
        p.chmod(0o777)  # the box writes as uid 65534 (also /out/branch_log.jsonl, results.partial.jsonl)
    runner._eval_temp_dir = lambda: contextlib.nullcontext(str(out))
    if a.branch:
        spec_text = Path(a.branch).read_text()
        old = out / "branch.json"
        assert not old.exists() or old.read_text() == spec_text, "resuming with a different branch spec"
        old.write_text(spec_text)

    partial = out / "results.partial.jsonl"
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            with contextlib.suppress(json.JSONDecodeError):  # a line cut by the interruption
                r = json.loads(line)
                done[r["trajectory_id"]] = r
    spec = validation_spec(a.identity, n=a.n, start=a.start)
    todo = tuple(x for x in spec.batch if x[0] not in done)
    for t in (out / "trace").iterdir():  # traces of games that did not finish
        if t.name.split(".")[0] not in {str(s) for s in done}:
            t.unlink()
    if done:
        print(f"resuming: {len(done)} finished, {len(todo)} to play", flush=True)

    evidence = None
    if todo:
        image = resolve_image(None, "arena") if a.image == "arena" else a.image
        evidence = runner.eval_batch(Path(a.solution), dataclasses.replace(spec, batch=todo), image,
                                     now=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                     max_parallel_evals=a.parallel).to_dict()
        for r in evidence["results"]:
            done[r["trajectory_id"]] = r  # the arena's own record wins over the partial line
    results = [done[s] for s, _ in spec.batch]
    (out / "results.json").unlink(missing_ok=True)  # the box's own (uid 65534, 644): only this batch's seeds
    (out / "results.json").write_text(json.dumps(results))
    ev = evidence or {}
    ev["results"] = results
    ev["mean_progress"] = sum(r["progress"] for r in results) / len(results)
    (out / "evidence.json").write_text(json.dumps(ev, indent=1))
    print(out, ev["mean_progress"])


if __name__ == "__main__":
    sys.exit(main())
