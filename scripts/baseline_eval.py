#!/usr/bin/env python
"""Phase-1 baseline: run a real model over a real exam, report the two numbers
everything else waits on.

  1. **Solve rate** — overall and per template. The duel discriminates best
     when the incumbent solves 45-65% of tasks; far outside that band the exam
     must be retuned before any competition is meaningful.
  2. **Wall-clock per task** — which sets the round budget. 400 tasks and up
     to ``ROUND_MAX_ENTRANTS`` challengers must fit inside a 2-day cadence.

Runs the SAME pinned harness a duel uses (:mod:`epago.eval.harness`), so the
measured numbers are the numbers a competition would see — not a proxy.

Usage:
    .venv-eval/bin/python scripts/baseline_eval.py \\
        --model /path/to/model \\
        --corpus <corpus.db> --release SCI2 --n 200 \\
        [--seed 90210] [--out baseline.json] [--concurrency 16]

The seed is arbitrary but fixed: re-running with the same corpus, seed and
release replays the identical exam, so two baselines are comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago.environment.corpus import SqliteCorpus  # noqa: E402
from epago.environment.services import ResearchEnvironment  # noqa: E402
from epago.eval.harness import run_rollouts_batched  # noqa: E402
from epago.taskgen.generator import generate_tasks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="local model snapshot dir")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--release", default="SCI2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=90210)
    ap.add_argument("--out", type=Path, default=Path("baseline.json"))
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    corpus = SqliteCorpus(args.corpus)
    env = ResearchEnvironment(corpus)
    print(f"corpus: {corpus.doc_count()} docs   release: {args.release}", flush=True)

    t0 = time.time()
    tasks = generate_tasks(
        seed=args.seed, release=args.release, corpus=corpus, n=args.n, king_probe=None
    )
    print(f"minted {len(tasks)} tasks in {time.time()-t0:.0f}s "
          f"({dict(Counter(t.template for t in tasks))})", flush=True)

    from epago.eval.backend import VllmBackend

    t0 = time.time()
    backend = VllmBackend(args.model)
    print(f"engine up in {time.time()-t0:.0f}s", flush=True)

    done: list = []
    t_run = time.time()

    def on_result(i: int, res) -> None:
        done.append(res)
        if len(done) % 10 == 0:
            elapsed = time.time() - t_run
            rate = sum(r.correct for r in done) / len(done)
            print(f"  {len(done):4d}/{len(tasks)}  solve={rate:5.1%}  "
                  f"{elapsed/len(done):5.1f}s/task", flush=True)

    results = run_rollouts_batched(
        backend, tasks, env.tools_for_task, llm_judge=None,
        concurrency=args.concurrency, on_result=on_result,
    )
    wall = time.time() - t_run

    by_template: dict[str, list] = {}
    for task, res in zip(tasks, results):
        by_template.setdefault(task.template, []).append(res)

    turns = [r.turns for r in results]
    times = [r.wall_time_s for r in results]
    errors = [r for r in results if r.error]
    overall = sum(r.correct for r in results) / max(len(results), 1)

    print()
    print("=" * 64)
    print(f"SOLVE RATE overall : {overall:6.1%}   (duel band: 45-65%)")
    for tmpl, rs in sorted(by_template.items()):
        print(f"  {tmpl:22s} {sum(r.correct for r in rs)/len(rs):6.1%}  (n={len(rs)})")
    print(f"TIMING  wall total : {wall/60:.1f} min for {len(results)} tasks "
          f"(concurrency {args.concurrency})")
    print(f"        per task   : mean {statistics.mean(times):.1f}s  "
          f"p50 {statistics.median(times):.1f}s  max {max(times):.1f}s")
    print(f"        turns      : mean {statistics.mean(turns):.1f}  max {max(turns)}")
    print(f"ERRORS             : {len(errors)}")
    exam_wall_min = wall / 60 * (400 / max(len(results), 1))
    print(f"PROJECTED          : one 400-task sweep ≈ {exam_wall_min:.0f} min; "
          f"a 12-entrant round ≈ {exam_wall_min * 13 / 60:.1f} h")
    print("=" * 64)

    args.out.write_text(json.dumps({
        "model": str(args.model), "corpus": str(args.corpus),
        "release": args.release, "seed": args.seed, "n": len(results),
        "solve_rate": overall,
        "by_template": {
            t: sum(r.correct for r in rs) / len(rs) for t, rs in by_template.items()
        },
        "wall_s": wall, "mean_task_s": statistics.mean(times),
        "mean_turns": statistics.mean(turns), "errors": len(errors),
        "answers": [
            {"task_id": t.task_id, "template": t.template, "expected": t.answer,
             "got": r.answer, "correct": r.correct, "turns": r.turns,
             "judge_tier": r.judge_tier, "error": r.error}
            for t, r in zip(tasks, results)
        ],
    }, indent=1))
    print(f"full record -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
