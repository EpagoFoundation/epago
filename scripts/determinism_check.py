#!/usr/bin/env python
"""How far apart do two runs of the SAME model land on the same tasks?

Runs one model over the same tasks TWICE and compares. The answer is never
"identical": ``EPAGO_VLLM_DETERMINISTIC=1`` with ``--concurrency 1`` is the
lowest-noise configuration the harness has (no CUDA graphs, batch of one, so
batch shape is not a variable) and it still does not tie, because the fused MoE
kernels reduce in a nondeterministic order below anything that flag controls.
Measured on the reference stack, a 400-task self-comparison disagreed on 84
tasks (21%).

So read this as a noise measurement, not a pass/fail gate. It lowers the floor;
it does not reach zero, and nothing downstream needs it to — the calibration
duel measures whatever remains and the adaptive delta prices it. For the number
the validator actually clamps on (the paired score-gap standard error), use
``scripts/calibrate_noise.py``.

Usage:
    EPAGO_VLLM_DETERMINISTIC=1 .venv-eval/bin/python scripts/determinism_check.py \\
        --model /models/tongyi-30b-awq4 --corpus data/corpus-med-10k/corpus.db \\
        --n 12 --concurrency 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago.environment.corpus import SqliteCorpus  # noqa: E402
from epago.environment.services import ResearchEnvironment  # noqa: E402
from epago.eval.harness import run_rollouts_batched  # noqa: E402
from epago.taskgen.generator import generate_tasks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--release", default="SCI2")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    corpus = SqliteCorpus(args.corpus)
    env = ResearchEnvironment(corpus)
    tasks = generate_tasks(seed=args.seed, release=args.release, corpus=corpus,
                           n=args.n, king_probe=None)
    print(f"{len(tasks)} tasks · concurrency {args.concurrency}", flush=True)

    from epago.eval.backend import VllmBackend
    backend = VllmBackend(args.model)

    def sweep(tag):
        res = run_rollouts_batched(backend, tasks, env.tools_for_task,
                                   llm_judge=None, concurrency=args.concurrency)
        acc = sum(r.correct for r in res) / len(res)
        print(f"  run {tag}: solve {acc:.1%}", flush=True)
        return res

    a = sweep("A")
    b = sweep("B")

    same_answer = sum(1 for x, y in zip(a, b) if (x.answer or "") == (y.answer or ""))
    same_correct = sum(1 for x, y in zip(a, b) if x.correct == y.correct)
    diffs = [(t.task_id, x.answer, y.answer) for t, x, y in zip(tasks, a, b)
             if (x.answer or "") != (y.answer or "")]

    print()
    print("=" * 60)
    print(f"identical answers : {same_answer}/{len(tasks)}")
    print(f"identical correct : {same_correct}/{len(tasks)}")
    acc_a = sum(r.correct for r in a) / len(a)
    acc_b = sum(r.correct for r in b) / len(b)
    print(f"solve A vs B      : {acc_a:.1%} vs {acc_b:.1%}   (gap {abs(acc_a-acc_b)*100:.1f} pp)")
    if diffs:
        print("\ndivergences:")
        for tid, x, y in diffs[:8]:
            print(f"  {tid}: {str(x)[:40]!r} != {str(y)[:40]!r}")
    print("=" * 60)
    print(f"per-task answer disagreement: {len(tasks) - same_answer}/{len(tasks)} "
          f"({(len(tasks) - same_answer) / len(tasks):.0%}) — expected nonzero; "
          "compare against calibrate_noise.py for the clamped floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
