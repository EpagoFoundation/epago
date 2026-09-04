#!/usr/bin/env python
"""Measure the real noise floor and the winning margin it implies.

Runs the king against itself on one fresh task set (two full sweeps of the
graded path a real duel takes). Because it is the same weights twice, any
per-task correctness flip is pure harness noise. The mean flip rate is the
noise floor the validator clamps the adaptive delta from below with.

This is the same measurement the running validator performs on a schedule
(``_maybe_calibrate`` -> ``add_noise_sample``); running it here once, before
genesis, seeds the first real round with a measured floor instead of the
static cross-GPU fallback. Pass ``--state DIR`` to write the sample into that
validator state's ``noise_floor_samples``.

Usage:
    .venv-eval/bin/python scripts/calibrate_noise.py \\
        --model models/tongyi-30b-awq4 --corpus data/corpus-med-10k/corpus.db \\
        --n 200 --state /var/lib/epago
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago import constants  # noqa: E402
from epago.core.stats import adaptive_delta, noise_floor_from_calibration  # noqa: E402
from epago.environment.corpus import SqliteCorpus  # noqa: E402
from epago.environment.services import ResearchEnvironment  # noqa: E402
from epago.eval.harness import run_rollouts_batched  # noqa: E402
from epago.taskgen.generator import generate_tasks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--release", default="SCI2")
    ap.add_argument("--n", type=int, default=constants.N_PUB_TASKS)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="0 = harness default (batched); 1 = one-at-a-time")
    ap.add_argument("--state", type=Path, default=None,
                    help="validator state dir to seed the measured sample into")
    args = ap.parse_args()

    corpus = SqliteCorpus(args.corpus)
    env = ResearchEnvironment(corpus)
    tasks = generate_tasks(seed=args.seed, release=args.release, corpus=corpus,
                           n=args.n, king_probe=None)
    ordered = sorted(tasks, key=lambda t: t.task_id)
    print(f"king-vs-king · {len(ordered)} tasks · release {args.release}", flush=True)

    from epago.eval.backend import VllmBackend
    backend = VllmBackend(args.model)

    kw = {"concurrency": args.concurrency} if args.concurrency else {}

    def sweep(tag: str):
        res = run_rollouts_batched(backend, ordered, env.tools_for_task, llm_judge=None, **kw)
        acc = sum(r.correct for r in res) / len(res)
        print(f"  sweep {tag}: king solve {acc:.1%}", flush=True)
        return res

    a = sweep("A")
    b = sweep("B")

    import statistics

    signed = [int(y.correct) - int(x.correct) for x, y in zip(a, b)]
    n = len(signed)
    flip_rate = sum(abs(d) for d in signed) / n          # per-question noise (informative)
    # The noise floor is the score gap's standard error -- the same quantity
    # run_calibration_duel feeds the validator. It is what bounds a coin-flip
    # coronation, and it shrinks with n; the per-question flip rate does not.
    sem = statistics.stdev(signed) / (n ** 0.5) if n > 1 else 0.0
    acc_a = sum(r.correct for r in a) / len(a)
    acc_b = sum(r.correct for r in b) / len(b)
    king_acc = (acc_a + acc_b) / 2.0

    floor = noise_floor_from_calibration([sem])
    # Implied margin a challenger must clear, at this king's accuracy.
    delta = adaptive_delta(king_acc, floor)
    headroom = constants.DELTA_C * (1.0 - king_acc)
    noise_term = constants.DELTA_NOISE_MULTIPLIER * floor

    print()
    print("=" * 60)
    print(f"per-question flip rate     : {flip_rate:.4f}  ({sum(abs(d) for d in signed)}/{n})")
    print(f"score gap A-B              : {(acc_b - acc_a) * 100:+.1f} pp")
    print(f"noise floor (score-gap SE) : {sem:.4f}")
    print(f"king accuracy              : {king_acc:.1%}")
    print(f"  headroom term  c*(1-acc) : {headroom:.4f}")
    print(f"  noise term  {constants.DELTA_NOISE_MULTIPLIER:g}*floor    : {noise_term:.4f}")
    print(f"  -> winning margin delta  : {delta:.4f}  ({'noise' if noise_term > headroom else 'headroom'}-bound)")
    print(f"static fallback floor      : {constants.CROSS_GPU_NOISE_BUDGET:.4f}")
    print("=" * 60)

    if args.state is not None:
        from epago.validator.state import ValidatorState
        st = ValidatorState.load(args.state)
        st.add_noise_sample(sem)
        st.save()
        print(f"seeded sample {sem:.4f} into {args.state}/state.json "
              f"({len(st.noise_floor_samples)} sample(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
