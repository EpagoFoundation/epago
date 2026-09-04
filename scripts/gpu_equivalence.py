#!/usr/bin/env python
"""Does sharding a task set across GPU replicas change the score?

This is the measurement that decides whether multi-GPU eval may ship. The
design's claim is that replication changes only *where* an episode runs, never
*what* it produces — so a sharded run must not disagree with a one-replica run
by any more than one-replica runs disagree with each other.

Runs one model over one task set at several replica counts, compares every run
against the same one-replica reference, and times each configuration, so one
invocation yields both the agreement table and the speedup table.

**Always pass ``--baseline``, and read the comparison against it.** Whether a
sharded run is "the same" is not a question that has a bit-identity answer on
real hardware: a quantized MoE checkpoint under vLLM is not reproducible even
against itself — the same prompt, same engine, batch of one, greedy, seeded,
``enforce_eager`` on and prefix caching off, still decodes differently run to
run, because the fused MoE kernels reduce in nondeterministic order. So "0
divergences" is not an available result for ANY configuration, and a test that
demands it would fail a single-GPU validator too.

The question that does have an answer is the one the network already asks in
:func:`epago.eval.duel.run_calibration_duel`: **does replication add noise
beyond the noise this box already has?** ``--baseline`` runs the one-replica
configuration twice, which measures the box's own floor; every sharded
configuration is then compared against the same reference run in the same
units. If the 1-vs-K score-gap standard error sits at the 1-vs-1 floor,
replication contributed nothing, and the validator's noise floor already
prices it.

The score-gap standard error printed here is exactly what a calibration duel
feeds into the adaptive acceptance floor, so it can be read against
``constants.CROSS_GPU_NOISE_BUDGET``.

Run it the way a validator scores a duel::

    .venv-eval/bin/python scripts/gpu_equivalence.py \\
        --model /models/tongyi-30b-awq4 --corpus data/corpus-med-10k/corpus.db \\
        --devices 0,1,2,3 --replicas 1,2,4 --n 128 --baseline

Adding ``EPAGO_VLLM_DETERMINISTIC=1 --concurrency 1`` runs the lowest-noise
configuration the harness has (no CUDA graphs, batch of one, so batch shape is
not a variable). It lowers the floor; it does not reach zero.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago.environment.corpus import SqliteCorpus  # noqa: E402
from epago.environment.services import ResearchEnvironment  # noqa: E402
from epago.eval.pool import GpuPool, SweepRequest  # noqa: E402
from epago.taskgen.generator import generate_tasks  # noqa: E402


def run_config(model, devices, tasks, env, concurrency):
    """One sweep of ``tasks`` over ``len(devices)`` replicas of ``model``."""
    pool = GpuPool(devices)
    phases = (("sweep", tuple(tasks)),)
    started = time.monotonic()
    try:
        out = pool.run_sweeps(
            [SweepRequest(Path(model), phases, label="model")],
            session_factory=env.tools_for_task,
            concurrency=concurrency,
        )
        wall = time.monotonic() - started
        if out[0].error is not None:
            raise out[0].error
        results = out[0].phases["sweep"]
        stats = pool.stats.snapshot()
    finally:
        pool.close()
    return results, wall, stats


def compare(reference, other):
    """Agreement between two runs, in the units a calibration duel uses.

    ``gap_se`` is ``stdev(d_i)/sqrt(n)`` over the per-task correctness
    differences — the standard error of the mean paired difference, which is
    the quantity a duel's verdict actually turns on and the quantity
    :func:`epago.eval.duel.run_calibration_duel` feeds into the noise floor.
    """
    n = len(reference)
    same_answer = sum(1 for a, b in zip(reference, other) if (a.answer or "") == (b.answer or ""))
    same_correct = sum(1 for a, b in zip(reference, other) if a.correct == b.correct)
    d = [int(b.correct) - int(a.correct) for a, b in zip(reference, other)]
    gap_se = statistics.stdev(d) / (n ** 0.5) if n > 1 else 0.0
    diffs = [
        (a.task_id, a.answer, b.answer)
        for a, b in zip(reference, other)
        if (a.answer or "") != (b.answer or "")
    ]
    return {
        "n": n,
        "identical_answers": same_answer,
        "identical_correct": same_correct,
        "mean_paired_diff": sum(d) / n if n else 0.0,
        "gap_se": gap_se,
        "answer_diffs": diffs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--release", default="SCI2")
    ap.add_argument("--n", type=int, default=24, help="tasks in the fixed set")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--devices", default="", help="CUDA devices, e.g. 2,3,4,5 (default: all)")
    ap.add_argument("--replicas", default="1,2,4", help="replica counts to measure")
    ap.add_argument("--concurrency", type=int, default=None, help="rollouts in flight per replica")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="repeat the one-replica run, to separate real divergence from engine noise",
    )
    ap.add_argument("--json", type=Path, default=None, help="write the raw numbers here")
    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        from epago.eval.pool import resolve_devices

        devices = list(resolve_devices())
    counts = [int(c) for c in args.replicas.split(",") if c.strip()]
    if max(counts) > len(devices):
        raise SystemExit(f"need {max(counts)} devices, have {devices}")

    corpus = SqliteCorpus(args.corpus)
    env = ResearchEnvironment(corpus)
    tasks = generate_tasks(
        seed=args.seed, release=args.release, corpus=corpus, n=args.n, king_probe=None
    )
    print(
        f"{len(tasks)} tasks · devices {devices} · replicas {counts} · "
        f"concurrency {args.concurrency} · deterministic="
        f"{os.environ.get('EPAGO_VLLM_DETERMINISTIC', '')!r}",
        flush=True,
    )

    runs: dict[str, dict] = {}
    order = [(c, f"r{c}") for c in counts]
    if args.baseline:
        order.insert(1, (counts[0], f"r{counts[0]}-repeat"))
    for count, name in order:
        results, wall, stats = run_config(
            args.model, devices[:count], tasks, env, args.concurrency
        )
        acc = sum(r.correct for r in results) / len(results)
        runs[name] = {
            "replicas": count,
            "wall_s": wall,
            "load_s": stats.load_seconds,
            "generate_s": stats.generate_seconds,
            "accuracy": acc,
            "results": results,
        }
        print(
            f"  {name:>12}: {wall:7.1f}s wall  ({stats.load_seconds:6.1f}s load, "
            f"{stats.generate_seconds:8.1f}s replica-generate)  solve {acc:.1%}",
            flush=True,
        )

    ref_name = f"r{counts[0]}"
    reference = runs[ref_name]["results"]
    baseline_name = f"r{counts[0]}-repeat"
    print("\n" + "=" * 78)
    print(f"AGREEMENT vs {ref_name} ({len(tasks)} tasks)")
    print(f"  {'run':>14}  {'same ans':>9}  {'same corr':>10}  {'mean d':>8}  {'gap SE':>8}  solve")
    report: dict = {"tasks": len(tasks), "devices": devices, "runs": {}}
    for name, run in runs.items():
        if name == ref_name:
            continue
        c = compare(reference, run["results"])
        print(
            f"  {name:>14}  {c['identical_answers']:>4}/{c['n']:<4}  "
            f"{c['identical_correct']:>5}/{c['n']:<4}  "
            f"{c['mean_paired_diff']:>+8.3f}  {c['gap_se']:>8.4f}  "
            f"{runs[ref_name]['accuracy']:.1%} -> {run['accuracy']:.1%}"
        )
        report["runs"][name] = {
            "replicas": run["replicas"],
            "wall_s": run["wall_s"],
            "accuracy": run["accuracy"],
            **{k: v for k, v in c.items() if k != "answer_diffs"},
        }
    if baseline_name in report["runs"]:
        floor = report["runs"][baseline_name]["gap_se"]
        print(
            f"\n  The {baseline_name} row is this box's own floor: one replica against "
            f"itself.\n  Replication is free of extra noise when every sharded row's gap "
            f"SE sits at\n  or below {floor:.4f}. (CROSS_GPU_NOISE_BUDGET = "
            f"{2 / 400:.4f}.)"
        )

    print("\nSPEEDUP (wall clock, load included)")
    base = runs[ref_name]["wall_s"]
    print(f"  {'replicas':>9}  {'wall_s':>9}  {'speedup':>8}  {'efficiency':>10}")
    for name, run in runs.items():
        if name.endswith("-repeat"):
            continue
        n = run["replicas"]
        speedup = base / run["wall_s"]
        print(f"  {n:>9}  {run['wall_s']:>9.1f}  {speedup:>8.2f}x  {speedup / n:>9.0%}")
    report["runs"][ref_name] = {
        "replicas": counts[0],
        "wall_s": base,
        "accuracy": runs[ref_name]["accuracy"],
    }
    print("=" * 78)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    if baseline_name not in report["runs"]:
        print("no --baseline: nothing to compare the sharded rows against")
        return 0
    floor = report["runs"][baseline_name]["gap_se"]
    sharded = [
        r["gap_se"] for name, r in report["runs"].items()
        if name not in (ref_name, baseline_name)
    ]
    if not sharded:
        return 0
    print(
        "NO EXTRA NOISE FROM REPLICATION"
        if max(sharded) <= floor
        else f"replication gap SE up to {max(sharded):.4f} against a {floor:.4f} floor"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
