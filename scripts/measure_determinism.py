#!/usr/bin/env python
"""M0: pre-launch determinism and cost measurement for the eval stack.

Runs K generated tasks through the real rollout harness TWICE with the pinned
backend and measures what the mechanism's math assumes:

* per-task **answer agreement** between the two passes (engine determinism);
* per-task **correctness disagreement** — the empirical same-box noise floor
  that ``delta``'s ``DELTA_NOISE_MULTIPLIER`` clamp must cover;
* **wall clock per rollout** and the projected GPU-hours of one full duel
  (``N_PUB_TASKS + N_PRIV_TASKS`` tasks x 2 models);
* with ``--compare other.json``: the **cross-hardware** disagreement rate
  between this box and another box's report (run the same command with the
  same ``--seed`` on both boxes, ship one JSON across).

The measured numbers decide launch parameters: the noise floor feeds the
``delta`` clamp (must stay well below the headroom term), and the GPU-hours
projection sizes validator hardware and the 48h SLA. See docs/TESTNET.md,
"Known unknowns to measure".

Backends:
  --backend vllm      the pinned production engine; requires --model-dir and a
                      GPU box with the eval extras installed. Without
                      torch/vllm this prints SKIP and exits 0 (never fails a
                      CPU-only CI lane).
  --backend scripted  deterministic scripted double; runs end-to-end anywhere
                      and doubles as this script's self-test: agreement must
                      be exactly 1.0 and the noise floor exactly 0.0.

Usage:
    # box A (GPU):
    .venv/bin/python scripts/measure_determinism.py --model-dir /models/seed --out a.json
    # box B (different GPU/driver), then cross-compare:
    .venv/bin/python scripts/measure_determinism.py --model-dir /models/seed \\
        --out b.json --compare a.json
    # CI self-test:
    .venv/bin/python scripts/measure_determinism.py --backend scripted
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

DEFAULT_TASKS = 40
DEFAULT_SEED = 1234
FIXTURE_DOCS = 240


def _scripted_backend(tasks):
    """Deterministic double: answers ~half the tasks correctly (stable by id)."""
    from epago.eval.backend import ScriptedBackend

    known = {
        t.question: t.answer
        for t in sorted(tasks, key=lambda t: t.task_id)[:: 2]
    }
    return ScriptedBackend(answers=known)


def _run_pass(backend, tasks, env):
    """One full pass over the sorted task list; returns per-task rollout rows."""
    from epago.eval.harness import run_rollout

    rows = []
    for task in tasks:
        result = run_rollout(backend, task, env.tools_for_task(task))
        rows.append(
            {
                "task_id": task.task_id,
                "answer": result.answer,
                "correct": bool(result.correct),
                "wall_time_s": result.wall_time_s,
                "turns": result.turns,
                "error": result.error,
            }
        )
    return rows


def _metrics(pass_a, pass_b):
    from epago import constants

    n = len(pass_a)
    disagreements = [
        a["task_id"] for a, b in zip(pass_a, pass_b) if a["answer"] != b["answer"]
    ]
    answer_agreement = (n - len(disagreements)) / n
    noise_floor = sum(
        1 for a, b in zip(pass_a, pass_b) if a["correct"] != b["correct"]
    ) / n
    # Paired score gap, the same statistic run_calibration_duel feeds the
    # adaptive floor: d_i in {-1, 0, +1} per task, then stdev/sqrt(n).
    diffs = [float(a["correct"]) - float(b["correct"]) for a, b in zip(pass_a, pass_b)]
    gap_se = (statistics.stdev(diffs) / (len(diffs) ** 0.5)) if len(diffs) > 1 else 0.0
    walls = [row["wall_time_s"] for row in pass_a + pass_b]
    mean_wall = statistics.fmean(walls)
    duel_rollouts = (constants.N_PUB_TASKS + constants.N_PRIV_TASKS) * 2
    return {
        "tasks": n,
        "answer_agreement": answer_agreement,
        "disagreement_task_ids": disagreements,
        "correctness_noise_floor": noise_floor,
        "mean_rollout_wall_s": mean_wall,
        "median_rollout_wall_s": statistics.median(walls),
        "max_rollout_wall_s": max(walls),
        "duel_rollouts": duel_rollouts,
        # This pass drives rollouts SEQUENTIALLY, so multiplying per-episode
        # wall by episode count prices out the continuous batching every real
        # duel uses. Measured on one RTX 5090 at ROLLOUT_CONCURRENCY=32, a
        # 1000-task sweep takes 31.3 min and a full duel 1.04 GPU-h, against
        # ~30 h projected here -- treat this as a loose upper bound for sizing
        # a single rollout, never as the cost of a duel.
        "sequential_upper_bound_gpu_hours": duel_rollouts * mean_wall / 3600.0,
        "delta_noise_multiplier": constants.DELTA_NOISE_MULTIPLIER,
        # The clamp is compared against what run_calibration_duel returns: the
        # standard error of the paired score gap, NOT the per-task flip rate
        # above. Feeding a flip rate here compares different units and inflates
        # the clamp several-fold.
        "score_gap_se": gap_se,
        "implied_noise_clamp": constants.DELTA_NOISE_MULTIPLIER * gap_se,
        "static_cross_gpu_budget": constants.CROSS_GPU_NOISE_BUDGET,
        "delta_headroom_at_ema_0_5": constants.DELTA_C * 0.5,
    }


def _cross_compare(report: dict, other: dict) -> dict:
    mine = {row["task_id"]: row for row in report["pass_a"]}
    theirs = {row["task_id"]: row for row in other.get("pass_a", [])}
    shared = sorted(set(mine) & set(theirs))
    if not shared:
        return {"shared_tasks": 0, "note": "no overlapping task_ids; run both boxes with the same --seed and corpus"}
    answer_disagree = [t for t in shared if mine[t]["answer"] != theirs[t]["answer"]]
    correct_disagree = sum(1 for t in shared if mine[t]["correct"] != theirs[t]["correct"])
    out = {
        "shared_tasks": len(shared),
        "cross_answer_agreement": (len(shared) - len(answer_disagree)) / len(shared),
        "cross_answer_disagreement_task_ids": answer_disagree,
        "cross_correctness_noise_floor": correct_disagree / len(shared),
    }
    for key in ("harness_digest", "seed", "corpus_digest"):
        if report.get(key) != other.get(key):
            out.setdefault("pin_mismatches", []).append(
                {"key": key, "mine": report.get(key), "theirs": other.get(key)}
            )
    return out


@app.command()
def main(
    model_dir: Optional[Path] = typer.Option(
        None, help="Local model snapshot directory (required for --backend vllm)."
    ),
    backend: str = typer.Option("vllm", help="'vllm' (pinned engine) or 'scripted' (self-test)."),
    corpus: Optional[Path] = typer.Option(
        None, help="Corpus snapshot to generate tasks from (default: temp fixture corpus)."
    ),
    tasks: int = typer.Option(DEFAULT_TASKS, help="Number of tasks to measure (K)."),
    seed: int = typer.Option(DEFAULT_SEED, help="Taskgen seed; use the same on both boxes."),
    out: Optional[Path] = typer.Option(None, help="Write the JSON report here."),
    compare: Optional[Path] = typer.Option(
        None, help="Another box's JSON report to cross-compare against."
    ),
) -> None:
    """Measure eval-stack determinism, noise floor, and duel cost projection."""
    if backend not in ("vllm", "scripted"):
        typer.echo(f"error: unknown backend {backend!r}", err=True)
        raise typer.Exit(code=2)

    if backend == "vllm":
        try:
            import torch  # noqa: F401
            import vllm  # noqa: F401
        except ImportError as exc:
            typer.echo(f"SKIP: torch/vllm not installed on this box ({exc}).")
            typer.echo("Install the eval extras (pip install -e '.[eval]') on a GPU box,")
            typer.echo("or run --backend scripted for the CI self-test.")
            raise typer.Exit(code=0)
        if model_dir is None or not model_dir.is_dir():
            typer.echo("error: --backend vllm requires --model-dir <snapshot dir>", err=True)
            raise typer.Exit(code=2)

    from epago.config import load_config
    from epago.environment.corpus import SqliteCorpus
    from epago.environment.fixtures import build_fixture_corpus
    from epago.environment.services import ResearchEnvironment
    from epago.environment.sync import corpus_digest
    from epago.eval.harness import harness_digest
    from epago.taskgen.generator import generate_tasks

    try:
        release = load_config().eval.taskgen_release
    except Exception:  # noqa: BLE001 - chain.toml optional for a measurement run
        release = "R1"

    with tempfile.TemporaryDirectory(prefix="epago-m0-") as tmp:
        if corpus is None:
            corpus_path = Path(tmp) / "corpus.db"
            store = build_fixture_corpus(corpus_path, n_docs=FIXTURE_DOCS, seed=7)
            typer.echo(f"corpus: fixture ({store.doc_count()} docs)")
        else:
            corpus_path = corpus
            store = SqliteCorpus(corpus_path)
            typer.echo(f"corpus: {corpus_path} ({store.doc_count()} docs)")
        c_digest = corpus_digest(corpus_path)

        typer.echo(f"minting {tasks} tasks (release {release}, seed {seed})")
        task_list = sorted(
            generate_tasks(seed=seed, release=release, corpus=store, n=tasks),
            key=lambda t: t.task_id,
        )

        if backend == "scripted":
            engine = _scripted_backend(task_list)
            model_label = "scripted"
        else:
            from epago.eval.backend import backend_factory

            engine = backend_factory(model_dir, kind="vllm")
            model_label = str(model_dir)

        env = ResearchEnvironment(store)
        typer.echo(f"pass A: {len(task_list)} rollouts")
        pass_a = _run_pass(engine, task_list, env)
        typer.echo(f"pass B: {len(task_list)} rollouts")
        pass_b = _run_pass(engine, task_list, env)
        engine.close()
        store.close()

    metrics = _metrics(pass_a, pass_b)
    report = {
        "kind": "epago-determinism-report-v1",
        "backend": backend,
        "model_dir": model_label,
        "seed": seed,
        "taskgen_release": release,
        "corpus_digest": c_digest,
        "harness_digest": harness_digest(),
        "metrics": metrics,
        "pass_a": pass_a,
        "pass_b": pass_b,
    }

    typer.echo("")
    typer.echo(f"answer agreement:        {metrics['answer_agreement']:.4f}")
    if metrics["disagreement_task_ids"]:
        typer.echo(f"disagreeing tasks:       {', '.join(metrics['disagreement_task_ids'])}")
    typer.echo(f"correctness noise floor: {metrics['correctness_noise_floor']:.4f}")
    typer.echo(
        f"implied noise clamp:     {metrics['implied_noise_clamp']:.4f} "
        f"(= {metrics['delta_noise_multiplier']:g} x noise floor; static fallback "
        f"{metrics['delta_noise_multiplier'] * metrics['static_cross_gpu_budget']:.4f})"
    )
    typer.echo(
        f"delta headroom at EMA .5: {metrics['delta_headroom_at_ema_0_5']:.4f} "
        "(the clamp must stay below this or duels can never accept)"
    )
    typer.echo(f"mean wall per rollout:   {metrics['mean_rollout_wall_s']:.3f}s")
    typer.echo(
        f"sequential upper bound:  {metrics['duel_rollouts']} rollouts "
        f"~= {metrics['sequential_upper_bound_gpu_hours']:.2f} GPU-hours "
        "(unbatched; a batched duel measured 1.04 GPU-h)"
    )
    typer.echo(f"score-gap SE:            {metrics['score_gap_se']:.4f}")

    if compare is not None:
        other = json.loads(Path(compare).read_text())
        cross = _cross_compare(report, other)
        report["cross"] = cross
        typer.echo("")
        typer.echo(f"cross-hardware vs {compare}:")
        for key, value in cross.items():
            typer.echo(f"  {key}: {value}")

    if out is not None:
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        typer.echo(f"\nreport written: {out}")

    if backend == "scripted":
        # Self-test contract: the scripted double is pure, so any wobble here
        # is a harness/environment determinism bug, not engine noise.
        if metrics["answer_agreement"] != 1.0 or metrics["correctness_noise_floor"] != 0.0:
            typer.echo("SELF-TEST FAIL: scripted backend must be perfectly deterministic", err=True)
            raise typer.Exit(code=1)
        typer.echo("\nSELF-TEST PASS: harness + environment are deterministic on this box")


if __name__ == "__main__":
    app()
