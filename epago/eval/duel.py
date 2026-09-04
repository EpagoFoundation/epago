"""The paired duel.

Both models answer the same tasks under the identical harness, so every
per-task difference d_i cancels task difficulty and isolates model skill. All
source of randomness the protocol controls is pinned to chain data: task order
is sorted by task_id, decoding is greedy under the harness, and the bootstrap
seed derives from (block_hash_at_reveal, author_hotkey). Two validators running
the same spec therefore pose the identical exam and, from identical per-task
results, derive a bit-identical DuelOutcome.

The per-task results themselves are NOT identical, and the mechanism does not
need them to be. The engine is not bit-reproducible (see backend.py): the same
checkpoint re-scored on the same 400 tasks flips correctness on ~21% of them on
the reference stack, with a paired score-gap standard error of ~0.030 at n=128.
That is what run_calibration_duel measures, what noise_floor_from_calibration
feeds to adaptive_delta, and what the 99.9% bootstrap LCB prices. Nothing here
assumes a deterministic scorer.
"""

from __future__ import annotations

import logging
import os
import statistics
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from epago.core.stats import (
    adaptive_delta,
    bootstrap_lcb,
    bootstrap_seed,
    paired_half,
    public_task_seed,
    round_bootstrap_seed,
    round_public_seed,
)
from epago.core.types import DuelHalf, DuelOutcome, DuelSpec, RolloutResult, Task
from epago.eval.backend import ModelBackend
from epago.eval.harness import run_rollout, run_rollouts_batched
from epago.eval.judge import LlmJudge

if TYPE_CHECKING:
    from epago.core.types import RoundDuelSpec, RoundResult
    from epago.environment.services import ResearchEnvironment
    from epago.eval.pool import GpuPool, SweepResult

logger = logging.getLogger(__name__)

ProgressFn = Callable[[dict], None]

__all__ = ["DuelSpec", "run_calibration_duel", "run_duel", "run_round_duel"]


def _run_pooled(
    pool: "GpuPool",
    env: "ResearchEnvironment",
    llm_judge: LlmJudge | None,
    requests: list,
    tier_counts: Counter[str],
    on_progress: ProgressFn | None,
    extra: dict | None = None,
) -> list["SweepResult"]:
    """Run sweep requests across the GPU pool, reporting progress as the
    sequential path reports it.

    The only thing that changes versus the single-backend path is *when* events
    arrive: replicas run concurrently, so a caller reading /progress sees the
    models interleaved instead of one after the other. Event fields, and the
    results themselves, are identical. ``tier_counts`` is mutated from replica
    threads, hence the lock — a Counter increment is not atomic.
    """
    totals = {(i, phase): len(tasks) for i, r in enumerate(requests) for phase, tasks in r.phases}
    lock = threading.Lock()

    def on_result(request_index: int, phase: str, index: int, res) -> None:
        with lock:
            tier_counts[res.judge_tier] += 1
        if on_progress is None:
            return
        event = {
            "phase": phase,
            "model": requests[request_index].label,
            "task_id": res.task_id,
            "index": index + 1,
            "total": totals[(request_index, phase)],
            "correct": res.correct,
        }
        if extra:
            event.update(extra)
        on_progress(event)

    return pool.run_sweeps(
        requests,
        session_factory=env.tools_for_task,
        llm_judge=llm_judge,
        on_result=on_result,
    )


def _safe_rollout(
    backend: ModelBackend,
    task: Task,
    env: "ResearchEnvironment",
    llm_judge: LlmJudge | None,
) -> RolloutResult:
    """One rollout that can never abort a duel: any crash — tool session
    construction included — scores the task incorrect for that model."""
    try:
        session = env.tools_for_task(task)
        return run_rollout(backend, task, session, llm_judge=llm_judge)
    except Exception as exc:  # noqa: BLE001 - a crashed rollout is a wrong answer
        return RolloutResult(
            task_id=task.task_id,
            answer=None,
            correct=False,
            turns=0,
            wall_time_s=0.0,
            judge_tier="none",
            error=f"rollout_crash: {type(exc).__name__}: {exc}",
        )


def run_duel(
    spec: DuelSpec,
    env: "ResearchEnvironment",
    backend_factory: Callable[[Path], ModelBackend],
    llm_judge: LlmJudge | None = None,
    *,
    on_progress: ProgressFn | None = None,
    pool: "GpuPool | None" = None,
) -> DuelOutcome:
    """Run one champion-vs-challenger duel and return the verdict inputs.

    Acceptance requires both halves: the public-half bootstrap LCB must clear
    the adaptive effect floor AND the private half must show a positive mean
    difference — the private pool is the overfitting tripwire, so a challenger
    tuned to the public distribution alone never passes. Backend lifecycle
    belongs to the caller's factory (the eval server keeps the king warm and
    evicts the challenger) — except under ``EPAGO_EVAL_LOW_VRAM``, where the
    king's engine is released here as soon as its rollouts finish.

    With a :class:`~epago.eval.pool.GpuPool` the two sweeps become two work
    items over the box's replicas — on 8 cards that is 4 replicas each, and the
    exam is sharded across them. ``pool=None`` (a one-GPU validator, or any
    caller that has not opted in) runs the original single-backend path
    untouched, LOW_VRAM swap included.
    """
    tier_counts: Counter[str] = Counter()
    phases: list[tuple[str, list[Task]]] = [
        ("public", sorted(spec.public_tasks, key=lambda t: t.task_id)),
        ("private", sorted(spec.private_tasks, key=lambda t: t.task_id)),
    ]

    def sweep(model: str, backend: ModelBackend) -> dict[str, list]:
        """All phases for one model, episodes step-batched to keep the engine
        saturated. Task order (and therefore pairing) is fixed by task_id."""
        out: dict[str, list] = {}
        for phase, ordered in phases:
            def report(index: int, res, *, _phase=phase, _total=len(ordered)) -> None:
                tier_counts[res.judge_tier] += 1
                if on_progress is not None:
                    on_progress(
                        {
                            "phase": _phase,
                            "model": model,
                            "task_id": res.task_id,
                            "index": index + 1,
                            "total": _total,
                            "correct": res.correct,
                        }
                    )

            out[phase] = run_rollouts_batched(
                backend,
                ordered,
                env.tools_for_task,
                llm_judge=llm_judge,
                on_result=report,
            )
        return out

    if pool is not None:
        from epago.eval.pool import SweepRequest

        frozen = tuple((phase, tuple(tasks)) for phase, tasks in phases)
        sweeps = _run_pooled(
            pool,
            env,
            llm_judge,
            [
                SweepRequest(spec.king_dir, frozen, label="king"),
                SweepRequest(spec.challenger_dir, frozen, label="challenger"),
            ],
            tier_counts,
            on_progress,
        )
        for result in sweeps:
            if result.error is not None:
                # A duel needs both sweeps; half a duel is not a verdict.
                raise result.error
        king_results, chall_results = sweeps[0].phases, sweeps[1].phases
    else:
        # Low-VRAM mode: one engine resident at a time — the king's rollouts run
        # first and its engine is released before the challenger loads, halving
        # peak memory at the cost of one model reload per duel. Pairing is by
        # task, not by time, so results are unchanged.
        low_vram = os.environ.get("EPAGO_EVAL_LOW_VRAM", "").lower() in ("1", "true", "yes")

        king = backend_factory(spec.king_dir)
        king_results = sweep("king", king)
        if low_vram:
            king.close()
        challenger = backend_factory(spec.challenger_dir)
        chall_results = sweep("challenger", challenger)

    def triples(phase: str) -> list[tuple[str, bool, bool]]:
        ordered = dict(phases)[phase]
        return [
            (task.task_id, k.correct, c.correct)
            for task, k, c in zip(ordered, king_results[phase], chall_results[phase])
        ]

    public_triples = triples("public")
    private_triples = triples("private")

    public = paired_half([k for _, k, _ in public_triples], [c for _, _, c in public_triples])
    private = paired_half(
        [k for _, k, _ in private_triples], [c for _, _, c in private_triples]
    )

    boot_seed = bootstrap_seed(spec.block_hash_at_reveal, spec.author_hotkey)
    pub_seed = public_task_seed(spec.block_hash_at_reveal, spec.author_hotkey)
    lcb_pub = bootstrap_lcb(public.diffs, boot_seed)
    delta = adaptive_delta(spec.king_acc_ema, spec.noise_floor)
    return DuelOutcome(
        public=public,
        private=private,
        lcb_pub=lcb_pub,
        delta=delta,
        accepted=lcb_pub > delta and private.mu_hat > 0.0,
        boot_seed_hex=f"{boot_seed:016x}",
        public_seed_hex=f"{pub_seed:016x}",
        public_task_results=tuple(
            (task_id, int(c) - int(k)) for task_id, k, c in public_triples
        ),
        judge_tier_counts=tuple(sorted(tier_counts.items())),
    )


def run_round_duel(
    spec: "RoundDuelSpec",
    env: "ResearchEnvironment",
    backend_factory: Callable[[Path], ModelBackend],
    llm_judge: LlmJudge | None = None,
    *,
    on_progress: ProgressFn | None = None,
    pool: "GpuPool | None" = None,
) -> list["RoundResult"]:
    """Run one competition round: every entrant against the king on one exam.

    The king answers the exam **once** and its per-task results are reused for
    every pairing. That is both the point and the economy of a round: the field
    is compared on identical questions against identical king answers, so the
    only thing separating two entrants is their own output, and N entrants cost
    N+1 sweeps instead of 2N.

    Each entrant is scored exactly as a solo duel would score it — same paired
    statistic, same bootstrap LCB, same adaptive floor — so a round result and a
    one-off duel result mean the same thing. Picking the winner is the caller's
    job (:mod:`epago.validator.service`); this returns every entrant's outcome
    including the losers, because they still earn arena credit and cooldowns.

    An entrant whose sweep raises is scored as a total loss rather than aborting
    the round: one broken checkpoint must not deny every other entrant its duel.

    A round is the case multi-GPU was built for: N+1 *independent* sweeps of one
    exam, with no ordering constraint between them (the comparison happens
    afterwards, on the score vectors). With a :class:`~epago.eval.pool.GpuPool`
    all N+1 are submitted at once and the pool decides replication;
    ``pool=None`` runs them one after another exactly as before. One reporting
    difference in the pooled path: sweeps finish concurrently, so every
    entrant's ``judge_tier_counts`` carries the round's full tally rather than
    the running total that happened to exist when it was scored. That field is
    telemetry — it is not an input to any verdict.
    """
    from epago.core.types import RoundResult  # local: keeps core import-light

    public = sorted(spec.public_tasks, key=lambda t: t.task_id)
    private = sorted(spec.private_tasks, key=lambda t: t.task_id)
    phases: list[tuple[str, list[Task]]] = [("public", public), ("private", private)]
    tier_counts: Counter[str] = Counter()

    def sweep(model: str, backend: ModelBackend) -> dict[str, list]:
        out: dict[str, list] = {}
        for phase, ordered in phases:
            def report(index: int, res, *, _phase=phase, _model=model, _total=len(ordered)) -> None:
                tier_counts[res.judge_tier] += 1
                if on_progress is not None:
                    on_progress(
                        {
                            "round": spec.round,
                            "phase": _phase,
                            "model": _model,
                            "task_id": res.task_id,
                            "index": index + 1,
                            "total": _total,
                            "correct": res.correct,
                        }
                    )

            out[phase] = run_rollouts_batched(
                backend, ordered, env.tools_for_task, llm_judge=llm_judge, on_result=report
            )
        return out

    entrant_sweeps: list | None = None
    if pool is not None:
        from epago.eval.pool import SweepRequest

        frozen = tuple((phase, tuple(tasks)) for phase, tasks in phases)
        sweeps = _run_pooled(
            pool,
            env,
            llm_judge,
            [SweepRequest(spec.king_dir, frozen, label="king")]
            + [SweepRequest(e.challenger_dir, frozen, label=e.digest) for e in spec.entrants],
            tier_counts,
            on_progress,
            extra={"round": spec.round},
        )
        if sweeps[0].error is not None:
            # Without the king's answers there is nothing to compare anyone to.
            raise sweeps[0].error
        king_results = sweeps[0].phases
        entrant_sweeps = sweeps[1:]
    else:
        king = backend_factory(spec.king_dir)
        king_results = sweep("king", king)
        # The king's engine is released before any entrant loads: a round holds
        # one challenger at a time, so peak memory does not grow with the field.
        king.close()

    delta = adaptive_delta(spec.king_acc_ema, spec.noise_floor)
    results: list[RoundResult] = []
    for position, entrant in enumerate(spec.entrants):
        if entrant_sweeps is not None:
            swept = entrant_sweeps[position]
            if swept.error is not None:
                logger.warning("entrant %s failed its sweep: %s", entrant.digest, swept.error)
                results.append(
                    RoundResult(entrant=entrant, outcome=_forfeit(spec, entrant, delta))
                )
                continue
            chall_results = swept.phases
        else:
            try:
                backend = backend_factory(entrant.challenger_dir)
                chall_results = sweep(entrant.digest, backend)
                backend.close()
            except Exception as exc:  # noqa: BLE001 - one bad entrant never ends the round
                logger.warning("entrant %s failed its sweep: %s", entrant.digest, exc)
                results.append(
                    RoundResult(entrant=entrant, outcome=_forfeit(spec, entrant, delta))
                )
                continue

        halves = {}
        diffs_by_phase = {}
        for phase, ordered in phases:
            triples = [
                (task.task_id, k.correct, c.correct)
                for task, k, c in zip(ordered, king_results[phase], chall_results[phase])
            ]
            halves[phase] = paired_half(
                [k for _, k, _ in triples], [c for _, _, c in triples]
            )
            diffs_by_phase[phase] = triples

        boot_seed = round_bootstrap_seed(spec.round_block_hash, entrant.digest)
        lcb_pub = bootstrap_lcb(halves["public"].diffs, boot_seed)
        results.append(
            RoundResult(
                entrant=entrant,
                outcome=DuelOutcome(
                    public=halves["public"],
                    private=halves["private"],
                    lcb_pub=lcb_pub,
                    delta=delta,
                    accepted=lcb_pub > delta and halves["private"].mu_hat > 0.0,
                    boot_seed_hex=f"{boot_seed:016x}",
                    public_seed_hex=f"{round_public_seed(spec.round_block_hash, spec.round):016x}",
                    public_task_results=tuple(
                        (task_id, int(c) - int(k)) for task_id, k, c in diffs_by_phase["public"]
                    ),
                    judge_tier_counts=tuple(sorted(tier_counts.items())),
                ),
            )
        )
    return results


def _forfeit(spec: "RoundDuelSpec", entrant, delta: float) -> DuelOutcome:
    """Outcome for an entrant whose sweep could not be run at all.

    Scored as a maximal loss rather than skipped, so a checkpoint that reliably
    crashes the harness is priced by the cooldown ladder instead of being a free
    way to occupy a slot in every round.
    """
    n_pub = max(len(spec.public_tasks), 1)
    n_priv = max(len(spec.private_tasks), 1)
    empty_pub = DuelHalf(
        n_tasks=n_pub, diffs=(-1,) * n_pub, mu_hat=-1.0, king_acc=0.0, challenger_acc=0.0
    )
    empty_priv = DuelHalf(
        n_tasks=n_priv, diffs=(-1,) * n_priv, mu_hat=-1.0, king_acc=0.0, challenger_acc=0.0
    )
    boot_seed = round_bootstrap_seed(spec.round_block_hash, entrant.digest)
    return DuelOutcome(
        public=empty_pub,
        private=empty_priv,
        lcb_pub=-1.0,
        delta=delta,
        accepted=False,
        boot_seed_hex=f"{boot_seed:016x}",
        public_seed_hex=f"{round_public_seed(spec.round_block_hash, spec.round):016x}",
    )


def run_calibration_duel(
    king_dir: Path,
    tasks: list[Task],
    env: "ResearchEnvironment",
    backend_factory: Callable[[Path], ModelBackend],
    llm_judge: LlmJudge | None = None,
    *,
    pool: "GpuPool | None" = None,
) -> float:
    """King versus itself on one task set; returns the score-gap std error.

    Any nonzero disagreement is pure harness noise (engine nondeterminism, tool
    flakiness, grader instability) since the model is literally the same weights
    twice. What a duel actually decides on is the mean paired difference, so the
    quantity that bounds a coin-flip verdict is that mean's standard error
    (std(d_i)/sqrt(n)), not the per-task flip rate. The validator feeds this into
    its noise floor, which clamps the duel's adaptive delta from below.

    The measurement has to run the *same* path a real duel runs, or it
    understates what it is supposed to bound. This previously passed
    ``llm_judge=None`` and rolled out one task at a time, so it excluded both
    the fallback judge — a second model, and the least deterministic grader in
    the stack — and the step-batched execution real duels use, where batch
    composition perturbs numerics. Both are live sources of disagreement in a
    scored duel, so leaving them out biased the floor low and let borderline
    verdicts through as coin flips.
    """
    if not tasks:
        raise ValueError("calibration duel needs at least one task")
    ordered = sorted(tasks, key=lambda t: t.task_id)
    if pool is not None:
        # Two sweeps of the same weights, which the pool spreads over the box
        # the same way it spreads a duel. On a multi-GPU validator the two runs
        # therefore land on *different* cards, which is the right measurement:
        # a real duel on this box also puts king and challenger on different
        # cards, so any cross-replica disagreement is noise the floor must
        # already cover. On a single-GPU validator this branch is never taken.
        from epago.eval.pool import SweepRequest

        phase = (("calibration", tuple(ordered)),)
        sweeps = _run_pooled(
            pool,
            env,
            llm_judge,
            [
                SweepRequest(king_dir, phase, label="calibration-a"),
                SweepRequest(king_dir, phase, label="calibration-b"),
            ],
            Counter(),
            None,
        )
        for result in sweeps:
            if result.error is not None:
                raise result.error
        first = sweeps[0].phases["calibration"]
        second = sweeps[1].phases["calibration"]
    else:
        backend = backend_factory(king_dir)
        first = run_rollouts_batched(backend, ordered, env.tools_for_task, llm_judge=llm_judge)
        second = run_rollouts_batched(backend, ordered, env.tools_for_task, llm_judge=llm_judge)
    diffs = [int(b.correct) - int(a.correct) for a, b in zip(first, second)]
    if not diffs:
        raise ValueError("calibration duel produced no comparable rollouts")
    # Return the score-gap standard error, not the raw per-task flip rate. A
    # duel decides on the *mean* difference mu_hat, whose noise is std(d_i)/sqrt(n)
    # -- shrinking with n. Feeding the per-task rate (which does not shrink)
    # into the delta noise clamp demanded an impossible margin once real
    # calibration data replaced the tiny static fallback. This is the same unit
    # as CROSS_GPU_NOISE_BUDGET (a score-level budget) and as adaptive_delta's
    # noise term. Identical weights -> all d_i == 0 -> stdev 0 -> 0.0.
    n = len(diffs)
    if n < 2:
        return 0.0
    return statistics.stdev(diffs) / (n ** 0.5)
