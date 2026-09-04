"""Deterministic task generation.

:func:`generate_tasks` is a pure function of ``(seed, release, corpus)``: all
randomness flows through one ``numpy`` PCG64 stream seeded by the caller, and
templates only ever consume that stream. Two validators deriving the same seed
from chain data (:func:`epago.core.stats.public_task_seed`) mint byte-identical
task lists, which is what makes duel verdicts replayable by auditors.

The optional :class:`KingProbe` hook adds the difficulty band filter: tasks
the current king solves far outside ``[KING_SOLVE_BAND_LOW,
KING_SOLVE_BAND_HIGH]`` carry almost no discrimination signal in a paired
duel, so they are dropped at mint time. The probe consumes no generator
randomness, so a probe-filtered run is still deterministic given the same
probe behaviour.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

import numpy as np

from epago import constants
from epago.core.types import Task
from epago.taskgen.qa import verify_task
from epago.taskgen.templates import base_mixture, templates_for_release

if TYPE_CHECKING:  # protocol-only coupling to the environment subsystem
    from epago.environment.corpus import CorpusStore

logger = logging.getLogger(__name__)

# Mint attempts allowed per requested task before the run counts as exhausted.
_ATTEMPTS_PER_TASK = 25
_MIN_ATTEMPT_BUDGET = 200
# Redraw the template mixture every this many accepted tasks so one long run
# rotates through differently weighted mixes (still purely rng-driven).
_MIXTURE_ROTATE_EVERY = 50
# Below this yield the batch is unusable for a fixed-size holdout half.
_MIN_YIELD_FRACTION = 0.9


class GenerationExhausted(RuntimeError):
    """The attempt budget ran out with fewer than 90% of requested tasks."""


class KingProbe(Protocol):
    """Estimates the current king's solve rate on one task via k rollouts.

    Implemented by the eval subsystem (it owns the inference harness); taskgen
    only consumes the rate for the difficulty band filter.
    """

    def solve_rate(self, task: Task, k: int = 4) -> float: ...


def generate_tasks(
    seed: int,
    release: str,
    corpus: "CorpusStore",
    n: int,
    king_probe: "KingProbe | None" = None,
) -> list[Task]:
    """Mint ``n`` QA-verified tasks deterministically from ``seed``.

    Pipeline per candidate: template mint -> content-id dedup -> QA
    (:func:`epago.taskgen.qa.verify_task`) -> optional king difficulty band.
    Dropped candidates are counted and logged — never silently truncated.
    Returns fewer than ``n`` (with a warning) only when the attempt budget is
    exhausted above the 90% yield floor; below it raises
    :class:`GenerationExhausted` so callers never duel on a starved holdout.
    """
    if n <= 0:
        return []
    rng = np.random.Generator(np.random.PCG64(seed))
    templates = templates_for_release(release)
    names = [t.name for t in templates]

    def draw_weights() -> np.ndarray:
        mixture = base_mixture(names, rng)
        return np.asarray([mixture[name] for name in names], dtype=np.float64)

    weights = draw_weights()
    next_rotate = _MIXTURE_ROTATE_EVERY
    budget = max(_ATTEMPTS_PER_TASK * n, _MIN_ATTEMPT_BUDGET)
    tasks: list[Task] = []
    seen_ids: set[str] = set()
    dropped: Counter[str] = Counter()

    for _ in range(budget):
        if len(tasks) >= n:
            break
        if len(tasks) >= next_rotate:
            weights = draw_weights()
            next_rotate += _MIXTURE_ROTATE_EVERY
        template = templates[int(rng.choice(len(templates), p=weights))]
        task = template.mint(corpus, rng)
        if task is None:
            dropped["mint_failed"] += 1
            continue
        if task.task_id in seen_ids:
            dropped["duplicate"] += 1
            continue
        # Same content -> same verdict; remember the id even if QA drops it
        # so a re-mint of the same fact is not re-checked.
        seen_ids.add(task.task_id)
        report = verify_task(task, corpus)
        if not report.ok:
            dropped["qa_failed"] += 1
            logger.debug("taskgen qa drop %s: %s", task.task_id, report.failures)
            continue
        if king_probe is not None:
            rate = king_probe.solve_rate(task)
            if not (constants.KING_SOLVE_BAND_LOW <= rate <= constants.KING_SOLVE_BAND_HIGH):
                dropped["difficulty_band"] += 1
                continue
        tasks.append(task)

    if dropped:
        logger.info(
            "taskgen: %d/%d tasks minted, %d candidates dropped (%s)",
            len(tasks), n, sum(dropped.values()), dict(dropped),
        )
    if len(tasks) < n:
        logger.warning(
            "taskgen: attempt budget %d exhausted at %d/%d tasks (release=%s)",
            budget, len(tasks), n, release,
        )
        if len(tasks) < _MIN_YIELD_FRACTION * n:
            raise GenerationExhausted(
                f"minted {len(tasks)}/{n} tasks after {budget} attempts "
                f"(release={release}, dropped={dict(dropped)})"
            )
    return tasks


def task_ids_digest(tasks: Iterable[Task]) -> str:
    """Order-independent commitment over a task set, for audit records.

    Sorted so two validators (or an auditor replaying a verdict) get the same
    digest regardless of mint order.
    """
    joined = "\n".join(sorted(t.task_id for t in tasks))
    return "sha256:" + hashlib.sha256(joined.encode()).hexdigest()
