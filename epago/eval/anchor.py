"""External benchmark anchor — the eval-of-the-eval.

Internal king accuracy climbing while scores on a public external benchmark
(e.g. GAIA-Text or xbench-DeepSearch) stay flat is the signature of a gamed
task generator. On a schedule the validator runs the current king over a
user-provided benchmark file through the SINGLE rollout harness
(:func:`epago.eval.harness.run_rollout`) and the standard judge cascade, then
records the drift between the internal and external curves.

Anchor results are strictly observational: they never touch verdicts, weights,
or coronation. Divergence above the alert threshold raises an alarm, never a
halt.

Benchmark files are user-provided JSONL — one ``{"id"?, "question", "answer",
"aliases"?}`` object per line. The repo ships none (benchmark licensing
varies); the file's byte digest is recorded with every run so scores are only
comparable when the digests match.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epago.core.types import RolloutResult, Task, TaskOrigin
from epago.eval.backend import ModelBackend
from epago.eval.harness import run_rollout
from epago.eval.judge import LlmJudge

#: Marker written into ``Task.template`` so anchor rollouts are identifiable
#: anywhere a task or its telemetry surfaces.
ANCHOR_TEMPLATE = "anchor"

_NO_SOURCES = (
    "no external sources available in anchor mode; answer from your own "
    "knowledge with <answer>...</answer>"
)


@dataclass(frozen=True)
class AnchorTask:
    """One benchmark item: a question with a literal answer and aliases."""

    task_id: str
    question: str
    answer: str
    aliases: tuple[str, ...]


def load_benchmark(path: Path) -> tuple[str, list[AnchorTask]]:
    """Read a JSONL benchmark file into ``(benchmark_digest, tasks)``.

    The digest is ``sha256:<hex>`` over the file's exact bytes, and tasks keep
    file order, so two validators holding the same file load byte-identical
    benchmarks in the same deterministic order. Lines missing an ``id`` get a
    positional one.
    """
    raw = Path(path).read_bytes()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    tasks: list[AnchorTask] = []
    for lineno, line in enumerate(raw.decode("utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        tasks.append(
            AnchorTask(
                task_id=str(obj.get("id") or f"anchor-{lineno:05d}"),
                question=str(obj["question"]),
                answer=str(obj["answer"]),
                aliases=tuple(str(a) for a in obj.get("aliases", ())),
            )
        )
    return digest, tasks


class AnchorSession:
    """Minimal tool session for closed-book benchmark rollouts.

    External benchmarks have no pinned corpus behind them, so when no
    retrieval environment is available ``search``/``browse`` return an
    explicit no-sources observation — the harness protocol still functions,
    the model just has nothing to retrieve and must answer from weights.
    """

    def search(self, query: str) -> str:
        return _NO_SOURCES

    def search_page(self, queries: list[str]) -> str:
        return _NO_SOURCES

    def visit_pages(self, targets: list[str]) -> str:
        return _NO_SOURCES

    def browse(self, doc_id: str) -> str:
        return _NO_SOURCES


@dataclass(frozen=True)
class AnchorReport:
    """One anchor run's outcome, JSON-serializable via :meth:`to_dict`."""

    benchmark_digest: str
    n_tasks: int
    n_correct: int
    accuracy: float
    per_task: tuple[tuple[str, bool], ...]
    judge_tier_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_digest": self.benchmark_digest,
            "n_tasks": self.n_tasks,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "per_task": [[task_id, bool(ok)] for task_id, ok in self.per_task],
            "judge_tier_counts": [[tier, int(n)] for tier, n in self.judge_tier_counts],
        }


def _to_core_task(anchor_task: AnchorTask) -> Task:
    return Task(
        task_id=anchor_task.task_id,
        question=anchor_task.question,
        answer=anchor_task.answer,
        aliases=anchor_task.aliases,
        evidence_doc_ids=(),
        masked_doc_ids=(),
        origin=TaskOrigin.GENERATED_PUBLIC,
        template=ANCHOR_TEMPLATE,
        hops=1,
    )


def run_anchor(
    model_dir: Path,
    benchmark_path: Path,
    backend_factory: Callable[[Path], ModelBackend],
    env: Any = None,
    llm_judge: LlmJudge | None = None,
    max_tasks: int | None = None,
) -> AnchorReport:
    """Run one model over a benchmark file under the single harness.

    Each :class:`AnchorTask` becomes a core :class:`Task` (template
    ``"anchor"``) and goes through :func:`epago.eval.harness.run_rollout` with
    the environment's tools when ``env`` is given, else the closed-book
    :class:`AnchorSession`; answers are judged by the standard cascade. A
    crashed rollout scores incorrect — an anchor run never aborts. Purely
    observational: nothing here feeds verdicts, weights, or coronation.
    """
    benchmark_digest, anchor_tasks = load_benchmark(Path(benchmark_path))
    if max_tasks is not None:
        anchor_tasks = anchor_tasks[:max_tasks]
    # The backend is released when the run ends, the same contract the probe
    # gate keeps. It matters beyond tidiness on a multi-GPU box: there the
    # factory hands out a *leased* pool replica, and a lease that is never
    # returned costs the pool that card for the rest of the process.
    backend = backend_factory(Path(model_dir))
    tier_counts: Counter[str] = Counter()
    per_task: list[tuple[str, bool]] = []
    n_correct = 0
    try:
        for anchor_task in anchor_tasks:
            task = _to_core_task(anchor_task)
            try:
                session = env.tools_for_task(task) if env is not None else AnchorSession()
                result = run_rollout(backend, task, session, llm_judge=llm_judge)
            except Exception as exc:  # noqa: BLE001 - a crashed rollout is a wrong answer
                result = RolloutResult(
                    task_id=task.task_id,
                    answer=None,
                    correct=False,
                    turns=0,
                    wall_time_s=0.0,
                    judge_tier="none",
                    error=f"rollout_crash: {type(exc).__name__}: {exc}",
                )
            tier_counts[result.judge_tier] += 1
            per_task.append((anchor_task.task_id, bool(result.correct)))
            n_correct += int(result.correct)
    finally:
        backend.close()
    n_tasks = len(anchor_tasks)
    return AnchorReport(
        benchmark_digest=benchmark_digest,
        n_tasks=n_tasks,
        n_correct=n_correct,
        accuracy=n_correct / n_tasks if n_tasks else 0.0,
        per_task=tuple(per_task),
        judge_tier_counts=tuple(sorted(tier_counts.items())),
    )


def divergence(internal_acc_ema: float, history: list[dict]) -> float | None:
    """Drift between the internal accuracy curve and the anchor curve.

    Given past anchor records (each ``{block, accuracy, internal_ema_at_run}``
    in run order), returns::

        (internal_acc_ema - history[0]["internal_ema_at_run"])   # internal gain
      - (history[-1]["accuracy"] - history[0]["accuracy"])       # anchor gain

    i.e. how much more the internal king EMA has climbed since the first
    anchor run than the external benchmark score has. A sustained positive
    value means internal accuracy is inflating relative to an unmoving
    external yardstick — the task generator is likely being gamed. Returns
    ``None`` with fewer than two records (no gain to measure).
    """
    if len(history) < 2:
        return None
    first, latest = history[0], history[-1]
    internal_gain = float(internal_acc_ema) - float(first["internal_ema_at_run"])
    anchor_gain = float(latest["accuracy"]) - float(first["accuracy"])
    return internal_gain - anchor_gain
