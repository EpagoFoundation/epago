"""Cheap pre-duel gates.

Every probe here must run in minutes on a CPU-only validator: a challenger
that would waste a full duel's GPU budget (corrupt weights, absurd
reparametrizations, harness-breaking output, collapsed policies) is rejected
before an engine is ever loaded. All tensor access goes through a pure-numpy
safetensors reader so nothing degrades when torch is absent; when the
``safetensors`` package is installed it is used as an extra cross-check on
header parsing, import-guarded.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from epago import constants
from epago.core.types import RolloutResult, Task
from epago.eval.backend import ModelBackend
from epago.eval.harness import run_rollout

if TYPE_CHECKING:
    from epago.environment.services import ResearchEnvironment


@dataclass(frozen=True, slots=True)
class ProbeFailure:
    code: str
    detail: str


# --- safetensors reading (pure numpy, torch-free) ----------------------------

_MAX_HEADER_BYTES = 100_000_000
_NAN_SAMPLE_TENSORS = 16
_NAN_SAMPLE_ELEMENTS = 1 << 22

_DTYPES: dict[str, tuple[str, bool]] = {
    "F64": ("<f8", True),
    "F32": ("<f4", True),
    "F16": ("<f2", True),
    "BF16": ("<u2", True),
    "I64": ("<i8", False),
    "I32": ("<i4", False),
    "I16": ("<i2", False),
    "I8": ("|i1", False),
    "U8": ("|u1", False),
    "BOOL": ("|b1", False),
}


def _read_header(path: Path) -> tuple[int, dict]:
    """Parse a safetensors header with the stdlib: 8-byte little-endian length,
    then a JSON map of name -> {dtype, shape, data_offsets}."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) < 8:
            raise ValueError("file shorter than the 8-byte header length")
        header_len = int.from_bytes(prefix, "little")
        if not 0 < header_len <= _MAX_HEADER_BYTES:
            raise ValueError(f"implausible header length {header_len}")
        raw = fh.read(header_len)
        if len(raw) < header_len:
            raise ValueError("truncated header")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return header_len, header


def _check_entry(name: str, info: dict, data_bytes: int) -> str | None:
    dtype = info.get("dtype")
    if dtype not in _DTYPES:
        return f"{name}: unknown dtype {dtype!r}"
    begin, end = info.get("data_offsets", (None, None))
    if not (isinstance(begin, int) and isinstance(end, int) and 0 <= begin <= end <= data_bytes):
        return f"{name}: data_offsets {info.get('data_offsets')!r} outside data section"
    itemsize = np.dtype(_DTYPES[dtype][0]).itemsize
    expected = math.prod(info.get("shape", ())) * itemsize
    if end - begin != expected:
        return f"{name}: byte span {end - begin} != shape-implied {expected}"
    return None


def _load_tensor(path: Path, header_len: int, info: dict, max_elements: int | None = None) -> np.ndarray:
    """Load one tensor as a flat float array via numpy; BF16 is widened to
    float32 by bit-shifting so no ML framework is required."""
    np_dtype, _ = _DTYPES[info["dtype"]]
    begin, end = info["data_offsets"]
    count = (end - begin) // np.dtype(np_dtype).itemsize
    if max_elements is not None:
        count = min(count, max_elements)
    arr = np.fromfile(path, dtype=np_dtype, count=count, offset=8 + header_len + begin)
    if info["dtype"] == "BF16":
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr


def _iter_shards(model_dir: Path) -> list[Path]:
    return sorted(p for p in model_dir.rglob("*.safetensors") if p.is_file())


def load_probe(model_dir: Path) -> list[ProbeFailure]:
    """Load-sanity gate: headers must parse, offsets must be consistent with
    file sizes, and a deterministic sample of float tensors must be finite.

    Uses ``safetensors.safe_open`` as a cross-check when installed
    (import-guarded); the primary path is the stdlib header parse plus numpy
    loading, so a torch-free validator still gets full NaN/Inf coverage on the
    sampled subset instead of a weaker file-size-only fallback.
    """
    failures: list[ProbeFailure] = []
    shards = _iter_shards(model_dir)
    if not shards:
        return [ProbeFailure("no_safetensors", f"no .safetensors files under {model_dir}")]

    float_tensors: list[tuple[Path, int, str, dict]] = []
    for shard in shards:
        try:
            header_len, header = _read_header(shard)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(ProbeFailure("header_parse", f"{shard.name}: {exc}"))
            continue
        data_bytes = shard.stat().st_size - 8 - header_len
        entries = {k: v for k, v in header.items() if k != "__metadata__"}
        if not entries:
            failures.append(ProbeFailure("empty_shard", f"{shard.name}: no tensors"))
            continue
        for name in sorted(entries):
            problem = _check_entry(name, entries[name], data_bytes)
            if problem:
                failures.append(ProbeFailure("bad_tensor_entry", f"{shard.name}: {problem}"))
            elif _DTYPES[entries[name]["dtype"]][1]:
                float_tensors.append((shard, header_len, name, entries[name]))
        try:
            from safetensors import safe_open
        except ImportError:
            safe_open = None
        if safe_open is not None:
            try:
                with safe_open(shard, framework="numpy") as fh:
                    list(fh.keys())
            except Exception as exc:  # noqa: BLE001 - any open failure is a probe failure
                failures.append(ProbeFailure("safetensors_open", f"{shard.name}: {exc}"))

    if failures:
        return failures

    float_tensors.sort(key=lambda t: t[2])
    stride = max(1, len(float_tensors) // _NAN_SAMPLE_TENSORS)
    for shard, header_len, name, info in float_tensors[::stride][:_NAN_SAMPLE_TENSORS]:
        values = _load_tensor(shard, header_len, info, max_elements=_NAN_SAMPLE_ELEMENTS)
        if values.size and not np.isfinite(values.astype(np.float64, copy=False)).all():
            failures.append(ProbeFailure("non_finite", f"{shard.name}: {name} contains NaN/Inf"))
    return failures


def _tensor_norms(model_dir: Path) -> dict[str, float]:
    """Per-tensor L2 norms for every float tensor, computed in float64."""
    norms: dict[str, float] = {}
    for shard in _iter_shards(model_dir):
        header_len, header = _read_header(shard)
        for name, info in header.items():
            if name == "__metadata__" or not _DTYPES.get(info.get("dtype"), ("", False))[1]:
                continue
            values = _load_tensor(shard, header_len, info).astype(np.float64, copy=False)
            norms[name] = float(np.linalg.norm(values))
    return norms


def _per_tensor_ratio_failures(
    chall_norms: dict[str, float], king_norms: dict[str, float]
) -> list[ProbeFailure]:
    """Cap each shared tensor's norm ratio against the king.

    Defends against the local scaling symmetry: multiplying one tensor by α
    and a functionally adjacent tensor by 1/α (e.g. an RMSNorm gain against
    the projection it feeds) preserves the network function exactly, letting a
    re-skinned king look arbitrarily far from the original in weight space.
    """
    eps = 1e-12
    failures = []
    for name in sorted(chall_norms.keys() & king_norms.keys()):
        ratio = (chall_norms[name] + eps) / (king_norms[name] + eps)
        worst = max(ratio, 1.0 / ratio)
        if worst > constants.NORM_SANITY_MAX_LAYER_RATIO:
            failures.append(
                ProbeFailure(
                    "layer_norm_ratio",
                    f"{name}: norm ratio {worst:.2f} exceeds "
                    f"{constants.NORM_SANITY_MAX_LAYER_RATIO}",
                )
            )
    return failures


def _global_ratio_failures(
    chall_norms: dict[str, float], king_norms: dict[str, float]
) -> list[ProbeFailure]:
    """Cap the whole-model norm ratio against the king.

    Defends against the global scaling symmetry: uniformly rescaling all
    weights (absorbed by normalization layers, and by the argmax invariance of
    greedy decoding at the output logits) preserves greedy behavior while
    shifting every per-tensor norm by the same factor, which per-tensor caps
    alone would tolerate up to their limit.
    """
    chall_global = math.sqrt(sum(v * v for v in chall_norms.values()))
    king_global = math.sqrt(sum(v * v for v in king_norms.values()))
    eps = 1e-12
    ratio = (chall_global + eps) / (king_global + eps)
    worst = max(ratio, 1.0 / ratio)
    if worst > constants.NORM_SANITY_MAX_GLOBAL_RATIO:
        return [
            ProbeFailure(
                "global_norm_ratio",
                f"global norm ratio {worst:.2f} exceeds {constants.NORM_SANITY_MAX_GLOBAL_RATIO}",
            )
        ]
    return []


def norm_sanity_probe(model_dir: Path, king_dir: Path) -> list[ProbeFailure]:
    """Weight-norm sanity of a challenger versus the incumbent king.

    The two ratio checks each defend against a named function-preserving
    reparametrization symmetry (see their docstrings); the tensor-set check
    rejects challengers that add or drop tensors, which the config lock cannot
    see. Norms come from the pure-numpy reader, so this runs torch-free.
    """
    try:
        chall_norms = _tensor_norms(model_dir)
        king_norms = _tensor_norms(king_dir)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        return [ProbeFailure("norm_read", f"failed reading tensors: {exc}")]
    failures: list[ProbeFailure] = []
    if chall_norms.keys() != king_norms.keys():
        extra = sorted(chall_norms.keys() - king_norms.keys())
        missing = sorted(king_norms.keys() - chall_norms.keys())
        failures.append(
            ProbeFailure("tensor_set", f"extra={extra[:5]} missing={missing[:5]}")
        )
    failures.extend(_per_tensor_ratio_failures(chall_norms, king_norms))
    failures.extend(_global_ratio_failures(chall_norms, king_norms))
    return failures


def probe_task_set(tasks: list[Task], n_tasks: int | None = None) -> list[Task]:
    """The tasks the format gate runs on: retrievable, low-hop, and balanced
    across the release's task shapes.

    The probe measures whether a submission can *drive the protocol*, so its
    task set must not smuggle in research difficulty. Taking the first N by
    ``task_id`` — which is a content hash, i.e. a lottery — did exactly that:
    it drew the full difficulty mixture, so the gate's real subject was "can
    this model solve a random exam sample", and the pass bar had to be met on
    tasks the incumbent king itself could not finish.

    Ordering: ``hops`` ascending, so a 2-hop join is preferred over a 9-hop
    synthesis, then ``task_id`` to break ties deterministically.

    There is deliberately no "pick the trivially retrievable ones" step. It
    was tried and measured: ranking the pool by whether one search with the
    question's own words surfaces its evidence put ALL 100 tasks of the live
    probe pool in the same bucket (rank > 10, i.e. never). That is by design —
    SCI2 rewrote the templates precisely so that questions stop handing
    full-text search their own source — so on this release no task is trivial,
    and a gate that assumed otherwise would be lying. What makes the gate
    difficulty-free is the *criterion* instead (see :func:`format_probe`):
    protocol compliance is counted per generation, not per solved task.

    Across templates the pick is round-robin in template-name order, so every
    shape the release mints is represented rather than whichever one won the
    hash lottery: a model whose formatting only survives one task shape must
    not pass.

    Pure and deterministic: same pool in, same slice out, on every validator.
    Returned in ``task_id`` order so rollout order is stable too.
    """
    limit = constants.FORMAT_PROBE_TASKS if n_tasks is None else n_tasks
    buckets: dict[str, list[Task]] = {}
    for task in sorted(tasks, key=lambda t: (t.hops, t.task_id)):
        buckets.setdefault(task.template, []).append(task)
    chosen: list[Task] = []
    while len(chosen) < limit and any(buckets.values()):
        for name in sorted(buckets):
            if len(chosen) >= limit:
                break
            if buckets[name]:
                chosen.append(buckets[name].pop(0))
    return sorted(chosen, key=lambda t: t.task_id)


def format_probe(
    backend: ModelBackend, probe_tasks: list[Task], env: "ResearchEnvironment"
) -> list[ProbeFailure]:
    """Protocol-compliance gate: can this submission drive the agent loop?

    Defends against harness-breaking submissions: a model that cannot emit
    well-formed action turns would score zero over an entire duel while
    burning its full GPU budget, so it is rejected after FORMAT_PROBE_TASKS
    cheap rollouts instead. Correctness is never required — the criterion is
    that an episode ends with a parsed ``<answer>`` and no protocol error
    (malformed twice, turn cap, timeout, crash).

    Compliance is judged per *generation*, not per solved task: the gate is
    the fraction of the model's outputs that carried a parseable action
    (:data:`~epago.constants.FORMAT_PROBE_MIN_COMPLIANCE`). Running out of
    turns or time is not a format failure — a model can speak the protocol
    perfectly and still fail to find an answer on a real corpus, and the old
    criterion ("ended with a parsed ``<answer>``") could not tell the two
    apart. That conflation is what made the gate unreachable: measured on the
    pinned reference model, 66% of probe episodes ended in an answer against a
    bar of 90%, while 76% of its generations were perfectly well formed.

    A model that formats perfectly but never commits to an answer therefore
    passes here — deliberately. :func:`degenerate_probe` is the gate for that,
    and it runs on these same rollouts.
    """
    tasks = probe_task_set(probe_tasks)
    if not tasks:
        return [ProbeFailure("no_probe_tasks", "format probe needs at least one task")]
    return _format_failures(_probe_rollouts(backend, tasks, env))


def _probe_rollouts(
    backend: ModelBackend, tasks: list[Task], env: "ResearchEnvironment"
) -> list[RolloutResult]:
    results: list[RolloutResult] = []
    for task in tasks:
        try:
            results.append(run_rollout(backend, task, env.tools_for_task(task)))
        except Exception as exc:  # noqa: BLE001 - a crash is a probe failure, not an abort
            results.append(
                RolloutResult(task.task_id, None, False, 0, 0.0, "none", f"crash: {exc}")
            )
    return results


#: Episode outcomes that are the model's protocol failing rather than the task
#: being hard. A crash is a failure of the submission; running out of turns or
#: time is not, and the format gate must not charge for it.
_CRASH_PREFIXES = ("crash", "rollout_crash", "judge_crash")


def compliance_rate(results: list[RolloutResult]) -> tuple[int, int]:
    """(generations carrying a parseable action, generations attempted).

    One turn is one generation, so this is the model's protocol hit rate over
    everything it emitted — the quantity the format gate is actually about.
    """
    generations = sum(r.turns for r in results)
    malformed = sum(r.malformed_actions for r in results)
    return generations - malformed, generations


def _format_failures(results: list[RolloutResult]) -> list[ProbeFailure]:
    crashed = [r for r in results if r.error and r.error.startswith(_CRASH_PREFIXES)]
    if crashed:
        return [
            ProbeFailure(
                "probe_crash",
                f"{len(crashed)}/{len(results)} probe rollouts crashed: {crashed[0].error}",
            )
        ]
    good, generations = compliance_rate(results)
    if generations == 0:
        return [ProbeFailure("no_generations", "probe produced no generations")]
    rate = good / generations
    if rate < constants.FORMAT_PROBE_MIN_COMPLIANCE:
        return [
            ProbeFailure(
                "format_probe",
                f"{good}/{generations} generations carried a well-formed action "
                f"({rate:.0%}), need {constants.FORMAT_PROBE_MIN_COMPLIANCE:.0%}",
            )
        ]
    return []


def degenerate_probe(results: list[RolloutResult]) -> list[ProbeFailure]:
    """Collapsed-policy gate: probe answers must not be constant or all empty.

    A model that emits the same (or no) answer regardless of the question can
    still pass the format probe; it must never reach a paired duel because a
    constant policy turns the duel into a pure coin flip on task composition.
    """
    if not results:
        return [ProbeFailure("no_results", "degenerate probe needs at least one rollout")]
    answers = [r.answer for r in results]
    non_empty = [a for a in answers if a and a.strip()]
    if not non_empty:
        return [ProbeFailure("empty_answers", f"all {len(results)} probe answers empty")]
    if len(results) >= 2 and len(set(non_empty)) == 1:
        return [
            ProbeFailure(
                "constant_answers",
                f"all {len(results)} probe answers identical: {non_empty[0][:40]!r}",
            )
        ]
    return []


def run_probes(challenger_dir: Path, king_dir: Path) -> list[ProbeFailure]:
    """Weight-level pre-duel gate: load sanity, then norm sanity vs the king.

    This is the torch-free floor every challenger must clear before any
    rollout is spent. Behavioural gates (format, degenerate) need an inference
    backend and are composed on top by :func:`make_probe_runner`.
    """
    failures = load_probe(challenger_dir)
    if failures:
        return failures
    return norm_sanity_probe(challenger_dir, king_dir)


def make_probe_runner(
    backend_factory: Callable[[Path], ModelBackend],
    env: "ResearchEnvironment",
    probe_tasks_fn: Callable[[], list[Task]],
) -> Callable[[Path, Path], list[ProbeFailure]]:
    """Compose the full pre-duel gate for the validator loop.

    Order is cheapest-first and short-circuits: weight-level checks spend no
    GPU; behavioural probes run one bounded rollout set and feed both the
    format and degenerate gates from the same trajectories.
    """

    def _run(challenger_dir: Path, king_dir: Path) -> list[ProbeFailure]:
        failures = run_probes(challenger_dir, king_dir)
        if failures:
            return failures
        tasks = probe_task_set(probe_tasks_fn())
        if not tasks:
            return [ProbeFailure("no_probe_tasks", "probe task supply is empty")]
        backend = backend_factory(challenger_dir)
        try:
            results = _probe_rollouts(backend, tasks, env)
        finally:
            backend.close()
        failures = _format_failures(results)
        if failures:
            return failures
        return degenerate_probe(results)

    return _run
