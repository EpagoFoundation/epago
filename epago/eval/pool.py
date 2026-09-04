"""Multi-GPU eval: one full model replica per device, sweeps sharded across them.

A validator box usually has more than one GPU and the evaluator used exactly
one of them. This module is the placement layer that fixes that, under three
rules that are not negotiable:

**Replication, never tensor parallelism.** A sharded model changes reduction
order and therefore logits, so a 2-GPU validator and an 8-GPU validator would
disagree on borderline episodes and quorum would fracture. Every replica here
is a *whole* model on *one* device, loaded by the same
:class:`~epago.eval.backend.VllmBackend` a single-GPU validator loads, in a
process that can see exactly one GPU. There is nothing for a replica to
disagree with a single-GPU box about.

**One device is the floor, not a degenerate case.** With one visible GPU this
module is never constructed — :func:`resolve_devices` returns a single device,
the callers keep their existing single-backend path (``EPAGO_EVAL_LOW_VRAM``
swap included), and behaviour is byte-for-byte what it was. Multi-GPU is an
optimization; if it were a requirement, validation would centralize onto big
boxes and the network would lose its cheap validators.

**Episodes are independent, so sharding is sound.** An episode is one agent
rollout: its prompt is a pure function of its own transcript, and nothing in
:class:`~epago.eval.harness.Episode` reads another episode's state. Splitting a
task list across replicas therefore cannot change *which* text a model is asked
to continue. What it can change is *batch composition* — which episodes happen
to decode together — and on a continuous-batching engine batch shape perturbs
kernel numerics; ``EPAGO_VLLM_DETERMINISTIC`` (enforce_eager + concurrency 1)
exists to take that variable off the table.

It does not take the last one off. Measured on the reference stack (Qwen3-MoE
4-bit under vLLM 0.27, RTX 5090): the same prompt, the same engine, batch of
one, greedy, seeded, CUDA graphs off and prefix caching off, still decodes
differently run to run — the fused MoE kernels reduce in a nondeterministic
order. So no configuration of this evaluator, sharded or not, reproduces itself
bit-for-bit, and "identical to a single-GPU run" is not an available standard
for anything.

The available standard is the one the network already uses. A calibration duel
(:func:`epago.eval.duel.run_calibration_duel`) runs the king against *itself*
and reports the score-gap standard error, and the adaptive acceptance floor is
clamped by it — that is precisely how engine noise is priced into a verdict.
The question for replication is therefore not "is it identical" but "does it
add noise beyond the floor this box already has", which
``scripts/gpu_equivalence.py`` measures directly by running the one-replica
configuration twice and comparing every sharded configuration against the same
reference.

The work graph
--------------

A competition round mints ONE exam and the king answers it ONCE; its result
vector is reused for every pairing (see
:class:`~epago.core.types.RoundDuelSpec`). So a round with N entrants is N+1
**independent sweeps** of the same task set — there is no king/challenger
ordering to respect at scheduling time, only at scoring time. A duel is the
N=1 case: two sweeps.

The scheduler is therefore a work pool over sweeps, not a fixed king/challenger
split of the devices. Dedicating half the box to the king would idle those
cards for 32/33 of a large round.

Replication factor, the whole policy
------------------------------------

::

    replicas_per_sweep = max(1, n_devices // n_pending_sweeps)

That single line covers both regimes the operator cares about, with no mode
switch to get wrong:

===========================  ==========  ==============================================
pending sweeps (8 devices)   replicas    behaviour
===========================  ==========  ==============================================
2 (one duel)                 4           each sweep spread over 4 cards — latency mode
4                            2           two-way spread, all 8 cards busy
9+ (a real round)            1           straight work queue over sweeps — throughput
===========================  ==========  ==============================================

Model residency and affinity
----------------------------

4-bit weights are ~17 GB against a 32 GB card, so a device holds exactly ONE
model and switching costs a full load. Slots therefore keep their engine
between calls (the king stays warm across duels, which is the case that
matters) and a free slot prefers pending work for the model it already holds.
Affinity is worth little *within* one big round — 33 sweeps of 33 distinct
checkpoints must pay 33 loads no matter how they are ordered — but it is worth
a great deal *across* calls, where the king would otherwise be reloaded every
single duel.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from epago.core.types import RolloutResult, Task
from epago.eval.backend import ModelBackend

if TYPE_CHECKING:
    from epago.eval.judge import LlmJudge

logger = logging.getLogger(__name__)

__all__ = [
    "GpuPool",
    "PoolStats",
    "SweepRequest",
    "SweepResult",
    "build_pool",
    "place_engines",
    "plan_replicas",
    "resolve_devices",
    "shard_indices",
    "split_judge_device",
    "visible_devices",
]

#: Device selection for the eval pool. Either a count (``4`` = the first four
#: visible devices) or an explicit list of *logical* indices into the visible
#: set (``0,2,5``). A trailing comma forces list semantics for a single device
#: (``3,`` = "only logical device 3", where bare ``3`` would mean "three
#: devices"). Unset or ``all`` = every visible device.
DEVICES_ENV = "EPAGO_EVAL_GPUS"


# --- device discovery ---------------------------------------------------------


def _device_count() -> int:
    """How many CUDA devices this box exposes, without importing torch.

    The base validator install has no torch — device discovery has to work
    there too, or a CPU-side dry run of the wiring would explode. ``nvidia-smi``
    is asked first because it costs a fork instead of a multi-hundred-megabyte
    import; torch is the fallback for boxes where the binary is absent but the
    runtime is present (some containers).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        n = len([line for line in out.stdout.splitlines() if line.strip()])
        if n:
            return n
    except Exception:  # noqa: BLE001 - no nvidia-smi is a normal CPU box
        pass
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - no torch either: assume a single device
        return 1


def visible_devices() -> tuple[str, ...]:
    """The devices this process is allowed to touch, as CUDA identifies them.

    An externally set ``CUDA_VISIBLE_DEVICES`` is authoritative and is passed
    through verbatim (entries may be indices or ``GPU-<uuid>`` strings; both
    are valid values to hand a child process). This matters because the pool
    launches one worker per device with its own ``CUDA_VISIBLE_DEVICES``, and a
    child's value is interpreted against the *physical* enumeration, not
    against the parent's restriction — so the entries we forward must be the
    ones the operator (or the scheduler that started us) wrote down, never
    re-derived indices.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        entries = tuple(e.strip() for e in raw.split(",") if e.strip())
        # "-1" and "" are CUDA's ways of saying "no GPUs at all".
        return () if entries in ((), ("-1",)) else entries
    return tuple(str(i) for i in range(_device_count()))


def resolve_devices(spec: str | None = None) -> tuple[str, ...]:
    """Devices for the eval pool: :data:`DEVICES_ENV` applied to the visible set.

    ``spec`` defaults to the environment variable. Selection composes with an
    outer ``CUDA_VISIBLE_DEVICES`` because the indices are *logical* — they
    index the visible set the same way CUDA itself does, so
    ``CUDA_VISIBLE_DEVICES=4,5,6,7 EPAGO_EVAL_GPUS=0,1`` yields physical 4 and
    5, which is what an operator carving up a shared box expects.

    An out-of-range or unparseable selection is a configuration error and
    raises: silently falling back to "all GPUs" on a shared box would have this
    process trample a neighbour's cards.
    """
    devices = visible_devices()
    raw = (spec if spec is not None else os.environ.get(DEVICES_ENV, "")).strip()
    if not raw or raw.lower() == "all":
        return devices
    if "," in raw:
        try:
            picks = [int(e) for e in raw.split(",") if e.strip()]
        except ValueError as exc:
            raise ValueError(f"{DEVICES_ENV}={raw!r} is not a device list") from exc
        if any(i < 0 or i >= len(devices) for i in picks):
            raise ValueError(
                f"{DEVICES_ENV}={raw!r} selects a device outside the visible set "
                f"{devices!r}"
            )
        return tuple(devices[i] for i in picks)
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DEVICES_ENV}={raw!r} is neither a device count nor a comma list"
        ) from exc
    if count < 0:
        raise ValueError(f"{DEVICES_ENV}={raw!r} must not be negative")
    return devices[:count]


# --- planning -----------------------------------------------------------------


def plan_replicas(n_devices: int, n_sweeps: int) -> int:
    """Replicas per pending sweep — the entire allocation policy.

    Deliberately one line. A cleverer planner (per-sweep priorities, predicted
    episode cost, preemption) would be a second scheduling model for operators
    to reason about when a round runs slowly, and the win over integer division
    is bounded by the load imbalance across shards, which is small.
    """
    return max(1, n_devices // max(n_sweeps, 1))


def shard_indices(n_items: int, n_shards: int) -> list[list[int]]:
    """Round-robin index shards, ``shard s`` taking items ``s, s+k, s+2k, ...``.

    Round-robin rather than contiguous blocks because episode cost varies by an
    order of magnitude (a one-turn answer against a forty-turn search spiral)
    and task ids are sorted, so contiguous blocks can correlate cost with
    shard. Interleaving makes every shard a systematic sample of the exam.

    Static, one shard per replica, not a finer-grained work queue. That leaves
    real imbalance on the table — measured at 128 tasks over 2 replicas, the
    slower shard ran about 1.5x the faster one, because episode cost is
    heavy-tailed and the engine is not reproducible enough for a systematic
    sample to equalize it. Two reasons to accept that anyway: the imbalance is
    a sampling effect that shrinks with the shard size a real exam produces
    (hundreds of episodes per replica, not thirty-two), and the obvious fix —
    chopping each sweep into many small chunks — trades it for a tail on every
    chunk, since a chunk shorter than a few times ``ROLLOUT_CONCURRENCY`` spends
    its last steps with the engine half-empty. If a production round shows the
    imbalance persisting at full exam size, chunking at a size well above the
    concurrency is the change to make, and this scheduler already pulls
    multiple shards per replica.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    return [list(range(s, n_items, n_shards)) for s in range(n_shards)]


# --- work units ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepRequest:
    """One model answering one exam: the unit of parallel work.

    ``phases`` is ordered and each phase is scored separately, exactly as the
    sequential path scores them — the split is preserved rather than
    concatenated so that per-phase batch shapes match the single-GPU run.
    """

    model_dir: Path
    phases: tuple[tuple[str, tuple[Task, ...]], ...]
    label: str = ""


@dataclass(slots=True)
class SweepResult:
    """One sweep's per-phase results, or the error that stopped it.

    Failures are per-request rather than fatal for the batch: in a round, one
    checkpoint that cannot be loaded must not deny every other entrant its
    duel, and the caller already knows how to price a forfeit.
    """

    label: str
    phases: dict[str, list[RolloutResult]] = field(default_factory=dict)
    error: BaseException | None = None


@dataclass(slots=True)
class PoolStats:
    """Where a run's wall time went. Load time is the number that decides
    whether affinity scheduling earns its complexity, so it is measured rather
    than asserted."""

    loads: int = 0
    load_seconds: float = 0.0
    generate_seconds: float = 0.0
    wall_seconds: float = 0.0
    shards: int = 0

    @property
    def load_fraction(self) -> float:
        """Share of replica-seconds spent loading weights rather than decoding.

        The number that decides whether affinity scheduling is worth its
        complexity: high means the box is reloading checkpoints, low means it
        is answering questions.
        """
        busy = self.load_seconds + self.generate_seconds
        return self.load_seconds / busy if busy else 0.0

    def snapshot(self) -> "PoolStats":
        return PoolStats(
            loads=self.loads,
            load_seconds=self.load_seconds,
            generate_seconds=self.generate_seconds,
            wall_seconds=self.wall_seconds,
            shards=self.shards,
        )

    def since(self, earlier: "PoolStats") -> "PoolStats":
        return PoolStats(
            loads=self.loads - earlier.loads,
            load_seconds=self.load_seconds - earlier.load_seconds,
            generate_seconds=self.generate_seconds - earlier.generate_seconds,
            wall_seconds=self.wall_seconds - earlier.wall_seconds,
            shards=self.shards - earlier.shards,
        )


@dataclass(slots=True)
class _Shard:
    """One replica's slice of one sweep: the same model, a subset of the tasks."""

    request_index: int
    shard_index: int
    model_key: str
    model_dir: Path
    label: str
    #: phase -> (original task index, task) pairs, in task order within the shard
    work: dict[str, list[tuple[int, Task]]]


class _Slot:
    """One device and the single model it currently holds."""

    __slots__ = (
        "device", "engine", "model_key", "loads", "load_seconds", "generate_seconds", "busy",
    )

    def __init__(self, device: str) -> None:
        self.device = device
        self.engine: ModelBackend | None = None
        self.model_key: str | None = None
        self.loads = 0
        self.load_seconds = 0.0
        self.generate_seconds = 0.0
        #: Claimed by a sweep or a lease. Nothing else may put work on this
        #: device while it is set — two 17 GB engines do not fit on one card.
        self.busy = False


class PoolError(RuntimeError):
    """A replica could not be brought up or died mid-shard."""


# --- the pool -----------------------------------------------------------------


def _default_engine_factory(model_dir: Path, device: str) -> ModelBackend:
    from epago.eval.worker import WorkerBackend  # keeps vllm/subprocess late-bound

    return WorkerBackend(model_dir, device)


class GpuPool:
    """A fixed set of devices, each holding at most one model replica.

    Engines survive between :meth:`run_sweeps` calls on purpose: the king is
    swept again in the next duel and reloading 17 GB of weights every time
    would cost more than the parallelism saves. Eviction is lazy and per-slot —
    a slot drops its engine only when it is handed work for a different model —
    so residency is never assumed to be permanent, which is what a periodically
    recrowned king requires.

    ``engine_factory(model_dir, device) -> ModelBackend`` is injectable so the
    scheduling logic can be tested without a GPU; the default spawns a
    :class:`~epago.eval.worker.WorkerBackend` subprocess pinned to ``device``.
    """

    def __init__(
        self,
        devices: Sequence[str],
        *,
        engine_factory: Callable[[Path, str], ModelBackend] | None = None,
    ) -> None:
        if not devices:
            raise ValueError("a GPU pool needs at least one device")
        self._slots = [_Slot(str(d)) for d in devices]
        self._engine_factory = engine_factory or _default_engine_factory
        self._cond = threading.Condition()
        self._lock = self._cond  # the same mutex; a Condition is one
        self.stats = PoolStats()

    @property
    def width(self) -> int:
        """Number of devices, i.e. the maximum number of concurrent replicas."""
        return len(self._slots)

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(s.device for s in self._slots)

    def resident(self) -> dict[str, str]:
        """device -> resident model key, for /health and operator logs."""
        with self._lock:
            return {s.device: s.model_key for s in self._slots if s.model_key}

    def close(self) -> None:
        """Release every replica. Idempotent."""
        with self._lock:
            slots = list(self._slots)
        for slot in slots:
            self._release(slot)

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _release(slot: _Slot) -> None:
        engine = slot.engine
        slot.engine = None
        slot.model_key = None
        if engine is None:
            return
        try:
            engine.close()
        except Exception as exc:  # noqa: BLE001 - a stuck replica must not wedge the pool
            logger.warning("replica on device %s failed to close: %s", slot.device, exc)

    def _ensure_engine(
        self, slot: _Slot, model_key: str, model_dir: Path, label: str = ""
    ) -> ModelBackend:
        if slot.engine is not None and slot.model_key == model_key:
            return slot.engine
        self._release(slot)
        started = time.monotonic()
        engine = self._engine_factory(model_dir, slot.device)
        elapsed = time.monotonic() - started
        slot.engine = engine
        slot.model_key = model_key
        slot.loads += 1
        slot.load_seconds += elapsed
        with self._lock:
            self.stats.loads += 1
            self.stats.load_seconds += elapsed
        logger.info("loaded %s on device %s in %.1fs", label or model_key, slot.device, elapsed)
        return engine

    # -- leasing ---------------------------------------------------------------

    def borrow(self, model_dir: Path) -> ModelBackend:
        """One replica of ``model_dir``, held until the caller closes it.

        The escape hatch for the GPU work that is not a sweep — the pre-duel
        behavioural probes, which run a short bounded rollout set against one
        checkpoint. Without it those probes would build their own in-process
        engine on the default device, which is a device this pool already fills
        with a replica: two 17 GB engines, one 32 GB card, one OOM.

        Blocks until a device is free, prefers one already holding the
        checkpoint, and leaves the engine resident when the lease is closed so
        the next sweep of the same model does not reload it.
        """
        key = str(Path(model_dir).resolve())
        with self._cond:
            while True:
                slot = self._claim(key)
                if slot is not None:
                    break
                self._cond.wait()
        try:
            engine = self._ensure_engine(slot, key, Path(model_dir))
        except BaseException:
            self._unclaim([slot])
            raise
        return _LeasedBackend(self, slot, engine)

    def _claim(self, key: str | None) -> _Slot | None:
        """A free slot, preferring one that already holds ``key``. Caller holds the lock."""
        free = [s for s in self._slots if not s.busy]
        if not free:
            return None
        slot = next((s for s in free if s.model_key == key), free[0])
        slot.busy = True
        return slot

    def _unclaim(self, slots: Sequence[_Slot]) -> None:
        with self._cond:
            for slot in slots:
                slot.busy = False
            self._cond.notify_all()

    def _next_shard(self, slot: _Slot, pending: list[_Shard], failed: set[int]) -> _Shard | None:
        """Pick this slot's next shard, preferring one it need not reload for.

        Under the lock; ``pending`` is consumed in place. Shards belonging to a
        request that has already failed are dropped rather than run — the
        request's verdict is decided, and burning a GPU on the rest of a
        forfeited entrant's exam helps nobody.
        """
        with self._lock:
            while pending and pending[0].request_index in failed:
                pending.pop(0)
            for i, shard in enumerate(pending):
                if shard.request_index in failed:
                    continue
                if shard.model_key == slot.model_key:
                    return pending.pop(i)
            return pending.pop(0) if pending else None

    def run_sweeps(
        self,
        requests: Sequence[SweepRequest],
        *,
        session_factory: Callable[[Task], object],
        llm_judge: "LlmJudge | None" = None,
        concurrency: int | None = None,
        on_result: Callable[[int, str, int, RolloutResult], None] | None = None,
    ) -> list[SweepResult]:
        """Run every request, sharded across replicas, and stitch results back.

        Results come back in ``requests`` order and, within a phase, in task
        order — identical shape to running each sweep sequentially on one GPU.
        ``on_result(request_index, phase, task_index, result)`` fires from
        replica threads, so a caller that aggregates must be thread-safe.

        Per-episode execution is unchanged :func:`~epago.eval.harness.run_rollouts_batched`
        with the same ``ROLLOUT_CONCURRENCY``: each replica sees the same batch
        *sizes* a single-GPU sweep produces, only a different subset of tasks
        in them.
        """
        from epago.eval.harness import run_rollouts_batched

        if not requests:
            return []
        started = time.monotonic()
        before = self.stats.snapshot()
        # Claim every device that is not already leased out, and hold them for
        # the whole run. Replication is planned over what was actually claimed,
        # not over the nominal width, so a probe holding one card shrinks the
        # sweep instead of oversubscribing that card.
        with self._cond:
            while True:
                claimed = [s for s in self._slots if not s.busy]
                if claimed:
                    break
                self._cond.wait()
            for slot in claimed:
                slot.busy = True
        replicas = plan_replicas(len(claimed), len(requests))
        results = [SweepResult(label=r.label or str(r.model_dir)) for r in requests]
        slices: list[dict[str, list[tuple[int, Task]]]] = []
        for req in requests:
            per_shard: list[dict[str, list[tuple[int, Task]]]] = [{} for _ in range(replicas)]
            for phase, tasks in req.phases:
                for s, idxs in enumerate(shard_indices(len(tasks), replicas)):
                    per_shard[s][phase] = [(i, tasks[i]) for i in idxs]
            slices.append(per_shard)  # type: ignore[arg-type]

        pending: list[_Shard] = []
        for ri, req in enumerate(requests):
            results[ri].phases = {
                phase: [None] * len(tasks) for phase, tasks in req.phases  # type: ignore[misc]
            }
            key = str(Path(req.model_dir).resolve())
            for si in range(replicas):
                work = slices[ri][si]  # type: ignore[index]
                if not any(work.values()):
                    continue  # fewer tasks than replicas: nothing for this shard
                pending.append(
                    _Shard(
                        request_index=ri,
                        shard_index=si,
                        model_key=key,
                        model_dir=Path(req.model_dir),
                        label=req.label or key,
                        work=work,
                    )
                )
        total_shards = len(pending)
        failed: set[int] = set()
        #: Bounded re-placements for shards whose replica never came up; see the
        #: load-failure branch in ``drive``. Bounded so a box where every card is
        #: full terminates instead of passing one shard around forever.
        requeue_budget = len(claimed)
        judge = _LockedJudge(llm_judge) if (llm_judge is not None and self.width > 1) else llm_judge

        def drive(slot: _Slot, shard: _Shard | None) -> None:
            nonlocal requeue_budget
            while shard is not None:
                loaded = False
                try:
                    engine = self._ensure_engine(
                        slot, shard.model_key, shard.model_dir, shard.label
                    )
                    loaded = True
                    for phase, items in shard.work.items():
                        if not items:
                            continue
                        began = time.monotonic()

                        def report(pos: int, res: RolloutResult, *, _p=phase, _it=items,
                                   _ri=shard.request_index) -> None:
                            index = _it[pos][0]
                            results[_ri].phases[_p][index] = res
                            if on_result is not None:
                                on_result(_ri, _p, index, res)

                        run_rollouts_batched(
                            engine,
                            [t for _, t in items],
                            session_factory,
                            concurrency=concurrency,
                            llm_judge=judge,
                            on_result=report,
                        )
                        spent = time.monotonic() - began
                        slot.generate_seconds += spent
                        with self._lock:
                            self.stats.generate_seconds += spent
                except BaseException as exc:  # noqa: BLE001 - one bad sweep, not the batch
                    # A shard can only fail through the engine, so the replica
                    # is suspect: drop it rather than hand it the next sweep.
                    self._release(slot)
                    with self._lock:
                        requeue = not loaded and requeue_budget > 0
                        if requeue:
                            requeue_budget -= 1
                            pending.append(shard)
                    if requeue:
                        # The engine never came up, so this shard ran nothing.
                        # A card a neighbouring process filled between planning
                        # and loading is a placement problem, not a verdict, and
                        # charging it to the sweep threw away a whole entrant's
                        # duel — or aborted the round outright when the shard
                        # was the king's — while the pool's other replicas sat
                        # idle. Hand the shard back and retire this replica; the
                        # next thread to free up takes it. If none does, its
                        # tasks stay unscored and the request fails exactly as
                        # it did before, so this is never worse than forfeiting.
                        logger.warning(
                            "replica load failed on device %s (%s); requeued shard %d of %s",
                            slot.device, exc, shard.shard_index, shard.label,
                        )
                        with self._cond:
                            self._cond.notify_all()
                        return
                    logger.warning(
                        "sweep %s shard %d failed on device %s: %s",
                        shard.label, shard.shard_index, slot.device, exc,
                    )
                    with self._lock:
                        failed.add(shard.request_index)
                        results[shard.request_index].error = exc
                shard = self._next_shard(slot, pending, failed)

        # Deal one shard to every device before any device takes a second, and
        # do the dealing here rather than letting the threads race for it. In
        # the wide case (sweeps <= devices) that makes the placement a pure
        # function of the plan — every shard lands on its own card, whatever the
        # timing — instead of one fast replica draining the queue while seven
        # cards sit idle. Whatever is left over (the deep case) is pulled
        # dynamically, which is where a work queue earns its keep.
        opening = [(slot, self._next_shard(slot, pending, failed)) for slot in claimed]
        threads = [
            threading.Thread(
                target=drive, args=(slot, shard), name=f"epago-eval-gpu{slot.device}"
            )
            for slot, shard in opening
            if shard is not None
        ]
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        finally:
            self._unclaim(claimed)

        for res in results:
            if res.error is None and any(
                r is None for phase in res.phases.values() for r in phase
            ):
                res.error = PoolError(f"sweep {res.label!r} produced no result for some tasks")
        with self._lock:
            self.stats.shards += total_shards
            self.stats.wall_seconds += time.monotonic() - started
            run = self.stats.since(before)
        logger.info(
            "ran %d sweeps as %d shards over %d devices (%d replicas each): "
            "%.1fs wall, %.1fs loading across %d loads (%.0f%% of replica-seconds), "
            "%.1fs generating",
            len(requests), total_shards, len(claimed), replicas,
            run.wall_seconds, run.load_seconds, run.loads,
            100 * run.load_fraction, run.generate_seconds,
        )
        return results


class _LeasedBackend:
    """A pool replica borrowed for non-sweep work; closing returns the device.

    ``close()`` deliberately does *not* release the engine — the caller's
    contract with a plain backend factory is "close it when you are done with
    it", but here the pool owns residency, and evicting a checkpoint the next
    duel is about to sweep would throw away a load for nothing.
    """

    __slots__ = ("_pool", "_slot", "_engine", "device")

    def __init__(self, pool: "GpuPool", slot: _Slot, engine: ModelBackend) -> None:
        self._pool = pool
        self._slot = slot
        self._engine = engine
        #: Which card the lease was served from; readable after close, because
        #: callers log it.
        self.device = slot.device

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        return self._engine.generate(prompt, max_tokens, stop)

    def generate_many(self, prompts: list[str], max_tokens: int, stop: list[str]) -> list[str]:
        native = getattr(self._engine, "generate_many", None)
        if callable(native):
            return native(prompts, max_tokens, stop)
        return [self._engine.generate(p, max_tokens, stop) for p in prompts]

    def close(self) -> None:
        slot, self._slot = self._slot, None
        self._engine = None
        if slot is not None:
            self._pool._unclaim([slot])


class _LockedJudge:
    """Serializes judge calls: the fallback judge is one shared engine and vLLM
    is not re-entrant, so replica threads must queue at it. Judging is a
    four-token generation on the tail of a rollout, so the contention is
    negligible next to the episodes themselves."""

    __slots__ = ("_judge", "_lock")

    def __init__(self, judge: "LlmJudge") -> None:
        self._judge = judge
        self._lock = threading.Lock()

    def judge(self, question: str, truth: str, candidate: str) -> bool:
        with self._lock:
            return self._judge.judge(question, truth, candidate)


def split_judge_device(devices: Sequence[str]) -> tuple[tuple[str, ...], str | None]:
    """Carve one card out of the pool for the fallback LLM judge.

    The judge is a second model that has to stay resident *while* sweeps run —
    it is called on the tail of every episode whose answer the programmatic
    tiers cannot settle — so unlike the probes it cannot borrow a replica slot.
    On a pooled box it therefore gets a card of its own; the sweeps get the
    rest. Below three cards the split is refused (``(devices, None)``): giving
    the judge one of two cards would leave a single sweep device, which is not a
    pool at all, so those boxes keep today's arrangement of judge and engine
    sharing a card under ``EPAGO_VLLM_GPU_MEM_UTIL``.
    """
    found = tuple(devices)
    if len(found) < 3:
        return found, None
    return found[:-1], found[-1]


def place_engines(
    backend_factory: Callable[[Path], ModelBackend],
    *,
    kind: str = "vllm",
    judge_on_own_device: bool = False,
) -> tuple["GpuPool | None", Callable[[Path], ModelBackend], Callable[[Path], ModelBackend]]:
    """Decide where this box's three kinds of GPU work live.

    Returns ``(pool, probe_backend_factory, judge_backend_factory)``. A duel or
    round sweeps through ``pool``; the pre-duel behavioural probes borrow one of
    its replicas; the fallback judge, which must stay resident while replicas
    run, gets a card the pool never touches.

    Every one of those is ``(None, backend_factory, backend_factory)`` on a
    single-GPU box or a scripted backend — that is the whole point: nothing
    about placement exists until there is more than one card to place on.
    """
    if kind != "vllm":
        return None, backend_factory, backend_factory
    devices = resolve_devices()
    pool_devices, judge_device = (
        split_judge_device(devices) if judge_on_own_device else (devices, None)
    )
    pool = build_pool(devices=pool_devices)
    if pool is None:
        return None, backend_factory, backend_factory
    if judge_device is None:
        judge_factory = backend_factory
    else:
        logger.info("reserving device %s for the LLM judge", judge_device)

        def judge_factory(model_dir: Path) -> ModelBackend:
            from epago.eval.worker import WorkerBackend

            return WorkerBackend(model_dir, judge_device)

    return pool, pool.borrow, judge_factory


def build_pool(
    *,
    devices: Sequence[str] | None = None,
    engine_factory: Callable[[Path, str], ModelBackend] | None = None,
) -> GpuPool | None:
    """A pool for this box, or ``None`` when there is nothing to parallelize.

    ``None`` is the important return value: it is what keeps the single-GPU
    validator on the code path it has always run, LOW_VRAM swap and all. Every
    caller treats ``pool is None`` as "do exactly what you did before".
    """
    found = tuple(devices) if devices is not None else resolve_devices()
    if len(found) < 2:
        logger.info("eval pool disabled: %d usable device(s)", len(found))
        return None
    logger.info("eval pool over %d devices: %s", len(found), ",".join(found))
    return GpuPool(found, engine_factory=engine_factory)
