"""Multi-GPU placement tests: device resolution, the replication policy, the
sharded sweep scheduler, and the guarantee that sharding changes nothing a
verdict depends on.

No GPU, no torch, no vllm: every engine is a scripted double handed to the pool
through its injectable ``engine_factory``, so the scheduler's behaviour —
affinity, residency, replica counts, failure isolation, result stitching — is
exercised on a laptop and in CI. Because the double is exactly deterministic,
"sharding changes nothing a verdict depends on" is a meaningful assertion here:
it isolates the scheduler from the engine.

The real-hardware claim is weaker and lives in scripts/gpu_equivalence.py, which
needs cards. It is NOT bit-identical scores across replicas — no configuration
delivers that, since a quantized MoE under vLLM does not reproduce itself even
at batch of one with CUDA graphs and prefix caching off. It is that replication
adds no disagreement beyond the box's own measured floor, which is why that
script always compares sharded runs against a *repeated* one-replica baseline
rather than against zero.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from epago.core.types import Task, TaskOrigin
from epago.eval.duel import DuelSpec, run_calibration_duel, run_duel, run_round_duel
from epago.eval.pool import (
    GpuPool,
    SweepRequest,
    build_pool,
    plan_replicas,
    resolve_devices,
    shard_indices,
    visible_devices,
)

KING_DIR = Path("/models/king")
CHALL_DIR = Path("/models/chall")
KING_DIGEST = "hf:" + "a" * 40
CHALL_DIGEST = "hf:" + "b" * 40


# --- doubles ------------------------------------------------------------------


class FakeSession:
    def search(self, query: str) -> str:
        return f"results for {query!r}"

    def browse(self, doc_id: str) -> str:
        return f"contents of {doc_id}"


class FakeEnv:
    def tools_for_task(self, task: Task) -> FakeSession:
        return FakeSession()


class FakeEngine:
    """A scripted engine that remembers where it was placed.

    ``answers`` maps a task id to the answer text the model gives for it, so a
    'model' is fully determined by its own answer table — which is what lets a
    test assert that sharding never changes a score.
    """

    def __init__(self, model_dir: Path, device: str, answers: dict[str, str], log: list) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.answers = answers
        self.closed = 0
        self.batch_sizes: list[int] = []
        self._log = log
        log.append(("load", str(model_dir), device))

    @staticmethod
    def _task_id(prompt) -> str:
        if not isinstance(prompt, str):  # v4 harness passes chat messages
            prompt = "\n".join(str(m.get("content", "")) for m in prompt)
        return prompt.split("Question: ", 1)[1].split("\n", 1)[0].strip()

    def _answer(self, prompt: str) -> str:
        tid = self._task_id(prompt)
        return f"<answer>{self.answers.get(tid, 'wrong')}</answer>"

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        return self._answer(prompt)

    def generate_many(self, prompts: list[str], max_tokens: int, stop: list[str]) -> list[str]:
        self.batch_sizes.append(len(prompts))
        return [self._answer(p) for p in prompts]

    def close(self) -> None:
        self.closed += 1
        self._log.append(("close", str(self.model_dir), self.device))


def make_tasks(prefix: str, n: int) -> list[Task]:
    return [
        Task(
            task_id=f"{prefix}{i:03d}",
            question=f"{prefix}{i:03d}",
            answer=f"ans-{prefix}{i:03d}",
            aliases=(),
            evidence_doc_ids=("d1",),
            masked_doc_ids=(),
            origin=TaskOrigin.GENERATED_PUBLIC,
            template="t",
            hops=1,
        )
        for i in range(n)
    ]


def knower(known: set[str]) -> dict[str, str]:
    """Answer table for a model that gets ``known`` right and everything wrong."""
    return {tid: f"ans-{tid}" for tid in known}


class Factory:
    """Engine factory over a per-model-dir answer table, with a placement log."""

    def __init__(self, tables: dict[Path, dict[str, str]]) -> None:
        self.tables = {str(Path(k).resolve()): v for k, v in tables.items()}
        self.log: list = []
        self.engines: list[FakeEngine] = []
        self.fail: set[str] = set()
        self._lock = threading.Lock()

    def __call__(self, model_dir: Path, device: str) -> FakeEngine:
        key = str(Path(model_dir).resolve())
        if key in self.fail:
            raise RuntimeError(f"cannot load {key}")
        engine = FakeEngine(model_dir, device, self.tables.get(key, {}), self.log)
        with self._lock:
            self.engines.append(engine)
        return engine

    @property
    def loads(self) -> list[tuple[str, str]]:
        return [(m, d) for kind, m, d in self.log if kind == "load"]

    def backend_factory(self, model_dir: Path):
        """Same doubles, single-engine style, for the sequential comparison path."""
        return self(model_dir, "seq")


# --- device discovery ---------------------------------------------------------


def test_visible_devices_honours_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,5")
    assert visible_devices() == ("2", "3", "5")


def test_visible_devices_passes_uuids_through_verbatim(monkeypatch) -> None:
    # A scheduler may hand us UUIDs; they must survive to the child's env, so
    # nothing here may try to turn them into indices.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa,GPU-bbbb")
    assert visible_devices() == ("GPU-aaaa", "GPU-bbbb")


@pytest.mark.parametrize("raw", ["", "-1"])
def test_visible_devices_empty_means_no_gpus(monkeypatch, raw) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", raw)
    assert visible_devices() == ()


def test_resolve_devices_defaults_to_everything_visible(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("EPAGO_EVAL_GPUS", raising=False)
    assert resolve_devices() == ("0", "1", "2", "3")
    assert resolve_devices("all") == ("0", "1", "2", "3")


def test_resolve_devices_count_takes_the_first_n(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert resolve_devices("2") == ("0", "1")


def test_resolve_devices_list_indexes_the_visible_set(monkeypatch) -> None:
    """Selection composes with an outer restriction instead of fighting it."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert resolve_devices("0,1") == ("4", "5")
    assert resolve_devices("3,") == ("7",)  # trailing comma = list, not count


def test_resolve_devices_reads_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("EPAGO_EVAL_GPUS", "2")
    assert resolve_devices() == ("0", "1")


def test_resolve_devices_rejects_a_selection_outside_the_visible_set(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    with pytest.raises(ValueError):
        resolve_devices("0,9")
    with pytest.raises(ValueError):
        resolve_devices("nonsense")


def test_build_pool_returns_none_without_a_second_device(monkeypatch) -> None:
    """The single-GPU floor: no pool means every caller keeps its old path."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert build_pool() is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert build_pool() is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    pool = build_pool(engine_factory=Factory({}))
    assert pool is not None and pool.devices == ("3", "4")


# --- the replication policy ---------------------------------------------------


@pytest.mark.parametrize(
    ("devices", "sweeps", "expected"),
    [
        (1, 2, 1),      # one card: no replication to do
        (8, 2, 4),      # one duel on a big box: latency mode, 4 replicas a side
        (8, 4, 2),
        (8, 9, 1),      # a round of 8 entrants: straight work queue
        (8, 33, 1),     # a big round: still a work queue, never fewer than one
        (2, 2, 1),
        (4, 1, 4),      # a lone sweep gets the whole box
        (8, 0, 8),      # degenerate input must not divide by zero
    ],
)
def test_plan_replicas(devices, sweeps, expected) -> None:
    assert plan_replicas(devices, sweeps) == expected


def test_shard_indices_partitions_every_item_exactly_once() -> None:
    shards = shard_indices(200, 4)
    assert sorted(i for s in shards for i in s) == list(range(200))
    assert [len(s) for s in shards] == [50, 50, 50, 50]


def test_shard_indices_is_interleaved_and_deterministic() -> None:
    assert shard_indices(10, 3) == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert shard_indices(10, 3) == shard_indices(10, 3)


def test_shard_indices_tolerates_more_shards_than_items() -> None:
    shards = shard_indices(2, 4)
    assert shards == [[0], [1], [], []]


# --- the scheduler ------------------------------------------------------------


def phases_of(*groups: tuple[str, list[Task]]):
    return tuple((name, tuple(tasks)) for name, tasks in groups)


def test_sharded_sweep_matches_a_single_replica_exactly() -> None:
    """The property the whole design rests on: splitting a task list across
    replicas changes nothing about the results, only where they were produced."""
    tasks = make_tasks("t", 40)
    table = knower({t.task_id for t in tasks[:25]})
    phases = phases_of(("public", tasks[:20]), ("private", tasks[20:]))

    one = GpuPool(["0"], engine_factory=Factory({KING_DIR: table}))
    many = GpuPool(["0", "1", "2", "3"], engine_factory=Factory({KING_DIR: table}))
    solo = one.run_sweeps(
        [SweepRequest(KING_DIR, phases, "king")], session_factory=FakeEnv().tools_for_task
    )
    split = many.run_sweeps(
        [SweepRequest(KING_DIR, phases, "king")], session_factory=FakeEnv().tools_for_task
    )

    for phase in ("public", "private"):
        assert [(r.task_id, r.answer, r.correct) for r in solo[0].phases[phase]] == [
            (r.task_id, r.answer, r.correct) for r in split[0].phases[phase]
        ]


def test_results_come_back_in_task_order_not_completion_order() -> None:
    tasks = make_tasks("t", 37)
    pool = GpuPool(list("0123"), engine_factory=Factory({KING_DIR: knower(set())}))
    out = pool.run_sweeps(
        [SweepRequest(KING_DIR, phases_of(("public", tasks)), "king")],
        session_factory=FakeEnv().tools_for_task,
    )
    assert [r.task_id for r in out[0].phases["public"]] == [t.task_id for t in tasks]


def test_one_duel_on_eight_cards_replicates_each_side_four_ways() -> None:
    tasks = make_tasks("t", 32)
    factory = Factory({KING_DIR: knower(set()), CHALL_DIR: knower(set())})
    pool = GpuPool([str(i) for i in range(8)], engine_factory=factory)
    pool.run_sweeps(
        [
            SweepRequest(KING_DIR, phases_of(("public", tasks)), "king"),
            SweepRequest(CHALL_DIR, phases_of(("public", tasks)), "challenger"),
        ],
        session_factory=FakeEnv().tools_for_task,
    )
    placed = factory.loads
    assert len(placed) == 8                      # every card got a replica
    assert len({d for _, d in placed}) == 8      # each on its own device
    assert sum(1 for m, _ in placed if m == str(KING_DIR)) == 4
    assert sum(1 for m, _ in placed if m == str(CHALL_DIR)) == 4


def test_a_deep_queue_becomes_one_replica_per_sweep() -> None:
    """33 sweeps on 8 cards: no replication, just a work queue with affinity."""
    tasks = make_tasks("t", 8)
    dirs = [Path(f"/models/e{i}") for i in range(33)]
    factory = Factory({d: knower(set()) for d in dirs})
    pool = GpuPool([str(i) for i in range(8)], engine_factory=factory)
    out = pool.run_sweeps(
        [SweepRequest(d, phases_of(("public", tasks)), f"e{i}") for i, d in enumerate(dirs)],
        session_factory=FakeEnv().tools_for_task,
    )
    assert len(out) == 33
    assert all(len(r.phases["public"]) == 8 for r in out)
    assert len(factory.loads) == 33  # one load per distinct checkpoint, no more


def test_a_resident_model_is_not_reloaded_across_calls() -> None:
    """The king survives between duels — the case affinity actually pays for."""
    tasks = make_tasks("t", 16)
    factory = Factory({KING_DIR: knower(set()), CHALL_DIR: knower(set())})
    pool = GpuPool(list("0123"), engine_factory=factory)
    duel = [
        SweepRequest(KING_DIR, phases_of(("public", tasks)), "king"),
        SweepRequest(CHALL_DIR, phases_of(("public", tasks)), "challenger"),
    ]
    pool.run_sweeps(duel, session_factory=FakeEnv().tools_for_task)
    first = len(factory.loads)
    pool.run_sweeps(duel, session_factory=FakeEnv().tools_for_task)
    assert len(factory.loads) == first  # nothing reloaded: same two checkpoints

    # A new challenger evicts only what it needs to: the king stays put.
    other = Path("/models/chall2")
    factory.tables[str(other.resolve())] = knower(set())
    pool.run_sweeps(
        [
            SweepRequest(KING_DIR, phases_of(("public", tasks)), "king"),
            SweepRequest(other, phases_of(("public", tasks)), "challenger2"),
        ],
        session_factory=FakeEnv().tools_for_task,
    )
    new_loads = factory.loads[first:]
    assert new_loads and all(m == str(other) for m, _ in new_loads)


def test_a_device_never_holds_two_models_at_once() -> None:
    tasks = make_tasks("t", 24)
    dirs = [Path(f"/models/m{i}") for i in range(6)]
    factory = Factory({d: knower(set()) for d in dirs})
    pool = GpuPool(list("01"), engine_factory=factory)
    pool.run_sweeps(
        [SweepRequest(d, phases_of(("public", tasks)), f"m{i}") for i, d in enumerate(dirs)],
        session_factory=FakeEnv().tools_for_task,
    )
    live: dict[str, str] = {}
    for kind, model, device in factory.log:
        if kind == "load":
            assert device not in live, f"device {device} loaded {model} while holding {live[device]}"
            live[device] = model
        else:
            live.pop(device, None)
    assert set(pool.resident()) <= set(pool.devices)


def test_a_failed_checkpoint_does_not_take_the_batch_down() -> None:
    tasks = make_tasks("t", 12)
    good, bad = Path("/models/good"), Path("/models/bad")
    factory = Factory({good: knower({t.task_id for t in tasks}), bad: knower(set())})
    factory.fail.add(str(bad.resolve()))
    pool = GpuPool(list("01"), engine_factory=factory)
    out = pool.run_sweeps(
        [
            SweepRequest(bad, phases_of(("public", tasks)), "bad"),
            SweepRequest(good, phases_of(("public", tasks)), "good"),
        ],
        session_factory=FakeEnv().tools_for_task,
    )
    assert out[0].error is not None
    assert out[1].error is None
    assert all(r.correct for r in out[1].phases["public"])


def test_a_card_taken_by_a_neighbour_costs_a_replica_not_the_sweep() -> None:
    """A load that fails on ONE device must not forfeit the whole request.

    This is the shared-box case: another process fills a card between planning
    and loading, so exactly one replica cannot come up while every other card
    is fine. The shard has run nothing at that point, so it is re-placed on a
    healthy replica and the sweep still produces a full, correct result — where
    before, one occupied card scored an entrant a total loss (or aborted the
    round when the shard happened to be the king's).
    """
    tasks = make_tasks("t", 12)
    model = Path("/models/good")
    factory = Factory({model: knower({t.task_id for t in tasks})})
    factory.dead_devices = {"1"}
    original = factory.__call__

    def call(model_dir: Path, device: str):
        if device in factory.dead_devices:
            raise RuntimeError(f"no room on device {device}")
        return original(model_dir, device)

    pool = GpuPool(list("012"), engine_factory=call)
    out = pool.run_sweeps(
        [SweepRequest(model, phases_of(("public", tasks)), "good")],
        session_factory=FakeEnv().tools_for_task,
    )
    assert out[0].error is None
    assert len(out[0].phases["public"]) == len(tasks)
    assert all(r.correct for r in out[0].phases["public"])
    assert "1" not in {d for _, d in factory.loads}


def test_a_checkpoint_that_loads_nowhere_still_forfeits() -> None:
    """The re-placement budget is bounded: a genuinely unloadable checkpoint
    must still fail its request rather than circulate forever."""
    tasks = make_tasks("t", 6)
    bad = Path("/models/bad")
    factory = Factory({bad: knower(set())})
    factory.fail.add(str(bad.resolve()))
    pool = GpuPool(list("012"), engine_factory=factory)
    out = pool.run_sweeps(
        [SweepRequest(bad, phases_of(("public", tasks)), "bad")],
        session_factory=FakeEnv().tools_for_task,
    )
    assert out[0].error is not None


def test_progress_fires_once_per_task_with_stable_totals() -> None:
    tasks = make_tasks("t", 20)
    pool = GpuPool(list("0123"), engine_factory=Factory({KING_DIR: knower(set())}))
    seen: list[tuple[int, str, int]] = []
    lock = threading.Lock()

    def on_result(request_index, phase, index, res):
        with lock:
            seen.append((request_index, phase, index))

    pool.run_sweeps(
        [SweepRequest(KING_DIR, phases_of(("public", tasks[:12]), ("private", tasks[12:])), "k")],
        session_factory=FakeEnv().tools_for_task,
        on_result=on_result,
    )
    assert sorted(seen) == (
        [(0, "private", i) for i in range(8)] + [(0, "public", i) for i in range(12)]
    )


def test_close_releases_every_replica() -> None:
    tasks = make_tasks("t", 8)
    factory = Factory({KING_DIR: knower(set())})
    pool = GpuPool(list("012"), engine_factory=factory)
    pool.run_sweeps(
        [SweepRequest(KING_DIR, phases_of(("public", tasks)), "king")],
        session_factory=FakeEnv().tools_for_task,
    )
    assert pool.resident()
    pool.close()
    assert pool.resident() == {}
    assert all(e.closed == 1 for e in factory.engines)


def test_stats_separate_loading_from_generating() -> None:
    tasks = make_tasks("t", 8)
    factory = Factory({KING_DIR: knower(set())})
    pool = GpuPool(list("01"), engine_factory=factory)
    pool.run_sweeps(
        [SweepRequest(KING_DIR, phases_of(("public", tasks)), "king")],
        session_factory=FakeEnv().tools_for_task,
    )
    assert pool.stats.loads == 2
    assert pool.stats.shards == 2
    assert pool.stats.generate_seconds >= 0.0
    assert 0.0 <= pool.stats.load_fraction <= 1.0


def test_a_pool_needs_at_least_one_device() -> None:
    with pytest.raises(ValueError):
        GpuPool([])


# --- integration: the duel path -----------------------------------------------


def duel_spec(pub: list[Task], priv: list[Task]) -> DuelSpec:
    return DuelSpec(
        king_dir=KING_DIR,
        challenger_dir=CHALL_DIR,
        public_tasks=pub,
        private_tasks=priv,
        block_hash_at_reveal="0xabc123",
        author_hotkey="5" + "F" * 47,
        king_acc_ema=0.97,
        noise_floor=0.0005,
    )


def duel_tables(pub: list[Task], priv: list[Task]) -> dict[Path, dict[str, str]]:
    king = {t.task_id for t in pub[:20]} | {t.task_id for t in priv[:20]}
    chall = {t.task_id for t in pub[:34]} | {t.task_id for t in priv[:34]}
    return {KING_DIR: knower(king), CHALL_DIR: knower(chall)}


def test_pooled_duel_is_identical_to_the_single_gpu_duel() -> None:
    """A four-card validator and a one-card validator must return the same
    verdict inputs, or quorum is a lottery over hardware."""
    pub, priv = make_tasks("pub", 40), make_tasks("prv", 40)
    spec, tables = duel_spec(pub, priv), duel_tables(pub, priv)

    sequential = run_duel(spec, FakeEnv(), Factory(tables).backend_factory)
    pool = GpuPool(list("0123"), engine_factory=Factory(tables))
    pooled = run_duel(spec, FakeEnv(), Factory(tables).backend_factory, pool=pool)

    assert pooled.public_task_results == sequential.public_task_results
    assert pooled.public == sequential.public
    assert pooled.private == sequential.private
    assert pooled.lcb_pub == sequential.lcb_pub
    assert pooled.delta == sequential.delta
    assert pooled.accepted == sequential.accepted
    assert pooled.boot_seed_hex == sequential.boot_seed_hex
    assert pooled.judge_tier_counts == sequential.judge_tier_counts


def test_pooled_duel_never_calls_the_sequential_backend_factory() -> None:
    pub, priv = make_tasks("pub", 12), make_tasks("prv", 12)
    tables = duel_tables(pub, priv)
    unused = Factory(tables)
    pool = GpuPool(list("01"), engine_factory=Factory(tables))
    run_duel(duel_spec(pub, priv), FakeEnv(), unused.backend_factory, pool=pool)
    assert unused.loads == []


def test_pooled_duel_reports_progress_for_both_models() -> None:
    pub, priv = make_tasks("pub", 16), make_tasks("prv", 16)
    events: list[dict] = []
    lock = threading.Lock()

    def on_progress(event: dict) -> None:
        with lock:
            events.append(event)

    pool = GpuPool(list("0123"), engine_factory=Factory(duel_tables(pub, priv)))
    run_duel(
        duel_spec(pub, priv),
        FakeEnv(),
        Factory(duel_tables(pub, priv)).backend_factory,
        pool=pool,
        on_progress=on_progress,
    )
    assert len(events) == 64
    assert {e["model"] for e in events} == {"king", "challenger"}
    assert {e["phase"] for e in events} == {"public", "private"}
    assert {e["total"] for e in events} == {16}
    assert all(1 <= e["index"] <= 16 for e in events)


def test_a_dead_replica_fails_the_duel_rather_than_scoring_it() -> None:
    """Half a duel is not a verdict: an unloadable side must raise, never be
    silently scored as 200 wrong answers."""
    pub, priv = make_tasks("pub", 8), make_tasks("prv", 8)
    tables = duel_tables(pub, priv)
    factory = Factory(tables)
    factory.fail.add(str(CHALL_DIR.resolve()))
    pool = GpuPool(list("01"), engine_factory=factory)
    with pytest.raises(RuntimeError):
        run_duel(duel_spec(pub, priv), FakeEnv(), Factory(tables).backend_factory, pool=pool)


# --- integration: rounds and calibration --------------------------------------


def round_spec(pub, priv, entrant_dirs):
    from epago.core.types import Entrant, RoundDuelSpec

    return RoundDuelSpec(
        king_dir=KING_DIR,
        entrants=tuple(
            Entrant(
                digest=f"hf:{chr(97 + i) * 40}",
                repo=f"m{i}/EPAGO-DR-4B-x",
                author_hotkey=f"hk-{i}",
                challenger_dir=d,
            )
            for i, d in enumerate(entrant_dirs)
        ),
        public_tasks=pub,
        private_tasks=priv,
        round=7,
        round_block_hash="0xabc123",
        king_acc_ema=0.97,
        noise_floor=0.0005,
    )


def test_pooled_round_matches_the_sequential_round() -> None:
    pub, priv = make_tasks("pub", 30), make_tasks("prv", 30)
    dirs = [Path(f"/models/e{i}") for i in range(5)]
    tables = {KING_DIR: knower({t.task_id for t in pub[:10]} | {t.task_id for t in priv[:10]})}
    for i, d in enumerate(dirs):
        tables[d] = knower(
            {t.task_id for t in pub[: 10 + 4 * i]} | {t.task_id for t in priv[: 10 + 4 * i]}
        )
    spec = round_spec(pub, priv, dirs)

    sequential = run_round_duel(spec, FakeEnv(), Factory(tables).backend_factory)
    pool = GpuPool(list("0123"), engine_factory=Factory(tables))
    pooled = run_round_duel(spec, FakeEnv(), Factory(tables).backend_factory, pool=pool)

    assert [r.entrant.digest for r in pooled] == [r.entrant.digest for r in sequential]
    for got, want in zip(pooled, sequential):
        assert got.outcome.public_task_results == want.outcome.public_task_results
        assert got.outcome.lcb_pub == want.outcome.lcb_pub
        assert got.outcome.accepted == want.outcome.accepted
        assert got.outcome.public == want.outcome.public
        assert got.outcome.private == want.outcome.private


def test_pooled_round_sweeps_the_king_once_and_each_entrant_once() -> None:
    pub, priv = make_tasks("pub", 12), make_tasks("prv", 12)
    dirs = [Path(f"/models/e{i}") for i in range(4)]
    tables = {KING_DIR: knower(set()), **{d: knower(set()) for d in dirs}}
    factory = Factory(tables)
    pool = GpuPool(list("01"), engine_factory=factory)
    run_round_duel(
        round_spec(pub, priv, dirs), FakeEnv(), Factory(tables).backend_factory, pool=pool
    )
    loaded = [m for m, _ in factory.loads]
    assert loaded.count(str(KING_DIR)) == 1          # N+1 sweeps, not 2N
    assert sorted(loaded) == sorted([str(KING_DIR)] + [str(d) for d in dirs])


def test_a_broken_entrant_forfeits_and_the_round_continues() -> None:
    pub, priv = make_tasks("pub", 12), make_tasks("prv", 12)
    dirs = [Path("/models/ok"), Path("/models/broken")]
    tables = {
        KING_DIR: knower(set()),
        dirs[0]: knower({t.task_id for t in pub} | {t.task_id for t in priv}),
        dirs[1]: knower(set()),
    }
    factory = Factory(tables)
    factory.fail.add(str(dirs[1].resolve()))
    pool = GpuPool(list("01"), engine_factory=factory)
    results = run_round_duel(
        round_spec(pub, priv, dirs), FakeEnv(), Factory(tables).backend_factory, pool=pool
    )
    assert len(results) == 2
    assert results[0].outcome.accepted            # the healthy entrant still ran
    assert not results[1].outcome.accepted
    assert results[1].outcome.lcb_pub == -1.0     # scored as a forfeit


def test_a_broken_king_stops_the_round() -> None:
    pub, priv = make_tasks("pub", 8), make_tasks("prv", 8)
    dirs = [Path("/models/ok")]
    tables = {KING_DIR: knower(set()), dirs[0]: knower(set())}
    factory = Factory(tables)
    factory.fail.add(str(KING_DIR.resolve()))
    pool = GpuPool(list("01"), engine_factory=factory)
    with pytest.raises(RuntimeError):
        run_round_duel(
            round_spec(pub, priv, dirs), FakeEnv(), Factory(tables).backend_factory, pool=pool
        )


def test_pooled_calibration_ties_when_the_model_is_deterministic() -> None:
    """Identical weights answering twice must disagree nowhere, whichever card
    each run lands on — a nonzero floor here would be the pool inventing noise."""
    tasks = make_tasks("cal", 24)
    tables = {KING_DIR: knower({t.task_id for t in tasks[:12]})}
    pool = GpuPool(list("0123"), engine_factory=Factory(tables))
    rate = run_calibration_duel(
        KING_DIR, tasks, FakeEnv(), Factory(tables).backend_factory, None, pool=pool
    )
    assert rate == 0.0
    sequential = run_calibration_duel(
        KING_DIR, tasks, FakeEnv(), Factory(tables).backend_factory, None
    )
    assert rate == sequential


# --- leasing and non-sweep GPU work -------------------------------------------


def test_borrow_hands_back_a_replica_and_keeps_it_resident() -> None:
    """The probe gate borrows a card instead of building its own engine on top
    of one — two 17 GB engines do not fit on a 32 GB card."""
    factory = Factory({KING_DIR: knower(set())})
    pool = GpuPool(list("01"), engine_factory=factory)
    leased = pool.borrow(KING_DIR)
    assert leased.generate("Question: t000\n", 16, []) == "<answer>wrong</answer>"
    assert leased.generate_many(["Question: t000\n"], 16, []) == ["<answer>wrong</answer>"]
    leased.close()
    assert len(factory.loads) == 1
    assert pool.resident()  # closing the lease returns the device, not the weights

    again = pool.borrow(KING_DIR)
    again.close()
    assert len(factory.loads) == 1  # affinity: the resident replica was reused


def test_a_sweep_never_schedules_onto_a_leased_device() -> None:
    tasks = make_tasks("t", 12)
    factory = Factory({KING_DIR: knower(set()), CHALL_DIR: knower(set())})
    pool = GpuPool(list("012"), engine_factory=factory)
    leased = pool.borrow(CHALL_DIR)
    lent_device = leased.device
    try:
        pool.run_sweeps(
            [SweepRequest(KING_DIR, phases_of(("public", tasks)), "king")],
            session_factory=FakeEnv().tools_for_task,
        )
    finally:
        leased.close()
    # Three cards, one leased: the sweep replicated over the other two only, and
    # the leased card still holds what it was lent.
    king_devices = {d for m, d in factory.loads if m == str(KING_DIR)}
    assert lent_device not in king_devices
    assert len(king_devices) == 2


def test_a_failed_borrow_releases_the_device() -> None:
    factory = Factory({KING_DIR: knower(set())})
    factory.fail.add(str(KING_DIR.resolve()))
    pool = GpuPool(list("01"), engine_factory=factory)
    with pytest.raises(RuntimeError):
        pool.borrow(KING_DIR)
    factory.fail.clear()
    leased = pool.borrow(KING_DIR)  # would block forever if the slot leaked
    leased.close()


def test_split_judge_device_reserves_one_card_above_two() -> None:
    from epago.eval.pool import split_judge_device

    assert split_judge_device(["0", "1", "2", "3"]) == (("0", "1", "2"), "3")
    # Two cards: reserving one would leave a single sweep device, which is not a
    # pool — so the judge keeps sharing, exactly as it does today.
    assert split_judge_device(["0", "1"]) == (("0", "1"), None)
    assert split_judge_device(["0"]) == (("0",), None)


def test_place_engines_is_inert_without_a_second_device(monkeypatch) -> None:
    from epago.eval.pool import place_engines

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("EPAGO_EVAL_GPUS", raising=False)
    base = Factory({}).backend_factory
    assert place_engines(base) == (None, base, base)


def test_place_engines_is_inert_for_a_scripted_backend(monkeypatch) -> None:
    from epago.eval.pool import place_engines

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    base = Factory({}).backend_factory
    assert place_engines(base, kind="scripted") == (None, base, base)


def test_place_engines_routes_probes_to_the_pool(monkeypatch) -> None:
    from epago.eval.pool import place_engines

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("EPAGO_EVAL_GPUS", raising=False)
    base = Factory({}).backend_factory
    pool, probe_factory, judge_factory = place_engines(base)
    assert pool is not None and pool.devices == ("0", "1", "2", "3")
    assert probe_factory == pool.borrow
    assert judge_factory is base  # no local judge wanted: nothing reserved


def test_place_engines_reserves_a_card_when_the_judge_is_local(monkeypatch) -> None:
    from epago.eval.pool import place_engines

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("EPAGO_EVAL_GPUS", raising=False)
    base = Factory({}).backend_factory
    pool, _, judge_factory = place_engines(base, judge_on_own_device=True)
    assert pool is not None
    assert pool.devices == ("0", "1", "2")  # the judge's card is not the pool's
    assert judge_factory is not base


def test_needs_local_judge_engine(monkeypatch) -> None:
    from epago.config import load_config
    from epago.eval.judge import needs_local_judge_engine

    cfg = load_config()
    for name in (
        "EPAGO_ENABLE_LLM_JUDGE",
        "EPAGO_JUDGE_API_BASE",
        "EPAGO_JUDGE_API_MODEL",
        "EPAGO_JUDGE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert needs_local_judge_engine(cfg) is False  # opt-in, and off by default

    monkeypatch.setenv("EPAGO_ENABLE_LLM_JUDGE", "1")
    pinned = _cfg_with_judge_digest(cfg, "sha256:" + "ab" * 32)
    assert needs_local_judge_engine(pinned) is True
    assert needs_local_judge_engine(cfg) is (
        set(cfg.eval.judge_digest.split(":", 1)[-1]) != {"0"}
    )

    # A fully configured hosted judge holds no local weights: no card needed.
    monkeypatch.setenv("EPAGO_JUDGE_API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("EPAGO_JUDGE_API_MODEL", "m")
    monkeypatch.setenv("EPAGO_JUDGE_API_KEY", "k")
    assert needs_local_judge_engine(pinned) is False
    # A half-configured one falls through to the local model, which does.
    monkeypatch.delenv("EPAGO_JUDGE_API_KEY")
    assert needs_local_judge_engine(pinned) is True


def _cfg_with_judge_digest(cfg, digest: str):
    from dataclasses import replace

    return replace(cfg, eval=replace(cfg.eval, judge_digest=digest))


# --- integration: the eval server ---------------------------------------------


def test_eval_server_duels_through_the_pool_and_reports_residency(tmp_path) -> None:
    """The server hands /duel to the pool and /health tells the operator which
    card is holding what — the only visibility an operator has into placement."""
    from fastapi.testclient import TestClient

    from epago.eval.server import create_app

    king_dir, chall_dir = tmp_path / "king", tmp_path / "chall"
    king_dir.mkdir()
    chall_dir.mkdir()
    pub, priv = make_tasks("pub", 20), make_tasks("prv", 20)
    tables = {
        king_dir: knower({t.task_id for t in pub[:8]} | {t.task_id for t in priv[:8]}),
        chall_dir: knower({t.task_id for t in pub[:14]} | {t.task_id for t in priv[:14]}),
    }
    factory = Factory(tables)
    pool = GpuPool(list("0123"), engine_factory=factory)
    unused = Factory(tables)
    app = create_app(
        FakeEnv(),
        unused.backend_factory,
        cache_dir=tmp_path / "cache",
        materialize=lambda ref, cache: {KING_DIGEST: king_dir, CHALL_DIGEST: chall_dir}[ref.digest],
        probe_runner=lambda challenger, king: [],
        pool=pool,
    )
    client = TestClient(app)

    def wire(task):
        return {
            "task_id": task.task_id,
            "question": task.question,
            "answer": task.answer,
            "aliases": list(task.aliases),
            "evidence_doc_ids": list(task.evidence_doc_ids),
            "masked_doc_ids": list(task.masked_doc_ids),
            "origin": task.origin.value,
            "template": task.template,
            "hops": task.hops,
        }

    body = {
        "king": {"repo": "org/king", "digest": KING_DIGEST},
        "challenger": {"repo": "org/chall", "digest": CHALL_DIGEST},
        "public_tasks": [wire(t) for t in pub],
        "private_tasks": [wire(t) for t in priv],
        "block_hash_at_reveal": "0xfeedbeef",
        "author_hotkey": "5" + "G" * 47,
        "king_acc_ema": 0.9,
        "noise_floor": 0.001,
    }
    outcome = client.post("/duel", json=body)
    assert outcome.status_code == 200, outcome.text
    assert outcome.json()["public"]["n_tasks"] == 20

    # The sweeps ran on pool replicas, not on the server's own backend cache.
    assert unused.loads == []
    assert len(factory.loads) == 4  # 2 sweeps x 2 replicas on 4 cards

    health = client.get("/health").json()
    assert sorted(health["gpu_devices"]) == ["0", "1", "2", "3"]
    assert set(health["gpu_residency"].values()) == {
        str(king_dir.resolve()), str(chall_dir.resolve())
    }
    # Both checkpoints stay resident: nothing is evicted between duels, which is
    # how the king stays warm for the next challenger.
    assert len(health["gpu_residency"]) == 4


def test_sharding_preserves_multi_turn_episodes_of_varying_length() -> None:
    """Shard boundaries must not touch per-episode logic.

    The single-turn case cannot catch a scheduler that leaks state between
    episodes, because there is no state to leak. Here every episode searches a
    task-dependent number of times before answering, so an episode's own
    transcript is the only thing that can produce its answer — and a sharded run
    must still reproduce the sequential run turn for turn.
    """
    from epago.eval.harness import run_rollouts_batched

    tasks = make_tasks("t", 30)

    class Searcher:
        """Answers after (task index mod 5) searches, then reports its own path."""

        def __init__(self, model_dir, device="seq") -> None:
            self.device = device
            self.closed = 0

        @staticmethod
        def _reply(prompt) -> str:
            if not isinstance(prompt, str):  # v4 harness passes chat messages
                prompt = "\n".join(str(m.get("content", "")) for m in prompt)
            tid = prompt.split("Question: ", 1)[1].split("\n", 1)[0].strip()
            want = int(tid[1:]) % 5
            done = prompt.count("<tool_response>")
            if done < want:
                return (
                    '<tool_call>{"name": "search", "arguments": {"query": ["'
                    + f"{tid} step {done}"
                    + '"]}}</tool_call>'
                )
            return f"<answer>ans-{tid}</answer>"

        def generate(self, prompt, max_tokens, stop):
            return self._reply(prompt)

        def generate_many(self, prompts, max_tokens, stop):
            return [self._reply(p) for p in prompts]

        def close(self):
            self.closed += 1

    sequential = run_rollouts_batched(
        Searcher(KING_DIR), tasks, FakeEnv().tools_for_task
    )
    pool = GpuPool(list("01234"), engine_factory=Searcher)
    out = pool.run_sweeps(
        [SweepRequest(KING_DIR, phases_of(("public", tasks)), "king")],
        session_factory=FakeEnv().tools_for_task,
    )
    assert out[0].error is None
    sharded = out[0].phases["public"]
    assert [(r.task_id, r.answer, r.correct, r.turns, r.error) for r in sharded] == [
        (r.task_id, r.answer, r.correct, r.turns, r.error) for r in sequential
    ]
    assert {r.turns for r in sharded} == {1, 2, 3, 4, 5}  # the episodes really did differ


# --- replica process lifecycle (no GPU: the protocol layer only) --------------


def _spawn_replica():
    """A worker process with the protocol socket wired up but no model loaded."""
    import os
    import socket
    import subprocess
    import sys

    from epago.eval.worker import FD_ENV, _package_root

    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    env = {**os.environ, FD_ENV: str(child.fileno()), "PYTHONPATH": _package_root()}
    proc = subprocess.Popen(
        [sys.executable, "-m", "epago.eval.worker"],
        env=env,
        pass_fds=(child.fileno(),),
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    child.close()
    parent.settimeout(30)
    return proc, parent


def test_a_replica_answers_the_protocol_before_any_model_is_loaded() -> None:
    from epago.eval.worker import _recv, _send

    proc, sock = _spawn_replica()
    try:
        _send(sock, {"op": "generate", "prompts": ["x"], "max_tokens": 4, "stop": []})
        reply = _recv(sock)
        assert reply["ok"] is False and "no model loaded" in reply["error"]
        _send(sock, {"op": "wat"})
        assert _recv(sock) == {"ok": False, "error": "unknown op 'wat'"}
        _send(sock, {"op": "close"})
        assert _recv(sock) == {"ok": True}
        assert proc.wait(timeout=30) is not None
    finally:
        sock.close()
        if proc.poll() is None:  # pragma: no cover - only on a failed assertion
            proc.kill()


def test_a_replica_exits_when_its_parent_goes_away() -> None:
    """A replica that outlives its parent is a card nobody can reclaim.

    The engine runs in a further child of the replica, and vLLM leaves
    non-daemon threads behind, so neither returning from the loop nor relying on
    reparenting is enough — the replica takes its whole process group down.
    """
    proc, sock = _spawn_replica()
    try:
        sock.close()  # exactly what a dead parent looks like from over there
        assert proc.wait(timeout=30) is not None
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a failed assertion
            proc.kill()
