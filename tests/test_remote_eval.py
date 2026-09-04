"""Remote eval split tests: wire serializers, server endpoints, client runner,
and the wiring switch.

Everything runs on ScriptedBackend plus stub materializers over
fastapi.testclient — no torch, no vllm, no network. The core property under
test: a duel shipped over the wire produces a DuelOutcome bit-identical to the
same duel run in-process on the same inputs.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from epago.core.types import DuelSpec, ModelRef, Task, TaskOrigin
from epago.eval.backend import ModelBackend, ScriptedBackend
from epago.eval.duel import run_calibration_duel, run_duel
from epago.eval.probes import ProbeFailure
from epago.eval.remote import (
    DirRefIndex,
    DuelRequest,
    RemoteEvalError,
    RemoteEvalRunner,
    outcome_from_wire,
    outcome_to_wire,
    ref_from_wire,
    ref_to_wire,
    task_from_wire,
    task_to_wire,
)
from epago.eval.server import create_app
TEMPLATE_TOML = Path(__file__).parent / "data" / "chain-template.toml"

KING_REF = ModelRef(repo="org/king", digest="hf:" + "a" * 40)
CHALL_REF = ModelRef(repo="org/challenger", digest="hf:" + "b" * 40)


class FakeSession:
    def __init__(self, docs: dict[str, str]) -> None:
        self.docs = docs

    def search(self, query: str) -> str:
        return f"results for {query!r}: " + " ".join(sorted(self.docs))

    def browse(self, doc_id: str) -> str:
        return self.docs.get(doc_id, f"no document {doc_id!r}")


class FakeEnv:
    def tools_for_task(self, task: Task) -> FakeSession:
        return FakeSession({d: f"contents of {d}" for d in task.evidence_doc_ids})


def make_task(task_id: str, question: str, answer: str) -> Task:
    return Task(
        task_id=task_id,
        question=question,
        answer=answer,
        aliases=(f"alias of {answer}",),
        evidence_doc_ids=("d1", "d2"),
        masked_doc_ids=("m1",),
        origin=TaskOrigin.GENERATED_PUBLIC,
        template="test",
        hops=2,
    )


def _tasks(prefix: str, n: int) -> list[Task]:
    return [
        make_task(f"{prefix}{i:03d}", f"What is {prefix}{i:03d}?", f"ans-{prefix}{i:03d}")
        for i in range(n)
    ]


def _knower(known_ids: set[str]) -> ScriptedBackend:
    def policy(prompt: str) -> str:
        task_id = re.search(r"What is (\S+)\?", prompt).group(1)
        if task_id in known_ids:
            return f"<answer>ans-{task_id}</answer>"
        return f"<answer>wrong-{task_id}</answer>"

    return ScriptedBackend(policy)


def _factory(known_by_dir: dict[Path, set[str]]):
    def factory(model_dir: Path) -> ModelBackend:
        return _knower(known_by_dir[Path(model_dir)])

    return factory


def _stub_materializer(dirs: dict[str, Path]):
    """Materialize hook double: digest -> pre-agreed local dir, no network."""

    def materialize(ref: ModelRef, cache_dir: Path) -> Path:
        return dirs[ref.digest]

    return materialize


@pytest.fixture()
def arena(tmp_path):
    """A server + runner pair over scripted backends and stub materialization."""
    king_dir = tmp_path / "king"
    chall_dir = tmp_path / "challenger"
    pub, priv = _tasks("pub", 30), _tasks("prv", 30)
    king_known = {t.task_id for t in pub[:15]} | {t.task_id for t in priv[:15]}
    chall_known = (
        {t.task_id for t in pub[:20]}
        | {t.task_id for t in priv[:20]}
    )
    factory = _factory({king_dir: king_known, chall_dir: chall_known})
    app = create_app(
        FakeEnv(),
        factory,
        cache_dir=tmp_path / "server-cache",
        materialize=_stub_materializer(
            {KING_REF.digest: king_dir, CHALL_REF.digest: chall_dir}
        ),
        probe_runner=lambda challenger, king: [
            ProbeFailure("stub_probe", f"{challenger.name}|{king.name}")
        ],
    )
    index = DirRefIndex()
    index.register(KING_REF, king_dir)
    index.register(CHALL_REF, chall_dir)
    runner = RemoteEvalRunner(
        ref_resolver=index.resolve, client=TestClient(app), retry_wait_s=0.0
    )
    spec = DuelSpec(
        king_dir=king_dir,
        challenger_dir=chall_dir,
        public_tasks=pub,
        private_tasks=priv,
        block_hash_at_reveal="0xfeedbeef",
        author_hotkey="5" + "G" * 47,
        king_acc_ema=0.95,
        noise_floor=0.0005,
        round_id="r1-test",
    )
    return {
        "runner": runner,
        "spec": spec,
        "factory": factory,
        "king_dir": king_dir,
        "chall_dir": chall_dir,
        "app": app,
    }


# --- serializers ---------------------------------------------------------------


def test_task_wire_round_trip_is_exact() -> None:
    task = make_task("t007", "What is t007?", "ans-t007")
    over_json = json.loads(json.dumps(task_to_wire(task)))
    assert task_from_wire(over_json) == task
    assert isinstance(task_from_wire(over_json).aliases, tuple)


def test_ref_wire_round_trip_is_exact() -> None:
    assert ref_from_wire(json.loads(json.dumps(ref_to_wire(KING_REF)))) == KING_REF


def test_outcome_wire_round_trip_is_exact(arena) -> None:
    outcome = run_duel(arena["spec"], FakeEnv(), arena["factory"])
    over_json = json.loads(json.dumps(outcome_to_wire(outcome)))
    restored = outcome_from_wire(over_json)
    assert restored == outcome
    assert isinstance(restored.public.diffs, tuple)
    assert isinstance(restored.public_task_results, tuple)
    assert isinstance(restored.judge_tier_counts, tuple)


def test_duel_request_wire_round_trip_is_exact(arena) -> None:
    spec = arena["spec"]
    request = DuelRequest(
        king=KING_REF,
        challenger=CHALL_REF,
        public_tasks=spec.public_tasks,
        private_tasks=spec.private_tasks,
        block_hash_at_reveal=spec.block_hash_at_reveal,
        author_hotkey=spec.author_hotkey,
        king_acc_ema=spec.king_acc_ema,
        noise_floor=spec.noise_floor,
        round_id=spec.round_id,
    )
    assert DuelRequest.from_wire(json.loads(json.dumps(request.to_wire()))) == request


# --- full round trip ------------------------------------------------------------


def test_remote_duel_matches_local_bit_identically(arena) -> None:
    local = run_duel(arena["spec"], FakeEnv(), arena["factory"])
    remote = arena["runner"].run_duel(arena["spec"], None, None, None)
    assert remote == local
    # The new fields ride the wire too.
    assert remote.public_task_results == local.public_task_results
    assert remote.judge_tier_counts == local.judge_tier_counts
    assert remote.judge_invocation_rate == local.judge_invocation_rate


def test_remote_calibration_matches_local(arena) -> None:
    tasks = _tasks("cal", 10)
    local = run_calibration_duel(arena["king_dir"], tasks, FakeEnv(), arena["factory"])
    remote = arena["runner"].run_calibration_duel(arena["king_dir"], tasks, None, None)
    assert remote == local == 0.0


def test_remote_probes_compose_server_probe_runner(arena) -> None:
    failures = arena["runner"].run_probes(arena["chall_dir"], arena["king_dir"])
    assert [f.code for f in failures] == ["stub_probe"]
    assert failures[0].detail == "challenger|king"


def test_king_ref_resolver_overrides_dir_lookup(arena) -> None:
    runner = RemoteEvalRunner(
        ref_resolver=lambda d: (_ for _ in ()).throw(KeyError("must not be called")),
        king_ref_resolver=lambda: KING_REF,
        client=TestClient(arena["app"]),
        retry_wait_s=0.0,
    )
    rate = runner.run_calibration_duel(Path("/nowhere/king"), _tasks("cal", 4), None, None)
    assert rate == 0.0


def test_dir_ref_index_resolves_and_reports_unknown(tmp_path) -> None:
    index = DirRefIndex()
    index.register(KING_REF, tmp_path / "king")
    assert index.resolve(tmp_path / "king") == KING_REF
    with pytest.raises(KeyError, match="no ModelRef registered"):
        index.resolve(tmp_path / "stranger")


# --- auth -----------------------------------------------------------------------


def test_bearer_auth_enforced_on_posts_health_open(arena, tmp_path) -> None:
    app = create_app(
        FakeEnv(),
        arena["factory"],
        cache_dir=tmp_path / "cache2",
        materialize=_stub_materializer(
            {KING_REF.digest: arena["king_dir"], CHALL_REF.digest: arena["chall_dir"]}
        ),
        token="sekrit",
    )
    client = TestClient(app)
    assert client.get("/health").status_code == 200

    body = {"king": ref_to_wire(KING_REF), "tasks": [task_to_wire(t) for t in _tasks("cal", 2)]}
    assert client.post("/calibrate", json=body).status_code == 401
    assert (
        client.post(
            "/calibrate", json=body, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    ok = client.post("/calibrate", json=body, headers={"Authorization": "Bearer sekrit"})
    assert ok.status_code == 200 and ok.json() == {"rate": 0.0}

    probe_body = {"challenger": ref_to_wire(CHALL_REF), "king": ref_to_wire(KING_REF)}
    assert client.post("/probes", json=probe_body).status_code == 401

    unauth_runner = RemoteEvalRunner(
        ref_resolver=lambda d: KING_REF, client=TestClient(app), retry_wait_s=0.0
    )
    with pytest.raises(RemoteEvalError, match="401"):
        unauth_runner.run_calibration_duel(arena["king_dir"], _tasks("cal", 2), None, None)

    auth_runner = RemoteEvalRunner(
        token="sekrit",
        ref_resolver=lambda d: KING_REF,
        client=TestClient(app),
        retry_wait_s=0.0,
    )
    assert auth_runner.run_calibration_duel(
        arena["king_dir"], _tasks("cal", 2), None, None
    ) == 0.0


def test_token_defaults_to_env(arena, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EPAGO_EVAL_TOKEN", "envtoken")
    app = create_app(
        FakeEnv(),
        arena["factory"],
        cache_dir=tmp_path / "cache3",
        materialize=_stub_materializer({KING_REF.digest: arena["king_dir"]}),
    )
    client = TestClient(app)
    body = {"king": ref_to_wire(KING_REF), "tasks": [task_to_wire(t) for t in _tasks("cal", 2)]}
    assert client.post("/calibrate", json=body).status_code == 401
    assert (
        client.post(
            "/calibrate", json=body, headers={"Authorization": "Bearer envtoken"}
        ).status_code
        == 200
    )


# --- single-duel lock -------------------------------------------------------------


def test_concurrent_duel_gets_409(arena, tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_materialize(ref: ModelRef, cache_dir: Path) -> Path:
        entered.set()
        assert release.wait(timeout=30.0), "test orchestration stalled"
        return {KING_REF.digest: arena["king_dir"], CHALL_REF.digest: arena["chall_dir"]}[
            ref.digest
        ]

    app = create_app(
        FakeEnv(),
        arena["factory"],
        cache_dir=tmp_path / "cache4",
        materialize=blocking_materialize,
    )
    spec = arena["spec"]
    request = DuelRequest(
        king=KING_REF,
        challenger=CHALL_REF,
        public_tasks=spec.public_tasks,
        private_tasks=spec.private_tasks,
        block_hash_at_reveal=spec.block_hash_at_reveal,
        author_hotkey=spec.author_hotkey,
        king_acc_ema=spec.king_acc_ema,
        noise_floor=spec.noise_floor,
        round_id=spec.round_id,
    ).to_wire()

    client = TestClient(app)
    first: dict = {}

    def run_first() -> None:
        first["response"] = client.post("/duel", json=request)

    worker = threading.Thread(target=run_first)
    worker.start()
    try:
        assert entered.wait(timeout=30.0)
        # The first duel holds the single-duel lock; a second POST is refused.
        second = client.post("/duel", json=request)
        assert second.status_code == 409
        assert client.get("/health").json()["busy"] is True
        # The runner waits out a busy (409) eval rather than abandoning the duel,
        # but the wait is bounded — here it gives up quickly with a clear 409 error.
        busy_runner = RemoteEvalRunner(
            ref_resolver=lambda d: KING_REF,
            client=TestClient(app),
            retry_wait_s=0.0,
            busy_wait_s=0.02,
            busy_max_wait_s=0.05,
        )
        with pytest.raises(RemoteEvalError, match="409"):
            busy_runner.run_duel(spec, None, None, None)
    finally:
        release.set()
        worker.join(timeout=60.0)
    assert first["response"].status_code == 200
    outcome = outcome_from_wire(first["response"].json())
    assert outcome == run_duel(spec, FakeEnv(), arena["factory"])
    # Lock released: the server accepts the next duel.
    after = client.post("/duel", json=request)
    assert after.status_code == 200


# --- client transport behavior ------------------------------------------------------


def test_runner_retries_transient_errors_then_raises(monkeypatch) -> None:
    import httpx

    attempts: list[int] = []

    class FlakyClient:
        def post(self, path, json=None, headers=None):
            attempts.append(1)
            raise httpx.ConnectError("boom")

    runner = RemoteEvalRunner(
        ref_resolver=lambda d: KING_REF,
        client=FlakyClient(),
        retries=3,
        retry_wait_s=0.0,
    )
    with pytest.raises(RemoteEvalError, match="unreachable after 3 attempts"):
        runner.run_calibration_duel(Path("/x"), _tasks("cal", 1), None, None)
    assert len(attempts) == 3


def test_server_rejects_malformed_duel_request(arena, tmp_path) -> None:
    app = create_app(
        FakeEnv(),
        arena["factory"],
        cache_dir=tmp_path / "cache5",
        materialize=_stub_materializer({}),
    )
    client = TestClient(app)
    response = client.post("/duel", json={"king": {"repo": "x"}})
    assert response.status_code == 422


# --- wiring switch ------------------------------------------------------------------


@pytest.fixture(scope="module")
def wiring_corpus(tmp_path_factory) -> Path:
    from epago.environment.fixtures import build_fixture_corpus

    path = tmp_path_factory.mktemp("corpus") / "corpus.db"
    build_fixture_corpus(path, n_docs=240, seed=7)
    return path


def test_wiring_switch_uses_remote_runner_and_registers_refs(
    tmp_path, monkeypatch, wiring_corpus
) -> None:
    from epago import constants
    from epago.chain.client import MockChainClient
    from epago.config import load_config
    from epago.validator.wiring import build_production_deps

    monkeypatch.setattr(constants, "N_PRIV_TASKS", 8)
    monkeypatch.setenv("EPAGO_EVAL_URL", "http://gpu-box:8793")
    monkeypatch.setenv("EPAGO_EVAL_TOKEN", "sekrit")

    deps = build_production_deps(
        cfg=load_config(TEMPLATE_TOML),
        chain=MockChainClient(),
        state_dir=tmp_path / "state",
        corpus_path=wiring_corpus,
        cache_dir=tmp_path / "cache",
        wallet_hotkey="v0",
    )
    assert isinstance(deps.run_duel.__self__, RemoteEvalRunner)
    assert deps.run_duel.__self__ is deps.run_calibration_duel.__self__
    assert deps.run_probes.__self__ is deps.run_duel.__self__
    assert deps.llm_judge is None  # EPAGO_ENABLE_LLM_JUDGE not set

    # The materialize wrapper registers (ref, dir) pairs for reverse lookup.
    assert deps.materialize is not None
    snapshot = tmp_path / "snap"
    import epago.model.store as store

    monkeypatch.setattr(store, "materialize_model", lambda ref, cache: snapshot)
    got = deps.materialize(KING_REF, tmp_path / "cache")
    assert got == snapshot
    runner: RemoteEvalRunner = deps.run_duel.__self__
    assert runner._ref_resolver(snapshot) == KING_REF


def test_wiring_without_env_stays_in_process(tmp_path, monkeypatch, wiring_corpus) -> None:
    from epago import constants
    from epago.chain.client import MockChainClient
    from epago.config import load_config
    from epago.eval.duel import run_duel as local_run_duel
    from epago.validator.wiring import build_production_deps

    monkeypatch.setattr(constants, "N_PRIV_TASKS", 8)
    monkeypatch.delenv("EPAGO_EVAL_URL", raising=False)

    deps = build_production_deps(
        cfg=load_config(TEMPLATE_TOML),
        chain=MockChainClient(),
        state_dir=tmp_path / "state",
        corpus_path=wiring_corpus,
        cache_dir=tmp_path / "cache",
        wallet_hotkey="v0",
    )
    # In-process, but possibly bound to this box's GPU pool: on a multi-GPU
    # validator the local duel function is wrapped in a partial carrying the
    # pool, on a single-GPU one it is passed through untouched.
    assert getattr(deps.run_duel, "func", deps.run_duel) is local_run_duel
    assert deps.materialize is None
