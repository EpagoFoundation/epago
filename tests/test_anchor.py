"""External benchmark anchor tests: loading, closed-book rollouts, divergence
math, the validator's scheduled hook, and the dashboard section.

The anchor is observational by contract — nothing here touches verdicts,
weights, or coronation, and the tests pin that the hook only ever records
history and raises the machine-readable alarm.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from epago import constants
from epago.chain.client import MockChainClient
from epago.config import load_config
from epago.dashboard.export import DashboardInputs, export_dashboard
from epago.eval.anchor import (
    AnchorSession,
    AnchorTask,
    divergence,
    load_benchmark,
    run_anchor,
)
from epago.eval.backend import ScriptedBackend
from epago.validator.service import Deps, ValidatorService
from epago.validator.state import ValidatorState

BENCH_ITEMS = [
    {"id": "gq-1", "question": "What color is the clear daytime sky?", "answer": "blue"},
    {"question": "How many sides does a hexagon have?", "answer": "6", "aliases": ["six"]},
    {"id": "gq-3", "question": "What is the capital of France?", "answer": "Paris"},
]


def write_benchmark(path: Path, items=None) -> Path:
    path.write_text("\n".join(json.dumps(i) for i in (items or BENCH_ITEMS)) + "\n")
    return path


# ------------------------------------------------------------- load_benchmark


def test_load_benchmark_digest_and_ordering(tmp_path):
    path = write_benchmark(tmp_path / "bench.jsonl")
    digest, tasks = load_benchmark(path)
    assert digest == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    # File order is the deterministic order; missing ids become positional.
    assert [t.task_id for t in tasks] == ["gq-1", "anchor-00001", "gq-3"]
    assert tasks[1].aliases == ("six",)
    assert tasks[0] == AnchorTask(
        task_id="gq-1", question="What color is the clear daytime sky?", answer="blue", aliases=()
    )
    # Same bytes -> same digest and same tasks on a second load.
    assert load_benchmark(path) == (digest, tasks)


def test_load_benchmark_digest_tracks_bytes(tmp_path):
    d1, _ = load_benchmark(write_benchmark(tmp_path / "a.jsonl"))
    d2, _ = load_benchmark(write_benchmark(tmp_path / "b.jsonl", BENCH_ITEMS[:2]))
    assert d1 != d2


# ----------------------------------------------------------------- run_anchor


def test_run_anchor_scores_k_of_n(tmp_path):
    path = write_benchmark(tmp_path / "bench.jsonl")
    answers = {  # two right, one wrong
        BENCH_ITEMS[0]["question"]: "blue",
        BENCH_ITEMS[1]["question"]: "six",  # alias tier
        BENCH_ITEMS[2]["question"]: "Lyon",
    }
    report = run_anchor(
        tmp_path, path, lambda model_dir: ScriptedBackend(answers=answers)
    )
    assert (report.n_tasks, report.n_correct) == (3, 2)
    assert report.accuracy == pytest.approx(2 / 3)
    assert report.per_task == (("gq-1", True), ("anchor-00001", True), ("gq-3", False))
    assert dict(report.judge_tier_counts) == {"exact": 1, "alias": 1, "none": 1}
    assert report.benchmark_digest.startswith("sha256:")
    round_trip = report.to_dict()
    assert round_trip["accuracy"] == report.accuracy
    assert round_trip["per_task"][0] == ["gq-1", True]


def test_run_anchor_closed_book_session(tmp_path):
    """With no env, tools answer with the explicit no-sources observation and
    the harness protocol still completes."""
    assert "no external sources available in anchor mode" in AnchorSession().search("q")
    assert "no external sources available in anchor mode" in AnchorSession().browse("doc-1")

    def policy(prompt: str) -> str:
        if "no external sources available in anchor mode" in prompt:
            return "<answer>blue</answer>"
        return '<tool_call>{"name": "search", "arguments": {"query": ["sky color"]}}</tool_call>'

    path = write_benchmark(tmp_path / "bench.jsonl", [BENCH_ITEMS[0]])
    report = run_anchor(tmp_path, path, lambda model_dir: ScriptedBackend(policy))
    assert report.accuracy == 1.0
    assert report.per_task == (("gq-1", True),)


def test_run_anchor_max_tasks_and_env(tmp_path):
    path = write_benchmark(tmp_path / "bench.jsonl")

    class RecordingEnv:
        def __init__(self):
            self.tasks = []

        def tools_for_task(self, task):
            self.tasks.append(task)
            return AnchorSession()

    env = RecordingEnv()
    answers = {i["question"]: i["answer"] for i in BENCH_ITEMS}
    report = run_anchor(
        tmp_path,
        path,
        lambda model_dir: ScriptedBackend(answers=answers),
        env=env,
        max_tasks=2,
    )
    assert report.n_tasks == 2
    assert [tid for tid, _ in report.per_task] == ["gq-1", "anchor-00001"]
    # The env's tool session was used and tasks carry the anchor marker.
    assert [t.template for t in env.tasks] == ["anchor", "anchor"]


def test_run_anchor_crashed_rollout_scores_incorrect(tmp_path):
    path = write_benchmark(tmp_path / "bench.jsonl", [BENCH_ITEMS[0]])

    class ExplodingEnv:
        def tools_for_task(self, task):
            raise RuntimeError("no session for you")

    report = run_anchor(
        tmp_path, path, lambda model_dir: ScriptedBackend(answers={}), env=ExplodingEnv()
    )
    assert report.per_task == (("gq-1", False),)
    assert report.accuracy == 0.0


# ----------------------------------------------------------------- divergence


def rec(block: int, accuracy: float, ema: float) -> dict:
    return {"block": block, "accuracy": accuracy, "internal_ema_at_run": ema}


def test_divergence_needs_two_records():
    assert divergence(0.9, []) is None
    assert divergence(0.9, [rec(1, 0.5, 0.5)]) is None


def test_divergence_rising_internal_flat_anchor_is_positive():
    history = [rec(1, 0.50, 0.50), rec(2, 0.50, 0.68)]
    assert divergence(0.70, history) == pytest.approx(0.20)


def test_divergence_parallel_rise_is_zero():
    history = [rec(1, 0.50, 0.50), rec(2, 0.70, 0.70)]
    assert divergence(0.70, history) == pytest.approx(0.0)


# --------------------------------------------------------------- service hook


class FakePool:
    epoch = 1
    digest = "sha256:" + "e" * 64

    def sample(self, n, seed):
        return []

    def rotation_due(self, current_block):
        return False

    def rotate(self, current_block):
        return None


class FakeCorpus:
    def get(self, doc_id):
        return None

    def search(self, query, k=5):
        return []


def make_service(tmp_path, answers=None):
    cfg = load_config()
    chain = MockChainClient(identity_hotkey="validator-0")
    state = ValidatorState.load(tmp_path / "state")
    genesis = tmp_path / "genesis"
    genesis.mkdir(exist_ok=True)
    canned = answers if answers is not None else {i["question"]: i["answer"] for i in BENCH_ITEMS}
    deps = Deps(
        chain=chain,
        cfg=cfg,
        state=state,
        corpus=FakeCorpus(),
        env=None,
        backend_factory=lambda model_dir: ScriptedBackend(answers=dict(canned)),
        run_duel=lambda *a, **k: None,
        run_calibration_duel=lambda *a: 0.0,
        run_probes=lambda *a: [],
        generate_tasks=lambda **k: [],
        task_ids_digest=lambda tasks: "d",
        private_pool=FakePool(),
        wallet_hotkey="validator-0",
        clock=chain.current_block,
        materialize=lambda ref, cache_dir: genesis,
        cache_dir=tmp_path / "cache",
    )
    return SimpleNamespace(service=ValidatorService(deps), chain=chain, state=state)


def test_hook_noop_without_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("EPAGO_ANCHOR_BENCHMARK", raising=False)
    h = make_service(tmp_path)
    h.service.tick()
    assert h.state.anchor_history == []
    assert h.state.last_anchor_block == 0


def test_hook_noop_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("EPAGO_ANCHOR_BENCHMARK", str(tmp_path / "nope.jsonl"))
    h = make_service(tmp_path)
    h.service.tick()
    assert h.state.anchor_history == []


def test_hook_records_history_on_interval(tmp_path, monkeypatch):
    bench = write_benchmark(tmp_path / "bench.jsonl")
    monkeypatch.setenv("EPAGO_ANCHOR_BENCHMARK", str(bench))
    monkeypatch.setattr(constants, "ANCHOR_INTERVAL_BLOCKS", 10)
    h = make_service(tmp_path)

    h.service.tick()
    assert len(h.state.anchor_history) == 1
    record = h.state.anchor_history[0]
    assert record["accuracy"] == 1.0
    assert record["n_tasks"] == 3
    assert record["block"] == h.chain.current_block()
    assert record["internal_ema_at_run"] == h.state.king_acc_ema
    assert record["benchmark_digest"].startswith("sha256:")
    assert record["divergence"] is None  # first run: nothing to diverge from
    assert h.state.last_anchor_block == record["block"]
    # The record is published into the audit trail (no holdout: immediate).
    published = list((h.state.state_dir / "audit" / "published").glob("*anchor*"))
    assert len(published) == 1
    assert json.loads(published[0].read_text())["accuracy"] == 1.0

    # Interval not elapsed -> no second run.
    h.service.tick()
    assert len(h.state.anchor_history) == 1

    # Interval elapsed -> second run, divergence now computable (zero drift).
    h.chain.advance(10)
    h.service.tick()
    assert len(h.state.anchor_history) == 2
    assert h.state.anchor_history[1]["divergence"] == pytest.approx(0.0)


def test_hook_alert_on_divergence(tmp_path, monkeypatch):
    bench = write_benchmark(tmp_path / "bench.jsonl")
    monkeypatch.setenv("EPAGO_ANCHOR_BENCHMARK", str(bench))
    monkeypatch.setattr(constants, "ANCHOR_INTERVAL_BLOCKS", 10)
    h = make_service(tmp_path)
    # Seed history: internal EMA has since climbed from 0.2 to 0.5 while the
    # anchor stays at 1.0 -> divergence 0.3 > ANCHOR_DIVERGENCE_ALERT (0.10).
    h.state.anchor_history.append(rec(10, 1.0, 0.2))
    h.state.last_anchor_block = 10
    assert h.state.king_acc_ema == 0.5

    h.service.tick()
    assert len(h.state.anchor_history) == 2
    assert h.state.anchor_history[1]["divergence"] == pytest.approx(0.3)
    assert h.state.last_error is not None
    assert h.state.last_error["code"] == "anchor_divergence"
    assert h.state.last_error["divergence"] == pytest.approx(0.3)
    assert h.state.last_error["threshold"] == constants.ANCHOR_DIVERGENCE_ALERT
    # Alarm, never a halt: the loop keeps ticking and the king is untouched.
    king_digest = h.state.king.ref.digest
    h.service.tick()
    assert h.state.king.ref.digest == king_digest


def test_hook_failure_degrades(tmp_path, monkeypatch):
    bench = tmp_path / "bench.jsonl"
    bench.write_text("this is not json\n")
    monkeypatch.setenv("EPAGO_ANCHOR_BENCHMARK", str(bench))
    monkeypatch.setattr(constants, "ANCHOR_INTERVAL_BLOCKS", 10)
    h = make_service(tmp_path)
    h.service.tick()  # must not raise
    assert h.state.anchor_history == []
    assert h.state.last_error["code"] == "anchor_failed"


def test_state_roundtrips_anchor_history(tmp_path):
    state = ValidatorState.load(tmp_path / "state")
    state.anchor_history = [
        {"block": 100, "accuracy": 0.4, "internal_ema_at_run": 0.5, "divergence": None},
        {"block": 200, "accuracy": 0.4, "internal_ema_at_run": 0.6, "divergence": 0.1},
    ]
    state.last_anchor_block = 200
    state.save()
    loaded = ValidatorState.load(tmp_path / "state")
    assert loaded.anchor_history == state.anchor_history
    assert loaded.last_anchor_block == 200


# ------------------------------------------------------------------ dashboard


def test_dashboard_anchor_section(tmp_path):
    state = {
        "anchor_history": [
            rec(100, 0.40, 0.50) | {"divergence": None, "benchmark_digest": "sha256:" + "a" * 64},
            rec(200, 0.40, 0.75) | {"divergence": 0.25, "benchmark_digest": "sha256:" + "a" * 64},
        ]
    }
    data = export_dashboard(DashboardInputs(state=state, audit_records=[], cfg=load_config()))
    anchor = data["anchor"]
    assert [p["block"] for p in anchor["series"]] == [100, 200]
    assert anchor["series"][1] == {
        "block": 200,
        "king_digest": "",
        "accuracy": 0.40,
        "internal_ema": 0.75,
        "divergence": 0.25,
    }
    assert anchor["latest_divergence"] == 0.25
    assert anchor["alert"] is True
    assert anchor["alert_threshold"] == constants.ANCHOR_DIVERGENCE_ALERT
    assert anchor["benchmark_digest"] == "sha256:" + "a" * 64
    assert anchor["last_run_block"] == 200


def test_dashboard_anchor_empty_safe():
    data = export_dashboard(DashboardInputs(state={}, audit_records=[], cfg=load_config()))
    anchor = data["anchor"]
    assert anchor["series"] == []
    assert anchor["latest_divergence"] is None
    assert anchor["alert"] is False


def test_anchor_releases_its_backend(tmp_path):
    """An anchor run must hand the engine back. On a multi-GPU validator the
    factory serves a leased pool replica, and a lease that is never returned
    takes that card out of the pool permanently."""
    from epago.eval.anchor import run_anchor

    class CountingBackend(ScriptedBackend):
        def __init__(self, answers):
            super().__init__(answers=answers)
            self.closed = 0

        def close(self):
            self.closed += 1

    made: list[CountingBackend] = []

    def factory(model_dir):
        backend = CountingBackend({i["question"]: i["answer"] for i in BENCH_ITEMS})
        made.append(backend)
        return backend

    path = tmp_path / "bench.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in BENCH_ITEMS))
    report = run_anchor(tmp_path / "model", path, factory)
    assert report.n_tasks == len(BENCH_ITEMS)
    assert [b.closed for b in made] == [1]


def test_anchor_releases_its_backend_when_a_rollout_explodes(tmp_path):
    from epago.eval.anchor import run_anchor

    class Exploding:
        def __init__(self):
            self.closed = 0

        def generate(self, prompt, max_tokens, stop):
            raise MemoryError("card fell over")

        def close(self):
            self.closed += 1

    backend = Exploding()
    path = tmp_path / "bench.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in BENCH_ITEMS))
    report = run_anchor(tmp_path / "model", path, lambda d: backend)
    assert report.n_correct == 0  # every rollout scored wrong, none aborted the run
    assert backend.closed == 1


def test_coronation_forces_an_anchor_run_next_tick(tmp_path, monkeypatch):
    """A new king must be measured on the external benchmark immediately —
    per-crown transfer evidence, not a weekly average over two reigns."""
    from tests.test_validator import add_challenger, make_harness, settle

    h = make_harness(tmp_path)
    path = write_benchmark(tmp_path / "bench.jsonl", [BENCH_ITEMS[0]])
    monkeypatch.setenv("EPAGO_ANCHOR_BENCHMARK", str(path))
    # Cadence says: anchor not due for ~a week.
    h.state.last_anchor_block = h.chain.current_block()

    answers = {i["question"]: i["answer"] for i in BENCH_ITEMS}
    h.service.deps.backend_factory = lambda model_dir: ScriptedBackend(answers=answers)
    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)

    assert h.state.king is not None  # coronation happened
    # The crown reset the cadence, so the anchor already ran on the very next
    # tick inside settle() — despite the weekly cadence saying "not due".
    assert h.state.anchor_history, "anchor did not run after coronation"
    rec = h.state.anchor_history[-1]
    assert rec["king_digest"] == h.state.king.ref.digest
    assert rec["n_tasks"] == 1
    assert h.state.last_anchor_block >= h.state.king.crowned_block
