"""Eval subsystem tests: harness loop, judge cascade, probes, paired duel.

Everything runs on ScriptedBackend plus a local fake environment — no torch,
no vllm, no network. The fakes implement exactly the ToolSession /
ResearchEnvironment surface the eval code targets, so these tests do not
import the environment package.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

from epago import constants
from epago.core.types import RolloutResult, Task, TaskOrigin
from epago.eval.backend import ModelBackend, ScriptedBackend, backend_factory
from epago.eval.duel import DuelSpec, run_calibration_duel, run_duel
from epago.eval.harness import harness_digest, run_rollout
from epago.eval.judge import ADVERSARIAL_SUITE, LlmJudge, judge_answer, normalize, sanitize
from epago.eval.probes import degenerate_probe


class FakeSession:
    """Minimal WebToolSession double: deterministic search page and visit."""

    def __init__(self, docs: dict[str, str]) -> None:
        self.docs = docs

    def search_page(self, queries: list[str]) -> str:
        return "\n".join(
            f"results for {q!r}: " + " ".join(sorted(self.docs)) for q in queries
        )

    def visit_pages(self, targets: list[str]) -> str:
        return "\n".join(self.docs.get(t, f"no document {t!r}") for t in targets)


class FakeEnv:
    """Minimal ResearchEnvironment double."""

    def tools_for_task(self, task: Task) -> FakeSession:
        return FakeSession({d: f"contents of {d}" for d in task.evidence_doc_ids})


def make_task(
    task_id: str = "t000",
    question: str = "What is the capital of France?",
    answer: str = "Paris",
    aliases: tuple[str, ...] = (),
) -> Task:
    return Task(
        task_id=task_id,
        question=question,
        answer=answer,
        aliases=aliases,
        evidence_doc_ids=("d1", "d2"),
        masked_doc_ids=(),
        origin=TaskOrigin.GENERATED_PUBLIC,
        template="test",
        hops=1,
    )


# --- harness -----------------------------------------------------------------


def test_harness_parses_actions_and_judges() -> None:
    def policy(prompt: str) -> str:
        if "<tool_response>" not in prompt:
            return '<tool_call>{"name": "search", "arguments": {"query": ["capital of France"]}}</tool_call>'
        if "the d1 document" not in prompt:
            return '<tool_call>{"name": "visit", "arguments": {"url": ["d1"]}}</tool_call>'
        return "<answer>Paris</answer>"

    result = run_rollout(
        ScriptedBackend(policy), make_task(), FakeSession({"d1": "the d1 document"})
    )
    assert result.answer == "Paris"
    assert result.correct is True
    assert result.judge_tier == "exact"
    assert result.turns == 3
    assert result.error is None


def test_harness_respects_turn_cap() -> None:
    backend = ScriptedBackend(
        lambda prompt: '<tool_call>{"name": "search", "arguments": {"query": ["again"]}}</tool_call>'
    )
    result = run_rollout(backend, make_task(), FakeSession({}), max_turns=5)
    assert result.turns == 5
    assert result.answer is None
    assert result.correct is False
    assert result.error == "turn_cap"


def test_harness_malformed_terminates_after_one_nudge() -> None:
    backend = ScriptedBackend(lambda prompt: "I refuse to use tags.")
    result = run_rollout(backend, make_task(), FakeSession({}))
    assert result.answer is None
    assert result.correct is False
    assert result.error == "malformed_action"
    assert result.turns == 2


def test_harness_malformed_then_recovers_via_nudge() -> None:
    def policy(prompt: str) -> str:
        if "no tool call and no answer" in prompt:
            return "<answer>Paris</answer>"
        return "just chatting"

    result = run_rollout(ScriptedBackend(policy), make_task(), FakeSession({}))
    assert result.answer == "Paris"
    assert result.error is None


def test_harness_truncates_answer() -> None:
    long_answer = "x" * (constants.ANSWER_MAX_CHARS * 2)
    backend = ScriptedBackend(lambda prompt: f"<answer>{long_answer}</answer>")
    result = run_rollout(backend, make_task(), FakeSession({}))
    assert result.answer is not None
    assert len(result.answer) == constants.ANSWER_MAX_CHARS


def test_action_parsed_from_the_last_tag_not_the_first() -> None:
    """The defect this fixes: a model narrating the protocol before obeying it
    ("put it in <answer> tags ... <answer>0.8</answer>") had the NARRATION
    graded, so a correct episode scored wrong. Measured on the reference model
    it cost 5 correct answers in 300 episodes."""
    from epago.eval.harness import parse_action

    raw = (
        "The snippet gives 0.8 days. Provide the answer in <answer> tags.\n\n"
        "Thus final answer: <answer>0.8</answer>"
    )
    assert parse_action(raw) == ("answer", "0.8")

    task = make_task(question="How many days?", answer="0.8")
    backend = ScriptedBackend(lambda prompt: raw)
    result = run_rollout(backend, task, FakeSession({}))
    assert result.answer == "0.8"
    assert result.correct is True


def test_action_parsing_unchanged_when_there_is_exactly_one_match() -> None:
    """Behaviour must be bit-identical on the overwhelming common case."""
    from epago.eval.harness import parse_action

    assert parse_action("<answer>Paris</answer>") == ("answer", "Paris")
    assert parse_action("<search>capital of France</search>") == (
        "search",
        "capital of France",
    )
    assert parse_action("<browse>d1</browse>") == ("browse", "d1")
    assert parse_action("<answer>  padded  </answer>") == ("answer", "padded")
    assert parse_action("no action at all") is None


def test_closed_think_block_is_not_an_action_but_an_open_one_still_counts() -> None:
    """Reasoning checkpoints narrate inside <think>. A CLOSED block is the model
    talking to itself; an OPEN one means the stop sequence cut the generation
    mid-thought, and the action it cut on is the only action there is."""
    from epago.eval.harness import parse_action

    assert parse_action(
        "<think>I could <search>wrong guess</search> first</think>"
        "<search>real query</search>"
    ) == ("search", "real query")
    assert parse_action("<think>I should <search>truncated</search>") == (
        "search",
        "truncated",
    )
    assert parse_action("<think>only thinking, no action</think>") is None


def test_harness_digest_is_pinned_sha256() -> None:
    digest = harness_digest()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert harness_digest() == digest


def test_harness_digest_covers_the_tool_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    # A search result becomes prompt bytes, so retrieval semantics must move
    # the digest: two validators searching differently are not comparable.
    before = harness_digest()
    monkeypatch.setattr(constants, "TOOL_SURFACE", constants.TOOL_SURFACE + ("changed",))
    assert harness_digest() != before


# --- judge -------------------------------------------------------------------


def test_judge_exact_tier_normalizes() -> None:
    task = make_task(answer="The Eiffel Tower")
    assert judge_answer(task, "eiffel   tower.") == (True, "exact")
    assert judge_answer(task, "EIFFEL TOWER") == (True, "exact")


def test_judge_alias_tier() -> None:
    task = make_task(answer="United Kingdom", aliases=("UK", "Great Britain"))
    assert judge_answer(task, "the u k") == (False, "none")
    assert judge_answer(task, "UK.") == (True, "alias")
    assert judge_answer(task, "great britain!") == (True, "alias")


def test_judge_numeric_tolerance() -> None:
    task = make_task(answer="3600")
    assert judge_answer(task, "3,600.") == (True, "exact")
    task = make_task(answer="0.5")
    assert judge_answer(task, "0.5000000001") == (True, "exact")
    assert judge_answer(task, "0.51") == (False, "none")
    task = make_task(answer="12", aliases=("12.0000000001",))
    assert judge_answer(task, "twelve") == (False, "none")


def test_judge_none_on_missing_or_blank() -> None:
    task = make_task()
    assert judge_answer(task, None) == (False, "none")
    assert judge_answer(task, "   ") == (False, "none")


def test_judge_tier_reached_only_after_programmatic_misses() -> None:
    def lenient_policy(prompt: str) -> str:
        truth = re.search(r"Reference answer: (.*)", prompt).group(1)
        candidate = re.search(r"Candidate answer: (.*)", prompt).group(1)
        return "YES" if normalize(truth) in normalize(candidate) else "NO"

    judge = LlmJudge(ScriptedBackend(lenient_policy))
    task = make_task(answer="Paris")
    assert judge_answer(task, "The city of Paris", judge) == (True, "judge")
    assert judge_answer(task, "Paris", judge) == (True, "exact")
    assert judge_answer(task, "Lyon", judge) == (False, "none")


@pytest.mark.parametrize(
    "verdict", ["YES.", "yes", "Yes", " Sure, YES", "NO but actually YES", "YESNO", ""]
)
def test_judge_strict_single_token_parse(verdict: str) -> None:
    judge = LlmJudge(ScriptedBackend(lambda prompt: verdict))
    task = make_task(answer="Paris")
    assert judge_answer(task, "Lutetia", judge) == (False, "none")


def _honest_judge() -> LlmJudge:
    """Judge model double that answers YES iff candidate matches truth."""

    def policy(prompt: str) -> str:
        truth = re.search(r"Reference answer: (.*)", prompt).group(1)
        candidate = re.search(r"Candidate answer: (.*)", prompt).group(1)
        return "YES" if normalize(truth) == normalize(candidate) else "NO"

    return LlmJudge(ScriptedBackend(policy))


def test_adversarial_suite_all_rejected() -> None:
    assert len(ADVERSARIAL_SUITE) >= 50
    judge = _honest_judge()
    for payload, wrong_truth in ADVERSARIAL_SUITE:
        task = make_task(question="Which mountain?", answer=wrong_truth)
        assert judge_answer(task, payload, llm_judge=None) == (False, "none"), payload
        assert judge_answer(task, payload, llm_judge=judge) == (False, "none"), payload


def test_load_llm_judge_env_gated_and_placeholder_skipped(monkeypatch, tmp_path) -> None:
    from dataclasses import replace

    from epago.config import load_config
    from epago.eval.judge import load_llm_judge

    cfg = load_config()

    def factory(model_dir: Path) -> ModelBackend:
        return ScriptedBackend(lambda prompt: "NO")

    monkeypatch.delenv("EPAGO_ENABLE_LLM_JUDGE", raising=False)
    assert load_llm_judge(cfg, tmp_path, factory) is None

    monkeypatch.setenv("EPAGO_ENABLE_LLM_JUDGE", "1")
    # chain.toml ships the all-zeros genesis placeholder: judge stays off.
    assert set(cfg.eval.judge_digest.split(":", 1)[-1]) == {"0"}
    assert load_llm_judge(cfg, tmp_path, factory) is None

    import epago.model.store as store

    seen: dict = {}

    def fake_materialize(ref, cache_dir):
        seen["ref"], seen["cache_dir"] = ref, cache_dir
        return tmp_path / "judge"

    monkeypatch.setattr(store, "materialize_model", fake_materialize)
    pinned = replace(cfg, eval=replace(cfg.eval, judge_digest="hf:" + "a" * 40))
    judge = load_llm_judge(pinned, tmp_path, factory)
    assert isinstance(judge, LlmJudge)
    assert seen["ref"].repo == cfg.eval.judge_repo
    assert seen["ref"].digest == "hf:" + "a" * 40
    assert seen["cache_dir"] == tmp_path


class _FakeHttpResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self) -> bytes:
        return self._body


def test_api_judge_backend_sends_greedy_request_and_parses_token(monkeypatch) -> None:
    import json as _json

    from epago.eval.judge import ApiJudgeBackend

    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = _json.loads(req.data)
        captured["auth"] = req.headers.get("Authorization")
        return _FakeHttpResp(_json.dumps({"choices": [{"message": {"content": "YES"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = ApiJudgeBackend("https://api.example.com/v1", "pinned-judge", "sk-test")
    assert backend.generate("Verdict:", max_tokens=4, stop=["\n"]) == "YES"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["temperature"] == 0.0        # greedy
    assert captured["body"]["model"] == "pinned-judge"
    assert captured["auth"] == "Bearer sk-test"
    # LlmJudge over the API backend applies the same strict single-token parse.
    assert LlmJudge(backend).judge("q", "Paris", "Paris") is True


def test_api_judge_fails_closed_on_network_error(monkeypatch) -> None:
    from epago.eval.judge import ApiJudgeBackend

    def boom(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    # A judge outage must read as NO, never crash the duel.
    assert ApiJudgeBackend("https://x/v1", "m", "k").generate("p", 4, ["\n"]) == ""


def test_api_judge_env_gating_and_precedence(monkeypatch, tmp_path) -> None:
    from epago.config import load_config
    from epago.eval.judge import load_api_judge, load_llm_judge

    for var in ("EPAGO_JUDGE_API_BASE", "EPAGO_JUDGE_API_MODEL", "EPAGO_JUDGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert load_api_judge() is None                        # nothing set

    monkeypatch.setenv("EPAGO_JUDGE_API_BASE", "https://api.example.com/v1")
    assert load_api_judge() is None                        # base but no model/key
    monkeypatch.setenv("EPAGO_JUDGE_API_MODEL", "pinned-judge")
    monkeypatch.setenv("EPAGO_JUDGE_API_KEY", "sk-test")
    assert isinstance(load_api_judge(), LlmJudge)

    # With the API configured, load_llm_judge takes the API path even though the
    # shipped judge_digest is still the genesis placeholder (no local model).
    monkeypatch.setenv("EPAGO_ENABLE_LLM_JUDGE", "1")
    cfg = load_config()
    assert set(cfg.eval.judge_digest.split(":", 1)[-1]) == {"0"}
    judge = load_llm_judge(cfg, tmp_path, lambda d: ScriptedBackend(lambda p: "NO"))
    assert isinstance(judge, LlmJudge)


def test_sanitize_strips_injection_surface() -> None:
    for payload, _ in ADVERSARIAL_SUITE:
        cleaned = sanitize(payload)
        assert len(cleaned) <= constants.ANSWER_MAX_CHARS
        assert "<" not in cleaned and ">" not in cleaned
        assert "\n" not in cleaned
        assert not any(unicodedata.category(ch) in ("Cc", "Cf") for ch in cleaned)


# --- probes ------------------------------------------------------------------


def _probe_result(task_id: str, answer: str | None) -> RolloutResult:
    return RolloutResult(task_id, answer, False, 1, 0.0, "none", None)


def test_probe_task_set_is_easy_balanced_and_deterministic() -> None:
    """The format gate must test formatting, not research luck: it takes the
    lowest-hop tasks of EVERY template rather than the first N by task_id
    (a content hash, i.e. a lottery over the full difficulty mixture)."""
    from epago.eval.probes import probe_task_set

    def task(tid: str, template: str, hops: int) -> Task:
        return Task(
            task_id=tid,
            question=f"q {tid}",
            answer="a",
            aliases=(),
            evidence_doc_ids=("d1",),
            masked_doc_ids=(),
            origin=TaskOrigin.GENERATED_PUBLIC,
            template=template,
            hops=hops,
        )

    pool = [task(f"tk-a{i}", "described_finding", 1) for i in range(10)]
    pool += [task(f"tk-b{i}", "cross_doc_join", 2) for i in range(10)]
    pool += [task(f"tk-c{i}", "monster", 9) for i in range(10)]

    chosen = probe_task_set(pool, n_tasks=6)
    assert len(chosen) == 6
    assert sorted(t.template for t in chosen) == [
        "cross_doc_join", "cross_doc_join",
        "described_finding", "described_finding",
        "monster", "monster",
    ]
    assert [t.task_id for t in chosen] == sorted(t.task_id for t in chosen)
    assert probe_task_set(list(reversed(pool)), n_tasks=6) == chosen

    # Fewer templates than slots: it fills up rather than under-running.
    assert len(probe_task_set(pool[:10], n_tasks=6)) == 6
    assert probe_task_set([], n_tasks=6) == []


def _gen_result(task_id: str, *, turns: int, malformed: int, error=None, answer="a"):
    return RolloutResult(task_id, answer, False, turns, 0.0, "none", error, malformed)


def test_format_gate_counts_generations_not_solved_tasks() -> None:
    """The gate must be failable only by BAD OUTPUT, never by a hard task: a
    model that formats every turn perfectly and still runs out of turns passes,
    while a model that mostly emits junk fails."""
    from epago.eval.probes import _format_failures

    perfect_but_stuck = [
        _gen_result(f"t{i}", turns=40, malformed=0, error="turn_cap", answer=None)
        for i in range(20)
    ]
    assert _format_failures(perfect_but_stuck) == []

    junk = [_gen_result(f"t{i}", turns=4, malformed=3) for i in range(20)]
    failures = _format_failures(junk)
    assert failures and failures[0].code == "format_probe"
    assert "25%" in failures[0].detail

    crashed = [_gen_result("t0", turns=0, malformed=0, error="crash: ValueError")]
    assert _format_failures(crashed)[0].code == "probe_crash"


def test_format_gate_threshold_is_the_measured_compliance_floor() -> None:
    """Exactly at the floor passes; one malformed generation more does not."""
    from epago import constants
    from epago.eval.probes import _format_failures

    total = 1000
    good = round(constants.FORMAT_PROBE_MIN_COMPLIANCE * total)
    at_floor = [_gen_result("t0", turns=total, malformed=total - good)]
    assert _format_failures(at_floor) == []
    below = [_gen_result("t0", turns=total, malformed=total - good + 1)]
    assert _format_failures(below)[0].code == "format_probe"


def test_degenerate_probe_fires_on_constant_answers() -> None:
    results = [_probe_result(f"p{i}", "same thing") for i in range(5)]
    failures = degenerate_probe(results)
    assert [f.code for f in failures] == ["constant_answers"]


def test_degenerate_probe_fires_on_empty_answers() -> None:
    results = [_probe_result(f"p{i}", None) for i in range(3)]
    assert [f.code for f in degenerate_probe(results)] == ["empty_answers"]
    assert [f.code for f in degenerate_probe([])] == ["no_results"]


def test_degenerate_probe_passes_varied_answers() -> None:
    results = [_probe_result(f"p{i}", f"answer {i}") for i in range(5)]
    assert degenerate_probe(results) == []


# --- duel --------------------------------------------------------------------

KING_DIR = Path("/models/king")
CHALL_DIR = Path("/models/challenger")


def _duel_tasks(prefix: str, n: int) -> list[Task]:
    return [
        make_task(
            task_id=f"{prefix}{i:03d}",
            question=f"What is {prefix}{i:03d}?",
            answer=f"ans-{prefix}{i:03d}",
        )
        for i in range(n)
    ]


def _knower(known_ids: set[str]) -> ScriptedBackend:
    """Model that answers correctly on known task ids, wrong elsewhere."""

    def policy(prompt: str) -> str:
        task_id = re.search(r"What is (\S+)\?", prompt).group(1)
        if task_id in known_ids:
            return f"<answer>ans-{task_id}</answer>"
        return f"<answer>wrong-{task_id}</answer>"

    return ScriptedBackend(policy)


def _factory(known_by_dir: dict[Path, set[str]]):
    def factory(model_dir: Path) -> ModelBackend:
        return _knower(known_by_dir[model_dir])

    return factory


def _spec(pub: list[Task], priv: list[Task]) -> DuelSpec:
    return DuelSpec(
        king_dir=KING_DIR,
        challenger_dir=CHALL_DIR,
        public_tasks=pub,
        private_tasks=priv,
        block_hash_at_reveal="0xabc123",
        author_hotkey="5" + "F" * 47,
        king_acc_ema=0.97,          # permissive headroom delta
        noise_floor=0.0005,
    )


def test_run_duel_challenger_wins() -> None:
    pub, priv = _duel_tasks("pub", 100), _duel_tasks("prv", 100)
    king_known = {t.task_id for t in pub[:50]} | {t.task_id for t in priv[:50]}
    chall_known = {t.task_id for t in pub[:60]} | {t.task_id for t in priv[:60]}
    factory = _factory({KING_DIR: king_known, CHALL_DIR: chall_known})

    outcome = run_duel(_spec(pub, priv), FakeEnv(), factory)
    assert outcome.public.king_acc == pytest.approx(0.5)
    assert outcome.public.challenger_acc == pytest.approx(0.6)
    assert outcome.public.mu_hat == pytest.approx(0.1)
    assert outcome.private.mu_hat == pytest.approx(0.1)
    assert outcome.lcb_pub > outcome.delta
    assert outcome.accepted is True
    assert re.fullmatch(r"[0-9a-f]{16}", outcome.boot_seed_hex)
    assert re.fullmatch(r"[0-9a-f]{16}", outcome.public_seed_hex)
    # Per-task results cover the public half in scored (sorted) order and
    # reproduce the half's diffs exactly.
    assert [tid for tid, _ in outcome.public_task_results] == sorted(t.task_id for t in pub)
    assert tuple(d for _, d in outcome.public_task_results) == outcome.public.diffs


def test_run_duel_king_vs_king_not_accepted() -> None:
    pub, priv = _duel_tasks("pub", 50), _duel_tasks("prv", 50)
    known = {t.task_id for t in pub[:25]} | {t.task_id for t in priv[:25]}
    factory = _factory({KING_DIR: known, CHALL_DIR: known})

    outcome = run_duel(_spec(pub, priv), FakeEnv(), factory)
    assert outcome.public.mu_hat == pytest.approx(0.0)
    assert outcome.private.mu_hat == pytest.approx(0.0)
    assert outcome.accepted is False


def test_run_calibration_duel_measures_zero_noise() -> None:
    tasks = _duel_tasks("cal", 20)
    known = {t.task_id for t in tasks[:10]}
    factory = _factory({KING_DIR: known})
    assert run_calibration_duel(KING_DIR, tasks, FakeEnv(), factory) == 0.0
    with pytest.raises(ValueError):
        run_calibration_duel(KING_DIR, [], FakeEnv(), factory)


def test_run_duel_is_deterministic() -> None:
    pub, priv = _duel_tasks("pub", 40), _duel_tasks("prv", 40)
    king_known = {t.task_id for t in pub[:20]} | {t.task_id for t in priv[:20]}
    chall_known = {t.task_id for t in pub[:26]} | {t.task_id for t in priv[:26]}
    factory = _factory({KING_DIR: king_known, CHALL_DIR: chall_known})

    first = run_duel(_spec(pub, priv), FakeEnv(), factory)
    second = run_duel(_spec(pub, priv), FakeEnv(), factory)
    assert first == second
    # The new per-task and judge-stat fields are part of the determinism
    # contract too — they must be populated and identical across runs.
    assert first.public_task_results and first.public_task_results == second.public_task_results
    assert first.judge_tier_counts and first.judge_tier_counts == second.judge_tier_counts


def test_run_duel_counts_judge_tiers_across_all_rollouts() -> None:
    pub, priv = _duel_tasks("pub", 6), _duel_tasks("prv", 4)
    all_ids = {t.task_id for t in pub} | {t.task_id for t in priv}
    king_known = all_ids  # king answers everything exactly
    chall_known: set[str] = set()  # challenger answers everything wrong -> "none"
    factory = _factory({KING_DIR: king_known, CHALL_DIR: chall_known})

    outcome = run_duel(_spec(pub, priv), FakeEnv(), factory)
    counts = dict(outcome.judge_tier_counts)
    total_rollouts = 2 * (len(pub) + len(priv))
    assert sum(counts.values()) == total_rollouts
    assert counts["exact"] == len(pub) + len(priv)  # king's rollouts
    assert counts["none"] == len(pub) + len(priv)   # challenger's
    assert outcome.judge_invocation_rate == 0.0
    assert outcome.judge_tier_counts == tuple(sorted(counts.items()))


def test_run_duel_survives_crashing_rollout() -> None:
    pub, priv = _duel_tasks("pub", 10), _duel_tasks("prv", 10)

    class CrashyEnv(FakeEnv):
        def tools_for_task(self, task: Task) -> FakeSession:
            if task.task_id == "pub003":
                raise RuntimeError("corpus shard offline")
            return super().tools_for_task(task)

    all_ids = {t.task_id for t in pub} | {t.task_id for t in priv}
    factory = _factory({KING_DIR: all_ids, CHALL_DIR: all_ids})
    outcome = run_duel(_spec(pub, priv), CrashyEnv(), factory)
    assert outcome.public.king_acc == pytest.approx(0.9)
    assert outcome.public.challenger_acc == pytest.approx(0.9)


# --- backend -----------------------------------------------------------------


def test_scripted_backend_answers_map_and_stops() -> None:
    backend = ScriptedBackend(answers={"What is q1?": "a1"})
    prompt = "system\n\nQuestion: What is q1?\n"
    out = backend.generate(prompt, 64, ["</answer>"])
    assert out == "<answer>a1</answer>"
    chatty = ScriptedBackend(lambda p: "<search>x</search> trailing junk")
    assert chatty.generate("p", 64, ["</search>", "</answer>"]) == "<search>x</search>"
    with pytest.raises(ValueError):
        ScriptedBackend()


class FakeTokenizer:
    """Tokenizer double: records how the chat template was invoked."""

    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return [7, len(messages[0]["content"]), 9]


class FakeEngine:
    """vLLM LLM double: records the prompts it was handed."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.seen: list[list] = []

    def get_tokenizer(self):
        return self._tokenizer

    def generate(self, prompts, params, use_tqdm=False):
        self.seen.append(list(prompts))
        out = []
        for _ in prompts:
            out.append(
                type("Req", (), {"outputs": [type("Out", (), {"text": "<answer>x</answer>"})()]})()
            )
        return out


def make_fake_vllm(monkeypatch, tokenizer):
    """Install a fake ``vllm`` module and return the engine it will construct."""
    import sys
    import types

    from epago.eval.backend import VllmBackend

    engine = FakeEngine(tokenizer)
    module = types.ModuleType("vllm")
    module.LLM = lambda **kwargs: engine
    module.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(VllmBackend, "_engines", {}, raising=False)
    return engine


def test_chat_prompt_uses_the_models_own_template_as_one_user_message() -> None:
    """Model-agnostic by construction: the harness prompt goes in as a single
    user message and the tokenizer's own template decides the framing, so no
    vendor's turn markers are written down anywhere in the eval code."""
    from epago.eval.backend import chat_prompt

    tokenizer = FakeTokenizer()
    rendered = chat_prompt(tokenizer, "harness prompt")

    assert rendered == {"prompt_token_ids": [7, len("harness prompt"), 9]}
    assert tokenizer.calls == [
        {
            "messages": [{"role": "user", "content": "harness prompt"}],
            "tokenize": True,
            "add_generation_prompt": True,
        }
    ]
    # Pure function of (template, prompt): no clock, no rng, no environment.
    assert chat_prompt(FakeTokenizer(), "harness prompt") == rendered


def test_chat_prompt_accepts_every_transformers_return_shape() -> None:
    """``apply_chat_template(tokenize=True)`` has returned a flat list, a
    BatchEncoding-like mapping, and a batch-of-one nesting across transformers
    versions; a validator cannot pin one version network-wide."""
    from epago.eval.backend import chat_prompt

    class Mapping(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    class Nested(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return [[1, 2, 3]]

    class Strings(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return "<|im_start|>user"

    assert chat_prompt(Mapping(), "p") == {"prompt_token_ids": [1, 2, 3]}
    assert chat_prompt(Nested(), "p") == {"prompt_token_ids": [1, 2, 3]}
    # Not tokens: fall back to the raw prompt rather than feed the engine junk.
    assert chat_prompt(Strings(), "p") is None


def test_chat_prompt_falls_back_to_the_raw_prompt_without_a_template() -> None:
    """A challenger that ships a plain base checkpoint must not be penalized:
    no template means the prompt is passed through exactly as before."""
    from epago.eval.backend import chat_prompt

    class BaseTokenizer(FakeTokenizer):
        chat_template = None

    assert chat_prompt(BaseTokenizer(), "harness prompt") is None
    assert chat_prompt(None, "harness prompt") is None
    assert chat_prompt(object(), "harness prompt") is None


def test_chat_prompt_survives_a_template_that_raises() -> None:
    from epago.eval.backend import chat_prompt

    class BrokenTokenizer(FakeTokenizer):
        def apply_chat_template(self, *a, **k):
            raise ValueError("template needs a system message")

    assert chat_prompt(BrokenTokenizer(), "harness prompt") is None


def test_vllm_backend_templates_single_and_batched_generation(monkeypatch) -> None:
    """Both entry points must render identically — a probe rollout (generate)
    and a duel batch step (generate_many) have to put the same bytes in front
    of the model, or in-process and remote eval disagree."""
    from epago.eval.backend import VllmBackend

    tokenizer = FakeTokenizer()
    engine = make_fake_vllm(monkeypatch, tokenizer)
    backend = VllmBackend(Path("/models/king"))

    backend.generate("prompt one", 16, ["</answer>"])
    backend.generate_many(["prompt one", "prompt two!"], 16, ["</answer>"])

    assert engine.seen[0] == [{"prompt_token_ids": [7, 10, 9]}]
    assert engine.seen[1] == [
        {"prompt_token_ids": [7, 10, 9]},
        {"prompt_token_ids": [7, 11, 9]},
    ]
    assert engine.seen[1][0] == engine.seen[0][0]


def test_vllm_backend_passes_raw_prompts_for_a_base_checkpoint(monkeypatch) -> None:
    from epago.eval.backend import VllmBackend

    class BaseTokenizer(FakeTokenizer):
        chat_template = None

    engine = make_fake_vllm(monkeypatch, BaseTokenizer())
    backend = VllmBackend(Path("/models/base"))

    backend.generate("prompt one", 16, ["</answer>"])
    backend.generate_many(["prompt two"], 16, ["</answer>"])

    assert engine.seen == [["prompt one"], ["prompt two"]]


def test_prompt_templating_happens_only_in_the_backend() -> None:
    """The harness must hand the backend the RAW harness prompt, so the one
    place a chat template is applied is the backend that owns the tokenizer.
    Remote eval (epago.eval.server) runs the same in-process backend behind
    an HTTP hop, so this is what keeps a duel scored on a GPU box and a duel
    scored in-process identical; templating here as well would double-wrap
    the prompt on exactly one of the two paths."""
    from epago.eval.harness import SYSTEM_PROMPT

    seen: list[str] = []

    def policy(prompt: str) -> str:
        seen.append(prompt)
        return "<answer>Paris</answer>"

    run_rollout(ScriptedBackend(policy), make_task(), FakeSession({}))

    assert seen and seen[0].startswith(SYSTEM_PROMPT)
    assert "Question: What is the capital of France?" in seen[0]


def test_backend_factory_guards_vllm() -> None:
    import importlib.util

    # The vllm guard only fires when the eval extra is absent; on a GPU/eval box
    # where vllm is installed, construction proceeds past the import guard.
    if importlib.util.find_spec("vllm") is None:
        with pytest.raises(RuntimeError, match="vllm"):
            backend_factory(Path("/models/king"), kind="vllm")
    with pytest.raises(ValueError):
        backend_factory(Path("/models/king"), kind="mystery")


# ---------------------------------------------------------------- throughput


class RecordingBackend:
    """Scripted backend that records call shapes and close() for batch tests."""

    def __init__(self, answers: dict[str, str]):
        self._answers = answers
        self.batch_sizes: list[int] = []
        self.closed = 0

    def _answer_for(self, prompt: str) -> str:
        for needle, out in self._answers.items():
            if needle in prompt:
                return out
        return "<answer>unknown</answer>"

    def generate(self, prompt, max_tokens, stop):
        return self._answer_for(prompt)

    def generate_many(self, prompts, max_tokens, stop):
        self.batch_sizes.append(len(prompts))
        return [self._answer_for(p) for p in prompts]

    def close(self):
        self.closed += 1


def _throughput_tasks(n=12):
    return [
        Task(
            task_id=f"bt-{i:03d}",
            question=f"question {i}",
            answer=f"ans-{i}",
            aliases=(),
            evidence_doc_ids=("d",),
            masked_doc_ids=(),
            origin=TaskOrigin.GENERATED_PUBLIC,
            template="t",
        )
        for i in range(n)
    ]


def test_batched_rollouts_equal_sequential():
    from epago.eval.harness import run_rollout, run_rollouts_batched

    tasks = _throughput_tasks()
    answers = {f"question {i}": f"<answer>ans-{i}</answer>" for i in range(0, 12, 2)}
    answers.update({f"question {i}": "<answer>wrong</answer>" for i in range(1, 12, 2)})

    seq_backend = RecordingBackend(answers)
    sequential = [run_rollout(seq_backend, t, FakeSession({})) for t in tasks]

    batch_backend = RecordingBackend(answers)
    batched = run_rollouts_batched(
        batch_backend, tasks, lambda t: FakeSession({}), concurrency=5
    )

    assert [(r.task_id, r.answer, r.correct, r.turns, r.error) for r in batched] == [
        (r.task_id, r.answer, r.correct, r.turns, r.error) for r in sequential
    ]
    assert max(batch_backend.batch_sizes) == 5  # engine actually saw batches
    assert sum(batch_backend.batch_sizes) == 12  # one turn per episode here


def test_batched_concurrency_one_matches_default():
    from epago.eval.harness import run_rollouts_batched

    tasks = _throughput_tasks(6)
    answers = {f"question {i}": f"<answer>ans-{i}</answer>" for i in range(6)}
    a = run_rollouts_batched(RecordingBackend(answers), tasks, lambda t: FakeSession({}), concurrency=1)
    b = run_rollouts_batched(RecordingBackend(answers), tasks, lambda t: FakeSession({}), concurrency=32)
    assert [(r.task_id, r.correct) for r in a] == [(r.task_id, r.correct) for r in b]


def test_low_vram_mode_releases_king_and_matches_default(monkeypatch, tmp_path):
    (tmp_path / "king").mkdir()
    (tmp_path / "chall").mkdir()
    pub = _throughput_tasks(8)
    answers = {f"question {i}": f"<answer>ans-{i}</answer>" for i in range(8)}
    spec = DuelSpec(
        king_dir=tmp_path / "king",
        challenger_dir=tmp_path / "chall",
        public_tasks=pub,
        private_tasks=_throughput_tasks(4),
        block_hash_at_reveal="0xabc",
        author_hotkey="hk",
        king_acc_ema=0.5,
        noise_floor=0.001,
    )
    made: dict[str, RecordingBackend] = {}

    def factory(model_dir):
        b = RecordingBackend(answers if "chall" in str(model_dir) else {})
        made[str(model_dir)] = b
        return b

    monkeypatch.delenv("EPAGO_EVAL_LOW_VRAM", raising=False)
    normal = run_duel(spec, FakeEnv(), factory)
    king_backend = made[str(tmp_path / "king")]
    assert king_backend.closed == 0

    made.clear()
    monkeypatch.setenv("EPAGO_EVAL_LOW_VRAM", "1")
    low = run_duel(spec, FakeEnv(), factory)
    assert made[str(tmp_path / "king")].closed == 1  # released before challenger
    assert made[str(tmp_path / "chall")].closed == 0  # caller's job, unchanged

    assert low.lcb_pub == normal.lcb_pub
    assert low.public_task_results == normal.public_task_results
    assert low.accepted == normal.accepted


# --- competition rounds -------------------------------------------------------


def _round_spec(pub, priv, entrant_dirs, king_acc_ema=0.97):
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
        king_acc_ema=king_acc_ema,
        noise_floor=0.0005,
    )


def test_round_duel_scores_every_entrant_on_one_exam() -> None:
    from epago.eval.duel import run_round_duel

    pub, priv = _duel_tasks("pub", 60), _duel_tasks("prv", 60)
    king_known = {t.task_id for t in pub[:30]} | {t.task_id for t in priv[:30]}
    strong = {t.task_id for t in pub[:50]} | {t.task_id for t in priv[:50]}
    weak = {t.task_id for t in pub[:31]} | {t.task_id for t in priv[:31]}
    a_dir, b_dir = Path("/models/a"), Path("/models/b")
    factory = _factory({KING_DIR: king_known, a_dir: strong, b_dir: weak})

    results = run_round_duel(_round_spec(pub, priv, [a_dir, b_dir]), FakeEnv(), factory)

    assert [r.entrant.challenger_dir for r in results] == [a_dir, b_dir]
    # Both were scored against the SAME king answers, so their public halves
    # have identical length and the comparison between them is exam-free.
    assert results[0].outcome.public.n_tasks == results[1].outcome.public.n_tasks
    assert results[0].outcome.public.king_acc == results[1].outcome.public.king_acc
    # The stronger entrant has the higher LCB.
    assert results[0].outcome.lcb_pub > results[1].outcome.lcb_pub
    assert results[0].outcome.accepted


def test_round_duel_sweeps_the_king_exactly_once() -> None:
    """N entrants must cost N+1 sweeps, not 2N."""
    from epago.eval.duel import run_round_duel

    pub, priv = _duel_tasks("pub", 20), _duel_tasks("prv", 20)
    known = {t.task_id for t in pub[:10]} | {t.task_id for t in priv[:10]}
    dirs = [Path(f"/models/{c}") for c in "abc"]
    loaded: list[Path] = []
    base = _factory({KING_DIR: known, **{d: known for d in dirs}})

    def counting_factory(model_dir: Path):
        loaded.append(model_dir)
        return base(model_dir)

    run_round_duel(_round_spec(pub, priv, dirs), FakeEnv(), counting_factory)
    assert loaded.count(KING_DIR) == 1
    assert len(loaded) == 4


def test_round_duel_is_deterministic() -> None:
    from epago.eval.duel import run_round_duel

    pub, priv = _duel_tasks("pub", 40), _duel_tasks("prv", 40)
    known = {t.task_id for t in pub[:20]} | {t.task_id for t in priv[:20]}
    better = {t.task_id for t in pub[:28]} | {t.task_id for t in priv[:28]}
    d = Path("/models/a")
    factory = _factory({KING_DIR: known, d: better})
    spec = _round_spec(pub, priv, [d])

    first = run_round_duel(spec, FakeEnv(), factory)
    second = run_round_duel(spec, FakeEnv(), factory)
    assert first[0].outcome == second[0].outcome


def test_round_duel_survives_a_broken_entrant() -> None:
    """One checkpoint that cannot even load must not deny the rest their duel."""
    from epago.eval.duel import run_round_duel

    pub, priv = _duel_tasks("pub", 20), _duel_tasks("prv", 20)
    known = {t.task_id for t in pub[:10]} | {t.task_id for t in priv[:10]}
    better = {t.task_id for t in pub[:18]} | {t.task_id for t in priv[:18]}
    good, bad = Path("/models/good"), Path("/models/bad")
    base = _factory({KING_DIR: known, good: better})

    def factory(model_dir: Path):
        if model_dir == bad:
            raise RuntimeError("cannot load weights")
        return base(model_dir)

    results = run_round_duel(_round_spec(pub, priv, [bad, good]), FakeEnv(), factory)

    assert len(results) == 2
    assert results[0].outcome.lcb_pub == -1.0          # forfeit, priced as a loss
    assert not results[0].outcome.accepted
    assert results[1].outcome.accepted                  # the good one still ran


# --- harness v4: native protocol ----------------------------------------------


def test_parse_native_turn_variants() -> None:
    from epago.eval.harness import parse_native_turn

    # A plain tool call, with thinking noise around it.
    turn = parse_native_turn(
        '<think>I should search.</think>\n'
        '<tool_call>{"name": "search", "arguments": {"query": ["a", "b"]}}</tool_call>'
    )
    assert turn == ("call", ("search", {"query": ["a", "b"]}))
    # </answer> is a stop sequence, so the closing tag may never be generated.
    assert parse_native_turn("<think>done</think><answer>42%") == ("answer", "42%")
    # An answer beats a tool call — the model has finished.
    both = (
        '<tool_call>{"name": "search", "arguments": {"query": ["x"]}}</tool_call>'
        "<answer>final</answer>"
    )
    assert parse_native_turn(both) == ("answer", "final")
    # Broken JSON in the call is malformed, not a crash.
    assert parse_native_turn("<tool_call>{not json}</tool_call>") is None
    # A tool call inside a CLOSED think block is the model talking to itself.
    assert (
        parse_native_turn(
            '<think><tool_call>{"name": "search", "arguments": {}}</tool_call></think>'
        )
        is None
    )


def test_web_tool_session_serp_and_visit(tmp_path) -> None:
    from epago.environment.corpus import Document, SqliteCorpus
    from epago.environment.services import WebToolSession

    corpus = SqliteCorpus.create(tmp_path / "web.db")
    corpus.add_documents(
        [
            Document(
                doc_id="ep-w1",
                url="https://example.org/w1",
                title="A Study of Kefir Microbiology",
                text="A Study of Kefir Microbiology\n\nThe accuracy was 91.2%.",
            ),
            Document(
                doc_id="ep-w2",
                url="https://example.org/w2",
                title="Kefir Sensory Properties in Goats",
                text="Kefir Sensory Properties in Goats\n\nThe yield reached 40%.",
            ),
        ]
    )
    session = WebToolSession(corpus, frozenset())
    page = session.search_page(["kefir"])
    # Web-shaped: numbered results carrying title, url and id.
    assert 'Search results for "kefir":' in page
    assert "url: https://example.org/w1 (id: ep-w1)" in page
    # visit resolves the URL the model copied off the results page.
    out = session.visit_pages(["https://example.org/w1"])
    assert "The accuracy was 91.2%." in out
    # ...and the raw id, and a pasted "url (id: x)" string.
    assert "yield reached 40%" in session.visit_pages(["ep-w2"])
    assert "accuracy" in session.visit_pages(["https://example.org/w1 (id: ep-w1)"])
    # An unknown target is a recoverable observation, not an exception.
    assert "Could not open" in session.visit_pages(["https://elsewhere.org/x"])


def test_episode_budget_closes_tools_then_accepts_answer() -> None:
    from epago.eval.harness import Episode, _BUDGET_MSG

    task = make_task()
    ep = Episode(task, FakeSession({}), max_turns=10)
    # Blow the char budget artificially, then start a turn.
    ep.messages.append({"role": "user", "content": "x" * (constants.TRANSCRIPT_MAX_CHARS + 1)})
    assert ep.begin_turn()
    assert ep.budget_closed
    assert ep.messages[-1]["content"] == _BUDGET_MSG
    # A tool call after closure gets the tools-closed observation...
    ep.advance('<tool_call>{"name": "search", "arguments": {"query": ["q"]}}</tool_call>')
    assert "tools are closed" in ep.messages[-1]["content"]
    # ...and an answer still ends the episode normally.
    assert ep.begin_turn()
    ep.advance("<answer>Paris</answer>")
    assert ep.done and ep.answer == "Paris"
    # But a model that never answers runs out two turns after closure.
    ep2 = Episode(task, FakeSession({}), max_turns=10)
    ep2.messages.append({"role": "user", "content": "x" * (constants.TRANSCRIPT_MAX_CHARS + 1)})
    while ep2.begin_turn():
        ep2.advance('<tool_call>{"name": "search", "arguments": {"query": ["q"]}}</tool_call>')
    assert ep2.error == "context_budget"


def test_batched_rollouts_survive_a_poisoned_batch() -> None:
    """A backend that rejects a whole batch (vLLM does this when ONE prompt is
    over-long) must cost only the offending episode, never the sweep."""
    from dataclasses import replace

    from epago.eval.harness import run_rollouts_batched

    class BatchPoisonBackend:
        def generate(self, prompt, max_tokens, stop):
            text = prompt if isinstance(prompt, str) else " ".join(
                str(m.get("content", "")) for m in prompt
            )
            if "poison" in text:
                raise ValueError("prompt too long")
            return "<answer>Paris</answer>"

        def generate_many(self, prompts, max_tokens, stop):
            return [self.generate(p, max_tokens, stop) for p in prompts]

        def close(self):
            return None

    tasks = [
        make_task(),
        replace(make_task(), task_id="tk-poison", question="poison question"),
        replace(make_task(), task_id="tk-c"),
    ]
    results = run_rollouts_batched(
        BatchPoisonBackend(), tasks, lambda t: FakeSession({}), concurrency=3
    )
    assert [r.answer for r in results] == ["Paris", None, "Paris"]
    assert results[1].error is not None and "generate_failed" in results[1].error


def test_deterministic_budgets_bind_before_the_clock() -> None:
    """v4.1: the wall clock is a hang net (>= 3600 s); the budgets that end an
    episode are the turn cap and the transcript char budget — pure functions
    of the transcript, identical at any load on any hardware."""
    from epago.eval.harness import _BUDGET_MSG, Episode

    assert constants.ROLLOUT_TIMEOUT_S >= 3600
    # A think-heavy episode is closed by the transcript budget, gracefully.
    ep = Episode(make_task(), FakeSession({}), max_turns=100)
    big_turn = "<think>" + "x" * 50_000 + "</think>" + \
        '<tool_call>{"name": "search", "arguments": {"query": ["q"]}}</tool_call>'
    while ep.begin_turn() and not ep.budget_closed:
        ep.advance(big_turn)
    assert ep.budget_closed and ep.messages[-1]["content"] == _BUDGET_MSG
    assert ep.begin_turn()
    ep.advance("<answer>Paris</answer>")
    assert ep.done and ep.answer == "Paris" and ep.error is None


def test_repeated_search_returns_a_pointer_not_a_second_page():
    """A repeat costs a full page of context and can return nothing new.

    Search is deterministic, so the second run of a query the model has
    already seen is guaranteed to show it what it already has. Measured on the
    two-hop tasks, a quarter of all searches were exact repeats while tool
    output filled 85% of the context.
    """
    from epago.environment.services import WebToolSession

    from epago.environment.corpus import Document, SearchHit

    class _Corpus:
        """Two documents, deterministic ranking — enough to see a repeat."""

        def search(self, query, k=10, mask_doc_ids=frozenset()):
            # A full page, because the saving being tested is a page of
            # context: a one-result stub would make the pointer look
            # expensive when in practice a page is ~1,900 chars.
            tag = "g" if "gamma" in query.lower() else "d"
            return [
                SearchHit(
                    doc_id=f"{tag}{i}",
                    title=f"A Study Of Something Numbered {i}",
                    snippet="lorem ipsum dolor sit amet " * 4,
                    score=1.0 - i / 100,
                )
                for i in range(k)
            ]

        def get(self, doc_id, mask_doc_ids=frozenset()):
            return Document(
                doc_id=doc_id,
                url=f"http://x/{doc_id}",
                title=doc_id,
                text="t",
                category="c",
            )

    session = WebToolSession(_Corpus(), frozenset())

    first = session.search_page(["alpha beta"])
    assert "d0" in first

    # Same terms, different order and quoting: the result set is identical, so
    # this is the same query.
    again = session.search_page(['"beta"  alpha'])
    assert "You already ran this search" in again
    assert "d0" not in again
    assert session.repeated_searches == 1
    assert len(again) < len(first)

    # A genuinely different query is still answered in full.
    other = session.search_page(["gamma"])
    assert "You already ran this search" not in other


def test_turn_cap_asks_for_an_answer_instead_of_cutting_the_episode_off():
    """Both budgets end an episode the same way: by asking it to commit.

    The character budget always closed the tools and demanded an answer. The
    turn cap did not -- it ended the episode outright. That difference decided
    real results: once search pages got cheaper the binding limit moved from
    characters to turns, and 51% of episodes ended at the cap while holding
    71% of their evidence, never asked for a verdict.
    """
    from epago.core.types import Task, TaskOrigin
    from epago.eval.harness import _TURN_RESERVE, Episode
    from epago.environment.services import ToolSession

    class _Corpus:
        def search(self, query, k=10, mask_doc_ids=frozenset()):
            return []

        def get(self, doc_id, mask_doc_ids=frozenset()):
            return None

    task = Task(
        task_id="t1",
        question="q",
        answer="a",
        aliases=(),
        evidence_doc_ids=("d1",),
        masked_doc_ids=(),
        origin=TaskOrigin.GENERATED_PUBLIC,
        template="x",
        hops=1,
    )
    ep = Episode(task, ToolSession(_Corpus(), frozenset()), max_turns=6)

    # Burn turns up to the reserve; nothing should be closed yet.
    for _ in range(6 - _TURN_RESERVE):
        assert ep.begin_turn()
    assert not ep.budget_closed

    # Crossing into the reserve closes the tools and asks for an answer.
    assert ep.begin_turn()
    assert ep.budget_closed
    assert ep.messages[-1]["role"] == "user"

    # The episode still ends, rather than looping forever.
    for _ in range(10):
        if not ep.begin_turn():
            break
    assert ep.done
