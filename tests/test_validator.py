"""Validator subsystem tests: MockChainClient + fake eval/taskgen deps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from epago import constants
from epago.chain.client import MockChainClient, NeuronView
from epago.config import load_config
from epago.core.reveal import (
    build_king_pointer,
    build_reveal,
    build_verdict,
    parse_king_pointer,
    parse_verdict,
)
from epago.core.types import (
    AuditRecord,
    DuelHalf,
    DuelOutcome,
    KingPointer,
    ModelRef,
    SubmissionStatus,
    Verdict,
    VerdictDecision,
)
from epago.model.validation import IntakeFailure
from epago.validator.audit import audit16, record_digest
from epago.validator.intake import (
    COOLDOWN_BLOCKS,
    COOLDOWN_MAX_BLOCKS,
    apply_cooldown,
    cooldown_duration,
    cooldown_triggered,
    cooldown_until,
    queue_pressure_scale,
)
from epago.validator.service import Deps, ValidatorService
from epago.validator.state import ValidatorState, difficulty_from_dict, difficulty_to_dict

VALIDATOR_HK = "validator-0"
ROUND_AUTHORITY_HK = "round-authority"

LOCK_CONFIG = {
    "architectures": ["EpagoForCausalLM"],
    "vocab_size": 151936,
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "model_type": "epago",
    "tie_word_embeddings": True,
    "rope_theta": 5000000.0,
    "rope_scaling": None,
    "max_position_embeddings": 262144,
}


def hf_digest(char: str) -> str:
    return "hf:" + char * 40


def make_model_dir(tmp_path, name: str, weights: bytes, config: dict | None = None):
    d = tmp_path / "models" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config if config is not None else LOCK_CONFIG))
    (d / "model.safetensors").write_bytes(weights)
    return d


def make_half(mu: float, king_acc: float, chall_acc: float) -> DuelHalf:
    return DuelHalf(
        n_tasks=200, diffs=(1, 0, -1, 1), mu_hat=mu, king_acc=king_acc, challenger_acc=chall_acc
    )


def make_outcome(
    lcb=0.08, delta=0.03, accepted=True, mu_priv=0.06, judge_tier_counts=()
) -> DuelOutcome:
    return DuelOutcome(
        public=make_half(0.09, 0.55, 0.64),
        private=make_half(mu_priv, 0.55, 0.61),
        lcb_pub=lcb,
        delta=delta,
        accepted=accepted,
        boot_seed_hex="ab" * 8,
        public_seed_hex="cd" * 8,
        judge_tier_counts=tuple(judge_tier_counts),
    )


class FakePrivatePool:
    def __init__(self):
        self.epoch = 1
        self.digest = "sha256:" + "e" * 64
        self.rotations: list[int] = []

    def sample(self, n: int, seed: int) -> list:
        return [f"priv-{seed % 997}-{i}" for i in range(4)]

    def rotation_due(self, current_block: int) -> bool:
        return False

    def commitment(self) -> str:
        return f"ep1|{self.epoch}|{self.digest.removeprefix('sha256:')[:16]}"

    def rotate(self, current_block: int):
        self.rotations.append(current_block)
        self.epoch += 1
        self.digest = "sha256:" + f"{self.epoch:064d}"
        return self.commitment()


class FakeCorpus:
    """Minimal corpus double."""

    def __init__(self, docs: dict[str, str] | None = None):
        self.docs = docs if docs is not None else dict(DEFAULT_DOCS)

    def get(self, doc_id: str):
        text = self.docs.get(doc_id)
        return None if text is None else SimpleNamespace(doc_id=doc_id, text=text)

    def search(self, query: str, k: int = 5) -> list:
        return []


DEFAULT_DOCS = {
    "doc-hab": "The orbital habitat was renamed Aurora Station after refurbishment.",
    "doc-relay": "Engineers commissioned the deepwater relay called Meridian Anchor.",
    "doc-vault": "The archive vessel is known to its crew as Basalt Vault.",
}


def make_harness(
    tmp_path,
    outcome: DuelOutcome | None = None,
    probe_failures=None,
    sign=None,
    chain_toml: str | None = None,
    model_config: dict | None = None,
):
    """``chain_toml``/``model_config`` select the generation under test: the
    default is the pinned 4B contract, and tests/test_generation_30b.py drives
    the same harness with the 30B MoE contract and its real config."""
    cfg = load_config(chain_toml) if chain_toml else load_config()
    # Nothing is evaluated until the round authority opens a competition, so
    # every harness names one; `open_round` is what actually triggers a duel.
    cfg = replace(cfg, chain=replace(cfg.chain, round_authority_hotkey=ROUND_AUTHORITY_HK))
    chain = MockChainClient(identity_hotkey=VALIDATOR_HK)
    chain.add_neuron(NeuronView(uid=0, hotkey="burn-hk", coldkey="burn-ck", stake=0.0, validator_permit=False))
    chain.add_neuron(NeuronView(uid=1, hotkey=VALIDATOR_HK, coldkey="ck-val-00", stake=100.0, validator_permit=True))

    state = ValidatorState.load(tmp_path / "state")
    genesis_dir = make_model_dir(tmp_path, "genesis", b"K" * 1000, config=model_config)
    dirs = {cfg.seed.seed_digest: genesis_dir}

    holder = {
        "outcome": outcome or make_outcome(),
        "probe_failures": list(probe_failures or []),
        "duel_specs": [],
    }

    def materialize(ref: ModelRef, cache_dir):
        return dirs[ref.digest]

    def run_duel(spec, env, backend_factory, llm_judge):
        holder["duel_specs"].append(spec)
        # An "outcomes" queue serves per-call results (e.g. a winning round
        # duel followed by a failing confirmation); the single "outcome" is
        # the steady-state answer once the queue drains.
        queue = holder.get("outcomes")
        out = queue.pop(0) if queue else holder["outcome"]
        if isinstance(out, Exception):
            raise out
        # Echo per-task results the way the eval subsystem would, so the
        # difficulty plumbing can be exercised. String placeholder tasks
        # (no .task_id) leave the holder outcome untouched.
        public = tuple(
            (t.task_id, (1, -1, 0, 1)[i % 4])
            for i, t in enumerate(spec.public_tasks)
            if hasattr(t, "task_id")
        )
        if public:
            out = replace(out, public_task_results=public)
        return out

    def run_probes(challenger_dir, king_dir):
        return list(holder["probe_failures"])

    def generate_tasks(*, seed, release, corpus, n, king_probe):
        return [f"task-{seed % 997}-{i}" for i in range(4)]

    def task_ids_digest(tasks):
        return hashlib.sha256("|".join(str(t) for t in tasks).encode()).hexdigest()

    deps = Deps(
        chain=chain,
        cfg=cfg,
        state=state,
        corpus=FakeCorpus(),
        env=None,
        backend_factory=None,
        run_duel=run_duel,
        run_calibration_duel=lambda *a: 0.005,
        run_probes=run_probes,
        generate_tasks=generate_tasks,
        task_ids_digest=task_ids_digest,
        private_pool=FakePrivatePool(),
        wallet_hotkey=VALIDATOR_HK,
        clock=chain.current_block,
        materialize=materialize,
        cache_dir=tmp_path / "cache",
        sign=sign,
    )
    service = ValidatorService(deps)
    return SimpleNamespace(
        service=service, chain=chain, state=state, cfg=cfg, dirs=dirs, tmp=tmp_path,
        holder=holder, model_config=model_config,
    )


def add_challenger(
    h,
    name: str,
    hotkey: str,
    coldkey: str,
    uid: int,
    digest_char: str,
    # Distinct per challenger by default: identical weights are now rejected as
    # a copy, which is the point — a shared default would be an unrealistic
    # fixture that hid the check.
    weights: bytes | None = None,
    king_digest: str | None = None,
    register: bool = True,
    reveal: bool = True,
):
    digest = hf_digest(digest_char)
    weights = weights if weights is not None else (digest_char.encode() * 1000)[:1000]
    repo = f"{name}/{h.cfg.chain.name}-{hotkey[:8].lower()}-v1"
    if digest not in h.dirs:
        h.dirs[digest] = make_model_dir(
            h.tmp, f"chal-{name}-{digest_char}", weights, config=h.model_config
        )
    if register:
        h.chain.add_neuron(
            NeuronView(uid=uid, hotkey=hotkey, coldkey=coldkey, stake=1.0, validator_permit=False)
        )
    payload = build_reveal(king_digest or h.cfg.seed.seed_digest, ModelRef(repo, digest))
    if reveal:
        h.chain.inject_reveal(hotkey, payload)
    return digest, repo, payload


def published_verdicts(chain: MockChainClient) -> list[str]:
    return [rp.payload for rp in chain.read_revealed_payloads() if rp.payload.startswith("ev3|")]


def open_round(h, round_no: int | None = None) -> int:
    """Publish an `er1` round start from the authority and let it reveal.

    Blocks are advanced past ROUND_MIN_INTERVAL_BLOCKS first, because the chain
    client drops a round published too soon after the previous one.
    """
    from epago import constants as _c
    from epago.core.reveal import build_round_start

    h.round_no = round_no if round_no is not None else getattr(h, "round_no", 0) + 1
    h.chain.advance(_c.ROUND_MIN_INTERVAL_BLOCKS + 1)
    h.chain.inject_reveal(ROUND_AUTHORITY_HK, build_round_start(h.round_no))
    h.chain.advance(1)
    return h.round_no


def settle(h, ticks: int = 2, round_no: int | None = None):
    """Open a round, tick, let timelock reveals surface, then tick again.

    Verdicts reveal ~5 blocks after publication, exactly as on the live chain.
    """
    from epago import constants as _c

    h.service.tick()          # intake first, so the field exists before the round
    open_round(h, round_no)
    for _ in range(ticks):
        h.service.tick()
        h.chain.advance(_c.VERDICT_REVEAL_BLOCKS + 1)
    h.chain.advance(_c.WEIGHT_INTERVAL_BLOCKS)  # cross the weight-set interval
    h.service.tick()



# --------------------------------------------------------------------------- tests


def _generator_release_harness(tmp_path, **kw):
    """A harness pinned to a generator release, whatever the contract ships.

    Tests about generator behaviour must not inherit the live contract's
    release: when the shipped contract moved to a sealed pool these tests
    started exercising the wrong path and failing for a reason that had nothing
    to do with what they assert. Pinning the release here keeps each test about
    the behaviour it names.
    """
    import dataclasses

    h = make_harness(tmp_path, **kw)
    h.service.cfg = dataclasses.replace(
        h.cfg, eval=dataclasses.replace(h.cfg.eval, taskgen_release="SCI4")
    )
    return h


def test_genuine_improver_accept_coronation_phase_a_burn(tmp_path):
    h = make_harness(tmp_path)
    digest, repo, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    settle(h)

    # ev1 verdict committed with an ACCEPT decision.
    evs = published_verdicts(h.chain)
    assert len(evs) == 1
    verdict = parse_verdict(evs[0], VALIDATOR_HK, h.chain.block)
    assert verdict.decision is VerdictDecision.ACCEPT
    assert verdict.challenger_digest == digest

    # Bootstrap mode (1 evaluator < bootstrap_min_evaluators): coronation fires.
    assert h.state.king is not None
    assert h.state.king.ref.digest == digest
    assert h.state.king.ref.repo == repo
    assert h.state.king.author_hotkey == "hk-alice"
    assert h.state.statuses[digest] == SubmissionStatus.ACCEPTED.value

    # King mirrored into validator-controlled storage at coronation.
    mirror = tmp_path / "state" / "king_mirror" / digest.replace(":", "_")
    assert (mirror / "model.safetensors").read_bytes() == b"a" * 1000

    # Phase A (counters below activation): the full emission burns (uid 0).
    assert h.chain.last_weights == {0: 1.0}

    # Durable state round-trips.
    reloaded = ValidatorState.load(tmp_path / "state")
    assert reloaded.king is not None and reloaded.king.ref.digest == digest
    assert reloaded.clean_duels == 1


def test_phase_b_weights_go_to_new_king(tmp_path):
    h = make_harness(tmp_path)
    # Force the deterministic phase gate (counters + age).
    h.state.clean_duels = constants.PHASE_B_MIN_CLEAN_DUELS + 10
    h.state.organic_dethrones = constants.PHASE_B_MIN_DETHRONES
    h.state.genesis_block = -constants.PHASE_B_MIN_BLOCKS
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    settle(h)

    assert h.state.king.ref.digest == digest
    weights = h.chain.last_weights
    king_uid = 2
    assert weights[king_uid] == max(weights.values())
    assert weights[king_uid] > 0.5
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_whole_field_duels_and_one_winner_is_crowned(tmp_path):
    """A round evaluates every entrant, not just the first in the queue.

    Under the old first-come flow only the head of the queue was dueled and the
    rest dropped as `stale_parent` the moment it won — so a better checkpoint
    submitted an hour later never got measured at all. A round crowns the best
    of the field, and the entrants that lost are near-misses, not stale.
    """
    h = make_harness(tmp_path)
    d_alice, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    d_bob, _, _ = add_challenger(h, "bob", "hk-bob", "ck-bob-001", uid=3, digest_char="b")
    d_carol, _, _ = add_challenger(h, "carol", "hk-carol", "ck-carol-1", uid=4, digest_char="c")

    settle(h)

    # Every entrant was dueled on the same exam, then the provisional winner
    # was confirmed once on a fresh one.
    assert len(h.holder["duel_specs"]) == 4
    exams = {tuple(spec.public_tasks) for spec in h.holder["duel_specs"][:3]}
    assert len(exams) == 1, "the field must answer one identical exam"
    assert tuple(h.holder["duel_specs"][3].public_tasks) not in exams, (
        "the confirmation duel must be a fresh sample, or it replays the luck"
        " it exists to rule out"
    )

    # The fake outcome is the same for all three, so the tie breaks on digest:
    # carol has the highest. Exactly one is crowned.
    crowned = {d for d in (d_alice, d_bob, d_carol)
               if h.state.statuses[d] == SubmissionStatus.ACCEPTED.value}
    assert crowned == {d_carol}
    assert h.state.king.ref.digest == d_carol
    # The others beat the king too — runners-up, not losers, so no cooldown.
    assert h.state.statuses[d_alice] == SubmissionStatus.NEAR_MISS.value
    assert h.state.statuses[d_bob] == SubmissionStatus.NEAR_MISS.value
    assert h.state.cooldowns == {}
    assert h.state.queue == []

    # New reveals against the old king digest still drop at intake.
    h.chain.advance()
    add_challenger(
        h, "dave", "hk-dave", "ck-dave-01", uid=5, digest_char="d",
        king_digest=h.cfg.seed.seed_digest,
    )
    settle(h)
    assert h.state.statuses[hf_digest("d")] == SubmissionStatus.STALE_PARENT.value
    assert h.state.queue == []


def test_duplicate_digest_first_reveal_owns_it(tmp_path):
    h = make_harness(tmp_path)
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    h.chain.advance()  # bob reveals the same digest one block later
    add_challenger(h, "bob", "hk-bob", "ck-bob-001", uid=3, digest_char="a")

    settle(h)

    assert h.state.seen_digests[digest] == "hk-alice"
    rejections = [
        e for e in h.state.intake_log if e["hotkey"] == "hk-bob" and e["code"] == "duplicate_digest"
    ]
    assert rejections, "second reveal of the same digest must reject as duplicate_digest"
    # Alice's submission proceeded to a verdict under her ownership.
    assert h.state.king.author_hotkey == "hk-alice"


def test_failed_probes_memory_and_cooldown(tmp_path):
    h = make_harness(
        tmp_path, probe_failures=[IntakeFailure("format_probe", "17/20 invalid outputs")]
    )
    digest, _, payload = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    settle(h)

    assert h.state.statuses[digest] == SubmissionStatus.FAILED_PROBES.value
    assert digest in h.state.failure_memory
    # Probe failure starts a hotkey cooldown (there is no bond to burn).
    cd = h.state.cooldowns["hk-alice"]
    assert cd["strikes"] == 1
    assert cd["until_block"] > h.chain.block - COOLDOWN_BLOCKS  # live cooldown
    assert published_verdicts(h.chain) == []  # no duel, no verdict
    assert h.state.king.ref.digest == h.cfg.seed.seed_digest

    # Failure memory prevents requeue on a fresh reveal of the same digest.
    h.chain.advance()
    h.chain.inject_reveal("hk-alice", payload)
    settle(h)
    assert h.state.queue == []
    assert not h.holder["duel_specs"]

    # Cooldown policy: everything that is not an acceptance or a near-miss
    # pays. The old rule only fired below BOND_BURN_LCB_THRESHOLD, which left
    # the -0.05..0 band free — exactly where a noise-perturbed copy of the king
    # scores — so farming the arena pool and the false-acceptance tail was free.
    assert cooldown_triggered(SubmissionStatus.QUEUED, probes_failed=True)
    assert cooldown_triggered(SubmissionStatus.FAILED_PROBES)
    assert cooldown_triggered(SubmissionStatus.DUEL_LOST)
    assert cooldown_triggered(SubmissionStatus.FAILED_INTAKE)
    assert not cooldown_triggered(SubmissionStatus.NEAR_MISS)
    assert not cooldown_triggered(SubmissionStatus.ACCEPTED)


def test_queue_pressure_scale_circuit_breaker():
    assert queue_pressure_scale(0, 6.0) == 1.0
    # 5 queued * 6h = 36h estimated latency: still within the breaker.
    assert queue_pressure_scale(5, 6.0) == 1.0
    # 12 queued -> 78h estimate, 42h over the 36h breaker -> two doublings.
    assert queue_pressure_scale(12, 6.0) == 4.0


def test_cooldown_duration_escalates_and_caps():
    assert cooldown_duration(1) == COOLDOWN_BLOCKS
    assert cooldown_duration(2) == min(2 * COOLDOWN_BLOCKS, COOLDOWN_MAX_BLOCKS)
    assert cooldown_duration(50) == COOLDOWN_MAX_BLOCKS          # strike cap
    # A queue the box can actually clear must not be punished: at the measured
    # duel cost a depth of 12 projects well inside QUEUE_BREAKER_HOURS, so the
    # breaker stays out of the way.
    assert cooldown_duration(1, queue_depth=12) == COOLDOWN_BLOCKS
    # It still fires on a backlog that would genuinely blow the SLA.
    assert cooldown_duration(1, queue_depth=48) == min(          # breaker scale
        COOLDOWN_BLOCKS * 4, COOLDOWN_MAX_BLOCKS
    )
    assert cooldown_duration(50, queue_depth=1000) == COOLDOWN_MAX_BLOCKS


def test_cooldown_strikes_escalate_and_reset(tmp_path):
    st = ValidatorState.load(tmp_path / "state")
    first = apply_cooldown(st, "ck-x", block=1_000)
    assert first["strikes"] == 1
    assert first["until_block"] == 1_000 + COOLDOWN_BLOCKS
    # Second decisive loss inside the memory window doubles the cooldown.
    second = apply_cooldown(st, "ck-x", block=2_000)
    assert second["strikes"] == 2
    assert second["until_block"] == 2_000 + 2 * COOLDOWN_BLOCKS
    assert cooldown_until(st, "ck-x", 2_001) == second["until_block"]
    assert cooldown_until(st, "ck-x", second["until_block"]) is None  # expired
    # A strike far outside the memory window starts over at 1.
    third = apply_cooldown(st, "ck-x", block=2_000 + 200_000)
    assert third["strikes"] == 1
    # Cooldowns survive a state reload.
    st.save()
    reloaded = ValidatorState.load(tmp_path / "state")
    assert reloaded.cooldowns["ck-x"]["strikes"] == 1


def test_near_miss_earns_a_retry_not_an_arena_seat(tmp_path):
    h = make_harness(tmp_path, outcome=make_outcome(lcb=0.01, delta=0.03, accepted=False, mu_priv=0.005))
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    settle(h)

    assert h.state.statuses[digest] == SubmissionStatus.NEAR_MISS.value
    assert digest not in h.state.failure_memory
    # A near-miss no longer takes an arena seat: the arena is the roster of
    # former kings, which is what makes "burn until something has been
    # crowned" true at genesis. What a near-miss earns is the right to try
    # again on fresh tasks.
    assert h.state.arena == []
    assert digest in h.state.near_misses
    assert h.state.cooldowns == {}  # good-faith attempt: no cooldown

    verdict = parse_verdict(published_verdicts(h.chain)[0], VALIDATOR_HK, h.chain.block)
    assert verdict.decision is VerdictDecision.REJECT
    assert h.state.king.ref.digest == h.cfg.seed.seed_digest  # no coronation


def test_self_dethrone_inherits_reign_other_hotkey_resets(tmp_path):
    st = ValidatorState.load(tmp_path / "state")
    st.set_king(
        ModelRef("a/EPAGO-DR-4B-x", hf_digest("a")), "hk-a", crowned_block=100,
        coronation_lcb=0.05, coronation_delta=0.02,
    )
    assert st.king.reign_started_block == 100
    dethrones = st.organic_dethrones

    # Same hotkey re-crowns itself: reign clock inherited (anti-salami-slicing).
    st.set_king(
        ModelRef("a/EPAGO-DR-4B-y", hf_digest("b")), "hk-a", crowned_block=500,
        coronation_lcb=0.04,
    )
    assert st.king.reign_started_block == 100
    assert st.king.crowned_block == 500
    assert st.organic_dethrones == dethrones

    # Different hotkey: reign resets and the organic-dethrone counter advances.
    st.set_king(
        ModelRef("b/EPAGO-DR-4B-z", hf_digest("c")), "hk-b", crowned_block=900,
        coronation_lcb=0.06,
    )
    assert st.king.reign_started_block == 900
    assert st.organic_dethrones == dethrones + 1


def test_quorum_three_evaluators_requires_theta_stake(tmp_path):
    h = make_harness(tmp_path)
    h.chain.add_neuron(NeuronView(uid=5, hotkey="val-1", coldkey="ck-v1", stake=100.0, validator_permit=True))
    h.chain.add_neuron(NeuronView(uid=6, hotkey="val-2", coldkey="ck-v2", stake=100.0, validator_permit=True))
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    # The other evaluators are active (they posted verdicts) but reject.
    reject = build_verdict(
        Verdict(digest, VerdictDecision.REJECT, -1000, 0, 25000, 1, 1, "ab" * 8)
    )
    h.chain.publish_commitment_as("val-1", reject)
    h.chain.publish_commitment_as("val-2", reject)

    settle(h)

    # 3 active evaluators -> no bootstrap mode; our ACCEPT alone is 100/300
    # stake < theta=0.51, so no coronation yet.
    assert h.state.king.ref.digest == h.cfg.seed.seed_digest
    assert digest in h.state.candidates

    # The others re-run the duel and flip to ACCEPT (latest verdict wins).
    h.chain.advance()
    accept = build_verdict(
        Verdict(digest, VerdictDecision.ACCEPT, 80000, 60000, 25000, 1, 1, "cd" * 8)
    )
    h.chain.publish_commitment_as("val-1", accept)
    h.chain.publish_commitment_as("val-2", accept)

    settle(h)

    assert h.state.king.ref.digest == digest
    assert digest not in h.state.candidates


def test_audit16_matches_record_and_sla_report(tmp_path):
    h = make_harness(tmp_path)
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    h.chain.advance(3)  # some reveal->verdict latency

    settle(h)

    verdict = parse_verdict(published_verdicts(h.chain)[0], VALIDATOR_HK, h.chain.block)
    lines = (tmp_path / "state" / "audit" / "audit.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = AuditRecord(**json.loads(lines[0]))

    # The 16-hex digest committed on-chain matches the recomputed digest of
    # the replayable audit record.
    assert verdict.audit_digest == audit16(record)
    assert record.challenger_digest == digest
    assert record.accepted is True
    assert record.private_pool_epoch == 1
    assert record.harness_digest.startswith("sha256:")

    report = h.service.sla_report()
    assert report["n"] == 1
    # Reveal-to-verdict now includes the wait for the next competition, so the
    # measured latency is dominated by the round cadence rather than by how
    # fast the box evaluates. See docs/DESIGN.md §9 on what the SLA means under
    # rounds.
    assert report["p50_blocks"] == report["p95_blocks"]
    assert report["p50_blocks"] >= constants.ROUND_MIN_INTERVAL_BLOCKS
    assert report["sla_target_blocks"] > 0

    # Public tasks staged for delayed publication, not yet released. A sealed
    # release stages a second artifact beside the round record -- the round's
    # questions in full -- so name the one under test rather than counting the
    # directory, which otherwise fails whenever the contract changes release.
    delayed_dir = tmp_path / "state" / "audit" / "delayed"
    assert len(list(delayed_dir.glob("*_round[0-9]*.json"))) == 1
    assert list((tmp_path / "state" / "audit" / "published").glob("*.json")) == []


def test_transient_duel_error_requeues_front(tmp_path):
    h = make_harness(tmp_path)
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    h.holder["outcome"] = RuntimeError("backend OOM")

    settle(h)

    # Machine-readable error state; submission back at the queue head.
    assert h.state.last_error["code"] == "round_duel_failed"
    assert h.state.queue and h.state.queue[0].digest == digest
    assert digest not in h.state.failure_memory

    # Next tick, with the backend healthy again, the duel completes.
    h.holder["outcome"] = make_outcome()
    h.chain.advance()
    settle(h)
    assert h.state.king.ref.digest == digest


# ------------------------------------------------------------------ cooldowns


def test_a_hotkey_is_spent_by_its_one_submission(tmp_path):
    """One submission per hotkey, permanently.

    A second reveal from the same hotkey is refused whatever happened to the
    first, so an attempt costs a registration burn rather than being free. That
    price is the point: without it the cheapest strategy is to submit many
    mediocre checkpoints and let the duels find one that got lucky on its
    holdout, and every one of those costs validators a full rollout sweep.

    Note this makes the hotkey cooldown unreachable for the same hotkey -- it
    is spent before a cooldown could bite. The cooldown ledger is left in place
    because it still records strikes for the dashboard and for any future rule
    that keys on an author across hotkeys.
    """
    h = make_harness(
        tmp_path, outcome=make_outcome(lcb=-0.20, delta=0.03, accepted=False, mu_priv=-0.05)
    )
    d_a, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    settle(h)
    assert h.state.statuses[d_a] == SubmissionStatus.DUEL_LOST.value
    assert h.state.spent_hotkeys["hk-alice"] == d_a

    # A different model from the SAME hotkey is refused at intake, with a
    # machine-readable code naming the submission that spent it.
    h.chain.advance()
    d_b, _, _ = add_challenger(
        h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="b", register=False
    )
    h.service.tick()

    spent = [
        e for e in h.state.intake_log
        if e["hotkey"] == "hk-alice" and e["code"] == "hotkey_spent"
    ]
    assert spent, "second submission from a spent hotkey must be refused"
    assert d_a[:16] in spent[0]["detail"]
    assert h.state.queue == []
    assert d_b not in h.state.statuses


def test_audit_signature_signs_canonical_unsigned_digest(tmp_path):
    def fake_sign(data: bytes) -> str:
        return "sig:" + hashlib.sha256(data).hexdigest()

    h = make_harness(tmp_path, sign=fake_sign)
    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    settle(h)

    verdict = parse_verdict(published_verdicts(h.chain)[0], VALIDATOR_HK, h.chain.block)
    (line,) = (tmp_path / "state" / "audit" / "audit.jsonl").read_text().strip().splitlines()
    raw = json.loads(line)

    # The stored record carries a signature...
    assert raw["validator_signature"].startswith("sig:")
    # ...which does NOT perturb the digest the on-chain ev1 verdict commits to.
    record = AuditRecord(**raw)
    assert verdict.audit_digest == audit16(record)

    # Canonical-unsigned recompute from the raw JSON alone (auditor's view):
    unsigned = dict(raw)
    unsigned["validator_signature"] = ""
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest[:16] == verdict.audit_digest
    assert digest == record_digest(record)
    # The signature is over exactly that digest's bytes.
    assert raw["validator_signature"] == fake_sign(digest.encode())


# --------------------------------------------------------- difficulty feed


def test_difficulty_controller_fed_from_duel_and_persisted(tmp_path):
    # Pinned to a generator release: this test injects its own generate_tasks,
    # which a sealed-pool release never calls.
    h = _generator_release_harness(
        tmp_path, outcome=make_outcome(judge_tier_counts=(("exact", 95), ("judge", 5)))
    )

    def template_tasks(*, seed, release, corpus, n, king_probe):
        return [
            SimpleNamespace(
                task_id=f"pt-{i}", template="tpl-alpha" if i % 2 == 0 else "tpl-beta"
            )
            for i in range(4)
        ]

    h.service.deps.generate_tasks = template_tasks
    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)

    # Echoed public diffs: pt-0 -> +1, pt-1 -> -1, pt-2 -> 0, pt-3 -> +1.
    ctrl = h.service.difficulty
    assert ctrl._disc["tpl-alpha"].value == pytest.approx(0.5)   # |+1|, |0|
    assert ctrl._disc["tpl-beta"].value == pytest.approx(1.0)    # |-1|, |+1|
    assert ctrl._solve["tpl-alpha"].value == pytest.approx(0.55)  # public king_acc
    assert ctrl._judge["tpl-beta"].value == pytest.approx(0.05)   # 5/100 judged

    # The audit record consumed the real judge invocation rate and tiers.
    (line,) = (tmp_path / "state" / "audit" / "audit.jsonl").read_text().strip().splitlines()
    raw = json.loads(line)
    assert raw["judge_invocation_rate"] == pytest.approx(0.05)
    assert raw["extra"]["judge_tier_counts"] == [["exact", 95], ["judge", 5]]
    assert len(raw["extra"]["public_diffs"]) == 4

    # Controller state survives a restart via the persisted snapshot.
    st = ValidatorState.load(tmp_path / "state")
    assert st.difficulty["disc"]["tpl-beta"][0] == pytest.approx(1.0)
    from epago.taskgen.difficulty import DifficultyController

    restored = difficulty_from_dict(DifficultyController(), st.difficulty)
    assert difficulty_to_dict(restored) == st.difficulty


def test_near_miss_retry_requires_fresh_reveal_and_is_bounded(tmp_path):
    """A near-miss earns exactly NEAR_MISS_RETRIES re-duels, and only via a NEW
    reveal (newer block => fresh seed); the original reveal never replays."""
    h = make_harness(tmp_path, outcome=make_outcome(lcb=0.01, delta=0.03, accepted=False))
    hotkey = "hk-alice"
    digest, repo, _payload = add_challenger(h, "alice", hotkey, "ck-alice-01", uid=2, digest_char="a")

    settle(h)
    assert h.state.statuses[digest] == "near_miss"
    duels_after_first = len(h.holder["duel_specs"])
    assert duels_after_first == 1

    # The original reveal is still the hotkey's latest — it must NOT re-enqueue.
    settle(h)
    assert len(h.holder["duel_specs"]) == duels_after_first

    # A fresh reveal of the same digest re-enters the queue exactly once.
    from epago.core.reveal import build_reveal
    from epago.core.types import ModelRef

    # Real reveals land at a FUTURE block (timelock), always ahead of the scan.
    payload = build_reveal(h.state.king.ref.digest, ModelRef(repo=repo, digest=digest))
    h.chain.publish_reveal_as(hotkey, payload, blocks_until_reveal=5)
    h.chain.advance(6)
    settle(h)
    assert len(h.holder["duel_specs"]) == duels_after_first + 1
    assert h.state.statuses[digest] == "near_miss"  # near-missed again
    assert h.state.near_misses[digest]["retries"] == 1

    # Third attempt: retry budget exhausted — ignored despite a fresh reveal.
    h.chain.publish_reveal_as(hotkey, payload, blocks_until_reveal=5)
    h.chain.advance(6)
    settle(h)
    assert len(h.holder["duel_specs"]) == duels_after_first + 1


# --- king pointer: what makes a validator startable ---------------------------


def _authority_cfg(h, authority: str):
    """Point the config at a coronation authority (chain.toml ships none)."""
    chain_section = replace(h.cfg.chain, king_authority_hotkey=authority)
    h.cfg = replace(h.cfg, chain=chain_section)
    h.service.cfg = h.cfg
    h.service.deps.cfg = h.cfg
    return h


def test_fresh_validator_adopts_the_king_from_the_chain_pointer(tmp_path):
    """A box with an empty state directory must be able to join.

    Without a pointer the only king it could name was the genesis seed, so every
    live challenge failed the stale_parent gate, no duel ever ran, no verdict was
    ever posted, and the validator could never catch up.
    """
    h = make_harness(tmp_path)
    _authority_cfg(h, "authority-hk")

    king_ref = ModelRef(repo="team/EPAGO-DR-4B-live", digest=hf_digest("f"))
    h.dirs[king_ref.digest] = make_model_dir(tmp_path, "live-king", b"L" * 1000)
    pointer = build_king_pointer(
        KingPointer(
            repo=king_ref.repo,
            digest=king_ref.digest,
            author_hotkey="hk-champion",
            crowned_block=500,
            reign_started_block=200,
            coronation_lcb_e6=40_000,
            coronation_delta_e6=25_000,
        )
    )
    h.chain.inject_reveal("authority-hk", pointer)
    h.chain.advance(constants.VERDICT_REVEAL_BLOCKS)

    h.service.tick()

    king = h.state.king
    assert king.ref == king_ref
    assert king.author_hotkey == "hk-champion"
    assert king.crowned_block == 500
    # The reign clock is adopted verbatim; recomputing it locally would hand the
    # incumbent a fresh decay curve on every restart.
    assert king.reign_started_block == 200
    assert king.coronation_delta == pytest.approx(0.025)


def test_pointer_from_a_non_authority_hotkey_is_ignored(tmp_path):
    """The pointer names who takes 80% of emission."""
    h = make_harness(tmp_path)
    _authority_cfg(h, "authority-hk")

    usurper = build_king_pointer(
        KingPointer(
            repo="evil/EPAGO-DR-4B-x",
            digest=hf_digest("9"),
            author_hotkey="hk-attacker",
            crowned_block=500,
            reign_started_block=500,
            coronation_lcb_e6=99_000,
            coronation_delta_e6=1,
        )
    )
    h.chain.inject_reveal("some-other-hk", usurper)
    h.chain.advance(constants.VERDICT_REVEAL_BLOCKS)

    h.service.tick()
    assert h.state.king.ref.digest == h.cfg.seed.seed_digest


def test_stale_pointer_replay_does_not_roll_the_throne_back(tmp_path):
    h = make_harness(tmp_path)
    _authority_cfg(h, "authority-hk")
    new_digest, old_digest = hf_digest("f"), hf_digest("9")
    for d, name in ((new_digest, "new"), (old_digest, "old")):
        h.dirs[d] = make_model_dir(tmp_path, f"king-{name}", b"K" * 1000)

    def pointer(digest, crowned):
        return build_king_pointer(
            KingPointer(
                repo="team/EPAGO-DR-4B-k",
                digest=digest,
                author_hotkey="hk-champion",
                crowned_block=crowned,
                reign_started_block=crowned,
                coronation_lcb_e6=40_000,
                coronation_delta_e6=25_000,
            )
        )

    h.chain.inject_reveal("authority-hk", pointer(new_digest, 900))
    h.chain.advance(constants.VERDICT_REVEAL_BLOCKS)
    h.service.tick()
    assert h.state.king.ref.digest == new_digest

    # An older coronation republished later must not win.
    h.chain.inject_reveal("authority-hk", pointer(old_digest, 100))
    h.chain.advance(constants.VERDICT_REVEAL_BLOCKS)
    h.service.tick()
    assert h.state.king.ref.digest == new_digest


def test_authority_publishes_a_pointer_when_it_crowns(tmp_path):
    h = make_harness(tmp_path, outcome=make_outcome(lcb=0.09, delta=0.03, accepted=True))
    _authority_cfg(h, VALIDATOR_HK)
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    settle(h)

    pointers = [
        rp.payload for rp in h.chain.read_revealed_payloads() if rp.payload.startswith("ek1|")
    ]
    assert len(pointers) == 1
    parsed = parse_king_pointer(pointers[0], VALIDATOR_HK, h.chain.block)
    assert parsed.digest == digest
    assert parsed.author_hotkey == "hk-alice"
    assert parsed.coronation_delta == pytest.approx(0.03)


# --- emission inputs must come from chain, not from local duels ---------------


def test_the_arena_roster_is_derived_from_the_chain_succession(tmp_path):
    """An auditing validator runs no duels, so a local arena list is empty for
    it while the scoring validator pays one out -- a straight weight
    divergence, which Yuma then penalises everyone for.

    The roster therefore comes from accepted verdicts, which are the coronation
    record: each accept crowns a challenger and dethrones whoever held the
    crown before it.
    """
    h = make_harness(tmp_path, outcome=make_outcome(lcb=0.09, delta=0.03,
                                                    accepted=True, mu_priv=0.02))
    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)

    # One coronation so far: it displaced the genesis king, so the roster holds
    # whoever was displaced, never the challenger that just took the crown.
    entries = h.service._arena_entries(h.chain.block)
    assert "hk-alice" not in [e.hotkey for e in entries]


def test_a_near_miss_never_reaches_the_arena_roster(tmp_path):
    """Only being dethroned seats a hotkey, so nothing is paid before a crown."""
    h = make_harness(tmp_path, outcome=make_outcome(lcb=0.01, delta=0.03,
                                                    accepted=False, mu_priv=0.005))
    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)

    assert h.service._arena_entries(h.chain.block) == []


def test_chain_arena_excludes_plain_losses(tmp_path):
    h = make_harness(tmp_path, outcome=make_outcome(lcb=-0.02, delta=0.03, accepted=False))
    add_challenger(h, "bob", "hk-bob", "ck-bob-01", uid=3, digest_char="b")
    settle(h)

    assert h.service._arena_entries(h.chain.block) == []


# --- the private pool is committed before it grades anything ------------------


def test_no_duel_runs_before_the_active_pool_is_committed(tmp_path):
    """Committing on the way out chain-stamped a digest ~6 days after every
    verdict that pool had produced, which proved nothing while it was secret."""
    h = make_harness(tmp_path)
    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    published: list[str] = []
    real_publish = h.chain.publish_reveal

    def refuse_pool_commitment(payload, blocks):
        if payload.startswith("ep1|"):
            return False   # rate-limited
        published.append(payload)
        return real_publish(payload, blocks)

    h.chain.publish_reveal = refuse_pool_commitment
    settle(h)
    assert h.holder["duel_specs"] == []          # held off, no duel
    assert published_verdicts(h.chain) == []

    h.chain.publish_reveal = real_publish
    settle(h)
    assert h.holder["duel_specs"]                 # committed, duel proceeds
    assert h.state.committed_pool_epoch == 1
    commitments = [
        rp.payload for rp in h.chain.read_revealed_payloads() if rp.payload.startswith("ep1|")
    ]
    assert commitments and commitments[0].split("|")[1] == "1"


# --- pricing every losing attempt ---------------------------------------------


def test_a_near_copy_of_the_king_pays_a_cooldown(tmp_path):
    """The arena-farming hole.

    A noise-perturbed copy of the king scores lcb ~ 0: not byte-identical so the
    exact-copy gate misses it, far under the norm-sanity ratio so the probes
    miss it, and above the old -0.05 cooldown threshold so it cost nothing.
    Roughly half of those draw a positive LCB and book near-miss credit against
    the arena pool, which made repeat submission a free draw on both the arena
    budget and the 1-in-1000 false-acceptance tail.
    """
    h = make_harness(tmp_path, outcome=make_outcome(lcb=-0.004, delta=0.03, accepted=False))
    digest, _, _ = add_challenger(h, "mallory", "hk-mal", "ck-mal-01", uid=4, digest_char="a")

    settle(h)

    assert h.state.statuses[digest] == SubmissionStatus.DUEL_LOST.value
    assert "hk-mal" in h.state.cooldowns
    assert cooldown_until(h.state, "hk-mal", h.chain.block) is not None


def test_a_near_miss_still_costs_nothing(tmp_path):
    h = make_harness(tmp_path, outcome=make_outcome(lcb=0.01, delta=0.03, accepted=False,
                                                    mu_priv=0.005))
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    settle(h)

    assert h.state.statuses[digest] == SubmissionStatus.NEAR_MISS.value
    assert h.state.cooldowns == {}


def test_overfit_rejection_pays_even_though_it_cleared_the_floor(tmp_path):
    """lcb > delta but the private half failed: that is the generator-overfit
    attack, and it must not be cheaper than an honest loss."""
    h = make_harness(tmp_path, outcome=make_outcome(lcb=0.09, delta=0.03, accepted=False,
                                                    mu_priv=-0.01))
    digest, _, _ = add_challenger(h, "eve", "hk-eve", "ck-eve-01", uid=5, digest_char="b")

    settle(h)

    assert h.state.statuses[digest] == SubmissionStatus.DUEL_LOST.value
    assert "hk-eve" in h.state.cooldowns


# --- intake must not depend on how often the validator polls -------------------


def test_supersession_is_resolved_over_all_history_not_the_scan_window(tmp_path):
    """Resolving "latest per hotkey" over the slice since the last poll made a
    validator's queue depend on its tick cadence: a box that had just restarted
    and one ticking steadily admitted different challenges from identical chain
    state."""
    h = make_harness(tmp_path)
    first, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    h.chain.advance(50)
    second, _, _ = add_challenger(h, "alice2", "hk-alice", "ck-alice-01", uid=2,
                                  digest_char="b", register=False)

    # A validator whose scan pointer is already past the first reveal must still
    # see it superseded, not treat the second as the only entry it ever knew.
    h.state.last_scan_block = h.chain.block
    settle(h)

    queued = {q.digest for q in h.state.queue} | set(h.state.statuses)
    assert second in queued
    assert first not in {q.digest for q in h.state.queue}


# --- competition rounds -------------------------------------------------------


def test_nothing_is_evaluated_until_the_authority_opens_a_round(tmp_path):
    """The trigger is a liveness dependency, by design.

    With no `er1` on chain the queue fills and the king keeps earning, but no
    duel runs and no verdict is committed. This is the cost of an owner-held
    trigger and the reason the mechanism spec's "no privileged operator"
    requirement no longer holds.
    """
    h = make_harness(tmp_path)
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    for _ in range(4):                       # plenty of ticks, no round trigger
        h.service.tick()
        h.chain.advance(constants.VERDICT_REVEAL_BLOCKS + 1)

    assert h.holder["duel_specs"] == []
    assert published_verdicts(h.chain) == []
    assert [q.digest for q in h.state.queue] == [digest]   # queued, waiting
    assert h.state.king.ref.digest == h.cfg.seed.seed_digest


def test_a_round_from_a_non_authority_hotkey_is_ignored(tmp_path):
    """Whoever can open a round chooses the exam entropy and the timing."""
    from epago.core.reveal import build_round_start

    h = make_harness(tmp_path)
    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    h.service.tick()

    h.chain.advance(constants.ROUND_MIN_INTERVAL_BLOCKS + 1)
    h.chain.inject_reveal("some-other-hotkey", build_round_start(1))
    h.chain.advance(1)
    for _ in range(3):
        h.service.tick()
        h.chain.advance(constants.VERDICT_REVEAL_BLOCKS + 1)

    assert h.holder["duel_specs"] == []
    assert h.state.last_round_run == 0


def test_a_round_opened_too_soon_is_ignored(tmp_path):
    """The 2-day cadence is enforced by every validator, not by the caller.

    Otherwise the authority could run rounds back to back and hand a favoured
    miner as many exam draws as it liked.
    """
    from epago.core.reveal import build_round_start

    h = make_harness(tmp_path)
    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)                                   # round 1 lands legitimately
    assert h.state.last_round_run == 1

    add_challenger(h, "bob", "hk-bob", "ck-bob-001", uid=3, digest_char="b",
                   king_digest=h.state.king.ref.digest)
    h.service.tick()
    h.chain.advance(100)                        # far short of the interval
    h.chain.inject_reveal(ROUND_AUTHORITY_HK, build_round_start(2))
    h.chain.advance(1)
    duels_before = len(h.holder["duel_specs"])
    for _ in range(3):
        h.service.tick()
        h.chain.advance(constants.VERDICT_REVEAL_BLOCKS + 1)

    assert len(h.holder["duel_specs"]) == duels_before
    assert h.state.last_round_run == 1


def test_a_replayed_round_number_is_ignored(tmp_path):
    from epago.core.reveal import build_round_start

    h = make_harness(tmp_path)
    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)
    duels_before = len(h.holder["duel_specs"])

    h.chain.advance(constants.ROUND_MIN_INTERVAL_BLOCKS + 1)
    h.chain.inject_reveal(ROUND_AUTHORITY_HK, build_round_start(1))  # same number
    h.chain.advance(1)
    for _ in range(3):
        h.service.tick()
        h.chain.advance(constants.VERDICT_REVEAL_BLOCKS + 1)

    assert len(h.holder["duel_specs"]) == duels_before


def test_submissions_after_the_round_opens_wait_for_the_next_one(tmp_path):
    """A reveal that has already seen the round-start hash cannot enter it.

    The block hash at the trigger mints the exam. Admitting a checkpoint
    committed after that point would hand its author the questions while the
    weights were still changeable — the one thing commit-reveal exists to stop.
    """
    # Near-miss outcome: nobody is crowned, so the late entrant is not dropped
    # as stale and we can see it simply waiting for the next round.
    h = make_harness(tmp_path, outcome=make_outcome(lcb=0.01, delta=0.03, accepted=False))
    early, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    h.service.tick()
    open_round(h)

    late, _, _ = add_challenger(h, "bob", "hk-bob", "ck-bob-001", uid=3, digest_char="b")
    for _ in range(3):
        h.service.tick()
        h.chain.advance(constants.VERDICT_REVEAL_BLOCKS + 1)

    dueled = {spec.challenger_dir for spec in h.holder["duel_specs"]}
    assert len(dueled) == 1                       # only the early one ran
    assert h.state.statuses[early] in (
        SubmissionStatus.ACCEPTED.value, SubmissionStatus.NEAR_MISS.value
    )
    assert late in {q.digest for q in h.state.queue}   # still waiting


def test_the_whole_field_answers_one_exam(tmp_path):
    h = make_harness(tmp_path)
    for i, (name, hk) in enumerate(
        [("alice", "hk-alice"), ("bob", "hk-bob"), ("carol", "hk-carol")]
    ):
        add_challenger(h, name, hk, f"ck-{name}-01", uid=2 + i, digest_char="abc"[i])

    settle(h)

    specs = h.holder["duel_specs"]
    assert len(specs) == 4  # 3 entrants + the winner's confirmation
    assert len({tuple(s.public_tasks) for s in specs[:3]}) == 1
    assert len({tuple(s.private_tasks) for s in specs[:3]}) == 1
    # The exam is keyed on the round, never on an entrant.
    assert len({s.block_hash_at_reveal for s in specs}) == 1
    # The confirmation is the same round on fresh public AND private samples.
    assert tuple(specs[3].public_tasks) != tuple(specs[0].public_tasks)
    assert tuple(specs[3].private_tasks) != tuple(specs[0].private_tasks)


def test_only_the_winner_is_committed_as_accept(tmp_path):
    h = make_harness(tmp_path)
    for i, (name, hk) in enumerate(
        [("alice", "hk-alice"), ("bob", "hk-bob"), ("carol", "hk-carol")]
    ):
        add_challenger(h, name, hk, f"ck-{name}-01", uid=2 + i, digest_char="abc"[i])

    settle(h)

    verdicts = [parse_verdict(p, VALIDATOR_HK, h.chain.block) for p in published_verdicts(h.chain)]
    assert len(verdicts) == 3
    accepts = [v for v in verdicts if v.decision is VerdictDecision.ACCEPT]
    assert len(accepts) == 1, "a round crowns exactly one model"
    # The rejected two beat the king as well, so on-chain they read as
    # near-misses and collect arena credit rather than cooldowns.
    rejects = [v for v in verdicts if v.decision is VerdictDecision.REJECT]
    assert all(v.is_near_miss for v in rejects)
    assert all(v.round == 1 for v in verdicts)
    assert h.state.cooldowns == {}


def test_round_field_is_capped_and_the_overflow_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ROUND_MAX_ENTRANTS", 2)
    # Near-miss outcome so no coronation drops the deferred pair as stale.
    h = make_harness(tmp_path, outcome=make_outcome(lcb=0.01, delta=0.03, accepted=False))
    for i, ch in enumerate("abcd"):
        add_challenger(h, f"m{ch}", f"hk-{ch}", f"ck-{ch}-01", uid=2 + i, digest_char=ch)

    h.service.tick()
    open_round(h)
    for _ in range(3):
        h.service.tick()
        h.chain.advance(constants.VERDICT_REVEAL_BLOCKS + 1)

    assert len(h.holder["duel_specs"]) == 2
    # The two that were cut keep their place for the next competition.
    assert len(h.state.queue) == 2


def test_a_stolen_checkpoint_cannot_enter_the_same_round(tmp_path):
    """Two entrants with identical weights: the earlier reveal keeps the slot.

    Digest ownership alone does not catch this. An `hf:` digest is a revision
    hash, so the same weights pushed to a second repo get a different digest and
    enter as a separate challenger. Under rounds the thief and the victim end up
    in the same field on the same exam, score identically, and the digest
    tie-break decides the crown.
    """
    h = make_harness(tmp_path)
    stolen = b"S" * 1000

    victim, _, _ = add_challenger(
        h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a", weights=stolen
    )
    h.chain.advance()
    thief, _, _ = add_challenger(
        h, "mallory", "hk-mal", "ck-mal-0001", uid=3, digest_char="b", weights=stolen
    )
    assert victim != thief  # different digests, identical weights

    settle(h)

    assert h.state.statuses[thief] == SubmissionStatus.FAILED_INTAKE.value
    assert h.state.failure_memory[thief]["code"] == "duplicate_weights"
    # Only the victim was dueled (plus its confirmation); the thief never was.
    assert len(h.holder["duel_specs"]) == 2
    assert all("-aaaaaaaa" in s.round_id for s in h.holder["duel_specs"])
    assert h.state.king.ref.digest == victim       # and the victim took the crown


def test_reentered_weights_are_rejected_across_rounds(tmp_path):
    """Re-uploading already-dueled weights under a new digest is terminal.

    One-duel-per-digest is a rule about revision hashes; the same safetensors
    pushed to a second repo mint a fresh digest and would buy a second draw on
    the same content. The fingerprint registry makes weights that have ever
    dueled terminal under every digest — and the re-entry cools the hotkey
    down, so grinding noise-free copies of old submissions costs a retrain.
    """
    h = make_harness(tmp_path, outcome=make_outcome(lcb=-0.01, accepted=False, mu_priv=-0.01))
    stale = b"W" * 1000

    first, _, _ = add_challenger(
        h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a", weights=stale
    )
    settle(h)
    assert h.state.statuses[first] == SubmissionStatus.DUEL_LOST.value
    assert h.state.king.ref.digest == h.cfg.seed.seed_digest  # nobody crowned

    again, _, _ = add_challenger(
        h, "mallory", "hk-mal", "ck-mal-0001", uid=3, digest_char="b", weights=stale
    )
    settle(h, round_no=2)

    assert h.state.statuses[again] == SubmissionStatus.FAILED_INTAKE.value
    assert first in h.state.failure_memory[again]["detail"]
    assert "hk-mal" in h.state.cooldowns  # known content is not a free retry

    # The registry is durable: a restarted validator still knows the weights.
    reloaded = ValidatorState.load(h.tmp / "state")
    assert reloaded.seen_fingerprints == h.state.seen_fingerprints
    assert first in reloaded.seen_fingerprints.values()


def test_unconfirmed_winner_is_demoted_not_crowned(tmp_path):
    """A provisional winner that fails its confirmation duel settles near-miss.

    One 99.9% clear is one lottery ticket; the crown requires two independent
    ones. The demoted entrant keeps everything a near-miss earns — arena
    credit, the re-duel right, no cooldown — because it did beat the king once
    and may well be genuinely better.
    """
    h = make_harness(tmp_path)
    h.holder["outcomes"] = [
        make_outcome(lcb=0.08, accepted=True),                              # round duel
        make_outcome(lcb=0.005, accepted=False, mu_priv=-0.01),             # confirmation
    ]
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")

    settle(h)

    assert len(h.holder["duel_specs"]) == 2
    assert h.state.king.ref.digest == h.cfg.seed.seed_digest   # crown unmoved
    assert h.state.statuses[digest] == SubmissionStatus.NEAR_MISS.value
    assert "hk-alice" not in h.state.cooldowns
    assert digest in h.state.near_misses                       # re-duel right intact
    verdicts = published_verdicts(h.chain)
    assert verdicts and all(v.split("|")[2] == "R" for v in verdicts)


def test_confirmation_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "CORONATION_CONFIRMATION_DUELS", 0)
    h = make_harness(tmp_path)
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)
    assert len(h.holder["duel_specs"]) == 1
    assert h.state.king.ref.digest == digest


def test_near_ties_crown_the_earlier_reveal(tmp_path):
    """Inside one noise floor, reveal order decides — not the digest.

    Two LCBs closer than the calibrated king-vs-king noise are the same
    measurement, and the digest tie-break handed exactly that band to anyone
    who perturbed a pending challenger's weights and out-drew it by luck.
    """
    from epago.core.types import Entrant, RoundResult

    def result(digest: str, lcb: float) -> RoundResult:
        return RoundResult(
            entrant=Entrant(digest=digest, repo="r/x", author_hotkey="hk", challenger_dir=tmp_path),
            outcome=make_outcome(lcb=lcb),
        )

    original, copy = result("hf:" + "a" * 40, 0.080), result("hf:" + "z" * 40, 0.081)
    blocks = {original.entrant.digest: 5, copy.entrant.digest: 40}

    # Within the floor: the earlier reveal wins despite the lower draw.
    pick = ValidatorService._pick_winner(
        [original, copy], noise_floor=0.03, reveal_blocks=blocks
    )
    assert pick is original

    # A gap the floor can resolve is a real difference: the higher LCB wins.
    clear = result("hf:" + "z" * 40, 0.140)
    pick = ValidatorService._pick_winner(
        [original, clear], noise_floor=0.03, reveal_blocks=blocks
    )
    assert pick is clear

    # No calibration yet: raw comparison, digest last.
    pick = ValidatorService._pick_winner(
        [original, copy], noise_floor=0.0, reveal_blocks=blocks
    )
    assert pick is copy


# --- API-key round trigger ----------------------------------------------------


def test_api_trigger_opens_a_round_from_the_current_block(tmp_path):
    """The local trigger mints a round seeded by the CURRENT chain block, whose
    hash the owner cannot choose — so an API trigger leaks the exam no more than
    the on-chain one does."""
    from epago.validator.roundapi import RoundTrigger

    h = make_harness(tmp_path)
    # Replace the chain authority with the local latch.
    h.cfg = replace(h.cfg, chain=replace(h.cfg.chain, round_authority_hotkey=""))
    h.service.cfg = h.cfg
    trigger = RoundTrigger()
    h.service._round_trigger = trigger

    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    h.service.tick()                       # intake; no trigger yet
    assert h.holder["duel_specs"] == []

    h.chain.advance(constants.ROUND_MIN_INTERVAL_BLOCKS + 1)
    trigger.request()                      # owner POSTs with the API key
    for _ in range(3):
        h.service.tick()
        h.chain.advance(constants.VERDICT_REVEAL_BLOCKS + 1)

    assert h.holder["duel_specs"], "the API trigger did not open a round"
    assert h.state.last_round_run == 1
    assert h.state.last_round_block > 0
    spec = h.holder["duel_specs"][0]
    # The exam seed is the current block's hash, not anything the caller chose.
    assert spec.block_hash_at_reveal == h.chain.block_hash(h.state.last_round_block)


def test_api_trigger_respects_the_minimum_interval(tmp_path):
    from epago.validator.roundapi import RoundTrigger

    h = make_harness(tmp_path)
    h.cfg = replace(h.cfg, chain=replace(h.cfg.chain, round_authority_hotkey=""))
    h.service.cfg = h.cfg
    trigger = RoundTrigger()
    h.service._round_trigger = trigger

    add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    h.service.tick()
    h.chain.advance(constants.ROUND_MIN_INTERVAL_BLOCKS + 1)
    trigger.request()
    for _ in range(3):
        h.service.tick(); h.chain.advance(constants.VERDICT_REVEAL_BLOCKS + 1)
    duels_after_first = len(h.holder["duel_specs"])
    assert duels_after_first >= 1

    # A second request too soon is ignored.
    add_challenger(h, "bob", "hk-bob", "ck-bob-001", uid=3, digest_char="b",
                   king_digest=h.state.king.ref.digest)
    h.service.tick()
    h.chain.advance(100)                    # far short of the interval
    trigger.request()
    for _ in range(3):
        h.service.tick(); h.chain.advance(constants.VERDICT_REVEAL_BLOCKS + 1)
    assert len(h.holder["duel_specs"]) == duels_after_first  # no new round


def test_round_trigger_rejects_a_bad_key():
    """The latch only rises for a request the handler accepted; a wrong key
    never reaches it."""
    import hmac
    from epago.validator.roundapi import RoundTrigger

    trigger = RoundTrigger()
    # The handler compares in constant time; here we assert the latch semantics
    # the handler depends on: request() raises it, take() lowers it once.
    assert trigger.take() is False
    trigger.request()
    assert trigger.take() is True
    assert trigger.take() is False
    # a burst collapses to one pending round
    trigger.request(); trigger.request(); trigger.request()
    assert trigger.take() is True
    assert trigger.take() is False
    assert trigger.total_requests == 4
    assert hmac.compare_digest("k", "k")   # the primitive the handler uses


def test_a_spent_hotkey_is_refused_even_after_a_win(tmp_path):
    """Winning does not refill the hotkey.

    A crowned miner that wants to defend with a better model registers a fresh
    hotkey like anyone else. Otherwise the incumbent alone would have unlimited
    free attempts, which is the opposite of the pressure the rule creates.
    """
    h = make_harness(
        tmp_path, outcome=make_outcome(lcb=0.09, delta=0.03, accepted=True, mu_priv=0.02)
    )
    d_a, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)
    assert h.state.spent_hotkeys["hk-alice"] == d_a

    h.chain.advance()
    d_b, _, _ = add_challenger(
        h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="b", register=False
    )
    h.service.tick()
    assert d_b not in h.state.statuses


def test_the_spent_ledger_survives_a_restart(tmp_path):
    """A hotkey must not be refilled by a validator restart.

    The ledger is the whole enforcement, so if it lived only in memory a
    miner could wait out a redeploy and submit again for free.
    """
    from epago.validator.state import ValidatorState

    st = ValidatorState.load(tmp_path / "state")
    st.spent_hotkeys["hk-alice"] = "d" * 64
    st.save()

    reloaded = ValidatorState.load(tmp_path / "state")
    assert reloaded.spent_hotkeys == {"hk-alice": "d" * 64}


# --- a validator that runs no duels computes the same weights -----------------


def test_a_validator_with_no_local_king_derives_it_from_chain(tmp_path):
    """An auditing validator runs no duels, so it has no local king.

    Without a chain-derived king it would compute an empty board and burn
    everything while the scoring validator paid a champion. That is a straight
    weight divergence, and Yuma penalises both sides for it.
    """
    h = make_harness(
        tmp_path, outcome=make_outcome(lcb=0.09, delta=0.03, accepted=True, mu_priv=0.02)
    )
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)

    derived = h.service._king_from_chain(h.chain.block)
    assert derived is not None
    assert derived[0] == digest
    assert derived[1] == "hk-alice"


def test_the_chain_king_is_preferred_and_a_disagreement_is_recorded(tmp_path):
    """Local state losing to chain is the safe direction.

    A local king the network never crowned means this box is out of consensus;
    paying the chain's king keeps the vector in agreement while the alarm stays
    visible in status output.
    """
    from epago.core.types import ModelRef

    h = make_harness(
        tmp_path, outcome=make_outcome(lcb=0.09, delta=0.03, accepted=True, mu_priv=0.02)
    )
    digest, _, _ = add_challenger(h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a")
    settle(h)

    # Force local state to a king the chain never crowned.
    h.state.set_king(
        ModelRef("z/EPAGO-DR-4B-z", hf_digest("f")), "hk-zed",
        crowned_block=h.chain.block, coronation_lcb=0.05, coronation_delta=0.02,
    )
    king = h.service._king_emission_state(h.chain.block)

    assert king is not None
    assert king.hotkey == "hk-alice"  # the chain's king, not the local one
    assert h.state.last_error["code"] == "king_disagrees_with_chain"


def test_no_coronation_yet_means_no_king_and_the_emission_burns(tmp_path):
    """Genesis: nothing has been crowned, so there is nobody to pay."""
    h = make_harness(tmp_path)
    assert h.service._king_from_chain(h.chain.block) is None


# --- private submissions stay inside their author's prefix -------------------


def test_a_private_submission_outside_its_authors_prefix_is_refused(tmp_path):
    """The prefix is the ownership boundary for a private upload.

    A public `hf:` ref is owned by a repository whose name must carry the
    author's hotkey. A private `sha256:` ref has no owner field — the
    credential sealed to that hotkey can only write under its own prefix, so a
    ref pointing elsewhere cannot have been written by its claimed author.
    The digest cannot catch this: anyone able to read the bytes can compute it,
    so without the prefix rule one miner could reveal a rival's upload as its
    own.
    """
    from epago.core.types import ModelRef
    from epago.validator.intake import validate_submission_prefix

    sha = "sha256:" + "a" * 64
    mine = ModelRef(repo="submissions/hk-alice/EPAGO-DR-4B", digest=sha)
    assert validate_submission_prefix(mine, "hk-alice") is None

    theirs = ModelRef(repo="submissions/hk-bob/EPAGO-DR-4B", digest=sha)
    failure = validate_submission_prefix(theirs, "hk-alice")
    assert failure is not None and failure[0] == "wrong_prefix"

    loose = ModelRef(repo="anywhere/EPAGO-DR-4B", digest=sha)
    assert validate_submission_prefix(loose, "hk-alice")[0] == "wrong_prefix"


def test_a_public_submission_is_unaffected_by_the_prefix_rule(tmp_path):
    """`hf:` refs live in a repository the author owns, which the repo-name
    rule already checks. Applying a key-prefix rule to them would be wrong."""
    from epago.core.types import ModelRef
    from epago.validator.intake import validate_submission_prefix

    public = ModelRef(repo="alice/EPAGO-DR-4B-x", digest=hf_digest("a"))
    assert validate_submission_prefix(public, "hk-alice") is None


def test_a_crowned_model_is_fetchable_from_its_public_location(tmp_path):
    """A challenger lives where only its author can write and only the scoring
    validator can read. That is correct while it is a challenger, and broken
    the moment it is crowned: every other party needs the king.

    Coronation republishes it under `kings/<digest>/`, and because that path is
    derived from the digest, any party constructs it without a manifest, a
    pointer, or a call to whoever published it.
    """
    from epago.core.types import ModelRef
    from epago.publishing.publisher import king_object_repo
    from epago.validator.service import ValidatorService

    sha = "sha256:" + "c" * 64
    private = ModelRef(repo="submissions/hk-alice/model", digest=sha)

    fallbacks = ValidatorService._public_fallbacks(private)
    assert [f.repo for f in fallbacks] == [king_object_repo(sha)]
    # The digest travels unchanged, so a fallback is only accepted once its
    # bytes rehash to the committed value — an impostor serving something else
    # at that path is caught rather than trusted.
    assert fallbacks[0].digest == sha


def test_a_public_model_needs_no_fallback(tmp_path):
    """An `hf:` primary pins a revision of one specific repository; another
    repository's revision hash proves nothing about content equality."""
    from epago.core.types import ModelRef
    from epago.validator.service import ValidatorService

    public = ModelRef(repo="alice/EPAGO-DR-4B-x", digest=hf_digest("a"))
    assert ValidatorService._public_fallbacks(public) == ()


def test_the_public_copy_does_not_fall_back_to_itself(tmp_path):
    from epago.core.types import ModelRef
    from epago.publishing.publisher import king_object_repo
    from epago.validator.service import ValidatorService

    sha = "sha256:" + "d" * 64
    already = ModelRef(repo=king_object_repo(sha), digest=sha)
    assert ValidatorService._public_fallbacks(already) == ()


# --- the credential mailbox is published on a cadence ------------------------


def test_no_mailbox_is_published_when_private_upload_is_not_configured(tmp_path, monkeypatch):
    """Private upload is optional. With no object store configured a validator
    publishes nothing and public submission keeps working unchanged."""
    for var in ("EPAGO_R2_PARENT_ACCESS_KEY", "EPAGO_R2_PARENT_SECRET_KEY",
                "EPAGO_R2_ACCOUNT_ID", "EPAGO_S3_BUCKET", "EPAGO_S3_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)

    h = make_harness(tmp_path)
    h.service._maybe_publish_mailbox(h.chain.block)

    assert h.state.mailbox_digest == ""
    assert not (h.state.state_dir / "publications" / "mailbox").exists()


def test_a_mailbox_failure_never_stops_the_validator(tmp_path, monkeypatch):
    """A box that stopped scoring because a credential refresh failed would be
    trading a working subnet for a storage problem."""
    monkeypatch.setenv("EPAGO_R2_PARENT_ACCESS_KEY", "parent")
    monkeypatch.setenv("EPAGO_R2_PARENT_SECRET_KEY", "secret")
    monkeypatch.setenv("EPAGO_R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("EPAGO_S3_BUCKET", "bucket")
    monkeypatch.setenv("EPAGO_S3_ENDPOINT", "https://acct.r2.cloudflarestorage.com")

    h = make_harness(tmp_path)
    monkeypatch.setattr(
        h.service, "_mailbox_recipients", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    h.service._maybe_publish_mailbox(h.chain.block)  # must not raise
    assert h.state.last_error["code"] == "mailbox_publish_failed"


def test_only_ed25519_hotkeys_receive_an_envelope(tmp_path):
    """sr25519 signs but has no encryption, so there is nothing to seal to.

    Such a miner is skipped here and told why at intake, rather than handed an
    envelope it can never open.
    """
    h = make_harness(tmp_path)

    class _Neuron:
        def __init__(self, hotkey, ed):
            self.hotkey = hotkey
            self.ed25519_public = ed

    h.service.chain.neurons = lambda: [  # type: ignore[method-assign]
        _Neuron("hk-ed", b"\x01" * 32),
        _Neuron("hk-sr", None),
    ]
    assert h.service._mailbox_recipients() == {"hk-ed": b"\x01" * 32}


def test_the_mailbox_is_not_reissued_on_every_tick(tmp_path, monkeypatch):
    """Reissuing constantly would invalidate an upload already in flight."""
    monkeypatch.setenv("EPAGO_R2_PARENT_ACCESS_KEY", "parent")
    monkeypatch.setenv("EPAGO_R2_PARENT_SECRET_KEY", "secret")
    monkeypatch.setenv("EPAGO_R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("EPAGO_S3_BUCKET", "bucket")
    monkeypatch.setenv("EPAGO_S3_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("EPAGO_MAILBOX_INTERVAL_BLOCKS", "600")

    h = make_harness(tmp_path)
    calls = []
    monkeypatch.setattr(h.service, "_mailbox_recipients", lambda: calls.append(1) or {})

    h.service._maybe_publish_mailbox(1000)
    h.service._maybe_publish_mailbox(1001)  # inside the interval
    assert len(calls) == 1

    h.service._maybe_publish_mailbox(1000 + 600)
    assert len(calls) == 2


def test_a_private_submission_needs_a_hotkey_credentials_can_be_sealed_to():
    """The mailbox failure mode is silence: an sr25519 miner never receives an
    envelope and has nowhere to learn why. Intake says it once, plainly, with
    the command that fixes it."""
    import nacl.signing
    from scalecodec.utils.ss58 import ss58_encode

    from epago.core.types import ModelRef
    from epago.validator.intake import validate_sealable_hotkey

    ed_public = bytes(nacl.signing.SigningKey.generate().verify_key)
    ed_hotkey = ss58_encode(ed_public.hex(), 42)
    sha = "sha256:" + "e" * 64

    private = ModelRef(repo=f"submissions/{ed_hotkey}/model", digest=sha)
    assert validate_sealable_hotkey(private, ed_hotkey) is None

    # A key that is not a valid Ed25519 point cannot be sealed to.
    bad_hotkey = ss58_encode((b"\xff" * 32).hex(), 42)
    failure = validate_sealable_hotkey(ModelRef(repo="submissions/x/m", digest=sha), bad_hotkey)
    assert failure is not None
    assert failure[0] == "hotkey_not_sealable"
    assert "ed25519" in failure[1].lower()


def test_a_public_submission_does_not_care_about_the_curve():
    """A public upload needs no credential, so the curve is irrelevant."""
    from epago.core.types import ModelRef
    from epago.validator.intake import validate_sealable_hotkey

    public = ModelRef(repo="alice/EPAGO-DR-4B-x", digest=hf_digest("a"))
    assert validate_sealable_hotkey(public, "not-even-an-address") is None


def test_a_sealed_release_draws_the_public_half_from_its_committed_pool(tmp_path, monkeypatch):
    """Which source serves the public half is decided by the release name, so
    an audit record is self-describing: a replay knows which verification path
    applies without being told separately."""
    import json

    from epago.taskgen.sealed_pool import pool_digest

    pool = tmp_path / "pool.jsonl"
    rows = [
        {"task_id": f"tk-{i:04d}", "question": f"q{i}", "answer": f"a{i}",
         "evidence_doc_ids": [f"d{i}"], "template": "bridge_intersection", "hops": 3}
        for i in range(40)
    ]
    pool.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    import dataclasses

    h = make_harness(tmp_path)
    h.service.cfg = dataclasses.replace(
        h.cfg,
        eval=dataclasses.replace(
            h.cfg.eval,
            taskgen_release="POOL1",
            public_pool_path=str(pool),
            public_pool_digest=pool_digest(pool.read_bytes()),
        ),
    )

    tasks = h.service._public_tasks(seed=99, n=10)
    assert len(tasks) == 10
    assert len({t.task_id for t in tasks}) == 10
    # Same seed, same exam; different seed, different exam.
    assert [t.task_id for t in h.service._public_tasks(99, 10)] == [t.task_id for t in tasks]
    assert [t.task_id for t in h.service._public_tasks(100, 10)] != [t.task_id for t in tasks]


def test_a_sealed_release_without_a_pool_refuses_rather_than_falling_back(tmp_path, monkeypatch):
    """Silently falling back to the generator would run a different exam than
    the contract names, and record it as if it were the committed one."""
    from epago.taskgen.sealed_pool import SealedPoolError

    import dataclasses

    h = make_harness(tmp_path)
    h.service.cfg = dataclasses.replace(
        h.cfg,
        eval=dataclasses.replace(h.cfg.eval, taskgen_release="POOL1", public_pool_path=""),
    )

    with pytest.raises(SealedPoolError, match="public_pool_path is empty"):
        h.service._public_tasks(seed=1, n=10)


def _sealed_release_harness(tmp_path, n=40):
    """A harness whose contract names a sealed pool, with its manifest committed."""
    import dataclasses
    import json

    from epago.taskgen.sealed_pool import Manifest, load_pool, pool_digest, write_manifest

    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "\n".join(
            json.dumps(
                {"task_id": f"tk-{i:04d}", "question": f"q{i}", "answer": f"a{i}",
                 "evidence_doc_ids": [f"d{i}"], "template": "bridge_intersection", "hops": 3}
            )
            for i in range(n)
        )
        + "\n"
    )
    digest = pool_digest(pool.read_bytes())
    manifest_digest = write_manifest(
        Manifest.from_pool(load_pool(pool, digest), digest), tmp_path / "manifest.json"
    )
    h = make_harness(tmp_path)
    h.service.cfg = dataclasses.replace(
        h.cfg,
        eval=dataclasses.replace(
            h.cfg.eval,
            taskgen_release="POOL1",
            public_pool_path=str(pool),
            public_pool_digest=digest,
            public_pool_manifest_path=str(tmp_path / "manifest.json"),
            public_pool_manifest_digest=manifest_digest,
        ),
    )
    return h


def test_a_round_retires_the_tasks_it_asked(tmp_path):
    """Published tasks are training data. Drawing them again would let a
    challenger trained after that publication answer part of its exam from
    memory rather than from research."""
    h = _sealed_release_harness(tmp_path)

    first = h.service._public_tasks(seed=5, n=10)
    h.service._stage_public_pool_round(1, first)
    assert set(h.service.state.served_public_task_ids) == {t.task_id for t in first}

    # Even replaying the *same* seed cannot bring a published task back.
    second = h.service._public_tasks(seed=5, n=10)
    assert not ({t.task_id for t in second} & {t.task_id for t in first})


def test_a_staged_round_publishes_its_tasks_in_full(tmp_path):
    """An auditor re-grades against the released file, so it needs the question
    and answer text — and the digests that say which pool it came from."""
    import json

    h = _sealed_release_harness(tmp_path)
    tasks = h.service._public_tasks(seed=5, n=10)
    h.service._stage_public_pool_round(7, tasks)

    staged = sorted(h.service.audit_log.delayed_dir.glob("*publicpool-round000007*"))
    assert len(staged) == 1
    payload = json.loads(staged[0].read_text())
    assert payload["round"] == 7
    assert payload["manifest_digest"] == h.service.cfg.eval.public_pool_manifest_digest
    assert {t["task_id"] for t in payload["tasks"]} == {t.task_id for t in tasks}
    assert all(t["question"] and t["answer"] for t in payload["tasks"])


def test_a_round_is_not_published_before_its_delay_elapses(tmp_path):
    """Releasing immediately would hand the next challenger a live answer key."""
    h = _sealed_release_harness(tmp_path)
    h.service._stage_public_pool_round(2, h.service._public_tasks(seed=5, n=10))

    assert not list(h.service.audit_log.published_dir.glob("*publicpool*"))
    released = h.service.audit_log.release_due(h.service.deps.clock())
    assert not [p for p in released if "publicpool" in p.name]

    later = h.service.deps.clock() + constants.AUDIT_PUBLISH_DELAY_BLOCKS
    assert [p for p in h.service.audit_log.release_due(later) if "publicpool" in p.name]




def test_a_generator_release_stages_no_pool_round(tmp_path):
    """Generator-served tasks regenerate from a seed and the round record
    already pins them, so a second publication would be dead weight."""
    h = _generator_release_harness(tmp_path)
    h.service._stage_public_pool_round(1, h.service._public_tasks(seed=5, n=5))
    assert not list(h.service.audit_log.delayed_dir.glob("*publicpool*"))
    assert h.service.state.served_public_task_ids == []


def test_served_ids_survive_a_restart(tmp_path):
    """Losing them would silently re-serve published tasks, which is exactly the
    leak the exclusion exists to prevent."""
    from epago.validator.state import ValidatorState

    h = _sealed_release_harness(tmp_path)
    tasks = h.service._public_tasks(seed=5, n=10)
    h.service._stage_public_pool_round(1, tasks)
    h.service.state.save()

    reloaded = ValidatorState.load(h.service.state.state_dir)
    assert set(reloaded.served_public_task_ids) == {t.task_id for t in tasks}
