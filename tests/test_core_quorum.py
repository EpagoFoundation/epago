"""Tests for epago.core.quorum — coronation as a pure function of chain state."""

from __future__ import annotations

from epago.config import QuorumSection
from epago.core.quorum import active_evaluators, derive_coronation, evaluate_quorum
from epago.core.types import EvaluatorInfo, Verdict, VerdictDecision

DIGEST = "hf:" + "c" * 40
OTHER_DIGEST = "hf:" + "d" * 40
AUDIT16 = "00" * 8

CFG = QuorumSection(
    theta=0.51,
    active_window_duels=20,
    verdict_timeout_blocks=7200,
    bootstrap_min_evaluators=3,
)


def verdict(hotkey: str, decision: VerdictDecision, block: int, digest: str = DIGEST) -> Verdict:
    return Verdict(
        challenger_digest=digest,
        decision=decision,
        lcb_pub_e6=30_000 if decision is VerdictDecision.ACCEPT else -1_000,
        mu_priv_e6=1_000,
        delta_e6=25_000,
        round=1,
        private_pool_epoch=1,
        audit_digest=AUDIT16,
        validator_hotkey=hotkey,
        block=block,
    )


def evaluators(*stakes: float) -> list[EvaluatorInfo]:
    return [
        EvaluatorInfo(hotkey=f"val-{i}", stake=s, last_verdict_block=100)
        for i, s in enumerate(stakes)
    ]


class TestActiveEvaluators:
    def test_window_filter(self):
        evs = [
            EvaluatorInfo(hotkey="fresh", stake=10, last_verdict_block=95),
            EvaluatorInfo(hotkey="edge", stake=10, last_verdict_block=80),
            EvaluatorInfo(hotkey="stale", stake=10, last_verdict_block=79),
        ]
        active = active_evaluators(evs, current_block=100, active_window_blocks=20)
        assert [e.hotkey for e in active] == ["fresh", "edge"]


class TestQuorumThreshold:
    def test_reached_exactly_at_theta_boundary(self):
        # active stake 100, theta 0.51 -> exactly 51 accept-stake crowns (>=, not >)
        evs = evaluators(51, 25, 24)
        vs = [verdict("val-0", VerdictDecision.ACCEPT, 10)]
        status = evaluate_quorum(DIGEST, vs, evs, CFG)
        assert status.accept_stake == 51
        assert status.active_stake == 100
        assert not status.bootstrap_mode
        assert status.reached

    def test_not_reached_just_below_theta(self):
        evs = evaluators(50.9, 25, 24.1)
        vs = [verdict("val-0", VerdictDecision.ACCEPT, 10)]
        assert not evaluate_quorum(DIGEST, vs, evs, CFG).reached

    def test_rejects_do_not_count_toward_accept(self):
        evs = evaluators(40, 40, 20)
        vs = [
            verdict("val-0", VerdictDecision.ACCEPT, 10),
            verdict("val-1", VerdictDecision.REJECT, 11),
        ]
        status = evaluate_quorum(DIGEST, vs, evs, CFG)
        assert status.accept_stake == 40
        assert status.reject_stake == 40
        assert not status.reached

    def test_verdicts_for_other_challenger_ignored(self):
        evs = evaluators(60, 20, 20)
        vs = [verdict("val-0", VerdictDecision.ACCEPT, 10, digest=OTHER_DIGEST)]
        status = evaluate_quorum(DIGEST, vs, evs, CFG)
        assert status.accept_stake == 0
        assert not status.reached


class TestLatestVerdictSupersedes:
    def test_rerun_supersedes_own_earlier_verdict(self):
        evs = evaluators(60, 20, 20)
        vs = [
            verdict("val-0", VerdictDecision.REJECT, 10),
            verdict("val-0", VerdictDecision.ACCEPT, 20),  # re-run supersedes
        ]
        status = evaluate_quorum(DIGEST, vs, evs, CFG)
        assert status.accept_stake == 60
        assert status.reject_stake == 0
        assert status.reached

    def test_earlier_verdict_does_not_supersede_later(self):
        evs = evaluators(60, 20, 20)
        vs = [
            verdict("val-0", VerdictDecision.ACCEPT, 20),
            verdict("val-0", VerdictDecision.REJECT, 10),  # older, arrives later in list
        ]
        status = evaluate_quorum(DIGEST, vs, evs, CFG)
        assert status.accept_stake == 60
        assert status.reached


class TestNonEvaluatorStake:
    def test_non_evaluator_verdict_ignored(self):
        evs = evaluators(50, 25, 25)
        vs = [
            verdict("val-0", VerdictDecision.ACCEPT, 10),
            verdict("outsider", VerdictDecision.ACCEPT, 11),  # not in evaluator set
        ]
        status = evaluate_quorum(DIGEST, vs, evs, CFG)
        assert status.accept_stake == 50
        assert status.active_stake == 100
        assert not status.reached


class TestBootstrapMode:
    def test_single_accept_crowns_below_min_evaluators(self):
        evs = evaluators(1.0, 1.0)  # 2 < bootstrap_min_evaluators=3
        vs = [verdict("val-0", VerdictDecision.ACCEPT, 10)]
        status = evaluate_quorum(DIGEST, vs, evs, CFG)
        assert status.bootstrap_mode
        assert status.reached

    def test_bootstrap_reject_alone_does_not_crown(self):
        evs = evaluators(1.0)
        vs = [verdict("val-0", VerdictDecision.REJECT, 10)]
        status = evaluate_quorum(DIGEST, vs, evs, CFG)
        assert status.bootstrap_mode
        assert not status.reached

    def test_exits_bootstrap_at_min_evaluators(self):
        evs = evaluators(1.0, 1.0, 1.0)
        status = evaluate_quorum(DIGEST, [], evs, CFG)
        assert not status.bootstrap_mode


class TestDeriveCoronation:
    def test_crossing_block_is_the_verdict_that_crossed(self):
        evs = evaluators(30, 30, 40)
        vs = [
            verdict("val-0", VerdictDecision.ACCEPT, 10),  # 30 < 51
            verdict("val-1", VerdictDecision.ACCEPT, 15),  # 60 >= 51  <- crossing
            verdict("val-2", VerdictDecision.ACCEPT, 25),  # after quorum
        ]
        event = derive_coronation(DIGEST, vs, evs, CFG, reveal_block=5, current_block=100)
        assert event is not None
        assert event.block == 15
        assert event.accept_stake == 100
        assert event.active_stake == 100

    def test_pending_returns_none(self):
        evs = evaluators(30, 40, 30)
        vs = [verdict("val-0", VerdictDecision.ACCEPT, 10)]
        assert derive_coronation(DIGEST, vs, evs, CFG, reveal_block=5, current_block=100) is None

    def test_timeout_is_judged_on_the_crossing_block_not_the_clock(self):
        """A coronation cannot expire just because the reader asked late.

        Judging the timeout against ``current_block`` made the answer depend on
        when a validator happened to poll: one that ticked inside the window
        crowned the challenger while one that was restarting got None for the
        same chain data, and the two boxes then disagreed about the king
        permanently. The crossing block is a property of the verdicts, so every
        observer gets the same answer whenever it asks.
        """
        evs = evaluators(60, 20, 20)  # normal (non-bootstrap) mode
        reveal = 5
        vs = [verdict("val-0", VerdictDecision.ACCEPT, reveal + 10)]

        event = derive_coronation(DIGEST, vs, evs, CFG, reveal_block=reveal, current_block=50)
        assert event is not None and event.block == reveal + 10
        # Asked long after the window closed: same answer, same block.
        very_late = reveal + CFG.verdict_timeout_blocks * 100
        late_event = derive_coronation(
            DIGEST, vs, evs, CFG, reveal_block=reveal, current_block=very_late
        )
        assert late_event == event

    def test_quorum_arriving_after_the_window_does_not_crown(self):
        evs = evaluators(60, 20, 20)
        reveal = 5
        crossed_late = [
            verdict("val-0", VerdictDecision.ACCEPT, reveal + CFG.verdict_timeout_blocks + 1)
        ]
        assert derive_coronation(
            DIGEST, crossed_late, evs, CFG, reveal_block=reveal, current_block=10**9
        ) is None

        # Exactly on the boundary still crowns.
        at_limit = [
            verdict("val-0", VerdictDecision.ACCEPT, reveal + CFG.verdict_timeout_blocks)
        ]
        assert derive_coronation(
            DIGEST, at_limit, evs, CFG, reveal_block=reveal, current_block=10**9
        ) is not None

    def test_withdrawn_accept_does_not_date_the_coronation(self):
        """A validator that accepted then re-ran into a REJECT has no live
        ACCEPT, so its withdrawn verdict must not supply the crossing block."""
        evs = evaluators(60, 40, 10)  # three evaluators: normal, not bootstrap
        vs = [
            verdict("val-0", VerdictDecision.ACCEPT, 10),
            verdict("val-0", VerdictDecision.REJECT, 20),
            verdict("val-1", VerdictDecision.ACCEPT, 30),
        ]
        # val-1 alone holds 40 of 110 stake, below theta: nothing crowns, and in
        # particular val-0's withdrawn block-10 ACCEPT does not date one.
        assert derive_coronation(DIGEST, vs, evs, CFG, reveal_block=5, current_block=100) is None

    def test_bootstrap_crossing_block_is_first_accept(self):
        evs = evaluators(1.0)
        vs = [
            verdict("val-0", VerdictDecision.REJECT, 10),
            verdict("val-0", VerdictDecision.ACCEPT, 12),
        ]
        event = derive_coronation(DIGEST, vs, evs, CFG, reveal_block=5, current_block=50)
        assert event is not None
        assert event.block == 12

    def test_dissent_recorded_in_event_verdicts(self):
        evs = evaluators(60, 20, 20)
        vs = [
            verdict("val-1", VerdictDecision.REJECT, 9),   # dissenter
            verdict("val-0", VerdictDecision.ACCEPT, 12),  # crosses 60 >= 51
        ]
        event = derive_coronation(DIGEST, vs, evs, CFG, reveal_block=5, current_block=100)
        assert event is not None
        assert event.block == 12
        decisions = {(v.validator_hotkey, v.decision) for v in event.verdicts}
        assert ("val-1", VerdictDecision.REJECT) in decisions
        assert ("val-0", VerdictDecision.ACCEPT) in decisions
