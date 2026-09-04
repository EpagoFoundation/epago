"""Coronation as a pure function of chain state.

Given the set of on-chain verdict commitments and validator stakes, every
honest party derives the same king at the same block. No validator can crown
or block alone; weight-setting follows the derived king mechanically, so a
validator that disagrees with quorum still converges (its dissent stays on
record in its own verdict commitment).
"""

from __future__ import annotations

from dataclasses import dataclass

from epago.config import QuorumSection
from epago.core.types import CoronationEvent, EvaluatorInfo, Verdict, VerdictDecision


def _stake_sum(stake_by_hotkey: dict[str, float], hotkeys) -> float:
    """Float summation in sorted-hotkey order.

    Different validators read commitments in different orders; summing in a
    canonical order guarantees every observer computes the identical stake
    total, so a theta-boundary comparison can never split the network.
    """
    return sum(stake_by_hotkey[h] for h in sorted(hotkeys))


@dataclass(frozen=True, slots=True)
class QuorumStatus:
    challenger_digest: str
    accept_stake: float
    reject_stake: float
    active_stake: float
    theta: float
    reached: bool
    bootstrap_mode: bool


def active_evaluators(
    evaluators: list[EvaluatorInfo],
    current_block: int,
    active_window_blocks: int,
) -> list[EvaluatorInfo]:
    """Evaluators that posted any verdict within the active window.

    Stake with no recent verdicts is a follower, not an evaluator: it neither
    counts toward nor blocks quorum.
    """
    return [e for e in evaluators if current_block - e.last_verdict_block <= active_window_blocks]


def evaluate_quorum(
    challenger_digest: str,
    verdicts: list[Verdict],
    evaluators: list[EvaluatorInfo],
    cfg: QuorumSection,
) -> QuorumStatus:
    stake_by_hotkey = {e.hotkey: e.stake for e in evaluators}
    active_stake = _stake_sum(stake_by_hotkey, stake_by_hotkey)
    bootstrap_mode = len(evaluators) < cfg.bootstrap_min_evaluators

    # Latest verdict per validator wins; a validator re-running a duel
    # supersedes its own earlier commitment.
    latest: dict[str, Verdict] = {}
    for v in verdicts:
        if v.challenger_digest != challenger_digest:
            continue
        if v.validator_hotkey not in stake_by_hotkey:
            continue
        prev = latest.get(v.validator_hotkey)
        if prev is None or v.block > prev.block:
            latest[v.validator_hotkey] = v

    accept_stake = _stake_sum(
        stake_by_hotkey, (h for h, v in latest.items() if v.decision is VerdictDecision.ACCEPT)
    )
    reject_stake = _stake_sum(
        stake_by_hotkey, (h for h, v in latest.items() if v.decision is VerdictDecision.REJECT)
    )

    if bootstrap_mode:
        # Explicitly-labeled degraded mode for genesis / low-evaluator regimes:
        # a single ACCEPT crowns. Exits automatically once evaluator count rises.
        reached = any(v.decision is VerdictDecision.ACCEPT for v in latest.values())
    else:
        reached = active_stake > 0 and accept_stake >= cfg.theta * active_stake

    return QuorumStatus(
        challenger_digest=challenger_digest,
        accept_stake=accept_stake,
        reject_stake=reject_stake,
        active_stake=active_stake,
        theta=cfg.theta,
        reached=reached,
        bootstrap_mode=bootstrap_mode,
    )


def derive_coronation(
    challenger_digest: str,
    verdicts: list[Verdict],
    evaluators: list[EvaluatorInfo],
    cfg: QuorumSection,
    reveal_block: int,
    current_block: int = 0,
) -> CoronationEvent | None:
    """Return the coronation event, or None if quorum was never reached in time.

    The coronation block is the block of the verdict that pushed accept-stake
    over theta, making the event's ordering unambiguous across observers.

    The timeout is judged against that **crossing block**, never against
    ``current_block``. Comparing to the caller's clock made the answer depend on
    *when* it asked: a validator that polled during the window crowned the
    challenger, while one that was restarting or busy past
    ``reveal_block + verdict_timeout_blocks`` got ``None`` for the very same
    chain data and permanently lapsed a king the rest of the subnet had already
    adopted. ``current_block`` is retained only so callers can keep passing it.
    """
    status = evaluate_quorum(challenger_digest, verdicts, evaluators, cfg)
    if not status.reached:
        return None
    # Total order: block ties broken by hotkey so every observer derives the
    # same crossing verdict regardless of the order it read commitments in.
    relevant = sorted(
        (v for v in verdicts if v.challenger_digest == challenger_digest),
        key=lambda v: (v.block, v.validator_hotkey),
    )
    crossing_block = _crossing_block(relevant, evaluators, cfg, status.bootstrap_mode)
    if crossing_block is None:
        return None
    if crossing_block - reveal_block > cfg.verdict_timeout_blocks:
        return None  # quorum arrived, but only after the challenge had lapsed
    return CoronationEvent(
        challenger_digest=challenger_digest,
        block=crossing_block,
        accept_stake=status.accept_stake,
        active_stake=status.active_stake,
        verdicts=tuple(relevant),
    )


def lapsed(reveal_block: int, current_block: int, cfg: QuorumSection) -> bool:
    """True once a challenge with no coronation can no longer get one."""
    return current_block - reveal_block > cfg.verdict_timeout_blocks


def _crossing_block(
    ordered_verdicts: list[Verdict],
    evaluators: list[EvaluatorInfo],
    cfg: QuorumSection,
    bootstrap_mode: bool,
) -> int | None:
    """The block at which accept-stake first crossed theta, or None.

    Returning None rather than the last verdict's block matters: a fabricated
    crossing block would be compared against the timeout and could crown a
    challenger that never actually reached quorum.
    """
    stake_by_hotkey = {e.hotkey: e.stake for e in evaluators}
    active_stake = _stake_sum(stake_by_hotkey, stake_by_hotkey)
    # The superseded prefix cannot crown: a validator that accepted and then
    # re-ran the duel into a REJECT has no live ACCEPT, so replaying from its
    # withdrawn verdict would date the coronation to a block where quorum did
    # not actually hold.
    final: dict[str, Verdict] = {}
    for v in ordered_verdicts:
        if v.validator_hotkey in stake_by_hotkey:
            final[v.validator_hotkey] = v

    latest: dict[str, Verdict] = {}
    for v in ordered_verdicts:
        if v.validator_hotkey not in stake_by_hotkey:
            continue
        if final[v.validator_hotkey] is not v:
            continue  # superseded by a later verdict from the same validator
        latest[v.validator_hotkey] = v
        if bootstrap_mode:
            if v.decision is VerdictDecision.ACCEPT:
                return v.block
            continue
        accept = _stake_sum(
            stake_by_hotkey, (h for h, lv in latest.items() if lv.decision is VerdictDecision.ACCEPT)
        )
        if active_stake > 0 and accept >= cfg.theta * active_stake:
            return v.block
    return None
