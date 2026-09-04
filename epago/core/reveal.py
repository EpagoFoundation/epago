"""Wire formats for on-chain strings.

Four formats live on the timelock-reveal channel:

* ``e2`` challenge reveals, submitted by miners::

      e2|<king_digest>|<challenger_repo>|<challenger_digest>

  The author is **not** on the wire. It is the hotkey that signed the
  commitment, which the chain attests and :meth:`ChainClient.read_revealed_submissions`
  supplies. The predecessor format ``e1`` carried a self-declared author field
  that nothing verified; because intake keys ownership, cooldowns and emission
  attribution off that field, any hotkey could submit a deliberately losing
  checkpoint under a rival's identity and put the rival on an escalating
  cooldown. ``e1`` payloads no longer parse.

* ``ev3`` verdict commitments, published by validators after a duel::

      ev3|<challenger_digest>|<A|R>|<lcb_pub_e6>|<mu_priv_e6>|<delta_e6>|<round>|<priv_epoch>|<audit16>

  ``delta_e6`` carries the adaptive floor. It is per-validator (it depends on
  that box's measured noise floor and king-accuracy EMA), so without it on the
  wire nobody else could tell an accept from a near-miss, and the arena split
  was not reproducible from chain state. ``round`` attributes the verdict to the
  competition that produced it.

* ``er1`` round starts, published by the round authority::

      er1|<round>

  Opens a competition. Nothing is evaluated until one lands, and only the
  configured authority hotkey can publish one.

* ``ek1`` king pointers, published by the coronation authority::

      ek1|<repo>|<digest>|<author_hotkey>|<crowned_block>|<reign_started_block>|<lcb_e6>|<delta_e6>

  Everything :func:`epago.core.emissions.compute_weights` needs about the king,
  so a validator with an empty state directory can adopt the current king and
  compute the same weight vector as everyone else.

* ``ep1`` private-pool commitments (built in :mod:`epago.validator.wiring`).

All are versioned; payloads with unknown versions are dropped at intake with a
one-time warning, never errors.
"""

from __future__ import annotations

import re

from epago import constants
from epago.core.types import (
    ChallengeReveal,
    KingPointer,
    ModelRef,
    RoundStart,
    Verdict,
    VerdictDecision,
    is_valid_digest,
)

_AUDIT16_RE = re.compile(r"^[0-9a-f]{16}$")
#: Canonical signed decimal. ``int()`` alone accepts underscores, surrounding
#: whitespace, unicode digits and redundant signs, so two different payloads
#: would parse to the same verdict — an avoidable malleability in a string that
#: quorum ordering depends on.
_INT_RE = re.compile(r"^-?(0|[1-9][0-9]*)$")


class WireFormatError(ValueError):
    """The string does not parse as the expected on-chain format."""


def _parse_int(raw: str, field: str) -> int:
    if not _INT_RE.match(raw):
        raise WireFormatError(f"{field} must be a canonical decimal integer, got {raw!r}")
    return int(raw)


def _require_digest(digest: str, field: str) -> str:
    if not is_valid_digest(digest):
        raise WireFormatError(f"invalid {field}: {digest!r}")
    return digest


def _require_no_separator(value: str, field: str) -> str:
    """Reject embedded ``|`` so a field can never forge extra fields."""
    if "|" in value or not value:
        raise WireFormatError(f"invalid {field}: {value!r}")
    return value


# --- e2: challenge reveal ----------------------------------------------------


def build_reveal(king_digest: str, challenger: ModelRef) -> str:
    _require_digest(king_digest, "king digest")
    _require_no_separator(challenger.repo, "challenger repo")
    return "|".join(
        (constants.REVEAL_VERSION, king_digest, challenger.repo, challenger.digest)
    )


def parse_reveal(
    raw: str, reveal_block: int, block_hash_at_reveal: str, author_hotkey: str
) -> ChallengeReveal:
    """Parse an ``e2`` payload.

    ``author_hotkey`` is supplied by the caller from the chain's own record of
    who signed the commitment — it is never read out of ``raw``.
    """
    parts = raw.split("|")
    if len(parts) != 4 or parts[0] != constants.REVEAL_VERSION:
        raise WireFormatError(f"not a {constants.REVEAL_VERSION} reveal: {raw!r}")
    _, king_digest, repo, digest = parts
    _require_digest(king_digest, "king digest")
    _require_digest(digest, "challenger digest")
    _require_no_separator(repo, "challenger repo")
    return ChallengeReveal(
        king_digest=king_digest,
        challenger=ModelRef(repo=repo, digest=digest),
        author_hotkey=author_hotkey,
        reveal_block=reveal_block,
        block_hash_at_reveal=block_hash_at_reveal,
    )


# --- ev3: verdict commitment -------------------------------------------------


def build_verdict(v: Verdict) -> str:
    if not _AUDIT16_RE.match(v.audit_digest):
        raise WireFormatError(f"audit digest must be 16 lowercase hex chars: {v.audit_digest!r}")
    _require_digest(v.challenger_digest, "challenger digest")
    return "|".join(
        (
            constants.VERDICT_VERSION,
            v.challenger_digest,
            v.decision.value,
            str(v.lcb_pub_e6),
            str(v.mu_priv_e6),
            str(v.delta_e6),
            str(v.round),
            str(v.private_pool_epoch),
            v.audit_digest,
        )
    )


def parse_verdict(raw: str, validator_hotkey: str, block: int) -> Verdict:
    parts = raw.split("|")
    if len(parts) != 9 or parts[0] != constants.VERDICT_VERSION:
        raise WireFormatError(f"not a {constants.VERDICT_VERSION} verdict: {raw!r}")
    _, digest, decision, lcb_e6, mu_e6, delta_e6, rnd, epoch, audit16 = parts
    _require_digest(digest, "challenger digest")
    if decision not in ("A", "R"):
        raise WireFormatError(f"invalid decision: {decision!r}")
    if not _AUDIT16_RE.match(audit16):
        raise WireFormatError(f"invalid audit digest: {audit16!r}")
    delta = _parse_int(delta_e6, "delta_e6")
    if delta < 0:
        raise WireFormatError(f"delta_e6 must be non-negative, got {delta}")
    pool_epoch = _parse_int(epoch, "private_pool_epoch")
    if pool_epoch < 0:
        raise WireFormatError(f"private_pool_epoch must be non-negative, got {pool_epoch}")
    round_no = _parse_int(rnd, "round")
    if round_no < 1:
        raise WireFormatError(f"round must be >= 1, got {round_no}")
    return Verdict(
        challenger_digest=digest,
        decision=VerdictDecision(decision),
        lcb_pub_e6=_parse_int(lcb_e6, "lcb_pub_e6"),
        mu_priv_e6=_parse_int(mu_e6, "mu_priv_e6"),
        delta_e6=delta,
        round=round_no,
        private_pool_epoch=pool_epoch,
        audit_digest=audit16,
        validator_hotkey=validator_hotkey,
        block=block,
    )


# --- ek1: king pointer -------------------------------------------------------


def build_king_pointer(p: KingPointer) -> str:
    _require_digest(p.digest, "king digest")
    _require_no_separator(p.repo, "king repo")
    return "|".join(
        (
            constants.KING_POINTER_VERSION,
            p.repo,
            p.digest,
            p.author_hotkey,
            str(p.crowned_block),
            str(p.reign_started_block),
            str(p.coronation_lcb_e6),
            str(p.coronation_delta_e6),
        )
    )


def parse_king_pointer(raw: str, publisher_hotkey: str, block: int) -> KingPointer:
    parts = raw.split("|")
    if len(parts) != 8 or parts[0] != constants.KING_POINTER_VERSION:
        raise WireFormatError(f"not a {constants.KING_POINTER_VERSION} pointer: {raw!r}")
    _, repo, digest, author, crowned, reign_started, lcb_e6, delta_e6 = parts
    _require_digest(digest, "king digest")
    _require_no_separator(repo, "king repo")
    crowned_block = _parse_int(crowned, "crowned_block")
    reign_block = _parse_int(reign_started, "reign_started_block")
    if crowned_block < 0 or reign_block < 0:
        raise WireFormatError("king pointer blocks must be non-negative")
    if reign_block > crowned_block:
        # A reign cannot start after the coronation that reported it; inherited
        # reigns move the clock backwards, never forwards.
        raise WireFormatError(
            f"reign_started_block {reign_block} is after crowned_block {crowned_block}"
        )
    delta = _parse_int(delta_e6, "coronation_delta_e6")
    if delta < 0:
        raise WireFormatError(f"coronation_delta_e6 must be non-negative, got {delta}")
    return KingPointer(
        repo=repo,
        digest=digest,
        author_hotkey=author,
        crowned_block=crowned_block,
        reign_started_block=reign_block,
        coronation_lcb_e6=_parse_int(lcb_e6, "coronation_lcb_e6"),
        coronation_delta_e6=delta,
        publisher_hotkey=publisher_hotkey,
        block=block,
    )


# --- er1: round start --------------------------------------------------------


def build_round_start(round_no: int) -> str:
    if round_no < 1:
        raise WireFormatError(f"round must be >= 1, got {round_no}")
    return "|".join((constants.ROUND_START_VERSION, str(round_no)))


def parse_round_start(raw: str, authority_hotkey: str, block: int, block_hash: str) -> RoundStart:
    """Parse an ``er1`` payload.

    The round number is the only field: the block and its hash come from the
    chain's own stamp, so the authority cannot backdate a round or choose the
    entropy that mints its exam.
    """
    parts = raw.split("|")
    if len(parts) != 2 or parts[0] != constants.ROUND_START_VERSION:
        raise WireFormatError(f"not a {constants.ROUND_START_VERSION} round start: {raw!r}")
    round_no = _parse_int(parts[1], "round")
    if round_no < 1:
        raise WireFormatError(f"round must be >= 1, got {round_no}")
    return RoundStart(
        round=round_no,
        authority_hotkey=authority_hotkey,
        block=block,
        block_hash=block_hash,
    )
