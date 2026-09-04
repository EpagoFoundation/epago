"""Submission admission: cheap checks first, zero GPU below the probe layer.

Every rejection here costs the validator microseconds and is remembered where
deterministic, so a hostile flood of reveals degrades into cheap bookkeeping
rather than burned duel-hours. The admission ladder, cheapest first:

1. duplicate digest    — on-chain reveal ordering adjudicates ownership: the
                         first reveal of a digest owns it, later reveals of the
                         same digest by other hotkeys reject as ``duplicate_digest``.
2. stale parent        — the reveal's king digest no longer matches the crowned
                         king. After a dethrone, every in-flight challenger
                         built on the old king drops at once.
3. self-challenge      — the sitting king's author cannot duel itself for free.
4. cooldown            — the author hotkey is serving a cooldown from an
                         earlier decisive loss or probe failure.
5. failure memory      — digests that already failed deterministically never
                         re-enter the queue.
6. repo validation     — repo pattern plus hotkey-prefix anti-impersonation.

Identity is the **hotkey** — the registered neuron on the metagraph, the same
key that carries UID, weights, and emissions. No coldkey resolution is done.

Spam discipline is time, not money. Earlier designs sketched an "advisory
bond" that never actually moved stake — a fiction this module no longer
pretends to enforce. The real, fully enforceable penalty is a **cooldown**:
after a decisive loss (public LCB below ``constants.BOND_BURN_LCB_THRESHOLD``)
or a probe failure, the author *hotkey* is barred from intake for
:data:`COOLDOWN_BLOCKS`, doubling on repeat offenses within
:data:`COOLDOWN_MEMORY_BLOCKS` up to :data:`COOLDOWN_MAX_BLOCKS`, and scaling
with queue pressure (:func:`queue_pressure_scale`). Near-misses and honest
losses (LCB at or above the threshold) incur no cooldown: good-faith attempts
stay free. Enforcement is a machine-readable ``cooldown`` rejection at intake
with the expiry block, so a barred miner needs no human to tell it when to
come back.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from epago import constants
from epago.chain.client import ChainClient
from epago.config import EpagoConfig
from epago.core.types import SubmissionStatus
from epago.chain.mailbox import submission_prefix
from epago.model.validation import validate_repo_name
from epago.validator.state import QueuedSubmission, ValidatorState

#: Planning estimate for one full duel (materialize + probes + both halves).
#: Measured end to end on one RTX 5090: 0.71 GPU-h for 800 episodes plus a
#: 179s admission probe and a 56s engine load. Held at 1.5 for margin on slower
#: cards and larger exams. The queue breaker multiplies this by queue depth, so
#: an inflated estimate throttles honest miners long before the box is busy.
EST_DUEL_HOURS: float = 1.5


def _env(name: str, default: int) -> int:
    raw = os.environ.get(f"EPAGO_{name}")
    return default if raw is None else int(raw)


#: Base cooldown after one decisive loss or probe failure: ~24h of blocks.
COOLDOWN_BLOCKS: int = _env("COOLDOWN_BLOCKS", 7_200)
#: Strike memory: decisive losses this close together escalate the cooldown.
COOLDOWN_MEMORY_BLOCKS: int = _env("COOLDOWN_MEMORY_BLOCKS", 100_800)
#: Hard bound on a single cooldown, escalation and queue pressure included.
COOLDOWN_MAX_BLOCKS: int = _env("COOLDOWN_MAX_BLOCKS", 43_200)

#: Statuses that permanently resolve a digest; reveals of such digests are ignored.
_TERMINAL_STATUSES = frozenset(
    s.value
    for s in (
        SubmissionStatus.ACCEPTED,
        SubmissionStatus.NEAR_MISS,
        SubmissionStatus.DUEL_LOST,
        SubmissionStatus.FAILED_PROBES,
        SubmissionStatus.FAILED_INTAKE,
    )
)


def validate_submission_prefix(ref, hotkey: str) -> tuple[str, str] | None:
    """A private (``sha256:``) submission must sit in its author's own prefix.

    Returns ``(code, detail)`` on failure, or ``None`` when the ref is fine.

    Public ``hf:`` refs are unaffected: they live in a repository the author
    owns, which :func:`validate_repo_name` already checks.

    For private uploads the prefix *is* the ownership boundary. The credential
    the validator sealed to this hotkey can only write under
    ``submissions/<hotkey>/``, so a ref pointing elsewhere cannot have been
    written by its claimed author. Rejecting it here is what stops one miner
    revealing a rival's upload as its own — the digest could not, because a
    digest is computable by anyone who can read the bytes.
    """
    if ref.backend != "oci":
        return None
    expected = submission_prefix(hotkey)
    repo = str(ref.repo)
    if not repo.startswith(expected):
        return (
            "wrong_prefix",
            f"a private submission from {hotkey} must live under {expected}",
        )
    return None


def validate_sealable_hotkey(ref, hotkey: str) -> tuple[str, str] | None:
    """A private submission requires a hotkey credentials can be sealed to.

    Returns ``(code, detail)`` on failure, ``None`` when the hotkey is usable.

    Public ``hf:`` submissions are unaffected: they need no credential, so the
    curve does not matter. This is deliberately checked at intake rather than
    left to the mailbox, because the mailbox failure mode is *silence* — a
    miner with an sr25519 hotkey simply never receives an envelope, and has
    nowhere to learn why. Failing the submission says it plainly, once, with
    the command that fixes it.
    """
    if ref.backend != "oci":
        return None
    from epago.chain.envelope import EnvelopeError, x25519_public_from_ed25519

    try:
        from scalecodec.utils.ss58 import ss58_decode

        x25519_public_from_ed25519(bytes.fromhex(ss58_decode(hotkey)))
    except EnvelopeError:
        return (
            "hotkey_not_sealable",
            f"{hotkey} is not an Ed25519 hotkey, so upload credentials cannot be "
            "encrypted to it. Create one with "
            "`btcli wallet new-hotkey --key-type ed25519`, or submit publicly "
            "with an hf: reference instead.",
        )
    except Exception:  # noqa: BLE001 - a decode problem is not a curve verdict
        return None
    return None


def _consume_near_miss_retry(state, digest: str, reveal_block: int) -> bool:
    """One fresh-holdout re-duel per near-miss (``NEAR_MISS_RETRIES``).

    The retry must arrive as a NEW reveal, strictly after the near-miss
    verdict: a newer reveal block means a new public seed and therefore new
    tasks. Re-processing the original reveal would deterministically replay
    the identical holdout — a wasted duel, not a retry — so it stays ignored.
    """
    meta = state.near_misses.get(digest)
    if not meta:
        return False
    if int(meta.get("retries", 0)) >= constants.NEAR_MISS_RETRIES:
        return False
    if reveal_block <= int(meta.get("verdict_block", 0)):
        return False
    meta["retries"] = int(meta.get("retries", 0)) + 1
    return True


@dataclass(frozen=True, slots=True)
class IntakeResult:
    author_hotkey: str
    repo: str
    digest: str
    status: SubmissionStatus
    code: str
    detail: str = ""


def queue_pressure_scale(queue_depth: int, est_duel_hours: float = EST_DUEL_HOURS) -> float:
    """Queue circuit breaker: cooldown multiplier from projected verdict latency.

    Published formula (deterministic, so miners can compute it themselves)::

        est_latency_hours = (queue_depth + 1) * est_duel_hours
        steps             = ceil(max(est_latency_hours - QUEUE_BREAKER_HOURS, 0)
                                 / QUEUE_BREAKER_HOURS)
        scale             = 2 ** steps

    While the projected latency stays within ``constants.QUEUE_BREAKER_HOURS``
    the scale is 1; beyond that it doubles for every additional breaker-width
    of backlog. Bad-faith spam that would blow the SLA earns proportionally
    longer cooldowns instead of requiring anyone to close the queue.
    """
    est_latency_hours = (queue_depth + 1) * est_duel_hours
    over = est_latency_hours - constants.QUEUE_BREAKER_HOURS
    steps = math.ceil(over / constants.QUEUE_BREAKER_HOURS) if over > 0 else 0
    return 2.0**steps


def cooldown_triggered(status: SubmissionStatus, probes_failed: bool = False) -> bool:
    """True when a resolved submission should cool its author hotkey down.

    Everything that is neither an acceptance nor a near-miss pays. The older
    rule only fired below ``constants.BOND_BURN_LCB_THRESHOLD`` (-0.05), which
    left the entire band ``-0.05 <= lcb <= 0`` free — and that is exactly where
    a noise-perturbed copy of the king lands. Those submissions cost nothing to
    make and nothing to lose: not byte-identical, so the exact-copy gate misses
    them; far below the norm-sanity ratio, so the probes miss them; and roughly
    half draw a positive LCB, which books near-miss credit against the arena
    pool. Repeating that was a free draw on both the arena budget and the
    1-in-1000 false-acceptance tail. A loss now costs ~24h of intake silence,
    doubling on repeat, which is what makes the per-duel significance bar mean
    something across many attempts.

    Near-misses stay free on purpose: they are the honest-attempt case the arena
    pool exists to fund, and their retry allowance is already bounded.
    """
    if probes_failed:
        return True
    return status not in (SubmissionStatus.ACCEPTED, SubmissionStatus.NEAR_MISS)


def decisive_loss(lcb_pub: float) -> bool:
    """True for a submission that was never plausibly an improvement.

    Distinct from :func:`cooldown_triggered`: this reports *how bad* an attempt
    was (it drives the intake record and dashboards), while the other decides
    whether to charge for it at all.
    """
    return lcb_pub < constants.BOND_BURN_LCB_THRESHOLD


def cooldown_duration(strikes: int, queue_depth: int = 0) -> int:
    """Cooldown length in blocks for the Nth strike under current queue load.

    Published formula::

        duration = min(COOLDOWN_BLOCKS * 2 ** (strikes - 1)
                       * queue_pressure_scale(queue_depth),
                       COOLDOWN_MAX_BLOCKS)
    """
    base = COOLDOWN_BLOCKS * (2 ** max(strikes - 1, 0))
    return int(min(base * queue_pressure_scale(queue_depth), COOLDOWN_MAX_BLOCKS))


def apply_cooldown(
    state: ValidatorState, hotkey: str, block: int, queue_depth: int = 0
) -> dict[str, Any]:
    """Record a decisive-loss strike against ``hotkey`` and start its cooldown.

    Strikes escalate while they land within :data:`COOLDOWN_MEMORY_BLOCKS` of
    the previous one and reset to 1 otherwise. The entry is persisted in
    ``state.cooldowns`` and returned for logging.
    """
    prev = state.cooldowns.get(hotkey)
    if prev is not None and block - int(prev.get("last_strike_block", 0)) <= COOLDOWN_MEMORY_BLOCKS:
        strikes = int(prev.get("strikes", 0)) + 1
    else:
        strikes = 1
    entry = {
        "until_block": block + cooldown_duration(strikes, queue_depth),
        "strikes": strikes,
        "last_strike_block": block,
    }
    state.cooldowns[hotkey] = entry
    return entry


def cooldown_until(state: ValidatorState, hotkey: str, current_block: int) -> int | None:
    """The block a live cooldown on ``hotkey`` expires at, or None if free."""
    entry = state.cooldowns.get(hotkey)
    if entry is None:
        return None
    until = int(entry["until_block"])
    return until if until > current_block else None


def scan_and_enqueue(
    chain: ChainClient,
    state: ValidatorState,
    cfg: EpagoConfig,
    current_king_digest: str,
) -> list[IntakeResult]:
    """Read revealed submissions and admit the valid ones into the duel queue.

    Reveals are processed in on-chain order — ``(reveal_block, hotkey)`` with
    the hotkey as a deterministic tie-break — so every validator resolves
    digest ownership identically.
    """
    results: list[IntakeResult] = []
    current_block = chain.current_block()
    registered_hotkeys = {n.hotkey for n in chain.neurons()}
    king_author = state.king.author_hotkey if state.king is not None else ""
    queued_digests = {q.digest for q in state.queue}

    # Full history, deliberately: "latest reveal per hotkey wins" has to be
    # resolved over every reveal, not over the slice since the last poll.
    # Scanning a window made supersession depend on tick cadence, so a validator
    # that had just restarted and one that had been ticking all along admitted
    # different challenges from identical chain state. Re-reading is cheap
    # because seen_digests, statuses and the queue make every repeat a no-op.
    reveals = sorted(
        chain.read_revealed_submissions(0),
        key=lambda r: (r.reveal_block, r.author_hotkey),
    )

    def reject(reveal, status: SubmissionStatus, code: str, detail: str = "") -> None:
        res = IntakeResult(
            author_hotkey=reveal.author_hotkey,
            repo=reveal.challenger.repo,
            digest=reveal.challenger.digest,
            status=status,
            code=code,
            detail=detail,
        )
        results.append(res)
        state.log_intake(res.author_hotkey, res.digest, code, detail, current_block)

    for reveal in reveals:
        digest = reveal.challenger.digest
        hotkey = reveal.author_hotkey

        owner = state.seen_digests.get(digest)
        if owner is not None and owner != hotkey:
            reject(
                reveal,
                SubmissionStatus.FAILED_INTAKE,
                "duplicate_digest",
                f"digest first revealed by {owner}; reveal ordering owns it",
            )
            continue
        if digest == current_king_digest:
            continue  # already crowned; nothing to duel
        if digest in queued_digests:
            continue
        status = state.statuses.get(digest)
        if status in _TERMINAL_STATUSES:
            if status == SubmissionStatus.NEAR_MISS.value and _consume_near_miss_retry(
                state, digest, reveal.reveal_block
            ):
                pass  # fresh reveal, fresh seed: the near-miss re-duel right
            else:
                continue  # permanently resolved

        # One submission per hotkey, permanently. A hotkey that has already put
        # a model forward is spent, whatever happened to it: to try again a
        # miner registers a fresh hotkey and pays the registration burn.
        #
        # The burn is what this rule buys. Without it an attempt is free, so
        # the cheapest strategy is to submit many mediocre checkpoints and let
        # the duels find one that got lucky on its holdout -- every one of
        # which costs validators a full rollout sweep. Priced attempts make a
        # miner spend the compute on training instead of on lottery tickets.
        #
        # Checked after the terminal-status branch so a near-miss can still
        # take its one re-duel: that is the same submission being re-judged on
        # fresh tasks, not a second one.
        spent = state.spent_hotkeys.get(hotkey)
        if spent is not None and spent != digest:
            reject(
                reveal,
                SubmissionStatus.FAILED_INTAKE,
                "hotkey_spent",
                f"hotkey {hotkey} already submitted {spent[:16]}; one per hotkey",
            )
            continue

        if reveal.king_digest != current_king_digest:
            state.seen_digests.setdefault(digest, hotkey)
            state.statuses[digest] = SubmissionStatus.STALE_PARENT.value
            reject(
                reveal,
                SubmissionStatus.STALE_PARENT,
                "stale_parent",
                f"built on {reveal.king_digest}, king is {current_king_digest}",
            )
            continue

        if king_author and hotkey == king_author:
            reject(
                reveal,
                SubmissionStatus.FAILED_INTAKE,
                "self_challenge",
                "author already holds the crown",
            )
            continue

        # Cooldown gate: hotkey-level (the registered neuron identity). Not a
        # deterministic property of the checkpoint -> no failure memory; the
        # miner may re-reveal after the expiry block in the rejection detail.
        until = cooldown_until(state, hotkey, current_block)
        if until is not None:
            reject(
                reveal,
                SubmissionStatus.FAILED_INTAKE,
                "cooldown",
                f"hotkey {hotkey} in cooldown until block {until}",
            )
            continue

        if digest in state.failure_memory:
            memo = state.failure_memory[digest]
            reject(
                reveal,
                SubmissionStatus.FAILED_INTAKE,
                "failure_memory",
                f"previously failed: {memo.get('code', '')}",
            )
            continue

        if hotkey not in registered_hotkeys:
            # Not a deterministic property of the checkpoint — no failure memory.
            reject(reveal, SubmissionStatus.FAILED_INTAKE, "unknown_hotkey", "not registered")
            continue

        # A private submission must land in the author's own prefix. The
        # credential issued to that hotkey can only write there, so a ref
        # pointing anywhere else is either a mistake or an attempt to claim
        # another miner's upload — and the digest alone would not catch the
        # second, since a miner can compute the digest of bytes it can see.
        # A private upload needs a hotkey the validator can seal credentials
        # to. sr25519 signs but has no encryption, so such a miner would never
        # receive a readable envelope — and without this it would see only
        # silence, with nothing anywhere explaining it.
        curve_failure = validate_sealable_hotkey(reveal.challenger, hotkey)
        if curve_failure is not None:
            reject(reveal, SubmissionStatus.FAILED_INTAKE, *curve_failure)
            continue

        prefix_failure = validate_submission_prefix(reveal.challenger, hotkey)
        if prefix_failure is not None:
            state.seen_digests.setdefault(digest, hotkey)
            state.statuses[digest] = SubmissionStatus.FAILED_INTAKE.value
            state.record_failure(digest, prefix_failure[0], prefix_failure[1], current_block)
            reject(reveal, SubmissionStatus.FAILED_INTAKE, *prefix_failure)
            continue

        # The repo-name rule is the ownership check for a PUBLIC submission:
        # `owner/EPAGO-...`, where the owner must carry the author's hotkey
        # prefix. A private upload has no repository and no owner field — its
        # ownership boundary is the key prefix checked above, which is enforced
        # by the credential itself rather than by a naming convention. Applying
        # the repo pattern to a key prefix would reject every private
        # submission for having one slash too many.
        name_failure = (
            validate_repo_name(reveal.challenger.repo, hotkey, cfg)
            if reveal.challenger.backend == "hf"
            else None
        )
        if name_failure is not None:
            state.seen_digests.setdefault(digest, hotkey)
            state.statuses[digest] = SubmissionStatus.FAILED_INTAKE.value
            state.record_failure(digest, name_failure.code, name_failure.detail, current_block)
            reject(reveal, SubmissionStatus.FAILED_INTAKE, name_failure.code, name_failure.detail)
            continue

        sub = QueuedSubmission(
            repo=reveal.challenger.repo,
            digest=digest,
            king_digest=reveal.king_digest,
            author_hotkey=hotkey,
            reveal_block=reveal.reveal_block,
            block_hash_at_reveal=reveal.block_hash_at_reveal,
            enqueued_block=current_block,
        )
        state.enqueue(sub)
        queued_digests.add(digest)
        state.seen_digests[digest] = hotkey
        # Spent at the moment the submission reaches the queue, not at the
        # verdict. Waiting for a verdict would let a hotkey hold several
        # in-flight submissions at once, which is the flooding this rule exists
        # to price. Earlier would be worse: a malformed payload or an
        # unregistered hotkey would burn a registration over a typo, punishing
        # honest error rather than gaming.
        state.spent_hotkeys.setdefault(hotkey, digest)
        state.statuses[digest] = SubmissionStatus.QUEUED.value
        res = IntakeResult(
            author_hotkey=hotkey,
            repo=reveal.challenger.repo,
            digest=digest,
            status=SubmissionStatus.QUEUED,
            code="queued",
            detail=f"queue_scale={queue_pressure_scale(len(state.queue) - 1)}",
        )
        results.append(res)
        state.log_intake(hotkey, digest, "queued", res.detail, current_block)

    state.last_scan_block = current_block + 1
    return results
