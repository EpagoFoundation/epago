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
4. attempts            — the author hotkey has used its allowance of rounds
                         (``constants.MAX_ATTEMPTS_PER_HOTKEY``).
5. failure memory      — digests that already failed deterministically never
                         re-enter the queue.
6. repo validation     — repo pattern plus hotkey-prefix anti-impersonation.

Identity is the **hotkey** — the registered neuron on the metagraph, the same
key that carries UID, weights, and emissions. No coldkey resolution is done.

Spam discipline is an **attempt allowance**, not time and not money. A hotkey
may put a model into at most ``MAX_ATTEMPTS_PER_HOTKEY`` rounds, one model per
round, each a different model. An attempt is used when a round takes the model
into its field, whatever happens to it there; a reveal refused here, or replaced
by the same hotkey's later reveal before any round took it, uses none. Counting
starts at ``ATTEMPTS_FROM_ROUND``, so every hotkey starts that round with the
full allowance. A near-miss may re-enter the same model once on fresh tasks, and
that re-entry is one of its attempts.

The allowance replaced two earlier rules: one submission per hotkey forever,
which made a model the new king had made stale cost its author a fresh
registration, and an escalating cooldown after each loss, which at one round a
day routinely kept a hotkey out of the very next round.
"""

from __future__ import annotations

from dataclasses import dataclass

from epago import constants
from epago.chain.client import ChainClient
from epago.config import EpagoConfig
from epago.core.types import SubmissionStatus
from epago.chain.mailbox import submission_prefix
from epago.model.validation import validate_repo_name
from epago.validator.state import QueuedSubmission, ValidatorState

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


def _near_miss_retry_open(state, digest: str, reveal_block: int) -> bool:
    """True when a near-miss may re-enter on this reveal (``NEAR_MISS_RETRIES``).

    The retry must arrive as a NEW reveal, strictly after the near-miss
    verdict: a newer reveal block means a new public seed and therefore new
    tasks. Re-processing the original reveal would deterministically replay
    the identical holdout — a wasted duel, not a retry — so it stays ignored.
    Nothing is consumed here: the retry is only spent once the re-entry is
    actually queued, so a re-reveal refused for another reason keeps it.
    """
    meta = state.near_misses.get(digest)
    if not meta:
        return False
    if int(meta.get("retries", 0)) >= constants.NEAR_MISS_RETRIES:
        return False
    return reveal_block > int(meta.get("verdict_block", 0))


@dataclass(frozen=True, slots=True)
class IntakeResult:
    author_hotkey: str
    repo: str
    digest: str
    status: SubmissionStatus
    code: str
    detail: str = ""


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
        # Already refused as stale and still naming the same old king: the reveal
        # cannot become current again, so it is not re-logged on every tick. A
        # fresh reveal of the same model against the current king goes through.
        if status == SubmissionStatus.STALE_PARENT.value and reveal.king_digest != current_king_digest:
            continue
        near_miss_retry = False
        if status in _TERMINAL_STATUSES:
            if status == SubmissionStatus.NEAR_MISS.value and _near_miss_retry_open(
                state, digest, reveal.reveal_block
            ):
                near_miss_retry = True  # fresh reveal, fresh seed: the near-miss re-duel right
            else:
                continue  # permanently resolved: each attempt is a different model

        # The attempt allowance. A hotkey may put a model into at most
        # MAX_ATTEMPTS_PER_HOTKEY rounds; the round that takes the model charges
        # the attempt (see ValidatorService._run_round), so this only has to
        # refuse a hotkey that has none left. The allowance never refills, so
        # the refusal is final for this reveal and is recorded like one — which
        # also keeps it from being re-logged on every tick. A near-miss re-entry
        # is checked here too: it is one of the attempts.
        used = state.attempts_used(hotkey)
        if used >= constants.MAX_ATTEMPTS_PER_HOTKEY:
            state.seen_digests.setdefault(digest, hotkey)
            if not near_miss_retry:
                state.statuses[digest] = SubmissionStatus.FAILED_INTAKE.value
            reject(
                reveal,
                SubmissionStatus.FAILED_INTAKE,
                "attempts_exhausted",
                f"hotkey {hotkey} has used all {constants.MAX_ATTEMPTS_PER_HOTKEY} attempts",
            )
            continue

        # A contract that takes private submissions only refuses a public one:
        # its weights were readable by every rival the moment it was revealed.
        # A property of the reveal itself, so it is remembered like one.
        if cfg.chain.private_submissions_only and reveal.challenger.backend == "hf":
            detail = "this subnet takes private submissions only; upload with `epago miner upload`"
            state.seen_digests.setdefault(digest, hotkey)
            state.statuses[digest] = SubmissionStatus.FAILED_INTAKE.value
            state.record_failure(digest, "public_submission", detail, current_block)
            reject(reveal, SubmissionStatus.FAILED_INTAKE, "public_submission", detail)
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

        # One model per hotkey per round, and the latest counts: a newer
        # admitted reveal replaces whatever the hotkey still has waiting. A model
        # already taken into the running round stays there — that round has it —
        # and the new one waits for the next. Replaced models used no attempt.
        in_round = {
            e.get("digest") for e in (state.round_in_progress or {}).get("entrants", [])
        }
        for old in [
            q for q in state.queue
            if q.author_hotkey == hotkey and q.digest != digest and q.digest not in in_round
        ]:
            state.queue.remove(old)
            queued_digests.discard(old.digest)
            state.statuses[old.digest] = SubmissionStatus.SUPERSEDED.value
            results.append(IntakeResult(
                author_hotkey=hotkey,
                repo=old.repo,
                digest=old.digest,
                status=SubmissionStatus.SUPERSEDED,
                code="superseded",
                detail=f"replaced by {digest}",
            ))
            state.log_intake(hotkey, old.digest, "superseded", f"replaced by {digest}", current_block)

        if near_miss_retry:
            state.near_misses[digest]["retries"] = int(state.near_misses[digest].get("retries", 0)) + 1
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
        state.statuses[digest] = SubmissionStatus.QUEUED.value
        res = IntakeResult(
            author_hotkey=hotkey,
            repo=reveal.challenger.repo,
            digest=digest,
            status=SubmissionStatus.QUEUED,
            code="queued",
            detail=f"attempts used {used} of {constants.MAX_ATTEMPTS_PER_HOTKEY}",
        )
        results.append(res)
        state.log_intake(hotkey, digest, "queued", res.detail, current_block)

    state.last_scan_block = current_block + 1
    return results
