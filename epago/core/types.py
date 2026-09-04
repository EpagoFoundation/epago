"""Shared value types.

Every subsystem codes against these types; nothing here imports from other
Epago modules or third-party ML libraries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HF_DIGEST_RE = re.compile(r"^hf:[0-9a-f]{40}$")
SS58_RE = re.compile(r"^5[1-9A-HJ-NP-Za-km-z]{47}$")


def is_valid_digest(digest: str) -> bool:
    return bool(SHA256_DIGEST_RE.match(digest) or HF_DIGEST_RE.match(digest))


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A content-addressed model snapshot: the digest is the binding commitment."""

    repo: str
    digest: str

    def __post_init__(self) -> None:
        if not is_valid_digest(self.digest):
            raise ValueError(f"invalid digest: {self.digest!r}")

    @property
    def backend(self) -> str:
        return "hf" if self.digest.startswith("hf:") else "oci"


class SubmissionStatus(StrEnum):
    QUEUED = "queued"
    STALE_PARENT = "stale_parent"
    FAILED_INTAKE = "failed_intake"
    FAILED_PROBES = "failed_probes"
    DUEL_LOST = "duel_lost"
    NEAR_MISS = "near_miss"
    ACCEPTED = "accepted"
    LAPSED = "lapsed"


@dataclass(frozen=True, slots=True)
class ChallengeReveal:
    """A parsed on-chain challenge reveal (format ``e1``)."""

    king_digest: str
    challenger: ModelRef
    author_hotkey: str
    reveal_block: int
    block_hash_at_reveal: str


class TaskOrigin(StrEnum):
    GENERATED_PUBLIC = "generated_public"
    GENERATED_PRIVATE = "generated_private"


@dataclass(frozen=True, slots=True)
class Task:
    """A single research task with a corpus-verifiable answer.

    ``masked_doc_ids`` are removed from search and browse for this task's
    rollouts so the answer must be re-derived, not looked up at its origin.
    """

    task_id: str
    question: str
    answer: str
    aliases: tuple[str, ...]
    evidence_doc_ids: tuple[str, ...]
    masked_doc_ids: tuple[str, ...]
    origin: TaskOrigin
    template: str
    hops: int = 1


@dataclass(frozen=True, slots=True)
class RolloutResult:
    task_id: str
    answer: str | None
    correct: bool
    turns: int
    wall_time_s: float
    judge_tier: str          # "exact" | "alias" | "judge" | "none"
    error: str | None = None
    #: Generations in this episode that carried no parseable action. Protocol
    #: compliance is a property of the model's OUTPUT, so it is counted per
    #: generation; whether the episode also found the answer is a separate
    #: question (``correct``) and a much harder one. The format probe grades on
    #: this field precisely so it cannot be failed by a hard task.
    malformed_actions: int = 0


@dataclass(frozen=True, slots=True)
class DuelHalf:
    """Paired outcome on one holdout half."""

    n_tasks: int
    diffs: tuple[int, ...]              # d_i = chall_i - king_i, each in {-1, 0, +1}
    mu_hat: float
    king_acc: float
    challenger_acc: float


@dataclass(frozen=True, slots=True)
class DuelOutcome:
    public: DuelHalf
    private: DuelHalf
    lcb_pub: float
    delta: float
    accepted: bool
    boot_seed_hex: str
    public_seed_hex: str
    # Per-task paired differences, (task_id, d_i), in scored order. This is what
    # makes an audit record's LCB independently recomputable.
    public_task_results: tuple[tuple[str, int], ...] = ()
    # Judge-tier usage across both models' rollouts, e.g. (("exact", 380), ...).
    judge_tier_counts: tuple[tuple[str, int], ...] = ()

    @property
    def judge_invocation_rate(self) -> float:
        total = sum(n for _, n in self.judge_tier_counts)
        judged = sum(n for tier, n in self.judge_tier_counts if tier == "judge")
        return judged / total if total else 0.0


@dataclass(frozen=True, slots=True)
class DuelSpec:
    """Everything one paired duel needs, in raw chain-derived form.

    The eval subsystem derives seeds and the adaptive floor from these inputs
    itself (via ``core.stats``) so a spec can never carry a seed or threshold
    that disagrees with the public derivation rules.
    """

    king_dir: Path
    challenger_dir: Path
    public_tasks: list[Task]
    private_tasks: list[Task]
    block_hash_at_reveal: str
    author_hotkey: str
    king_acc_ema: float
    noise_floor: float
    round_id: str = ""


@dataclass(frozen=True, slots=True)
class Entrant:
    """One challenger in a round's field."""

    digest: str
    repo: str
    author_hotkey: str
    challenger_dir: Path


@dataclass(frozen=True, slots=True)
class RoundDuelSpec:
    """One competition round: every entrant against the king on one exam.

    The exam is minted once per round from the round-start block hash, so the
    field is compared on identical questions rather than on a per-submission
    draw. That removes exam luck from the comparison between rivals — under a
    per-submission exam two challengers of equal skill could be separated purely
    by which questions each happened to draw.

    The king answers the exam exactly once and its results are reused for every
    pairing, which is also what keeps a large field affordable: N entrants cost
    N+1 sweeps, not 2N.
    """

    king_dir: Path
    entrants: tuple[Entrant, ...]
    public_tasks: list[Task]
    private_tasks: list[Task]
    round: int
    round_block_hash: str
    king_acc_ema: float
    noise_floor: float


@dataclass(frozen=True, slots=True)
class RoundResult:
    """One entrant's outcome within a round."""

    entrant: Entrant
    outcome: DuelOutcome


class VerdictDecision(StrEnum):
    ACCEPT = "A"
    REJECT = "R"


@dataclass(frozen=True, slots=True)
class Verdict:
    """A validator's signed duel verdict, committed on-chain (format ``ev3``)."""

    challenger_digest: str
    decision: VerdictDecision
    lcb_pub_e6: int                     # lcb_pub * 1e6, signed integer
    mu_priv_e6: int
    delta_e6: int                       # the adaptive floor this duel was judged against
    round: int                          # the competition round that produced it
    private_pool_epoch: int
    audit_digest: str                   # first 16 hex of the audit record digest
    validator_hotkey: str = ""
    block: int = 0

    @property
    def lcb_pub(self) -> float:
        return self.lcb_pub_e6 / 1e6

    @property
    def mu_priv(self) -> float:
        return self.mu_priv_e6 / 1e6

    @property
    def delta(self) -> float:
        return self.delta_e6 / 1e6

    @property
    def is_near_miss(self) -> bool:
        """Rejected, but measurably better than the king on both halves.

        Two kinds of rejection qualify, and both are honest attempts the arena
        pool exists to fund:

        * ``0 < lcb <= delta`` — probably better, not provably;
        * ``lcb > delta`` — provably better, but another entrant in the same
          round was better still. A round crowns one winner, so a genuine
          improver can be rejected purely for placing second; charging it a
          cooldown for that would punish exactly the behaviour the subnet wants.

        ``mu_priv > 0`` is what separates both from the overfit case: a
        challenger that clears the public floor while losing the private half
        was tuned to the generator, and that is a plain loss.

        Chain-derivable, because ``delta``, ``lcb`` and ``mu_priv`` all ride on
        the wire — which is what lets a validator that ran no duel compute the
        same arena split as the one that did.
        """
        return (
            self.decision is VerdictDecision.REJECT
            and self.lcb_pub_e6 > 0
            and self.mu_priv_e6 > 0
        )


@dataclass(frozen=True, slots=True)
class RoundStart:
    """A competition round opened by the round authority (format ``er1``).

    The chain stamps the reveal block, and the block hash there mints the
    round's exam. That ordering is what keeps the exam honest: the trigger is
    published *after* every entrant's weights are already committed, so nobody —
    including the authority — can know the questions while a checkpoint can
    still be changed. Entrants are exactly the challengers revealed strictly
    before ``block``; anything revealed later would have seen this hash and
    waits for the next round.
    """

    round: int
    authority_hotkey: str = ""
    block: int = 0
    block_hash: str = ""


@dataclass(frozen=True, slots=True)
class KingPointer:
    """The on-chain king record (format ``ek1``).

    Carries every field :func:`epago.core.emissions.compute_weights` needs, so a
    validator booting with no state directory adopts the live king and the live
    reign clock instead of falling back to the genesis seed.
    """

    repo: str
    digest: str
    author_hotkey: str
    crowned_block: int
    reign_started_block: int
    coronation_lcb_e6: int
    coronation_delta_e6: int
    publisher_hotkey: str = ""
    block: int = 0

    @property
    def ref(self) -> ModelRef:
        return ModelRef(repo=self.repo, digest=self.digest)

    @property
    def coronation_lcb(self) -> float:
        return self.coronation_lcb_e6 / 1e6

    @property
    def coronation_delta(self) -> float:
        return self.coronation_delta_e6 / 1e6


@dataclass(frozen=True, slots=True)
class KingState:
    ref: ModelRef
    author_hotkey: str
    crowned_block: int
    reign_started_block: int            # inherited on self-dethrone, reset otherwise
    acc_ema: float
    coronation_lcb: float
    # The floor the crowning duel cleared. Lives on the king (not beside it) so
    # the coronation bonus is computed from the same record the ``ek1`` pointer
    # publishes, and every validator reads one consistent pair.
    coronation_delta: float = 0.0


@dataclass(slots=True)
class EvaluatorInfo:
    hotkey: str
    stake: float
    last_verdict_block: int = 0


@dataclass(frozen=True, slots=True)
class CoronationEvent:
    challenger_digest: str
    block: int
    accept_stake: float
    active_stake: float
    verdicts: tuple[Verdict, ...]


@dataclass(slots=True)
class AuditRecord:
    """One duel's replayable verdict record. Serialized as canonical JSON."""

    round_id: str
    block_hash_at_reveal: str
    author_hotkey: str
    king_repo: str
    king_digest: str
    challenger_repo: str
    challenger_digest: str
    corpus_digest: str
    taskgen_release: str
    public_seed: str
    public_task_ids_digest: str
    private_pool_digest: str
    private_pool_epoch: int
    n_private_tasks: int
    boot_seed: str
    king_acc_ema: float
    delta_threshold: float
    mu_hat_pub: float
    lcb_pub: float
    mu_hat_priv: float
    accepted: bool
    harness_digest: str
    judge_model_digest: str
    eval_code_digest: str
    judge_invocation_rate: float
    revealed_at_block: int
    intake_at_block: int
    verdict_at_block: int
    validator_hotkey: str
    #: The sealed public pool this exam was drawn from, when the release names
    #: one. Recorded per verdict rather than read from the contract at replay
    #: time: a contract can be edited, and an auditor must be able to pin the
    #: pool that *this* duel actually used. Empty for generator-served releases.
    public_pool_digest: str = ""
    #: The task-id manifest the round's selection is reproducible from. Pinned
    #: in the record as well as the contract so a replay needs nothing but the
    #: verdict to know which id list to check against.
    public_pool_manifest_digest: str = ""
    validator_signature: str = ""
    extra: dict = field(default_factory=dict)
