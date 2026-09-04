"""Durable validator state.

All mechanism-relevant validator memory lives in one JSON document under the
state directory, written atomically (temp file + ``os.replace``). A validator
process can be killed at any block and restarted with no operator action: the
queue, failure memory, king, counters, and SLA records all survive, and every
transition is re-derivable from chain data plus the audit log.

Nothing in this module talks to the chain or the GPU; it is pure bookkeeping
so the orchestration loop in :mod:`epago.validator.service` stays testable.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from epago.core.emissions import ArenaEntry, inherits_reign, trim_arena
from epago.core.types import KingState, ModelRef, SubmissionStatus

_STATE_FILE = "state.json"

NOISE_SAMPLE_CAP = 20    # capped-deque semantics: only the freshest samples count
INTAKE_LOG_CAP = 500


@dataclass(slots=True)
class QueuedSubmission:
    """One admitted challenge waiting for its duel."""

    repo: str
    digest: str
    king_digest: str
    author_hotkey: str
    reveal_block: int
    block_hash_at_reveal: str
    enqueued_block: int


_QUEUED_FIELDS = frozenset(f.name for f in fields(QueuedSubmission))


def _king_to_dict(king: KingState | None) -> dict[str, Any] | None:
    if king is None:
        return None
    return {
        "repo": king.ref.repo,
        "digest": king.ref.digest,
        "author_hotkey": king.author_hotkey,
        "crowned_block": king.crowned_block,
        "reign_started_block": king.reign_started_block,
        "acc_ema": king.acc_ema,
        "coronation_lcb": king.coronation_lcb,
        "coronation_delta": king.coronation_delta,
    }


def _king_from_dict(data: dict[str, Any] | None) -> KingState | None:
    if data is None:
        return None
    return KingState(
        ref=ModelRef(repo=data["repo"], digest=data["digest"]),
        author_hotkey=data["author_hotkey"],
        crowned_block=int(data["crowned_block"]),
        reign_started_block=int(data["reign_started_block"]),
        acc_ema=float(data["acc_ema"]),
        coronation_lcb=float(data["coronation_lcb"]),
        # Older state files kept the floor in a sibling field; default rather
        # than fail so an existing validator restarts without intervention.
        coronation_delta=float(data.get("coronation_delta", 0.0)),
    )


class ValidatorState:
    """The validator's durable memory.

    Spam discipline is hotkey **cooldowns** (see
    :mod:`epago.validator.intake`), persisted in :attr:`cooldowns` as
    ``{hotkey: {until_block, strikes, last_strike_block}}`` so an escalation
    ladder survives restarts. :attr:`burned_bonds` is a retired mechanism —
    the "advisory bond" was never an extrinsic and is no longer accounted; the
    attribute stays (always empty for new state) only so status tooling that
    prints it keeps working against old state files.
    """

    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.king: KingState | None = None
        self.king_acc_ema: float = 0.5
        self.king_coronation_delta: float = 0.0
        self.queue: list[QueuedSubmission] = []
        self.failure_memory: dict[str, dict[str, Any]] = {}   # digest -> {code, detail, block}
        self.seen_digests: dict[str, str] = {}                # digest -> owner hotkey (first reveal)
        # hotkey -> the one digest it ever put forward. A hotkey gets a single
        # submission, permanently: to try again a miner registers a new one and
        # pays the registration burn. That burn is the whole point -- it prices
        # every attempt, so flooding the queue with speculative checkpoints
        # costs real TAO instead of being free.
        self.spent_hotkeys: dict[str, str] = {}
        # Credential mailbox: when it was last published, and the digest of
        # what went out. Persisted so a restart does not immediately reissue
        # every miner's credentials, which would invalidate an upload already
        # in flight.
        self.last_mailbox_block: int | None = None
        self.mailbox_digest: str = ""
        # Weight fingerprint -> the digest that first dueled with those bytes.
        # Digest identity is a revision hash, so the same weights re-uploaded
        # (or re-sharded) get a fresh digest and would re-enter the one-duel-
        # per-digest rule as new content. The fingerprint is content identity:
        # once weights have dueled, every later digest carrying them is a
        # resubmission, whoever reveals it.
        self.seen_fingerprints: dict[str, str] = {}
        self.statuses: dict[str, str] = {}                    # digest -> SubmissionStatus value
        self.near_misses: dict[str, dict] = {}                # digest -> {verdict_block, retries}
        self.candidates: dict[str, dict[str, Any]] = {}       # digest -> coronation metadata
        self.noise_floor_samples: list[float] = []
        self.clean_duels: int = 0
        self.organic_dethrones: int = 0
        self.genesis_block: int = 0
        self.arena: list[ArenaEntry] = []
        self.burned_bonds: dict[str, float] = {}              # retired; kept for status output
        self.cooldowns: dict[str, dict[str, Any]] = {}        # hotkey -> {until_block, strikes, ...}
        # A round commits one ev3 per entrant, and the commitment pallet
        # rate-limits writes per hotkey, so deferred verdicts are a queue and
        # not a single slot: a dropped verdict never coronates.
        self.pending_verdicts: list[str] = []
        self.pending_king_pointer: str | None = None          # ek1 awaiting a re-commit
        self.current_round: int = 0
        self.last_round_run: int = 0
        self.last_round_block: int = 0
        self.sla: list[dict[str, Any]] = []
        self.intake_log: list[dict[str, Any]] = []
        self.last_scan_block: int = 0
        self.difficulty: dict[str, Any] = {}                  # DifficultyController snapshot
        self.last_weights_block: int | None = None
        self.last_pool_publish_block: int = 0
        self.committed_pool_epoch: int = -1   # ep1 landed for this private-pool epoch
        # Public sealed-pool task ids this validator has already asked and
        # published. Rounds draw from the unserved remainder, so a task that has
        # been published in full — and is therefore training data from that
        # moment on — is never asked again. Persisted because losing it would
        # silently re-serve published tasks after a restart, which is exactly
        # the leak the exclusion exists to prevent.
        self.served_public_task_ids: list[str] = []
        self.anchor_history: list[dict[str, Any]] = []    # external-anchor runs (observational)
        self.last_anchor_block: int = 0
        self.tick_count: int = 0
        self.pending_mirror: bool = False
        self.last_error: dict[str, Any] | None = None

    # ---- persistence ---------------------------------------------------

    @classmethod
    def load(cls, state_dir: str | Path) -> "ValidatorState":
        state = cls(state_dir)
        state.state_dir.mkdir(parents=True, exist_ok=True)
        path = state.state_dir / _STATE_FILE
        if not path.exists():
            return state
        data = json.loads(path.read_text())
        state.king = _king_from_dict(data.get("king"))
        state.king_acc_ema = float(data.get("king_acc_ema", 0.5))
        state.king_coronation_delta = float(data.get("king_coronation_delta", 0.0))
        # Filter unknown keys so old state files (e.g. with the retired
        # ``bond`` field) still load without operator intervention.
        state.queue = [
            QueuedSubmission(**{k: v for k, v in q.items() if k in _QUEUED_FIELDS})
            for q in data.get("queue", [])
        ]
        state.failure_memory = dict(data.get("failure_memory", {}))
        state.seen_digests = dict(data.get("seen_digests", {}))
        state.spent_hotkeys = dict(data.get("spent_hotkeys", {}))
        state.last_mailbox_block = data.get("last_mailbox_block")
        state.mailbox_digest = str(data.get("mailbox_digest", ""))
        state.seen_fingerprints = dict(data.get("seen_fingerprints", {}))
        state.statuses = dict(data.get("statuses", {}))
        state.near_misses = {k: dict(v) for k, v in data.get("near_misses", {}).items()}
        state.candidates = dict(data.get("candidates", {}))
        state.noise_floor_samples = [float(x) for x in data.get("noise_floor_samples", [])]
        state.clean_duels = int(data.get("clean_duels", 0))
        state.organic_dethrones = int(data.get("organic_dethrones", 0))
        state.genesis_block = int(data.get("genesis_block", 0))
        state.arena = [ArenaEntry(**a) for a in data.get("arena", [])]
        state.burned_bonds = {h: float(v) for h, v in data.get("burned_bonds", {}).items()}
        state.cooldowns = dict(data.get("cooldowns", {}))
        pending = data.get("pending_verdicts")
        if pending is None:  # older state files carried a single slot
            legacy = data.get("pending_verdict")
            pending = [legacy] if legacy else []
        state.pending_verdicts = list(pending)
        state.pending_king_pointer = data.get("pending_king_pointer")
        state.current_round = int(data.get("current_round", 0))
        state.last_round_run = int(data.get("last_round_run", 0))
        state.last_round_block = int(data.get("last_round_block", 0))
        state.sla = list(data.get("sla", []))
        state.intake_log = list(data.get("intake_log", []))
        state.last_scan_block = int(data.get("last_scan_block", 0))
        state.difficulty = dict(data.get("difficulty", {}))
        lwb = data.get("last_weights_block")
        state.last_weights_block = int(lwb) if lwb is not None else None
        state.last_pool_publish_block = int(data.get("last_pool_publish_block", 0))
        state.served_public_task_ids = [str(i) for i in data.get("served_public_task_ids", [])]
        state.committed_pool_epoch = int(data.get("committed_pool_epoch", -1))
        state.anchor_history = list(data.get("anchor_history", []))
        state.last_anchor_block = int(data.get("last_anchor_block", 0))
        state.tick_count = int(data.get("tick_count", 0))
        state.pending_mirror = bool(data.get("pending_mirror", False))
        state.last_error = data.get("last_error")
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "king": _king_to_dict(self.king),
            "king_acc_ema": self.king_acc_ema,
            "king_coronation_delta": self.king_coronation_delta,
            "queue": [asdict(q) for q in self.queue],
            "failure_memory": self.failure_memory,
            "seen_digests": self.seen_digests,
            "spent_hotkeys": self.spent_hotkeys,
            "last_mailbox_block": self.last_mailbox_block,
            "mailbox_digest": self.mailbox_digest,
            "seen_fingerprints": self.seen_fingerprints,
            "statuses": self.statuses,
            # Persisted because ``load`` reads it back: the near-miss retry
            # ledger bounds how many fresh-holdout re-duels one digest may
            # claim (NEAR_MISS_RETRIES). Leaving it out of the snapshot reset
            # every miner's allowance on each validator restart, so the bound
            # was only as strong as the process uptime.
            "near_misses": self.near_misses,
            "candidates": self.candidates,
            "noise_floor_samples": self.noise_floor_samples,
            "clean_duels": self.clean_duels,
            "organic_dethrones": self.organic_dethrones,
            "genesis_block": self.genesis_block,
            "arena": [asdict(a) for a in self.arena],
            "burned_bonds": self.burned_bonds,
            "cooldowns": self.cooldowns,
            "pending_verdicts": self.pending_verdicts,
            "pending_king_pointer": self.pending_king_pointer,
            "current_round": self.current_round,
            "last_round_run": self.last_round_run,
            "last_round_block": self.last_round_block,
            "sla": self.sla,
            "intake_log": self.intake_log,
            "last_scan_block": self.last_scan_block,
            "difficulty": self.difficulty,
            "last_weights_block": self.last_weights_block,
            "last_pool_publish_block": self.last_pool_publish_block,
            "served_public_task_ids": sorted(self.served_public_task_ids),
            "committed_pool_epoch": self.committed_pool_epoch,
            "anchor_history": self.anchor_history,
            "last_anchor_block": self.last_anchor_block,
            "tick_count": self.tick_count,
            "pending_mirror": self.pending_mirror,
            "last_error": self.last_error,
        }

    def save(self) -> None:
        """Atomic write: temp file in the same directory, then rename."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / _STATE_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=1))
        os.replace(tmp, path)

    # ---- queue ----------------------------------------------------------

    def enqueue(self, sub: QueuedSubmission) -> None:
        self.queue.append(sub)

    def requeue_front(self, sub: QueuedSubmission) -> None:
        """Put a submission back at the head after a transient failure."""
        self.queue.insert(0, sub)

    # ---- memory & records ------------------------------------------------

    def queue_pending_verdict(self, payload: str) -> None:
        """Defer a verdict the chain would not accept this window."""
        if payload not in self.pending_verdicts:
            self.pending_verdicts.append(payload)

    def record_failure(self, digest: str, code: str, detail: str, block: int = 0) -> None:
        """Deterministic failures are remembered forever: the same digest can
        never re-enter the queue and burn compute twice."""
        self.failure_memory[digest] = {"code": code, "detail": detail, "block": block}

    def add_noise_sample(self, sample: float) -> None:
        self.noise_floor_samples.append(float(sample))
        if len(self.noise_floor_samples) > NOISE_SAMPLE_CAP:
            self.noise_floor_samples = self.noise_floor_samples[-NOISE_SAMPLE_CAP:]

    def log_intake(self, hotkey: str, digest: str, code: str, detail: str, block: int) -> None:
        self.intake_log.append(
            {"hotkey": hotkey, "digest": digest, "code": code, "detail": detail, "block": block}
        )
        if len(self.intake_log) > INTAKE_LOG_CAP:
            self.intake_log = self.intake_log[-INTAKE_LOG_CAP:]

    def record_sla(
        self, digest: str, revealed_at: int, intake_at: int, verdict_at: int | None
    ) -> None:
        self.sla.append(
            {
                "digest": digest,
                "revealed_at": revealed_at,
                "intake_at": intake_at,
                "verdict_at": verdict_at,
            }
        )

    def record_verdict(
        self,
        *,
        digest: str,
        hotkey: str,
        status: SubmissionStatus,
        lcb_pub: float,
        verdict_block: int,
    ) -> None:
        """Record a duel resolution: status and the near-miss retry right.

        A near-miss no longer earns arena credit. The arena is the roster of
        former kings (:func:`epago.core.emissions.trim_arena`), which is what
        makes "burn until something has been crowned" true at genesis: if
        near-misses seated themselves there, the arena could be paying before
        any king had ever existed.

        What a near-miss still earns is the right to try again on fresh tasks.
        Decisive-loss penalties are cooldowns, applied by the service through
        :func:`epago.validator.intake.apply_cooldown`, not recorded here.
        """
        self.statuses[digest] = status.value
        if status is SubmissionStatus.NEAR_MISS and lcb_pub > 0:
            # Near-miss retry ledger: a fresh reveal (fresh seed, fresh tasks)
            # may re-enter the queue up to NEAR_MISS_RETRIES times; the retry
            # count survives its own re-duel so the right is bounded.
            prior = self.near_misses.get(digest, {})
            self.near_misses[digest] = {
                "verdict_block": verdict_block,
                "retries": int(prior.get("retries", 0)),
            }

    # ---- king ------------------------------------------------------------

    def set_king(
        self,
        ref: ModelRef,
        author_hotkey: str,
        crowned_block: int,
        coronation_lcb: float,
        coronation_delta: float = 0.0,
        acc_ema: float | None = None,
    ) -> KingState:
        """Install a new king, computing reign inheritance.

        A self-dethrone (same hotkey, per :func:`epago.core.emissions.inherits_reign`)
        keeps the previous ``reign_started_block`` so salami-slicing improvements
        buys no fresh reign clock; a different hotkey resets it and counts as
        an organic dethrone toward Phase B activation.
        """
        prev = self.king
        if prev is not None and inherits_reign(prev.author_hotkey, author_hotkey):
            reign_started = prev.reign_started_block
        else:
            reign_started = crowned_block
            if prev is not None:
                self.organic_dethrones += 1
                # The displaced king steps back into the arena and keeps
                # earning while the roster holds it. A self-dethrone is not a
                # displacement -- the same hotkey is still king, and seating it
                # in its own arena would pay one party twice out of a budget
                # that is supposed to reward having been beaten.
                self.arena.append(
                    ArenaEntry(
                        hotkey=prev.author_hotkey,
                        dethroned_block=crowned_block,
                        reign_blocks=max(crowned_block - prev.reign_started_block, 0),
                    )
                )
                self.arena = trim_arena(self.arena)
        self.king = KingState(
            ref=ref,
            author_hotkey=author_hotkey,
            crowned_block=crowned_block,
            reign_started_block=reign_started,
            acc_ema=self.king_acc_ema if acc_ema is None else acc_ema,
            coronation_lcb=coronation_lcb,
            coronation_delta=coronation_delta,
        )
        self.king_coronation_delta = coronation_delta
        return self.king

    def adopt_king(
        self,
        ref: ModelRef,
        author_hotkey: str,
        crowned_block: int,
        reign_started_block: int,
        coronation_lcb: float,
        coronation_delta: float,
    ) -> KingState:
        """Install a king announced on-chain, reign clock included.

        Distinct from :meth:`set_king`, which *derives* the reign clock from a
        coronation this validator observed. An auditing validator did not
        observe one, so it takes the authority's published clock verbatim —
        recomputing it locally would restart the reign at adoption time and
        hand the incumbent a fresh decay curve on every validator restart.
        Adoption is not a dethrone this box witnessed, so it does not count
        toward ``organic_dethrones``.
        """
        self.king = KingState(
            ref=ref,
            author_hotkey=author_hotkey,
            crowned_block=crowned_block,
            reign_started_block=reign_started_block,
            acc_ema=self.king_acc_ema,
            coronation_lcb=coronation_lcb,
            coronation_delta=coronation_delta,
        )
        self.king_coronation_delta = coronation_delta
        return self.king

    def mirror_king(self, king_dir: Path, mirror_dir: Path) -> Path:
        """Copy the crowned snapshot into validator-controlled storage.

        King-loss is impossible by construction: at every coronation the full
        snapshot is mirrored under the validator's own state directory, so a
        miner deleting or corrupting their upstream repo can never orphan the
        subnet's baseline. Future duels, calibration runs, and re-coronations
        after a challenger flood always have a locally owned copy of the king
        to evaluate against — there is no failure mode that requires a human
        to re-seed the subnet.
        """
        if self.king is None:
            raise RuntimeError("no king to mirror")
        mirror_dir = Path(mirror_dir)
        mirror_dir.mkdir(parents=True, exist_ok=True)
        target = mirror_dir / self.king.ref.digest.replace(":", "_")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(king_dir, target)
        self.pending_mirror = False
        return target


# ---- difficulty controller persistence ---------------------------------------
#
# ``epago.taskgen.difficulty.DifficultyController`` holds per-template EMAs in
# private dicts and has no (de)serialization of its own; that module is owned
# by taskgen and stays untouched, so the EMA snapshot logic lives here. The
# accessors below reach into ``_solve``/``_judge``/``_disc``/``_retired``
# deliberately — the snapshot is (value, count) pairs per template plus the
# retired set, which is the controller's entire durable state.


def difficulty_to_dict(controller: Any) -> dict[str, Any]:
    """Snapshot a ``DifficultyController`` into a JSON-safe dict."""

    def dump(store: dict[str, Any]) -> dict[str, list]:
        return {t: [ema.value, ema.count] for t, ema in store.items()}

    return {
        "solve": dump(controller._solve),
        "judge": dump(controller._judge),
        "disc": dump(controller._disc),
        "retired": sorted(controller._retired),
    }


def difficulty_from_dict(controller: Any, data: dict[str, Any]) -> Any:
    """Restore a snapshot into a fresh ``DifficultyController`` (returned).

    Each EMA is materialized through the controller's public observe methods
    (which construct it with the correct horizon), then its value/count are
    overwritten with the persisted numbers.
    """
    if not data:
        return controller
    restore = (
        (controller.observe, controller._solve, data.get("solve", {})),
        (controller.observe_judge_rate, controller._judge, data.get("judge", {})),
        (controller.observe_discrimination, controller._disc, data.get("disc", {})),
    )
    for observe, store, snapshot in restore:
        for template, (value, count) in snapshot.items():
            observe(template, float(value) if value is not None else 0.0)
            ema = store[template]
            ema.value = value
            ema.count = int(count)
    controller._retired.update(data.get("retired", []))
    return controller
