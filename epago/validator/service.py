"""Validator orchestration loop ("validator-in-a-box").

One :meth:`ValidatorService.tick` performs a full pass: miner intake, a whole
competition round when the round authority has triggered one, verdict
commitment, quorum-derived coronation, private-pool rotation, calibration
scheduling, weight setting, and state persistence. The
async :meth:`ValidatorService.run_forever` just repeats ``tick`` — tests drive
``tick`` directly against fakes.

All heavy subsystems (eval harness, task generation, model backends) arrive as
constructor-injected callables through :class:`Deps`; this module never imports
them at the top level, so it runs on a machine with no torch/vllm/bittensor
installed.

There are no human-intervention paths anywhere in this loop. Failures are
handled by construction (king mirroring makes king-loss impossible), by
automatic requeue with machine-readable ``state.last_error``, or by
deterministic degradation (bootstrap quorum mode, burn-all Phase A weights).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from epago import constants
from epago.chain.client import ChainClient
from epago.config import EpagoConfig
from epago.core.emissions import (
    ArenaEntry,
    KingEmissionState,
    phase_b_active,
    trim_arena,
)
from epago.core.quorum import derive_coronation, lapsed
from epago.core.reveal import build_king_pointer, build_verdict
from epago.core.stats import (
    derive_seed,
    noise_floor_from_calibration,
    round_confirmation_private_seed,
    round_confirmation_public_seed,
    round_private_seed,
    round_public_seed,
    update_acc_ema,
)
from epago.core.types import (
    CoronationEvent,
    DuelOutcome,
    DuelSpec,
    Entrant,
    KingPointer,
    KingState,
    ModelRef,
    RoundDuelSpec,
    RoundResult,
    RoundStart,
    SubmissionStatus,
    Verdict,
    VerdictDecision,
)
from epago.model.validation import (
    exact_copy_of_king,
    validate_challenger_folder,
    weight_fingerprint,
)
from epago.taskgen.difficulty import DifficultyController
from epago.validator.audit import (
    AuditLog,
    audit16,
    build_audit_record,
    package_code_digest,
    record_digest,
)
from epago.validator.intake import apply_cooldown, cooldown_triggered, scan_and_enqueue
from epago.validator.state import (
    QueuedSubmission,
    ValidatorState,
    difficulty_from_dict,
    difficulty_to_dict,
)

logger = logging.getLogger(__name__)

#: Nominal 12-second blocks.
BLOCKS_PER_HOUR = 300

#: Calibration duel cadence, in ticks (king-vs-king on fresh tasks -> noise sample).
CALIBRATION_EVERY_TICKS = 50


class PrivatePoolLike(Protocol):
    """The slice of taskgen's ``PrivatePool`` this loop depends on."""

    epoch: int
    digest: str

    def sample(self, n: int, seed: int) -> list: ...

    def rotation_due(self, current_block: int) -> bool: ...

    def commitment(self) -> str:
        """The ``ep1`` payload committing the pool that is active *now*."""

    def rotate(self, current_block: int) -> str | None:
        """Rotate to a fresh pool; returns the incoming pool's ``ep1``
        commitment, or None if there is nothing to commit."""


@dataclass
class Deps:
    """Constructor-injected dependencies for :class:`ValidatorService`.

    ``run_duel`` / ``run_calibration_duel`` / ``run_probes`` come from the eval
    subsystem, ``generate_tasks`` / ``task_ids_digest`` / ``private_pool`` from
    taskgen. They are injected as callables/objects (never imported here) so
    the whole loop is testable with fakes and buildable before those
    subsystems exist.

    * ``run_duel(spec: core.types.DuelSpec, env, backend_factory, llm_judge) -> DuelOutcome``
    * ``run_round_duel(spec: core.types.RoundDuelSpec, env, backend_factory, llm_judge)
      -> list[RoundResult]`` — optional batch runner for a whole round
    * ``run_calibration_duel(king_dir, tasks, env, backend_factory) -> float``
      (king vs itself on fresh tasks, returning the score-gap STANDARD ERROR
      ``stdev(d_i)/sqrt(n)`` — the same unit as ``adaptive_delta``'s noise term,
      not the per-task flip rate, which does not shrink with n)
    * ``run_probes(challenger_dir, king_dir) -> list[IntakeFailure]``
    * ``generate_tasks(seed=, release=, corpus=, n=, king_probe=) -> list[Task]``
    * ``task_ids_digest(tasks) -> str``
    * ``clock() -> int`` — current chain block.
    * ``materialize(ref: ModelRef, cache_dir: Path) -> Path`` — optional
      override for :func:`epago.model.store.materialize_model` (used by tests).
    * ``sign(digest_bytes) -> hex signature`` — signs each audit record's
      canonical unsigned digest. ``None`` (tests, keyless soaks) leaves
      ``validator_signature`` empty; the audit16 commitment is unaffected
      either way (see :mod:`epago.validator.audit`).

    Every field added after the initial release carries a default so existing
    constructors (tests, soaks, wiring) keep working unchanged.
    """

    chain: ChainClient
    cfg: EpagoConfig
    state: ValidatorState
    corpus: Any
    env: Any
    backend_factory: Any
    run_duel: Callable[..., DuelOutcome]
    run_calibration_duel: Callable[..., float]
    run_probes: Callable[..., list]
    generate_tasks: Callable[..., list]
    task_ids_digest: Callable[[list], str]
    private_pool: PrivatePoolLike
    wallet_hotkey: str
    clock: Callable[[], int]
    llm_judge: Any = None
    #: Batch runner for a whole competition round. Optional: without it the
    #: service falls back to one ``run_duel`` per entrant over the same exam,
    #: which is correct but re-sweeps the king for every challenger.
    run_round_duel: Callable[..., list] | None = None
    materialize: Callable[[ModelRef, Path], Path] | None = None
    cache_dir: Path = Path(".epago-cache")
    sign: Callable[[bytes], str] | None = None


def compute_sla_report(records: list[dict]) -> dict:
    """p50/p95 reveal-to-verdict latency in blocks over resolved submissions."""
    latencies = sorted(
        int(r["verdict_at"]) - int(r["revealed_at"])
        for r in records
        if r.get("verdict_at") is not None
    )
    target = constants.SLA_TARGET_HOURS * BLOCKS_PER_HOUR
    if not latencies:
        return {"n": 0, "p50_blocks": None, "p95_blocks": None, "sla_target_blocks": target}

    def pct(q: float) -> int:
        return latencies[min(int(round(q * (len(latencies) - 1))), len(latencies) - 1)]

    return {
        "n": len(latencies),
        "p50_blocks": pct(0.50),
        "p95_blocks": pct(0.95),
        "sla_target_blocks": target,
    }


class ValidatorService:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.chain = deps.chain
        self.cfg = deps.cfg
        self.state = deps.state
        self.audit_log = AuditLog(self.state.state_dir)
        self.mirror_dir = self.state.state_dir / "king_mirror"
        # Difficulty telemetry survives restarts; taskgen wiring reads
        # ``service.difficulty`` for mint-mixture weights.
        self.difficulty = difficulty_from_dict(DifficultyController(), self.state.difficulty)
        self._code_digest = package_code_digest()
        self._warned_burn_fallback = False
        # Local API-key round trigger (optional). The server flips a latch; the
        # tick loop reads it in `_pending_round`. Injected by tests; started
        # from config here when an address and key are configured.
        self._round_trigger = getattr(deps, "round_trigger", None)
        self._round_server = None
        self._maybe_start_round_trigger()
        # "Active in the last N duels" proxied in blocks via the SLA duel cadence:
        # a validator that hasn't posted a verdict within N duel-SLAs is a follower.
        self._active_window_blocks = (
            self.cfg.quorum.active_window_duels * constants.SLA_TARGET_HOURS * BLOCKS_PER_HOUR
        )

    # ------------------------------------------------------------------ loop

    async def run_forever(self, poll_interval_s: float = 30.0) -> None:
        """Tick until cancelled. Failures degrade, they never halt: any tick
        exception is recorded machine-readably and the next tick proceeds."""
        while True:
            try:
                await asyncio.to_thread(self.tick)
            except Exception as exc:  # noqa: BLE001 - no human-intervention paths
                logger.exception("tick failed")
                self.state.last_error = {
                    "code": "tick_failure",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "block": self._safe_block(),
                }
                try:
                    self.state.save()
                except OSError:
                    logger.exception("state save failed after tick failure")
            await asyncio.sleep(poll_interval_s)

    def tick(self) -> None:
        """One full synchronous pass. This is what tests drive."""
        block = self.deps.clock()
        if self.state.genesis_block == 0:
            self.state.genesis_block = block
        self._ensure_genesis_king(block)
        # Before anything reads the king: an auditing validator takes the throne
        # from the authority's on-chain pointer, so intake compares challenges
        # against the king the subnet actually has.
        self._sync_king_from_pointer()

        # Re-commit anything a prior tick could not land (commitment rate
        # limit). Do this first and alone so it wins this window's write budget.
        if self.state.pending_verdicts:
            head = self.state.pending_verdicts[0]
            if self.chain.publish_reveal(head, constants.VERDICT_REVEAL_BLOCKS):
                self.state.pending_verdicts.pop(0)
        elif self.state.pending_king_pointer is not None:
            if self.chain.publish_reveal(
                self.state.pending_king_pointer, constants.VERDICT_REVEAL_BLOCKS
            ):
                self.state.pending_king_pointer = None

        try:
            scan_and_enqueue(self.chain, self.state, self.cfg, self.state.king.ref.digest)
        except Exception as exc:  # noqa: BLE001 - chain reads are best-effort
            self.state.last_error = {
                "code": "intake_scan_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }

        # Competitions only run when the round authority triggers one. With no
        # trigger the queue keeps filling and nothing is evaluated — a
        # deliberate liveness dependency on a privileged key.
        pending = self._pending_round()
        if pending is not None:
            if self._ensure_pool_committed(block):
                self._run_round(pending)
            elif self._round_trigger is not None and not pending.authority_hotkey:
                # The API latch was consumed to mint this round, but the private
                # pool's ``ep1`` is not on chain yet (the commitment pallet
                # rate-limits writes), so nothing ran. The latch is a boolean
                # that ``take()`` already lowered, so without re-arming it the
                # operator's trigger is silently swallowed and the round has to
                # be requested again by hand. The on-chain trigger re-derives
                # itself from ``latest_round`` every tick and needs no such fix.
                logger.info(
                    "round %d deferred: private pool not committed yet; re-arming the trigger",
                    pending.round,
                )
                self._round_trigger.request()
        self._derive_coronations(self.deps.clock())
        self._retry_mirror()
        self._rotate_private_pool(self.deps.clock())
        self._maybe_publish_mailbox(self.deps.clock())
        self._publish_pool_manifest()
        self._maybe_calibrate(self.deps.clock())
        self._maybe_anchor(self.deps.clock())
        try:
            self.maybe_set_weights()
        except Exception as exc:  # noqa: BLE001 - periodic emission set; never abort the tick
            self.state.last_error = {
                "code": "set_weights_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": self.deps.clock(),
            }
            logger.warning("[chain] set_weights failed (non-fatal, retries next interval): %s", exc)
        self._audit_housekeeping(self.deps.clock())
        self.state.tick_count += 1
        self.state.save()

    # ------------------------------------------------------------- genesis

    def _ensure_genesis_king(self, block: int) -> None:
        """At genesis the pinned seed snapshot is the king. It has no author,
        so it earns nothing (weights burn until the first coronation), but the
        duel baseline always exists — there is no empty-throne state."""
        if self.state.king is not None:
            return
        self.state.king = KingState(
            ref=ModelRef(repo=self.cfg.chain.seed_repo, digest=self.cfg.seed.seed_digest),
            author_hotkey="",
            crowned_block=block,
            reign_started_block=block,
            acc_ema=self.state.king_acc_ema,
            coronation_lcb=0.0,
            coronation_delta=0.0,
        )

    # -------------------------------------------------------- king pointer

    @property
    def _is_king_authority(self) -> bool:
        return bool(
            self.cfg.chain.king_authority_hotkey
            and self.cfg.chain.king_authority_hotkey == self.deps.wallet_hotkey
        )

    def _sync_king_from_pointer(self) -> None:
        """Adopt the authority's ``ek1`` king pointer when it names a newer king.

        This is what makes a validator startable. Without it the only king a box
        with an empty state directory could name was the genesis seed: every
        live ``e2`` reveal then failed the ``stale_parent`` gate, no duel ever
        ran, no verdict was ever posted, and the box could not rejoin — the same
        trap the scoring validator falls into if it loses its state directory.

        The authority is pinned in ``chain.toml``, so a pointer from any other
        hotkey is ignored. Only strictly newer coronations are adopted, so a
        replayed older pointer cannot roll the throne backwards.
        """
        authority = self.cfg.chain.king_authority_hotkey
        if not authority or self._is_king_authority:
            return
        try:
            pointer = self.chain.read_king_pointer(authority)
        except Exception as exc:  # noqa: BLE001 - a chain read never halts the loop
            self.state.last_error = {
                "code": "king_pointer_read_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": self._safe_block(),
            }
            return
        if pointer is None:
            return
        current = self.state.king
        if current is not None and pointer.digest == current.ref.digest:
            return
        if current is not None and pointer.crowned_block < current.crowned_block:
            return  # stale pointer replay
        self.state.adopt_king(
            ref=pointer.ref,
            author_hotkey=pointer.author_hotkey,
            crowned_block=pointer.crowned_block,
            reign_started_block=pointer.reign_started_block,
            coronation_lcb=pointer.coronation_lcb,
            coronation_delta=pointer.coronation_delta,
        )
        self._mirror_current_king()
        self._drop_stale(pointer.digest)
        logger.info(
            "adopted king %s from %s pointer at block %d",
            pointer.digest,
            authority,
            pointer.block,
        )

    def _publish_king_pointer(self) -> None:
        """Publish the current king so every other validator can find it."""
        if not self._is_king_authority or self.state.king is None:
            return
        king = self.state.king
        if not king.author_hotkey:
            return  # the genesis seed has no author and crowns nobody
        payload = build_king_pointer(
            KingPointer(
                repo=king.ref.repo,
                digest=king.ref.digest,
                author_hotkey=king.author_hotkey,
                crowned_block=king.crowned_block,
                reign_started_block=king.reign_started_block,
                coronation_lcb_e6=round(king.coronation_lcb * 1e6),
                coronation_delta_e6=max(round(king.coronation_delta * 1e6), 0),
            )
        )
        if not self.chain.publish_reveal(payload, constants.VERDICT_REVEAL_BLOCKS):
            # Same rate limit that defers verdicts; retried from the head of
            # later ticks so the pointer is never silently lost.
            self.state.pending_king_pointer = payload

    # ------------------------------------------------------------- the duel

    def _materialize(self, ref: ModelRef) -> Path:
        if self.deps.materialize is not None:
            return Path(self.deps.materialize(ref, self.deps.cache_dir))
        from epago.model.store import materialize_model  # heavy deps stay late-bound

        return materialize_model(
            ref, self.deps.cache_dir, fallbacks=self._public_fallbacks(ref)
        )

    @staticmethod
    def _public_fallbacks(ref: ModelRef) -> tuple[ModelRef, ...]:
        """Where else a crowned model can be fetched from.

        A challenger uploads into ``submissions/<its own hotkey>/``, which only
        that miner can write and only this validator can read. That is the
        point while it is a challenger — and a problem the moment it is
        crowned, because every other party then needs the king and none of
        them can reach a private prefix.

        Coronation republishes it to ``kings/<digest>/``, and because that
        location is *derived from the digest* any party can construct it
        without a manifest, a pointer, or a call to whoever published it.
        Content addressing does the rest: a fallback is only accepted after
        its bytes rehash to the committed digest, so an impostor serving
        something else at that path is caught rather than trusted.

        Only for ``sha256:`` refs. An ``hf:`` primary pins a revision of one
        specific repository, and a different repository's revision hash proves
        nothing about content equality.
        """
        if ref.backend != "oci":
            return ()
        from epago.publishing.publisher import king_object_repo

        public = king_object_repo(ref.digest)
        if ref.repo == public:
            return ()  # already the public copy
        return (ModelRef(repo=public, digest=ref.digest),)

    def _transient(self, sub: QueuedSubmission, code: str, exc: Exception, block: int) -> None:
        """Transient failure: requeue at the front, record machine-readable state."""
        self.state.requeue_front(sub)
        self.state.last_error = {
            "code": code,
            "digest": sub.digest,
            "detail": f"{type(exc).__name__}: {exc}",
            "block": block,
        }

    # ------------------------------------------------------------ competition

    def _maybe_start_round_trigger(self) -> None:
        """Start the API-key trigger server when configured (and not injected)."""
        import os

        if self._round_trigger is not None:
            return  # a test or caller supplied one
        bind = self.cfg.chain.round_api_bind
        key = os.environ.get("EPAGO_ROUND_API_KEY", "")
        if not bind:
            return
        if not key:
            logger.warning(
                "round_api_bind is set but EPAGO_ROUND_API_KEY is empty; "
                "refusing to serve an unauthenticated trigger"
            )
            return
        from epago.validator.roundapi import RoundTrigger, serve_round_trigger

        self._round_trigger = RoundTrigger()
        self._round_server = serve_round_trigger(bind, key, self._round_trigger)

    def _pending_round(self) -> "RoundStart | None":
        """The round to run now, or None.

        Nothing is evaluated until a round is triggered — a deliberate liveness
        dependency: with no trigger the queue keeps filling, the king keeps
        earning, and no duel ever runs. Two trigger sources, checked in order:

        1. the **local API-key latch** (:mod:`epago.validator.roundapi`), which
           mints a round from the CURRENT chain block — its hash, which the
           owner cannot choose, seeds the exam, so an API trigger cannot leak
           the questions any more than the on-chain one could;
        2. the **on-chain authority hotkey**, whose ``er1`` every validator can
           see and time-check independently.

        The minimum interval is enforced for both, so neither can run rounds
        back to back.
        """
        if self._round_trigger is not None and self._round_trigger.take():
            block = self._safe_block()
            since = block - self.state.last_round_block
            if self.state.last_round_block and since < constants.ROUND_MIN_INTERVAL_BLOCKS:
                logger.warning(
                    "round trigger ignored: only %d blocks since round %d, minimum is %d",
                    since, self.state.last_round_run, constants.ROUND_MIN_INTERVAL_BLOCKS,
                )
            else:
                try:
                    block_hash = self.chain.block_hash(block)
                except Exception as exc:  # noqa: BLE001 - never halt the loop
                    self.state.last_error = {
                        "code": "round_block_hash_failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "block": block,
                    }
                    return None
                return RoundStart(
                    round=self.state.last_round_run + 1,
                    authority_hotkey="",  # local trigger has no on-chain author
                    block=block,
                    block_hash=block_hash,
                )

        authority = self.cfg.chain.round_authority_hotkey
        if not authority:
            return None
        try:
            latest = self.chain.latest_round(authority)
        except Exception as exc:  # noqa: BLE001 - a chain read never halts the loop
            self.state.last_error = {
                "code": "round_read_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": self._safe_block(),
            }
            return None
        if latest is None or latest.round <= self.state.last_round_run:
            return None
        return latest

    def _field_for(self, start: "RoundStart") -> list[QueuedSubmission]:
        """The entrants for a round, in a deterministic order.

        Only challenges revealed **strictly before** the round-start block are
        eligible. A reveal landing at or after that block has already seen the
        hash that mints the exam, so admitting it would hand its author the
        questions before the weights were fixed — the one property the whole
        commit-reveal design exists to protect. Those wait for the next round.

        Ordered by (reveal_block, digest) and capped at
        ``ROUND_MAX_ENTRANTS``: every entrant costs a full sweep, so an
        unbounded field would blow the SLA. The overflow keeps its place in the
        queue for the next round, and the cut is logged rather than silent.
        """
        eligible = [q for q in self.state.queue if q.reveal_block < start.block]
        eligible.sort(key=lambda q: (q.reveal_block, q.digest))
        if len(eligible) > constants.ROUND_MAX_ENTRANTS:
            deferred = eligible[constants.ROUND_MAX_ENTRANTS:]
            logger.warning(
                "round %d field capped at %d; %d challenger(s) deferred to the next round: %s",
                start.round,
                constants.ROUND_MAX_ENTRANTS,
                len(deferred),
                ", ".join(q.digest for q in deferred),
            )
            eligible = eligible[: constants.ROUND_MAX_ENTRANTS]
        return eligible

    def _run_round(self, start: "RoundStart") -> None:
        """Evaluate a whole competition round: gate, duel, settle, pick a winner."""
        assert self.state.king is not None
        block = self.deps.clock()
        self.state.current_round = start.round

        field = self._field_for(start)
        if not field:
            logger.info("round %d opened with an empty field", start.round)
            self.state.last_round_run = start.round
            self.state.last_round_block = start.block
            return

        try:
            king_dir = self._materialize(self.state.king.ref)
        except Exception as exc:  # noqa: BLE001 - retry the whole round next tick
            self.state.last_error = {
                "code": "king_materialize_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }
            return

        entrants, admitted = self._admit_entrants(field, king_dir, start, block)
        if not entrants:
            self.state.last_round_run = start.round
            self.state.last_round_block = start.block
            return

        pre_round_ema = self.state.king_acc_ema
        noise_floor = noise_floor_from_calibration(self.state.noise_floor_samples)
        try:
            public_tasks = self._public_tasks(
                round_public_seed(start.block_hash, start.round), constants.N_PUB_TASKS
            )
            private_tasks = self.deps.private_pool.sample(
                constants.N_PRIV_TASKS, round_private_seed(start.block_hash, start.round)
            )
        except Exception as exc:  # noqa: BLE001 - retry the whole round next tick
            self.state.last_error = {
                "code": "taskgen_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }
            return

        spec = RoundDuelSpec(
            king_dir=king_dir,
            entrants=tuple(entrants),
            public_tasks=public_tasks,
            private_tasks=private_tasks,
            round=start.round,
            round_block_hash=start.block_hash,
            king_acc_ema=pre_round_ema,
            noise_floor=noise_floor,
        )
        try:
            results = self._run_round_duel(spec)
        except Exception as exc:  # noqa: BLE001 - retry the whole round next tick
            self.state.last_error = {
                "code": "round_duel_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }
            return

        winner = self._pick_winner(
            results,
            noise_floor=noise_floor,
            reveal_blocks={d: s.reveal_block for d, s in admitted.items()},
        )
        confirmation: RoundResult | None = None
        if winner is not None and constants.CORONATION_CONFIRMATION_DUELS > 0:
            confirmation = self._confirm_winner(winner, spec, start)
            if confirmation is None or not confirmation.outcome.accepted:
                # An unconfirmed win settles as a near-miss: the entrant showed
                # lcb > delta once, keeps its arena credit and its re-duel
                # right, and pays no cooldown — but a coronation is a
                # 99.9%-confidence event and one clear is not two. A failed
                # confirmation *mint* demotes as well, never crowns: the
                # conservative direction for an event this hard to reverse.
                winner = None
        for result in results:
            self._settle_entrant(
                result, admitted[result.entrant.digest], spec, start,
                is_winner=result is winner, pre_round_ema=pre_round_ema,
            )

        # One EMA update per round, from the king's measured accuracy on the
        # round's exam — the king answered it once, so there is one observation.
        if results:
            sample = results[0].outcome
            observed = (sample.public.king_acc + sample.private.king_acc) / 2.0
            self.state.king_acc_ema = update_acc_ema(pre_round_ema, observed)
            self.state.clean_duels += len(results)
            self._feed_difficulty(sample, public_tasks)

        round_record: dict[str, Any] = {
            "round": start.round,
            "round_block": start.block,
            "public_seed": f"{round_public_seed(start.block_hash, start.round):016x}",
            "task_ids": [getattr(t, "task_id", str(t)) for t in public_tasks],
            "entrants": [e.digest for e in entrants],
        }
        if confirmation is not None:
            # The confirmation exam is replayable from its seed alone; the
            # record pins the outcome so an auditor can tell a confirmed crown
            # from a demoted one without re-running anything.
            round_record["confirmation"] = {
                "digest": confirmation.entrant.digest,
                "public_seed": f"{round_confirmation_public_seed(start.block_hash, start.round):016x}",
                "lcb_pub": confirmation.outcome.lcb_pub,
                "mu_hat_priv": confirmation.outcome.private.mu_hat,
                "confirmed": bool(confirmation.outcome.accepted),
            }
        self.audit_log.stage_delayed(
            f"round{start.round:06d}",
            json.dumps(round_record, sort_keys=True),
            self.deps.clock() + constants.AUDIT_PUBLISH_DELAY_BLOCKS,
        )
        self._stage_public_pool_round(start.round, public_tasks)
        self.state.last_round_run = start.round
        self.state.last_round_block = start.block
        self.state.last_error = None

    def _admit_entrants(
        self, field: list[QueuedSubmission], king_dir: Path, start: "RoundStart", block: int
    ) -> tuple[list[Entrant], dict[str, QueuedSubmission]]:
        """Run the per-checkpoint gates and return the survivors.

        Cheap deterministic checks first, then the GPU probes — same ladder a
        one-off duel used, just applied to a field. A rejection here resolves
        the submission immediately; it never reaches the exam.
        """
        entrants: list[Entrant] = []
        admitted: dict[str, QueuedSubmission] = {}
        # Weight fingerprint -> the entrant that claimed it first in this round.
        # Digest ownership cannot catch a copy on its own: an ``hf:`` digest is a
        # revision hash, so the same weights pushed to a second repo get a new
        # digest and enter as a separate challenger. Under rounds that is acute —
        # thief and victim sit in the SAME field on the SAME exam, score
        # identically, and the digest tie-break hands the crown to whichever id
        # sorts higher. The field is processed in (reveal_block, digest) order,
        # so the earlier reveal keeps the slot.
        claimed: dict[str, str] = {}
        for sub in field:
            def resolve(status: SubmissionStatus, code: str, detail: str, cool: bool = False) -> None:
                self.state.record_failure(sub.digest, code, detail, block)
                self.state.record_verdict(
                    digest=sub.digest, hotkey=sub.author_hotkey, status=status,
                    lcb_pub=0.0, verdict_block=block,
                )
                self.state.record_sla(sub.digest, sub.reveal_block, sub.enqueued_block, block)
                if cool:
                    apply_cooldown(
                        self.state, sub.author_hotkey, block, queue_depth=len(self.state.queue)
                    )
                self.state.queue = [q for q in self.state.queue if q.digest != sub.digest]

            try:
                challenger_dir = self._materialize(ModelRef(repo=sub.repo, digest=sub.digest))
            except Exception as exc:  # noqa: BLE001 - unreachable weights are a forfeit
                resolve(
                    SubmissionStatus.FAILED_INTAKE,
                    "materialize_failed",
                    f"{type(exc).__name__}: {exc}",
                )
                continue

            failures = validate_challenger_folder(challenger_dir, king_dir, self.cfg)
            if failures:
                resolve(
                    SubmissionStatus.FAILED_INTAKE,
                    failures[0].code,
                    "; ".join(f"{f.code}: {f.detail}" for f in failures),
                )
                continue
            if exact_copy_of_king(challenger_dir, king_dir):
                resolve(
                    SubmissionStatus.FAILED_INTAKE,
                    "exact_copy",
                    "challenger shards are byte-identical to the king",
                )
                continue

            probe_failures = list(self.deps.run_probes(challenger_dir, king_dir))
            if probe_failures:
                resolve(
                    SubmissionStatus.FAILED_PROBES,
                    "probes",
                    "; ".join(f"{f.code}: {f.detail}" for f in probe_failures),
                    cool=True,
                )
                continue

            fingerprint = weight_fingerprint(challenger_dir)
            owner = claimed.get(fingerprint)
            if owner is not None:
                resolve(
                    SubmissionStatus.FAILED_INTAKE,
                    "duplicate_weights",
                    f"identical weights already entered this round as {owner}",
                )
                continue
            # Cross-round content identity. One-duel-per-digest is a rule about
            # revision hashes, and the same weights re-uploaded (new repo, new
            # sharding) mint a fresh digest — a free second draw on the same
            # content. The fingerprint registry closes that: weights that have
            # ever dueled are terminal under every digest, and re-entering
            # known content cools the hotkey down like any other junk
            # submission — a retry must at least be a retrain.
            prior = self.state.seen_fingerprints.get(fingerprint)
            if prior is not None and prior != sub.digest:
                resolve(
                    SubmissionStatus.FAILED_INTAKE,
                    "duplicate_weights",
                    f"identical weights already dueled as {prior}",
                    cool=True,
                )
                continue
            claimed[fingerprint] = sub.digest
            self.state.seen_fingerprints.setdefault(fingerprint, sub.digest)

            entrants.append(
                Entrant(
                    digest=sub.digest,
                    repo=sub.repo,
                    author_hotkey=sub.author_hotkey,
                    challenger_dir=challenger_dir,
                )
            )
            admitted[sub.digest] = sub
        return entrants, admitted

    def _run_round_duel(self, spec: RoundDuelSpec) -> list[RoundResult]:
        """Delegate to the batch runner, or fall back to per-entrant duels.

        The fallback exists so a Deps built with only ``run_duel`` (tests,
        soaks, an older eval server) still produces correct round results: the
        exam is already minted by the caller and handed to every entrant
        unchanged, so the field is still compared on identical questions. It
        costs one extra king sweep per entrant, which is why the batch runner
        is preferred whenever it is wired.
        """
        if self.deps.run_round_duel is not None:
            return list(
                self.deps.run_round_duel(
                    spec, self.deps.env, self.deps.backend_factory, self.deps.llm_judge
                )
            )
        results: list[RoundResult] = []
        for entrant in spec.entrants:
            outcome = self.deps.run_duel(
                DuelSpec(
                    king_dir=spec.king_dir,
                    challenger_dir=entrant.challenger_dir,
                    public_tasks=spec.public_tasks,
                    private_tasks=spec.private_tasks,
                    block_hash_at_reveal=spec.round_block_hash,
                    author_hotkey=entrant.digest,
                    king_acc_ema=spec.king_acc_ema,
                    noise_floor=spec.noise_floor,
                    round_id=f"round{spec.round:06d}-{entrant.digest[-8:]}",
                ),
                self.deps.env,
                self.deps.backend_factory,
                self.deps.llm_judge,
            )
            results.append(RoundResult(entrant=entrant, outcome=outcome))
        return results

    def _confirm_winner(
        self, winner: RoundResult, spec: RoundDuelSpec, start: "RoundStart"
    ) -> RoundResult | None:
        """Re-duel the provisional winner once on a fresh exam.

        The round already showed ``lcb_pub > delta`` once; this asks for the
        same clear on an independent sample before the ACCEPT is committed.
        For a genuinely better model the second clear is the expected outcome;
        for a lucky noise-copy the probabilities multiply, which is the whole
        point — no per-attempt bar can stay meaningful against a fleet of
        lottery tickets, but squaring the tail can. Costs two sweeps, paid only
        when a coronation is on the table. Returns None when the confirmation
        exam could not be minted or run; the caller treats that as unconfirmed.
        """
        try:
            public_tasks = self.deps.generate_tasks(
                seed=round_confirmation_public_seed(start.block_hash, start.round),
                release=self.cfg.eval.taskgen_release,
                corpus=self.deps.corpus,
                n=constants.N_PUB_TASKS,
                king_probe=None,
            )
            private_tasks = self.deps.private_pool.sample(
                constants.N_PRIV_TASKS,
                round_confirmation_private_seed(start.block_hash, start.round),
            )
            confirm_spec = RoundDuelSpec(
                king_dir=spec.king_dir,
                entrants=(winner.entrant,),
                public_tasks=public_tasks,
                private_tasks=private_tasks,
                round=spec.round,
                round_block_hash=spec.round_block_hash,
                king_acc_ema=spec.king_acc_ema,
                noise_floor=spec.noise_floor,
            )
            results = self._run_round_duel(confirm_spec)
        except Exception as exc:  # noqa: BLE001 - unconfirmed, never crowned
            self.state.last_error = {
                "code": "confirmation_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": self.deps.clock(),
            }
            return None
        return results[0] if results else None

    @staticmethod
    def _pick_winner(
        results: list[RoundResult],
        *,
        noise_floor: float = 0.0,
        reveal_blocks: dict[str, int] | None = None,
    ) -> RoundResult | None:
        """The single entrant crowned by this round, or None.

        Highest LCB among those that cleared both halves — but two LCBs closer
        than one noise floor are the same measurement, and inside that band the
        EARLIER reveal wins. Without the band the tie-break was the digest,
        which handed a coin flip to anyone who copied a pending challenger,
        perturbed the weights (new digest, same skill), and revealed before the
        round closed; reveal order makes theft strictly worse than the
        original, because a perturbed copy cannot out-score its source beyond
        noise. The digest stays as the final deterministic tie-break so every
        validator picks the same winner from the same results. Everyone else is
        rejected — including entrants that beat the king outright, who are
        runners-up rather than losers and are paid from the arena pool
        accordingly.
        """
        eligible = [r for r in results if r.outcome.accepted]
        if not eligible:
            return None
        blocks = reveal_blocks or {}

        def key(r: RoundResult) -> tuple[float, int, str]:
            lcb = r.outcome.lcb_pub
            band = lcb if noise_floor <= 0 else math.floor(lcb / noise_floor)
            return (band, -blocks.get(r.entrant.digest, 0), r.entrant.digest)

        return max(eligible, key=key)

    def _settle_entrant(
        self,
        result: RoundResult,
        sub: QueuedSubmission,
        spec: RoundDuelSpec,
        start: "RoundStart",
        *,
        is_winner: bool,
        pre_round_ema: float,
    ) -> None:
        """Record, commit and file one entrant's result."""
        entrant, outcome = result.entrant, result.outcome
        verdict_block = self.deps.clock()
        round_id = f"round{start.round:06d}-{entrant.digest[-8:]}"
        assert self.state.king is not None

        record = build_audit_record(
            round_id=round_id,
            block_hash_at_reveal=start.block_hash,
            author_hotkey=entrant.author_hotkey,
            king_repo=self.state.king.ref.repo,
            king_digest=self.state.king.ref.digest,
            challenger_repo=entrant.repo,
            challenger_digest=entrant.digest,
            corpus_digest=self.cfg.eval.corpus_digest,
            taskgen_release=self.cfg.eval.taskgen_release,
            public_seed=outcome.public_seed_hex,
            public_task_ids_digest=self.deps.task_ids_digest(spec.public_tasks),
            # Pinned per verdict, not read from the contract at replay time: a
            # contract can be edited, and an auditor must be able to pin the
            # pool this duel actually used.
            public_pool_digest=getattr(self.cfg.eval, "public_pool_digest", ""),
            public_pool_manifest_digest=getattr(
                self.cfg.eval, "public_pool_manifest_digest", ""
            ),
            private_pool_digest=self.deps.private_pool.digest,
            private_pool_epoch=self.deps.private_pool.epoch,
            n_private_tasks=outcome.private.n_tasks,
            boot_seed=outcome.boot_seed_hex,
            king_acc_ema=pre_round_ema,
            delta_threshold=outcome.delta,
            mu_hat_pub=outcome.public.mu_hat,
            lcb_pub=outcome.lcb_pub,
            mu_hat_priv=outcome.private.mu_hat,
            accepted=is_winner,
            harness_digest=self._code_digest,
            judge_model_digest=self.cfg.eval.judge_digest,
            eval_code_digest=self._code_digest,
            judge_invocation_rate=outcome.judge_invocation_rate,
            revealed_at_block=sub.reveal_block,
            intake_at_block=sub.enqueued_block,
            verdict_at_block=verdict_block,
            validator_hotkey=self.deps.wallet_hotkey,
            extra={
                "round": start.round,
                "round_block": start.block,
                "cleared_floor": outcome.accepted,
                "public_diffs": [[tid, int(d)] for tid, d in outcome.public_task_results],
                "judge_tier_counts": [[tier, int(n)] for tier, n in outcome.judge_tier_counts],
            },
        )
        unsigned_digest = record_digest(record)
        if self.deps.sign is not None:
            record.validator_signature = self.deps.sign(unsigned_digest.encode())
        self.audit_log.append(record)

        # Only the winner is committed as ACCEPT: a round crowns one model, so
        # quorum must see exactly one candidate to coronate.
        verdict = Verdict(
            challenger_digest=entrant.digest,
            decision=VerdictDecision.ACCEPT if is_winner else VerdictDecision.REJECT,
            lcb_pub_e6=round(outcome.lcb_pub * 1e6),
            mu_priv_e6=round(outcome.private.mu_hat * 1e6),
            delta_e6=max(round(outcome.delta * 1e6), 0),
            round=start.round,
            private_pool_epoch=self.deps.private_pool.epoch,
            audit_digest=audit16(record),
        )
        payload = build_verdict(verdict)
        if not self.chain.publish_reveal(payload, constants.VERDICT_REVEAL_BLOCKS):
            self.state.queue_pending_verdict(payload)

        status = self._classify_round(outcome, is_winner=is_winner)
        if status is SubmissionStatus.DUEL_LOST:
            self.state.record_failure(
                entrant.digest, "duel_lost", f"lcb_pub={outcome.lcb_pub:.6f}", verdict_block
            )
        if cooldown_triggered(status):
            apply_cooldown(
                self.state, entrant.author_hotkey, verdict_block, queue_depth=len(self.state.queue)
            )
        self.state.record_verdict(
            digest=entrant.digest,
            hotkey=entrant.author_hotkey,
            status=status,
            lcb_pub=outcome.lcb_pub,
            verdict_block=verdict_block,
        )
        self.state.record_sla(entrant.digest, sub.reveal_block, sub.enqueued_block, verdict_block)
        self.state.queue = [q for q in self.state.queue if q.digest != entrant.digest]

        # EVERY entrant this box dueled becomes a coronation candidate, not just
        # the one it would crown. Quorum may crown an entrant this validator
        # rejected — its own private half may have disagreed — and weight-setting
        # has to follow the chain-derived king regardless. Registering only the
        # local winner left a dissenting validator with no candidate to adopt,
        # so it kept the old king forever while the rest of the subnet moved on.
        self.state.candidates[entrant.digest] = {
            "repo": entrant.repo,
            "author_hotkey": entrant.author_hotkey,
            "reveal_block": sub.reveal_block,
            "king_digest": sub.king_digest,
            "lcb": outcome.lcb_pub,
            "delta": outcome.delta,
            "round": start.round,
            # The coronation window runs from the round, not from the reveal.
            # Submissions wait for the next competition, and with a ~2-day
            # cadence and a ~24h timeout every one of them would lapse before
            # its round ever opened.
            "round_block": start.block,
        }

    @staticmethod
    def _classify_round(outcome: DuelOutcome, *, is_winner: bool) -> SubmissionStatus:
        """Status for one entrant, given the round's winner.

        An entrant that cleared the floor but did not win is a **near-miss**,
        not a loss: it beat the king and only lost the round, so it earns arena
        credit and pays no cooldown.
        """
        if is_winner:
            return SubmissionStatus.ACCEPTED
        if outcome.accepted or (0.0 < outcome.lcb_pub and outcome.private.mu_hat > 0.0):
            return SubmissionStatus.NEAR_MISS
        return SubmissionStatus.DUEL_LOST

    @staticmethod
    def _classify(outcome: DuelOutcome) -> SubmissionStatus:
        if outcome.accepted:
            return SubmissionStatus.ACCEPTED
        if 0.0 < outcome.lcb_pub <= outcome.delta:
            return SubmissionStatus.NEAR_MISS
        return SubmissionStatus.DUEL_LOST

    # ------------------------------------------------------- difficulty feed

    def _feed_difficulty(self, outcome: DuelOutcome, public_tasks: list) -> None:
        """Feed per-template duel telemetry into the difficulty controller.

        Joins ``outcome.public_task_results`` with the minted public task list
        (tasks carry ``.template``). The outcome exposes king accuracy at half
        granularity only, so each observed template records the duel's public
        king accuracy; discrimination (mean ``|d_i|``) is genuinely
        per-template. The controller snapshot is persisted with the rest of
        the state so telemetry survives restarts.
        """
        template_by_id: dict[str, str] = {}
        for task in public_tasks:
            task_id = getattr(task, "task_id", None)
            template = getattr(task, "template", None)
            if task_id and template:
                template_by_id[task_id] = template
        diffs_by_template: dict[str, list[int]] = {}
        for task_id, d_i in outcome.public_task_results:
            template = template_by_id.get(task_id)
            if template is not None:
                diffs_by_template.setdefault(template, []).append(int(d_i))
        if not diffs_by_template:
            return
        judge_rate = outcome.judge_invocation_rate
        king_acc = outcome.public.king_acc
        for template in sorted(diffs_by_template):
            diffs = diffs_by_template[template]
            self.difficulty.observe(template, king_acc)
            self.difficulty.observe_discrimination(
                template, sum(abs(d) for d in diffs) / len(diffs)
            )
            self.difficulty.observe_judge_rate(template, judge_rate)
        self.state.difficulty = difficulty_to_dict(self.difficulty)

    # -------------------------------------------------------------- coronation

    def _derive_coronations(self, block: int) -> None:
        if not self.state.candidates:
            return
        verdicts = self.chain.read_verdicts()
        evaluators = self.chain.evaluators(self._active_window_blocks)
        events: list[tuple[CoronationEvent, dict]] = []
        for digest in sorted(self.state.candidates):
            meta = self.state.candidates[digest]
            # Anchor the timeout on the round that produced the candidate.
            anchor = int(meta.get("round_block", meta["reveal_block"]))
            event = derive_coronation(
                digest,
                verdicts,
                evaluators,
                self.cfg.quorum,
                reveal_block=anchor,
                current_block=block,
            )
            if event is not None:
                events.append((event, meta))
            elif lapsed(anchor, block, self.cfg.quorum):
                # Stop tracking it for coronation, but do NOT overwrite a status
                # the duel already decided. Every entrant is a candidate, so
                # most candidates lapse by design — a near-miss whose status was
                # replaced by LAPSED fell out of the terminal set and could
                # re-enter the queue forever, bypassing its retry budget.
                if self.state.statuses.get(digest) == SubmissionStatus.QUEUED.value:
                    self.state.statuses[digest] = SubmissionStatus.LAPSED.value
                self.state.candidates.pop(digest)
        for event, meta in sorted(events, key=lambda p: (p[0].block, p[0].challenger_digest)):
            assert self.state.king is not None
            if meta["king_digest"] != self.state.king.ref.digest:
                # Crowned-in-the-same-pass sibling made this parent stale.
                self.state.statuses[event.challenger_digest] = SubmissionStatus.STALE_PARENT.value
                self.state.candidates.pop(event.challenger_digest, None)
                continue
            self._crown(event, meta)

    def _crown(self, event: CoronationEvent, meta: dict) -> None:
        digest = event.challenger_digest
        ref = ModelRef(repo=meta["repo"], digest=digest)
        # Coronation terms come from the accepting verdicts on chain, not from
        # this box's own duel. delta and lcb are per-validator (each box has its
        # own noise floor and king-accuracy EMA), so reading them locally made
        # the coronation bonus — and therefore the whole weight vector —
        # disagree between validators that had crowned the identical king.
        # Yuma then penalises everyone for the divergence. Lowest accepting
        # LCB and highest accepting delta are the conservative pair, and both
        # are a pure function of chain state for every observer.
        accepts = [v for v in event.verdicts if v.decision is VerdictDecision.ACCEPT]
        if accepts:
            lcb = min(v.lcb_pub for v in accepts)
            delta = max(v.delta for v in accepts)
        else:  # bootstrap paths that crowned without a parsed accept
            lcb = float(meta.get("lcb") or 0.0)
            delta = float(meta.get("delta") or 0.0)
        self.state.set_king(
            ref=ref,
            author_hotkey=meta["author_hotkey"],
            crowned_block=event.block,
            coronation_lcb=lcb,
            coronation_delta=delta,
        )
        self.state.candidates.pop(digest, None)
        self.state.statuses[digest] = SubmissionStatus.ACCEPTED.value
        self._mirror_current_king()
        self._drop_stale(digest)
        self._publish_king_pointer()
        # Every coronation is a transfer experiment: the new king runs the
        # external anchor benchmark on the next tick instead of waiting out
        # the weekly cadence, so "did the exam gain show up on a real
        # deep-research benchmark?" is answered per crown, publicly.
        self.state.last_anchor_block = event.block - constants.ANCHOR_INTERVAL_BLOCKS

    def _mirror_current_king(self) -> None:
        assert self.state.king is not None
        try:
            king_dir = self._materialize(self.state.king.ref)
            self.state.mirror_king(king_dir, self.mirror_dir)
            self._publish_king_publicly(king_dir)
        except Exception as exc:  # noqa: BLE001 - retried every tick, never halts
            self.state.pending_mirror = True
            self.state.last_error = {
                "code": "mirror_pending",
                "digest": self.state.king.ref.digest,
                "detail": f"{type(exc).__name__}: {exc}",
                "block": self._safe_block(),
            }

    def _publish_king_publicly(self, king_dir: "Path") -> None:
        """Move a crowned model from its author's private prefix into the open.

        A challenger uploads where only it can read, so losing does not expose
        a checkpoint to the rivals that beat it. Winning reverses that: the
        model collecting emissions has to be fetchable and re-scorable by
        anyone, or the subnet is asking to be trusted rather than checked.

        A public `hf:` king needs nothing done — it is already readable — so
        only private submissions are republished.

        Failure is recorded and retried with the mirror rather than raised. A
        king that is crowned but not yet published is a transparency delay; a
        validator that halted on an upload error would be an outage.
        """
        king = self.state.king
        if king is None or king.ref.backend != "oci":
            return
        try:
            from epago.publishing.publisher import publish_king

            published = publish_king(king_dir, king.ref.digest)
            logger.info("king published for public audit at %s", published.repo)
        except Exception as exc:  # noqa: BLE001 - never halt the loop
            self.state.pending_mirror = True
            self.state.last_error = {
                "code": "king_publish_pending",
                "digest": king.ref.digest,
                "detail": f"{type(exc).__name__}: {exc}",
                "block": self._safe_block(),
            }

    def _retry_mirror(self) -> None:
        if self.state.pending_mirror and self.state.king is not None:
            self._mirror_current_king()

    def _drop_stale(self, new_king_digest: str) -> None:
        """After a dethrone every *undueled* challenger built on the old king
        drops at once — its measured improvement no longer means anything.

        Entrants that already ran in the round that produced this coronation
        keep the status their duel gave them. They were measured against the
        king that was reigning at the time, which is exactly what the mechanism
        asked of them; the round's runners-up are near-misses, not stale
        submissions, and they have arena credit riding on that distinction.
        """
        kept: list[QueuedSubmission] = []
        for sub in self.state.queue:
            if sub.king_digest != new_king_digest:
                self.state.statuses[sub.digest] = SubmissionStatus.STALE_PARENT.value
            else:
                kept.append(sub)
        self.state.queue = kept
        for digest, meta in list(self.state.candidates.items()):
            if meta.get("king_digest") != new_king_digest:
                if self.state.statuses.get(digest) == SubmissionStatus.QUEUED.value:
                    self.state.statuses[digest] = SubmissionStatus.STALE_PARENT.value
                self.state.candidates.pop(digest)

    # ----------------------------------------------------- pool & calibration

    def _ensure_pool_committed(self, block: int) -> bool:
        """Commit the active private pool on-chain before it grades anything.

        Returns True when this epoch's ``ep1`` is on chain. ``_process_one``
        holds off until it is: the commitment is what pins the validator to one
        exact task set while that set is still secret. Committing after the fact
        — which is what rotation used to do — let a validator pick its private
        tasks once it already knew what outcome it wanted to justify, and the
        chain stamp made that look rigorous.
        """
        pool = self.deps.private_pool
        if self.state.committed_pool_epoch == pool.epoch:
            return True
        commitment = getattr(pool, "commitment", None)
        if commitment is None:
            return True  # fakes and older pools carry no commitment channel
        try:
            payload = commitment()
        except Exception as exc:  # noqa: BLE001
            self.state.last_error = {
                "code": "pool_commitment_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }
            return False
        if not payload:
            return True
        if self.chain.publish_reveal(str(payload), constants.VERDICT_REVEAL_BLOCKS):
            self.state.committed_pool_epoch = pool.epoch
            return True
        # Rate-limited; retried next tick. No duel runs on an uncommitted pool.
        self.state.last_error = {
            "code": "pool_commitment_deferred",
            "detail": f"epoch {pool.epoch} commitment not accepted yet",
            "block": block,
        }
        return False

    def _public_tasks(self, seed: int, n: int) -> list:
        """The public half, from the generator or from a sealed pool.

        Which one is decided by the release name alone, so an audit record is
        self-describing: a replay reading it knows which verification path
        applies without being told separately.

        A sealed pool is loaded against the digest pinned in the contract. That
        pin is what makes the pool a commitment rather than a claim — it was
        fixed before the round opened, so the exam existed before the
        challenger's weights were frozen, and it cannot be swapped afterwards.
        Selection is still seeded by a block hash nobody chose, so *which*
        questions get asked is unknown even to whoever minted the pool.
        """
        from epago.taskgen.sealed_pool import is_sealed_release

        release = self.cfg.eval.taskgen_release
        if not is_sealed_release(release):
            return self.deps.generate_tasks(
                seed=seed,
                release=release,
                corpus=self.deps.corpus,
                n=n,
                king_probe=None,
            )

        from epago.taskgen.sealed_pool import SealedPoolError, load_pool, select

        path = getattr(self.cfg.eval, "public_pool_path", "")
        digest = getattr(self.cfg.eval, "public_pool_digest", "")
        if not path:
            raise SealedPoolError(
                f"release {release!r} is served from a sealed pool, but "
                "eval.public_pool_path is empty"
            )
        # Already-published tasks are excluded, so rounds are disjoint. A round
        # publishes its tasks in full once the delay elapses, which makes them
        # training data from that moment on; drawing them again would let a
        # challenger trained after that publication answer part of its exam
        # from memory rather than from research.
        return select(
            load_pool(path, digest), seed, n, exclude=set(self.state.served_public_task_ids)
        )

    def _stage_public_pool_round(self, round_no: int, public_tasks: list) -> None:
        """Publish a sealed-pool round's tasks, and retire them from the pool.

        Two things happen here and they are deliberately not separable.

        The tasks are staged for release on the same delay as the rest of the
        round record, because a sealed pool's questions are not derivable from
        a seed: releasing them is a real disclosure, where a generator-served
        exam was always reconstructible by anyone holding the corpus. Delayed
        transparency is what lets an auditor re-grade the exam without handing
        the next challenger a live answer key.

        The ids are then marked served, immediately and regardless of when the
        file actually releases. Marking at release time instead would leave a
        window in which a round could re-draw tasks already queued for
        publication — the leak has begun the moment the file is staged, not the
        moment it lands.

        Generator-served releases skip both: their tasks regenerate from a seed
        and the existing round record already pins them.
        """
        from epago.taskgen.sealed_pool import is_sealed_release, round_payload, round_stage_name

        if not is_sealed_release(self.cfg.eval.taskgen_release) or not public_tasks:
            return
        try:
            digest = self.deps.task_ids_digest(public_tasks)
            self.audit_log.stage_delayed(
                round_stage_name(round_no, digest),
                round_payload(
                    public_tasks,
                    round_no=round_no,
                    task_ids_digest=digest,
                    pool_digest_value=getattr(self.cfg.eval, "public_pool_digest", ""),
                    manifest_digest=getattr(self.cfg.eval, "public_pool_manifest_digest", ""),
                ),
                self.deps.clock() + constants.AUDIT_PUBLISH_DELAY_BLOCKS,
            )
        except Exception as exc:  # noqa: BLE001 - a staging fault must not void a verdict
            self.state.last_error = {
                "code": "public_pool_publish_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": self.state.last_round_block,
            }
            return
        # Only after staging succeeded: marking ids served on a path that
        # failed to stage would retire tasks no auditor will ever receive.
        served = set(self.state.served_public_task_ids)
        served.update(getattr(t, "task_id", str(t)) for t in public_tasks)
        self.state.served_public_task_ids = sorted(served)

    def _publish_pool_manifest(self) -> None:
        """Copy the sealed pool's task-id manifest into the published tree.

        The manifest is the only artifact an auditor can redraw a round's
        selection from, so a verdict is checkable exactly as far as this file
        is reachable. It lives wherever the operator minted it, which is
        outside every directory the publisher syncs; without this step the
        replay's ``tasks`` check reports SKIP for every verdict and the
        sealed-pool audit trail does not work for anyone but us.

        Run every tick rather than once at startup, because the contract can be
        pointed at a different pool while the box is running. A published
        manifest that no longer matches the pinned digest is worse than none at
        all: an auditor would redraw from the wrong id list and see failures
        that are not real. Copying is a no-op once the bytes match, so the cost
        is one comparison per tick.

        The digest is verified before anything is copied. A manifest that does
        not match its commitment is refused loudly and left unpublished --
        publishing it would hand auditors a file the contract does not vouch
        for.

        Ids only: no question and no answer is disclosed by this file, so it
        publishes immediately rather than on the transparency delay that
        governs a round's actual questions.
        """
        from epago.taskgen.sealed_pool import SealedPoolError, is_sealed_release, load_manifest

        if not is_sealed_release(self.cfg.eval.taskgen_release):
            return
        source = str(getattr(self.cfg.eval, "public_pool_manifest_path", "") or "")
        digest = str(getattr(self.cfg.eval, "public_pool_manifest_digest", "") or "")
        if not source:
            return

        try:
            # Verified against the pinned digest, not merely read: this is the
            # step that stops a stale or swapped manifest reaching auditors.
            load_manifest(source, digest)
            payload = Path(source).read_bytes()
            target = self.state.state_dir / "publications" / Path(source).name
            if target.exists() and target.read_bytes() == payload:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(payload)
            tmp.replace(target)
            logger.info("published pool manifest %s (%s)", target.name, digest[:23])
        except (SealedPoolError, OSError) as exc:
            self.state.last_error = {
                "code": "pool_manifest_publish_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": self.deps.clock(),
            }

    def _maybe_publish_mailbox(self, block: int) -> None:
        """Re-issue upload credentials for every registered miner.

        This is what makes private submission possible: a miner cannot upload
        into the validator's bucket without a credential, and there is no
        private channel to hand one over. So every miner's credential is sealed
        to its own hotkey and all of them are published in one public file.

        Republished on a cadence well inside the credential lifetime, because
        the credentials expire. That is deliberate — a leaked key is then
        bounded in time rather than valid forever, and a miner that never
        submits stops holding a live one.

        Entirely optional. With no object store configured, nothing is
        published and public Hugging Face submission continues to work
        unchanged; private upload is a choice about who can read a challenger
        before it wins, not a different protocol.

        Never raises. A validator that stopped scoring because a credential
        refresh failed would be trading a working subnet for a storage
        problem.
        """
        import os

        interval = int(os.environ.get("EPAGO_MAILBOX_INTERVAL_BLOCKS", "600"))
        last = self.state.last_mailbox_block
        if last is not None and block - last < interval:
            return
        parent_key = os.environ.get("EPAGO_R2_PARENT_ACCESS_KEY", "").strip()
        parent_secret = os.environ.get("EPAGO_R2_PARENT_SECRET_KEY", "").strip()
        account_id = os.environ.get("EPAGO_R2_ACCOUNT_ID", "").strip()
        bucket = os.environ.get("EPAGO_S3_BUCKET", "").strip()
        endpoint = os.environ.get("EPAGO_S3_ENDPOINT", "").strip()
        if not all((parent_key, parent_secret, account_id, bucket, endpoint)):
            return  # private upload is not configured on this box

        try:
            from epago.chain.credentials import mint_upload_credentials
            from epago.chain.mailbox import (
                DEFAULT_TTL_SECONDS,
                MAILBOX_KEY,
                build_mailbox,
                write_mailbox,
            )

            recipients = self._mailbox_recipients()
            if not recipients:
                self.state.last_mailbox_block = block
                return

            def issue(hotkey: str, prefix: str, expires_at: int) -> dict:
                creds = mint_upload_credentials(
                    endpoint=endpoint,
                    account_id=account_id,
                    parent_access_key_id=parent_key,
                    parent_secret_access_key=parent_secret,
                    bucket=bucket,
                    prefix=prefix,
                    ttl_seconds=DEFAULT_TTL_SECONDS,
                )
                return creds.as_payload(endpoint=endpoint, bucket=bucket, prefix=prefix)

            mailbox = build_mailbox(
                recipients, issue, signer_seed=self._validator_signing_seed()
            )
            path = self.state.state_dir / "publications" / MAILBOX_KEY
            digest = write_mailbox(mailbox, path)
            self.state.last_mailbox_block = block
            self.state.mailbox_digest = digest
            logger.info(
                "published credential mailbox for %d miners (%s)",
                len(mailbox.envelopes),
                digest[:23],
            )
        except Exception as exc:  # noqa: BLE001 - never halt the loop
            self.state.last_error = {
                "code": "mailbox_publish_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }

    def _mailbox_recipients(self) -> dict[str, bytes]:
        """Registered miners that can receive a sealed credential.

        An ss58 address *is* a 32-byte public key with a checksum, so the key
        needs no extra chain query and no registration step — it is already in
        the metagraph.

        What the address does not say is the curve. sr25519 and Ed25519 keys are
        both 32 bytes and look alike, and only Ed25519 converts to X25519 for
        sealing. That is resolved by trying: :func:`build_mailbox` skips a
        recipient whose key will not seal, so an sr25519 hotkey is simply
        absent from the mailbox rather than handed an envelope it can never
        open. Intake tells such a miner why, so the silence is explained
        somewhere it will look.
        """
        try:
            from scalecodec.utils.ss58 import ss58_decode
        except ImportError:  # pragma: no cover - environment-dependent
            return {}
        recipients: dict[str, bytes] = {}
        for neuron in self.chain.neurons():
            declared = getattr(neuron, "ed25519_public", None)
            if declared:
                recipients[neuron.hotkey] = bytes(declared)
                continue
            try:
                recipients[neuron.hotkey] = bytes.fromhex(ss58_decode(neuron.hotkey))
            except Exception:  # noqa: BLE001 - an undecodable address is not a recipient
                continue
        return recipients

    def _validator_signing_seed(self) -> bytes | None:
        """This validator's Ed25519 seed, for signing credential payloads.

        Absent is not fatal: envelopes go out unsigned and a miner can still
        open them. It does mean a miner cannot distinguish this validator's
        credentials from a forgery, so the absence is worth logging loudly.
        """
        wallet = getattr(self.deps, "wallet", None)
        seed = getattr(getattr(wallet, "hotkey", None), "private_key", None)
        return bytes(seed)[:32] if seed else None

    def _rotate_private_pool(self, block: int) -> None:
        pool = self.deps.private_pool
        rotation_due = getattr(pool, "rotation_due", None)
        if rotation_due is None or not rotation_due(block):
            return
        try:
            payload = pool.rotate(block)
        except Exception as exc:  # noqa: BLE001
            self.state.last_error = {
                "code": "pool_rotation_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }
            return
        # The outgoing pool has been written out in full (delayed transparency);
        # the payload commits the INCOMING pool, before it grades a single duel.
        if payload and self.chain.publish_reveal(str(payload), constants.VERDICT_REVEAL_BLOCKS):
            self.state.committed_pool_epoch = pool.epoch
        self.state.last_pool_publish_block = block

    def _maybe_calibrate(self, block: int) -> None:
        """Every N ticks: king-vs-king on fresh tasks. Any nonzero disagreement
        is pure harness noise and feeds the adaptive-delta noise floor."""
        if self.state.tick_count == 0 or self.state.tick_count % CALIBRATION_EVERY_TICKS != 0:
            return
        assert self.state.king is not None
        try:
            king_dir = self._materialize(self.state.king.ref)
            seed = derive_seed(self.chain.block_hash(block), self.deps.wallet_hotkey, b"calibration")
            tasks = self.deps.generate_tasks(
                seed=seed,
                release=self.cfg.eval.taskgen_release,
                corpus=self.deps.corpus,
                n=constants.N_PUB_TASKS,
                king_probe=None,
            )
            # The judge rides along: it is part of the graded path a real duel
            # takes, so excluding it would measure a noise floor the duel does
            # not actually run at.
            rate = float(
                self.deps.run_calibration_duel(
                    king_dir,
                    tasks,
                    self.deps.env,
                    self.deps.backend_factory,
                    self.deps.llm_judge,
                )
            )
            self.state.add_noise_sample(rate)
        except Exception as exc:  # noqa: BLE001 - calibration is opportunistic
            self.state.last_error = {
                "code": "calibration_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }

    def _maybe_anchor(self, block: int) -> None:
        """External benchmark anchor — the eval-of-the-eval (opt-in, observational).

        When ``EPAGO_ANCHOR_BENCHMARK`` points at an existing JSONL benchmark
        file, run the current king over it every ``ANCHOR_INTERVAL_BLOCKS``
        (bounded by ``EPAGO_ANCHOR_MAX_TASKS``, default 100) and record the
        drift between internal EMA gain and anchor accuracy gain. Divergence
        above ``ANCHOR_DIVERGENCE_ALERT`` sets a machine-readable alarm and
        nothing else: anchor results never touch verdicts, weights, or
        coronation, and a failed run degrades exactly like calibration.
        """
        benchmark = os.environ.get("EPAGO_ANCHOR_BENCHMARK", "").strip()
        if not benchmark or not Path(benchmark).exists():
            return
        if block - self.state.last_anchor_block < constants.ANCHOR_INTERVAL_BLOCKS:
            return
        assert self.state.king is not None
        try:
            from epago.eval.anchor import divergence, run_anchor  # keeps eval late-bound

            king_dir = self._materialize(self.state.king.ref)
            # The benchmark's questions need the benchmark's own corpus behind
            # the tools (EPAGO_ANCHOR_CORPUS, e.g. the FRAMES wiki snapshot);
            # falling back to the duel env would handicap every king equally
            # but make the absolute score meaningless against published runs.
            anchor_env = self.deps.env
            anchor_corpus = os.environ.get("EPAGO_ANCHOR_CORPUS", "").strip()
            if anchor_corpus and Path(anchor_corpus).exists():
                from epago.environment.corpus import SqliteCorpus
                from epago.environment.services import ResearchEnvironment

                anchor_env = ResearchEnvironment(SqliteCorpus(anchor_corpus))
            report = run_anchor(
                king_dir,
                Path(benchmark),
                self.deps.backend_factory,
                env=anchor_env,
                llm_judge=self.deps.llm_judge,
                max_tasks=int(os.environ.get("EPAGO_ANCHOR_MAX_TASKS", "100")),
            )
            record: dict[str, Any] = {
                "block": block,
                "king_digest": self.state.king.ref.digest,
                "benchmark_digest": report.benchmark_digest,
                "accuracy": report.accuracy,
                "n_tasks": report.n_tasks,
                "internal_ema_at_run": self.state.king_acc_ema,
            }
            drift = divergence(self.state.king_acc_ema, [*self.state.anchor_history, record])
            record["divergence"] = drift
            self.state.anchor_history.append(record)
            self.state.last_anchor_block = block
            # Anchor scores carry no holdout, so they publish into the audit
            # trail immediately (released by this tick's housekeeping pass).
            self.audit_log.stage_delayed(
                f"anchor-{block}", json.dumps(record, sort_keys=True), block
            )
            if drift is not None and drift > constants.ANCHOR_DIVERGENCE_ALERT:
                self.state.last_error = {
                    "code": "anchor_divergence",
                    "divergence": drift,
                    "threshold": constants.ANCHOR_DIVERGENCE_ALERT,
                    "block": block,
                }
        except Exception as exc:  # noqa: BLE001 - the anchor is an alarm, never a halt
            self.state.last_error = {
                "code": "anchor_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }

    # ----------------------------------------------------------------- weights

    def _succession_from_chain(self, block: int) -> list[tuple[str, str, int]]:
        """The coronation order, read from chain: ``(digest, hotkey, block)``.

        Accepted ``ev3`` verdicts ARE the coronation record, so the throne's
        whole history is public chain state. Deriving it here rather than
        trusting a published pointer is what lets a validator that runs no
        duels compute the same weight vector as the one that does: it reads
        the same verdicts, walks the same succession, and arrives at the same
        king and the same arena. No authority publishes anything, and no
        central API is consulted.

        Raises nothing: on a failed chain read the caller falls back to local
        state, so a transient RPC error degrades rather than zeroing the board.
        """
        verdicts = self.chain.read_verdicts()
        evaluators = {e.hotkey for e in self.chain.evaluators(self._active_window_blocks)}
        authors = {
            r.challenger.digest: r.author_hotkey
            for r in self.chain.read_revealed_submissions(0)
        }
        # One coronation per challenger, at the block it was FIRST accepted.
        # A second validator confirming the same crowning is corroboration,
        # not a second coronation.
        crowned_at: dict[str, int] = {}
        for v in sorted(verdicts, key=lambda x: x.block):
            if v.validator_hotkey not in evaluators:
                continue
            if v.decision is not VerdictDecision.ACCEPT:
                continue
            crowned_at.setdefault(v.challenger_digest, v.block)

        out: list[tuple[str, str, int]] = []
        for digest, crowned_block in sorted(crowned_at.items(), key=lambda kv: kv[1]):
            hotkey = authors.get(digest)
            if hotkey is None:
                continue  # no live reveal to attribute the crown to
            out.append((digest, hotkey, crowned_block))
        return out

    def _king_from_chain(self, block: int) -> tuple[str, str, int] | None:
        """The reigning king, derived from chain. ``None`` before any coronation.

        This is what removes the need for a king-authority hotkey. A pointer
        published by a named authority would say the same thing, but it would
        also make the throne depend on one box staying alive and honest; the
        succession is already on chain, so reading it is both simpler and
        harder to lie about.
        """
        try:
            succession = self._succession_from_chain(block)
        except Exception as exc:  # noqa: BLE001 - a chain read never halts the loop
            self.state.last_error = {
                "code": "king_derivation_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }
            return None
        return succession[-1] if succession else None

    def _arena_entries(self, block: int) -> list[ArenaEntry]:
        """The former-kings roster, derived from the same chain succession.

        Every validator has to reach the same arena split or their weight
        vectors disagree and Yuma penalises the lot. Local state only ever
        holds this box's own view, so an auditing validator -- which runs no
        duels -- would compute an empty arena while the scoring validator paid
        one out.
        """
        try:
            succession = self._succession_from_chain(block)
        except Exception as exc:  # noqa: BLE001 - never halt weight setting
            self.state.last_error = {
                "code": "arena_derivation_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": block,
            }
            return list(self.state.arena)

        # Each coronation dethrones the previous holder, and it is the
        # DETHRONED one that takes a seat.
        entries: list[ArenaEntry] = []
        previous: tuple[str, int] | None = None
        for _, hotkey, crowned_block in succession:
            if previous is not None and previous[0] != hotkey:
                # A self-dethrone is not a displacement: the same hotkey still
                # wears the crown, and seating it in its own arena would pay
                # one party twice out of a budget meant to reward being beaten.
                entries.append(
                    ArenaEntry(
                        hotkey=previous[0],
                        dethroned_block=crowned_block,
                        reign_blocks=max(crowned_block - previous[1], 0),
                    )
                )
            previous = (hotkey, crowned_block)

        reigning = succession[-1][1] if succession else ""
        return trim_arena([e for e in entries if e.hotkey != reigning])

    def _king_emission_state(self, block: int) -> "KingEmissionState | None":
        """The king the weight vector should pay, preferring chain over local.

        A validator that runs no duels has no local king, so without this it
        would compute an empty board and burn everything while the scoring
        validator paid a champion -- a straight divergence that Yuma penalises
        both for. The succession is public, so it is read rather than trusted
        to a published pointer.

        When both exist and disagree, the chain wins and the difference is
        recorded. Local state losing to chain is the safe direction: it means
        this box crowned something the network did not, and paying the
        network's king keeps the vector in consensus while the alarm is
        visible in status output.
        """
        local = self.state.king
        derived = self._king_from_chain(block)

        if derived is None:
            if local is None or not local.author_hotkey:
                return None
            return KingEmissionState(
                hotkey=local.author_hotkey,
                reign_started_block=local.reign_started_block,
                crowned_block=local.crowned_block,
                coronation_lcb=local.coronation_lcb,
                coronation_delta=self.state.king_coronation_delta,
            )

        digest, hotkey, crowned_block = derived
        if local is not None and local.ref.digest != digest:
            self.state.last_error = {
                "code": "king_disagrees_with_chain",
                "detail": f"local {local.ref.digest[:16]} vs chain {digest[:16]}",
                "block": block,
            }
            logger.warning(
                "[weights] local king %s disagrees with the chain succession %s; "
                "paying the chain's king",
                local.ref.digest[:16],
                digest[:16],
            )

        # Reign start: the earliest consecutive coronation by this same hotkey,
        # so a self-dethrone does not reset the bleed clock. That rule already
        # governs local state; deriving it the same way keeps a chain-reading
        # validator on the identical emission curve.
        reign_started = crowned_block
        try:
            succession = self._succession_from_chain(block)
        except Exception:  # noqa: BLE001 - already reported by the caller path
            succession = []
        for _, prev_hotkey, prev_block in reversed(succession[:-1]):
            if prev_hotkey != hotkey:
                break
            reign_started = prev_block

        # Coronation terms are per-validator, so a chain-reading box has none.
        # Local terms are used when they describe this same king; otherwise the
        # bonus is simply absent, which costs the king a little and never pays
        # anyone a share they did not earn.
        if local is not None and local.ref.digest == digest:
            lcb, delta = local.coronation_lcb, self.state.king_coronation_delta
        else:
            lcb, delta = 0.0, 0.0

        return KingEmissionState(
            hotkey=hotkey,
            reign_started_block=reign_started,
            crowned_block=crowned_block,
            coronation_lcb=lcb,
            coronation_delta=delta,
        )

    def maybe_set_weights(self) -> None:
        """Set weights every ``WEIGHT_INTERVAL_BLOCKS``.

        Phase gate: before :func:`epago.core.emissions.phase_b_active` the full
        emission burns (weight 1.0 on the burn key). In Phase B the weight
        vector is :func:`epago.core.emissions.compute_weights` over the king
        emission state and arena entries.
        Bad-faith challengers are disciplined by intake cooldowns (see
        :mod:`epago.validator.intake`), not by weight manipulation.
        """
        block = self.deps.clock()
        last = self.state.last_weights_block
        if last is not None and block - last < constants.WEIGHT_INTERVAL_BLOCKS:
            return
        neurons = self.chain.neurons()
        if not neurons:
            return
        uid_by_hotkey = {n.hotkey: n.uid for n in neurons}
        # An explicitly configured burn hotkey, or the lowest UID as a fallback.
        # The fallback is not a burn — UID 0 is an ordinary registered neuron,
        # and in Phase A it would collect the subnet's entire emission — so it
        # is announced rather than assumed.
        burn_neuron = min(neurons, key=lambda n: n.uid)
        burn_hotkey = self.cfg.chain.burn_hotkey or burn_neuron.hotkey
        if not self.cfg.chain.burn_hotkey and not self._warned_burn_fallback:
            logger.warning(
                "[chain] burn_hotkey is unset; unallocatable emission goes to uid %d (%s), "
                "which is a live neuron and not a burn",
                burn_neuron.uid,
                burn_neuron.hotkey,
            )
            self._warned_burn_fallback = True

        phase_b = phase_b_active(
            clean_duels=self.state.clean_duels,
            organic_dethrones=self.state.organic_dethrones,
            blocks_since_genesis=block - self.state.genesis_block,
            min_duels=constants.PHASE_B_MIN_CLEAN_DUELS,
            min_dethrones=constants.PHASE_B_MIN_DETHRONES,
            min_blocks=constants.PHASE_B_MIN_BLOCKS,
        )
        if not phase_b:
            hotkey_weights = {burn_hotkey: 1.0}
        else:
            king_emission = self._king_emission_state(block)
            from epago.core.emissions import compute_weights

            hotkey_weights = compute_weights(
                king_emission,
                self._arena_entries(block),
                block,
                self.cfg.emissions,
                burn_hotkey,
            )

        uid_weights: dict[int, float] = {}
        for hotkey, weight in hotkey_weights.items():
            uid = uid_by_hotkey.get(hotkey, burn_neuron.uid)  # unmapped mass burns
            uid_weights[uid] = uid_weights.get(uid, 0.0) + weight
        self.chain.set_weights(uid_weights)
        self.state.last_weights_block = block

    # -------------------------------------------------------------------- audit

    def _audit_housekeeping(self, block: int) -> None:
        if self.audit_log.chain_checkpoint_due():
            self.chain.publish_status(self.audit_log.checkpoint_payload())
            self.audit_log.mark_checkpointed()
        self.audit_log.release_due(block)

    # ---------------------------------------------------------------------- SLA

    def sla_report(self) -> dict:
        return compute_sla_report(self.state.sla)

    # ------------------------------------------------------------------- misc

    def _safe_block(self) -> int:
        try:
            return self.deps.clock()
        except Exception:  # noqa: BLE001
            return -1
