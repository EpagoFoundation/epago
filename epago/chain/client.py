"""Thin chain adapter, written against bittensor SDK 10.x.

All SDK access goes through :class:`ChainClient` so the rest of the codebase
stays testable against :class:`MockChainClient`.

On-chain channels — chosen to match what the chain actually provides:

* **Timelock-reveal channel** (``set_reveal_commitment`` /
  ``get_all_revealed_commitments``): multi-entry per hotkey, each entry
  chain-stamped with its reveal block. Everything that must accumulate or be
  trustlessly ordered flows here: miner ``e2`` challenges, validator ``ev3``
  verdicts, ``ep1`` pool commitments, ``ek1`` king pointers, and ``er1`` round
  starts.
  Verdict blocks therefore come from the chain, never from self-report.
* **Plaintext commitment slot** (``set_commitment`` / ``get_all_commitments``):
  ONE string per hotkey, hard-capped at 128 bytes, newest write wins. Only the
  compact ``es1`` status (audit checkpoint + active pool digest) lives here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from epago import constants
from epago.core.reveal import (
    WireFormatError,
    parse_king_pointer,
    parse_reveal,
    parse_round_start,
    parse_verdict,
)
from epago.core.types import (
    ChallengeReveal,
    EvaluatorInfo,
    KingPointer,
    RoundStart,
    Verdict,
)

logger = logging.getLogger(__name__)

STATUS_MAX_BYTES = 128


def _decode_revealed_commitment(com: object) -> str:
    """Decode one revealed-commitment blob into its UTF-8 payload.

    The chain stores a SCALE compact-length prefix followed by the payload bytes.
    substrate-interface hands the blob back as either a ``0x``-hex string or an
    already-decoded byte-string (both still carrying the compact-length prefix), so
    we normalise to bytes, strip the prefix by its compact mode, and decode. The
    SDK's own helper assumes hex-only and crashes on the raw form.
    """
    if isinstance(com, (bytes, bytearray)):
        raw = bytes(com)
    elif isinstance(com, (list, tuple)) and com and all(isinstance(x, int) for x in com):
        # This runtime hands the blob back as a tuple of byte values, not bytes
        # or a hex string; normalise straight to bytes (prefix stripped below).
        raw = bytes(b & 0xFF for b in com)
    else:
        s = str(com)
        if s.startswith("0x"):
            try:
                raw = bytes.fromhex(s[2:])
            except ValueError:
                raw = s.encode("latin-1", "ignore")
        else:
            raw = s.encode("latin-1", "ignore")
    if not raw:
        return ""
    mode = raw[0] & 0b11
    offset = 1 if mode == 0 else 2 if mode == 1 else 4
    return bytes(raw[offset:]).decode("utf-8", errors="ignore")


@dataclass(frozen=True, slots=True)
class NeuronView:
    uid: int
    hotkey: str
    coldkey: str
    stake: float
    validator_permit: bool


@dataclass(frozen=True, slots=True)
class RevealedPayload:
    hotkey: str
    block: int
    payload: str


class ChainClient(ABC):
    @abstractmethod
    def current_block(self) -> int: ...

    @abstractmethod
    def block_hash(self, block: int) -> str: ...

    @abstractmethod
    def neurons(self) -> list[NeuronView]: ...

    @abstractmethod
    def read_revealed_payloads(self, since_block: int = 0) -> list[RevealedPayload]:
        """Every timelock-revealed entry on the subnet since a block."""

    @abstractmethod
    def publish_reveal(self, payload: str, blocks_until_reveal: int) -> bool:
        """Publish through the timelock-reveal channel (multi-entry, chain-stamped).

        Returns True if the commit landed. The commitment pallet rate-limits writes
        per hotkey, and the SDK reports a throttled write as an unsuccessful response
        rather than raising — callers must check the result and retry, or a verdict
        can be silently lost (and coronation never derived)."""

    @abstractmethod
    def publish_status(self, payload: str) -> None:
        """Write the plaintext status slot (single entry, <=128 bytes, newest wins)."""

    @abstractmethod
    def read_status(self) -> dict[str, str]:
        """hotkey -> current plaintext status slot."""

    @abstractmethod
    def set_weights(self, weights: dict[int, float]) -> None:
        """Weights extrinsic; the SDK carries commit-reveal (v4) internally."""

    # ---- derived views (shared by all implementations) -----------------------

    def read_revealed_submissions(self, since_block: int = 0) -> list[ChallengeReveal]:
        """Latest valid ``e2`` challenge per hotkey.

        ``since_block`` bounds the *scan*, not the supersession rule. Resolving
        "latest per hotkey" over a window would make a validator's queue depend
        on its polling cadence — a box that just restarted (window from block 0)
        and a box ticking every 30 s would admit different challenges from
        identical chain state. Callers that need correct supersession pass 0;
        the parameter stays for cheap incremental reads that do not feed intake.

        The author is ``rp.hotkey``, the hotkey the chain recorded as signing
        the commitment. It is never taken from the payload.
        """
        latest: dict[str, RevealedPayload] = {}
        for rp in self.read_revealed_payloads(since_block):
            if not rp.payload.startswith(constants.REVEAL_VERSION + "|"):
                continue
            prev = latest.get(rp.hotkey)
            if prev is None or rp.block > prev.block:
                latest[rp.hotkey] = rp
        out: list[ChallengeReveal] = []
        for rp in latest.values():
            try:
                out.append(
                    parse_reveal(
                        rp.payload,
                        rp.block,
                        self.block_hash(rp.block),
                        author_hotkey=rp.hotkey,
                    )
                )
            except WireFormatError as exc:
                logger.warning("dropping malformed reveal from %s: %s", rp.hotkey, exc)
        return out

    def read_king_pointer(self, authority_hotkey: str) -> KingPointer | None:
        """The newest ``ek1`` king pointer published by ``authority_hotkey``.

        Pointers from any other hotkey are ignored outright: the pointer names
        who earns the king's share, so accepting one from an arbitrary neuron
        would hand the throne to whoever wrote last. An empty authority disables
        the lookup rather than trusting everyone.
        """
        if not authority_hotkey:
            return None
        best: KingPointer | None = None
        for rp in self.read_revealed_payloads(0):
            if rp.hotkey != authority_hotkey:
                continue
            if not rp.payload.startswith(constants.KING_POINTER_VERSION + "|"):
                continue
            try:
                pointer = parse_king_pointer(rp.payload, publisher_hotkey=rp.hotkey, block=rp.block)
            except WireFormatError as exc:
                logger.warning("dropping malformed king pointer from %s: %s", rp.hotkey, exc)
                continue
            # Tie-break by crowned_block so two pointers landing in one block
            # still resolve identically for every reader.
            key = (pointer.block, pointer.crowned_block)
            if best is None or key > (best.block, best.crowned_block):
                best = pointer
        return best

    def read_round_starts(self, authority_hotkey: str) -> list[RoundStart]:
        """Every valid ``er1`` round start from the authority, oldest first.

        Rounds published by any other hotkey are ignored: the trigger decides
        when the field is evaluated and on which exam, so honouring an arbitrary
        neuron's round would hand that neuron the schedule. Two further rules
        are enforced here rather than at the caller, so every validator derives
        the same round history from the same chain data:

        * round numbers must strictly increase — a replayed or duplicated number
          is dropped;
        * consecutive starts must be at least
          ``constants.ROUND_MIN_INTERVAL_BLOCKS`` apart, so the cadence is a
          property of the chain and not of how often the authority runs the
          command.
        """
        if not authority_hotkey:
            return []
        out: list[RoundStart] = []
        for rp in self.read_revealed_payloads(0):
            if rp.hotkey != authority_hotkey:
                continue
            if not rp.payload.startswith(constants.ROUND_START_VERSION + "|"):
                continue
            try:
                start = parse_round_start(
                    rp.payload,
                    authority_hotkey=rp.hotkey,
                    block=rp.block,
                    block_hash=self.block_hash(rp.block),
                )
            except WireFormatError as exc:
                logger.warning("dropping malformed round start from %s: %s", rp.hotkey, exc)
                continue
            if out:
                previous = out[-1]
                if start.round <= previous.round:
                    logger.warning(
                        "dropping round %d from %s: not newer than %d",
                        start.round,
                        rp.hotkey,
                        previous.round,
                    )
                    continue
                if start.block - previous.block < constants.ROUND_MIN_INTERVAL_BLOCKS:
                    logger.warning(
                        "dropping round %d from %s: only %d blocks after round %d, minimum is %d",
                        start.round,
                        rp.hotkey,
                        start.block - previous.block,
                        previous.round,
                        constants.ROUND_MIN_INTERVAL_BLOCKS,
                    )
                    continue
            out.append(start)
        return out

    def latest_round(self, authority_hotkey: str) -> RoundStart | None:
        rounds = self.read_round_starts(authority_hotkey)
        return rounds[-1] if rounds else None

    def read_verdicts(self, since_block: int = 0) -> list[Verdict]:
        """All ``ev3`` verdicts from validator-permit hotkeys.

        The verdict block is the chain's reveal block — a validator cannot
        backdate or forward-date its verdicts.
        """
        permitted = {n.hotkey for n in self.neurons() if n.validator_permit}
        verdicts: list[Verdict] = []
        for rp in self.read_revealed_payloads(since_block):
            if not rp.payload.startswith(constants.VERDICT_VERSION + "|"):
                continue
            if rp.hotkey not in permitted:
                continue
            try:
                verdicts.append(parse_verdict(rp.payload, validator_hotkey=rp.hotkey, block=rp.block))
            except WireFormatError as exc:
                logger.warning("dropping malformed verdict from %s: %s", rp.hotkey, exc)
        return verdicts

    def evaluators(self, active_window_blocks: int) -> list[EvaluatorInfo]:
        current = self.current_block()
        last_verdict: dict[str, int] = {}
        for v in self.read_verdicts():
            last_verdict[v.validator_hotkey] = max(last_verdict.get(v.validator_hotkey, 0), v.block)
        infos = []
        for n in self.neurons():
            if not n.validator_permit:
                continue
            lvb = last_verdict.get(n.hotkey, 0)
            if lvb and current - lvb <= active_window_blocks:
                infos.append(EvaluatorInfo(hotkey=n.hotkey, stake=n.stake, last_verdict_block=lvb))
        return infos


class BittensorChainClient(ChainClient):
    """Production client over the bittensor SDK (verified against 9.12.2).

    Runtime-API reads (``neurons_lite``) are read raw from storage when the
    SDK's encoder drifts against a live-upgraded runtime; see
    :meth:`_neurons_via_storage`. Extrinsics and storage reads go unchanged.
    """

    def __init__(self, wallet, netuid: int, network: str) -> None:
        import bittensor as bt

        self.wallet = wallet
        self.netuid = netuid
        self.subtensor = bt.Subtensor(network=network)
        self._install_hyperparam_fallback()
        if constants.COMMIT_REVEAL_REQUIRED and not self.subtensor.commit_reveal_enabled(netuid):
            raise RuntimeError(
                f"commit-reveal weights is not enabled on netuid {netuid}; "
                "refusing to start (COMMIT_REVEAL_REQUIRED). Enable it with: "
                f"btcli sudo set --netuid {netuid} --param commit_reveal_weights_enabled --value 1"
            )
        self._neurons_cache: list[NeuronView] = []
        self._neurons_cache_block = -1

    def _install_hyperparam_fallback(self) -> None:
        """Make commit-reveal ``set_weights`` survive runtime-API drift.

        The SDK's set_weights path calls ``get_subnet_hyperparameters`` (a
        runtime API whose bare-u16 netuid the live runtime wants wrapped as a
        Composite, so it raises) purely to read ``tempo`` and
        ``commit_reveal_period``. Wrap it to fall back to raw storage for those
        two fields; the drand encryption and the commit extrinsic then submit
        normally (same storage/extrinsic paths already proven working).
        """
        from types import SimpleNamespace

        st = self.subtensor
        original = st.get_subnet_hyperparameters

        def _fallback(netuid, block=None):
            try:
                return original(netuid, block=block)
            except Exception:  # noqa: BLE001 - runtime-API encoding drift
                sub = st.substrate

                def _one(sf: str) -> int:
                    v = sub.query("SubtensorModule", sf, [netuid]).value
                    return int(v[0] if isinstance(v, (list, tuple)) else v)

                return SimpleNamespace(
                    tempo=_one("Tempo"),
                    commit_reveal_period=_one("RevealPeriodEpochs"),
                )

        st.get_subnet_hyperparameters = _fallback

    def current_block(self) -> int:
        return int(self.subtensor.get_current_block())

    def block_hash(self, block: int) -> str:
        return str(self.subtensor.get_block_hash(block))

    def neurons(self) -> list[NeuronView]:
        block = self.current_block()
        if block == self._neurons_cache_block:
            return self._neurons_cache
        try:
            views = self._neurons_via_sdk()
        except Exception as exc:  # noqa: BLE001 - NeuronInfoRuntimeApi drift; go raw
            logger.debug("neurons_lite unavailable (%s); reading raw storage", type(exc).__name__)
            views = self._neurons_via_storage()
        self._neurons_cache, self._neurons_cache_block = views, block
        return views

    def _neurons_via_sdk(self) -> list[NeuronView]:
        return [
            NeuronView(
                uid=int(n.uid),
                hotkey=str(n.hotkey),
                coldkey=str(n.coldkey),
                stake=float(getattr(n.total_stake, "tao", n.total_stake)),
                validator_permit=bool(n.validator_permit),
            )
            for n in self.subtensor.neurons_lite(self.netuid)
        ]

    def _neurons_via_storage(self) -> list[NeuronView]:
        """Reconstruct the neuron set from raw storage.

        The SDK's ``neurons_lite`` calls ``NeuronInfoRuntimeApi``, whose param
        encoding drifts against a live-upgraded runtime (a bare ``u16`` netuid the
        current runtime wants wrapped as a Composite). Storage reads do not go
        through that path — same class of breakage already worked around in
        :meth:`read_revealed_payloads`. uid<->hotkey from ``Keys``, coldkey from
        ``Owner``, per-subnet stake from ``TotalHotkeyAlpha``, validator permit
        from the netuid's permit vector.
        """
        from bittensor.core.chain_data.utils import decode_account_id

        sub = self.subtensor.substrate
        permit = list(sub.query("SubtensorModule", "ValidatorPermit", [self.netuid]).value)
        views: list[NeuronView] = []
        for uid, hk in sub.query_map("SubtensorModule", "Keys", [self.netuid]):
            uid = int(uid)
            hotkey = decode_account_id(hk.value[0]) if hasattr(hk, "value") else str(hk)
            try:
                coldkey = str(decode_account_id(sub.query("SubtensorModule", "Owner", [hotkey]).value))
            except Exception:  # noqa: BLE001 - coldkey is not load-bearing for scoring
                coldkey = ""
            try:
                raw = sub.query("SubtensorModule", "TotalHotkeyAlpha", [hotkey, self.netuid]).value
                if isinstance(raw, (list, tuple)):  # runtime returns the u64 wrapped
                    raw = raw[0] if raw else 0
                stake = float(raw) / 1e9
            except Exception:  # noqa: BLE001 - stake feeds relative quorum weight only
                stake = 0.0
            views.append(
                NeuronView(
                    uid=uid,
                    hotkey=str(hotkey),
                    coldkey=coldkey,
                    stake=stake,
                    validator_permit=bool(permit[uid]) if uid < len(permit) else False,
                )
            )
        views.sort(key=lambda v: v.uid)
        return views

    def read_revealed_payloads(self, since_block: int = 0) -> list[RevealedPayload]:
        # The SDK's get_all_revealed_commitments / get_revealed_commitment_by_hotkey
        # crash on this chain runtime: substrate-interface returns revealed Bytes in a
        # MIX of 0x-hex and already-decoded byte-string forms, and the SDK decoder
        # assumes hex (bytes.fromhex) all-or-nothing — one malformed neighbour blinds
        # the whole read. We query the raw storage and decode each entry defensively,
        # so a foreign or malformed commitment can never stall our intake.
        try:
            qmap = self.subtensor.substrate.query_map(
                module="Commitments",
                storage_function="RevealedCommitments",
                params=[self.netuid],
            )
        except Exception as exc:  # noqa: BLE001 - a read failure degrades, never crashes the loop
            logger.warning("read_revealed_payloads: raw storage query failed: %s", exc)
            return []
        from bittensor.core.chain_data.utils import decode_account_id

        out: list[RevealedPayload] = []
        for key, value in qmap:
            raw_key = getattr(key, "value", key)
            # This runtime hands the map key back as the raw 32-byte account
            # (sometimes wrapped one level); decode to ss58 so intake can match
            # it against the registered neuron set.
            inner = (
                raw_key[0]
                if isinstance(raw_key, (list, tuple)) and raw_key and isinstance(raw_key[0], (list, tuple))
                else raw_key
            )
            try:
                hotkey = decode_account_id(inner)
            except Exception:  # noqa: BLE001 - fall back to the raw form, never crash
                hotkey = str(raw_key)
            for entry in getattr(value, "value", value) or []:
                try:
                    block = int(entry[1])
                    if block < since_block:
                        continue
                    payload = _decode_revealed_commitment(entry[0])
                    if payload:
                        out.append(RevealedPayload(hotkey, block, payload))
                except Exception as exc:  # noqa: BLE001 - skip malformed, never crash
                    logger.debug("skipping undecodable revealed entry from %s: %s", hotkey, exc)
        out.sort(key=lambda rp: (rp.block, rp.hotkey))
        return out

    def publish_reveal(self, payload: str, blocks_until_reveal: int) -> bool:
        # raise_error=False: a rate-limited commit comes back as success=False WITHOUT
        # raising, so we must inspect the response and let the caller retry.
        # v9 returns (success, reveal_block); older SDKs returned a result object.
        resp = self.subtensor.set_reveal_commitment(
            wallet=self.wallet,
            netuid=self.netuid,
            data=payload,
            blocks_until_reveal=blocks_until_reveal,
        )
        ok = bool(resp[0]) if isinstance(resp, tuple) else bool(
            getattr(resp, "success", getattr(resp, "is_success", False))
        )
        if not ok:
            logger.warning("publish_reveal not accepted (rate-limited?)")
        return ok

    def publish_status(self, payload: str) -> None:
        if len(payload.encode()) > STATUS_MAX_BYTES:
            raise ValueError(
                f"status payload is {len(payload.encode())} bytes; the plaintext "
                f"commitment slot is capped at {STATUS_MAX_BYTES}"
            )
        ok = self.subtensor.set_commitment(
            wallet=self.wallet, netuid=self.netuid, data=payload
        )
        if ok is False:  # v9 returns a bool; older SDKs returned None on success
            raise RuntimeError("set_commitment was not accepted by the chain")

    def read_status(self) -> dict[str, str]:
        return {
            str(hk): str(data)
            for hk, data in (self.subtensor.get_all_commitments(self.netuid) or {}).items()
        }

    def set_weights(self, weights: dict[int, float]) -> None:
        uids = sorted(weights)
        total = sum(weights.values()) or 1.0
        # v9 returns (success, message) and handles commit-reveal internally when
        # the subnet has it enabled; no raise_error kwarg.
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uids,
            weights=[weights[u] / total for u in uids],
        )
        ok, msg = result if isinstance(result, tuple) else (bool(result), "")
        if not ok:
            raise RuntimeError(f"set_weights was not accepted: {msg}")


@dataclass
class MockChainClient(ChainClient):
    """In-memory chain for tests, soaks, and the adversarial sandbox.

    Mirrors the real channels: an accumulating reveal history (multi-entry,
    block-stamped) and a single overwriting status slot per hotkey.
    """

    block: int = 100
    _neurons: list[NeuronView] = field(default_factory=list)
    _reveals: list[RevealedPayload] = field(default_factory=list)
    _status: dict[str, str] = field(default_factory=dict)
    last_weights: dict[int, float] = field(default_factory=dict)
    identity_hotkey: str = "validator-0"

    def advance(self, blocks: int = 1) -> None:
        self.block += blocks

    def add_neuron(self, n: NeuronView) -> None:
        self._neurons.append(n)

    def inject_reveal(self, hotkey: str, payload: str, block: int | None = None) -> None:
        self._reveals.append(RevealedPayload(hotkey, self.block if block is None else block, payload))

    def current_block(self) -> int:
        return self.block

    def block_hash(self, block: int) -> str:
        import hashlib

        return "0x" + hashlib.blake2b(str(block).encode(), digest_size=32).hexdigest()

    def neurons(self) -> list[NeuronView]:
        return list(self._neurons)

    def read_revealed_payloads(self, since_block: int = 0) -> list[RevealedPayload]:
        out = [rp for rp in self._reveals if rp.block >= since_block and rp.block <= self.block]
        out.sort(key=lambda rp: (rp.block, rp.hotkey))
        return out

    def publish_reveal(self, payload: str, blocks_until_reveal: int) -> bool:
        self._reveals.append(
            RevealedPayload(self.identity_hotkey, self.block + blocks_until_reveal, payload)
        )
        return True

    def publish_reveal_as(self, hotkey: str, payload: str, blocks_until_reveal: int = 0) -> None:
        self._reveals.append(RevealedPayload(hotkey, self.block + blocks_until_reveal, payload))

    def publish_status(self, payload: str) -> None:
        if len(payload.encode()) > STATUS_MAX_BYTES:
            raise ValueError(f"status payload exceeds {STATUS_MAX_BYTES} bytes")
        self._status[self.identity_hotkey] = payload

    def read_status(self) -> dict[str, str]:
        return dict(self._status)

    def set_weights(self, weights: dict[int, float]) -> None:
        self.last_weights = dict(weights)

    # -- legacy aliases kept for older tests/scripts ---------------------------

    def submit_challenge(self, reveal_payload: str, blocks_until_reveal: int) -> None:
        self.publish_reveal(reveal_payload, blocks_until_reveal)

    def publish_commitment_as(self, hotkey: str, payload: str) -> None:
        self.publish_reveal_as(hotkey, payload)
