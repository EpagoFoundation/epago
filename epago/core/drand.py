"""drand randomness beacon — non-gameable, unbiasable post-commit task selection.

Seeding task generation from the reveal block hash makes the exam *fresh* (the
hash does not exist until the challenger is committed), but a block hash is
**biasable**: whoever produces the block can withhold one whose hash yields an
unfavourable draw. For a subnet miner that is a weak attack (miners do not
produce subtensor blocks), but the robust primitive is a threshold-randomness
beacon that *no* single party can predict or bias.

This module derives the task seed from a **future drand round** instead. drand
(League of Entropy) emits a verifiable, unbiasable random value every ``period``
seconds on a fixed schedule; round R's value is a deterministic public number
that becomes known only when the network produces it. So:

* **Non-gameable** — the round used for a submission is computed to fall *after*
  the reveal block, so its randomness does not exist when the miner commits.
  A miner running the identical public pipeline still cannot predict the draw,
  and — unlike a block hash — cannot bias it (they are not a drand node).
* **Deterministic** — every validator (and every later auditor) fetches the same
  round and gets the same bytes, so they mint byte-identical task sets and the
  θ-quorum on the LCB holds.
* **Replayable** — the round number is derived from public chain data, so anyone
  can re-fetch the same round and reproduce the seed forever after.

Determinism note: :func:`round_at` and :func:`beacon_seed` are pure. Only
:class:`HttpDrand` touches the network, and it fetches a value that is identical
for everyone, so it does not break the "pure function of chain data" contract —
it only *materialises* a public number the chain already pointed at.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol

# League of Entropy public beacons. quicknet is unchained (a round's signature
# does not depend on the previous one) and fast — the right default for a
# post-commit seed. Pin the chain hash so a mirror cannot serve a different chain.
QUICKNET_CHAIN_HASH = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
QUICKNET_GENESIS_TIME = 1692803367
QUICKNET_PERIOD_S = 3

DEFAULT_HTTP_ENDPOINTS = (
    "https://api.drand.sh",
    "https://drand.cloudflare.com",
)


@dataclass(frozen=True, slots=True)
class DrandInfo:
    """Beacon schedule parameters — enough to map a wall-clock time to a round."""

    genesis_time: int
    period_s: int
    chain_hash: str

    @classmethod
    def quicknet(cls) -> "DrandInfo":
        return cls(QUICKNET_GENESIS_TIME, QUICKNET_PERIOD_S, QUICKNET_CHAIN_HASH)


def round_at(info: DrandInfo, unix_time: int) -> int:
    """The drand round in effect at ``unix_time`` (pure, deterministic).

    Round 1 is emitted at ``genesis_time``; round N at
    ``genesis_time + (N-1)*period``. Times before genesis clamp to round 1.
    """
    if unix_time <= info.genesis_time:
        return 1
    return (unix_time - info.genesis_time) // info.period_s + 1


def round_after(info: DrandInfo, block_time_unix: int, delay_s: int) -> int:
    """The round guaranteed to be produced at least ``delay_s`` after a block.

    ``delay_s`` is the safety margin ensuring the round's randomness does not yet
    exist when the challenger commits — the whole point of post-commit entropy.
    """
    return round_at(info, block_time_unix + max(delay_s, 0)) + 1


def beacon_seed(randomness: bytes, hotkey: str, label: bytes) -> int:
    """8-byte seed from beacon randomness + hotkey + a domain-separating label.

    Same shape as :func:`epago.core.stats.derive_seed` so the two are
    interchangeable at the call site; the entropy source is what differs.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(randomness)
    h.update(hotkey.encode())
    h.update(label)
    return int.from_bytes(h.digest(), "big")


class Beacon(Protocol):
    """A drand beacon: schedule info plus a round -> randomness lookup."""

    def info(self) -> DrandInfo: ...

    def randomness(self, round_number: int) -> bytes: ...


@dataclass(slots=True)
class MockDrand:
    """Deterministic in-memory beacon for tests and offline soaks.

    Randomness is ``blake2b(chain_hash || round)`` — stable per round, distinct
    across rounds — so tests exercise the exact seeding path without network.
    """

    _info: DrandInfo

    @classmethod
    def quicknet(cls) -> "MockDrand":
        return cls(DrandInfo.quicknet())

    def info(self) -> DrandInfo:
        return self._info

    def randomness(self, round_number: int) -> bytes:
        return hashlib.blake2b(
            f"{self._info.chain_hash}:{round_number}".encode(), digest_size=32
        ).digest()


class HttpDrand:
    """Fetches rounds from the League of Entropy HTTP relays (the network seam).

    The returned ``randomness`` is a public value identical for every caller, so
    determinism across validators holds. The chain hash is pinned; a full BLS
    signature verification is a hardening step left as a documented seam (it
    needs a pairing library) — it guards only against a malicious relay, not
    against non-determinism.
    """

    def __init__(
        self,
        info: DrandInfo | None = None,
        endpoints: tuple[str, ...] = DEFAULT_HTTP_ENDPOINTS,
        timeout_s: float = 10.0,
    ) -> None:
        self._info = info or DrandInfo.quicknet()
        self._endpoints = endpoints
        self._timeout_s = timeout_s

    def info(self) -> DrandInfo:
        return self._info

    def randomness(self, round_number: int) -> bytes:
        last_exc: Exception | None = None
        for base in self._endpoints:
            url = f"{base.rstrip('/')}/{self._info.chain_hash}/public/{round_number}"
            try:
                with urllib.request.urlopen(url, timeout=self._timeout_s) as resp:
                    data = json.loads(resp.read())
                if int(data["round"]) != round_number:
                    raise ValueError(f"drand returned round {data['round']} != {round_number}")
                return bytes.fromhex(data["randomness"])
            except Exception as exc:  # noqa: BLE001 - try the next relay
                last_exc = exc
        raise RuntimeError(f"all drand relays failed for round {round_number}: {last_exc}")
