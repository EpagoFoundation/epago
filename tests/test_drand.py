"""drand beacon: round derivation and seeding are pure and deterministic, so
two validators (or a later auditor) fetching the same round mint identical tasks."""
from __future__ import annotations

from epago.core.drand import (
    DrandInfo,
    MockDrand,
    beacon_seed,
    round_after,
    round_at,
)


def test_round_at_boundaries() -> None:
    info = DrandInfo(genesis_time=1000, period_s=3, chain_hash="ch")
    assert round_at(info, 999) == 1          # before genesis clamps to round 1
    assert round_at(info, 1000) == 1         # at genesis
    assert round_at(info, 1002) == 1         # within the first period
    assert round_at(info, 1003) == 2         # one period later
    assert round_at(info, 1009) == 4


def test_round_after_is_strictly_in_the_future() -> None:
    info = DrandInfo(genesis_time=1000, period_s=3, chain_hash="ch")
    # A block at t=1000 with a 12s safety margin must map to a round strictly
    # beyond the round covering t=1012 — so its randomness cannot exist yet.
    now_round = round_at(info, 1012)
    assert round_after(info, 1000, delay_s=12) == now_round + 1


def test_beacon_seed_is_deterministic_and_domain_separated() -> None:
    r = b"\x01" * 32
    s_pub = beacon_seed(r, "hk", b"public")
    assert s_pub == beacon_seed(r, "hk", b"public")          # deterministic
    assert s_pub != beacon_seed(r, "hk", b"private")         # label separates domains
    assert s_pub != beacon_seed(r, "hk2", b"public")         # hotkey separates authors
    assert s_pub != beacon_seed(b"\x02" * 32, "hk", b"public")  # randomness matters


def test_mock_beacon_randomness_stable_per_round_distinct_across() -> None:
    b = MockDrand.quicknet()
    assert b.randomness(100) == b.randomness(100)   # stable
    assert b.randomness(100) != b.randomness(101)   # distinct


def test_two_validators_derive_identical_seed_from_same_round() -> None:
    # Independent beacon instances (different validators) + same round -> same seed.
    b1, b2 = MockDrand.quicknet(), MockDrand.quicknet()
    info = b1.info()
    rnd = round_after(info, block_time_unix=1_700_000_000, delay_s=30)
    seed1 = beacon_seed(b1.randomness(rnd), "author-hk", b"private")
    seed2 = beacon_seed(b2.randomness(rnd), "author-hk", b"private")
    assert seed1 == seed2
