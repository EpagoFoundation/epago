"""Tests for epago.core.stats — deterministic duel statistics.

The known-answer seed tests hardcode expected integers: any change to the seed
derivation (hash function, digest size, byte order, label handling, input
encoding) must break these tests loudly, because it would silently fork every
validator's task selection and bootstrap draws.
"""

from __future__ import annotations

import pytest

from epago import constants
from epago.core import stats
from epago.core.types import DuelHalf

BLOCK_HASH = "0x" + "ab" * 32
HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


class TestDeriveSeed:
    def test_known_answer_public(self):
        assert stats.public_task_seed(BLOCK_HASH, HOTKEY) == 641636554155131871

    def test_known_answer_boot(self):
        assert stats.bootstrap_seed(BLOCK_HASH, HOTKEY) == 8272034804287572166

    def test_labels_separate_domains(self):
        assert stats.public_task_seed(BLOCK_HASH, HOTKEY) != stats.bootstrap_seed(
            BLOCK_HASH, HOTKEY
        )

    def test_sensitive_to_hotkey(self):
        assert stats.derive_seed(BLOCK_HASH, HOTKEY + "x", b"public") == 15460669576133326700

    def test_sensitive_to_block_hash(self):
        assert stats.derive_seed("0x" + "cd" * 32, HOTKEY, b"public") == 14845218136646518011

    def test_stable_across_calls(self):
        assert stats.derive_seed(BLOCK_HASH, HOTKEY, b"x") == stats.derive_seed(
            BLOCK_HASH, HOTKEY, b"x"
        )

    def test_fits_in_eight_bytes(self):
        assert 0 <= stats.derive_seed(BLOCK_HASH, HOTKEY, b"public") < 2**64


class TestBootstrapLcb:
    # 200 diffs: 80 wins, 40 losses, 80 ties -> mean 0.2
    DIFFS = (1, 0, -1, 1, 1, 0, 0, 1, -1, 0) * 20

    def test_deterministic_per_seed(self):
        a = stats.bootstrap_lcb(self.DIFFS, seed=1234)
        b = stats.bootstrap_lcb(self.DIFFS, seed=1234)
        assert a == b

    def test_different_seeds_generally_differ(self):
        a = stats.bootstrap_lcb(self.DIFFS, seed=1)
        b = stats.bootstrap_lcb(self.DIFFS, seed=2)
        assert a != b

    def test_known_answer(self):
        seed = stats.bootstrap_seed(BLOCK_HASH, HOTKEY)
        assert stats.bootstrap_lcb(self.DIFFS, seed) == pytest.approx(0.03)

    def test_monotone_in_alpha(self):
        seed = 999
        lcb_strict = stats.bootstrap_lcb(self.DIFFS, seed, alpha=0.001)
        lcb_loose = stats.bootstrap_lcb(self.DIFFS, seed, alpha=0.05)
        assert lcb_strict <= lcb_loose

    def test_lcb_below_mean(self):
        mean = sum(self.DIFFS) / len(self.DIFFS)
        assert stats.bootstrap_lcb(self.DIFFS, seed=7) < mean

    def test_degenerate_all_equal(self):
        assert stats.bootstrap_lcb((1,) * 50, seed=3) == pytest.approx(1.0)


class TestPairedHalf:
    def test_math(self):
        king = [True, True, False, False]
        chall = [True, False, True, True]
        half = stats.paired_half(king, chall)
        assert isinstance(half, DuelHalf)
        assert half.n_tasks == 4
        assert half.diffs == (0, -1, 1, 1)
        assert half.mu_hat == pytest.approx(0.25)
        assert half.king_acc == pytest.approx(0.5)
        assert half.challenger_acc == pytest.approx(0.75)

    def test_diffs_in_range(self):
        half = stats.paired_half([True, False], [False, True])
        assert set(half.diffs) <= {-1, 0, 1}

    def test_unequal_lengths_rejected(self):
        with pytest.raises(ValueError, match="equal length"):
            stats.paired_half([True], [True, False])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            stats.paired_half([], [])


class TestAdaptiveDelta:
    def test_headroom_dominates(self):
        # 0.05 * (1 - 0.5) = 0.025 > 1 * 0.001
        assert stats.adaptive_delta(0.5, noise_floor=0.001) == pytest.approx(0.025)

    def test_noise_clamp_dominates(self):
        # 0.05 * (1 - 0.99) = 0.0005 < multiplier * 0.01
        assert stats.adaptive_delta(0.99, noise_floor=0.01) == pytest.approx(
            constants.DELTA_NOISE_MULTIPLIER * 0.01
        )

    def test_clamp_is_multiplier_times_floor(self):
        nf = 0.02
        assert stats.adaptive_delta(1.0, noise_floor=nf) == pytest.approx(
            constants.DELTA_NOISE_MULTIPLIER * nf
        )

    def test_never_below_clamp(self):
        for ema in (0.0, 0.5, 0.9, 1.0):
            assert stats.adaptive_delta(ema, noise_floor=0.005) >= (
                constants.DELTA_NOISE_MULTIPLIER * 0.005
            )


class TestUpdateAccEma:
    def test_default_k(self):
        # alpha = 2 / (10 + 1)
        expected = (2 / 11) * 0.6 + (9 / 11) * 0.5
        assert stats.update_acc_ema(0.5, 0.6) == pytest.approx(expected)

    def test_fixed_point(self):
        assert stats.update_acc_ema(0.55, 0.55) == pytest.approx(0.55)

    def test_moves_toward_observation(self):
        assert 0.5 < stats.update_acc_ema(0.5, 1.0) < 1.0
        assert 0.0 < stats.update_acc_ema(0.5, 0.0) < 0.5


class TestNoiseFloor:
    def test_fallback_without_calibration(self):
        assert stats.noise_floor_from_calibration([]) == pytest.approx(
            constants.CROSS_GPU_NOISE_BUDGET
        )

    def test_takes_max_of_rates(self):
        assert stats.noise_floor_from_calibration([0.001, 0.004, 0.002]) == pytest.approx(0.004)

    def test_zero_rates_floored_at_epsilon(self):
        assert stats.noise_floor_from_calibration([0.0, 0.0]) == pytest.approx(1e-6)
