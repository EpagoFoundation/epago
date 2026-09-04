"""Tests for epago.core.emissions — reign decay, bonus, arena, burn fallbacks."""

from __future__ import annotations

import pytest

from epago.config import EmissionsSection
from epago.core.emissions import (
    ARENA_MAX_KINGS,
    ArenaEntry,
    KingEmissionState,
    compute_weights,
    coronation_bonus_factor,
    inherits_reign,
    phase_b_active,
    king_share_at,
    reign_decay_factor,
)

# Matches chain.toml: the king and the arena split the whole emission, so these
# two are the entire budget and must sum to 1.
CFG = EmissionsSection(
    king_share=0.90,
    arena_share=0.10,
    reign_halflife_blocks=216_000,
    coronation_bonus_cap=3.0,
    coronation_bonus_blocks=7_200,
)

POOL = CFG.king_share + CFG.arena_share
BURN = "burn-hotkey"


def king(
    reign_started: int = 1_000,
    crowned: int = 1_000,
    lcb: float = 0.0,
    delta: float = 0.0,
) -> KingEmissionState:
    return KingEmissionState(
        hotkey="king-hk",
        reign_started_block=reign_started,
        crowned_block=crowned,
        coronation_lcb=lcb,
        coronation_delta=delta,
    )


class TestNormalization:
    def test_shares_sum_to_one_full_population(self):
        arena = [
            ArenaEntry(hotkey="nm-1", dethroned_block=1_000),
            ArenaEntry(hotkey="nm-2", dethroned_block=900),
        ]
        w = compute_weights(king(), arena, current_block=1_000, cfg=CFG, burn_hotkey=BURN)
        assert sum(w.values()) == pytest.approx(1.0)
        assert all(v > 0 for v in w.values())

    def test_shares_sum_to_one_king_only(self):
        w = compute_weights(king(), [], current_block=1_000, cfg=CFG, burn_hotkey=BURN)
        assert sum(w.values()) == pytest.approx(1.0)
        assert w["king-hk"] == pytest.approx(0.90)
        assert w[BURN] == pytest.approx(0.10)  # empty arena burns

    def test_no_king_burns_everything(self):
        w = compute_weights(None, [], current_block=1_000, cfg=CFG, burn_hotkey=BURN)
        assert w == {BURN: 1.0}


class TestReignDecay:
    def test_halves_at_halflife(self):
        assert reign_decay_factor(216_000, 216_000) == pytest.approx(0.5)

    def test_no_decay_at_age_zero(self):
        assert reign_decay_factor(0, 216_000) == pytest.approx(1.0)

    def test_quarter_at_two_halflives(self):
        assert reign_decay_factor(432_000, 216_000) == pytest.approx(0.25)

    def test_share_walks_from_the_top_of_the_band_to_the_floor(self):
        """90% at coronation, 85% after the window, and half way in between.

        Linear rather than exponential because the schedule is a promise a
        miner can check against a block number.
        """
        assert king_share_at(0, CFG) == pytest.approx(CFG.king_share)
        assert king_share_at(CFG.reign_decay_blocks // 2, CFG) == pytest.approx(
            (CFG.king_share + CFG.king_share_floor) / 2
        )
        assert king_share_at(CFG.reign_decay_blocks, CFG) == pytest.approx(
            CFG.king_share_floor
        )

    def test_the_share_never_falls_below_the_floor(self):
        """An unchallenged reign bleeds, but only so far."""
        forever = CFG.reign_decay_blocks * 100
        assert king_share_at(forever, CFG) == pytest.approx(CFG.king_share_floor)

    def test_bled_mass_flows_to_the_arena_roster(self):
        current = 1_000 + CFG.reign_decay_blocks
        arena = [ArenaEntry(hotkey="former-king", dethroned_block=current)]
        w = compute_weights(king(), arena, current_block=current, cfg=CFG, burn_hotkey=BURN)
        assert w["king-hk"] == pytest.approx(CFG.king_share_floor)
        assert w["former-king"] == pytest.approx(1.0 - CFG.king_share_floor)
        assert BURN not in w

    def test_bled_mass_burns_when_no_one_has_been_dethroned_yet(self):
        current = 1_000 + CFG.reign_decay_blocks
        w = compute_weights(king(), [], current_block=current, cfg=CFG, burn_hotkey=BURN)
        assert w["king-hk"] == pytest.approx(CFG.king_share_floor)
        assert w[BURN] == pytest.approx(1.0 - CFG.king_share_floor)


class TestSelfDethrone:
    def test_same_hotkey_inherits_reign(self):
        assert inherits_reign("hotkey-A", "hotkey-A")

    def test_different_hotkey_resets_reign(self):
        assert not inherits_reign("hotkey-A", "hotkey-B")

    def test_inherited_reign_keeps_bleeding(self):
        # A self-dethroned king whose reign started before the window earns the
        # floor, not the top of the band, even though it was crowned just now.
        current = 1_000 + CFG.reign_decay_blocks
        k = king(reign_started=1_000, crowned=current, lcb=0.01, delta=0.02)  # lcb/delta < 1
        w = compute_weights(k, [], current_block=current, cfg=CFG, burn_hotkey=BURN)
        assert w["king-hk"] == pytest.approx(CFG.king_share_floor)


class TestCoronationBonus:
    def test_caps_at_configured_multiplier(self):
        # lcb/delta = 10 caps at 3.0
        assert coronation_bonus_factor(1.0, 0.1, 0, CFG) == pytest.approx(3.0)

    def test_decays_linearly_over_window(self):
        assert coronation_bonus_factor(1.0, 0.1, 3_600, CFG) == pytest.approx(2.0)

    def test_expires_at_window_end(self):
        assert coronation_bonus_factor(1.0, 0.1, 7_200, CFG) == 1.0
        assert coronation_bonus_factor(1.0, 0.1, 100_000, CFG) == 1.0

    def test_no_bonus_when_barely_above_delta(self):
        # magnitude <= 1 -> no boost
        assert coronation_bonus_factor(0.02, 0.02, 0, CFG) == pytest.approx(1.0)

    def test_no_bonus_for_nonpositive_delta(self):
        assert coronation_bonus_factor(0.5, 0.0, 0, CFG) == 1.0

    def test_the_bonus_cannot_reach_past_the_top_of_the_band(self):
        """A coronation may hold the king at 90%, never above it.

        Before the band existed the bonus multiplied without an upper bound and
        a capped 3x wanted 2.70 of a 1.0 pool, so a fresh king took everything
        and every former king earned nothing for the whole bonus window --
        the opposite of what the arena is for.
        """
        arena = [ArenaEntry(hotkey="former-king", dethroned_block=1_000)]
        w = compute_weights(
            king(lcb=0.5, delta=0.01), arena, current_block=1_000, cfg=CFG, burn_hotkey=BURN
        )
        assert w["king-hk"] == pytest.approx(CFG.king_share)
        assert w["former-king"] == pytest.approx(1.0 - CFG.king_share)
        assert sum(w.values()) == pytest.approx(1.0)


class TestArenaPool:
    def test_split_is_equal_among_the_former_kings_on_the_roster(self):
        """Equal, not decayed.

        The roster is already bounded at ARENA_MAX_KINGS, so recency is
        expressed by who is on it at all. A decay curve on top would leave the
        third seat worth almost nothing while still occupying it, which is a
        worse description of "the three most recent kings" than an even split.
        """
        current = 10_000
        arena = [
            ArenaEntry(hotkey="king-1", dethroned_block=9_000),
            ArenaEntry(hotkey="king-2", dethroned_block=8_000),
        ]
        w = compute_weights(king(reign_started=current), arena, current, CFG, BURN)
        assert w["king-1"] == pytest.approx(w["king-2"])

    def test_only_the_three_most_recent_kings_earn(self):
        """A king displaced more than ARENA_MAX_KINGS times leaves entirely.

        The arena is a short memory of who recently held the crown, not a
        pension list that grows without bound.
        """
        current = 10_000
        arena = [
            ArenaEntry(hotkey=f"king-{i}", dethroned_block=9_000 - i * 100)
            for i in range(5)
        ]
        w = compute_weights(king(reign_started=current), arena, current, CFG, BURN)
        # The reigning king is in the vector too; the arena roster is what is
        # being counted here.
        earning = {h for h in w if h.startswith("king-") and h != king().hotkey}
        assert len(earning) == ARENA_MAX_KINGS
        # The newest three, by when they fell.
        assert earning == {"king-0", "king-1", "king-2"}

    def test_a_hotkey_that_reigned_twice_holds_one_seat(self):
        """Otherwise a repeat king draws a double share of the same budget."""
        current = 10_000
        arena = [
            ArenaEntry(hotkey="repeat", dethroned_block=8_000),
            ArenaEntry(hotkey="repeat", dethroned_block=9_500),
            ArenaEntry(hotkey="other", dethroned_block=9_000),
        ]
        w = compute_weights(king(reign_started=current), arena, current, CFG, BURN)
        assert w["repeat"] == pytest.approx(w["other"])

    def test_empty_arena_mass_burns(self):
        w = compute_weights(king(), [], current_block=1_000, cfg=CFG, burn_hotkey=BURN)
        assert w[BURN] == pytest.approx(CFG.arena_share)


class TestBudgetIsConserved:
    """The king and the arena split one fixed budget.

    Crediting the arena with the decayed mass while the king separately kept a
    bonus multiple of it double-counted, so the raw vector summed above 1 and
    the final normalization silently rescaled every share. The case is not
    exotic: a self-dethrone inherits the reign clock, so a decayed reign and an
    open bonus window coincide exactly in the salami-slicing scenario the bonus
    exists to discourage.
    """

    @pytest.mark.parametrize(
        "reign_age,blocks_since_crowned,lcb",
        [
            (0, 10**9, 0.0),                              # fresh reign, no bonus
            (CFG.reign_halflife_blocks, 10**9, 0.0),      # decayed, no bonus
            (0, 0, 0.30),                                 # fresh reign, big bonus
            (CFG.reign_halflife_blocks, 0, 0.30),         # decayed AND bonused
            (CFG.reign_halflife_blocks * 2, 0, 0.30),
            (CFG.reign_halflife_blocks * 4, 3600, 0.09),
        ],
    )
    def test_vector_sums_to_one_without_rescaling(self, reign_age, blocks_since_crowned, lcb):
        current = 1_000_000
        k = KingEmissionState(
            hotkey="king-hk",
            reign_started_block=current - reign_age,
            crowned_block=current - blocks_since_crowned,
            coronation_lcb=lcb,
            coronation_delta=0.03,
        )
        arena = [ArenaEntry(hotkey="near-hk", dethroned_block=current)]
        weights = compute_weights(k, arena, current, CFG, burn_hotkey=BURN)

        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["king-hk"] <= POOL + 1e-9
        # Whatever the king did not take, the arena got — exactly.
        assert weights["king-hk"] + weights.get("near-hk", 0.0) == pytest.approx(POOL)

    def test_a_bonus_over_a_bled_reign_lifts_back_toward_the_ceiling(self):
        """A self-dethrone inherits the reign clock, so a bled share and an open
        bonus window coincide exactly in the slicing scenario the bonus is meant
        to discourage. The band keeps that bounded either way.
        """
        current = 1_000_000
        k = KingEmissionState(
            hotkey="king-hk",
            reign_started_block=current - CFG.reign_decay_blocks,  # at the floor
            crowned_block=current,        # bonus window wide open
            coronation_lcb=0.045,         # lcb/delta = 1.5x bonus
            coronation_delta=0.03,
        )
        arena = [ArenaEntry(hotkey="former-king", dethroned_block=current)]
        weights = compute_weights(k, arena, current, CFG, burn_hotkey=BURN)

        # floor * 1.5 = 1.275, clamped to the ceiling.
        assert weights["king-hk"] == pytest.approx(CFG.king_share)
        assert weights["king-hk"] + weights["former-king"] == pytest.approx(POOL)


class TestPhaseB:
    MIN = dict(min_duels=50, min_dethrones=1, min_blocks=100_800)

    def test_active_exactly_at_boundary(self):
        assert phase_b_active(50, 1, 100_800, **self.MIN)

    def test_inactive_when_any_condition_unmet(self):
        assert not phase_b_active(49, 1, 100_800, **self.MIN)
        assert not phase_b_active(50, 0, 100_800, **self.MIN)
        assert not phase_b_active(50, 1, 100_799, **self.MIN)

    def test_active_well_past_boundary(self):
        assert phase_b_active(500, 9, 1_000_000, **self.MIN)
