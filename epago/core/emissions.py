"""Emission schedule: reign decay, coronation bonus, arena pool.

Pure functions of chain-derived state so every validator computes identical
weight vectors. Design intent, encoded in the math:

* The king's share decays with reign age, from ``king_share`` to
  ``king_share_floor`` over ``reign_decay_blocks``, and then holds. An
  unchallenged incumbent bleeds to the arena but never below a defined
  majority, so the schedule is bounded at both ends and a miner can reason
  about what a reign is worth without solving for an asymptote.
* A self-dethrone (same hotkey re-crowning itself) inherits reign age, so
  slicing one improvement into many small coronations from one hotkey buys no
  fresh reigns; slicing across fresh hotkeys is deterred by registration burn.
* The coronation bonus scales with measured improvement, so revealing a full
  improvement at once weakly dominates slicing it.
* Former kings share the arena pool. A dethroned king does not fall to zero;
  it steps back into the arena and keeps earning while the roster holds it.
  The roster is bounded at :data:`ARENA_MAX_KINGS`, so a king that has been
  displaced that many times leaves entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

from epago.config import EmissionsSection


@dataclass(frozen=True, slots=True)
class ArenaEntry:
    """A former king, still earning while it stays on the roster."""

    hotkey: str
    #: Block at which this king was displaced. Ordering only -- the roster is
    #: trimmed by recency, not by an age curve, so a former king's share does
    #: not quietly decay to nothing between rotations.
    dethroned_block: int
    #: How long it reigned, in blocks. Recorded for audit and for the dashboard;
    #: it does not enter the split.
    reign_blocks: int = 0


@dataclass(frozen=True, slots=True)
class KingEmissionState:
    hotkey: str
    reign_started_block: int
    crowned_block: int
    coronation_lcb: float
    coronation_delta: float


#: How many former kings keep earning. Beyond this the oldest drops out
#: entirely, so the arena is a short memory of who recently held the crown
#: rather than an ever-growing pension list.
ARENA_MAX_KINGS = 3


def reign_decay_factor(reign_age_blocks: int, halflife_blocks: int) -> float:
    """Exponential halving. Retained for callers that want the raw curve."""
    return 0.5 ** (reign_age_blocks / halflife_blocks)


def king_share_at(reign_age_blocks: int, cfg: EmissionsSection) -> float:
    """The king's share of the whole emission at a given reign age.

    Linear from ``king_share`` to ``king_share_floor`` across
    ``reign_decay_blocks``, then flat at the floor. Linear rather than
    exponential because the schedule is a promise to miners: "90% falling to
    85% over three days" is checkable against a block number, where an
    asymptote that never quite arrives is not.

    Everything the king does not take is the arena's, so this single curve
    fixes both shares.
    """
    start = cfg.king_share
    floor = min(getattr(cfg, "king_share_floor", start), start)
    window = max(int(getattr(cfg, "reign_decay_blocks", 0)), 0)
    if window == 0 or reign_age_blocks >= window:
        return floor
    progress = max(reign_age_blocks, 0) / window
    return start - (start - floor) * progress


def coronation_bonus_factor(
    lcb: float, delta: float, blocks_since_crowned: int, cfg: EmissionsSection
) -> float:
    """Multiplier >= 1 on the king's share while the bonus window is open."""
    if blocks_since_crowned >= cfg.coronation_bonus_blocks or delta <= 0:
        return 1.0
    magnitude = min(lcb / delta, cfg.coronation_bonus_cap)
    remaining = 1.0 - blocks_since_crowned / cfg.coronation_bonus_blocks
    return 1.0 + max(magnitude - 1.0, 0.0) * remaining


def compute_weights(
    king: KingEmissionState | None,
    arena: list[ArenaEntry],
    current_block: int,
    cfg: EmissionsSection,
    burn_hotkey: str,
) -> dict[str, float]:
    """Hotkey -> weight, summing to 1. Unallocatable mass goes to the burn key."""
    weights: dict[str, float] = {}

    if king is None:
        weights[burn_hotkey] = 1.0
        return weights

    reign_age = max(current_block - king.reign_started_block, 0)
    base = king_share_at(reign_age, cfg)
    bonus = coronation_bonus_factor(
        king.coronation_lcb, king.coronation_delta, current_block - king.crowned_block, cfg
    )
    # The king and the arena split one fixed budget, so the arena gets exactly
    # what the king did not take. Crediting the arena with the decayed mass
    # while the king separately kept a bonus multiple of it double-counted: at
    # decay 0.5 with a 2x bonus the vector summed to 1.4, and the final
    # normalization then silently rescaled every share. That case is not
    # exotic — a self-dethrone inherits the reign clock, so a decayed reign and
    # an open bonus window coincide precisely in the salami-slicing scenario
    # the bonus is meant to discourage.
    king_pool = cfg.king_share + cfg.arena_share
    # The bonus can hold a fresh king at the top of the band but never above
    # it. Letting it multiply past `king_share` would let a coronation absorb
    # the arena's whole budget for the length of the bonus window, which is
    # the opposite of what the arena is for -- and it happened: at a 2x bonus
    # the king took the entire pool and every former king earned nothing.
    king_share = min(base * bonus, cfg.king_share)

    weights[king.hotkey] = king_share

    arena_budget = max(king_pool - king_share, 0.0)
    arena_weights = _arena_split(arena, current_block)
    for hotkey, frac in arena_weights.items():
        weights[hotkey] = weights.get(hotkey, 0.0) + arena_budget * frac
    if not arena_weights:
        weights[burn_hotkey] = weights.get(burn_hotkey, 0.0) + arena_budget

    total = sum(weights.values())
    return {h: w / total for h, w in weights.items() if w > 0}


def trim_arena(arena: list[ArenaEntry]) -> list[ArenaEntry]:
    """The most recent :data:`ARENA_MAX_KINGS` former kings, newest first.

    One entry per hotkey: a king that reigns twice occupies one seat, dated by
    its latest fall, rather than holding two and drawing a double share.
    """
    best: dict[str, ArenaEntry] = {}
    for e in sorted(arena, key=lambda x: x.dethroned_block):
        best[e.hotkey] = e
    ordered = sorted(best.values(), key=lambda x: x.dethroned_block, reverse=True)
    return ordered[:ARENA_MAX_KINGS]


def _arena_split(arena: list[ArenaEntry], current_block: int) -> dict[str, float]:
    """Equal shares among the former kings on the roster.

    Equal rather than decayed: the roster is already bounded, so recency is
    expressed by who is on it at all. Adding a decay curve on top would make
    the third seat worth almost nothing while still occupying it, which is a
    worse description of "the three most recent kings" than an even split.

    ``current_block`` is unused and kept so the signature stays stable for
    callers and for a future recency weighting.
    """
    roster = trim_arena(arena)
    if not roster:
        return {}
    share = 1.0 / len(roster)
    return {e.hotkey: share for e in roster}


def inherits_reign(previous_author_hotkey: str, new_author_hotkey: str) -> bool:
    """Self-dethrone keeps the old reign clock (anti-salami-slicing).

    Keyed on hotkey: a self-dethrone from the same hotkey inherits the reign
    clock. Slicing across freshly registered hotkeys is deterred by the
    per-hotkey registration burn, not by clock inheritance.
    """
    return previous_author_hotkey == new_author_hotkey


def phase_b_active(
    clean_duels: int,
    organic_dethrones: int,
    blocks_since_genesis: int,
    min_duels: int,
    min_dethrones: int,
    min_blocks: int,
) -> bool:
    """Deterministic emission activation — no operator switch exists."""
    return (
        clean_duels >= min_duels
        and organic_dethrones >= min_dethrones
        and blocks_since_genesis >= min_blocks
    )
