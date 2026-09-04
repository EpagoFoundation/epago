"""Deterministic duel statistics.

Everything in this module is a pure function of its inputs. Two validators (or
a validator and a later auditor) given the same chain data compute bit-identical
seeds, bootstrap draws, and verdict thresholds.
"""

from __future__ import annotations

import hashlib

import numpy as np

from epago import constants
from epago.core.types import DuelHalf


def derive_seed(block_hash: str, hotkey: str, label: bytes) -> int:
    """8-byte blake2b seed from public chain inputs; the label separates domains."""
    h = hashlib.blake2b(digest_size=8)
    h.update(block_hash.encode())
    h.update(hotkey.encode())
    h.update(label)
    return int.from_bytes(h.digest(), "big")


def public_task_seed(block_hash: str, hotkey: str) -> int:
    return derive_seed(block_hash, hotkey, b"public")


def bootstrap_seed(block_hash: str, hotkey: str) -> int:
    return derive_seed(block_hash, hotkey, b"boot")


def round_public_seed(round_block_hash: str, round_no: int) -> int:
    """Exam seed for a whole competition round.

    Keyed on the round, never on an entrant, so the entire field answers one
    identical exam. A per-entrant draw would let exam luck separate two
    challengers of equal skill, which is exactly the comparison a round is for.
    """
    return derive_seed(round_block_hash, str(round_no), b"public")


def round_private_seed(round_block_hash: str, round_no: int) -> int:
    """Private-half sampling seed for a round; same reasoning as the public one."""
    return derive_seed(round_block_hash, str(round_no), b"private")


def round_confirmation_public_seed(round_block_hash: str, round_no: int) -> int:
    """Exam seed for a coronation confirmation duel.

    A separate domain from the round exam: the confirmation must be a fresh
    sample, or it would replay the exact luck it exists to rule out. Still a
    pure function of public chain data, so any auditor re-mints the same exam.
    """
    return derive_seed(round_block_hash, str(round_no), b"confirm-public")


def round_confirmation_private_seed(round_block_hash: str, round_no: int) -> int:
    """Private-half sampling seed for a confirmation duel; same reasoning."""
    return derive_seed(round_block_hash, str(round_no), b"confirm-private")


def round_bootstrap_seed(round_block_hash: str, challenger_digest: str) -> int:
    """Bootstrap resampling seed for one entrant's LCB.

    Per-entrant, because the LCB is computed over that entrant's own paired
    differences; still a pure function of public chain data, so any auditor
    recomputes it.
    """
    return derive_seed(round_block_hash, challenger_digest, b"boot")


def paired_half(king_correct: list[bool], challenger_correct: list[bool]) -> DuelHalf:
    if len(king_correct) != len(challenger_correct):
        raise ValueError("paired halves must have equal length")
    if not king_correct:
        raise ValueError("empty holdout half")
    diffs = tuple(int(c) - int(k) for k, c in zip(king_correct, challenger_correct))
    n = len(diffs)
    return DuelHalf(
        n_tasks=n,
        diffs=diffs,
        mu_hat=sum(diffs) / n,
        king_acc=sum(king_correct) / n,
        challenger_acc=sum(challenger_correct) / n,
    )


def bootstrap_lcb(
    diffs: tuple[int, ...] | list[int],
    seed: int,
    b: int = constants.BOOTSTRAP_B,
    alpha: float = constants.EVAL_ALPHA,
) -> float:
    """One-sided lower confidence bound on the mean paired difference."""
    d = np.asarray(diffs, dtype=np.float64)
    n = d.shape[0]
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, n, size=(b, n))
    boot_means = d[idx].mean(axis=1)
    return float(np.quantile(boot_means, alpha))


def adaptive_delta(king_acc_ema: float, noise_floor: float, c: float = constants.DELTA_C) -> float:
    """Effect floor scaled to remaining headroom, clamped above harness noise.

    The clamp is a hard correctness constraint: verdicts within measured
    harness noise of the threshold would be coin flips.
    """
    headroom_delta = c * (1.0 - king_acc_ema)
    return max(headroom_delta, constants.DELTA_NOISE_MULTIPLIER * noise_floor)


def update_acc_ema(prev_ema: float, observed_acc: float, k: int = constants.KING_ACC_EMA_K) -> float:
    """K-duel EMA of the king's per-duel accuracy."""
    alpha = 2.0 / (k + 1.0)
    return alpha * observed_acc + (1.0 - alpha) * prev_ema


def noise_floor_from_calibration(disagreement_rates: list[float]) -> float:
    """Noise floor = max of recent king-vs-king score-gap standard errors.

    Calibration duels run the king against itself on fresh holdouts across the
    validator's own hardware; any nonzero score gap is pure harness noise, and
    its standard error is the coin-flip band a coronation must clear. Falls back
    to the static cross-GPU budget when no calibration data exists.
    """
    if not disagreement_rates:
        return constants.CROSS_GPU_NOISE_BUDGET
    return max(max(disagreement_rates), 1e-6)
