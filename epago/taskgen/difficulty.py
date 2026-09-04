"""Template difficulty control from live duel telemetry.

The paired-duel design only gains statistical power from tasks the king
neither always solves nor always fails, and from tasks where challenger and
king actually *differ* (nonzero ``|d_i|``). The controller tracks three EMAs
per template from duel results and shapes the mint mixture accordingly:

* king solve rate — templates outside ``[KING_SOLVE_BAND_LOW,
  KING_SOLVE_BAND_HIGH]`` are down-weighted (too easy or too hard);
* judge invocation rate — templates that frequently need the LLM judge tier
  are down-weighted (judge calls are the noisiest and costliest grades);
* discrimination (mean ``|d_i|``) — templates whose discrimination EMA stays
  ~0 over many observed duels are *retired* (weight exactly 0): they burn
  rollout budget without ever moving a verdict.

The controller shapes only the mixture; determinism of a single generation
run is untouched because its output feeds the seeded rng-jittered weights the
same way :func:`epago.taskgen.templates.base_mixture` does.
"""

from __future__ import annotations

import numpy as np

from epago import constants

# EMA horizons (in observations, converted via alpha = 2/(k+1)).
_SOLVE_EMA_K = 20
_JUDGE_EMA_K = 20
_DISC_EMA_K = 50
# Discrimination EMA below this, after enough observations, retires a template.
_DISC_RETIRE_FLOOR = 0.02
_DISC_RETIRE_MIN_OBS = 40
# Judge-tier grading on more than this fraction of tasks marks a template noisy.
_JUDGE_RATE_CEILING = 0.5
# Multiplicative penalties (down-weight, never hard-zero — only retirement zeroes).
_BAND_PENALTY = 0.25
_JUDGE_PENALTY = 0.5
_LOW_DISC_PENALTY = 0.5


class _Ema:
    """Simple EMA that seeds itself on the first observation."""

    __slots__ = ("value", "count", "_alpha")

    def __init__(self, k: int) -> None:
        self.value: float | None = None
        self.count = 0
        self._alpha = 2.0 / (k + 1.0)

    def observe(self, x: float) -> None:
        self.count += 1
        self.value = x if self.value is None else self._alpha * x + (1.0 - self._alpha) * self.value


class DifficultyController:
    """Per-template mint-mixture shaping from observed duel outcomes."""

    def __init__(
        self,
        band_low: float = constants.KING_SOLVE_BAND_LOW,
        band_high: float = constants.KING_SOLVE_BAND_HIGH,
    ) -> None:
        self.band_low = band_low
        self.band_high = band_high
        self._solve: dict[str, _Ema] = {}
        self._judge: dict[str, _Ema] = {}
        self._disc: dict[str, _Ema] = {}
        self._retired: set[str] = set()

    def _ema(self, store: dict[str, _Ema], template: str, k: int) -> _Ema:
        if template not in store:
            store[template] = _Ema(k)
        return store[template]

    # -- telemetry -------------------------------------------------------------

    def observe(self, template: str, king_correct_rate: float) -> None:
        """Record the king's per-duel solve rate on tasks of ``template``."""
        self._ema(self._solve, template, _SOLVE_EMA_K).observe(king_correct_rate)

    def observe_judge_rate(self, template: str, judge_invocation_rate: float) -> None:
        """Record how often tasks of ``template`` needed the judge grading tier."""
        self._ema(self._judge, template, _JUDGE_EMA_K).observe(judge_invocation_rate)

    def observe_discrimination(self, template: str, abs_d_rate: float) -> None:
        """Record mean ``|d_i|`` for ``template`` in one duel; retire dead templates.

        Retirement is one-way by design: a template that produced ~zero
        discrimination over :data:`_DISC_RETIRE_MIN_OBS` duels has demonstrated
        it cannot separate models on this corpus and release; it should come
        back only via a new release, not a lucky EMA drift.
        """
        ema = self._ema(self._disc, template, _DISC_EMA_K)
        ema.observe(abs_d_rate)
        if ema.count >= _DISC_RETIRE_MIN_OBS and (ema.value or 0.0) < _DISC_RETIRE_FLOOR:
            self._retired.add(template)

    @property
    def retired(self) -> frozenset[str]:
        return frozenset(self._retired)

    # -- mixture -----------------------------------------------------------------

    def mixture_weights(self, templates: list[str], rng: np.random.Generator) -> dict[str, float]:
        """Normalized mint weights: rng-jittered base times telemetry penalties.

        Templates with no telemetry yet keep full base weight (innocent until
        observed). If every template is retired the mixture falls back to
        uniform — generation must never stall entirely on controller state.
        """
        base = 0.5 + rng.random(len(templates))
        weights: dict[str, float] = {}
        for name, b in zip(templates, base):
            if name in self._retired:
                weights[name] = 0.0
                continue
            w = float(b)
            solve = self._solve.get(name)
            if solve is not None and solve.value is not None:
                if not (self.band_low <= solve.value <= self.band_high):
                    w *= _BAND_PENALTY
            judge = self._judge.get(name)
            if judge is not None and judge.value is not None and judge.value > _JUDGE_RATE_CEILING:
                w *= _JUDGE_PENALTY
            disc = self._disc.get(name)
            if disc is not None and disc.value is not None and disc.value < 2 * _DISC_RETIRE_FLOOR:
                w *= _LOW_DISC_PENALTY
            weights[name] = w
        total = sum(weights.values())
        if total <= 0.0:
            uniform = 1.0 / len(templates)
            return {name: uniform for name in templates}
        return {name: w / total for name, w in weights.items()}
