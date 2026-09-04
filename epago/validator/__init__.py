"""Validator-in-a-box: intake, duel orchestration, verdicts, coronation, weights."""

from epago.validator.service import Deps, DuelSpec, ValidatorService
from epago.validator.state import QueuedSubmission, ValidatorState

__all__ = ["Deps", "DuelSpec", "QueuedSubmission", "ValidatorService", "ValidatorState"]
