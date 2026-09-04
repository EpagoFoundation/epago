"""Evaluation subsystem: backends, the pinned rollout harness, the judge
cascade, pre-duel probes, the paired duel, and the persistent eval server.

Importable without torch or vllm; GPU dependencies are guarded inside the
backends that need them.
"""

from epago.eval.backend import ModelBackend, ScriptedBackend, VllmBackend, backend_factory
from epago.eval.duel import DuelSpec, run_calibration_duel, run_duel
from epago.eval.harness import SYSTEM_PROMPT, harness_digest, run_rollout
from epago.eval.judge import (
    ADVERSARIAL_SUITE,
    LlmJudge,
    judge_answer,
    load_llm_judge,
    normalize,
    sanitize,
)
from epago.eval.pool import GpuPool, SweepRequest, build_pool, resolve_devices
from epago.eval.probes import (
    ProbeFailure,
    degenerate_probe,
    format_probe,
    load_probe,
    norm_sanity_probe,
)
from epago.eval.remote import DirRefIndex, RemoteEvalError, RemoteEvalRunner
from epago.eval.server import create_app

__all__ = [
    "ADVERSARIAL_SUITE",
    "DirRefIndex",
    "DuelSpec",
    "GpuPool",
    "LlmJudge",
    "ModelBackend",
    "ProbeFailure",
    "RemoteEvalError",
    "RemoteEvalRunner",
    "SYSTEM_PROMPT",
    "ScriptedBackend",
    "SweepRequest",
    "VllmBackend",
    "backend_factory",
    "build_pool",
    "create_app",
    "degenerate_probe",
    "format_probe",
    "harness_digest",
    "judge_answer",
    "load_llm_judge",
    "load_probe",
    "norm_sanity_probe",
    "normalize",
    "resolve_devices",
    "run_calibration_duel",
    "run_duel",
    "run_rollout",
    "sanitize",
]
