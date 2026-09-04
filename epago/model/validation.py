"""Intake validation: config lock and file hygiene.

The king's ``config.json`` is the source of truth. Challengers must match it
structurally, contain no executable code paths, and stay within the size cap.
Violations fail at intake with machine-readable reasons and zero GPU spent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from epago import constants
from epago.config import EpagoConfig
from epago.model.store import safetensors_total_bytes

GENERIC_LOCK_KEYS = (
    "architectures",
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "model_type",
    "tie_word_embeddings",
    "rope_theta",
    "max_position_embeddings",
)

_ALLOWED_SUFFIXES = {".safetensors", ".json", ".txt", ".model", ".jinja"}


@dataclass(frozen=True, slots=True)
class IntakeFailure:
    code: str
    detail: str


def validate_repo_name(repo: str, author_hotkey: str, cfg: EpagoConfig) -> IntakeFailure | None:
    if not re.match(cfg.chain.repo_pattern, repo):
        return IntakeFailure("repo_pattern", f"{repo!r} does not match {cfg.chain.repo_pattern!r}")
    prefix = author_hotkey[:8].lower()
    if prefix not in repo.lower():
        return IntakeFailure(
            "hotkey_prefix",
            f"repo name must contain the author hotkey prefix {prefix!r} (anti-impersonation)",
        )
    return None


def validate_challenger_folder(
    challenger_dir: Path, king_dir: Path, cfg: EpagoConfig
) -> list[IntakeFailure]:
    failures: list[IntakeFailure] = []
    failures.extend(_check_file_hygiene(challenger_dir))
    failures.extend(_check_config_lock(challenger_dir, king_dir, cfg))
    failures.extend(_check_size(challenger_dir, king_dir))
    return failures


def _check_file_hygiene(folder: Path) -> list[IntakeFailure]:
    failures = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        # Skip dotfiles AND anything under a dot-directory — HF's snapshot download
        # leaves `.cache/huggingface/**/*.lock`/`*.metadata` tooling artifacts in the
        # materialized folder; they are not part of the content-addressed model. This
        # mirrors prepare_challenger, which already excludes dot-prefixed path parts.
        if any(part.startswith(".") for part in p.relative_to(folder).parts):
            continue
        if p.suffix == ".py":
            failures.append(IntakeFailure("python_file", f"{p.name}: no code in model repos"))
        elif p.suffix in (".bin", ".pt", ".pkl", ".ckpt"):
            failures.append(IntakeFailure("pickle_weights", f"{p.name}: safetensors only"))
        elif p.suffix not in _ALLOWED_SUFFIXES:
            failures.append(IntakeFailure("unexpected_file", f"{p.name}: not an allowed file type"))
    single = (folder / "model.safetensors").exists()
    index = (folder / "model.safetensors.index.json").exists()
    if not single and not index:
        failures.append(IntakeFailure("layout", "missing canonical safetensors layout"))
    return failures


def _check_config_lock(challenger_dir: Path, king_dir: Path, cfg: EpagoConfig) -> list[IntakeFailure]:
    try:
        chall = json.loads((challenger_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [IntakeFailure("config_missing", f"config.json unreadable: {exc}")]
    king = json.loads((king_dir / "config.json").read_text())

    failures = []
    if "auto_map" in chall:
        failures.append(IntakeFailure("auto_map", "auto_map is forbidden (no remote-code path)"))
    for key in (*GENERIC_LOCK_KEYS, *cfg.arch.extra_lock_keys):
        if chall.get(key) != king.get(key):
            failures.append(
                IntakeFailure(
                    "config_lock",
                    f"{key}: challenger={chall.get(key)!r} king={king.get(key)!r}",
                )
            )
    return failures


def _check_size(challenger_dir: Path, king_dir: Path) -> list[IntakeFailure]:
    king_bytes = safetensors_total_bytes(king_dir)
    chall_bytes = safetensors_total_bytes(challenger_dir)
    cap = int(king_bytes * constants.MAX_CHALLENGER_SIZE_RATIO)
    if chall_bytes > cap:
        return [
            IntakeFailure(
                "size_cap",
                f"safetensors bytes {chall_bytes} exceed cap {cap} "
                f"({constants.MAX_CHALLENGER_SIZE_RATIO}x king)",
            )
        ]
    return []


def weight_fingerprint(folder: Path) -> str:
    """Content fingerprint over a snapshot's weight shards.

    Two checkpoints with identical weights produce the same fingerprint no
    matter which repo they were uploaded to. That is the point: an ``hf:``
    digest is a *revision* hash, so the same weights pushed to a second repo get
    a different digest and slip past digest-ownership entirely.
    """
    from epago.model.store import file_sha256

    shards = sorted(p for p in folder.rglob("*.safetensors") if p.is_file())
    joined = "|".join(f"{p.name}:{file_sha256(p)}" for p in shards)
    return hashlib.sha256(joined.encode()).hexdigest()


def exact_copy_of_king(challenger_dir: Path, king_dir: Path) -> bool:
    """Per-shard content equality with the reigning king."""
    return weight_fingerprint(challenger_dir) == weight_fingerprint(king_dir)
