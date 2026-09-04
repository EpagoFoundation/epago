"""Replayable audit trail.

Every duel produces one :class:`~epago.core.types.AuditRecord` whose canonical
JSON digest is committed on-chain inside the ``ev3`` verdict (first 16 hex
chars). An auditor holding the same chain data, corpus snapshot, and code can
re-run the duel and recompute the identical digest, so a validator cannot
retroactively rewrite what it evaluated.

Signature convention (verification-order safe): :func:`record_digest` — and
therefore :func:`audit16`, the value the on-chain ``ev3`` verdict commits to —
is computed over the record's *canonical unsigned form*: the full record,
``extra`` included, with ``validator_signature`` forced to ``""``. The
validator then signs that digest and stores the signature alongside, in the
``validator_signature`` field of the logged record. Because signing never
changes the digest, an auditor can (1) recompute the unsigned digest from the
logged record, (2) match it against the on-chain ``ev3`` commitment, and
(3) verify the stored signature over that same digest — in any order.

The append-only JSONL log is additionally chain-checkpointed: every
``AUDIT_CHAIN_COMMIT_EVERY`` records the validator publishes
``ea1|<n_records>|<rolling_digest16>``, binding the whole local log to the
chain. Public task sets are published with a delay
(``AUDIT_PUBLISH_DELAY_BLOCKS``) — delayed transparency — so tasks can be
audited after the fact without leaking a live holdout.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from epago import constants
from epago.core.types import AuditRecord

_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def build_audit_record(
    *,
    round_id: str,
    block_hash_at_reveal: str,
    author_hotkey: str,
    king_repo: str,
    king_digest: str,
    challenger_repo: str,
    challenger_digest: str,
    corpus_digest: str,
    taskgen_release: str,
    public_seed: str,
    public_task_ids_digest: str,
    private_pool_digest: str,
    private_pool_epoch: int,
    n_private_tasks: int,
    boot_seed: str,
    king_acc_ema: float,
    delta_threshold: float,
    mu_hat_pub: float,
    lcb_pub: float,
    mu_hat_priv: float,
    accepted: bool,
    harness_digest: str,
    judge_model_digest: str,
    eval_code_digest: str,
    judge_invocation_rate: float,
    revealed_at_block: int,
    intake_at_block: int,
    verdict_at_block: int,
    validator_hotkey: str,
    public_pool_digest: str = "",
    public_pool_manifest_digest: str = "",
    validator_signature: str = "",
    extra: dict | None = None,
) -> AuditRecord:
    """Fill every field of :class:`~epago.core.types.AuditRecord` explicitly."""
    return AuditRecord(
        round_id=round_id,
        block_hash_at_reveal=block_hash_at_reveal,
        author_hotkey=author_hotkey,
        king_repo=king_repo,
        king_digest=king_digest,
        challenger_repo=challenger_repo,
        challenger_digest=challenger_digest,
        corpus_digest=corpus_digest,
        taskgen_release=taskgen_release,
        public_seed=public_seed,
        public_task_ids_digest=public_task_ids_digest,
        public_pool_digest=public_pool_digest,
        public_pool_manifest_digest=public_pool_manifest_digest,
        private_pool_digest=private_pool_digest,
        private_pool_epoch=private_pool_epoch,
        n_private_tasks=n_private_tasks,
        boot_seed=boot_seed,
        king_acc_ema=king_acc_ema,
        delta_threshold=delta_threshold,
        mu_hat_pub=mu_hat_pub,
        lcb_pub=lcb_pub,
        mu_hat_priv=mu_hat_priv,
        accepted=accepted,
        harness_digest=harness_digest,
        judge_model_digest=judge_model_digest,
        eval_code_digest=eval_code_digest,
        judge_invocation_rate=judge_invocation_rate,
        revealed_at_block=revealed_at_block,
        intake_at_block=intake_at_block,
        verdict_at_block=verdict_at_block,
        validator_hotkey=validator_hotkey,
        validator_signature=validator_signature,
        extra=dict(extra or {}),
    )


def canonical_json(record: AuditRecord, *, unsigned: bool = False) -> str:
    """Canonical serialization: sorted keys, no whitespace.

    With ``unsigned=True`` the ``validator_signature`` field is forced to
    ``""`` — the canonical unsigned form that :func:`record_digest` hashes.
    The default (signed) form is what :class:`AuditLog` writes to disk, so
    the stored line carries the signature while the digest ignores it.
    """
    data = asdict(record)
    if unsigned:
        data["validator_signature"] = ""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def record_digest(record: AuditRecord) -> str:
    """sha256 hex digest of the canonical *unsigned* JSON.

    The signature field is zeroed before hashing (see the module docstring):
    the on-chain ``ev3`` verdict commits to this digest, and the validator's
    signature signs this digest, so setting the signature afterwards can never
    invalidate what the verdict committed to.
    """
    return hashlib.sha256(canonical_json(record, unsigned=True).encode()).hexdigest()


def audit16(record: AuditRecord) -> str:
    """First 16 hex chars of the record digest — exactly what goes into ``ev3``."""
    return record_digest(record)[:16]


def package_code_digest() -> str:
    """Digest of every ``.py`` file in the installed epago package.

    Used for the audit record's ``harness_digest`` / ``eval_code_digest``
    fields so a replayer can verify it is running the same mechanism code.
    """
    import epago

    root = Path(epago.__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return "sha256:" + h.hexdigest()


class AuditLog:
    """Append-only JSONL audit log with rolling chain checkpoints.

    The rolling digest folds each canonical record line into a hash chain
    (``rolling = sha256(rolling + sha256(line))``), so the checkpoint payload
    commits to the entire history, not just the latest record.
    """

    def __init__(self, state_dir: str | Path) -> None:
        self.dir = Path(state_dir) / "audit"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "audit.jsonl"
        self.delayed_dir = self.dir / "delayed"
        self.published_dir = self.dir / "published"
        self.delayed_dir.mkdir(exist_ok=True)
        self.published_dir.mkdir(exist_ok=True)
        self.n_records = 0
        self.rolling_digest = "0" * 64
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._fold(line)
        # Never re-publish checkpoints for history that predates this process.
        every = constants.AUDIT_CHAIN_COMMIT_EVERY
        self._last_checkpoint = self.n_records - (self.n_records % every)

    def _fold(self, line: str) -> None:
        line_digest = hashlib.sha256(line.encode()).hexdigest()
        self.rolling_digest = hashlib.sha256(
            (self.rolling_digest + line_digest).encode()
        ).hexdigest()
        self.n_records += 1

    def append(self, record: AuditRecord) -> str:
        """Append one record; returns its full sha256 hex digest."""
        line = canonical_json(record)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
        self._fold(line)
        return record_digest(record)

    # ---- chain checkpoints -------------------------------------------------

    def chain_checkpoint_due(self) -> bool:
        return self.n_records - self._last_checkpoint >= constants.AUDIT_CHAIN_COMMIT_EVERY

    def checkpoint_payload(self) -> str:
        return f"ea1|{self.n_records}|{self.rolling_digest[:16]}"

    def mark_checkpointed(self) -> None:
        self._last_checkpoint = self.n_records

    # ---- delayed transparency ------------------------------------------------

    def stage_delayed(self, name: str, payload: str, publish_at_block: int) -> Path:
        """Stage a payload (e.g. a round's public task set) for delayed release."""
        safe = _NAME_SAFE_RE.sub("_", name)
        target = self.delayed_dir / f"{publish_at_block:012d}_{safe}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(payload)
        tmp.replace(target)
        return target

    def release_due(self, current_block: int) -> list[Path]:
        """Move every staged payload whose delay elapsed into the public dir."""
        released: list[Path] = []
        for p in sorted(self.delayed_dir.glob("*.json")):
            try:
                publish_at = int(p.name.split("_", 1)[0])
            except ValueError:
                continue
            if publish_at <= current_block:
                dest = self.published_dir / p.name
                p.replace(dest)
                released.append(dest)
        return released
