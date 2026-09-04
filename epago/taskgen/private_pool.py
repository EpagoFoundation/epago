"""Per-validator private holdout pool with delayed transparency.

Each validator holds a private task pool that models never see until after
rotation. The pool's canonical-JSON digest is committed on-chain in every
verdict that used it (``Verdict.private_pool_epoch`` + the audit record's
``private_pool_digest``), which pins the validator to one exact task set
*before* anyone can inspect it. On rotation the outgoing pool is published in
full — tasks, answers, evidence — as the exact canonical bytes the digest was
computed over, so anyone can replay every private verdict of that epoch
against the published file and the digest committed at verdict time. Private
tasks are therefore private only while active; afterwards they are audit
material.

Byte discipline: :meth:`PrivatePool.digest` and the file written by
:meth:`PrivatePool.rotate` are derived from the same
:meth:`PrivatePool.canonical_bytes`, so ``sha256(published file) ==
digest committed at verdict time`` holds exactly, not just semantically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epago import constants
from epago.core.types import Task, TaskOrigin

_POOL_FORMAT = "epago-private-pool-v1"
_STATE_FILENAME = "pool_state.json"


def _task_to_dict(task: Task) -> dict:
    return {
        "task_id": task.task_id,
        "question": task.question,
        "answer": task.answer,
        "aliases": list(task.aliases),
        "evidence_doc_ids": list(task.evidence_doc_ids),
        "masked_doc_ids": list(task.masked_doc_ids),
        "origin": task.origin.value,
        "template": task.template,
        "hops": task.hops,
    }


def _task_from_dict(data: dict) -> Task:
    return Task(
        task_id=data["task_id"],
        question=data["question"],
        answer=data["answer"],
        aliases=tuple(data["aliases"]),
        evidence_doc_ids=tuple(data["evidence_doc_ids"]),
        masked_doc_ids=tuple(data["masked_doc_ids"]),
        origin=TaskOrigin(data["origin"]),
        template=data["template"],
        hops=data["hops"],
    )


def _canonical_json(obj: dict) -> bytes:
    """Canonical JSON: sorted keys, no whitespace, ASCII-only escapes.

    ASCII escaping keeps the bytes independent of any downstream encoding
    choice; sorted keys and fixed separators make serialization injective for
    our payload shapes.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


@dataclass(frozen=True, slots=True)
class PrivatePool:
    """One epoch's private holdout. Immutable; rotation returns a new pool."""

    epoch: int
    created_block: int
    tasks: tuple[Task, ...]
    storage_path: Path | None = None

    # -- persistence ---------------------------------------------------------

    @classmethod
    def load(cls, directory: str | Path) -> "PrivatePool":
        directory = Path(directory)
        with open(directory / _STATE_FILENAME, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("format") != _POOL_FORMAT:
            raise ValueError(f"not a {_POOL_FORMAT} state file: {directory / _STATE_FILENAME}")
        return cls(
            epoch=data["epoch"],
            created_block=data["created_block"],
            tasks=tuple(_task_from_dict(t) for t in data["tasks"]),
            storage_path=directory,
        )

    def save(self, directory: str | Path | None = None) -> Path:
        """Persist the active pool state (validator-local, never published)."""
        target = Path(directory) if directory is not None else self.storage_path
        if target is None:
            raise ValueError("no storage directory given and pool has no storage_path")
        target.mkdir(parents=True, exist_ok=True)
        path = target / _STATE_FILENAME
        path.write_bytes(self.canonical_bytes())
        return path

    # -- commitment ----------------------------------------------------------

    def _payload(self) -> dict:
        # Tasks sorted by task_id: the digest must not depend on mint order.
        return {
            "format": _POOL_FORMAT,
            "epoch": self.epoch,
            "created_block": self.created_block,
            "tasks": [_task_to_dict(t) for t in sorted(self.tasks, key=lambda t: t.task_id)],
        }

    def canonical_bytes(self) -> bytes:
        """The exact bytes the digest commits to and rotation publishes."""
        return _canonical_json(self._payload())

    def digest(self) -> str:
        """Canonical-JSON digest, committed on-chain with every verdict."""
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()

    # -- use -------------------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator) -> list[Task]:
        """Draw ``n`` tasks without replacement over the sorted task list.

        Sorting before the draw keeps the sample a function of (pool digest,
        rng state) rather than of in-memory ordering.
        """
        ordered = sorted(self.tasks, key=lambda t: t.task_id)
        if not ordered:
            return []
        k = min(n, len(ordered))
        idx = rng.choice(len(ordered), size=k, replace=False)
        return [ordered[int(i)] for i in idx]

    def due_for_rotation(self, current_block: int) -> bool:
        return current_block - self.created_block >= constants.PRIVATE_POOL_ROTATION_BLOCKS

    def rotate(
        self, new_tasks: list[Task], current_block: int, publish_dir: Path
    ) -> "PrivatePool":
        """Publish this pool in full and return its epoch+1 successor.

        The published file is byte-for-byte :meth:`canonical_bytes`, named
        after the epoch and digest so auditors can locate it from a verdict's
        ``(private_pool_epoch, private_pool_digest)`` pair alone. The new pool
        is persisted to ``storage_path`` when one is set.
        """
        publish_dir = Path(publish_dir)
        publish_dir.mkdir(parents=True, exist_ok=True)
        digest_hex = self.digest().removeprefix("sha256:")
        out_path = publish_dir / f"private-pool-epoch{self.epoch:06d}-{digest_hex[:16]}.json"
        out_path.write_bytes(self.canonical_bytes())
        successor = PrivatePool(
            epoch=self.epoch + 1,
            created_block=current_block,
            tasks=tuple(new_tasks),
            storage_path=self.storage_path,
        )
        if successor.storage_path is not None:
            successor.save()
        return successor
