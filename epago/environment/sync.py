"""Corpus snapshot sync and digest verification.

The corpus snapshot is content-addressed exactly like model weights: the
``sha256:`` digest pinned in ``chain.toml`` (``eval.corpus_digest``) is the
binding commitment. Every validator materializes the same bytes or refuses to
evaluate — a corpus that drifts by even one page silently changes task
solvability and breaks cross-validator verdict agreement.

The bytes live in the object store, under
``{repo}/{digest}/corpus.db`` (shared-bucket layout). The digest in the key is
the same commitment verified after download, so the fetch is a pure function of
the pin — any validator pulls identical bytes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from epago.core.types import SHA256_DIGEST_RE
from epago.model.store import file_sha256

_SNAPSHOT_FILENAME = "corpus.db"
_VERIFIED_MARKER = ".verified"


class CorpusIntegrityError(RuntimeError):
    pass


def corpus_digest(db_path: Path) -> str:
    """``sha256:<hex>`` digest of the snapshot file bytes."""
    return "sha256:" + file_sha256(Path(db_path))


def verify_corpus(db_path: Path, expected_digest: str) -> None:
    if not SHA256_DIGEST_RE.match(expected_digest):
        raise CorpusIntegrityError(f"invalid corpus digest format: {expected_digest!r}")
    actual = corpus_digest(db_path)
    if actual != expected_digest:
        raise CorpusIntegrityError(
            f"corpus digest mismatch at {db_path}: expected {expected_digest}, got {actual}"
        )


def sync_corpus(repo: str, expected_digest: str, dest_dir: Path, store=None) -> Path:
    """Download the pinned corpus snapshot and verify it.

    Returns the local snapshot path. Idempotent: an already-verified snapshot
    is returned without network access. A failed verification removes the
    download so a retry starts clean. ``store`` is an injectable
    :class:`~epago.model.objectstore.ObjectStore` (defaults to env config).
    """
    if not SHA256_DIGEST_RE.match(expected_digest):
        raise CorpusIntegrityError(f"invalid corpus digest format: {expected_digest!r}")
    target = Path(dest_dir) / expected_digest.replace(":", "_")
    db_path = target / _SNAPSHOT_FILENAME
    marker = target / _VERIFIED_MARKER
    if marker.exists() and db_path.exists():
        return db_path

    from epago.model.objectstore import ObjectStore  # lazy: boto3 is a chain extra

    target.mkdir(parents=True, exist_ok=True)
    key = f"{repo}/{expected_digest.removeprefix('sha256:')}/{_SNAPSHOT_FILENAME}"
    store = store or ObjectStore()
    store.get_object(key, db_path)
    try:
        verify_corpus(db_path, expected_digest)
    except CorpusIntegrityError:
        shutil.rmtree(target, ignore_errors=True)
        raise
    marker.touch()
    return db_path


def publish_corpus(db_path: Path, repo: str, store=None) -> str:
    """Upload a corpus snapshot content-addressed; return its ``sha256:`` digest.

    Stored under ``{repo}/{digest}/corpus.db`` — the same key :func:`sync_corpus`
    reconstructs from ``(repo, digest)``. Idempotent: identical bytes yield the
    same key. Pin the returned digest in ``chain.toml`` as ``eval.corpus_digest``.
    """
    from epago.model.objectstore import ObjectStore  # lazy: boto3 is a chain extra

    db_path = Path(db_path)
    digest = corpus_digest(db_path)
    store = store or ObjectStore()
    store.put_object(f"{repo}/{digest.removeprefix('sha256:')}/{_SNAPSHOT_FILENAME}", db_path)
    return digest
