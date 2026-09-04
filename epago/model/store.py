"""Content-addressed model storage.

A :class:`~epago.core.types.ModelRef` pins an immutable snapshot: ``hf:`` refs
pin a Hugging Face revision hash, ``sha256:`` refs pin an OCI-style artifact
digest. The digest is the binding commitment — evaluation always materializes
exactly the committed snapshot, so weights cannot be swapped after reveal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from fnmatch import fnmatch
from pathlib import Path
from typing import Sequence

from epago.core.types import ModelRef

logger = logging.getLogger(__name__)

_CANONICAL_INDEX = "model.safetensors.index.json"
_CANONICAL_SINGLE = "model.safetensors"
#: ``*.jinja`` is on this list because modern checkpoints ship their chat
#: template as a standalone ``chat_template.jinja`` rather than a field in
#: ``tokenizer_config.json``. The eval backend renders every prompt through
#: that template (:func:`epago.eval.backend.chat_prompt`), so leaving it
#: undownloaded would silently evaluate an ``hf:`` submission as if it were a
#: base model — the exact defect this list must not reintroduce.
_ALLOW_PATTERNS = ("*.safetensors", "*.json", "*.txt", "*.jinja", "tokenizer*")

#: Upper bound on one materialized snapshot, shared with the the object store backend.
#: The intake size cap counts only ``.safetensors`` and only runs after the
#: download, so without this a repo full of allowlisted junk fills the disk
#: before any gate sees it.
MAX_SNAPSHOT_BYTES = 64 * 1024**3  # 64 GiB


class ModelStoreError(RuntimeError):
    pass


def file_sha256(path: Path, chunk_bytes: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def snapshot_digest(folder: Path) -> str:
    """Deterministic digest over a snapshot folder: sorted (relpath, sha256) pairs."""
    entries = []
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            entries.append((str(p.relative_to(folder)), file_sha256(p)))
    payload = json.dumps(entries, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def materialize_model(
    ref: ModelRef,
    cache_dir: Path,
    fallbacks: Sequence[ModelRef] = (),
    local_mirrors: Sequence[Path] = (),
) -> Path:
    """Materialize the pinned snapshot into the local cache and verify it.

    Returns the local snapshot path. Idempotent: an already-verified snapshot
    is returned without network access. Sources are tried in order:

    1. ``local_mirrors`` — directories of snapshots this operator copied at
       mirror time (see :meth:`ValidatorState.mirror_king`), keyed by digest
       (``<digest with : -> _>/``). Trusted because the operator recorded them
       from an already-verified snapshot; still digest-checked for ``sha256:``
       refs and layout-checked for ``hf:`` refs.
    2. The primary ``ref`` itself.
    3. ``fallbacks`` — remote mirror refs (e.g. from
       :class:`epago.publishing.publisher.MirrorResolver`). Attempted **only
       when the primary digest is ``sha256:``**: a content-addressed digest is
       recomputed over whatever any source serves, so a lying mirror is caught.
       An ``hf:`` primary pins a revision of one specific repo — a different
       repo's revision hash proves nothing about content equality with the
       original commitment, so remote fallbacks for ``hf:`` primaries are
       never accepted (use a local mirror instead).

    Raises :class:`ModelStoreError` with every attempted source's failure when
    nothing verifies.
    """
    target = cache_dir / ref.digest.replace(":", "_")
    marker = target / ".epago_verified"
    if marker.exists():
        return target
    errors: list[str] = []
    digest_dirname = ref.digest.replace(":", "_")

    for mirror_root in local_mirrors:
        source = Path(mirror_root).expanduser() / digest_dirname
        if not source.is_dir():
            continue
        try:
            shutil.rmtree(target, ignore_errors=True)
            _copy_snapshot(source, target)
            _verify(ref, target)
            marker.touch()
            return target
        except (OSError, ModelStoreError) as exc:
            errors.append(f"local mirror {source}: {exc}")
            shutil.rmtree(target, ignore_errors=True)

    try:
        target.mkdir(parents=True, exist_ok=True)
        if ref.backend == "hf":
            _materialize_hf(ref, target)
        else:
            _materialize_oci(ref, target)
        _verify(ref, target)
        marker.touch()
        return target
    except Exception as exc:  # noqa: BLE001 - fall through to verifiable mirrors
        if not fallbacks and not errors:
            raise
        errors.append(f"primary {ref.repo}@{ref.digest}: {exc}")
        shutil.rmtree(target, ignore_errors=True)

    if ref.digest.startswith("sha256:"):
        for fb in fallbacks:
            try:
                target.mkdir(parents=True, exist_ok=True)
                if fb.backend == "hf":
                    _materialize_hf(fb, target)
                else:
                    _materialize_oci(fb, target)
                _prune_hidden(target)
                _verify(ref, target)  # against the PRIMARY digest, never the mirror's
                marker.touch()
                return target
            except Exception as exc:  # noqa: BLE001 - each mirror judged independently
                errors.append(f"fallback {fb.repo}@{fb.digest}: {exc}")
                shutil.rmtree(target, ignore_errors=True)
    elif fallbacks:
        errors.append(
            "remote fallbacks skipped: primary digest is hf:, which is not "
            "content-addressed, so a mirror's bytes cannot be verified against it"
        )

    raise ModelStoreError(
        f"could not materialize {ref.repo}@{ref.digest} from any source: " + "; ".join(errors)
    )


def _copy_snapshot(source: Path, target: Path) -> None:
    """Copy a snapshot folder, skipping dotfiles (cache markers, HF metadata)."""
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(source.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(source)
        if any(part.startswith(".") for part in rel.parts):
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _prune_hidden(target: Path) -> None:
    """Drop dot-entries a downloader may add (e.g. ``.cache/``) before digesting."""
    for p in target.iterdir():
        if not p.name.startswith("."):
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def _materialize_hf(ref: ModelRef, target: Path) -> None:
    from huggingface_hub import HfApi, snapshot_download

    revision = ref.digest.removeprefix("hf:")
    _check_hf_repo_size(ref, revision, HfApi())
    snapshot_download(
        repo_id=ref.repo,
        revision=revision,
        local_dir=target,
        allow_patterns=list(_ALLOW_PATTERNS),
    )


def _check_hf_repo_size(ref: ModelRef, revision: str, api) -> None:
    """Refuse an oversized repo before downloading a byte of it.

    The intake size cap only counts ``.safetensors`` and only runs *after*
    materialization, so a repo carrying an arbitrarily large ``.json`` or
    ``.txt`` — both on the download allowlist — was fetched in full before
    anything checked it. That is free disk exhaustion against every validator
    that dequeues the submission. The listing is metadata only; a repo that
    refuses to report sizes is downloaded as before rather than rejected, since
    the cap is a bound on abuse and not a correctness gate.
    """
    try:
        info = api.repo_info(repo_id=ref.repo, revision=revision, files_metadata=True)
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        logger.info("could not size %s@%s before download: %s", ref.repo, revision, exc)
        return
    total = 0
    for sibling in getattr(info, "siblings", None) or ():
        name = getattr(sibling, "rfilename", "")
        if not any(fnmatch(name, pattern) for pattern in _ALLOW_PATTERNS):
            continue
        total += getattr(sibling, "size", None) or 0
    if total > MAX_SNAPSHOT_BYTES:
        raise ModelStoreError(
            f"{ref.repo}@{ref.digest}: downloadable files total {total} bytes, "
            f"cap is {MAX_SNAPSHOT_BYTES}"
        )


def _materialize_oci(ref: ModelRef, target: Path) -> None:
    """Download a ``sha256:`` snapshot from the object store."""
    from epago.model.objectstore import ObjectStore  # lazy: boto3 is a chain extra

    ObjectStore().download_snapshot(ref.repo, ref.digest, target)


def _verify(ref: ModelRef, target: Path) -> None:
    if ref.backend == "hf":
        # The revision hash was resolved by the hub; verify layout is present.
        if not (target / _CANONICAL_SINGLE).exists() and not (target / _CANONICAL_INDEX).exists():
            shutil.rmtree(target, ignore_errors=True)
            raise ModelStoreError(f"{ref.repo}@{ref.digest}: no canonical safetensors layout")
        return
    actual = snapshot_digest(target)
    if actual != ref.digest:
        shutil.rmtree(target, ignore_errors=True)
        raise ModelStoreError(f"digest mismatch: expected {ref.digest}, got {actual}")


def upload_model_folder(folder: Path, repo: str, backend: str = "hf") -> ModelRef:
    """Upload a trained checkpoint and return its pinned ref.

    ``hf`` pins a Hugging Face commit; ``oci`` stores the folder content-addressed
    on the object store and pins the resulting ``sha256:`` digest.
    """
    if backend == "oci":
        from epago.model.objectstore import ObjectStore  # lazy: boto3 is a chain extra

        digest = ObjectStore().upload_snapshot(repo, folder)
        return ModelRef(repo=repo, digest=digest)
    if backend != "hf":
        raise ModelStoreError(f"unknown backend {backend!r} (expected 'hf' or 'oci')")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo, exist_ok=True)
    info = api.upload_folder(folder_path=str(folder), repo_id=repo)
    commit = getattr(info, "oid", None) or getattr(info, "commit_id", None)
    if not commit:
        raise ModelStoreError("upload did not return a revision hash")
    return ModelRef(repo=repo, digest=f"hf:{commit}")


def safetensors_total_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*.safetensors"))
