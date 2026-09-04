"""Sync validator state to the object store.

One key namespace (``repo_id``) per validator inside the shared bucket, with a
stable layout mirroring the local state directory:

    {repo_id}/publications/...      rotated private pools, mirrors.json
    {repo_id}/audit/audit.jsonl     the append-only duel audit log
    {repo_id}/audit/published/...   delay-released public task sets
    {repo_id}/dashboard/...         dashboard.json (+ static site) if exported

``audit/delayed/`` is deliberately **never** uploaded — those payloads are
still under their transparency embargo; only what :meth:`AuditLog.release_due`
has moved into ``audit/published/`` ships.

Change detection is a local manifest (``state_dir/.publish_manifest.json``,
remote path -> content sha256): unchanged files are never re-uploaded, so a
``watch`` loop is cheap. :meth:`StatePublisher.sync` never raises — every
failure is collected into the returned :class:`PublishReport` so one bad file
(or a network blip) cannot stall the validator loop that calls it.

King snapshots mirror to the object store via :func:`publish_king_mirror` content-
addressed; the resulting ``repo@sha256:<digest>`` pins are recorded in
``publications/mirrors.json`` (see :func:`update_mirror_manifest`) which the
next ``sync`` ships, and :class:`MirrorResolver` turns back into fallback
:class:`~epago.core.types.ModelRef` candidates for
:func:`epago.model.store.materialize_model`. Because an object-store mirror is
content-addressed, its bytes are verifiable against the original ``sha256:``
commitment — unlike an ``hf:`` mirror, which only pins its own commit.

The the object store store (boto3 under the hood) is import-guarded inside methods;
every network surface takes an injectable ``store``/``downloader`` so tests run
hermetically.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from epago.core.types import ModelRef
from epago.model.store import file_sha256

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".publish_manifest.json"
MIRRORS_FILENAME = "mirrors.json"
MIRRORS_REPO_PATH = f"publications/{MIRRORS_FILENAME}"

#: Where a crowned model is published, readable by anyone.
#:
#: This prefix is the hinge of the whole submission design. A challenger
#: uploads into ``submissions/<hotkey>/`` where only its author can read it, so
#: a model that loses is never exposed to the rivals it lost to. The moment one
#: is crowned it moves here, because the model actually collecting emissions
#: has to be re-scorable by anyone — that is the difference between "trust the
#: validator" and "check the validator".
#:
#: Content-addressed by digest rather than by author, so the published bytes
#: are verifiable against the coronation commitment and a re-crowning of the
#: same model resolves to the same object instead of a second copy.
KINGS_PREFIX = "kings"


def king_object_repo(digest: str) -> str:
    """The public key prefix a crowned model is published under."""
    return f"{KINGS_PREFIX}/{digest.removeprefix('sha256:').removeprefix('hf:')}"


class PublishError(RuntimeError):
    pass


@dataclass(slots=True)
class PublishReport:
    """Outcome of one :meth:`StatePublisher.sync` pass."""

    repo_id: str
    uploaded: list[str] = field(default_factory=list)   # remote paths shipped this pass
    skipped: list[str] = field(default_factory=list)    # unchanged since last pass
    errors: list[tuple[str, str]] = field(default_factory=list)  # (remote path, error)

    @property
    def ok(self) -> bool:
        return not self.errors


def _make_store():
    """Late import so the package imports without boto3 (a chain extra) installed."""
    try:
        from epago.model.objectstore import ObjectStore
    except ImportError as exc:  # pragma: no cover - exercised via error message tests
        raise PublishError(
            "boto3 is not installed; `pip install 'epago[chain]'` to publish "
            "validator state to the object store"
        ) from exc
    return ObjectStore()


def _is_hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


class StatePublisher:
    """Mirror a validator's public artifacts to its key namespace in the bucket."""

    def __init__(
        self,
        state_dir: str | Path,
        repo_id: str,
        store: Any = None,
    ) -> None:
        self.state_dir = Path(state_dir).expanduser()
        self.repo_id = repo_id
        self._store = store
        self._manifest_path = self.state_dir / MANIFEST_FILENAME

    # ---- sources -----------------------------------------------------------

    def _iter_source_files(self) -> list[tuple[Path, str]]:
        """(local file, path in repo) pairs, in a deterministic order.

        Only publicly releasable artifacts are listed; anything still under
        embargo (``audit/delayed/``) or private (``private_pool/``,
        ``state.json`` internals, model caches) never appears here.
        """
        pairs: list[tuple[Path, str]] = []

        def add_tree(root: Path, prefix: str) -> None:
            if not root.is_dir():
                return
            for p in sorted(root.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(root)
                if _is_hidden(rel):
                    continue
                pairs.append((p, f"{prefix}/{rel.as_posix()}"))

        add_tree(self.state_dir / "publications", "publications")
        audit_log = self.state_dir / "audit" / "audit.jsonl"
        if audit_log.is_file():
            pairs.append((audit_log, "audit/audit.jsonl"))
        add_tree(self.state_dir / "audit" / "published", "audit/published")
        add_tree(self.state_dir / "dashboard", "dashboard")
        return pairs

    # ---- manifest ------------------------------------------------------------

    def _load_manifest(self) -> dict[str, str]:
        try:
            data = json.loads(self._manifest_path.read_text())
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, sort_keys=True, indent=1))
        os.replace(tmp, self._manifest_path)

    # ---- sync ------------------------------------------------------------------

    def sync(self) -> PublishReport:
        """Upload every new or changed public artifact. Never raises.

        Files whose upload fails stay out of the manifest, so the next sync
        retries exactly those.
        """
        report = PublishReport(repo_id=self.repo_id)
        manifest = self._load_manifest()
        try:
            store = self._store if self._store is not None else _make_store()
            store.require_bucket()  # fail fast if EPAGO_S3_BUCKET is unset
        except Exception as exc:  # noqa: BLE001 - sync must never raise
            report.errors.append((self.repo_id, f"{type(exc).__name__}: {exc}"))
            return report

        for local, remote in self._iter_source_files():
            try:
                digest = file_sha256(local)
            except OSError as exc:
                report.errors.append((remote, f"{type(exc).__name__}: {exc}"))
                continue
            if manifest.get(remote) == digest:
                report.skipped.append(remote)
                continue
            try:
                store.put_object(f"{self.repo_id}/{remote}", local)
            except Exception as exc:  # noqa: BLE001 - collect, keep shipping the rest
                report.errors.append((remote, f"{type(exc).__name__}: {exc}"))
                continue
            manifest[remote] = digest
            report.uploaded.append(remote)

        try:
            self._save_manifest(manifest)
        except OSError as exc:
            report.errors.append((MANIFEST_FILENAME, f"{type(exc).__name__}: {exc}"))
        return report


# --- king mirroring -------------------------------------------------------------


def publish_king_mirror(
    king_dir: Path,
    digest: str,
    repo_id: str,
    store: Any = None,
) -> ModelRef:
    """Upload a mirrored king snapshot content-addressed to the object store.

    Returns the mirror's pinned ref (``sha256:<digest>``). Because the mirror is
    content-addressed, this digest equals the original ``digest`` for identical
    bytes — so the mirror is a *verifiable* fallback for the coronation
    commitment (unlike an ``hf:`` mirror, which only pins its own commit).
    Local cache markers and other dotfiles are excluded before hashing so the
    digest reflects only the model's own content.
    """
    import tempfile

    from epago.model.store import _copy_snapshot

    store = store if store is not None else _make_store()
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "snapshot"
        _copy_snapshot(Path(king_dir).expanduser(), clean)  # skips dotfiles
        mirror_digest = store.upload_snapshot(repo_id, clean)
    if mirror_digest != digest:
        logger.warning(
            "king mirror content digest %s != original commitment %s (repo %s)",
            mirror_digest,
            digest,
            repo_id,
        )
    logger.info("mirrored king %s to %s@%s", digest, repo_id, mirror_digest)
    return ModelRef(repo=repo_id, digest=mirror_digest)


def publish_king(
    king_dir: Path,
    digest: str,
    store: Any = None,
) -> ModelRef:
    """Publish a newly crowned model where everyone can fetch it.

    Returns the published ref. The digest is recomputed from the uploaded
    bytes, so a mismatch against the coronation commitment is reported rather
    than silently published — publishing something *other* than what was
    crowned would be worse than not publishing at all.

    Idempotent by construction: the destination is derived from the digest, so
    re-publishing the same king overwrites identical bytes.
    """
    import tempfile

    from epago.model.store import _copy_snapshot

    store = store if store is not None else _make_store()
    repo_id = king_object_repo(digest)
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "snapshot"
        _copy_snapshot(Path(king_dir).expanduser(), clean)  # skips dotfiles
        published = store.upload_snapshot(repo_id, clean)
    if published != digest:
        logger.warning(
            "published king content digest %s != coronation commitment %s (at %s)",
            published,
            digest,
            repo_id,
        )
    logger.info("published king %s at %s", digest, repo_id)
    return ModelRef(repo=repo_id, digest=published)


def update_mirror_manifest(state_dir: str | Path, digest: str, mirror_ref: ModelRef) -> Path:
    """Record a king mirror in ``publications/mirrors.json``.

    Layout: ``{<original digest>: ["<repo>@<hf:revision>", ...]}``. Lives in
    ``publications/`` so the next :meth:`StatePublisher.sync` ships it, letting
    any :class:`MirrorResolver` (local or remote) map an orphaned primary ref
    to surviving mirrors.
    """
    pub_dir = Path(state_dir).expanduser() / "publications"
    pub_dir.mkdir(parents=True, exist_ok=True)
    path = pub_dir / MIRRORS_FILENAME
    mirrors: dict[str, list[str]] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                mirrors = {str(k): [str(x) for x in v] for k, v in loaded.items()}
        except ValueError:
            logger.warning("corrupt %s; rewriting", path)
    entry = f"{mirror_ref.repo}@{mirror_ref.digest}"
    entries = mirrors.setdefault(digest, [])
    if entry not in entries:
        entries.append(entry)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mirrors, sort_keys=True, indent=1))
    os.replace(tmp, path)
    return path


class MirrorResolver:
    """Turn a primary :class:`ModelRef` into fallback mirror refs.

    Sources, merged and cached after first use:

    - ``mirror_files``: local ``mirrors.json`` paths (a validator's own
      ``publications/mirrors.json``, or one fetched out-of-band);
    - ``publisher_repos``: key namespaces of other validators' publishers in the
      shared bucket, fetched from the object store (import-guarded; a ``downloader``
      taking a repo id and returning a local ``mirrors.json`` path is
      injectable).

    Resolution is availability routing only — it finds *where else* bytes
    claiming to be the snapshot live. Whether a fallback may be trusted is
    decided by :func:`epago.model.store.materialize_model`'s verification
    rules, never here.
    """

    def __init__(
        self,
        mirror_files: Sequence[str | Path] = (),
        publisher_repos: Sequence[str] = (),
        store: Any = None,
        downloader: Callable[[str], str | Path] | None = None,
    ) -> None:
        self._mirror_files = [Path(p).expanduser() for p in mirror_files]
        self._publisher_repos = list(publisher_repos)
        self._store = store
        self._downloader = downloader
        self._cache: dict[str, list[str]] | None = None
        self._tmpdir: str | None = None

    def _download(self, repo_id: str) -> str | Path:
        if self._downloader is not None:
            return self._downloader(repo_id)
        import tempfile

        store = self._store if self._store is not None else _make_store()
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="epago-mirrors-")
        dest = Path(self._tmpdir) / f"{repo_id.replace('/', '_')}.json"
        store.get_object(f"{repo_id}/{MIRRORS_REPO_PATH}", dest)
        return dest

    def _merge(self, target: dict[str, list[str]], path: Path, origin: str) -> None:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            logger.info("skipping unreadable mirrors.json from %s: %s", origin, exc)
            return
        if not isinstance(data, dict):
            return
        for digest, entries in data.items():
            if not isinstance(entries, list):
                continue
            bucket = target.setdefault(str(digest), [])
            for entry in entries:
                if isinstance(entry, str) and entry not in bucket:
                    bucket.append(entry)

    def _mirrors(self) -> dict[str, list[str]]:
        if self._cache is not None:
            return self._cache
        merged: dict[str, list[str]] = {}
        for path in self._mirror_files:
            self._merge(merged, path, str(path))
        for repo_id in self._publisher_repos:
            try:
                path = Path(self._download(repo_id))
            except Exception as exc:  # noqa: BLE001 - missing/offline publishers are routine
                logger.info("no mirrors.json from publisher %s: %s", repo_id, exc)
                continue
            self._merge(merged, path, repo_id)
        self._cache = merged
        return merged

    def refresh(self) -> None:
        """Drop the cache so the next :meth:`resolve` re-reads every source."""
        self._cache = None

    def resolve(self, ref: ModelRef) -> list[ModelRef]:
        """Fallback refs recorded for ``ref.digest``, primary itself excluded."""
        out: list[ModelRef] = []
        for entry in self._mirrors().get(ref.digest, []):
            repo, sep, digest = entry.rpartition("@")
            if not sep:
                continue
            try:
                mirror = ModelRef(repo=repo, digest=digest)
            except ValueError:
                logger.info("ignoring malformed mirror entry %r", entry)
                continue
            if mirror != ref and mirror not in out:
                out.append(mirror)
        return out
