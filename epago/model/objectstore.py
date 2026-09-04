"""The object store — content-addressed (``sha256:``) snapshots and validator artifacts.

Cloudflare R2 by default, reached with **boto3** over the S3 API (SigV4,
path-style addressing, region ``auto``). Nothing here is R2-specific beyond the
defaults: any S3-compatible endpoint works by setting ``EPAGO_S3_ENDPOINT``, so
the backend is a deployment choice rather than a protocol commitment.

boto3 rather than a vendor SDK on purpose. The obvious alternatives wrap
``huggingface_hub`` >= 1.0, whose upper half is incompatible with the pinned
``transformers`` that vLLM needs (which caps huggingface-hub < 1.0). boto3 has
no such dependency, so this backend coexists with the eval stack.

Content addressing: a snapshot is stored under ``{repo}/{digest}/…`` where the
digest is :func:`epago.model.store.snapshot_digest` over the folder. A
``sha256:`` ref therefore pins immutable bytes exactly the way an ``hf:``
revision pins a commit — download resolves one specific object set, and
:func:`epago.model.store._verify` recomputes the digest to prove it.

Config is env-driven (credentials are secrets and stay out of chain.toml):
``EPAGO_S3_ENDPOINT`` (for R2, ``https://<account-id>.r2.cloudflarestorage.com``),
``EPAGO_S3_REGION`` (default ``auto``), ``EPAGO_S3_BUCKET`` (required), and
``EPAGO_S3_ACCESS_KEY`` / ``EPAGO_S3_SECRET_KEY``. The boto3 client is
injectable, so the layout and addressing logic is unit-tested without network
or credentials.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from epago.model.store import MAX_SNAPSHOT_BYTES, ModelStoreError, snapshot_digest

DEFAULT_ENDPOINT = ""  # R2 endpoints are account-scoped; there is no useful default
DEFAULT_REGION = "auto"  # R2 ignores region but SigV4 requires one

#: Hard bound on the object count in one snapshot. Miners hold prefix-scoped
#: write credentials, so an object listing is attacker-controlled input: a
#: snapshot that blows this or :data:`MAX_SNAPSHOT_BYTES` is refused before a
#: single byte lands.
MAX_SNAPSHOT_FILES = 4096


def safe_relative_key(rel: str) -> str:
    """Validate one snapshot-relative object key, or raise.

    S3 keys are free-form strings written by whoever holds bucket credentials —
    they are NOT paths we control. ``target / rel`` with ``rel`` absolute
    (``/etc/cron.d/x``, reachable via a doubled separator in the key) or
    containing ``..`` escapes the snapshot directory entirely and writes
    anywhere the validator process can reach. Worse, escaped files land outside
    the folder :func:`~epago.model.store.snapshot_digest` hashes, so the write
    is invisible to digest verification. Every key component is therefore
    checked before it is ever joined to a local path.
    """
    if not rel or rel != rel.strip():
        raise ModelStoreError(f"unsafe object key {rel!r}: empty or padded")
    if "\\" in rel or "\x00" in rel:
        raise ModelStoreError(f"unsafe object key {rel!r}: illegal character")
    pure = PurePosixPath(rel)
    if pure.is_absolute():
        raise ModelStoreError(f"unsafe object key {rel!r}: absolute path")
    for part in pure.parts:
        if part in ("..", "."):
            raise ModelStoreError(f"unsafe object key {rel!r}: traversal segment {part!r}")
    return rel


def _resolved_within(target: Path, rel: str) -> Path:
    """Join ``rel`` under ``target`` and prove the result stays inside it.

    Belt and braces over :func:`safe_relative_key`: symlinks already present in
    the target tree could otherwise redirect a syntactically clean key.
    """
    dest = (target / safe_relative_key(rel)).resolve()
    root = target.resolve()
    if not dest.is_relative_to(root):
        raise ModelStoreError(f"unsafe object key {rel!r}: resolves outside {root}")
    return dest


class ObjectStore:
    """Content-addressed snapshot storage over the object store."""

    def __init__(
        self,
        bucket: str | None = None,
        endpoint: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        client=None,
    ) -> None:
        self.bucket = bucket or os.environ.get("EPAGO_S3_BUCKET", "")
        self._endpoint = endpoint or os.environ.get("EPAGO_S3_ENDPOINT", DEFAULT_ENDPOINT)
        self._region = region or os.environ.get("EPAGO_S3_REGION", DEFAULT_REGION)
        self._access = access_key or os.environ.get("EPAGO_S3_ACCESS_KEY", "")
        self._secret = secret_key or os.environ.get("EPAGO_S3_SECRET_KEY", "")
        self._client = client  # injectable for tests

    def client(self):
        if self._client is None:
            import boto3  # deferred: chain-extra dep, no huggingface-hub coupling
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                region_name=self._region,
                aws_access_key_id=self._access or None,
                aws_secret_access_key=self._secret or None,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            )
        return self._client

    def require_bucket(self) -> str:
        if not self.bucket:
            raise ModelStoreError("EPAGO_S3_BUCKET is not set")
        return self.bucket

    # ---- generic object ops (the shared-bucket primitives) -------------------

    def put_object(self, key: str, path: Path) -> None:
        """Upload a single local file to ``key`` in the bucket."""
        self.client().upload_file(str(path), self.require_bucket(), key)

    def get_object(self, key: str, dest: Path) -> None:
        """Download the object at ``key`` into ``dest`` (parents created)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client().download_file(self.require_bucket(), key, str(dest))

    def list_objects(self, prefix: str) -> list[tuple[str, int]]:
        """Every ``(key, size_bytes)`` under ``prefix`` (handles pagination)."""
        c = self.client()
        out: list[tuple[str, int]] = []
        token: str | None = None
        while True:
            kw = {"Bucket": self.require_bucket(), "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = c.list_objects_v2(**kw)
            out.extend((o["Key"], int(o.get("Size", 0))) for o in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return out

    def list_keys(self, prefix: str) -> list[str]:
        """Every object key under ``prefix`` (handles pagination)."""
        return [key for key, _ in self.list_objects(prefix)]

    # ---- content-addressed snapshots (model backend) -------------------------

    def upload_snapshot(self, repo: str, folder: Path) -> str:
        """Store a folder content-addressed under ``{repo}/{digest}/…``; return the ref.

        The digest is computed first, so the upload path is a pure function of the
        bytes — re-uploading identical content is idempotent and yields the same
        immutable ``sha256:`` reference.

        Size and object count are checked **before** the first byte goes up.
        The download path already bounds both, but bounding only there means an
        oversized upload is paid for in full — in bandwidth, in storage, and in
        the validator's egress — and only then refused. Checking here is what
        makes a miner's quota a limit rather than a bill.
        """
        self.require_bucket()
        files = [p for p in sorted(Path(folder).rglob("*")) if p.is_file()]
        if len(files) > MAX_SNAPSHOT_FILES:
            raise ModelStoreError(
                f"snapshot has {len(files)} files, cap is {MAX_SNAPSHOT_FILES}"
            )
        total = sum(p.stat().st_size for p in files)
        if total > MAX_SNAPSHOT_BYTES:
            raise ModelStoreError(
                f"snapshot is {total} bytes, cap is {MAX_SNAPSHOT_BYTES}"
            )
        digest = snapshot_digest(folder)  # "sha256:<hex>"
        prefix = f"{repo}/{digest.removeprefix('sha256:')}"
        for p in files:
            rel = p.relative_to(folder).as_posix()
            self.put_object(f"{prefix}/{rel}", p)
        return digest

    def download_snapshot(self, repo: str, digest: str, target: Path) -> None:
        """Materialize the snapshot pinned by ``(repo, digest)`` into ``target``.

        Every key is validated and bounded *before* the first download: the
        listing comes from a shared bucket, so it is untrusted input. A single
        bad key aborts the whole snapshot rather than being skipped — a partial
        snapshot would fail digest verification anyway, and failing loudly keeps
        the reason in the intake record.
        """
        bucket = self.require_bucket()
        prefix = f"{repo}/{digest.removeprefix('sha256:')}/"
        objects = self.list_objects(prefix)
        if not objects:
            raise ModelStoreError(f"no objects at s3://{bucket}/{prefix}")

        planned: list[tuple[str, str]] = []
        total_bytes = 0
        for key, size in objects:
            rel = key[len(prefix):]
            if not rel:
                continue
            safe_relative_key(rel)  # raises on absolute paths and traversal
            total_bytes += size
            planned.append((key, rel))
        if len(planned) > MAX_SNAPSHOT_FILES:
            raise ModelStoreError(
                f"snapshot at {prefix} has {len(planned)} objects, cap is {MAX_SNAPSHOT_FILES}"
            )
        if total_bytes > MAX_SNAPSHOT_BYTES:
            raise ModelStoreError(
                f"snapshot at {prefix} is {total_bytes} bytes, cap is {MAX_SNAPSHOT_BYTES}"
            )

        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        for key, rel in planned:
            self.get_object(key, _resolved_within(target, rel))
