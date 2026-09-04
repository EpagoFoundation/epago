"""Publishing tests: state sync to a the object store key namespace with manifest-based
change detection, content-addressed king mirroring, mirror resolution, and the
fallback-aware model store.

Hermetic: a :class:`FakeStore` stands in for :class:`ObjectStore` and a stub
downloader replaces the mirrors.json fetch — no network, no credentials, no
boto3 required at test time.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from epago.core.types import ModelRef
from epago.model import store
from epago.model.store import ModelStoreError, materialize_model, snapshot_digest
from epago.publishing.publisher import (
    MirrorResolver,
    StatePublisher,
    publish_king_mirror,
    update_mirror_manifest,
)


class FakeStore:
    """In-memory the object store store: the shared-bucket ops publishing/mirroring use."""

    def __init__(self, bucket: str = "drstore-test", fail_keys: set[str] | None = None):
        self.bucket = bucket
        self.objs: dict[str, bytes] = {}  # key -> bytes
        self.fail_keys = set(fail_keys or ())

    def require_bucket(self) -> str:
        if not self.bucket:
            raise ModelStoreError("EPAGO_S3_BUCKET is not set")
        return self.bucket

    def put_object(self, key: str, path) -> None:
        if key in self.fail_keys:
            raise RuntimeError(f"simulated upload failure for {key}")
        assert Path(path).is_file()
        self.objs[key] = Path(path).read_bytes()

    def get_object(self, key: str, dest) -> None:
        if key not in self.objs:
            raise KeyError(key)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objs[key])

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(k for k in self.objs if k.startswith(prefix))

    def upload_snapshot(self, repo: str, folder) -> str:
        digest = snapshot_digest(Path(folder))
        prefix = f"{repo}/{digest.removeprefix('sha256:')}"
        for p in sorted(Path(folder).rglob("*")):
            if p.is_file():
                self.objs[f"{prefix}/{p.relative_to(folder).as_posix()}"] = p.read_bytes()
        return digest


@pytest.fixture()
def state_dir(tmp_path: Path) -> Path:
    """A validator state dir with every publishable artifact plus embargoed ones."""
    sd = tmp_path / "state"
    (sd / "publications").mkdir(parents=True)
    (sd / "publications" / "pool_epoch_0001.json").write_text('{"epoch": 1}')
    (sd / "audit").mkdir()
    (sd / "audit" / "audit.jsonl").write_text('{"round_id": "r1"}\n')
    (sd / "audit" / "published").mkdir()
    (sd / "audit" / "published" / "000000000010_r1_tasks.json").write_text("[]")
    (sd / "audit" / "delayed").mkdir()
    (sd / "audit" / "delayed" / "000000000999_r2_tasks.json").write_text("[]")  # embargoed
    (sd / "dashboard").mkdir()
    (sd / "dashboard" / "dashboard.json").write_text('{"schema": "epd1"}')
    (sd / "state.json").write_text("{}")  # internal, never published
    return sd


# --- StatePublisher.sync -----------------------------------------------------------

REPO = "val/epago-state"


def _keyed(remotes: list[str]) -> set[str]:
    return {f"{REPO}/{r}" for r in remotes}


def test_sync_uploads_public_layout_and_nothing_else(state_dir):
    store_ = FakeStore()
    report = StatePublisher(state_dir, REPO, store=store_).sync()

    assert report.ok
    assert sorted(report.uploaded) == [
        "audit/audit.jsonl",
        "audit/published/000000000010_r1_tasks.json",
        "dashboard/dashboard.json",
        "publications/pool_epoch_0001.json",
    ]
    # Objects land under the validator's key namespace in the shared bucket.
    assert set(store_.objs) == _keyed(report.uploaded)
    # Embargoed and internal files never ship.
    assert not any("delayed" in k for k in store_.objs)
    assert not any(k.endswith("state.json") for k in store_.objs)


def test_sync_manifest_skips_unchanged_and_reships_changed(state_dir):
    first = StatePublisher(state_dir, REPO, store=FakeStore()).sync()
    assert len(first.uploaded) == 4 and first.skipped == []

    # Second sync with a fresh store: manifest persists on disk, so nothing ships.
    store2 = FakeStore()
    second = StatePublisher(state_dir, REPO, store=store2).sync()
    assert second.uploaded == []
    assert sorted(second.skipped) == sorted(first.uploaded)
    assert store2.objs == {}

    # Touch one file (content change) and add one: only those ship.
    (state_dir / "audit" / "audit.jsonl").write_text('{"round_id": "r1"}\n{"round_id": "r2"}\n')
    (state_dir / "publications" / "pool_epoch_0002.json").write_text('{"epoch": 2}')
    store3 = FakeStore()
    third = StatePublisher(state_dir, REPO, store=store3).sync()
    assert sorted(third.uploaded) == ["audit/audit.jsonl", "publications/pool_epoch_0002.json"]
    assert len(third.skipped) == 3
    assert set(store3.objs) == _keyed(third.uploaded)


def test_sync_collects_errors_and_retries_next_pass(state_dir):
    failing = FakeStore(fail_keys={f"{REPO}/dashboard/dashboard.json"})
    report = StatePublisher(state_dir, REPO, store=failing).sync()
    assert not report.ok
    assert [p for p, _ in report.errors] == ["dashboard/dashboard.json"]
    assert len(report.uploaded) == 3  # everything else still shipped

    # The failed file stayed out of the manifest, so the next sync retries it.
    retry = StatePublisher(state_dir, REPO, store=FakeStore()).sync()
    assert retry.ok
    assert retry.uploaded == ["dashboard/dashboard.json"]


def test_sync_never_raises_even_when_bucket_is_unreachable(state_dir):
    class ExplodingStore:
        def require_bucket(self):
            raise ConnectionError("bucket unreachable")

    report = StatePublisher(state_dir, REPO, store=ExplodingStore()).sync()
    assert not report.ok
    assert report.uploaded == []
    assert "bucket unreachable" in report.errors[0][1]


def test_sync_never_raises_when_bucket_is_unset(state_dir):
    report = StatePublisher(state_dir, REPO, store=FakeStore(bucket="")).sync()
    assert not report.ok
    assert report.uploaded == []
    assert "EPAGO_S3_BUCKET is not set" in report.errors[0][1]


def test_sync_on_empty_state_dir_is_a_clean_noop(tmp_path):
    store_ = FakeStore()
    report = StatePublisher(tmp_path / "empty", REPO, store=store_).sync()
    assert report.ok and report.uploaded == [] and report.skipped == []
    assert store_.objs == {}


# --- king mirroring -----------------------------------------------------------------


KING_DIGEST = "sha256:" + "ab" * 32


def _make_snapshot(path: Path, weights: bytes = b"king-weights") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "qwen3"}')
    (path / "model.safetensors").write_bytes(weights)
    return path


def test_publish_king_mirror_is_content_addressed_and_verifiable(tmp_path, state_dir):
    king_dir = _make_snapshot(tmp_path / "king")
    (king_dir / ".epago_verified").touch()  # marker must not enter the mirror digest
    expected_digest = snapshot_digest(_make_snapshot(tmp_path / "clean"))  # same bytes, no marker

    store_ = FakeStore()
    ref = publish_king_mirror(king_dir, KING_DIGEST, "val/epago-king-mirror", store=store_)
    # Mirror is content-addressed: its digest is the clean snapshot's, not KING_DIGEST.
    assert ref == ModelRef(repo="val/epago-king-mirror", digest=expected_digest)
    stored = store_.list_keys(f"val/epago-king-mirror/{expected_digest.removeprefix('sha256:')}/")
    assert any(k.endswith("/model.safetensors") for k in stored)
    assert not any(".epago_verified" in k for k in stored)  # dotfile excluded

    # The original->mirror pin is recorded keyed by the original commitment.
    path = update_mirror_manifest(state_dir, KING_DIGEST, ref)
    assert path == state_dir / "publications" / "mirrors.json"
    data = json.loads(path.read_text())
    assert data == {KING_DIGEST: [f"val/epago-king-mirror@{ref.digest}"]}

    # Recording again is idempotent; a second distinct mirror appends.
    update_mirror_manifest(state_dir, KING_DIGEST, ref)
    other = ModelRef(repo="val2/epago-king-mirror", digest="sha256:" + "9" * 64)
    update_mirror_manifest(state_dir, KING_DIGEST, other)
    data = json.loads(path.read_text())
    assert data[KING_DIGEST] == [
        f"val/epago-king-mirror@{ref.digest}",
        f"val2/epago-king-mirror@{other.digest}",
    ]

    # mirrors.json lives in publications/ so the next sync ships it.
    report = StatePublisher(state_dir, REPO, store=FakeStore()).sync()
    assert "publications/mirrors.json" in report.uploaded


# --- MirrorResolver -------------------------------------------------------------------


def test_mirror_resolver_reads_local_and_publisher_mirrors(tmp_path):
    d1 = "sha256:" + "1" * 64
    d2 = "sha256:" + "2" * 64
    local = tmp_path / "mirrors.json"
    local.write_text(json.dumps({KING_DIGEST: [f"val/mirror@{d1}", "garbage-no-at"]}))

    remote = tmp_path / "remote_mirrors.json"
    remote.write_text(json.dumps({KING_DIGEST: [f"val2/mirror@{d2}", f"val/mirror@{d1}"]}))
    fetched: list[str] = []

    def downloader(repo_id: str) -> Path:
        fetched.append(repo_id)
        if repo_id == "down/publisher":
            raise RuntimeError("publisher offline")
        return remote

    resolver = MirrorResolver(
        mirror_files=[local, tmp_path / "does-not-exist.json"],
        publisher_repos=["down/publisher", "val2/epago-state"],
        downloader=downloader,
    )
    primary = ModelRef(repo="miner/orig", digest=KING_DIGEST)
    refs = resolver.resolve(primary)
    assert refs == [
        ModelRef(repo="val/mirror", digest=d1),
        ModelRef(repo="val2/mirror", digest=d2),
    ]
    assert fetched == ["down/publisher", "val2/epago-state"]

    # Cached: a second resolve does not refetch; refresh() does.
    resolver.resolve(primary)
    assert len(fetched) == 2
    resolver.refresh()
    resolver.resolve(primary)
    assert len(fetched) == 4

    # Unknown digests resolve to nothing; the primary itself is never returned.
    assert resolver.resolve(ModelRef(repo="x/y", digest="sha256:" + "ff" * 32)) == []
    assert primary not in refs


def test_mirror_resolver_fetches_from_the_object_store(tmp_path):
    """The default (non-injected) downloader pulls mirrors.json from the store."""
    d1 = "sha256:" + "a" * 64
    store_ = FakeStore()
    mirrors = tmp_path / "m.json"
    mirrors.write_text(json.dumps({KING_DIGEST: [f"val/mirror@{d1}"]}))
    store_.put_object("val2/epago-state/publications/mirrors.json", mirrors)

    resolver = MirrorResolver(publisher_repos=["val2/epago-state"], store=store_)
    refs = resolver.resolve(ModelRef(repo="miner/orig", digest=KING_DIGEST))
    assert refs == [ModelRef(repo="val/mirror", digest=d1)]


# --- materialize_model fallbacks ---------------------------------------------------------


def test_materialize_prefers_local_mirror_without_network(tmp_path):
    snap = _make_snapshot(tmp_path / "src")
    digest = snapshot_digest(snap)
    mirror_root = tmp_path / "mirrors"
    mirrored = mirror_root / digest.replace(":", "_")
    shutil.copytree(snap, mirrored)
    (mirrored / ".epago_verified").touch()  # marker must not poison the digest

    ref = ModelRef(repo="miner/deleted-repo", digest=digest)
    # sha256 primary would hit the (unconfigured) oci backend; the local mirror wins first.
    out = materialize_model(ref, tmp_path / "cache", local_mirrors=[mirror_root])
    assert (out / "model.safetensors").read_bytes() == b"king-weights"
    assert (out / ".epago_verified").exists()
    # Idempotent: second call returns the verified cache without re-copying.
    assert materialize_model(ref, tmp_path / "cache") == out


def test_materialize_local_mirror_for_hf_primary(tmp_path):
    snap = _make_snapshot(tmp_path / "src")
    digest = "hf:" + "cd" * 20
    mirror_root = tmp_path / "mirrors"
    shutil.copytree(snap, mirror_root / digest.replace(":", "_"))

    ref = ModelRef(repo="miner/deleted-repo", digest=digest)
    out = materialize_model(ref, tmp_path / "cache", local_mirrors=[mirror_root])
    assert (out / "config.json").exists()


def test_materialize_sha256_fallback_verified_after_download(tmp_path, monkeypatch):
    good = _make_snapshot(tmp_path / "good")
    digest = snapshot_digest(good)
    evil = _make_snapshot(tmp_path / "evil", weights=b"swapped-weights")

    sources = {"val-evil/mirror": evil, "val-good/mirror": good}
    attempts: list[str] = []

    def fake_oci(ref: ModelRef, target: Path) -> None:
        attempts.append(ref.repo)
        if ref.repo == "miner/orig":
            raise RuntimeError("repo deleted upstream")
        src = sources[ref.repo]
        for p in src.iterdir():
            shutil.copy2(p, target / p.name)
        (target / ".cache").mkdir()  # downloader junk must not poison the digest
        (target / ".cache" / "junk").write_text("x")

    monkeypatch.setattr(store, "_materialize_oci", fake_oci)
    primary = ModelRef(repo="miner/orig", digest=digest)
    fallbacks = [
        ModelRef(repo="val-evil/mirror", digest="sha256:" + "1" * 64),
        ModelRef(repo="val-good/mirror", digest="sha256:" + "2" * 64),
    ]
    # sha256 primary uses the oci backend (raises here for miner/orig) -> content-addressed
    # fallbacks tried in order; the tampered mirror fails digest verification, the honest one passes.
    out = materialize_model(primary, tmp_path / "cache", fallbacks=fallbacks)
    assert attempts == ["miner/orig", "val-evil/mirror", "val-good/mirror"]
    assert (out / "model.safetensors").read_bytes() == b"king-weights"
    assert not (out / ".cache").exists()


def test_materialize_hf_primary_never_accepts_remote_fallbacks(tmp_path, monkeypatch):
    def fail_hf(ref: ModelRef, target: Path) -> None:
        raise RuntimeError("repo deleted upstream")

    monkeypatch.setattr(store, "_materialize_hf", fail_hf)
    primary = ModelRef(repo="miner/orig", digest="hf:" + "ab" * 20)
    fallback = ModelRef(repo="val/mirror", digest="sha256:" + "cd" * 32)
    with pytest.raises(ModelStoreError) as excinfo:
        materialize_model(primary, tmp_path / "cache", fallbacks=[fallback])
    message = str(excinfo.value)
    assert "remote fallbacks skipped" in message
    assert "repo deleted upstream" in message


def test_materialize_two_arg_call_still_works(tmp_path, monkeypatch):
    snap = _make_snapshot(tmp_path / "src")

    def fake_hf(ref: ModelRef, target: Path) -> None:
        for p in snap.iterdir():
            shutil.copy2(p, target / p.name)

    monkeypatch.setattr(store, "_materialize_hf", fake_hf)
    ref = ModelRef(repo="miner/orig", digest="hf:" + "ee" * 20)
    out = materialize_model(ref, tmp_path / "cache")
    assert (out / "model.safetensors").exists()


# --- fetch_public_state / s3:// source ----------------------------------------------


def test_fetch_public_state_and_object_store_source(tmp_path):
    from epago.miner import workflow

    store_ = FakeStore()
    doc = {"king": {"author_hotkey": "5" + "H" * 47}}
    src = tmp_path / "dashboard.json"
    src.write_text(json.dumps(doc))
    store_.put_object("val/epago-state/dashboard/dashboard.json", src)

    assert workflow.fetch_public_state("val/epago-state", "dashboard/dashboard.json", store=store_) == doc
    # A missing object resolves to None, never raises.
    assert workflow.fetch_public_state("val/epago-state", "dashboard/missing.json", store=store_) is None

    # load_public_state parses the s3:// scheme and defaults the path.
    import epago.model.objectstore as store_mod

    orig = store_mod.ObjectStore
    store_mod.ObjectStore = lambda *a, **k: store_  # default store -> our fake
    try:
        assert workflow.load_public_state("s3://val/epago-state") == doc
        store_.put_object(
            "val/epago-state/audit/published/round.json",
            _write(tmp_path / "r.json", json.dumps([1, 2])),
        )
        assert workflow.load_public_state(
            "s3://val/epago-state/audit/published/round.json"
        ) == [1, 2]
        with pytest.raises(FileNotFoundError):
            workflow.load_public_state("s3://val/epago-state/dashboard/missing.json")
        with pytest.raises(ValueError):
            workflow.load_public_state("s3://only-namespace")
    finally:
        store_mod.ObjectStore = orig

    # Local-path mode keeps working untouched.
    assert workflow.load_public_state(str(src)) == doc


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# --- CLI ---------------------------------------------------------------------------------


def test_publishing_cli_help_smoke():
    from epago.publishing.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("sync", "watch", "mirror-king"):
        assert command in result.output


def test_publishing_cli_sync_one_shot(state_dir, monkeypatch):
    from epago.publishing import publisher as publisher_mod
    from epago.publishing.cli import app

    store_ = FakeStore()
    monkeypatch.setattr(publisher_mod, "_make_store", lambda: store_)
    runner = CliRunner()
    result = runner.invoke(app, ["sync", "--state-dir", str(state_dir), "--repo-id", REPO])
    assert result.exit_code == 0, result.output
    assert "4 uploaded" in result.output
    assert len(store_.objs) == 4


def test_publishing_cli_mirror_king_records_manifest(tmp_path, state_dir, monkeypatch):
    from epago.publishing import publisher as publisher_mod
    from epago.publishing.cli import app

    king_dir = _make_snapshot(tmp_path / "king")
    expected_digest = snapshot_digest(king_dir)
    store_ = FakeStore()
    monkeypatch.setattr(publisher_mod, "_make_store", lambda: store_)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "mirror-king",
            "--king-dir", str(king_dir),
            "--digest", KING_DIGEST,
            "--repo-id", "val/epago-king-mirror",
            "--state-dir", str(state_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads((state_dir / "publications" / "mirrors.json").read_text())
    assert data == {KING_DIGEST: [f"val/epago-king-mirror@{expected_digest}"]}


# --- a crowned model becomes public -----------------------------------------


def test_a_crowned_model_is_published_where_anyone_can_fetch_it(tmp_path):
    """The hinge of the private-submission design.

    A challenger uploads into `submissions/<hotkey>/`, readable only by its
    author, so a model that loses is never handed to the rivals that beat it.
    Winning reverses that: the model collecting emissions must be fetchable
    and re-scorable by anyone, or the subnet asks to be trusted instead of
    checked.
    """
    from epago.publishing.publisher import king_object_repo, publish_king

    king_dir = tmp_path / "king"
    king_dir.mkdir()
    (king_dir / "model.safetensors").write_bytes(b"weights")
    (king_dir / ".cache_marker").write_text("local only")

    uploaded = {}

    class _Store:
        def upload_snapshot(self, repo, folder):
            uploaded["repo"] = repo
            uploaded["files"] = sorted(p.name for p in folder.rglob("*") if p.is_file())
            return "sha256:" + "a" * 64

    ref = publish_king(king_dir, "sha256:" + "a" * 64, store=_Store())

    # Published under the digest, not under the author: the bytes are
    # verifiable against the coronation commitment, and re-crowning the same
    # model resolves to the same object rather than a second copy.
    assert uploaded["repo"] == king_object_repo("sha256:" + "a" * 64)
    assert "submissions/" not in uploaded["repo"]
    assert ref.digest == "sha256:" + "a" * 64
    # Local cache markers never travel.
    assert uploaded["files"] == ["model.safetensors"]


def test_a_publish_reports_the_digest_of_what_it_actually_stored(tmp_path):
    """Publishing bytes that are not the crowned model would be worse than not
    publishing at all.

    The returned ref carries the digest recomputed from what was uploaded, not
    the digest that was claimed, so a caller comparing the two sees the
    mismatch instead of inheriting a false pin.
    """
    from epago.publishing.publisher import publish_king

    class _Store:
        def upload_snapshot(self, repo, folder):
            return "sha256:" + "b" * 64  # not what was crowned

    folder = tmp_path / "king"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x")

    ref = publish_king(folder, "sha256:" + "a" * 64, store=_Store())

    assert ref.digest == "sha256:" + "b" * 64
    assert ref.digest != "sha256:" + "a" * 64
