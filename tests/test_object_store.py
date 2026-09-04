"""the object store content-addressed model backend: upload/download round-trips to the
exact committed digest, so a ``sha256:`` ref pins immutable bytes. The boto3
client is faked in-memory — no network, no credentials, no huggingface-hub."""
from __future__ import annotations

from pathlib import Path

import pytest

from epago.model.objectstore import ObjectStore
from epago.model.store import ModelStoreError, snapshot_digest


class FakeS3:
    """Minimal in-memory S3: the three calls ObjectStore uses."""

    def __init__(self) -> None:
        self.objs: dict[tuple[str, str], bytes] = {}

    def upload_file(self, filename, bucket, key):
        self.objs[(bucket, key)] = Path(filename).read_bytes()

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objs[(bucket, key)])

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for (b, k) in self.objs if b == Bucket and k.startswith(Prefix))
        return {
            "Contents": [
                {"Key": k, "Size": len(self.objs[(Bucket, k)])} for k in keys
            ],
            "IsTruncated": False,
        }

    def put_raw(self, bucket, key, body=b"pwned"):
        """Plant an object under an arbitrary key, as any bucket writer can."""
        self.objs[(bucket, key)] = body


def _model_dir(tmp_path: Path) -> Path:
    src = tmp_path / "model"
    (src / "sub").mkdir(parents=True)
    (src / "config.json").write_text('{"architectures":["Qwen3ForCausalLM"]}')
    (src / "model.safetensors").write_bytes(b"\x00\x01weights")
    (src / "sub" / "tokenizer.json").write_text("{}")
    return src


def test_upload_is_content_addressed_and_matches_snapshot_digest(tmp_path) -> None:
    src = _model_dir(tmp_path)
    s3 = FakeS3()
    store = ObjectStore(bucket="epago-sn", client=s3)
    digest = store.upload_snapshot("org/EPAGO-DR-4B-x", src)

    assert digest == snapshot_digest(src)          # ref == the folder's content digest
    assert digest.startswith("sha256:")
    prefix = f"org/EPAGO-DR-4B-x/{digest.removeprefix('sha256:')}/"
    assert s3.objs and all(k.startswith(prefix) for (_, k) in s3.objs)  # stored under digest path


def test_download_round_trips_to_the_exact_digest(tmp_path) -> None:
    src = _model_dir(tmp_path)
    s3 = FakeS3()
    store = ObjectStore(bucket="epago-sn", client=s3)
    digest = store.upload_snapshot("org/model", src)

    out = tmp_path / "out"
    store.download_snapshot("org/model", digest, out)
    assert snapshot_digest(out) == digest                       # verify() would pass
    assert (out / "sub" / "tokenizer.json").read_text() == "{}"  # nested layout preserved
    assert (out / "model.safetensors").read_bytes() == b"\x00\x01weights"


def test_errors_on_missing_bucket_and_absent_snapshot(tmp_path) -> None:
    s3 = FakeS3()
    with pytest.raises(ModelStoreError):
        ObjectStore(bucket="", client=s3).upload_snapshot("r", _model_dir(tmp_path))
    with pytest.raises(ModelStoreError):  # nothing uploaded at that digest
        ObjectStore(bucket="b", client=s3).download_snapshot("r", "sha256:" + "0" * 64, tmp_path / "o")


def test_upload_model_folder_oci_backend_routes_to_the_object_store(tmp_path, monkeypatch) -> None:
    from epago.model import objectstore, store

    src = _model_dir(tmp_path)
    prebuilt = ObjectStore(bucket="epago-sn", client=FakeS3())
    monkeypatch.setattr(objectstore, "ObjectStore", lambda *a, **k: prebuilt)

    ref = store.upload_model_folder(src, "org/EPAGO-DR-4B-y", backend="oci")
    assert ref.backend == "oci"
    assert ref.repo == "org/EPAGO-DR-4B-y"
    assert ref.digest == snapshot_digest(src)


def test_upload_model_folder_rejects_unknown_backend(tmp_path) -> None:
    with pytest.raises(ModelStoreError):
        store_unknown()


def store_unknown():
    from epago.model.store import upload_model_folder

    upload_model_folder(Path("/x"), "r", backend="nope")


def test_corpus_publish_and_sync_round_trip(tmp_path) -> None:
    """publish_corpus stores under {repo}/{digest}/corpus.db; sync_corpus rebuilds it."""
    from epago.environment.sync import CorpusIntegrityError, corpus_digest, publish_corpus, sync_corpus

    db = tmp_path / "corpus.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\x01\x02\x03pages")
    store = ObjectStore(bucket="epago-sn", client=FakeS3())

    digest = publish_corpus(db, "epago/corpus", store=store)
    assert digest == corpus_digest(db)
    key = f"epago/corpus/{digest.removeprefix('sha256:')}/corpus.db"
    assert store.list_keys("epago/corpus/") == [key]

    out = sync_corpus("epago/corpus", digest, tmp_path / "dl", store=store)
    assert out.read_bytes() == db.read_bytes()  # verify_corpus passed

    # A malformed digest is rejected before any network access.
    with pytest.raises(CorpusIntegrityError):
        sync_corpus("epago/corpus", "not-a-digest", tmp_path / "dl2", store=store)


# --- untrusted object keys ----------------------------------------------------
#
# The bucket is shared and miners hold write credentials, so a listing is
# attacker-controlled input, not a path we produced. Joining a key straight onto
# the target directory let an object named to escape it write anywhere the
# validator process could reach — and because escaped files land outside the
# folder snapshot_digest() hashes, digest verification still passed.


@pytest.mark.parametrize(
    "evil_suffix",
    [
        "/etc/cron.d/pwn",                  # doubled separator -> absolute rel
        "../../../../etc/cron.d/pwn",       # classic traversal
        "sub/../../../escape.txt",          # traversal mid-key
        "./../escape.txt",
    ],
)
def test_download_refuses_keys_that_escape_the_target(tmp_path, evil_suffix) -> None:
    src = _model_dir(tmp_path)
    s3 = FakeS3()
    store = ObjectStore(bucket="epago-sn", client=s3)
    digest = store.upload_snapshot("org/model", src)

    prefix = f"org/model/{digest.removeprefix('sha256:')}/"
    s3.put_raw("epago-sn", prefix + evil_suffix)

    out = tmp_path / "out"
    with pytest.raises(ModelStoreError, match="unsafe object key"):
        store.download_snapshot("org/model", digest, out)


def test_escaping_key_writes_nothing_outside_the_target(tmp_path) -> None:
    """The refusal must happen before the first download, not part-way through."""
    src = _model_dir(tmp_path)
    s3 = FakeS3()
    store = ObjectStore(bucket="epago-sn", client=s3)
    digest = store.upload_snapshot("org/model", src)

    victim = tmp_path / "victim.txt"
    victim.write_text("original")
    prefix = f"org/model/{digest.removeprefix('sha256:')}/"
    s3.put_raw("epago-sn", prefix + f"../{victim.name}", b"overwritten")

    with pytest.raises(ModelStoreError):
        store.download_snapshot("org/model", digest, tmp_path / "out")
    assert victim.read_text() == "original"


def test_download_refuses_too_many_objects(tmp_path, monkeypatch) -> None:
    src = _model_dir(tmp_path)
    s3 = FakeS3()
    store = ObjectStore(bucket="epago-sn", client=s3)
    digest = store.upload_snapshot("org/model", src)
    monkeypatch.setattr("epago.model.objectstore.MAX_SNAPSHOT_FILES", 2)

    with pytest.raises(ModelStoreError, match="objects, cap is"):
        store.download_snapshot("org/model", digest, tmp_path / "out")


def test_download_refuses_oversized_snapshot(tmp_path, monkeypatch) -> None:
    src = _model_dir(tmp_path)
    s3 = FakeS3()
    store = ObjectStore(bucket="epago-sn", client=s3)
    digest = store.upload_snapshot("org/model", src)
    monkeypatch.setattr("epago.model.objectstore.MAX_SNAPSHOT_BYTES", 4)

    with pytest.raises(ModelStoreError, match="bytes, cap is"):
        store.download_snapshot("org/model", digest, tmp_path / "out")


def test_safe_keys_still_round_trip(tmp_path) -> None:
    """Nested and dotted-but-harmless names must keep working."""
    from epago.model.objectstore import safe_relative_key

    for ok in ("model.safetensors", "sub/tokenizer.json", "a.b/c..d/e.json"):
        assert safe_relative_key(ok) == ok


def test_an_oversized_upload_is_refused_before_any_bytes_move(tmp_path) -> None:
    """The download path already bounds size, but bounding only there means an
    oversized upload is paid for in full — bandwidth, storage and egress — and
    only then refused. Checking here makes a quota a limit rather than a bill.
    """
    import pytest

    from epago.model import objectstore
    from epago.model.store import ModelStoreError

    folder = tmp_path / "snapshot"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x" * 4096)

    put_calls = []

    class _Store(objectstore.ObjectStore):
        def require_bucket(self):
            return "bucket"

        def put_object(self, key, path):
            put_calls.append(key)

    store = _Store(bucket="bucket", access_key="a", secret_key="b", endpoint="https://e")

    monkey = objectstore.MAX_SNAPSHOT_BYTES
    try:
        objectstore.MAX_SNAPSHOT_BYTES = 100
        with pytest.raises(ModelStoreError, match="cap is 100"):
            store.upload_snapshot("submissions/hk/model", folder)
    finally:
        objectstore.MAX_SNAPSHOT_BYTES = monkey

    assert put_calls == [], "nothing should have been uploaded"
