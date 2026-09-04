"""Automated fresh private-pool source: shard selection is deterministic under a
beacon seed, and HfSnapshotSource packs a bounded slice into documents without
network/optional deps (the HF listing/download/parquet-read are injected)."""
from __future__ import annotations

from pathlib import Path

from epago.taskgen.ingest import HfSnapshotSource, select_shards
TEMPLATE_TOML = Path(__file__).parent / "data" / "chain-template.toml"


def test_select_shards_deterministic_and_bounded() -> None:
    shards = [f"data/train-{i:04d}.parquet" for i in range(20)]
    a = select_shards(shards, seed=123, k=4)
    b = select_shards(shards, seed=123, k=4)
    assert a == b                              # same seed -> identical slice
    assert len(a) == 4                         # bounded to k
    assert set(a).issubset(shards)             # real shards
    assert select_shards(shards, seed=999, k=4) != a  # seed changes the slice (overwhelmingly)
    # Listing order must not matter — only (manifest set, seed).
    assert select_shards(list(reversed(shards)), seed=123, k=4) == a
    assert select_shards(shards, seed=1, k=0) == []
    assert len(select_shards(shards[:3], seed=1, k=10)) == 3  # k clamps to available


def test_hf_snapshot_source_packs_bounded_docs_from_injected_shards() -> None:
    shards = [f"data/train-{i:04d}.parquet" for i in range(10)]
    # injected seams: list -> the manifest; download -> a marker path; read -> texts.
    texts_by_shard = {
        s: [f"Doc {s} line one.\nbody text here", f"Second doc in {s}."] for s in shards
    }
    downloaded: list[str] = []

    def fake_list(repo, revision):
        return shards

    def fake_download(repo, revision, shard):
        downloaded.append(shard)
        return Path(f"/fake/{shard}")

    def fake_read(path, column):
        shard = str(path).replace("/fake/", "")
        yield from texts_by_shard[shard]

    src = HfSnapshotSource(
        repo="Org/fineweb-edu", revision="abc123", seed=42, max_shards=3,
        list_shards_fn=fake_list, download_shard_fn=fake_download, read_texts_fn=fake_read,
    )
    docs = src.fetch_documents(budget=5)
    assert len(docs) == 5                                   # honours the budget
    assert all(d.category == "ingested" for d in docs)
    assert all(d.doc_id.startswith("ing-") for d in docs)
    assert all(d.url.startswith("hf://Org/fineweb-edu@abc123/") for d in docs)
    # Only downloaded the beacon-selected shards, and stopped once budget was met.
    assert set(downloaded).issubset(set(select_shards(shards, 42, 3)))

    # Determinism: a second identical source yields identical doc ids.
    src2 = HfSnapshotSource(
        repo="Org/fineweb-edu", revision="abc123", seed=42, max_shards=3,
        list_shards_fn=fake_list, download_shard_fn=fake_download, read_texts_fn=fake_read,
    )
    assert [d.doc_id for d in src2.fetch_documents(5)] == [d.doc_id for d in docs]


def test_private_source_config_optional_and_env_overridable(monkeypatch) -> None:
    from epago.config import load_config

    for v in ("EPAGO_PRIVATE_SOURCE_REPO", "EPAGO_PRIVATE_SOURCE_MAX_SHARDS", "EPAGO_REPO"):
        monkeypatch.delenv(v, raising=False)
    cfg = load_config(TEMPLATE_TOML)                       # chain.toml has no [private_source]
    assert cfg.private_source.repo == ""      # disabled by default
    monkeypatch.setenv("EPAGO_PRIVATE_SOURCE_REPO", "HuggingFaceFW/fineweb-edu")
    monkeypatch.setenv("EPAGO_PRIVATE_SOURCE_MAX_SHARDS", "8")
    cfg2 = load_config(TEMPLATE_TOML)
    assert cfg2.private_source.repo == "HuggingFaceFW/fineweb-edu"
    assert cfg2.private_source.max_shards == 8


def test_managed_pool_prefers_configured_fresh_feed_over_manual_ingest() -> None:
    from types import SimpleNamespace

    from epago.config import PrivateSourceSection
    from epago.taskgen.ingest import LocalDirSource
    from epago.validator.wiring import ManagedPrivatePool

    # configured fresh feed → automated HfSnapshotSource, carrying the secret seed
    feed = SimpleNamespace(_cfg=SimpleNamespace(
        private_source=PrivateSourceSection(repo="Org/fineweb-edu", revision="rev", max_shards=3)
    ), _ingest_dir=None)
    src = ManagedPrivatePool._make_source(feed, seed=7)
    assert isinstance(src, HfSnapshotSource)
    assert src.repo == "Org/fineweb-edu" and src.seed == 7 and src.max_shards == 3
    # no feed but a manual ingest dir → deprecated LocalDirSource override
    manual = SimpleNamespace(_cfg=SimpleNamespace(private_source=PrivateSourceSection()), _ingest_dir="/d")
    assert isinstance(ManagedPrivatePool._make_source(manual, 1), LocalDirSource)
    # neither → None → corpus fallback
    none_ = SimpleNamespace(_cfg=SimpleNamespace(private_source=PrivateSourceSection()), _ingest_dir=None)
    assert ManagedPrivatePool._make_source(none_, 1) is None


def test_hf_snapshot_source_skips_blank_texts() -> None:
    def fake_list(repo, revision):
        return ["a.parquet"]

    def fake_download(repo, revision, shard):
        return Path("/fake/a.parquet")

    def fake_read(path, column):
        yield "   "          # blank -> skipped
        yield "real content"

    src = HfSnapshotSource(
        repo="r", revision="v", seed=1, list_shards_fn=fake_list,
        download_shard_fn=fake_download, read_texts_fn=fake_read,
    )
    docs = src.fetch_documents(budget=10)
    assert len(docs) == 1 and docs[0].text == "real content"
