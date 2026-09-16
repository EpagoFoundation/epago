"""Production wiring seam tests.

Everything here exercises :mod:`epago.validator.wiring` — the one module where
the live subsystems meet — against the fixture corpus, so an interface drift
between service, eval, and taskgen fails in CI rather than on mainnet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from epago import constants
from epago.chain.client import MockChainClient, NeuronView
from epago.config import load_config
from epago.core.types import Task
from epago.environment.fixtures import build_fixture_corpus
from epago.validator.wiring import ManagedPrivatePool, build_production_deps
TEMPLATE_TOML = Path(__file__).parent / "data" / "chain-template.toml"


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("corpus") / "corpus.db"
    build_fixture_corpus(path, n_docs=240, seed=7)
    return path


@pytest.fixture()
def cfg():
    return load_config(TEMPLATE_TOML)


def _small_pool(monkeypatch):
    monkeypatch.setattr(constants, "N_PRIV_TASKS", 8)


def test_managed_pool_bootstraps_samples_and_rotates(tmp_path, corpus_path, cfg, monkeypatch):
    _small_pool(monkeypatch)
    from epago.environment.corpus import SqliteCorpus

    corpus = SqliteCorpus(corpus_path)
    pool = ManagedPrivatePool(tmp_path, corpus, cfg)

    assert pool.epoch == 1
    active_digest = pool.digest
    sample = pool.sample(4, seed=123)
    assert len(sample) == 4 and all(isinstance(t, Task) for t in sample)
    assert pool.sample(4, seed=123) == sample  # same seed, same draw

    # Before rotating, the commitment names the pool that is grading now.
    version, epoch, digest16 = pool.commitment().split("|")
    assert version == "ep1" and int(epoch) == 1
    assert active_digest.removeprefix("sha256:").startswith(digest16)

    payload = pool.rotate(current_block=50_000)
    assert payload is not None
    # Rotation commits the INCOMING pool, before it grades a single duel.
    # Committing the outgoing one chain-stamped a digest ~6 days after every
    # verdict it had already produced, which proved nothing about the tasks
    # while they were secret.
    version, epoch, digest16 = payload.split("|")
    assert version == "ep1" and int(epoch) == 2
    assert pool.digest.removeprefix("sha256:").startswith(digest16)

    # Delayed transparency: the published file hashes to the committed digest.
    published = list((tmp_path / "publications").glob("*.json"))
    assert len(published) == 1
    body = published[0].read_bytes()
    assert hashlib.sha256(body).hexdigest() == active_digest.removeprefix("sha256:")
    tasks = json.loads(body)["tasks"]
    assert tasks and all("answer" in t for t in tasks)

    assert pool.epoch == 2
    assert pool.digest != active_digest


def test_managed_pool_persists_across_restarts(tmp_path, corpus_path, cfg, monkeypatch):
    _small_pool(monkeypatch)
    from epago.environment.corpus import SqliteCorpus

    corpus = SqliteCorpus(corpus_path)
    first = ManagedPrivatePool(tmp_path, corpus, cfg)
    digest = first.digest
    again = ManagedPrivatePool(tmp_path, corpus, cfg)
    assert again.epoch == first.epoch
    assert again.digest == digest


def _audited_rows(tmp_path, corpus_path, cfg) -> list[dict]:
    """Rows for an audited pool file, minted over the fixture corpus."""
    from epago.environment.corpus import SqliteCorpus
    from epago.taskgen.private_pool import _task_to_dict

    minted = ManagedPrivatePool(tmp_path / "mint", SqliteCorpus(corpus_path), cfg)
    return [_task_to_dict(t) for t in minted._pool.tasks]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_managed_pool_refuses_a_missing_audited_dir(tmp_path, corpus_path, cfg, monkeypatch):
    _small_pool(monkeypatch)
    from epago.environment.corpus import SqliteCorpus

    monkeypatch.setenv("EPAGO_AUDITED_POOL_DIR", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="not a directory"):
        ManagedPrivatePool(tmp_path / "state", SqliteCorpus(corpus_path), cfg)
    assert not (tmp_path / "state" / "private_pool" / "pool_state.json").exists()


def test_managed_pool_refuses_an_audited_dir_with_no_unused_file(
    tmp_path, corpus_path, cfg, monkeypatch
):
    _small_pool(monkeypatch)
    from epago.environment.corpus import SqliteCorpus

    audited = tmp_path / "audited"
    audited.mkdir()
    (audited / "pool-epoch001.jsonl.used").write_text("")
    monkeypatch.setenv("EPAGO_AUDITED_POOL_DIR", str(audited))
    with pytest.raises(RuntimeError, match="no unused audited pool"):
        ManagedPrivatePool(tmp_path / "state", SqliteCorpus(corpus_path), cfg)


def test_managed_pool_uses_an_audited_file_once(tmp_path, corpus_path, cfg, monkeypatch):
    _small_pool(monkeypatch)
    from epago.environment.corpus import SqliteCorpus

    monkeypatch.delenv("EPAGO_AUDITED_POOL_DIR", raising=False)
    rows = _audited_rows(tmp_path, corpus_path, cfg)
    audited = tmp_path / "audited"
    audited.mkdir()
    _write_rows(audited / "pool-epoch001.jsonl", rows)
    monkeypatch.setenv("EPAGO_AUDITED_POOL_DIR", str(audited))

    pool = ManagedPrivatePool(tmp_path / "state", SqliteCorpus(corpus_path), cfg)
    assert sorted(t.task_id for t in pool._pool.tasks) == sorted(r["task_id"] for r in rows)
    assert (audited / "pool-epoch001.jsonl.used").exists()


def test_managed_pool_refuses_audited_tasks_citing_papers_the_corpus_lacks(
    tmp_path, corpus_path, cfg, monkeypatch
):
    _small_pool(monkeypatch)
    from epago.environment.corpus import SqliteCorpus

    monkeypatch.delenv("EPAGO_AUDITED_POOL_DIR", raising=False)
    rows = _audited_rows(tmp_path, corpus_path, cfg)
    rows[0]["evidence_doc_ids"] = ["ing-0000000000000000"]
    audited = tmp_path / "audited"
    audited.mkdir()
    _write_rows(audited / "pool-epoch001.jsonl", rows)
    monkeypatch.setenv("EPAGO_AUDITED_POOL_DIR", str(audited))

    with pytest.raises(ValueError, match="papers the corpus does not have"):
        ManagedPrivatePool(tmp_path / "state", SqliteCorpus(corpus_path), cfg)
    assert (audited / "pool-epoch001.jsonl").exists()  # refused, so not marked used


def test_managed_pool_refuses_a_saved_pool_citing_papers_the_corpus_lacks(
    tmp_path, corpus_path, cfg, monkeypatch
):
    _small_pool(monkeypatch)
    from dataclasses import replace

    from epago.environment.corpus import SqliteCorpus
    from epago.taskgen.private_pool import PrivatePool

    monkeypatch.delenv("EPAGO_AUDITED_POOL_DIR", raising=False)
    corpus = SqliteCorpus(corpus_path)
    saved = ManagedPrivatePool(tmp_path, corpus, cfg)._pool
    first = replace(saved.tasks[0], evidence_doc_ids=("ing-0000000000000000",))
    PrivatePool(
        epoch=saved.epoch,
        created_block=saved.created_block,
        tasks=(first,) + saved.tasks[1:],
        storage_path=saved.storage_path,
    ).save()

    with pytest.raises(ValueError, match="saved private pool epoch 1"):
        ManagedPrivatePool(tmp_path, corpus, cfg)


def test_managed_pool_drops_feed_tasks_whose_papers_are_not_in_the_corpus(
    tmp_path, corpus_path, cfg, monkeypatch
):
    _small_pool(monkeypatch)
    from dataclasses import replace

    import epago.taskgen.ingest as ingest
    from epago.environment.corpus import SqliteCorpus

    monkeypatch.delenv("EPAGO_AUDITED_POOL_DIR", raising=False)
    corpus = SqliteCorpus(corpus_path)
    minted = ManagedPrivatePool(tmp_path / "mint", corpus, cfg)._pool.tasks
    feed_tasks = [
        replace(t, task_id=f"feed-{i}", evidence_doc_ids=("ing-0000000000000000",))
        for i, t in enumerate(minted)
    ]
    monkeypatch.setattr(ManagedPrivatePool, "_make_source", lambda self, seed: object())
    monkeypatch.setattr(ingest, "build_private_tasks", lambda *args, **kwargs: feed_tasks)

    pool = ManagedPrivatePool(tmp_path / "state", corpus, cfg)
    ids = {t.task_id for t in pool._pool.tasks}
    assert ids and not any(i.startswith("feed-") for i in ids)


def test_build_production_deps_composes_without_gpu_extras(
    tmp_path, corpus_path, cfg, monkeypatch
):
    """Composition must not require torch/vllm — backends are lazy."""
    _small_pool(monkeypatch)
    chain = MockChainClient()
    chain.add_neuron(
        NeuronView(uid=0, hotkey="v0", coldkey="c0", stake=10.0, validator_permit=True)
    )
    deps = build_production_deps(
        cfg=cfg,
        chain=chain,
        state_dir=tmp_path / "state",
        corpus_path=corpus_path,
        cache_dir=tmp_path / "cache",
        wallet_hotkey="v0",
    )
    assert deps.clock() == chain.current_block()
    assert deps.private_pool.epoch == 1
    tasks = deps.generate_tasks(
        seed=42,
        release=cfg.eval.taskgen_release,
        corpus=deps.corpus,
        n=8,
        king_probe=None,
    )
    assert len(tasks) == 8
    assert deps.task_ids_digest(tasks).startswith("sha256:")
    session = deps.env.tools_for_task(tasks[0])
    assert isinstance(session.search("city"), str)


def test_corpus_digest_placeholder_skips_verification(tmp_path, corpus_path, cfg):
    # chain.toml ships the genesis placeholder digest; wiring must not reject
    # the corpus before seed_genesis pins the real one.
    assert set(cfg.eval.corpus_digest.removeprefix("sha256:")) == {"0"}
    chain = MockChainClient()
    build_production_deps(
        cfg=cfg,
        chain=chain,
        state_dir=tmp_path / "state",
        corpus_path=corpus_path,
        cache_dir=tmp_path / "cache",
        wallet_hotkey="v0",
    )
