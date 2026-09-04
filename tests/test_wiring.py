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
