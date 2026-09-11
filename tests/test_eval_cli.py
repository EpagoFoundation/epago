"""The eval server's command line: what it builds before it serves."""

from __future__ import annotations

from dataclasses import replace

from epago.config import load_config


def test_the_probe_pool_is_minted_by_the_generator_release(monkeypatch):
    """A sealed contract names a pool file (POOL1), not a generator. Minting the
    probe pool from it raised, so every /probes call failed with a 500."""
    from epago.eval.cli import probe_task_pool
    from epago.taskgen import generator

    cfg = load_config()
    cfg = replace(cfg, eval=replace(cfg.eval, taskgen_release="POOL1"))
    seen = {}
    monkeypatch.setattr(generator, "generate_tasks", lambda **kw: seen.update(kw) or [])

    probe_task_pool(cfg, corpus=None)

    assert seen["release"] == cfg.eval.generation_release
    assert seen["release"] != "POOL1"
