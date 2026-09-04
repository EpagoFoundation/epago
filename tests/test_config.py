"""Tests for epago.config — the chain.toml loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from epago.config import DEFAULT_CONFIG_PATH, load_config

#: Template contract used for text-manipulation tests (validation, env-path);
#: the default-load test below reads the real shipped default instead.
TEMPLATE_TOML = Path(__file__).parent / "data" / "chain-template.toml"
CHAIN_TOML = TEMPLATE_TOML.read_text()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests must not inherit EPAGO_* overrides from the invoking shell."""
    import os

    for key in list(os.environ):
        if key.startswith("EPAGO_"):
            monkeypatch.delenv(key)
    yield


class TestLoad:
    def test_loads_repo_chain_toml(self):
        cfg = load_config()
        assert cfg.chain.name == "EPAGO-DR-30B"
        assert cfg.chain.network == "finney"
        assert cfg.chain.repo_pattern == "^[^/]+/EPAGO-DR-30B-.+$"
        assert cfg.quorum.theta == pytest.approx(0.51)
        assert cfg.quorum.bootstrap_min_evaluators == 3
        assert cfg.emissions.king_share == pytest.approx(0.90)
        assert cfg.emissions.arena_share == pytest.approx(0.10)
        assert cfg.path == DEFAULT_CONFIG_PATH

    def test_list_fields_become_tuples(self):
        cfg = load_config()
        assert isinstance(cfg.arch.extra_lock_keys, tuple)
        assert "rope_scaling" in cfg.arch.extra_lock_keys

    def test_sections_are_frozen(self):
        cfg = load_config()
        with pytest.raises(AttributeError):
            cfg.chain.netuid = 99  # type: ignore[misc]


class TestEnvOverride:
    def test_bare_env_override(self, monkeypatch):
        monkeypatch.setenv("EPAGO_NETUID", "42")
        assert load_config().chain.netuid == 42

    def test_section_prefixed_override(self, monkeypatch):
        monkeypatch.setenv("EPAGO_QUORUM_THETA", "0.66")
        assert load_config().quorum.theta == pytest.approx(0.66)

    def test_override_preserves_type(self, monkeypatch):
        monkeypatch.setenv("EPAGO_VERDICT_TIMEOUT_BLOCKS", "100")
        v = load_config().quorum.verdict_timeout_blocks
        assert v == 100 and isinstance(v, int)

    def test_chain_toml_path_env(self, tmp_path, monkeypatch):
        alt = tmp_path / "alt.toml"
        alt.write_text(CHAIN_TOML.replace('name         = "EPAGO-DR-4B"', 'name = "EPAGO-ALT"'))
        monkeypatch.setenv("EPAGO_CHAIN_TOML", str(alt))
        cfg = load_config()
        assert cfg.chain.name == "EPAGO-ALT"
        assert cfg.path == alt


class TestValidation:
    def _write(self, tmp_path: Path, old: str, new: str) -> Path:
        assert old in CHAIN_TOML, f"fixture drift: {old!r} not in chain.toml"
        p = tmp_path / "chain.toml"
        p.write_text(CHAIN_TOML.replace(old, new))
        return p

    def test_share_sum_must_be_one(self, tmp_path):
        bad = self._write(tmp_path, "king_share        = 0.90", "king_share        = 0.70")
        with pytest.raises(ValueError, match="sum to 1.0"):
            load_config(bad)

    def test_theta_above_one_rejected(self, tmp_path):
        bad = self._write(tmp_path, "theta                 = 0.51", "theta                 = 1.5")
        with pytest.raises(ValueError, match="theta"):
            load_config(bad)

    def test_theta_zero_rejected(self, tmp_path):
        bad = self._write(tmp_path, "theta                 = 0.51", "theta                 = 0.0")
        with pytest.raises(ValueError, match="theta"):
            load_config(bad)

    def test_theta_of_exactly_one_allowed(self, tmp_path):
        ok = self._write(tmp_path, "theta                 = 0.51", "theta                 = 1.0")
        assert load_config(ok).quorum.theta == 1.0
