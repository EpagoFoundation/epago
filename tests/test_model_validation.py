"""Tests for epago.model.validation — intake gates on fake model folders.

These build tiny fake king/challenger snapshot directories (a config.json plus
dummy safetensors bytes); intake validation never parses tensor contents, so
placeholder bytes exercise exactly the code under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epago.config import load_config
from epago.model.validation import (
    exact_copy_of_king,
    validate_challenger_folder,
    validate_repo_name,
)

TEMPLATE_TOML = Path(__file__).parent / "data" / "chain-template.toml"

KING_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "vocab_size": 151_936,
    "hidden_size": 2_560,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9_728,
    "tie_word_embeddings": True,
    "rope_theta": 5_000_000,
    "rope_scaling": None,
    "max_position_embeddings": 262_144,
}

KING_WEIGHTS = bytes(range(100))  # 100 dummy bytes -> size cap = int(100 * 1.05) = 105


@pytest.fixture(scope="module")
def cfg():
    return load_config(TEMPLATE_TOML)


def write_model_dir(
    root: Path,
    name: str,
    config: dict | None = None,
    weights: bytes = KING_WEIGHTS,
    extra_files: dict[str, bytes] | None = None,
    with_weights: bool = True,
) -> Path:
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(KING_CONFIG if config is None else config))
    if with_weights:
        (d / "model.safetensors").write_bytes(weights)
    for fname, data in (extra_files or {}).items():
        (d / fname).write_bytes(data)
    return d


@pytest.fixture()
def king_dir(tmp_path):
    return write_model_dir(tmp_path, "king")


def codes(failures) -> set[str]:
    return {f.code for f in failures}


class TestCleanChallenger:
    def test_identical_structure_passes(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(tmp_path, "chall", weights=b"\xff" * 100)
        assert validate_challenger_folder(chall, king_dir, cfg) == []

    def test_allowed_extra_files_pass(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(
            tmp_path,
            "chall",
            weights=b"\xff" * 100,
            extra_files={
                "tokenizer.json": b"{}",
                "special_tokens_map.json": b"{}",
                "merges.txt": b"",
                "chat_template.jinja": b"",
            },
        )
        assert validate_challenger_folder(chall, king_dir, cfg) == []


class TestConfigLock:
    def test_generic_key_mismatch_detected(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(tmp_path, "chall", config={**KING_CONFIG, "hidden_size": 4096})
        failures = validate_challenger_folder(chall, king_dir, cfg)
        assert codes(failures) == {"config_lock"}
        assert any("hidden_size" in f.detail for f in failures)

    def test_extra_lock_key_mismatch_detected(self, tmp_path, king_dir, cfg):
        # rope_scaling is locked via arch.extra_lock_keys, not the generic set
        assert "rope_scaling" in cfg.arch.extra_lock_keys
        chall = write_model_dir(
            tmp_path, "chall", config={**KING_CONFIG, "rope_scaling": {"factor": 4.0}}
        )
        assert "config_lock" in codes(validate_challenger_folder(chall, king_dir, cfg))

    def test_missing_lock_key_detected(self, tmp_path, king_dir, cfg):
        conf = dict(KING_CONFIG)
        del conf["vocab_size"]
        chall = write_model_dir(tmp_path, "chall", config=conf)
        assert "config_lock" in codes(validate_challenger_folder(chall, king_dir, cfg))

    def test_auto_map_rejected(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(
            tmp_path, "chall", config={**KING_CONFIG, "auto_map": {"AutoModel": "x.Y"}}
        )
        assert "auto_map" in codes(validate_challenger_folder(chall, king_dir, cfg))

    def test_missing_config_json(self, tmp_path, king_dir, cfg):
        chall = tmp_path / "chall"
        chall.mkdir()
        (chall / "model.safetensors").write_bytes(b"\x00" * 10)
        assert "config_missing" in codes(validate_challenger_folder(chall, king_dir, cfg))


class TestFileHygiene:
    def test_python_file_rejected(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(
            tmp_path, "chall", extra_files={"modeling_custom.py": b"import os"}
        )
        assert "python_file" in codes(validate_challenger_folder(chall, king_dir, cfg))

    @pytest.mark.parametrize("fname", ["pytorch_model.bin", "model.pt", "state.pkl", "last.ckpt"])
    def test_pickle_weights_rejected(self, tmp_path, king_dir, cfg, fname):
        chall = write_model_dir(tmp_path, "chall", extra_files={fname: b"\x80\x04"})
        assert "pickle_weights" in codes(validate_challenger_folder(chall, king_dir, cfg))

    def test_unexpected_file_type_rejected(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(tmp_path, "chall", extra_files={"banner.png": b"\x89PNG"})
        assert "unexpected_file" in codes(validate_challenger_folder(chall, king_dir, cfg))

    def test_missing_canonical_layout_rejected(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(tmp_path, "chall", with_weights=False)
        assert "layout" in codes(validate_challenger_folder(chall, king_dir, cfg))

    def test_sharded_index_layout_accepted(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(
            tmp_path,
            "chall",
            with_weights=False,
            extra_files={
                "model.safetensors.index.json": b"{}",
                "model-00001-of-00001.safetensors": b"\xff" * 100,
            },
        )
        assert validate_challenger_folder(chall, king_dir, cfg) == []


class TestSizeCap:
    def test_over_cap_rejected(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(tmp_path, "chall", weights=b"\xff" * 106)  # cap is 105
        failures = validate_challenger_folder(chall, king_dir, cfg)
        assert "size_cap" in codes(failures)

    def test_at_cap_accepted(self, tmp_path, king_dir, cfg):
        chall = write_model_dir(tmp_path, "chall", weights=b"\xff" * 105)
        assert validate_challenger_folder(chall, king_dir, cfg) == []


class TestExactCopy:
    def test_identical_shards_detected(self, tmp_path, king_dir):
        chall = write_model_dir(tmp_path, "chall", weights=KING_WEIGHTS)
        assert exact_copy_of_king(chall, king_dir) is True

    def test_single_flipped_byte_not_exact_copy(self, tmp_path, king_dir):
        tweaked = bytes([KING_WEIGHTS[0] ^ 1]) + KING_WEIGHTS[1:]
        chall = write_model_dir(tmp_path, "chall", weights=tweaked)
        assert exact_copy_of_king(chall, king_dir) is False

    def test_same_bytes_different_shard_name_not_exact_copy(self, tmp_path, king_dir):
        chall = write_model_dir(
            tmp_path,
            "chall",
            with_weights=False,
            extra_files={"model-00001-of-00001.safetensors": KING_WEIGHTS},
        )
        assert exact_copy_of_king(chall, king_dir) is False


class TestRepoName:
    HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"

    def test_valid_repo_with_prefix(self, cfg):
        repo = f"myorg/EPAGO-DR-4B-{self.HOTKEY[:8]}-run1"
        assert validate_repo_name(repo, self.HOTKEY, cfg) is None

    def test_prefix_match_is_case_insensitive(self, cfg):
        repo = f"myorg/EPAGO-DR-4B-{self.HOTKEY[:8].upper()}-run1"
        assert validate_repo_name(repo, self.HOTKEY, cfg) is None

    def test_pattern_violation(self, cfg):
        fail = validate_repo_name("myorg/some-other-model", self.HOTKEY, cfg)
        assert fail is not None and fail.code == "repo_pattern"

    def test_missing_slash_rejected(self, cfg):
        fail = validate_repo_name("EPAGO-DR-4B-standalone", self.HOTKEY, cfg)
        assert fail is not None and fail.code == "repo_pattern"

    def test_missing_hotkey_prefix(self, cfg):
        fail = validate_repo_name("myorg/EPAGO-DR-4B-run1", self.HOTKEY, cfg)
        assert fail is not None and fail.code == "hotkey_prefix"
