"""The EPAGO-DR-30B generation contract against the real base-model config.

The fixture is the actual ``config.json`` of Alibaba-NLP/Tongyi-DeepResearch-
30B-A3B at the pinned revision (vendored, not fetched). What these tests pin
down: the intake config-lock, written for a 4B dense model, must hold for an
MoE — and specifically must catch the MoE-shaped smuggling the generic keys
never see, like growing the expert count or the per-token expert budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epago.config import load_config
from epago.model.validation import validate_challenger_folder, validate_repo_name

FIXTURE = Path(__file__).parent / "data" / "tongyi_deepresearch_30b_a3b_config.json"
GENERATION = Path(__file__).parent.parent / "chains" / "EPAGO-DR-30B.toml"

HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
GOOD_REPO = f"team/EPAGO-DR-30B-{HOTKEY[:8].lower()}-run1"


@pytest.fixture()
def cfg():
    return load_config(GENERATION)


@pytest.fixture()
def base_config() -> dict:
    return json.loads(FIXTURE.read_text())


def _model_dir(path: Path, config: dict, weights: bytes) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, sort_keys=True))
    (path / "model.safetensors").write_bytes(weights)
    return path


@pytest.fixture()
def king_dir(tmp_path, base_config) -> Path:
    return _model_dir(tmp_path / "king", base_config, b"K" * 2048)


def test_generation_contract_loads_with_real_pins(cfg):
    assert cfg.chain.name == "EPAGO-DR-30B"
    assert cfg.seed.seed_digest == "hf:4b0ac5767427a55d08a254f0367e2934976598e0"
    assert cfg.eval.taskgen_release == "SCI4"
    # The corpus is pinned for real: an all-zeros digest means nothing is
    # committed to and every validator would verify against a different file.
    assert cfg.eval.corpus_digest.startswith("sha256:")
    assert set(cfg.eval.corpus_digest.removeprefix("sha256:")) != {"0"}
    # The MoE structure is locked; these are the keys a smuggler would touch.
    for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size"):
        assert key in cfg.arch.extra_lock_keys


def test_fixture_is_the_moe_config(base_config):
    """Guard the fixture itself: if someone swaps it for a dense config, every
    other test here silently stops testing what it claims to."""
    assert base_config["architectures"] == ["Qwen3MoeForCausalLM"]
    assert base_config["num_experts"] == 128
    assert base_config["num_experts_per_tok"] == 8


def test_identical_moe_config_passes_the_lock(tmp_path, cfg, king_dir, base_config):
    challenger = _model_dir(tmp_path / "chal", base_config, b"C" * 2048)
    assert validate_challenger_folder(challenger, king_dir, cfg) == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_experts", 256),            # double the capacity
        ("num_experts_per_tok", 16),     # double the per-token compute
        ("moe_intermediate_size", 1536), # fatter experts
        ("decoder_sparse_step", 2),      # change which layers are sparse
        ("hidden_size", 4096),           # generic key still covered
        ("num_hidden_layers", 96),
    ],
)
def test_moe_smuggling_fails_the_lock(tmp_path, cfg, king_dir, base_config, key, value):
    tampered = dict(base_config)
    tampered[key] = value
    challenger = _model_dir(tmp_path / "chal", tampered, b"C" * 2048)
    failures = validate_challenger_folder(challenger, king_dir, cfg)
    assert any(f.code == "config_lock" and key in f.detail for f in failures), (
        f"tampered {key} was not caught"
    )


def test_auto_map_still_forbidden(tmp_path, cfg, king_dir, base_config):
    tampered = dict(base_config)
    tampered["auto_map"] = {"AutoModelForCausalLM": "evil.Model"}
    challenger = _model_dir(tmp_path / "chal", tampered, b"C" * 2048)
    failures = validate_challenger_folder(challenger, king_dir, cfg)
    assert any(f.code == "auto_map" for f in failures)


def test_30b_repo_pattern(cfg):
    assert validate_repo_name(GOOD_REPO, HOTKEY, cfg) is None
    # The 4B generation's repos do not pass the 30B pattern.
    old = f"team/EPAGO-DR-4B-{HOTKEY[:8].lower()}-run1"
    failure = validate_repo_name(old, HOTKEY, cfg)
    assert failure is not None and failure.code == "repo_pattern"


def test_size_cap_scales_from_the_king(tmp_path, cfg, king_dir, base_config):
    """1.05x is relative to the reigning king's bytes, so it needs no edits for
    a bigger base — but confirm it actually trips."""
    challenger = _model_dir(tmp_path / "chal", base_config, b"C" * 4096)  # ~2x king
    failures = validate_challenger_folder(challenger, king_dir, cfg)
    assert any(f.code == "size_cap" for f in failures)


# --- the full loop under the 30B contract -------------------------------------


def test_full_round_runs_under_the_30b_generation(tmp_path, base_config):
    """Intake, MoE config lock, a competition round, verdicts, and coronation —
    end to end under chains/EPAGO-DR-30B.toml with the real base config."""
    from test_validator import (
        add_challenger,
        make_harness,
        published_verdicts,
        settle,
    )

    h = make_harness(
        tmp_path,
        chain_toml=str(GENERATION),
        model_config=base_config,
    )
    assert h.cfg.chain.name == "EPAGO-DR-30B"

    digest, repo, _ = add_challenger(
        h, "alice", "hk-alice", "ck-alice-01", uid=2, digest_char="a"
    )
    assert "EPAGO-DR-30B" in repo  # repo naming follows the generation

    settle(h)

    # The MoE challenger cleared the lock, dueled, won, and was crowned.
    assert h.state.king.ref.digest == digest
    verdicts = published_verdicts(h.chain)
    assert len(verdicts) == 1 and "|A|" in verdicts[0]


def test_dense_4b_checkpoint_is_rejected_by_the_30b_lock(tmp_path, base_config):
    """A challenger shaped like the old 4B generation cannot enter the 30B one:
    every structural key disagrees, starting with the architecture itself."""
    from test_validator import LOCK_CONFIG, add_challenger, make_harness, settle
    from epago.core.types import SubmissionStatus

    h = make_harness(tmp_path, chain_toml=str(GENERATION), model_config=base_config)
    # Force the challenger dir to carry the DENSE config against the MoE king.
    digest, _, _ = add_challenger(h, "bob", "hk-bob", "ck-bob-001", uid=3, digest_char="b")
    (h.dirs[digest] / "config.json").write_text(json.dumps(LOCK_CONFIG))

    settle(h)

    assert h.state.statuses[digest] == SubmissionStatus.FAILED_INTAKE.value
    assert h.state.king.ref.digest == h.cfg.seed.seed_digest  # genesis keeps the throne
