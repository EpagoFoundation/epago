"""Miner tooling tests: preflight mirrors intake, submit round-trips e1,
king resolution scans chain state, and the CLIs at least present help.

No torch/vllm/bittensor: model dirs are tiny fakes (small config.json plus
dummy .safetensors bytes) and the chain is :class:`MockChainClient`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from epago.chain.client import MockChainClient, NeuronView
from epago.config import load_config
from epago import constants
from epago.core.reveal import build_reveal, build_verdict, parse_reveal
from epago.core.types import ModelRef, Verdict, VerdictDecision
from epago.miner import workflow
from epago.miner.workflow import NoKingError
TEMPLATE_TOML = Path(__file__).parent / "data" / "chain-template.toml"

AUTHOR_HK = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
GOOD_REPO = f"miner1/EPAGO-DR-4B-{AUTHOR_HK[:8].lower()}-run1"
HOTKEY = "5" + "H" * 47

_BASE_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "vocab_size": 1024,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "intermediate_size": 128,
    "model_type": "qwen3",
    "tie_word_embeddings": True,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "max_position_embeddings": 4096,
}


@pytest.fixture()
def cfg():
    return load_config(TEMPLATE_TOML)


def make_model_dir(
    path: Path,
    *,
    weight_bytes: bytes = b"king-weights-0123456789",
    config_overrides: dict | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = dict(_BASE_CONFIG)
    config.update(config_overrides or {})
    (path / "config.json").write_text(json.dumps(config, sort_keys=True))
    (path / "model.safetensors").write_bytes(weight_bytes)
    (path / "tokenizer_config.json").write_text("{}")
    return path


@pytest.fixture()
def king_dir(tmp_path: Path) -> Path:
    return make_model_dir(tmp_path / "king")


@pytest.fixture()
def challenger_dir(tmp_path: Path) -> Path:
    # Same locked config, different weights: a legitimate challenger.
    return make_model_dir(tmp_path / "challenger", weight_bytes=b"trained-weights-xyz-9876")


# --- preflight ----------------------------------------------------------------


def test_preflight_clean_challenger_passes(challenger_dir, king_dir, cfg):
    assert workflow.preflight(challenger_dir, king_dir, GOOD_REPO, AUTHOR_HK, cfg) == []


def test_preflight_catches_bad_repo_name(challenger_dir, king_dir, cfg):
    problems = workflow.preflight(challenger_dir, king_dir, "miner1/other-model", AUTHOR_HK, cfg)
    assert any(p.startswith("repo_pattern") for p in problems)

    no_prefix = "miner1/EPAGO-DR-4B-anonymous"
    problems = workflow.preflight(challenger_dir, king_dir, no_prefix, AUTHOR_HK, cfg)
    assert any(p.startswith("hotkey_prefix") for p in problems)


def test_preflight_catches_config_lock_violation(tmp_path, king_dir, cfg):
    hacked = make_model_dir(
        tmp_path / "hacked",
        weight_bytes=b"different-weights",
        config_overrides={"hidden_size": 4096},
    )
    problems = workflow.preflight(hacked, king_dir, GOOD_REPO, AUTHOR_HK, cfg)
    assert any(p.startswith("config_lock") and "hidden_size" in p for p in problems)


def test_preflight_catches_oversized_weights(tmp_path, king_dir, cfg):
    # King weights are ~23 bytes; 2 KiB blows the 1.05x size cap.
    bloated = make_model_dir(tmp_path / "bloated", weight_bytes=b"x" * 2048)
    problems = workflow.preflight(bloated, king_dir, GOOD_REPO, AUTHOR_HK, cfg)
    assert any(p.startswith("size_cap") for p in problems)


def test_preflight_catches_exact_copy_of_king(tmp_path, king_dir, cfg):
    copy_dir = workflow.prepare_challenger(king_dir, tmp_path / "copy")
    problems = workflow.preflight(copy_dir, king_dir, GOOD_REPO, AUTHOR_HK, cfg)
    assert any(p.startswith("exact_copy") for p in problems)


def test_preflight_catches_forbidden_files(tmp_path, king_dir, cfg):
    bad = make_model_dir(tmp_path / "bad", weight_bytes=b"other-weights")
    (bad / "backdoor.py").write_text("print('hi')")
    problems = workflow.preflight(bad, king_dir, GOOD_REPO, AUTHOR_HK, cfg)
    assert any(p.startswith("python_file") for p in problems)


# --- prepare_challenger ---------------------------------------------------------


def test_prepare_challenger_copies_snapshot_without_markers(tmp_path, king_dir):
    (king_dir / ".epago_verified").touch()
    out = workflow.prepare_challenger(king_dir, tmp_path / "out")
    assert (out / "config.json").read_text() == (king_dir / "config.json").read_text()
    assert (out / "model.safetensors").read_bytes() == (king_dir / "model.safetensors").read_bytes()
    assert not (out / ".epago_verified").exists()


def test_prepare_challenger_rejects_non_snapshot(tmp_path):
    empty = tmp_path / "not-a-model"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        workflow.prepare_challenger(empty, tmp_path / "out")


# --- submit ---------------------------------------------------------------------


def test_submit_builds_parseable_e2_payload():
    chain = MockChainClient()
    chain.identity_hotkey = HOTKEY
    king_digest = "sha256:" + "ab" * 32
    challenger = ModelRef(repo=GOOD_REPO, digest="hf:" + "cd" * 20)

    payload = workflow.submit(chain, king_digest, challenger, HOTKEY)
    assert payload == build_reveal(king_digest, challenger)
    assert payload.startswith("e2|")
    # Authorship is not on the wire; the chain attests who signed.
    assert HOTKEY not in payload

    chain.advance(constants.BLOCKS_UNTIL_REVEAL)
    reveals = chain.read_revealed_submissions(0)
    assert len(reveals) == 1
    reveal = reveals[0]
    assert reveal.king_digest == king_digest
    assert reveal.challenger == challenger
    assert reveal.author_hotkey == HOTKEY

    # Direct wire round-trip of the exact payload the chain stored.
    parsed = parse_reveal(
        payload,
        reveal_block=reveal.reveal_block,
        block_hash_at_reveal=reveal.block_hash_at_reveal,
        author_hotkey=HOTKEY,
    )
    assert parsed == reveal


def test_author_cannot_be_spoofed_by_the_submitter():
    """The e1 hole: a self-declared author let anyone submit under a rival's
    identity, and intake charges cooldowns and pays emission by author."""
    chain = MockChainClient()
    king_digest = "sha256:" + "ab" * 32
    challenger = ModelRef(repo=GOOD_REPO, digest="hf:" + "cd" * 20)

    payload = workflow.submit(chain, king_digest, challenger, hotkey=HOTKEY)
    # The attacker publishes the identical payload from its own hotkey.
    chain.publish_reveal_as("attacker-hotkey", payload)
    chain.advance(constants.BLOCKS_UNTIL_REVEAL)

    authors = {r.author_hotkey for r in chain.read_revealed_submissions(0)}
    assert authors == {chain.identity_hotkey, "attacker-hotkey"}
    # Each reveal is attributed to whoever signed it, never to a payload field.
    assert HOTKEY not in payload


# --- resolve_king_from_chain -----------------------------------------------------


def _accept_verdict(digest: str, block: int) -> Verdict:
    return Verdict(
        challenger_digest=digest,
        decision=VerdictDecision.ACCEPT,
        lcb_pub_e6=30_000,
        mu_priv_e6=20_000,
        delta_e6=25_000,
        round=1,
        private_pool_epoch=1,
        audit_digest="ab" * 8,
        block=block,
    )


def test_resolve_king_with_no_king_yet_raises_clear_error(cfg):
    # Default chain.toml ships an all-zeros placeholder seed digest.
    chain = MockChainClient()
    with pytest.raises(NoKingError) as excinfo:
        workflow.resolve_king_from_chain(chain, cfg)
    message = str(excinfo.value)
    assert "no coronation" in message
    assert "seed_genesis" in message


def test_resolve_king_falls_back_to_pinned_seed(cfg):
    pinned = dataclasses.replace(
        cfg, seed=dataclasses.replace(cfg.seed, seed_digest="hf:" + "a1" * 20)
    )
    chain = MockChainClient()
    king = workflow.resolve_king_from_chain(chain, pinned)
    assert king == ModelRef(repo=cfg.chain.seed_repo, digest="hf:" + "a1" * 20)


def test_resolve_king_from_accepted_verdict(cfg):
    chain = MockChainClient()
    chain.add_neuron(
        NeuronView(uid=0, hotkey="validator-0", coldkey="5Val", stake=1000.0, validator_permit=True)
    )
    challenger = ModelRef(repo=GOOD_REPO, digest="sha256:" + "ee" * 32)
    chain.inject_reveal(HOTKEY, build_reveal("sha256:" + "ab" * 32, challenger))
    chain.advance(5)
    chain.publish_commitment_as("validator-0", build_verdict(_accept_verdict(challenger.digest, 0)))

    king = workflow.resolve_king_from_chain(chain, cfg)
    assert king == challenger


# --- public state helpers ---------------------------------------------------------


def test_find_hotkey_entries_scans_nested_state():
    state = {
        "submissions": [
            {"author_hotkey": HOTKEY, "status": "accepted"},
            {"author_hotkey": "someone-else", "status": "queued"},
        ],
        "king": {"author_hotkey": HOTKEY},
    }
    entries = workflow.find_hotkey_entries(state, HOTKEY)
    assert {e.get("status", "king") for e in entries} == {"accepted", "king"}


# --- CLI smoke ---------------------------------------------------------------------


def test_miner_cli_help_smoke():
    from neurons.miner import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("prepare", "preflight", "submit", "status"):
        assert command in result.output


def test_miner_submit_dry_run_prints_payload():
    from neurons.miner import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "submit",
            "--repo", GOOD_REPO,
            "--digest", "hf:" + "cd" * 20,
            "--king-digest", "sha256:" + "ab" * 32,
            "--hotkey", HOTKEY,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    parsed = parse_reveal(
        result.output.strip(),
        reveal_block=1,
        block_hash_at_reveal="0xdead",
        author_hotkey=HOTKEY,
    )
    assert parsed.challenger == ModelRef(repo=GOOD_REPO, digest="hf:" + "cd" * 20)
