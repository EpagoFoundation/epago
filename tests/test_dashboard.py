"""Dashboard exporter tests — the export is part of the audit surface, so its
derivations from state.json + audit.jsonl are pinned here."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from epago import constants
from epago.config import load_config
from epago.dashboard.export import (
    DASHBOARD_SCHEMA,
    REFUSAL_FALLBACK,
    REFUSAL_REASONS,
    export_dashboard,
    export_from_chain,
    load_dashboard_inputs,
    write_dashboard,
)


def _audit_record(**kw) -> dict:
    base = {
        "round_id": "r1001-miner-a",
        "block_hash_at_reveal": "0xabc",
        "author_hotkey": "miner-a",
        "king_repo": "k/EPAGO-DR-4B-x",
        "king_digest": "sha256:" + "1" * 64,
        "challenger_repo": "m/EPAGO-DR-4B-y",
        "challenger_digest": "sha256:" + "2" * 64,
        "corpus_digest": "sha256:" + "0" * 64,
        "taskgen_release": "R1",
        "public_seed": "00" * 8,
        "public_task_ids_digest": "sha256:" + "3" * 64,
        "private_pool_digest": "sha256:" + "4" * 64,
        "private_pool_epoch": 1,
        "n_private_tasks": 200,
        "boot_seed": "00" * 8,
        "king_acc_ema": 0.55,
        "delta_threshold": 0.0225,
        "mu_hat_pub": 0.04,
        "lcb_pub": 0.03,
        "mu_hat_priv": 0.02,
        "accepted": True,
        "harness_digest": "sha256:" + "5" * 64,
        "judge_model_digest": "hf:" + "6" * 40,
        "eval_code_digest": "sha256:" + "7" * 64,
        "judge_invocation_rate": 0.01,
        "revealed_at_block": 1000,
        "intake_at_block": 1001,
        "verdict_at_block": 1005,
        "validator_hotkey": "validator-0",
        "validator_signature": "",
        "extra": {},
    }
    base.update(kw)
    return base


@pytest.fixture()
def state_dir(tmp_path: Path) -> Path:
    state = {
        "king": {
            "repo": "m/EPAGO-DR-4B-y",
            "digest": "sha256:" + "2" * 64,
            "author_hotkey": "miner-a",
            "author_coldkey": "5A",
            "crowned_block": 1005,
            "reign_started_block": 1005,
            "acc_ema": 0.57,
            "coronation_lcb": 0.03,
        },
        "king_acc_ema": 0.57,
        "king_coronation_delta": 0.0225,
        "queue": [{"digest": "sha256:" + "9" * 64}],
        "statuses": {
            "sha256:" + "2" * 64: "accepted",
            "sha256:" + "8" * 64: "failed_intake",
        },
        "candidates": {},
        "arena": [{"hotkey": "miner-b", "lcb_pub": 0.01, "verdict_block": 1010}],
        "noise_floor_samples": [0.004],
        "clean_duels": 3,
        "organic_dethrones": 1,
        "genesis_block": 900,
        "sla": [{"digest": "d", "revealed_at": 1000, "intake_at": 1001, "verdict_at": 1005}],
        "last_scan_block": 1020,
        "last_weights_block": 1015,
        "last_pool_publish_block": 0,
        "tick_count": 5,
        "failure_memory": {},
        "seen_digests": {},
        "burned_bonds": [],
        "intake_log": [],
        "pending_mirror": None,
        "last_error": None,
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    (tmp_path / "audit").mkdir()
    records = [
        _audit_record(),
        _audit_record(
            round_id="r1006-miner-b",
            author_hotkey="miner-b",
            challenger_digest="sha256:" + "a" * 64,
            lcb_pub=0.01,
            mu_hat_pub=0.02,
            accepted=False,
            king_acc_ema=0.57,
            verdict_at_block=1010,
        ),
        _audit_record(
            round_id="r1007-miner-c",
            author_hotkey="miner-c",
            challenger_digest="sha256:" + "b" * 64,
            lcb_pub=-0.08,
            mu_hat_pub=-0.05,
            accepted=False,
            king_acc_ema=0.57,
            verdict_at_block=1012,
        ),
        _audit_record(round_id="calib-1013", accepted=False, verdict_at_block=1013),
    ]
    (tmp_path / "audit" / "audit.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    return tmp_path


def test_export_sections_and_schema(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert data["schema"] == DASHBOARD_SCHEMA
    for key in (
        "king", "kpis", "lineage", "accuracy_series", "duels", "miners",
        "funnel", "quorum", "sla", "emissions", "tasks", "noise", "queue",
        "round_running", "refused", "links",
    ):
        assert key in data


def test_queue_rows_carry_position_and_waiting(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    rows = data["queue"]
    assert len(rows) == 1
    row = rows[0]
    assert row["position"] == 1
    assert row["digest"] == "sha256:" + "9" * 64
    # waiting is measured from the reveal block against the export block, so
    # it is never negative even on a stale snapshot
    assert row["waiting_blocks"] >= 0


def test_calibration_rounds_excluded_from_duels(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert len(data["duels"]) == 3
    assert all(not r["round_id"].startswith("calib-") for r in data["duels"])


def test_duel_outcomes_come_from_the_duel_not_the_lifecycle(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    by_author = {r["author_hotkey"]: r["outcome"] for r in data["duels"]}
    assert by_author == {"miner-a": "accepted", "miner-b": "near_miss", "miner-c": "duel_lost"}


def test_duel_detail_exposes_per_task_and_provenance(state_dir):
    records = [
        _audit_record(
            round_id="r2000-miner-d",
            author_hotkey="miner-d",
            verdict_at_block=2000,
            extra={
                "public_diffs": [["tk-a", 1], ["tk-b", 0], ["tk-c", -1], ["tk-d", 1]],
                "judge_tier_counts": [["exact", 3], ["none", 5]],
            },
        )
    ]
    (state_dir / "audit" / "audit.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    det = data["duels"][-1]["detail"]
    assert (det["won"], det["tied"], det["lost"]) == (2, 1, 1)
    assert det["judge_tiers"] == [["exact", 3], ["none", 5]]
    assert [t["task_id"] for t in det["tasks"]] == ["tk-a", "tk-b", "tk-c", "tk-d"]
    # Provenance carries exactly the digests a replayer feeds to replay_verdict.
    prov = det["provenance"]
    assert prov["corpus_digest"] and prov["judge_model_digest"]
    assert prov["public_seed"] and prov["eval_code_digest"]


def test_duel_detail_tolerates_empty_extra(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    det = data["duels"][0]["detail"]  # fixture records carry extra={}
    assert det["tasks"] == [] and (det["won"], det["tied"], det["lost"]) == (0, 0, 0)
    assert det["provenance"]["corpus_digest"]  # provenance still populated


def _chain_with(neurons, reveals):
    """MockChainClient seeded with (uid,hotkey,stake,permit) neurons and
    (hotkey, payload, block) reveals; current block advanced past them."""
    from epago.chain.client import MockChainClient, NeuronView

    chain = MockChainClient(identity_hotkey="val-1")
    for uid, hk, stake, permit in neurons:
        chain.add_neuron(NeuronView(uid=uid, hotkey=hk, coldkey="ck-" + hk, stake=stake, validator_permit=permit))
    for hk, payload, block in reveals:
        chain.inject_reveal(hk, payload, block=block)
    chain.block = max([b for _, _, b in reveals], default=100) + 5
    return chain


def test_export_from_chain_derives_king_and_duel():
    from epago.core.reveal import build_reveal, build_verdict
    from epago.core.types import ModelRef, Verdict, VerdictDecision

    king_digest = "hf:" + "a" * 40
    chal = ModelRef(repo="org/EPAGO-DR-4B-x", digest="hf:" + "b" * 40)
    verdict = Verdict(
        challenger_digest=chal.digest, decision=VerdictDecision.ACCEPT,
        lcb_pub_e6=62500, mu_priv_e6=250000, delta_e6=25000, round=1,
        private_pool_epoch=1, audit_digest="0" * 16,
    )
    chain = _chain_with(
        neurons=[(0, "val-1", 100.0, True), (1, "miner-1", 0.0, False)],
        reveals=[
            ("miner-1", build_reveal(king_digest, chal), 110),
            ("val-1", build_verdict(verdict), 120),
        ],
    )
    data = export_from_chain(chain, load_config())

    assert data["source"] == "chain"
    # Same schema as the local exporter — the leaderboard renders it unchanged.
    for key in ("king", "kpis", "duels", "miners", "lineage", "emissions"):
        assert key in data
    # King derived purely from the on-chain ACCEPT + the e1 challenge author.
    assert data["king"]["digest"] == chal.digest
    assert data["king"]["author_hotkey"] == "miner-1"
    assert data["king"]["repo"] == "org/EPAGO-DR-4B-x"
    # The duel carries the on-chain lcb / mu_priv exactly.
    assert len(data["duels"]) == 1
    d = data["duels"][0]
    assert d["outcome"] == "accepted"
    assert d["lcb"] == pytest.approx(0.0625)
    assert d["mu_priv"] == pytest.approx(0.25)
    assert d["author_hotkey"] == "miner-1"
    # The crowning verdict's private mean reaches the champion card and the lineage.
    assert data["king"]["coronation_mu_priv"] == pytest.approx(0.25)
    assert data["lineage"][-1]["mu_priv"] == pytest.approx(0.25)


def test_export_from_chain_classifies_near_miss_and_lost_without_king():
    from epago.core.reveal import build_reveal, build_verdict
    from epago.core.types import ModelRef, Verdict, VerdictDecision

    king_digest = "hf:" + "a" * 40
    nm = ModelRef(repo="org/EPAGO-DR-4B-nm", digest="hf:" + "c" * 40)   # reject, lcb>0
    lost = ModelRef(repo="org/EPAGO-DR-4B-l", digest="hf:" + "d" * 40)  # reject, lcb<=0
    v_nm = Verdict(nm.digest, VerdictDecision.REJECT, 10000, 5000, 25000, 1, 1, "0" * 16)
    v_lost = Verdict(lost.digest, VerdictDecision.REJECT, -80000, -50000, 25000, 1, 1, "0" * 16)
    chain = _chain_with(
        neurons=[(0, "val-1", 100.0, True), (1, "m-nm", 0.0, False), (2, "m-lost", 0.0, False)],
        reveals=[
            ("m-nm", build_reveal(king_digest, nm), 110),
            ("m-lost", build_reveal(king_digest, lost), 111),
            ("val-1", build_verdict(v_nm), 120),
            ("val-1", build_verdict(v_lost), 121),
        ],
    )
    data = export_from_chain(chain, load_config())

    assert data["king"] is None  # no ACCEPT anywhere → no crown
    by_digest = {d["digest"]: d["outcome"] for d in data["duels"]}
    assert by_digest[nm.digest] == "near_miss"
    assert by_digest[lost.digest] == "duel_lost"


def test_export_from_chain_ignores_verdicts_from_non_validators():
    """A verdict from a hotkey without validator_permit must not crown anyone."""
    from epago.core.reveal import build_reveal, build_verdict
    from epago.core.types import ModelRef, Verdict, VerdictDecision

    king_digest = "hf:" + "a" * 40
    chal = ModelRef(repo="org/EPAGO-DR-4B-x", digest="hf:" + "b" * 40)
    verdict = Verdict(chal.digest, VerdictDecision.ACCEPT, 62500, 250000, 25000, 1, 1, "0" * 16)
    chain = _chain_with(
        neurons=[(0, "impostor", 100.0, False), (1, "miner-1", 0.0, False)],
        reveals=[
            ("miner-1", build_reveal(king_digest, chal), 110),
            ("impostor", build_verdict(verdict), 120),  # not a permitted validator
        ],
    )
    data = export_from_chain(chain, load_config())
    assert data["king"] is None
    assert data["duels"] == []


def test_lineage_and_miners(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert [l["author_hotkey"] for l in data["lineage"]] == ["miner-a"]
    miners = {m["hotkey"]: m for m in data["miners"]}
    assert miners["miner-a"]["accepted"] == 1 and miners["miner-a"]["is_king"]
    assert miners["miner-b"]["near_miss"] == 1
    assert miners["miner-b"]["arena_credit"] == pytest.approx(0.01)
    assert data["miners"][0]["hotkey"] == "miner-a"  # crowned sorts first


def test_private_mean_is_shown_for_the_king_the_lineage_and_miners(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert data["king"]["coronation_mu_priv"] == pytest.approx(0.02)
    assert data["lineage"][0]["mu_priv"] == pytest.approx(0.02)
    miners = {m["hotkey"]: m for m in data["miners"]}
    assert miners["miner-a"]["best_mu_priv"] == pytest.approx(0.02)
    assert miners["miner-b"]["best_mu_priv"] == pytest.approx(0.02)


def test_a_miners_private_mean_comes_from_its_best_scoring_duel(tmp_path):
    state = {"king": {"digest": "hf:" + "0" * 40, "author_hotkey": ""}, "arena": []}
    (tmp_path / "state.json").write_text(json.dumps(state))
    (tmp_path / "audit").mkdir()
    records = [
        _audit_record(round_id="r1", challenger_digest="sha256:" + "c" * 64, accepted=False,
                      lcb_pub=0.01, mu_hat_priv=0.30, verdict_at_block=1001),
        _audit_record(round_id="r2", challenger_digest="sha256:" + "d" * 64, accepted=False,
                      lcb_pub=0.05, mu_hat_priv=-0.04, verdict_at_block=1002),
        _audit_record(round_id="r3", challenger_digest="sha256:" + "e" * 64, accepted=False,
                      lcb_pub=0.02, mu_hat_priv=0.50, verdict_at_block=1003),
    ]
    (tmp_path / "audit" / "audit.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    data = export_dashboard(load_dashboard_inputs(tmp_path, load_config()))
    (miner,) = data["miners"]
    # Paired with the best LCB (0.05), not the largest mean or the latest duel.
    assert miner["best_lcb"] == pytest.approx(0.05)
    assert miner["best_mu_priv"] == pytest.approx(-0.04)
    # A genesis king never won a duel, so it has no private mean to show.
    assert data["king"]["coronation_mu_priv"] is None


def test_kpis_and_noise_clamp(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    k = data["kpis"]
    assert k["duels_total"] == 3 and k["duels_accepted"] == 1
    assert k["phase"] == "burn"
    assert data["noise"]["floor"] == pytest.approx(0.004)
    assert k["delta_clamp"] == pytest.approx(constants.DELTA_NOISE_MULTIPLIER * 0.004)
    assert k["queue_depth"] == 1


def test_funnel_counts_statuses(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    counts = {r["key"]: r["count"] for r in data["funnel"]}
    assert counts["accepted"] == 1
    assert counts["failed_intake"] == 1
    assert counts["queued"] == 1


def _schedule_config():
    """The repo contract with its fixed burn off: the king-and-arena schedule."""
    cfg = load_config()
    return replace(cfg, emissions=replace(cfg.emissions, burn_share=0.0))


def test_emissions_burn_everything_before_the_phase_gate(state_dir):
    """Phase A pays nobody: the panel must say what the validator actually sets.

    The fixture is pre-gate (``kpis.phase == "burn"``), and in that regime
    ValidatorService.maybe_set_weights puts weight 1.0 on the burn key. The
    panel used to render the Phase B split anyway — a king with a 90% share
    next to a validator burning the whole emission.
    """
    data = export_dashboard(load_dashboard_inputs(state_dir, _schedule_config()))
    e = data["emissions"]
    assert data["kpis"]["phase"] == "burn"
    assert e == {"king": 0.0, "arena": 0.0, "burn": 1.0}


def test_emissions_under_a_fixed_burn(state_dir):
    """The mainnet contract pays the king 5% and burns the rest, gate or not."""
    e = export_dashboard(load_dashboard_inputs(state_dir, load_config()))["emissions"]
    assert {k: e[k] for k in ("king", "arena", "burn")} == pytest.approx(
        {"king": 0.05, "arena": 0.0, "burn": 0.95}
    )


def test_emissions_normalized(state_dir, monkeypatch):
    monkeypatch.setattr(constants, "PHASE_B_MIN_CLEAN_DUELS", 1)
    monkeypatch.setattr(constants, "PHASE_B_MIN_BLOCKS", 0)
    data = export_dashboard(load_dashboard_inputs(state_dir, _schedule_config()))
    e = data["emissions"]
    # King and arena split the whole budget; the dashboard must mirror
    # compute_weights rather than carry its own arithmetic.
    assert e["king"] + e["arena"] + e["burn"] == pytest.approx(1.0)
    assert e["king"] > 0.5


def test_write_dashboard_installs_site(state_dir, tmp_path):
    out = tmp_path / "dash"
    path = write_dashboard(state_dir, out, load_config())
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["schema"] == DASHBOARD_SCHEMA
    site = out / "index.html"
    assert site.exists()
    assert "EPAGO_DATA" in site.read_text()  # inline-data hook for demos


def test_export_deterministic(state_dir):
    cfg = load_config()
    a = export_dashboard(load_dashboard_inputs(state_dir, cfg))
    b = export_dashboard(load_dashboard_inputs(state_dir, cfg))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --- rounds -------------------------------------------------------------------


def _round_state_dir(tmp_path: Path) -> Path:
    """A state dir whose audit log holds two competition rounds."""
    state_dir = tmp_path / "rounds-state"
    (state_dir / "audit").mkdir(parents=True)
    records = [
        # round 7: three entrants, one winner, one true near-miss, one loss
        _audit_record(
            round_id="round000007-aaaaaaaa", author_hotkey="miner-a",
            challenger_digest="sha256:" + "a" * 64, accepted=True,
            lcb_pub=0.031, mu_hat_priv=0.02, verdict_at_block=2000,
            extra={"round": 7, "round_block": 1990},
        ),
        _audit_record(
            round_id="round000007-bbbbbbbb", author_hotkey="miner-b",
            challenger_digest="sha256:" + "b" * 64, accepted=False,
            lcb_pub=0.027, mu_hat_priv=0.01, verdict_at_block=2001,
            extra={"round": 7, "round_block": 1990},
        ),
        _audit_record(
            round_id="round000007-cccccccc", author_hotkey="miner-c",
            challenger_digest="sha256:" + "c" * 64, accepted=False,
            lcb_pub=-0.04, mu_hat_priv=-0.01, verdict_at_block=2002,
            extra={"round": 7, "round_block": 1990},
        ),
        # round 8: nobody cleared the bar
        _audit_record(
            round_id="round000008-dddddddd", author_hotkey="miner-d",
            challenger_digest="sha256:" + "d" * 64, accepted=False,
            lcb_pub=0.004, mu_hat_priv=0.003, verdict_at_block=16500,
            extra={"round": 8, "round_block": 16400},
        ),
        # a pre-rounds record: must not be claimed by any round
        _audit_record(round_id="r900-legacy", verdict_at_block=950, extra={}),
    ]
    (state_dir / "audit" / "audit.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    state = {
        "king": None, "king_acc_ema": 0.5, "king_coronation_delta": 0.0,
        "queue": [], "statuses": {}, "candidates": {}, "arena": [],
        "noise_floor_samples": [], "clean_duels": 4, "organic_dethrones": 1,
        "genesis_block": 900, "sla": [], "last_scan_block": 16600,
        "last_weights_block": None, "last_pool_publish_block": 0,
        "tick_count": 9, "failure_memory": {}, "seen_digests": {},
        "intake_log": [], "pending_mirror": None, "last_error": None,
        "current_round": 8, "last_round_run": 8,
    }
    (state_dir / "state.json").write_text(json.dumps(state))
    return state_dir


def test_rounds_section_groups_by_competition(tmp_path):
    data = export_dashboard(load_dashboard_inputs(_round_state_dir(tmp_path), load_config()))
    rounds = data["rounds"]
    assert [r["round"] for r in rounds] == [8, 7]  # newest first

    r7 = rounds[1]
    assert r7["entrants"] == 3
    assert r7["winner_hotkey"] == "miner-a"
    assert r7["winner_lcb"] == pytest.approx(0.031)
    # miner-b beat the king but lost the round: near-miss. miner-c just lost.
    assert r7["near_misses"] == 1

    r8 = rounds[0]
    assert r8["winner_hotkey"] == ""      # no coronation this round
    assert r8["best_lcb"] == pytest.approx(0.004)

    # The legacy record joined no round but is still in the duel table.
    assert any(d["round_id"] == "r900-legacy" for d in data["duels"])
    assert all(r["round"] != 0 for r in rounds)


def test_kpis_carry_the_round_counters(tmp_path):
    data = export_dashboard(load_dashboard_inputs(_round_state_dir(tmp_path), load_config()))
    assert data["kpis"]["current_round"] == 8
    assert data["kpis"]["last_round_run"] == 8
    # Every round-born duel row knows its round; the legacy one reads 0.
    rounds_seen = {d["round"] for d in data["duels"]}
    assert {0, 7, 8} <= rounds_seen


# --- round running, refused, links --------------------------------------------


def _update_state(state_dir: Path, **changes) -> None:
    state = json.loads((state_dir / "state.json").read_text())
    state.update(changes)
    (state_dir / "state.json").write_text(json.dumps(state))


def _digest(c: str) -> str:
    return "sha256:" + c * 64


# What a validator's own detail text can hold; none of it may reach the export.
_SECRET = "OSError: /root/secret/cache/model.safetensors via https://internal.example/x"


def test_round_running_is_null_when_idle_or_absent(state_dir):
    # The fixture predates the key: an older state file reads as idle.
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert data["round_running"] is None
    _update_state(state_dir, round_in_progress=None)
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert data["round_running"] is None


def test_round_running_carries_round_block_and_entrants(state_dir):
    entrants = [
        {"digest": _digest("c"), "author_hotkey": "miner-c", "repo": "submissions/miner-c/x"},
        {"digest": _digest("d"), "author_hotkey": "miner-d", "repo": "submissions/miner-d/y"},
    ]
    _update_state(state_dir, round_in_progress={"round": 9, "block": 1018, "entrants": entrants})
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert data["round_running"] == {"round": 9, "block": 1018, "entrants": entrants}


def test_refused_lists_only_submissions_that_never_dueled(state_dir):
    _update_state(
        state_dir,
        statuses={
            _digest("2"): "accepted",
            _digest("a"): "near_miss",
            _digest("b"): "duel_lost",
            _digest("c"): "failed_intake",
            _digest("e"): "failed_probes",
            _digest("f"): "stale_parent",
            _digest("9"): "queued",
        },
        failure_memory={
            _digest("b"): {"code": "duel_lost", "detail": "lcb_pub=-0.080000", "block": 1012},
            _digest("c"): {"code": "exact_copy", "detail": _SECRET, "block": 1016},
            _digest("e"): {"code": "probes", "detail": _SECRET, "block": 1017},
        },
        seen_digests={
            _digest("2"): "miner-a", _digest("a"): "miner-b", _digest("b"): "miner-c",
            _digest("c"): "miner-e", _digest("e"): "miner-f", _digest("f"): "miner-g",
            _digest("9"): "miner-h",
        },
        intake_log=[
            {"hotkey": "miner-g", "digest": _digest("f"), "code": "stale_parent", "detail": _SECRET, "block": 1011},
            # refused once, then admitted: no longer a refusal
            {"hotkey": "miner-h", "digest": _digest("9"), "code": "cooldown", "detail": _SECRET, "block": 1012},
            {"hotkey": "miner-h", "digest": _digest("9"), "code": "queued", "detail": "queue_scale=1", "block": 1013},
            # intake re-reads every reveal, so the same refusal repeats each tick
            {"hotkey": "miner-i", "digest": _digest("7"), "code": "hotkey_spent", "detail": _SECRET, "block": 1014},
            {"hotkey": "miner-i", "digest": _digest("7"), "code": "hotkey_spent", "detail": _SECRET, "block": 1015},
            # the king's digest revealed again by someone else
            {"hotkey": "miner-j", "digest": _digest("2"), "code": "duplicate_digest", "detail": _SECRET, "block": 1019},
        ],
    )
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    rows = data["refused"]
    # Newest first; the dueled ones (accepted, near miss, lost) are left to the duel table.
    assert [(r["author_hotkey"], r["code"], r["block"]) for r in rows] == [
        ("miner-j", "duplicate_digest", 1019),
        ("miner-f", "probes", 1017),
        ("miner-e", "exact_copy", 1016),
        ("miner-i", "hotkey_spent", 1014),
        ("miner-g", "stale_parent", 1011),
    ]
    for r in rows:
        assert set(r) == {"digest", "author_hotkey", "code", "reason", "block"}
        assert r["reason"] == REFUSAL_REASONS[r["code"]]
    # The validator's own detail text never leaves the box.
    dumped = json.dumps(data)
    assert "/root/secret" not in dumped
    assert "internal.example" not in dumped


def test_unknown_refusal_code_gets_the_fallback(state_dir):
    _update_state(
        state_dir,
        statuses={_digest("c"): "failed_intake", _digest("e"): "failed_intake"},
        failure_memory={
            _digest("c"): {"code": "brand_new_check", "detail": _SECRET, "block": 1016},
            # a code that is not a plain identifier is not exported either
            _digest("e"): {"code": "OSError at /root/secret", "detail": _SECRET, "block": 1015},
        },
        seen_digests={_digest("c"): "miner-e", _digest("e"): "miner-f"},
    )
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    by_hotkey = {r["author_hotkey"]: r for r in data["refused"]}
    assert by_hotkey["miner-e"]["code"] == "brand_new_check"
    assert by_hotkey["miner-e"]["reason"] == REFUSAL_FALLBACK
    assert by_hotkey["miner-f"]["code"] == "unknown"
    assert by_hotkey["miner-f"]["reason"] == REFUSAL_FALLBACK
    assert "/root/secret" not in json.dumps(data)


def test_links_point_at_the_published_files(state_dir):
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert data["links"] == {"audit_log": "../audit/audit.jsonl", "file_index": "../index.json"}
    # Relative to the page, so no host is baked into the export.
    assert all("://" not in v for v in data["links"].values())


# --- champion accuracy ---------------------------------------------------------------


def _with_acc_history(state_dir, history):
    s = json.loads((state_dir / "state.json").read_text())
    s["king_acc_history"] = history
    (state_dir / "state.json").write_text(json.dumps(s))


_TWO_ROUNDS = [
    {"round": 1, "block": 1100, "observed": 0.21, "ema": 0.21, "coronation": False},
    {"round": 2, "block": 1200, "observed": 0.10, "ema": 0.19, "coronation": True},
]


def test_no_champion_accuracy_is_shown_before_a_round_is_scored(state_dir):
    """The validator's 0.5 stand-in is not a measurement, so nothing is shown."""
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert data["accuracy_series"] == []
    kp = data["kpis"]
    assert (kp["king_acc_ema"], kp["king_acc_ema_prev"], kp["king_acc_genesis"], kp["king_acc_rounds"]) == (
        None, None, None, 0,
    )
    assert data["king"]["acc_ema"] is None


def test_champion_accuracy_comes_from_each_rounds_measurement(state_dir):
    _with_acc_history(state_dir, _TWO_ROUNDS)
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    assert [(p["round"], p["ema"], p["coronation"]) for p in data["accuracy_series"]] == [
        (1, 0.21, False), (2, 0.19, True),
    ]
    kp = data["kpis"]
    assert (kp["king_acc_genesis"], kp["king_acc_ema_prev"], kp["king_acc_ema"], kp["king_acc_rounds"]) == (
        0.21, 0.21, 0.19, 2,
    )
    assert data["king"]["acc_ema"] == 0.19


def test_the_next_bar_uses_measured_accuracy_not_only_the_noise_clamp(state_dir):
    """A weak king leaves headroom, and then the headroom term binds above the clamp."""
    from epago.core.stats import adaptive_delta, noise_floor_from_calibration

    _with_acc_history(state_dir, _TWO_ROUNDS)
    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    noise = noise_floor_from_calibration([0.004])
    assert data["kpis"]["delta_next"] == pytest.approx(adaptive_delta(0.19, noise))
    assert data["kpis"]["delta_next"] > data["kpis"]["delta_clamp"]


def test_before_any_round_the_next_bar_uses_the_stand_in(state_dir):
    from epago.core.stats import adaptive_delta, noise_floor_from_calibration

    data = export_dashboard(load_dashboard_inputs(state_dir, load_config()))
    expected = adaptive_delta(0.5, noise_floor_from_calibration([0.004]))
    assert data["kpis"]["delta_next"] == pytest.approx(expected)
