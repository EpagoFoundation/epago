"""Chain doctor + auditor verification tests.

``run_doctor`` is exercised against a duck-typed in-memory subtensor stub (no
network), covering every PASS/FAIL/WARN/SKIP branch. The replay-verdict
signature check is exercised with a REAL bittensor sr25519 keypair
(``//Alice``) — sign/verify needs no chain.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from epago.chain.doctor import commit_reveal_fix_command, run_doctor
from epago.core.stats import bootstrap_lcb

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("replay_verdict", ROOT / "scripts" / "replay_verdict.py")
replay = importlib.util.module_from_spec(_spec)
sys.modules["replay_verdict"] = replay
_spec.loader.exec_module(replay)


# ---------------------------------------------------------------------------
# stub chain
# ---------------------------------------------------------------------------


def _neuron(uid: int, hotkey: str, permit: bool = False, stake: float = 0.0):
    return SimpleNamespace(
        uid=uid, hotkey=hotkey, coldkey=f"ck-{hotkey}", total_stake=stake, validator_permit=permit
    )


def _wallet(ss58: str = "hk-val"):
    return SimpleNamespace(name="doctor", hotkey=SimpleNamespace(ss58_address=ss58))


class StubSubtensor:
    """Duck-typed subtensor: exactly the surface run_doctor touches."""

    def __init__(
        self,
        netuid: int = 7,
        commit_reveal: bool = True,
        liquid_alpha: bool = True,
        neurons=(),
        advance: bool = True,
        reveal_works: bool = True,
    ) -> None:
        self.netuid = netuid
        self._block = 1000
        self._advance = advance
        self._neurons = list(neurons)
        self._hyper = SimpleNamespace(
            tempo=360,
            commit_reveal_period=1,
            commit_reveal_weights_enabled=commit_reveal,
            liquid_alpha_enabled=liquid_alpha,
            alpha_low=0.7,
            alpha_high=0.9,
        )
        self._commitments: dict[str, str] = {}
        self._pending: list[tuple[int, str, str]] = []
        self._reveal_works = reveal_works

    def get_current_block(self) -> int:
        if self._advance:
            self._block += 1
        return self._block

    def get_subnet_hyperparameters(self, netuid: int):
        return self._hyper if netuid == self.netuid else None

    def neurons_lite(self, netuid: int):
        return list(self._neurons)

    def set_commitment(self, *, wallet, netuid, data, raise_error=True) -> None:
        self._commitments[wallet.hotkey.ss58_address] = data

    def get_all_commitments(self, netuid: int) -> dict[str, str]:
        return dict(self._commitments)

    def set_reveal_commitment(self, *, wallet, netuid, data, blocks_until_reveal, raise_error=True):
        if self._reveal_works:
            self._pending.append((self._block + blocks_until_reveal, wallet.hotkey.ss58_address, data))

    def get_all_revealed_commitments(self, netuid: int):
        if self._advance:
            self._block += 1  # polling a stub also moves time forward
        out: dict[str, list[tuple[int, str]]] = {}
        for block, hotkey, payload in self._pending:
            if block <= self._block:
                out.setdefault(hotkey, []).append((block, payload))
        return {hk: tuple(v) for hk, v in out.items()}


def _by_name(results):
    return {r.name: r for r in results}


# ---------------------------------------------------------------------------
# run_doctor
# ---------------------------------------------------------------------------


def test_doctor_happy_path_with_probes():
    stub = StubSubtensor(
        neurons=[_neuron(0, "hk-val", permit=True, stake=1234.5), _neuron(1, "hk-miner")]
    )
    results = run_doctor(
        "finney",
        7,
        probe_writes=True,
        subtensor=stub,
        wallet=_wallet("hk-val"),
        reveal_timeout_s=2.0,
        poll_interval_s=0.01,
    )
    by = _by_name(results)
    for name in (
        "chain",
        "subnet",
        "commit-reveal",
        "tempo",
        "liquid-alpha",
        "neurons",
        "wallet-registered",
        "wallet-permit",
        "wallet-stake",
        "probe-status",
        "probe-reveal",
    ):
        assert by[name].status == "PASS", f"{name}: {by[name].detail}"
    assert "uid 0" in by["wallet-registered"].detail
    assert "1 with validator permit" in by["neurons"].detail
    assert "1234.5" in by["wallet-stake"].detail
    assert "blocks" in by["probe-reveal"].detail  # measured reveal latency reported


def test_commit_reveal_disabled_fails_with_fix_command():
    stub = StubSubtensor(commit_reveal=False)
    by = _by_name(run_doctor("finney", 7, subtensor=stub))
    assert by["commit-reveal"].status == "FAIL"
    assert commit_reveal_fix_command(7) in by["commit-reveal"].detail
    assert "btcli sudo set --netuid 7 --param commit_reveal_weights_enabled --value 1" in (
        by["commit-reveal"].detail
    )


def test_missing_netuid_fails_and_downstream_skips():
    stub = StubSubtensor(netuid=7)
    by = _by_name(run_doctor("finney", 99, subtensor=stub))
    assert by["subnet"].status == "FAIL"
    assert "netuid 99" in by["subnet"].detail
    for name in ("commit-reveal", "tempo", "liquid-alpha", "neurons"):
        assert by[name].status == "SKIP"


def test_liquid_alpha_disabled_warns():
    stub = StubSubtensor(liquid_alpha=False)
    by = _by_name(run_doctor("finney", 7, subtensor=stub))
    assert by["liquid-alpha"].status == "WARN"


def test_wallet_not_registered_fails():
    stub = StubSubtensor(neurons=[_neuron(0, "hk-other", permit=True, stake=10.0)])
    by = _by_name(run_doctor("finney", 7, subtensor=stub, wallet=_wallet("hk-val")))
    assert by["wallet-registered"].status == "FAIL"
    assert "NOT registered" in by["wallet-registered"].detail
    assert by["wallet-permit"].status == "SKIP"
    assert by["wallet-stake"].status == "SKIP"


def test_wallet_without_permit_warns():
    stub = StubSubtensor(neurons=[_neuron(3, "hk-val", permit=False, stake=5.0)])
    by = _by_name(run_doctor("finney", 7, subtensor=stub, wallet=_wallet("hk-val")))
    assert by["wallet-registered"].status == "PASS"
    assert by["wallet-permit"].status == "WARN"


def test_no_wallet_skips_wallet_checks_and_probes_skip_by_default():
    stub = StubSubtensor()
    by = _by_name(run_doctor("finney", 7, subtensor=stub))
    assert by["wallet-registered"].status == "SKIP"
    assert by["probe-status"].status == "SKIP"
    assert by["probe-reveal"].status == "SKIP"


def test_probe_writes_without_wallet_skips():
    stub = StubSubtensor()
    by = _by_name(run_doctor("finney", 7, probe_writes=True, subtensor=stub))
    assert by["probe-status"].status == "SKIP"
    assert "wallet" in by["probe-status"].detail


def test_stalled_block_warns_not_fails():
    stub = StubSubtensor(advance=False)
    by = _by_name(
        run_doctor("finney", 7, subtensor=stub, block_advance_timeout_s=0.05, poll_interval_s=0.01)
    )
    assert by["chain"].status == "WARN"
    assert "did not advance" in by["chain"].detail


def test_reveal_probe_timeout_warns_with_guidance():
    stub = StubSubtensor(
        neurons=[_neuron(0, "hk-val", permit=True, stake=1.0)], reveal_works=False
    )
    by = _by_name(
        run_doctor(
            "finney",
            7,
            probe_writes=True,
            subtensor=stub,
            wallet=_wallet("hk-val"),
            reveal_timeout_s=0.05,
            poll_interval_s=0.01,
        )
    )
    assert by["probe-status"].status == "PASS"
    assert by["probe-reveal"].status == "WARN"
    assert "watch-reveals" in by["probe-reveal"].detail


def test_one_crashing_check_never_aborts_the_run():
    class Exploding(StubSubtensor):
        def neurons_lite(self, netuid):
            raise RuntimeError("rpc went away")

    stub = Exploding(neurons=[_neuron(0, "hk-val", permit=True, stake=1.0)])
    results = run_doctor("finney", 7, subtensor=stub, wallet=_wallet("hk-val"))
    by = _by_name(results)
    assert by["neurons"].status == "FAIL"
    assert "RuntimeError" in by["neurons"].detail
    # everything after the crash still reported
    assert by["wallet-registered"].status == "SKIP"
    assert by["probe-status"].status == "SKIP"
    assert by["probe-reveal"].status == "SKIP"


# ---------------------------------------------------------------------------
# replay_verdict: signature / lcb / mined / chain checks
# ---------------------------------------------------------------------------


def _make_record(validator_hotkey: str = "hk-val", signature: str = "") -> dict:
    diffs = [1, 0, 1, -1, 0, 1, 0, 0, 1, 1]
    pairs = [[f"task-{i:02d}", d] for i, d in enumerate(diffs)]
    boot_seed = 987654321
    return {
        "round_id": "r1",
        "block_hash_at_reveal": "0xabc",
        "author_hotkey": "hk-author",
        "king_repo": "org/king",
        "king_digest": "sha256:" + "0" * 64,
        "challenger_repo": "org/chall",
        "challenger_digest": "sha256:" + "1" * 64,
        "corpus_digest": "sha256:" + "2" * 64,
        "taskgen_release": "R1",
        "public_seed": "0",
        "public_task_ids_digest": "sha256:" + "3" * 64,
        "private_pool_digest": "sha256:" + "4" * 64,
        "private_pool_epoch": 1,
        "n_private_tasks": 200,
        "boot_seed": str(boot_seed),
        "king_acc_ema": 0.5,
        "delta_threshold": 0.02,
        "mu_hat_pub": sum(diffs) / len(diffs),
        "lcb_pub": bootstrap_lcb(tuple(diffs), boot_seed),
        "mu_hat_priv": 0.1,
        "accepted": True,
        "harness_digest": "sha256:" + "5" * 64,
        "judge_model_digest": "hf:" + "6" * 40,
        "eval_code_digest": "sha256:" + "7" * 64,
        "judge_invocation_rate": 0.1,
        "revealed_at_block": 100,
        "intake_at_block": 105,
        "verdict_at_block": 140,
        "validator_hotkey": validator_hotkey,
        "validator_signature": signature,
        "extra": {
            "public_diffs": pairs,
            "judge_tier_counts": [["exact", 300], ["judge", 30]],
        },
    }


_BOOT_SEED = 987654321


def _signed_record(keypair) -> dict:
    record = _make_record(validator_hotkey=keypair.ss58_address)
    digest = replay._unsigned_digest(record)
    record["validator_signature"] = keypair.sign(digest.encode()).hex()
    return record


def test_signature_roundtrip_with_real_keypair():
    import bittensor as bt

    record = _signed_record((getattr(bt, "Keypair", None) or bt.sp_core.Keypair).create_from_uri("//Alice"))
    check = replay._check_signature(record)
    assert check.status == "PASS", check.detail


def test_signature_tampered_record_fails():
    import bittensor as bt

    record = _signed_record((getattr(bt, "Keypair", None) or bt.sp_core.Keypair).create_from_uri("//Alice"))
    record["accepted"] = False  # rewrite history after signing
    assert replay._check_signature(record).status == "FAIL"


def test_signature_wrong_key_fails():
    import bittensor as bt

    alice = (getattr(bt, "Keypair", None) or bt.sp_core.Keypair).create_from_uri("//Alice")
    bob = (getattr(bt, "Keypair", None) or bt.sp_core.Keypair).create_from_uri("//Bob")
    record = _make_record(validator_hotkey=alice.ss58_address)
    digest = replay._unsigned_digest(record)
    record["validator_signature"] = bob.sign(digest.encode()).hex()
    assert replay._check_signature(record).status == "FAIL"


def test_unsigned_record_skips_signature():
    check = replay._check_signature(_make_record(signature=""))
    assert check.status == "SKIP"


def test_lcb_recomputes_from_public_diff_pairs():
    record = _make_record()
    check = replay._check_lcb(record, _BOOT_SEED)
    assert check.status == "PASS", check.detail


def test_lcb_mismatch_fails():
    record = _make_record()
    record["lcb_pub"] = float(record["lcb_pub"]) + 0.01
    assert replay._check_lcb(record, _BOOT_SEED).status == "FAIL"


def test_audit16_matches_signature_zeroed_digest():
    import bittensor as bt

    record = _signed_record((getattr(bt, "Keypair", None) or bt.sp_core.Keypair).create_from_uri("//Alice"))
    audit16 = replay._unsigned_digest(record)[:16]
    verdict = f"ev3|{record['challenger_digest']}|A|1000|2000|0|1|1|{audit16}"
    check = replay._check_audit16(record, verdict, None)
    assert check.status == "PASS", check.detail


def test_chain_check_skips_without_netuid():
    check = replay._check_chain(_make_record(), "finney", None)
    assert check.status == "SKIP"


def test_chain_cross_check_finds_matching_reveal(monkeypatch):
    import bittensor as bt

    record = _make_record()
    audit16 = replay._unsigned_digest(record)[:16]
    payload = f"ev3|{record['challenger_digest']}|A|1000|2000|0|1|1|{audit16}"

    class FakeSubtensor:
        def __init__(self, network=None):
            pass

        def get_all_revealed_commitments(self, netuid):
            return {"hk-val": ((140, payload),)}

    monkeypatch.setattr(bt, "Subtensor", FakeSubtensor)
    check = replay._check_chain(record, "finney", 7)
    assert check.status == "PASS", check.detail
    assert "140" in check.detail


def test_chain_cross_check_hotkey_mismatch_fails(monkeypatch):
    import bittensor as bt

    record = _make_record(validator_hotkey="hk-val")
    audit16 = replay._unsigned_digest(record)[:16]
    payload = f"ev3|{record['challenger_digest']}|A|1000|2000|0|1|1|{audit16}"

    class FakeSubtensor:
        def __init__(self, network=None):
            pass

        def get_all_revealed_commitments(self, netuid):
            return {"hk-imposter": ((140, payload),)}

    monkeypatch.setattr(bt, "Subtensor", FakeSubtensor)
    assert replay._check_chain(record, "finney", 7).status == "FAIL"


def test_chain_cross_check_no_match_fails(monkeypatch):
    import bittensor as bt

    class FakeSubtensor:
        def __init__(self, network=None):
            pass

        def get_all_revealed_commitments(self, netuid):
            return {"hk-val": ((140, "e1|not-a-verdict"),)}

    monkeypatch.setattr(bt, "Subtensor", FakeSubtensor)
    assert replay._check_chain(_make_record(), "finney", 7).status == "FAIL"


def test_chain_cross_check_skips_when_unreachable(monkeypatch):
    import bittensor as bt

    class FakeSubtensor:
        def __init__(self, network=None):
            raise ConnectionError("no chain here")

    monkeypatch.setattr(bt, "Subtensor", FakeSubtensor)
    check = replay._check_chain(_make_record(), "finney", 7)
    assert check.status == "SKIP"
    assert "unreachable" in check.detail
