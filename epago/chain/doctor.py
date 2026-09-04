"""Chain preflight doctor.

``run_doctor`` exercises everything Epago needs from a subtensor before a
validator or miner goes live: connectivity, the netuid, the hyperparameters
the mechanism depends on (commit-reveal weights above all), the neuron/stake
landscape, the operator's wallet registration, and — optionally — real write
probes through both on-chain channels the subnet uses (the 128-byte plaintext
status slot and the timelock-reveal channel, including measured reveal
latency).

Every check is individually wrapped: one failure reports FAIL and the run
continues, so a single ``epago chain check`` gives the full picture.

The subtensor is duck-typed (only the methods listed below are called), so
tests run against an in-memory stub with no network:

* ``get_current_block()``
* ``get_subnet_hyperparameters(netuid)``
* ``neurons_lite(netuid)``
* ``set_commitment(wallet=, netuid=, data=, raise_error=)`` /
  ``get_all_commitments(netuid)``
* ``set_reveal_commitment(wallet=, netuid=, data=, blocks_until_reveal=,
  raise_error=)`` / ``get_all_revealed_commitments(netuid)``
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str = ""


def _guard(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    """Run one check; any exception becomes a FAIL, never an aborted run."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the whole point is to keep going
        return CheckResult(name, FAIL, f"{type(exc).__name__}: {exc}")


def _stake_tao(neuron: Any) -> float:
    """total_stake is a Balance on the real SDK; plain float on stubs."""
    stake = getattr(neuron, "total_stake", 0.0)
    return float(getattr(stake, "tao", stake))


# --- raw-storage fallbacks ------------------------------------------------------
#
# ``get_subnet_hyperparameters`` and ``neurons_lite`` are runtime-API calls whose
# netuid encoding drifts against a live-upgraded runtime (it wants the bare u16
# wrapped as a Composite), so both raise on a chain that is otherwise perfectly
# healthy. :class:`epago.chain.client.BittensorChainClient` already routes the
# scoring path around exactly this; the doctor built its own bare Subtensor and
# did not, so a preflight against such a chain reported "netuid does not exist"
# and SKIPped eight of its eleven checks on a live subnet. These fallbacks read
# the same facts from storage and are attempted only after the runtime call has
# raised, so a stub subtensor (tests) never reaches them.


def _substrate(subtensor: Any) -> Any:
    return getattr(subtensor, "substrate", None)


def _unwrap(value: Any) -> Any:
    """This runtime returns several scalars wrapped in a one-element sequence."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _hyperparams_via_storage(subtensor: Any, netuid: int) -> Any:
    """The hyperparameters this doctor reports, read from raw storage.

    Returns ``None`` when the subnet genuinely does not exist (or when there is
    no substrate handle to read), so the caller can still report a real FAIL.
    """
    from types import SimpleNamespace

    sub = _substrate(subtensor)
    if sub is None:
        return None

    def query(name: str, default: Any = None) -> Any:
        try:
            return _unwrap(sub.query("SubtensorModule", name, [netuid]).value)
        except Exception:  # noqa: BLE001 - a missing item is a default, not a crash
            return default

    if not bool(query("NetworksAdded", False)):
        return None
    alpha = query("AlphaValues", (0, 0))
    alpha_low, alpha_high = (
        (alpha[0], alpha[1]) if isinstance(alpha, (list, tuple)) and len(alpha) >= 2 else (0, 0)
    )
    return SimpleNamespace(
        commit_reveal_weights_enabled=bool(query("CommitRevealWeightsEnabled", False)),
        tempo=int(query("Tempo", 0) or 0),
        commit_reveal_period=int(query("RevealPeriodEpochs", 0) or 0),
        liquid_alpha_enabled=bool(query("LiquidAlphaOn", False)),
        alpha_low=alpha_low,
        alpha_high=alpha_high,
    )


def _neurons_via_storage(subtensor: Any, netuid: int) -> list[Any]:
    """uid/hotkey/permit/stake from storage, shaped like ``neurons_lite`` rows."""
    from types import SimpleNamespace

    from bittensor.core.chain_data.utils import decode_account_id

    sub = _substrate(subtensor)
    if sub is None:
        return []
    permit = list(_unwrap(sub.query("SubtensorModule", "ValidatorPermit", [netuid]).value) or [])
    rows: list[Any] = []
    for uid, hk in sub.query_map("SubtensorModule", "Keys", [netuid]):
        uid = int(uid)
        raw = hk.value[0] if hasattr(hk, "value") else hk
        try:
            hotkey = str(decode_account_id(raw))
        except Exception:  # noqa: BLE001 - an undecodable key is still a neuron
            hotkey = str(raw)
        # A failed stake read must NOT become a confident zero. It did, and the
        # preflight then told an operator "total evaluator stake 0.000 tao" on
        # a subnet whose owner hotkey held 382,305 -- which reads as "nobody
        # has staked here" and is the opposite of the truth. `None` means
        # unknown, and the summary says so rather than inventing a number.
        try:
            stake = float(_unwrap(
                sub.query("SubtensorModule", "TotalHotkeyAlpha", [hotkey, netuid]).value
            ) or 0) / 1e9
        except Exception:  # noqa: BLE001 - unknown, not zero
            stake = None
        rows.append(
            SimpleNamespace(
                uid=uid,
                hotkey=hotkey,
                validator_permit=bool(permit[uid]) if uid < len(permit) else False,
                total_stake=stake,
            )
        )
    rows.sort(key=lambda n: n.uid)
    return rows


def commit_reveal_fix_command(netuid: int) -> str:
    """The exact btcli command that enables commit-reveal weights."""
    return f"btcli sudo set --netuid {netuid} --param commit_reveal_weights_enabled --value 1"


def run_doctor(
    network: str,
    netuid: int,
    wallet_name: str | None = None,
    wallet_hotkey: str | None = None,
    probe_writes: bool = False,
    *,
    subtensor: Any = None,
    wallet: Any = None,
    block_advance_timeout_s: float = 30.0,
    reveal_timeout_s: float = 90.0,
    poll_interval_s: float = 3.0,
) -> list[CheckResult]:
    """Run every preflight check and return the results in display order.

    ``subtensor`` / ``wallet`` accept pre-built (or stub) objects; when omitted
    they are constructed from ``network`` and ``wallet_name``/``wallet_hotkey``
    via the bittensor SDK. Read-only unless ``probe_writes`` is set AND a
    wallet is available.
    """
    results: list[CheckResult] = []

    # -- connect ---------------------------------------------------------------
    if subtensor is None:
        try:
            import bittensor as bt

            subtensor = bt.Subtensor(network=network)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            results.append(
                CheckResult("chain", FAIL, f"cannot connect to {network!r}: {type(exc).__name__}: {exc}")
            )
            for name in ("subnet", "commit-reveal", "tempo", "liquid-alpha", "neurons",
                         "wallet-registered", "probe-status", "probe-reveal"):
                results.append(CheckResult(name, SKIP, "chain unreachable"))
            return results

    # -- chain reachable + block advances ---------------------------------------
    def check_chain() -> CheckResult:
        b0 = int(subtensor.get_current_block())
        deadline = time.monotonic() + block_advance_timeout_s
        while True:
            b1 = int(subtensor.get_current_block())
            if b1 > b0:
                return CheckResult(
                    "chain", PASS, f"connected to {network!r}; block advancing ({b0} -> {b1})"
                )
            if time.monotonic() >= deadline:
                return CheckResult(
                    "chain",
                    WARN,
                    f"connected to {network!r} at block {b0}, but it did not advance within "
                    f"{block_advance_timeout_s:.0f}s — the node may be stalled or syncing",
                )
            time.sleep(min(poll_interval_s, max(deadline - time.monotonic(), 0.01)))

    results.append(_guard("chain", check_chain))

    # -- netuid exists -----------------------------------------------------------
    hyper: Any = None

    def check_subnet() -> CheckResult:
        nonlocal hyper
        note = ""
        try:
            hyper = subtensor.get_subnet_hyperparameters(netuid)
        except Exception as exc:  # runtime-API drift: read the same facts from storage
            hyper = _hyperparams_via_storage(subtensor, netuid)
            if hyper is None:
                raise
            note = f" (runtime API unavailable — {type(exc).__name__}; read from storage)"
        if hyper is None:
            return CheckResult("subnet", FAIL, f"netuid {netuid} does not exist on {network!r}")
        return CheckResult("subnet", PASS, f"netuid {netuid} exists on {network!r}{note}")

    results.append(_guard("subnet", check_subnet))

    # -- hyperparameters the mechanism depends on ---------------------------------
    def check_commit_reveal() -> CheckResult:
        if hyper is None:
            return CheckResult("commit-reveal", SKIP, "netuid not found")
        if bool(hyper.commit_reveal_weights_enabled):
            return CheckResult("commit-reveal", PASS, "commit_reveal_weights_enabled is on")
        return CheckResult(
            "commit-reveal",
            FAIL,
            "commit-reveal weights is DISABLED — Epago refuses to start without it. Fix: "
            + commit_reveal_fix_command(netuid),
        )

    def check_tempo() -> CheckResult:
        if hyper is None:
            return CheckResult("tempo", SKIP, "netuid not found")
        return CheckResult(
            "tempo",
            PASS,
            f"tempo={int(hyper.tempo)} blocks; commit_reveal_period="
            f"{int(hyper.commit_reveal_period)} epochs",
        )

    def check_liquid_alpha() -> CheckResult:
        if hyper is None:
            return CheckResult("liquid-alpha", SKIP, "netuid not found")
        if bool(hyper.liquid_alpha_enabled):
            return CheckResult(
                "liquid-alpha",
                PASS,
                f"enabled (alpha_low={hyper.alpha_low}, alpha_high={hyper.alpha_high})",
            )
        return CheckResult(
            "liquid-alpha", WARN, "liquid alpha disabled — bonds use the static alpha (not fatal)"
        )

    results.append(_guard("commit-reveal", check_commit_reveal))
    results.append(_guard("tempo", check_tempo))
    results.append(_guard("liquid-alpha", check_liquid_alpha))

    # -- neuron / stake landscape ---------------------------------------------------
    neurons: list[Any] | None = None

    def check_neurons() -> CheckResult:
        nonlocal neurons
        if hyper is None:
            return CheckResult("neurons", SKIP, "netuid not found")
        try:
            neurons = list(subtensor.neurons_lite(netuid))
        except Exception:  # same runtime-API drift; storage has the facts
            rows = _neurons_via_storage(subtensor, netuid)
            if not rows:
                raise  # nothing readable from storage either: report the real failure
            neurons = rows
        if not neurons:
            return CheckResult("neurons", WARN, f"netuid {netuid} has no registered neurons yet")
        permits = [n for n in neurons if bool(n.validator_permit)]
        known = [n for n in permits if getattr(n, "total_stake", None) is not None]
        summary = f"{len(neurons)} neurons; {len(permits)} with validator permit"
        if len(known) < len(permits):
            # A missing number is a missing number.
            return CheckResult(
                "neurons",
                WARN,
                f"{summary}; stake unreadable for {len(permits) - len(known)} of them "
                "(runtime API unavailable) — check stake on taostats before relying on it",
            )
        total = sum(_stake_tao(n) for n in known)
        if permits and total == 0.0:
            # Every permitted validator reading as exactly zero on a live
            # subnet is a decode failure, not a fact. `TotalHotkeyAlpha` comes
            # back as `[0]` under the runtime drift this module already routes
            # around, and reporting that as "0.000 tao" told an operator the
            # subnet was unstaked while its owner hotkey held 382,305.
            return CheckResult(
                "neurons",
                WARN,
                f"{summary}; stake reads as zero for every one of them, which on a "
                "live subnet means the runtime API drift rather than an unstaked "
                "subnet — read stake from taostats, not from here",
            )
        return CheckResult(
            "neurons",
            PASS,
            f"{summary}; total evaluator stake {total:,.3f} tao",
        )

    results.append(_guard("neurons", check_neurons))

    # -- wallet -----------------------------------------------------------------------
    want_wallet = wallet is not None or wallet_name is not None
    hotkey_ss58: str | None = None
    my_neuron: Any = None

    if not want_wallet:
        results.append(
            CheckResult("wallet-registered", SKIP, "no wallet supplied (--wallet-name)")
        )
    else:

        def check_registered() -> CheckResult:
            nonlocal wallet, hotkey_ss58, my_neuron
            if wallet is None:
                import bittensor as bt

                wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey or "default")
            hotkey_ss58 = str(wallet.hotkey.ss58_address)
            if neurons is None:
                return CheckResult("wallet-registered", SKIP, "neuron list unavailable")
            for n in neurons:
                if str(n.hotkey) == hotkey_ss58:
                    my_neuron = n
                    return CheckResult(
                        "wallet-registered", PASS, f"hotkey {hotkey_ss58} registered as uid {int(n.uid)}"
                    )
            return CheckResult(
                "wallet-registered",
                FAIL,
                f"hotkey {hotkey_ss58} is NOT registered on netuid {netuid} — fix: "
                f"btcli subnets register --netuid {netuid} "
                f"--wallet.name {wallet_name or getattr(wallet, 'name', 'default')} "
                f"--wallet.hotkey {wallet_hotkey or 'default'}",
            )

        results.append(_guard("wallet-registered", check_registered))

        def check_permit() -> CheckResult:
            if my_neuron is None:
                return CheckResult("wallet-permit", SKIP, "hotkey not registered")
            if bool(my_neuron.validator_permit):
                return CheckResult("wallet-permit", PASS, f"uid {int(my_neuron.uid)} holds a validator permit")
            return CheckResult(
                "wallet-permit",
                WARN,
                "no validator permit — ev3 verdicts from this hotkey are ignored by quorum; "
                "add stake and wait for the next epoch",
            )

        def check_stake() -> CheckResult:
            if my_neuron is None:
                return CheckResult("wallet-stake", SKIP, "hotkey not registered")
            return CheckResult(
                "wallet-stake", PASS, f"total stake {_stake_tao(my_neuron):.3f} tao"
            )

        results.append(_guard("wallet-permit", check_permit))
        results.append(_guard("wallet-stake", check_stake))

    # -- write probes -------------------------------------------------------------------
    if not probe_writes:
        results.append(CheckResult("probe-status", SKIP, "write probes disabled (--probe-writes)"))
        results.append(CheckResult("probe-reveal", SKIP, "write probes disabled (--probe-writes)"))
        return results
    if wallet is None or hotkey_ss58 is None:
        detail = "write probes need a wallet (--wallet-name)"
        results.append(CheckResult("probe-status", SKIP, detail))
        results.append(CheckResult("probe-reveal", SKIP, detail))
        return results

    def probe_status() -> CheckResult:
        payload = f"es1|doctor|{int(time.time())}"
        assert len(payload.encode()) <= 128
        subtensor.set_commitment(wallet=wallet, netuid=netuid, data=payload, raise_error=True)
        readback = (subtensor.get_all_commitments(netuid) or {}).get(hotkey_ss58)
        if readback == payload:
            return CheckResult(
                "probe-status", PASS, "128-byte plaintext status slot round-trips (es1 write + readback)"
            )
        return CheckResult(
            "probe-status",
            FAIL,
            f"wrote {payload!r} but read back {readback!r} — the status slot is not round-tripping",
        )

    results.append(_guard("probe-status", probe_status))

    def probe_reveal() -> CheckResult:
        payload = f"epago-doctor-reveal|{int(time.time())}"
        start_block = int(subtensor.get_current_block())
        started = time.monotonic()
        subtensor.set_reveal_commitment(
            wallet=wallet,
            netuid=netuid,
            data=payload,
            blocks_until_reveal=2,
            raise_error=True,
        )
        from epago.chain.client import _decode_revealed_commitment

        def _reveals_for_hotkey() -> list[tuple[int, str]]:
            """(block, payload) entries for our hotkey. Prefer the SDK method; fall back
            to a raw-storage decode when the SDK decoder crashes on this runtime's mixed
            hex/raw commitment encoding (see BittensorChainClient.read_revealed_payloads)."""
            try:
                revealed = subtensor.get_all_revealed_commitments(netuid) or {}
                return [(int(b), str(d)) for b, d in revealed.get(hotkey_ss58, ())]
            except Exception:  # noqa: BLE001 - fall back to raw decode below
                pass
            out: list[tuple[int, str]] = []
            try:
                qmap = subtensor.substrate.query_map(
                    module="Commitments", storage_function="RevealedCommitments", params=[netuid]
                )
                for key, value in qmap:
                    if str(getattr(key, "value", key)) != hotkey_ss58:
                        continue
                    for entry in getattr(value, "value", value) or []:
                        try:
                            out.append((int(entry[1]), _decode_revealed_commitment(entry[0])))
                        except Exception:  # noqa: BLE001 - skip undecodable neighbours
                            continue
            except Exception:  # noqa: BLE001 - a read failure just retries until deadline
                pass
            return out

        deadline = started + reveal_timeout_s
        while True:
            found_block = next((b for b, d in _reveals_for_hotkey() if d == payload), None)
            if found_block is not None:
                elapsed = time.monotonic() - started
                return CheckResult(
                    "probe-reveal",
                    PASS,
                    f"timelock channel round-trips: revealed at block {found_block} "
                    f"(+{found_block - start_block} blocks, {elapsed:.1f}s after submit)",
                )
            if time.monotonic() >= deadline:
                return CheckResult(
                    "probe-reveal",
                    WARN,
                    f"reveal not visible within {reveal_timeout_s:.0f}s (submitted at block "
                    f"{start_block}, blocks_until_reveal=2) — the timelock decryption pipeline "
                    "can lag; keep watching with: epago chain watch-reveals "
                    f"--network {network} --netuid {netuid}",
                )
            time.sleep(min(poll_interval_s, max(deadline - time.monotonic(), 0.01)))

    results.append(_guard("probe-reveal", probe_reveal))
    return results
