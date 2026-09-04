#!/usr/bin/env python
"""External auditor: replay a duel verdict from its published audit record.

Anyone — not only validators — can verify a verdict after the fact. Given an
audit record JSON, the pinned public corpus snapshot and (optionally, after
the rotation window) a published private pool file, this tool re-derives every
deterministic quantity the validator committed to and reports an explicit
PASS/FAIL per check:

  record        required AuditRecord fields are present
  corpus        corpus snapshot digest matches the record's pin
  seeds         public/bootstrap seeds re-derive from the recorded block hash
  tasks         public tasks regenerate from the recorded seed and their
                ids digest matches (needs epago.taskgen installed)
  lcb           the bootstrap LCB recomputes from recorded per-task diffs
                (extra.public_diffs, [[task_id, d], ...])
  audit16       the on-chain audit digest matches the canonical record hash
  signature     the validator's sr25519 signature verifies against its hotkey
                over the canonical-unsigned record digest (needs bittensor)
  pool          the published private pool file matches the recorded digest
  chain         a revealed on-chain ev3 verdict carries this record's audit16
                (needs --network/--netuid and chain access)

Checks whose inputs were not supplied (or whose module is not present in this
build) report SKIP, never silently pass. Exit code 0 iff no check FAILed.

Runs without torch/vllm; bittensor is only needed for the signature and
on-chain checks, which SKIP cleanly when it is absent or the chain is
unreachable.

WHAT THIS TOOL DOES NOT DO — and no tool can. It does not re-run the models.
The ``lcb`` check recomputes the bootstrap from the per-task difference vector
the validator RECORDED (``extra.public_diffs``); it proves the arithmetic and
the seed, not that those diffs describe real rollouts. Checking the diffs
themselves means re-scoring both checkpoints on the regenerated task set and
comparing distributions, because GPU inference is not bit-reproducible: on the
reference stack the same checkpoint re-scored on the same 400 tasks disagrees
with itself on ~21% of them (paired score-gap SE ~0.030 at n=128), on one card
as much as on eight. So the boundary is: everything this tool reports is exact
and binding; the honesty of the scores is a separate statistical check against
the calibrated noise floor, where wholesale fabrication is detectable and a
flipped task or two is neither detectable nor large enough to move a verdict
that cleared delta with margin.

Usage:
    .venv/bin/python scripts/replay_verdict.py AUDIT_RECORD.json \\
        --corpus corpus.db [--private-pool pool.json] \\
        [--verdict "ev3|..."] [--audit16 <16 hex>] \\
        [--network finney --netuid N]
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago import constants  # noqa: E402
from epago.core.stats import (  # noqa: E402
    bootstrap_lcb,
    bootstrap_seed,
    public_task_seed,
    round_bootstrap_seed,
    round_public_seed,
)
from epago.core.types import AuditRecord  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

_LCB_TOL = 1e-9  # bootstrap is bit-deterministic given the seed; exact up to float noise


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""


def _canonical_json(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _unsigned_digest(record: dict) -> str:
    """The binding record digest: canonical JSON with validator_signature zeroed.

    Mirrors :func:`epago.validator.audit.record_digest` — the signature is set
    to the empty string (not removed) before hashing, so the digest covers the
    exact document the validator signed.
    """
    return _sha256_hex(_canonical_json({**record, "validator_signature": ""}))


def _canonical_digests(record: dict) -> dict[str, str]:
    """Canonicalizations of the record's binding digest, preferred first.

    The authoritative convention is :func:`epago.validator.audit.record_digest`
    (canonical JSON with ``validator_signature`` zeroed). Fallback variants are
    tried and labeled so a mismatch report says exactly what was compared.
    """
    out: dict[str, str] = {}
    try:
        from epago.core.types import AuditRecord as _AR
        from epago.validator.audit import record_digest as _record_digest

        known = {f.name for f in fields(_AR)}
        if set(record) <= known and not (known - set(record)):
            out["epago.validator.audit.record_digest"] = _record_digest(_AR(**record))
    except ImportError:
        pass
    out.setdefault("canonical-json signature zeroed", _unsigned_digest(record))
    out["canonical-json minus signature"] = _sha256_hex(
        _canonical_json({k: v for k, v in record.items() if k != "validator_signature"})
    )
    out["canonical-json full record"] = _sha256_hex(_canonical_json(record))
    return out


def _bare(digest: str) -> str:
    return digest.split(":", 1)[-1].lower()


def _seed_forms(seed: int) -> set[str]:
    return {str(seed), f"{seed:x}", f"{seed:016x}", hex(seed)}


def _check_record(record: dict) -> Check:
    required = {f.name for f in fields(AuditRecord)} - {"validator_signature", "extra"}
    missing = sorted(required - set(record))
    if missing:
        return Check("record", "FAIL", f"missing fields: {', '.join(missing)}")
    return Check("record", "PASS", f"all {len(required)} required fields present")


def _check_corpus(record: dict, corpus: Optional[Path]) -> Check:
    if corpus is None:
        return Check("corpus", "SKIP", "no --corpus supplied")
    from epago.environment.sync import corpus_digest

    actual = corpus_digest(corpus)
    if actual == record["corpus_digest"]:
        return Check("corpus", "PASS", actual)
    return Check("corpus", "FAIL", f"expected {record['corpus_digest']}, got {actual}")


def _seed_candidates(record: dict) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Every derivation a verdict could legitimately have used, preferred first.

    A **round** keys its exam on the round number and each entrant's bootstrap
    on that entrant's digest (:func:`epago.core.stats.round_public_seed` /
    :func:`~epago.core.stats.round_bootstrap_seed`), so the whole field answers
    one exam. A **solo duel** keys both on the author hotkey. This auditor knew
    only the solo forms, so every verdict the shipped validator produces — it
    runs rounds — failed the seeds check, and the tasks check with it, for
    reasons that had nothing to do with the verdict being wrong.
    """
    block_hash = record["block_hash_at_reveal"]
    extra = record.get("extra") or {}
    pub: list[tuple[str, int]] = []
    boot: list[tuple[str, int]] = []
    round_no = extra.get("round")
    if round_no:
        pub.append(("round", round_public_seed(block_hash, int(round_no))))
        digest = record.get("challenger_digest")
        if digest:
            boot.append(("round", round_bootstrap_seed(block_hash, str(digest))))
    for hotkey_label in ("author_hotkey", "validator_hotkey"):
        hotkey = record.get(hotkey_label)
        if hotkey:
            pub.append((hotkey_label, public_task_seed(block_hash, hotkey)))
            boot.append((hotkey_label, bootstrap_seed(block_hash, hotkey)))
    return pub, boot


def _check_seeds(record: dict) -> tuple[Check, Optional[int], Optional[int]]:
    pub_candidates, boot_candidates = _seed_candidates(record)
    details = []
    ok = True
    pub_seed_int = boot_seed_int = None
    for label, recorded, candidates in (
        ("public_seed", str(record["public_seed"]), pub_candidates),
        ("boot_seed", str(record["boot_seed"]), boot_candidates),
    ):
        matched = None
        for how, seed in candidates:
            if recorded in _seed_forms(seed):
                matched = (how, seed)
                break
        if matched is None:
            ok = False
            details.append(f"{label} {recorded!r} does not re-derive from the block hash")
        else:
            details.append(f"{label} re-derives via {matched[0]}")
            if label == "public_seed":
                pub_seed_int = matched[1]
            else:
                boot_seed_int = matched[1]
    return Check("seeds", "PASS" if ok else "FAIL", "; ".join(details)), pub_seed_int, boot_seed_int


def _check_tasks(record: dict, corpus: Optional[Path], pub_seed: Optional[int]) -> Check:
    if corpus is None:
        return Check("tasks", "SKIP", "no --corpus supplied")
    try:
        from epago.taskgen.generator import generate_tasks, task_ids_digest
    except ImportError:
        return Check("tasks", "SKIP", "epago.taskgen not present in this build")
    if pub_seed is None:
        recorded = str(record["public_seed"])
        try:
            pub_seed = int(recorded, 16)
        except ValueError:
            try:
                pub_seed = int(recorded)
            except ValueError:
                return Check("tasks", "FAIL", f"unparseable public_seed {recorded!r}")
    from epago.environment.corpus import SqliteCorpus

    store = SqliteCorpus(corpus)
    try:
        # The exam size is a property of the verdict, not of the auditor's
        # environment: ``extra.public_diffs`` carries exactly one entry per
        # public task, so read it from the record and fall back to the local
        # constant only when a record predates that field. Using the auditor's
        # own N_PUB_TASKS made a replay fail whenever the validator ran a
        # different exam size, which reads as a forged verdict rather than a
        # mismatched knob.
        n_public = len(_diff_values((record.get("extra") or {}).get("public_diffs") or []))
        from epago.taskgen.sealed_pool import is_sealed_release

        if is_sealed_release(str(record["taskgen_release"])):
            # A sealed-pool exam is not regenerable from a seed: the questions
            # were worded by a model. It is still checkable, because selection
            # runs over the pool's task-id manifest, whose digest was committed
            # before the round, and because the tasks a round asked are
            # published in full once the disclosure delay elapses.
            #
            # Only the manifest is needed to prove selection was honest: the
            # ids alone reproduce the draw. The round file is what supplies the
            # question and answer text for re-grading.
            #
            # That is a real reduction in independence, stated rather than
            # hidden: without those files this check SKIPs, and a skip is never
            # reported as a pass.
            tasks = _sealed_pool_tasks(record, pub_seed, n_public or constants.N_PUB_TASKS)
            if isinstance(tasks, Check):
                return tasks
        else:
            tasks = generate_tasks(
                seed=pub_seed,
                release=record["taskgen_release"],
                corpus=store,
                n=n_public or constants.N_PUB_TASKS,
            )
    except Exception as exc:  # noqa: BLE001 - starved generation etc. is a finding
        return Check("tasks", "FAIL", f"regeneration failed: {type(exc).__name__}: {exc}")
    finally:
        store.close()
    actual = task_ids_digest(tasks)
    if _bare(actual) == _bare(str(record["public_task_ids_digest"])):
        return Check("tasks", "PASS", f"{len(tasks)} public tasks regenerate; ids digest matches")
    return Check(
        "tasks",
        "FAIL",
        f"regenerated {len(tasks)} tasks but digest {actual} != recorded "
        f"{record['public_task_ids_digest']} (check EPAGO_N_PUB_TASKS matches the release)",
    )



def _round_ids(path: Path) -> tuple[int, set[str]]:
    """A released round file's round number and the ids it published."""
    from epago.taskgen.sealed_pool import load_round_file

    data = json.loads(path.read_text())
    return int(data.get("round", -1)), {t.task_id for t in load_round_file(path)}


def _sealed_pool_tasks(record: dict, pub_seed: int, n_public: int):
    """Rebuild a sealed-pool round's tasks, or return a Check saying why not.

    The honest-selection proof runs over ids only. An auditor loads the
    manifest (digest pinned by the verdict), reconstructs which tasks earlier
    rounds already retired, redraws this round from the remainder, and gets the
    exact id set the validator must have asked. Nothing in that chain requires
    trusting the validator: the manifest is pinned before the round, and the
    exclusion set comes from round files each pinned by its own verdict.

    The round file then supplies the text to re-grade. Its ids are checked
    against the redrawn set, so a validator cannot publish a gentler exam than
    the one selection actually chose.
    """
    from epago.taskgen.sealed_pool import (
        SealedPoolError,
        load_manifest,
        load_round_file,
        select_ids,
    )

    manifest_path = os.environ.get("EPAGO_PUBLIC_POOL_MANIFEST", "").strip()
    rounds_dir = os.environ.get("EPAGO_PUBLIC_ROUNDS", "").strip()
    if not manifest_path or not rounds_dir:
        return Check(
            "tasks",
            "SKIP",
            f"release {record['taskgen_release']} is served from a sealed pool; set "
            "EPAGO_PUBLIC_POOL_MANIFEST to the pool's task-id manifest and "
            "EPAGO_PUBLIC_ROUNDS to the directory of released round files to check it",
        )

    this_round = int((record.get("extra") or {}).get("round", -1))
    try:
        manifest = load_manifest(
            manifest_path, str(record.get("public_pool_manifest_digest", ""))
        )
        # Every round strictly before this one retired its tasks from the pool,
        # so they were not eligible to be drawn again here.
        served: set[str] = set()
        this_round_file: Path | None = None
        for path in sorted(Path(rounds_dir).glob("*.json")):
            try:
                round_no, ids = _round_ids(path)
            except (SealedPoolError, ValueError, KeyError):
                continue  # not a round file; the directory holds other releases
            if round_no < this_round:
                served |= ids
            elif round_no == this_round:
                this_round_file = path

        # Checked before redrawing, not after. Redrawing first would subtract
        # every earlier round from the pool and report exhaustion, which reads
        # as a protocol failure when the truth is simply that this round's
        # disclosure delay has not elapsed yet.
        if this_round_file is None:
            return Check(
                "tasks",
                "SKIP",
                f"round {this_round} has not been released yet (public tasks publish "
                f"{constants.AUDIT_PUBLISH_DELAY_BLOCKS} blocks after the round); "
                "it cannot be re-graded until it does",
            )
        expected = select_ids(manifest.task_ids, pub_seed, n_public, exclude=served)
    except SealedPoolError as exc:
        return Check("tasks", "FAIL", str(exc))

    published = load_round_file(this_round_file)
    if {t.task_id for t in published} != set(expected):
        return Check(
            "tasks",
            "FAIL",
            f"{this_round_file.name} publishes a different task set than selection "
            "draws from the committed manifest",
        )
    return list(published)

def _diff_values(raw: list) -> list[int]:
    """Accept both the pair form ``[[task_id, d], ...]`` and bare ``[d, ...]``."""
    out: list[int] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append(int(item[1]))
        else:
            out.append(int(item))
    return out


def _public_task_ids(extra: dict) -> set[str]:
    """Task ids carried by the pair-form ``extra.public_diffs``."""
    ids: set[str] = set()
    for item in extra.get("public_diffs") or ():
        if isinstance(item, (list, tuple)) and len(item) == 2:
            ids.add(str(item[0]))
    return ids


def _check_lcb(record: dict, boot_seed: Optional[int]) -> Check:
    extra = record.get("extra") or {}
    diffs = None
    for key in ("public_diffs", "diffs_pub", "diffs"):
        diffs = extra.get(key, record.get(key))
        if diffs is not None:
            break
    if not diffs:
        return Check("lcb", "SKIP", "no per-task diffs recorded (extra.public_diffs absent)")
    if boot_seed is None:
        recorded = str(record["boot_seed"])
        try:
            boot_seed = int(recorded, 16)
        except ValueError:
            try:
                boot_seed = int(recorded)
            except ValueError:
                return Check("lcb", "FAIL", f"unparseable boot_seed {recorded!r}")
    diffs = _diff_values(diffs)
    mu = sum(diffs) / len(diffs)
    details = []
    ok = True
    if abs(mu - float(record["mu_hat_pub"])) > 1e-9:
        ok = False
        details.append(f"mean(diffs)={mu:+.6f} != recorded mu_hat_pub={record['mu_hat_pub']:+.6f}")
    lcb = bootstrap_lcb(tuple(diffs), boot_seed)
    if abs(lcb - float(record["lcb_pub"])) > _LCB_TOL:
        ok = False
        details.append(
            f"recomputed lcb={lcb:+.6f} != recorded {float(record['lcb_pub']):+.6f} "
            f"(B={constants.BOOTSTRAP_B}, alpha={constants.EVAL_ALPHA}; ensure EPAGO_BOOTSTRAP_B "
            "matches the release the validator ran)"
        )
    if ok:
        details.append(f"lcb recomputes to {lcb:+.6f} from {len(diffs)} diffs")
    return Check("lcb", "PASS" if ok else "FAIL", "; ".join(details))


def _check_audit16(record: dict, verdict: Optional[str], audit16: Optional[str]) -> Check:
    expected = None
    if verdict is not None:
        from epago.core.reveal import WireFormatError, parse_verdict

        try:
            expected = parse_verdict(verdict, validator_hotkey="", block=0).audit_digest
        except WireFormatError as exc:
            return Check("audit16", "FAIL", f"unparseable --verdict: {exc}")
    elif audit16 is not None:
        expected = audit16.lower()
    candidates = _canonical_digests(record)
    if expected is None:
        first_label, first_digest = next(iter(candidates.items()))
        return Check(
            "audit16",
            "SKIP",
            f"no --verdict/--audit16 to compare against; computed {first_digest[:16]} "
            f"({first_label})",
        )
    for label, digest in candidates.items():
        if digest[:16] == expected:
            return Check("audit16", "PASS", f"matches via {label}")
    return Check(
        "audit16",
        "FAIL",
        f"on-chain audit16 {expected} matches no canonicalization of this record "
        f"(computed {_unsigned_digest(record)[:16]} with the signature zeroed)",
    )


def _check_pool(record: dict, pool: Optional[Path]) -> Check:
    if pool is None:
        return Check("pool", "SKIP", "no --private-pool supplied (publishes after rotation)")
    from epago.model.store import file_sha256

    actual = file_sha256(pool)
    recorded = _bare(str(record["private_pool_digest"]))
    if actual == recorded:
        return Check("pool", "PASS", f"sha256:{actual}")
    return Check("pool", "FAIL", f"expected sha256:{recorded}, got sha256:{actual}")


def _check_signature(record: dict) -> Check:
    """Verify the validator's sr25519 signature over the canonical-unsigned digest."""
    sig = str(record.get("validator_signature") or "")
    hotkey = str(record.get("validator_hotkey") or "")
    if not sig:
        return Check("signature", "SKIP", "record is unsigned (validator_signature empty)")
    if not hotkey:
        return Check("signature", "FAIL", "signed record carries no validator_hotkey")
    try:
        import bittensor as bt

        # bittensor v11 removed the top-level Keypair; the low-level type moved
        # to bittensor.sp_core. Fall back so post-hoc verification keeps working.
        Keypair = getattr(bt, "Keypair", None) or bt.sp_core.Keypair
    except ImportError:
        return Check(
            "signature", "SKIP", "bittensor not installed — cannot verify the sr25519 signature"
        )
    try:
        sig_bytes = bytes.fromhex(sig.removeprefix("0x"))
    except ValueError:
        return Check("signature", "FAIL", f"validator_signature is not hex: {sig[:32]!r}…")
    try:
        keypair = Keypair(ss58_address=hotkey)
    except Exception as exc:  # noqa: BLE001 - malformed ss58 is a finding
        return Check("signature", "FAIL", f"invalid validator_hotkey ss58 {hotkey!r}: {exc}")
    digest = _unsigned_digest(record)
    candidates = (
        ("digest hex string", digest.encode()),
        ("digest bytes", bytes.fromhex(digest)),
        ("canonical-unsigned json", _canonical_json({**record, "validator_signature": ""}).encode()),
    )
    for label, message in candidates:
        try:
            if keypair.verify(message, sig_bytes):
                return Check(
                    "signature", "PASS", f"sr25519 signature by {hotkey} verifies over the {label}"
                )
        except Exception:  # noqa: BLE001 - a candidate form failing to verify is not fatal
            continue
    return Check(
        "signature",
        "FAIL",
        f"signature does not verify against {hotkey} over the canonical-unsigned "
        f"record digest {digest[:16]}…",
    )


def _check_chain(record: dict, network: Optional[str], netuid: Optional[int]) -> Check:
    """Cross-check: a revealed on-chain ev3 verdict must carry this record's audit16."""
    if netuid is None:
        return Check("chain", "SKIP", "no --netuid supplied (offline replay)")
    try:
        import bittensor as bt
    except ImportError:
        return Check("chain", "SKIP", "bittensor not installed — on-chain cross-check skipped")
    expected = {digest[:16] for digest in _canonical_digests(record).values()}
    entries: list[tuple[str, int, str]] = []
    try:
        subtensor = bt.Subtensor(network=network or "finney")
        try:
            # Preferred SDK path (works wherever the decoder is sound).
            revealed = subtensor.get_all_revealed_commitments(netuid) or {}
            for hk, evs in revealed.items():
                for b, p in evs:
                    entries.append((str(hk), int(b), str(p)))
        except Exception:  # noqa: BLE001 - SDK decoder crashes on this runtime; raw fallback
            from epago.chain.client import _decode_revealed_commitment

            qmap = subtensor.substrate.query_map(
                module="Commitments", storage_function="RevealedCommitments", params=[netuid]
            )
            for key, value in qmap:
                hk = str(getattr(key, "value", key))
                for e in getattr(value, "value", value) or []:
                    try:
                        entries.append((hk, int(e[1]), _decode_revealed_commitment(e[0])))
                    except Exception:  # noqa: BLE001 - skip undecodable neighbours
                        continue
    except Exception as exc:  # noqa: BLE001 - offline replay must degrade to SKIP
        return Check(
            "chain",
            "SKIP",
            f"chain unreachable ({type(exc).__name__}: {exc}) — cross-check skipped",
        )
    from epago.core.reveal import WireFormatError, parse_verdict

    hotkeys_seen = {hk for hk, _, _ in entries}
    scanned = 0
    for hotkey, block, payload in entries:
        if not str(payload).startswith(constants.VERDICT_VERSION + "|"):
            continue
        scanned += 1
        try:
            v = parse_verdict(str(payload), validator_hotkey=str(hotkey), block=int(block))
        except WireFormatError:
            continue
        if v.audit_digest not in expected:
            continue
        recorded_hotkey = str(record.get("validator_hotkey") or "")
        if recorded_hotkey and str(hotkey) != recorded_hotkey:
            return Check(
                "chain",
                "FAIL",
                f"audit16 {v.audit_digest} was revealed by {hotkey} at block {v.block}, "
                f"but the record claims validator_hotkey {recorded_hotkey}",
            )
        return Check(
            "chain",
            "PASS",
            f"ev3 verdict with audit16 {v.audit_digest} revealed by {hotkey} "
            f"at chain block {v.block} (decision {v.decision.value})",
        )
    return Check(
        "chain",
        "FAIL",
        f"no revealed ev3 verdict on netuid {netuid} ({len(hotkeys_seen)} hotkeys, "
        f"{scanned} verdicts scanned) carries this record's audit16",
    )


@app.command()
def main(
    audit_record: Path = typer.Argument(..., help="published audit record JSON"),
    corpus: Optional[Path] = typer.Option(None, help="pinned public corpus snapshot (corpus.db)"),
    private_pool: Optional[Path] = typer.Option(
        None, help="published private pool file (available after rotation)"
    ),
    verdict: Optional[str] = typer.Option(
        None, help="the on-chain ev3 verdict string to check audit16 against"
    ),
    audit16: Optional[str] = typer.Option(
        None, help="expected 16-hex audit digest (alternative to --verdict)"
    ),
    network: str = typer.Option(
        "finney", help="subtensor network for the on-chain cross-check (with --netuid)"
    ),
    netuid: Optional[int] = typer.Option(
        None, help="netuid to cross-check the revealed ev3 verdict on (omit for offline replay)"
    ),
) -> None:
    """Replay every deterministic quantity in AUDIT_RECORD and report PASS/FAIL."""
    try:
        record = json.loads(audit_record.expanduser().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(f"error: could not read audit record: {exc}", err=True)
        raise typer.Exit(code=1)

    checks: list[Check] = [_check_record(record)]
    if checks[0].status == "FAIL":
        pub_seed = boot_seed = None
    else:
        seed_check, pub_seed, boot_seed = _check_seeds(record)
        checks.append(_check_corpus(record, corpus))
        checks.append(seed_check)
        checks.append(_check_tasks(record, corpus, pub_seed))
        checks.append(_check_lcb(record, boot_seed))
        checks.append(_check_audit16(record, verdict, audit16))
        checks.append(_check_signature(record))
        checks.append(_check_pool(record, private_pool))
        checks.append(_check_chain(record, network, netuid))

    typer.echo("")
    typer.echo(f"replay of {audit_record.name} "
               f"(challenger {record.get('challenger_digest', '?')}, "
               f"accepted={record.get('accepted', '?')}):")
    width = max(len(c.name) for c in checks)
    failed = 0
    for c in checks:
        typer.echo(f"  {c.status:<4}  {c.name:<{width}}  {c.detail}")
        if c.status == "FAIL":
            failed += 1
    typer.echo("")
    if failed:
        typer.echo(f"REPLAY FAIL: {failed} check(s) failed — this verdict does not replay")
        raise typer.Exit(code=1)
    typer.echo("REPLAY PASS: every supplied input replays to the committed verdict")


if __name__ == "__main__":
    app()
