#!/usr/bin/env python
"""N-iteration adversarial soak of the validator loop on a mock chain.

Drives ``ValidatorService.tick`` with :class:`MockChainClient` plus the same
style of fakes the validator tests use: every iteration a cast of synthetic
miners submits — an improver (should be accepted and crowned), an exact
byte-copy of the king's weights (intake rejection), a degenerate config-lock
violator (intake rejection), and a near-miss. After each round the soak
asserts state invariants:

  * the duel queue drains,
  * failure memory grows monotonically and only for real failures,
  * the king lineage stays consistent (the king only ever changes to that
    round's improver digest, and at least one coronation happens),
  * audit records parse as AuditRecord and their audit16 digests match the
    on-chain ev3 verdict commitments.

Artifacts (final state.json, audit log, per-iteration report) are written
under ``docs/soak-artifacts/``. Runs entirely without torch/vllm/bittensor.

Exit codes: 0 pass (or SKIP when epago.validator.service is not present in
this build), 1 invariant failure.

Usage: .venv/bin/python scripts/sandbox_soak.py [--iterations 10]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path

# Small, fast mechanism numbers; must be set before epago imports.
os.environ.setdefault("EPAGO_N_PUB_TASKS", "200")
os.environ.setdefault("EPAGO_N_PRIV_TASKS", "48")
os.environ.setdefault("EPAGO_BOOTSTRAP_B", "1000")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from epago.chain.client import MockChainClient, NeuronView  # noqa: E402
from epago.config import load_config  # noqa: E402
from epago.core.reveal import build_reveal, build_round_start  # noqa: E402
from epago import constants
from epago.core.stats import adaptive_delta, bootstrap_lcb, bootstrap_seed, paired_half, public_task_seed  # noqa: E402
from epago.core.types import AuditRecord, DuelOutcome, ModelRef, Task, TaskOrigin  # noqa: E402
from epago.model.store import snapshot_digest  # noqa: E402

TICKS_PER_ITERATION = 6  # one competition per iteration, plus slack for coronation
ROUND_AUTHORITY_HK = "5RoundAuthority"


# --- synthetic model zoo ------------------------------------------------------

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


class ModelZoo:
    """Fake content-addressed model store: digest -> local dir, dir -> skill."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.by_digest: dict[str, Path] = {}
        self.skill_by_dir: dict[str, float] = {}
        self._n = 0

    def mint(
        self,
        skill: float,
        *,
        config_overrides: dict | None = None,
        copy_weights_of: Path | None = None,
    ) -> tuple[str, Path]:
        """Create a tiny fake model dir; returns (digest, dir)."""
        self._n += 1
        d = self.root / f"model-{self._n:04d}"
        d.mkdir(parents=True)
        config = dict(_BASE_CONFIG)
        config.update(config_overrides or {})
        (d / "config.json").write_text(json.dumps(config, sort_keys=True))
        if copy_weights_of is not None:
            # Byte-identical shards (the exact-copy intake case) but a distinct
            # snapshot digest, via an allowed auxiliary file.
            shutil.copy2(copy_weights_of / "model.safetensors", d / "model.safetensors")
            (d / "provenance.txt").write_text(f"shameless copy #{self._n}")
        else:
            (d / "model.safetensors").write_bytes(
                f"soak-weights-{self._n}-skill-{skill}".encode().ljust(2048, b"\0")
            )
        digest = snapshot_digest(d)
        self.by_digest[digest] = d
        self.skill_by_dir[str(d.resolve())] = skill
        return digest, d

    def alias(self, digest: str, target: Path) -> None:
        self.by_digest[digest] = target

    def skill_of_digest(self, digest: str) -> float:
        return self.skill_by_dir[str(self.by_digest[digest].resolve())]

    def materialize(self, ref: ModelRef, cache_dir: Path) -> Path:
        try:
            return self.by_digest[ref.digest]
        except KeyError:
            raise RuntimeError(f"soak zoo has no model for {ref.digest}") from None


# --- fake eval / taskgen dependencies -------------------------------------------


def _make_tasks(tag: str, n: int) -> list[Task]:
    return [
        Task(
            task_id=f"soak-{tag}-{i:03d}",
            question=f"What is item {i} of {tag}?",
            answer=f"ans-{tag}-{i:03d}",
            aliases=(),
            evidence_doc_ids=(f"fx-{i:05d}",),
            masked_doc_ids=(),
            origin=TaskOrigin.GENERATED_PUBLIC,
            template="soak",
            hops=1,
        )
        for i in range(n)
    ]


def fake_generate_tasks(*, seed: int, release: str, corpus, n: int, king_probe=None) -> list[Task]:
    return _make_tasks(f"pub{seed & 0xFFFFFFFF:08x}", n)


class FakePrivatePool:
    epoch = 1
    digest = "sha256:" + "77" * 32

    def sample(self, n: int, seed: int) -> list[Task]:
        return _make_tasks(f"prv{seed & 0xFFFFFFFF:08x}", n)

    def rotation_due(self, current_block: int) -> bool:
        return False

    def rotate(self, current_block: int) -> str | None:
        return None


def _solves(skill: float, index: int) -> bool:
    """Deterministic nested skill model: an evenly spread per-task difficulty
    grid, so a higher-skill model solves a strict superset of a lower one."""
    return ((index * 37) % 96) / 96.0 < skill


def make_fake_run_duel(zoo: ModelZoo):
    """Skill-based duel double computing the real paired statistics."""

    def run_duel(spec, env, backend_factory, llm_judge) -> DuelOutcome:
        king_skill = zoo.skill_by_dir[str(Path(spec.king_dir).resolve())]
        chall_skill = zoo.skill_by_dir[str(Path(spec.challenger_dir).resolve())]
        halves = []
        for tasks in (spec.public_tasks, spec.private_tasks):
            king_correct = [_solves(king_skill, i) for i in range(len(tasks))]
            chall_correct = [_solves(chall_skill, i) for i in range(len(tasks))]
            halves.append(paired_half(king_correct, chall_correct))
        public, private = halves
        boot_seed = bootstrap_seed(spec.block_hash_at_reveal, spec.author_hotkey)
        pub_seed = public_task_seed(spec.block_hash_at_reveal, spec.author_hotkey)
        lcb_pub = bootstrap_lcb(public.diffs, boot_seed)
        delta = adaptive_delta(spec.king_acc_ema, spec.noise_floor)
        return DuelOutcome(
            public=public,
            private=private,
            lcb_pub=lcb_pub,
            delta=delta,
            accepted=lcb_pub > delta and private.mu_hat > 0.0,
            boot_seed_hex=f"{boot_seed:016x}",
            public_seed_hex=f"{pub_seed:016x}",
        )

    return run_duel


# --- invariants ----------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.skips: list[str] = []

    def check(self, ok: bool, message: str) -> None:
        if not ok:
            self.failures.append(message)
            print(f"  INVARIANT FAIL: {message}")

    def skip(self, message: str) -> None:
        if message not in self.skips:
            self.skips.append(message)


def check_audit_log(state_dir: Path, chain: MockChainClient, report: Report) -> int:
    """Every audit record parses as AuditRecord, digest-verifies, and matches
    an on-chain ev3 commitment."""
    from epago.validator.audit import audit16

    log_path = state_dir / "audit" / "audit.jsonl"
    if not log_path.exists():
        report.skip("no audit log written yet")
        return 0
    known = {f.name for f in fields(AuditRecord)}
    onchain = {v.audit_digest for v in chain.read_verdicts()}
    n = 0
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        n += 1
        rec = json.loads(line)
        missing, unknown = known - set(rec), set(rec) - known
        report.check(not missing, f"audit record missing fields: {sorted(missing)}")
        report.check(not unknown, f"audit record has unknown fields: {sorted(unknown)}")
        if missing or unknown:
            continue
        record = AuditRecord(**rec)
        digest16 = audit16(record)
        report.check(
            digest16 in onchain,
            f"audit16 {digest16} for {record.challenger_digest[:20]}... has no "
            "matching on-chain ev3 commitment",
        )
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=REPO_ROOT / "docs" / "soak-artifacts"
    )
    args = parser.parse_args()

    try:
        from epago.taskgen.generator import task_ids_digest
        from epago.validator.service import Deps, ValidatorService
        from epago.validator.state import ValidatorState
    except ImportError as exc:
        print(f"SKIP: validator/taskgen modules not yet present in this build ({exc})")
        print("sandbox_soak will drive ValidatorService.tick once they land")
        return 0

    cfg = load_config()
    # The soak owns the round authority: every iteration opens one competition,
    # because nothing is evaluated until a round is triggered.
    cfg = replace(cfg, chain=replace(cfg.chain, round_authority_hotkey=ROUND_AUTHORITY_HK))
    run_dir = args.artifacts_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report = Report()

    with tempfile.TemporaryDirectory(prefix="epago-soak-") as tmp:
        tmp_path = Path(tmp)
        state_dir = tmp_path / "state"
        zoo = ModelZoo(tmp_path / "zoo")

        chain = MockChainClient(block=1000)
        chain.add_neuron(
            NeuronView(
                uid=0, hotkey=chain.identity_hotkey, coldkey="5Validator", stake=1000.0,
                validator_permit=True,
            )
        )

        # Genesis: the pinned seed snapshot is the king; alias its digest.
        _genesis_digest, genesis_dir = zoo.mint(skill=0.5)
        zoo.alias(cfg.seed.seed_digest, genesis_dir)

        deps = Deps(
            chain=chain,
            cfg=cfg,
            state=ValidatorState.load(state_dir),
            corpus=None,
            env=None,
            backend_factory=None,
            run_duel=make_fake_run_duel(zoo),
            run_calibration_duel=lambda *a, **k: 0.0,
            run_probes=lambda challenger_dir, king_dir: [],
            generate_tasks=fake_generate_tasks,
            task_ids_digest=task_ids_digest,
            private_pool=FakePrivatePool(),
            wallet_hotkey=chain.identity_hotkey,
            clock=chain.current_block,
            materialize=zoo.materialize,
            cache_dir=tmp_path / "cache",
        )
        svc = ValidatorService(deps)

        svc.tick()  # establishes the genesis king
        report.check(
            svc.state.king is not None
            and svc.state.king.ref.digest == cfg.seed.seed_digest,
            "genesis king was not established from the chain.toml seed pin",
        )
        chain.advance(1)

        lineage = [svc.state.king.ref.digest]
        coronations = 0
        failure_count_prev = len(svc.state.failure_memory)
        next_uid = 1
        iteration_rows = []

        for iteration in range(1, args.iterations + 1):
            king_digest = svc.state.king.ref.digest
            king_skill = zoo.skill_of_digest(king_digest)
            king_dir = zoo.by_digest[king_digest]

            # Improver strength varies per iteration: some are genuine steps,
            # some match the king exactly (lost duels), so long soaks produce a
            # gradual accuracy staircase instead of one saturating jump.
            # Steps sized against the bootstrap LCB at n=200 tasks: ~0.12 clears
            # the floor decisively, ~0.06 lands in near-miss territory, 0.0 loses.
            step = (0.12, 0.06, 0.0, 0.15, 0.09, 0.0)[iteration % 6]
            cast = {
                "improver": zoo.mint(skill=min(0.92, king_skill + step)),
                "copy": zoo.mint(skill=king_skill, copy_weights_of=king_dir),
                "degenerate": zoo.mint(skill=0.0, config_overrides={"hidden_size": 4096}),
                "nearmiss": zoo.mint(skill=min(0.99, king_skill + 0.01)),
            }
            digests = {}
            for role, (digest, _dir) in cast.items():
                hotkey = f"miner-{role}-{iteration:03d}"
                coldkey = f"5{role.capitalize()}{iteration:03d}xxxx"
                # Named after the HOTKEY: validate_repo_name checks the author
                # hotkey prefix, and the author is the hotkey that signs the
                # commitment. Naming it after the coldkey made every submission
                # die at `hotkey_prefix` before reaching the gates this soak
                # exists to exercise, so exact_copy, config_lock and coronation
                # were never actually tested.
                repo = f"{role}{iteration}/EPAGO-DR-30B-{hotkey[:8].lower()}"
                ref = ModelRef(repo=repo, digest=digest)
                chain.add_neuron(
                    NeuronView(uid=next_uid, hotkey=hotkey, coldkey=coldkey, stake=1.0,
                               validator_permit=False)
                )
                next_uid += 1
                chain.inject_reveal(hotkey, build_reveal(king_digest, ref))
                digests[role] = digest
            chain.advance(1)

            # Intake first, so the field exists, then open this iteration's
            # competition. The interval check is enforced by the chain client,
            # so the soak has to respect the real cadence.
            svc.tick()
            chain.advance(constants.ROUND_MIN_INTERVAL_BLOCKS + 1)
            chain.inject_reveal(ROUND_AUTHORITY_HK, build_round_start(iteration))
            chain.advance(1)

            for _ in range(TICKS_PER_ITERATION):
                svc.tick()
                chain.advance(1)
            # Let this iteration's timelocked ev3 verdicts reveal, then tick
            # once more so quorum derivation sees them — coronation lands in
            # the same iteration, as it would across polls on the live chain.
            chain.advance(constants.VERDICT_REVEAL_BLOCKS)
            svc.tick()
            chain.advance(1)

            # -- invariants -------------------------------------------------
            report.check(
                len(svc.state.queue) == 0,
                f"iter {iteration}: queue did not drain ({len(svc.state.queue)} left)",
            )
            grew = len(svc.state.failure_memory) - failure_count_prev
            report.check(grew >= 0, f"iter {iteration}: failure memory shrank ({grew})")
            # Real failures this round: copy + degenerate always; improver and
            # near-miss may add duel losses. Never more than the cast size.
            report.check(
                2 <= grew <= 4,
                f"iter {iteration}: failure memory grew by {grew}, expected 2..4",
            )
            failure_count_prev = len(svc.state.failure_memory)

            statuses = svc.state.statuses
            report.check(
                statuses.get(digests["copy"]) == "failed_intake",
                f"iter {iteration}: king-copy status {statuses.get(digests['copy'])!r}, "
                "expected failed_intake",
            )
            report.check(
                svc.state.failure_memory.get(digests["copy"], {}).get("code") == "exact_copy",
                f"iter {iteration}: king-copy failure code "
                f"{svc.state.failure_memory.get(digests['copy'], {}).get('code')!r}, "
                "expected exact_copy",
            )
            report.check(
                statuses.get(digests["degenerate"]) == "failed_intake",
                f"iter {iteration}: degenerate status "
                f"{statuses.get(digests['degenerate'])!r}, expected failed_intake",
            )
            report.check(
                svc.state.failure_memory.get(digests["degenerate"], {}).get("code")
                == "config_lock",
                f"iter {iteration}: degenerate failure code is not config_lock",
            )

            new_king = svc.state.king.ref.digest
            if new_king != king_digest:
                report.check(
                    new_king == digests["improver"],
                    f"iter {iteration}: king changed to {new_king[:20]}..., which is "
                    "not this round's improver",
                )
                lineage.append(new_king)
                coronations += 1
            audit_n = check_audit_log(state_dir, chain, report)
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "block": chain.block,
                    "king": new_king,
                    "failure_memory": len(svc.state.failure_memory),
                    "audit_records": audit_n,
                    "improver_status": statuses.get(digests["improver"]),
                    "nearmiss_status": statuses.get(digests["nearmiss"]),
                }
            )
            print(
                f"iteration {iteration:>2}/{args.iterations}: block={chain.block} "
                f"king={new_king[:18]}... improver={statuses.get(digests['improver'])} "
                f"failures={len(svc.state.failure_memory)} audit={audit_n}"
            )

        report.check(coronations >= 1, "no coronation happened across the whole soak")
        n_records = check_audit_log(state_dir, chain, report)

        # -- artifacts ------------------------------------------------------
        shutil.copy2(state_dir / "state.json", run_dir / "state.json")
        audit_src = state_dir / "audit" / "audit.jsonl"
        if audit_src.exists():
            shutil.copy2(audit_src, run_dir / "audit.jsonl")
        (run_dir / "report.json").write_text(
            json.dumps(
                {
                    "iterations": args.iterations,
                    "final_block": chain.block,
                    "king_lineage": lineage,
                    "coronations": coronations,
                    "audit_records": n_records,
                    "iteration_rows": iteration_rows,
                    "invariant_failures": report.failures,
                    "skipped_checks": report.skips,
                    "last_weights": chain.last_weights,
                },
                indent=2,
            )
        )

    print(f"artifacts written to {run_dir}")
    for s in report.skips:
        print(f"  SKIPPED CHECK: {s}")
    if report.failures:
        print(f"SOAK FAIL: {len(report.failures)} invariant violation(s)")
        return 1
    print(
        f"SOAK PASS: {args.iterations} iterations, {coronations} coronation(s), "
        f"lineage {' -> '.join(d[:14] for d in lineage)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
