#!/usr/bin/env python
"""End-to-end eval smoke on the fixture corpus with scripted backends.

Runs entirely without torch/vllm/bittensor:

  1. build the deterministic fixture corpus,
  2. mint QA-verified tasks with taskgen,
  3. king-vs-king duel with two equal-accuracy scripted backends that
     disagree per task (expect mu ~ 0, lcb < 0, accepted=False),
  4. improver-vs-king duel (expect accepted=True).

Exit codes:
  0  all assertions passed, or SKIP: a required module is not yet present
     in this build (printed explicitly, never silent)
  1  an assertion failed

Usage: .venv/bin/python scripts/smoke_eval.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Keep the smoke fast; must be set before epago.constants is imported.
os.environ.setdefault("EPAGO_BOOTSTRAP_B", "2000")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_PUB = 16
N_PRIV = 16
KING_DIR = Path("king")
CHALL_DIR = Path("challenger")
BLOCK_HASH = "0x" + "ab" * 32
HOTKEY = "5" + "F" * 47


def _import_pipeline():
    """Late-import the parallel-built modules; return (bindings, missing)."""
    bindings = {}
    missing = []
    for label, module, name in (
        ("build_fixture_corpus", "epago.environment.fixtures", "build_fixture_corpus"),
        ("ResearchEnvironment", "epago.environment.services", "ResearchEnvironment"),
        ("generate_tasks", "epago.taskgen.generator", "generate_tasks"),
        ("ScriptedBackend", "epago.eval.backend", "ScriptedBackend"),
        ("DuelSpec", "epago.eval.duel", "DuelSpec"),
        ("run_duel", "epago.eval.duel", "run_duel"),
    ):
        try:
            mod = __import__(module, fromlist=[name])
            bindings[label] = getattr(mod, name)
        except (ImportError, AttributeError) as exc:
            missing.append(f"{module}.{name} ({exc})")
    return bindings, missing


def _knower(scripted_backend, tasks, known_ids: set[str]):
    """Backend that answers known tasks correctly and everything else wrong."""
    answers = {t.question: t.answer for t in tasks if t.task_id in known_ids}
    return scripted_backend(answers=answers)


def _even_odd_ids(tasks) -> tuple[set[str], set[str]]:
    """Split each sorted half by parity so both sides solve exactly half."""
    ordered = sorted(tasks, key=lambda t: t.task_id)
    even = {t.task_id for i, t in enumerate(ordered) if i % 2 == 0}
    odd = {t.task_id for i, t in enumerate(ordered) if i % 2 == 1}
    return even, odd


def main() -> int:
    b, missing = _import_pipeline()
    if missing:
        print("SKIP: module(s) not yet present in this build:")
        for name in missing:
            print(f"  - {name}")
        print("smoke_eval will run end-to-end once the eval/taskgen/environment modules land")
        return 0

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="epago-smoke-") as tmp:
        print("smoke_eval: building fixture corpus")
        corpus = b["build_fixture_corpus"](Path(tmp) / "corpus.db", n_docs=240, seed=7)
        print(f"  corpus docs: {corpus.doc_count()}")

        print(f"smoke_eval: minting {N_PUB + N_PRIV} tasks (release R1)")
        tasks = b["generate_tasks"](seed=1234, release="R1", corpus=corpus, n=N_PUB + N_PRIV)
        if len(tasks) < N_PUB + N_PRIV:
            print(f"FAIL: taskgen yielded {len(tasks)}/{N_PUB + N_PRIV} tasks")
            return 1
        pub, priv = tasks[:N_PUB], tasks[N_PUB : N_PUB + N_PRIV]
        print(f"  tasks minted: {len(tasks)}")

        env = b["ResearchEnvironment"](corpus)
        spec = b["DuelSpec"](
            king_dir=KING_DIR,
            challenger_dir=CHALL_DIR,
            public_tasks=pub,
            private_tasks=priv,
            block_hash_at_reveal=BLOCK_HASH,
            author_hotkey=HOTKEY,
            king_acc_ema=0.5,
            noise_floor=0.0005,
        )

        # Two equal-accuracy "kings" that disagree on every task, emulating a
        # noisy-but-unbiased pair: each solves the complementary half.
        pub_even, pub_odd = _even_odd_ids(pub)
        priv_even, priv_odd = _even_odd_ids(priv)
        king_ids = pub_even | priv_even
        peer_ids = pub_odd | priv_odd
        all_ids = {t.task_id for t in tasks}

        def duel(chall_ids: set[str], name: str):
            backends = {
                KING_DIR: _knower(b["ScriptedBackend"], tasks, king_ids),
                CHALL_DIR: _knower(b["ScriptedBackend"], tasks, chall_ids),
            }
            outcome = b["run_duel"](spec, env, lambda model_dir: backends[model_dir])
            print(
                f"  {name}: mu_pub={outcome.public.mu_hat:+.4f} "
                f"lcb={outcome.lcb_pub:+.4f} delta={outcome.delta:.4f} "
                f"mu_priv={outcome.private.mu_hat:+.4f} accepted={outcome.accepted}"
            )
            return outcome

        print("smoke_eval: king-vs-king duel (expect mu~0, lcb<0, rejected)")
        outcome = duel(peer_ids, "king-vs-king")
        if abs(outcome.public.mu_hat) > 0.15:
            failures.append(f"king-vs-king mu {outcome.public.mu_hat:+.4f} not ~0")
        if outcome.lcb_pub >= 0:
            failures.append(f"king-vs-king lcb {outcome.lcb_pub:+.4f} not < 0")
        if outcome.accepted:
            failures.append("king-vs-king duel was accepted (must never be)")

        print("smoke_eval: improver-vs-king duel (expect accepted)")
        outcome = duel(all_ids, "improver-vs-king")
        if not outcome.accepted:
            failures.append(
                f"improver-vs-king was rejected (lcb={outcome.lcb_pub:+.4f}, "
                f"delta={outcome.delta:.4f}, mu_priv={outcome.private.mu_hat:+.4f})"
            )

        corpus.close()

    if failures:
        print(f"SMOKE FAIL ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE PASS: fixture corpus -> taskgen -> duels behaved as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
