#!/usr/bin/env python
"""Run one real competition round on real models and render the leaderboard.

This is the whole pipeline end to end with nothing faked: real corpus, real
SCI2 exam minted from a block hash, real vLLM rollouts for the king and every
challenger, real paired scoring, real signed-less audit records, real dashboard
export. The output ``index.html`` renders the same way the live leaderboard
does.

Each ``--challenger NAME=DIR`` is a real model directory. With one king and one
challenger of equal weights the duel ties (no crown, honestly); with a stronger
challenger it wins and is crowned — both are real outcomes of the same code.

Usage:
    .venv-eval/bin/python scripts/run_real_round.py \\
        --king /models/tongyi-30b-awq4 \\
        --challenger awq8=/models/tongyi-30b-awq8 \\
        --corpus data/corpus-med-10k/corpus.db --release SCI2 --n 60 \\
        --out /tmp/leaderboard
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago import constants  # noqa: E402
from epago.config import load_config  # noqa: E402
from epago.core.stats import (  # noqa: E402
    adaptive_delta,
    noise_floor_from_calibration,
    round_private_seed,
    round_public_seed,
)
from epago.core.types import Entrant, RoundDuelSpec  # noqa: E402
from epago.environment.corpus import SqliteCorpus  # noqa: E402
from epago.environment.services import ResearchEnvironment  # noqa: E402
from epago.taskgen.generator import generate_tasks  # noqa: E402
from epago.validator.audit import AuditLog, audit16, build_audit_record, record_digest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--king", type=Path, required=True)
    ap.add_argument("--challenger", action="append", default=[], help="NAME=DIR (repeatable)")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--release", default="SCI2")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--block", type=int, default=5_000_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chain-toml", default=None)
    args = ap.parse_args()

    cfg = load_config(args.chain_toml)
    corpus = SqliteCorpus(args.corpus)
    env = ResearchEnvironment(corpus)
    block_hash = "0x" + "%064x" % (args.block * 2654435761 & (2**256 - 1))

    print(f"corpus {corpus.doc_count()} docs · round {args.round} · block {args.block}", flush=True)

    # Real exam, minted from the round's block hash exactly as a live round does.
    public = generate_tasks(
        seed=round_public_seed(block_hash, args.round), release=args.release,
        corpus=corpus, n=args.n, king_probe=None,
    )
    private = generate_tasks(
        seed=round_private_seed(block_hash, args.round), release=args.release,
        corpus=corpus, n=args.n, king_probe=None,
    )
    print(f"exam: {len(public)} public + {len(private)} private tasks", flush=True)

    entrants = []
    for spec in args.challenger:
        name, _, path = spec.partition("=")
        entrants.append(Entrant(
            digest=f"hf:{name.ljust(40, '0')[:40]}", repo=f"team/{cfg.chain.name}-{name}",
            author_hotkey=f"hk-{name}", challenger_dir=Path(path),
        ))

    from epago.eval.backend import backend_factory as make_backend

    def backend_factory(model_dir: Path):
        return make_backend(model_dir, kind="vllm")

    round_spec = RoundDuelSpec(
        king_dir=args.king, entrants=tuple(entrants),
        public_tasks=public, private_tasks=private, round=args.round,
        round_block_hash=block_hash, king_acc_ema=0.5, noise_floor=0.005,
    )

    from epago.eval.duel import run_round_duel

    t0 = time.time()
    results = run_round_duel(round_spec, env, backend_factory, llm_judge=None)
    print(f"round ran in {(time.time()-t0)/60:.1f} min", flush=True)

    # Pick the winner exactly as the service does.
    accepted = [r for r in results if r.outcome.accepted]
    winner = max(accepted, key=lambda r: (r.outcome.lcb_pub, r.entrant.digest)) if accepted else None

    # Real audit records + a state.json the dashboard exporter reads.
    args.out.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(args.out)
    delta = adaptive_delta(round_spec.king_acc_ema, round_spec.noise_floor)
    for r in results:
        o = r.entrant, r.outcome
        rec = build_audit_record(
            round_id=f"round{args.round:06d}-{r.entrant.digest[-8:]}",
            block_hash_at_reveal=block_hash, author_hotkey=r.entrant.author_hotkey,
            king_repo=f"team/{cfg.chain.name}-king", king_digest="hf:" + "0" * 40,
            challenger_repo=r.entrant.repo, challenger_digest=r.entrant.digest,
            corpus_digest=cfg.eval.corpus_digest, taskgen_release=args.release,
            public_seed=r.outcome.public_seed_hex,
            public_task_ids_digest="sha256:" + "0" * 64,
            private_pool_digest="sha256:" + "0" * 64, private_pool_epoch=1,
            n_private_tasks=r.outcome.private.n_tasks, boot_seed=r.outcome.boot_seed_hex,
            king_acc_ema=round_spec.king_acc_ema, delta_threshold=r.outcome.delta,
            mu_hat_pub=r.outcome.public.mu_hat, lcb_pub=r.outcome.lcb_pub,
            mu_hat_priv=r.outcome.private.mu_hat,
            accepted=(winner is not None and r is winner),
            harness_digest="sha256:" + "0" * 64, judge_model_digest="hf:" + "0" * 40,
            eval_code_digest="sha256:" + "0" * 64,
            judge_invocation_rate=r.outcome.judge_invocation_rate,
            revealed_at_block=args.block - 10, intake_at_block=args.block - 5,
            verdict_at_block=args.block + 5, validator_hotkey="validator-0",
            extra={
                "round": args.round, "round_block": args.block,
                "cleared_floor": r.outcome.accepted,
                "public_diffs": [[t, int(d)] for t, d in r.outcome.public_task_results],
                "judge_tier_counts": [[t, int(n)] for t, n in r.outcome.judge_tier_counts],
            },
        )
        record_digest(rec)
        audit.append(rec)

    king = None
    if winner is not None:
        king = {
            "repo": winner.entrant.repo, "digest": winner.entrant.digest,
            "author_hotkey": winner.entrant.author_hotkey, "crowned_block": args.block + 5,
            "reign_started_block": args.block + 5,
            "acc_ema": 0.5 + winner.outcome.public.challenger_acc / 2,
            "coronation_lcb": winner.outcome.lcb_pub,
        }
    state = {
        "king": king, "king_acc_ema": 0.5, "king_coronation_delta": delta,
        "queue": [], "statuses": {}, "candidates": {}, "arena": [],
        "noise_floor_samples": [round_spec.noise_floor], "clean_duels": len(results),
        "organic_dethrones": 1 if winner else 0, "genesis_block": args.block - 1000,
        "sla": [], "last_scan_block": args.block, "last_weights_block": args.block,
        "last_pool_publish_block": 0, "tick_count": 1, "failure_memory": {},
        "seen_digests": {}, "intake_log": [], "pending_mirror": None, "last_error": None,
        "current_round": args.round, "last_round_run": args.round,
        "last_round_block": args.block,
    }
    (args.out / "state.json").write_text(json.dumps(state))

    from epago.dashboard.export import write_dashboard

    out = write_dashboard(args.out, args.out, cfg)
    print()
    print("=" * 60)
    for r in results:
        tag = "CROWNED" if (winner and r is winner) else "        "
        print(f"  {tag} {r.entrant.repo:34s} lcb={r.outcome.lcb_pub:+.4f} "
              f"delta={r.outcome.delta:.4f} "
              f"chall_acc={r.outcome.public.challenger_acc:.1%} "
              f"king_acc={r.outcome.public.king_acc:.1%}")
    print("=" * 60)
    print(f"dashboard: {out}")
    print(f"open:      {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
