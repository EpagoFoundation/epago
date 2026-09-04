"""Dashboard data exporter.

A pure file-to-file transform: the validator's durable artifacts (``state.json``
+ ``audit/audit.jsonl``) in, one versioned ``dashboard.json`` out. There is no
server and no privileged data source — anyone holding a validator's published
state can render the identical dashboard, which is the point: the dashboard is
an *audit surface*, not an owner's website.

Everything numeric is derived from the same records external auditors replay,
so a dashboard that disagrees with the audit trail is impossible by
construction.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from epago import constants
from epago.config import EpagoConfig
from epago.core.emissions import (
    coronation_bonus_factor,
    inherits_reign,
    phase_b_active,
    reign_decay_factor,
)
from epago.core.stats import noise_floor_from_calibration
from epago.core.types import VerdictDecision

DASHBOARD_SCHEMA = "epd1"

# Keep the export bounded: the site stays fast on years of history.
MAX_DUEL_ROWS = 200
MAX_SERIES_POINTS = 500


@dataclass(slots=True)
class DashboardInputs:
    state: dict[str, Any]
    audit_records: list[dict[str, Any]]
    cfg: EpagoConfig
    current_block: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_dashboard_inputs(state_dir: str | Path, cfg: EpagoConfig) -> DashboardInputs:
    state_dir = Path(state_dir)
    state = json.loads((state_dir / "state.json").read_text())
    audit_path = state_dir / "audit" / "audit.jsonl"
    records: list[dict[str, Any]] = []
    if audit_path.exists():
        for line in audit_path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return DashboardInputs(state=state, audit_records=records, cfg=cfg)


def export_dashboard(inputs: DashboardInputs) -> dict[str, Any]:
    state, records, cfg = inputs.state, inputs.audit_records, inputs.cfg
    records = sorted(records, key=lambda r: (r.get("verdict_at_block", 0), r.get("round_id", "")))
    block = inputs.current_block or _latest_block(state, records)

    return {
        "schema": DASHBOARD_SCHEMA,
        "chain": {"name": cfg.chain.name, "netuid": cfg.chain.netuid, "network": cfg.chain.network},
        "generated_at_block": block,
        "king": _king(state, block, cfg),
        "kpis": _kpis(state, records, block, cfg),
        "lineage": _lineage(records),
        "accuracy_series": _accuracy_series(records),
        "duels": _duel_rows(records, state),
        "rounds": _rounds(records),
        "miners": _miners(records, state),
        "funnel": _funnel(state),
        "queue": _queue(state, block),
        "quorum": _quorum(state, cfg),
        "sla": _sla(state, block),
        "emissions": _emissions(state, block, cfg),
        "tasks": _tasks(records, state, block),
        "anchor": _anchor(state),
        "noise": {
            "floor": noise_floor_from_calibration(state.get("noise_floor_samples", [])),
            "samples": state.get("noise_floor_samples", [])[-50:],
        },
    }


def write_dashboard(state_dir: str | Path, out_dir: str | Path, cfg: EpagoConfig) -> Path:
    """Export ``dashboard.json`` and install the static site next to it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = export_dashboard(load_dashboard_inputs(state_dir, cfg))
    out = out_dir / "dashboard.json"
    out.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True))
    site_src = _find_site_index()
    if site_src is not None:
        (out_dir / "index.html").write_text(site_src.read_text())
    return out


def _find_site_index() -> Path | None:
    """Locate ``leaderboard/index.html`` across install layouts.

    In a source checkout it sits three parents up from this module; in the
    container image it is copied to ``/app/leaderboard`` while the package runs
    from site-packages (so the relative path misses). Probe the plausible spots
    rather than assume one layout, and return ``None`` if the site is unbundled
    (``dashboard.json`` alone is still a valid, if headless, export).
    """
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "leaderboard" / "index.html",
        Path.cwd() / "leaderboard" / "index.html",
        Path("/app/leaderboard/index.html"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# --------------------------------------------------------------------------- sections


def _latest_block(state: dict, records: list[dict]) -> int:
    # `last_weights_block` is serialized as null until the first weight-set, so
    # a fresh validator's state.json would crash a naive max() here.
    candidates = [
        state.get("last_scan_block") or 0,
        state.get("last_weights_block") or 0,
    ]
    if records:
        candidates.append(records[-1].get("verdict_at_block", 0))
    return max(candidates)


def _king(state: dict, block: int, cfg: EpagoConfig) -> dict | None:
    king = state.get("king")
    if not king:
        return None
    reign_age = max(block - king.get("reign_started_block", block), 0)
    return {
        "repo": king.get("repo", ""),
        "digest": king.get("digest", ""),
        "author_hotkey": king.get("author_hotkey", ""),
        "crowned_block": king.get("crowned_block", 0),
        "reign_started_block": king.get("reign_started_block", 0),
        "reign_age_blocks": reign_age,
        "acc_ema": state.get("king_acc_ema", 0.0),
        "coronation_lcb": king.get("coronation_lcb", 0.0),
        "reign_decay": reign_decay_factor(reign_age, cfg.emissions.reign_halflife_blocks),
    }


def _kpis(state: dict, records: list[dict], block: int, cfg: EpagoConfig) -> dict:
    duels = [r for r in records if not r.get("round_id", "").startswith("calib-")]
    accepted = [r for r in duels if r.get("accepted")]
    prev_ema = duels[-2]["king_acc_ema"] if len(duels) >= 2 else None
    noise = noise_floor_from_calibration(state.get("noise_floor_samples", []))
    return {
        "king_acc_ema": state.get("king_acc_ema", 0.0),
        "king_acc_ema_prev": prev_ema,
        "duels_total": len(duels),
        "duels_accepted": len(accepted),
        "accept_rate": len(accepted) / len(duels) if duels else 0.0,
        "organic_dethrones": state.get("organic_dethrones", 0),
        "clean_duels": state.get("clean_duels", 0),
        "queue_depth": len(state.get("queue", [])),
        # Competitions are explicit rounds now: nothing is evaluated between
        # them, so "which round are we on" is the first thing a miner asks.
        "current_round": state.get("current_round", 0),
        "last_round_run": state.get("last_round_run", 0),
        "noise_floor": noise,
        "delta_clamp": constants.DELTA_NOISE_MULTIPLIER * noise,
        "phase": "live"
        if phase_b_active(
            state.get("clean_duels", 0),
            state.get("organic_dethrones", 0),
            block - state.get("genesis_block", block),
            constants.PHASE_B_MIN_CLEAN_DUELS,
            constants.PHASE_B_MIN_DETHRONES,
            constants.PHASE_B_MIN_BLOCKS,
        )
        else "burn",
    }


def _lineage(records: list[dict]) -> list[dict]:
    out = []
    prev_author = None
    for r in records:
        if not r.get("accepted"):
            continue
        author = r.get("author_hotkey", "")
        out.append(
            {
                "digest": r.get("challenger_digest", ""),
                "repo": r.get("challenger_repo", ""),
                "author_hotkey": author,
                "block": r.get("verdict_at_block", 0),
                "lcb": r.get("lcb_pub", 0.0),
                "delta": r.get("delta_threshold", 0.0),
                "self_dethrone": author == prev_author and prev_author is not None,
            }
        )
        prev_author = author
    return out[-MAX_SERIES_POINTS:]


def _accuracy_series(records: list[dict]) -> list[dict]:
    pts = [
        {
            "block": r.get("verdict_at_block", 0),
            "ema": r.get("king_acc_ema", 0.0),
            "coronation": bool(r.get("accepted")),
        }
        for r in records
        if not r.get("round_id", "").startswith("calib-")
    ]
    return pts[-MAX_SERIES_POINTS:]


def _rounds(records: list[dict]) -> list[dict]:
    """One row per competition round, newest first.

    Grouped from audit records via ``extra.round``. Records minted before the
    round mechanism (or calibration runs) carry no round and are left out —
    they still appear in the duel table, so nothing is hidden, just unclaimed
    by any round.
    """
    by_round: dict[int, list[dict]] = {}
    for r in records:
        rnd = (r.get("extra") or {}).get("round")
        if rnd:
            by_round.setdefault(int(rnd), []).append(r)

    out: list[dict] = []
    for rnd in sorted(by_round, reverse=True):
        rows = by_round[rnd]
        winner = next((r for r in rows if r.get("accepted")), None)
        lcbs = [r.get("lcb_pub", 0.0) for r in rows]
        out.append(
            {
                "round": rnd,
                "block": min(int(r.get("verdict_at_block", 0)) for r in rows),
                "entrants": len(rows),
                "winner_hotkey": (winner or {}).get("author_hotkey", ""),
                "winner_digest": (winner or {}).get("challenger_digest", ""),
                "winner_lcb": (winner or {}).get("lcb_pub", 0.0) if winner else None,
                "best_lcb": max(lcbs) if lcbs else 0.0,
                "near_misses": sum(
                    1
                    for r in rows
                    if not r.get("accepted")
                    and r.get("lcb_pub", 0.0) > 0
                    and r.get("mu_hat_priv", 0.0) > 0
                ),
            }
        )
    return out


def _duel_rows(records: list[dict], state: dict) -> list[dict]:
    rows = []
    for r in records:
        if r.get("round_id", "").startswith("calib-"):
            continue
        digest = r.get("challenger_digest", "")
        lcb = r.get("lcb_pub", 0.0)
        delta = r.get("delta_threshold", 0.0)
        # The duel's own verdict, never the submission's later lifecycle
        # (a near-miss that goes stale after the next dethrone still near-missed).
        outcome = (
            "accepted" if r.get("accepted") else ("near_miss" if 0.0 < lcb <= delta else "duel_lost")
        )
        rows.append(
            {
                "round_id": r.get("round_id", ""),
                "round": (r.get("extra") or {}).get("round", 0),
                "block": r.get("verdict_at_block", 0),
                "author_hotkey": r.get("author_hotkey", ""),
                "digest": digest,
                "mu_pub": r.get("mu_hat_pub", 0.0),
                "lcb": lcb,
                "delta": delta,
                "mu_priv": r.get("mu_hat_priv", 0.0),
                "king_acc_ema": r.get("king_acc_ema", 0.0),
                "judge_rate": r.get("judge_invocation_rate", 0.0),
                "outcome": outcome,
                "latency_blocks": max(
                    r.get("verdict_at_block", 0) - r.get("revealed_at_block", 0), 0
                ),
                # Everything a miner needs to see *why* they lost and reproduce it.
                "detail": _duel_detail(r),
            }
        )
    return rows[-MAX_DUEL_ROWS:]


def _duel_detail(r: dict) -> dict:
    """Per-duel diagnostics: which tasks the challenger won/tied/lost, how the
    judge scored, and the exact eval-data provenance to reproduce the verdict.

    Sourced entirely from the audit record (``extra`` + provenance digests), so
    it never reveals anything the audit trail does not already commit to. The
    per-task marks are challenger-minus-king: ``+1`` challenger won that task,
    ``0`` tie, ``-1`` king won.
    """
    extra = r.get("extra") or {}
    tasks = [
        {"task_id": tid, "diff": int(diff)}
        for tid, diff in (extra.get("public_diffs") or [])
    ]
    won = sum(1 for t in tasks if t["diff"] > 0)
    lost = sum(1 for t in tasks if t["diff"] < 0)
    tied = sum(1 for t in tasks if t["diff"] == 0)
    return {
        "tasks": tasks,
        "won": won,
        "tied": tied,
        "lost": lost,
        "judge_tiers": [[tier, int(n)] for tier, n in (extra.get("judge_tier_counts") or [])],
        # Eval-data + code fingerprints: a replayer feeds these to
        # scripts/replay_verdict.py and must recompute the same on-chain digest.
        "provenance": {
            "king_repo": r.get("king_repo", ""),
            "king_digest": r.get("king_digest", ""),
            "challenger_repo": r.get("challenger_repo", ""),
            "challenger_digest": r.get("challenger_digest", ""),
            "corpus_digest": r.get("corpus_digest", ""),
            "judge_model_digest": r.get("judge_model_digest", ""),
            "eval_code_digest": r.get("eval_code_digest", ""),
            "harness_digest": r.get("harness_digest", ""),
            "taskgen_release": r.get("taskgen_release", ""),
            "public_seed": r.get("public_seed", ""),
            "public_task_ids_digest": r.get("public_task_ids_digest", ""),
            "private_pool_digest": r.get("private_pool_digest", ""),
            "private_pool_epoch": r.get("private_pool_epoch", 0),
            "n_private_tasks": r.get("n_private_tasks", 0),
            "block_hash_at_reveal": r.get("block_hash_at_reveal", ""),
            "validator_hotkey": r.get("validator_hotkey", ""),
            "validator_signature": r.get("validator_signature", ""),
        },
    }


def _miners(records: list[dict], state: dict) -> list[dict]:
    by_author: dict[str, dict] = {}
    for r in records:
        if r.get("round_id", "").startswith("calib-"):
            continue
        a = by_author.setdefault(
            r.get("author_hotkey", "?"),
            {"attempts": 0, "accepted": 0, "near_miss": 0, "best_lcb": None, "last_block": 0},
        )
        a["attempts"] += 1
        lcb, delta = r.get("lcb_pub", 0.0), r.get("delta_threshold", 0.0)
        if r.get("accepted"):
            a["accepted"] += 1
        elif 0.0 < lcb <= delta:
            a["near_miss"] += 1
        a["best_lcb"] = lcb if a["best_lcb"] is None else max(a["best_lcb"], lcb)
        a["last_block"] = max(a["last_block"], r.get("verdict_at_block", 0))

    arena_credit: dict[str, float] = {}
    for e in state.get("arena", []):
        arena_credit[e.get("hotkey", "?")] = arena_credit.get(e.get("hotkey", "?"), 0.0) + e.get(
            "lcb_pub", 0.0
        )

    king_hotkey = (state.get("king") or {}).get("author_hotkey")
    out = []
    for hotkey, a in by_author.items():
        out.append(
            {
                "hotkey": hotkey,
                "is_king": hotkey == king_hotkey,
                "attempts": a["attempts"],
                "accepted": a["accepted"],
                "near_miss": a["near_miss"],
                "best_lcb": a["best_lcb"] or 0.0,
                "last_block": a["last_block"],
                "arena_credit": arena_credit.get(hotkey, 0.0),
            }
        )
    out.sort(key=lambda m: (-m["accepted"], -(m["best_lcb"]), -m["attempts"], m["hotkey"]))
    return out[:100]


def _queue(state: dict, block: int) -> list[dict]:
    """Every submission waiting for its duel, oldest reveal first.

    A miner's first question after submitting is "did it arrive, and where is
    it in line?" — answerable only from validator state, so the dashboard is
    where it has to surface. Waiting time is measured from the reveal block:
    that is the moment the chain stamped the entry, and the number every SLA
    figure elsewhere on the page is measured against.
    """
    rows = [
        {
            "position": i + 1,
            "repo": q.get("repo", ""),
            "digest": q.get("digest", ""),
            "author_hotkey": q.get("author_hotkey", ""),
            "king_digest": q.get("king_digest", ""),
            "reveal_block": q.get("reveal_block", 0),
            "enqueued_block": q.get("enqueued_block", 0),
            "waiting_blocks": max(block - int(q.get("reveal_block", 0)), 0),
        }
        for i, q in enumerate(
            sorted(state.get("queue", []), key=lambda q: (q.get("reveal_block", 0), q.get("digest", "")))
        )
    ]
    return rows


def _funnel(state: dict) -> list[dict]:
    """Intake funnel: where submissions ended, in pipeline order."""
    order = (
        ("failed_intake", "Failed intake"),
        ("stale_parent", "Stale parent"),
        ("failed_probes", "Failed probes"),
        ("duel_lost", "Lost duel"),
        ("near_miss", "Near miss"),
        ("accepted", "Crowned"),
    )
    counts: dict[str, int] = {}
    for status in state.get("statuses", {}).values():
        counts[status] = counts.get(status, 0) + 1
    queued = len(state.get("queue", []))
    rows = [{"stage": label, "key": key, "count": counts.get(key, 0)} for key, label in order]
    rows.insert(0, {"stage": "In queue", "key": "queued", "count": queued})
    return rows


def _quorum(state: dict, cfg: EpagoConfig) -> dict:
    candidates = []
    for digest, c in (state.get("candidates") or {}).items():
        candidates.append(
            {
                "digest": digest,
                "author_hotkey": c.get("author_hotkey", ""),
                "lcb": c.get("lcb", 0.0),
                "delta": c.get("delta", 0.0),
                "reveal_block": c.get("reveal_block", 0),
            }
        )
    candidates.sort(key=lambda c: -c["reveal_block"])
    return {
        "theta": cfg.quorum.theta,
        "bootstrap_min_evaluators": cfg.quorum.bootstrap_min_evaluators,
        "verdict_timeout_blocks": cfg.quorum.verdict_timeout_blocks,
        "pending_candidates": candidates[:20],
    }


def _sla(state: dict, block: int) -> dict:
    latencies = [
        max(rec.get("verdict_at_block", 0) - rec.get("revealed_at", rec.get("revealed_at_block", 0)), 0)
        if "verdict_at" not in rec
        else max(rec["verdict_at"] - rec.get("revealed_at", 0), 0)
        for rec in state.get("sla", [])
    ]
    latencies = [l for l in latencies if l >= 0]
    target_blocks = constants.SLA_TARGET_HOURS * 300  # ~300 blocks/hour
    p50 = statistics.median(latencies) if latencies else 0
    p95 = (
        statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20
        else (max(latencies) if latencies else 0)
    )
    return {
        "p50_blocks": p50,
        "p95_blocks": p95,
        "target_hours": constants.SLA_TARGET_HOURS,
        "target_blocks": target_blocks,
        "samples": len(latencies),
        "queue_depth": len(state.get("queue", [])),
        "breaker_hours": constants.QUEUE_BREAKER_HOURS,
        "series": latencies[-MAX_SERIES_POINTS:],
    }


def _emissions(state: dict, block: int, cfg: EpagoConfig) -> dict:
    king = state.get("king")
    e = cfg.emissions
    # Mirror ValidatorService.maybe_set_weights: before the phase gate opens the
    # entire emission burns, and a genesis king with no author earns nothing
    # either (compute_weights is handed ``None`` in that case). Rendering the
    # Phase B split regardless told every reader the king was collecting 90%
    # while the validator was in fact setting weight 1.0 on the burn key.
    phase_b = phase_b_active(
        state.get("clean_duels", 0),
        state.get("organic_dethrones", 0),
        block - state.get("genesis_block", block),
        constants.PHASE_B_MIN_CLEAN_DUELS,
        constants.PHASE_B_MIN_DETHRONES,
        constants.PHASE_B_MIN_BLOCKS,
    )
    if not king or not king.get("author_hotkey") or not phase_b:
        return {"king": 0.0, "arena": 0.0, "burn": 1.0}
    reign_age = max(block - king.get("reign_started_block", block), 0)
    decay = reign_decay_factor(reign_age, e.reign_halflife_blocks)
    bonus = coronation_bonus_factor(
        king.get("coronation_lcb", 0.0),
        state.get("king_coronation_delta", 0.0),
        block - king.get("crowned_block", block),
        e,
    )
    # Mirrors epago.core.emissions.compute_weights: one pooled budget, and the
    # arena takes exactly what the king does not.
    king_pool = e.king_share + e.arena_share
    king_share = min(e.king_share * decay * bonus, king_pool)
    leftover = max(king_pool - king_share, 0.0)
    # With no near-miss to pay, compute_weights hands the arena's share to the
    # burn key rather than to an empty pool, so the panel has to as well.
    has_arena = any(float(a.get("lcb_pub", 0.0)) > 0 for a in (state.get("arena") or []))
    arena = leftover if has_arena else 0.0
    burn = 0.0 if has_arena else leftover
    total = king_share + leftover
    return {
        "king": king_share / total,
        "arena": arena / total,
        "burn": burn / total,
        "reign_decay": decay,
        "bonus_factor": bonus,
        "configured": {
            "king": e.king_share,
            "arena": e.arena_share,
        },
    }


def _anchor(state: dict) -> dict:
    """External benchmark anchor: internal EMA vs external benchmark accuracy.

    Observational only — anchor runs never touch verdicts or weights; the
    alert flag mirrors the validator's ``anchor_divergence`` alarm.
    """
    history = state.get("anchor_history") or []
    series = [
        {
            "block": r.get("block", 0),
            "king_digest": r.get("king_digest", ""),
            "accuracy": r.get("accuracy", 0.0),
            "internal_ema": r.get("internal_ema_at_run", 0.0),
            "divergence": r.get("divergence"),
        }
        for r in history
    ][-MAX_SERIES_POINTS:]
    latest = history[-1] if history else {}
    latest_divergence = latest.get("divergence")
    return {
        "series": series,
        "latest_divergence": latest_divergence,
        "alert": latest_divergence is not None
        and latest_divergence > constants.ANCHOR_DIVERGENCE_ALERT,
        "alert_threshold": constants.ANCHOR_DIVERGENCE_ALERT,
        "benchmark_digest": latest.get("benchmark_digest", ""),
        "last_run_block": latest.get("block", 0),
        "interval_blocks": constants.ANCHOR_INTERVAL_BLOCKS,
    }


def _tasks(records: list[dict], state: dict, block: int) -> dict:
    duels = [r for r in records if not r.get("round_id", "").startswith("calib-")]
    judge_series = [
        {"block": r.get("verdict_at_block", 0), "rate": r.get("judge_invocation_rate", 0.0)}
        for r in duels
    ][-MAX_SERIES_POINTS:]
    last = duels[-1] if duels else {}
    return {
        "judge_rate_series": judge_series,
        "private_pool": {
            "epoch": last.get("private_pool_epoch", 0),
            "digest": last.get("private_pool_digest", ""),
            "n_tasks": last.get("n_private_tasks", 0),
            "last_publish_block": state.get("last_pool_publish_block", 0),
            "rotation_blocks": constants.PRIVATE_POOL_ROTATION_BLOCKS,
        },
        "taskgen_release": last.get("taskgen_release", ""),
        "n_public_per_duel": constants.N_PUB_TASKS,
        "n_private_per_duel": constants.N_PRIV_TASKS,
    }


# --------------------------------------------------------------------- from-chain
#
# The canonical exporter: a dashboard built *only* from public chain state — the
# on-chain ``ev3`` verdicts, ``e2`` challenges, and the metagraph — with no
# validator's local files. Anyone running this against the chain derives the
# byte-identical consensus view: the same king, the same coronation lineage, the
# same per-verdict ``lcb``/``mu_priv``. The tiers that only exist off-chain
# (``mu_pub``, the adaptive ``delta``, per-task diffs, judge tiers, and all local
# telemetry) are genuinely not on the chain, so they are left at neutral defaults
# and the output is stamped ``source: "chain"``. To recover the drill-down detail
# a caller additionally fetches a validator's published audit records and verifies
# each against the on-chain ``audit_digest`` — that enrichment is deliberately not
# done here, so this path stays a pure function of consensus state.


def export_from_chain(chain: Any, cfg: EpagoConfig, current_block: int | None = None) -> dict[str, Any]:
    """Build ``dashboard.json`` from chain state alone (see module note above)."""
    state, records, block = _synthesize_chain_inputs(chain, cfg, current_block)
    data = export_dashboard(DashboardInputs(state=state, audit_records=records, cfg=cfg, current_block=block))
    data["source"] = "chain"
    data["note"] = (
        "Canonical consensus view, derived from on-chain verdicts + metagraph. "
        "mu_pub, delta, per-task detail and local telemetry are off-chain and "
        "shown only when enriched from a validator's published audit records."
    )
    return data


def write_dashboard_from_chain(
    chain: Any, out_dir: str | Path, cfg: EpagoConfig, current_block: int | None = None
) -> Path:
    """Export the canonical chain-sourced ``dashboard.json`` + install the site."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = export_from_chain(chain, cfg, current_block)
    out = out_dir / "dashboard.json"
    out.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True))
    site_src = _find_site_index()
    if site_src is not None:
        (out_dir / "index.html").write_text(site_src.read_text())
    return out


def _synthesize_chain_inputs(
    chain: Any, cfg: EpagoConfig, current_block: int | None
) -> tuple[dict, list[dict], int]:
    """Turn chain reads into the ``(state, records)`` shape the exporter consumes.

    Every field that the chain does not carry is filled with a neutral default;
    outcomes and the coronation lineage are computed the same way the validator's
    own quorum code does, so the derived king matches what every observer sees.
    """
    block = current_block if current_block is not None else chain.current_block()
    verdicts = list(chain.read_verdicts())
    challenges = {c.challenger.digest: c for c in chain.read_revealed_submissions()}
    neurons = list(chain.neurons())
    stake_by = {n.hotkey: n.stake for n in neurons}
    permitted = {n.hotkey for n in neurons if n.validator_permit}

    coronations = _replay_coronations(verdicts, stake_by, permitted, cfg)
    king, organic_dethrones = _chain_king(coronations, challenges)

    records: list[dict] = []
    for v in sorted(verdicts, key=lambda x: (x.block, x.validator_hotkey)):
        if v.validator_hotkey not in permitted:
            continue
        ch = challenges.get(v.challenger_digest)
        lcb = v.lcb_pub
        accepted = v.decision is VerdictDecision.ACCEPT
        # ``ev3`` carries the floor the duel was judged against, so the outcome
        # is read straight off the wire. The previous synthetic delta had to
        # infer near-miss from the sign of the LCB alone, which mislabelled any
        # rejection that cleared the floor but failed the private half.
        delta = 0.0 if accepted else v.delta
        author = ch.author_hotkey if ch else v.validator_hotkey
        digest8 = v.challenger_digest.split(":")[-1][:8]
        records.append(
            {
                "round_id": f"chain-{v.block}-{author[:8]}-{digest8}",
                "author_hotkey": author,
                "challenger_repo": ch.challenger.repo if ch else "",
                "challenger_digest": v.challenger_digest,
                "king_digest": ch.king_digest if ch else "",
                "king_repo": "",
                "verdict_at_block": v.block,
                "revealed_at_block": ch.reveal_block if ch else v.block,
                "intake_at_block": ch.reveal_block if ch else v.block,
                "accepted": accepted,
                "lcb_pub": lcb,
                "mu_hat_pub": lcb,          # off-chain; lcb is its valid lower bound
                "delta_threshold": delta,
                "mu_hat_priv": v.mu_priv,   # on-chain
                "private_pool_epoch": v.private_pool_epoch,
                "king_acc_ema": 0.0,        # off-chain EMA — unknown from chain
                "judge_invocation_rate": 0.0,
                "block_hash_at_reveal": ch.block_hash_at_reveal if ch else "",
                "corpus_digest": "",
                "judge_model_digest": "",
                "eval_code_digest": "",
                "harness_digest": "",
                "taskgen_release": "",
                "public_seed": "",
                "public_task_ids_digest": "",
                "private_pool_digest": "",
                "n_private_tasks": 0,
                "validator_hotkey": v.validator_hotkey,
                "validator_signature": "",
                # No per-task detail rides the chain, but the round does (ev3),
                # so the rounds table renders identically from either source.
                "extra": {"round": v.round} if v.round else {},
            }
        )

    # Near-misses feed the arena pool exactly as the live emissions code expects.
    arena = [
        {"hotkey": r["author_hotkey"], "lcb_pub": r["lcb_pub"], "verdict_block": r["verdict_at_block"]}
        for r in records
        if not r["accepted"] and r["lcb_pub"] > 0.0
    ]

    genesis_block = min((v.block for v in verdicts), default=block)
    newest_round = max((v.round for v in verdicts), default=0)
    state = {
        "king": king,
        "king_acc_ema": 0.0,
        # The FLOOR the coronation cleared, not the LCB that cleared it: this
        # feeds coronation_bonus_factor's denominator, so putting the LCB here
        # made the bonus exactly 1.0 for every king.
        "king_coronation_delta": king.get("coronation_delta", 0.0) if king else 0.0,
        "organic_dethrones": organic_dethrones,
        "clean_duels": len(records),
        "current_round": newest_round,
        "last_round_run": newest_round,
        "queue": [],
        "statuses": {},
        "candidates": {},
        "arena": arena,
        "noise_floor_samples": [],
        "genesis_block": genesis_block,
        "sla": [],
        "last_scan_block": block,
        "last_weights_block": block,
        "last_pool_publish_block": 0,
        "anchor_history": [],
    }
    return state, records, block


def _replay_coronations(
    verdicts: list, stake_by: dict[str, float], permitted: set[str], cfg: EpagoConfig
) -> list[tuple[str, int, float, float]]:
    """Replay verdicts in chain order, emitting one row per crown.

    Each row is ``(challenger_digest, block, coronation_lcb, coronation_delta)``.
    The terms come from the ACCEPTING verdicts exactly as
    :meth:`epago.validator.service.ValidatorService._crown` takes them — lowest
    accepting LCB, highest accepting delta — because both ride on ``ev3`` and
    both feed the coronation bonus. Dropping them (they used to be hardcoded to
    zero downstream) made this canonical view compute a different emission split
    than the validator that produced the coronation, which is precisely the
    divergence the chain-derived view exists to rule out.

    Uses current stakes as the quorum weighting — the same approximation the live
    validator makes when it derives the present king. Bootstrap mode (fewer than
    the configured evaluators) crowns on a single ACCEPT, matching quorum.py.
    """
    theta = cfg.quorum.theta
    bootstrap = len(permitted) < cfg.quorum.bootstrap_min_evaluators
    active_stake = sum(stake_by[h] for h in sorted(permitted))
    latest: dict[str, dict[str, VerdictDecision]] = {}
    accepts: dict[str, list] = {}
    crowned: set[str] = set()
    coronations: list[tuple[str, int, float, float]] = []

    def terms(digest: str) -> tuple[float, float]:
        live = accepts.get(digest) or []
        if not live:
            return 0.0, 0.0
        return min(a.lcb_pub for a in live), max(a.delta for a in live)

    for v in sorted(verdicts, key=lambda x: (x.block, x.validator_hotkey)):
        if v.validator_hotkey not in permitted:
            continue
        d = v.challenger_digest
        by_val = latest.setdefault(d, {})
        by_val[v.validator_hotkey] = v.decision
        if v.decision is VerdictDecision.ACCEPT:
            accepts.setdefault(d, []).append(v)
        if d in crowned:
            continue
        if bootstrap:
            if v.decision is VerdictDecision.ACCEPT:
                coronations.append((d, v.block, *terms(d)))
                crowned.add(d)
        else:
            accept = sum(
                stake_by[h]
                for h in sorted(hk for hk, dec in by_val.items() if dec is VerdictDecision.ACCEPT)
            )
            if active_stake > 0 and accept >= theta * active_stake:
                coronations.append((d, v.block, *terms(d)))
                crowned.add(d)
    return coronations


def _chain_king(
    coronations: list[tuple[str, int, float, float]], challenges: dict
) -> tuple[dict | None, int]:
    """Fold the coronation lineage into the current king, honouring reign inheritance."""
    king: dict | None = None
    prev_author: str | None = None
    reign_started = 0
    organic = 0
    for digest, block, lcb, delta in coronations:
        ch = challenges.get(digest)
        author = ch.author_hotkey if ch else ""
        if prev_author is not None and inherits_reign(prev_author, author):
            pass  # self-dethrone: keep the running reign clock
        else:
            reign_started = block
            organic += 1
        king = {
            "repo": ch.challenger.repo if ch else "",
            "digest": digest,
            "author_hotkey": author,
            "crowned_block": block,
            "reign_started_block": reign_started,
            "coronation_lcb": lcb,
            "coronation_delta": delta,
        }
        prev_author = author
    return king, organic
