"""Remote eval: wire protocol and HTTP client.

A validator that owns no GPU delegates duels to a persistent eval server
(:mod:`epago.eval.server`, run via ``epago eval serve`` on the GPU box). The
wire format is deliberately dumb JSON — model *references* (repo + pinned
digest), serialized tasks, and the raw chain-derived duel inputs — so the
server materializes the exact committed snapshots itself and both sides
recompute identical outcomes from identical public data.

Everything here is torch-free: the client speaks httpx and (de)serializes
:mod:`epago.core.types`; the serializers are shared with the server so the two
sides can never drift on field names.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from epago.core.types import (
    DuelHalf,
    DuelOutcome,
    DuelSpec,
    ModelRef,
    Task,
    TaskOrigin,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DirRefIndex",
    "DuelRequest",
    "RemoteEvalError",
    "RemoteEvalRunner",
    "outcome_from_wire",
    "outcome_to_wire",
    "ref_from_wire",
    "ref_to_wire",
    "task_from_wire",
    "task_to_wire",
]


# --- (de)serializers ----------------------------------------------------------
# Shared by client and server. Round-trips are exact: tuples come back as
# tuples, floats survive JSON bit-identically (json uses repr round-tripping),
# so a deserialized DuelOutcome compares equal to the original dataclass.


def ref_to_wire(ref: ModelRef) -> dict:
    return {"repo": ref.repo, "digest": ref.digest}


def ref_from_wire(payload: dict) -> ModelRef:
    return ModelRef(repo=payload["repo"], digest=payload["digest"])


def task_to_wire(task: Task) -> dict:
    return {
        "task_id": task.task_id,
        "question": task.question,
        "answer": task.answer,
        "aliases": list(task.aliases),
        "evidence_doc_ids": list(task.evidence_doc_ids),
        "masked_doc_ids": list(task.masked_doc_ids),
        "origin": task.origin.value,
        "template": task.template,
        "hops": task.hops,
    }


def task_from_wire(payload: dict) -> Task:
    return Task(
        task_id=payload["task_id"],
        question=payload["question"],
        answer=payload["answer"],
        aliases=tuple(payload.get("aliases", ())),
        evidence_doc_ids=tuple(payload.get("evidence_doc_ids", ())),
        masked_doc_ids=tuple(payload.get("masked_doc_ids", ())),
        origin=TaskOrigin(payload.get("origin", TaskOrigin.GENERATED_PUBLIC.value)),
        template=payload.get("template", ""),
        hops=int(payload.get("hops", 1)),
    )


def _half_to_wire(half: DuelHalf) -> dict:
    return {
        "n_tasks": half.n_tasks,
        "diffs": list(half.diffs),
        "mu_hat": half.mu_hat,
        "king_acc": half.king_acc,
        "challenger_acc": half.challenger_acc,
    }


def _half_from_wire(payload: dict) -> DuelHalf:
    return DuelHalf(
        n_tasks=payload["n_tasks"],
        diffs=tuple(payload["diffs"]),
        mu_hat=payload["mu_hat"],
        king_acc=payload["king_acc"],
        challenger_acc=payload["challenger_acc"],
    )


def outcome_to_wire(outcome: DuelOutcome) -> dict:
    return {
        "public": _half_to_wire(outcome.public),
        "private": _half_to_wire(outcome.private),
        "lcb_pub": outcome.lcb_pub,
        "delta": outcome.delta,
        "accepted": outcome.accepted,
        "boot_seed_hex": outcome.boot_seed_hex,
        "public_seed_hex": outcome.public_seed_hex,
        "public_task_results": [[tid, d] for tid, d in outcome.public_task_results],
        "judge_tier_counts": [[tier, n] for tier, n in outcome.judge_tier_counts],
    }


def outcome_from_wire(payload: dict) -> DuelOutcome:
    return DuelOutcome(
        public=_half_from_wire(payload["public"]),
        private=_half_from_wire(payload["private"]),
        lcb_pub=payload["lcb_pub"],
        delta=payload["delta"],
        accepted=payload["accepted"],
        boot_seed_hex=payload["boot_seed_hex"],
        public_seed_hex=payload["public_seed_hex"],
        public_task_results=tuple(
            (tid, int(d)) for tid, d in payload.get("public_task_results", ())
        ),
        judge_tier_counts=tuple(
            (tier, int(n)) for tier, n in payload.get("judge_tier_counts", ())
        ),
    )


@dataclass(frozen=True)
class DuelRequest:
    """One duel over the wire: model refs plus the raw chain-derived inputs.

    Mirrors :class:`~epago.core.types.DuelSpec` except that models travel as
    pinned references — the server materializes them into its own cache, so
    weights are pulled from the content-addressed store, never uploaded.
    """

    king: ModelRef
    challenger: ModelRef
    public_tasks: list[Task]
    private_tasks: list[Task]
    block_hash_at_reveal: str
    author_hotkey: str
    king_acc_ema: float
    noise_floor: float
    round_id: str = ""

    def to_wire(self) -> dict:
        return {
            "king": ref_to_wire(self.king),
            "challenger": ref_to_wire(self.challenger),
            "public_tasks": [task_to_wire(t) for t in self.public_tasks],
            "private_tasks": [task_to_wire(t) for t in self.private_tasks],
            "block_hash_at_reveal": self.block_hash_at_reveal,
            "author_hotkey": self.author_hotkey,
            "king_acc_ema": self.king_acc_ema,
            "noise_floor": self.noise_floor,
            "round_id": self.round_id,
        }

    @classmethod
    def from_wire(cls, payload: dict) -> "DuelRequest":
        return cls(
            king=ref_from_wire(payload["king"]),
            challenger=ref_from_wire(payload["challenger"]),
            public_tasks=[task_from_wire(t) for t in payload["public_tasks"]],
            private_tasks=[task_from_wire(t) for t in payload["private_tasks"]],
            block_hash_at_reveal=payload["block_hash_at_reveal"],
            author_hotkey=payload["author_hotkey"],
            king_acc_ema=float(payload["king_acc_ema"]),
            noise_floor=float(payload["noise_floor"]),
            round_id=payload.get("round_id", ""),
        )

    def to_spec(self, king_dir: Path, challenger_dir: Path) -> DuelSpec:
        """Bind materialized snapshot directories to produce a runnable spec."""
        return DuelSpec(
            king_dir=king_dir,
            challenger_dir=challenger_dir,
            public_tasks=self.public_tasks,
            private_tasks=self.private_tasks,
            block_hash_at_reveal=self.block_hash_at_reveal,
            author_hotkey=self.author_hotkey,
            king_acc_ema=self.king_acc_ema,
            noise_floor=self.noise_floor,
            round_id=self.round_id,
        )


# --- dir <-> ref reverse index --------------------------------------------------


class DirRefIndex:
    """Reverse map from a materialized snapshot directory back to its ModelRef.

    The local eval contracts pass snapshot *directories* (the service
    materializes before probing/dueling); the wire carries *references*. The
    wiring layer registers every (ref, dir) pair as it materializes, and the
    remote runner reverse-resolves dirs at call time.
    """

    def __init__(self) -> None:
        self._by_dir: dict[str, ModelRef] = {}

    def register(self, ref: ModelRef, model_dir: Path | str) -> None:
        self._by_dir[str(Path(model_dir).resolve())] = ref

    def resolve(self, model_dir: Path | str) -> ModelRef:
        key = str(Path(model_dir).resolve())
        try:
            return self._by_dir[key]
        except KeyError:
            raise KeyError(
                f"no ModelRef registered for snapshot dir {key}; remote eval "
                "requires every model to be materialized through the "
                "registering wrapper"
            ) from None


# --- client ---------------------------------------------------------------------


class RemoteEvalError(RuntimeError):
    """A remote eval call failed in a way retries cannot fix."""


class RemoteEvalRunner:
    """Client-side eval runner: same call contracts as the in-process functions
    (:func:`epago.eval.duel.run_duel` and friends), executed on a remote eval
    server.

    ``env`` / ``backend_factory`` / ``llm_judge`` arguments are accepted for
    contract compatibility and ignored — the server owns its own environment,
    engines, and judge.

    Constructor arguments:

    * ``ref_resolver`` maps a locally materialized snapshot directory back to
      its :class:`ModelRef` (the local contracts pass directories, the wire
      needs references). Wiring passes :meth:`DirRefIndex.resolve`.
    * ``king_ref_resolver`` optionally supplies the current king's ref
      directly for calibration duels; when ``None`` the king's directory is
      reverse-resolved through ``ref_resolver``.
    * ``client`` injects a preconfigured ``httpx.Client`` (tests pass a
      ``fastapi.testclient.TestClient``); when ``None`` a client is built for
      ``base_url`` with no read timeout — a duel legitimately runs for hours.

    Transient transport errors are retried a bounded number of times; HTTP 401
    (bad/missing token) and 409 (server busy with another duel) raise
    :class:`RemoteEvalError` immediately with actionable messages.
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        token: str | None = None,
        ref_resolver: Callable[[Path], ModelRef],
        king_ref_resolver: Callable[[], ModelRef] | None = None,
        client: httpx.Client | None = None,
        retries: int = 3,
        retry_wait_s: float = 2.0,
        busy_wait_s: float = 5.0,
        busy_max_wait_s: float = 900.0,
    ) -> None:
        self._token = token
        self._ref_resolver = ref_resolver
        self._king_ref_resolver = king_ref_resolver
        self._retries = max(1, retries)
        self._retry_wait_s = retry_wait_s
        # A single-GPU eval serves one duel at a time and legitimately returns
        # 409 while a model loads, evicts, or another duel runs. Wait it out
        # rather than abandoning the round; bounded so a wedged server can't hang
        # the tick forever.
        self._busy_wait_s = busy_wait_s
        self._busy_max_wait_s = busy_max_wait_s
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(None, connect=30.0),
        )

    # -- transport ---------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        last_exc: Exception | None = None
        busy_waited = 0.0
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._client.post(path, json=payload, headers=headers)
            except httpx.TransportError as exc:
                last_exc = exc
                logger.warning(
                    "remote eval %s transport error (attempt %d/%d): %s",
                    path, attempt, self._retries, exc,
                )
                if attempt >= self._retries:
                    break
                time.sleep(self._retry_wait_s * attempt)
                continue
            if response.status_code == 401:
                raise RemoteEvalError(
                    f"eval server rejected {path}: 401 unauthorized — set "
                    "EPAGO_EVAL_TOKEN to the token configured on the server"
                )
            if response.status_code == 409:
                # Expected on a single GPU: wait out the busy window instead of
                # abandoning the duel. Does not consume the transport-retry budget.
                attempt -= 1
                if busy_waited >= self._busy_max_wait_s:
                    raise RemoteEvalError(
                        f"eval server busy on {path} for over "
                        f"{self._busy_max_wait_s:.0f}s (409); giving up this round"
                    )
                logger.info("remote eval %s busy (409); waited %.0fs, retrying", path, busy_waited)
                time.sleep(self._busy_wait_s)
                busy_waited += self._busy_wait_s
                continue
            if response.status_code >= 400:
                raise RemoteEvalError(
                    f"eval server {path} failed: {response.status_code} "
                    f"{response.text[:500]}"
                )
            return response.json()
        raise RemoteEvalError(
            f"eval server unreachable after {self._retries} attempts on {path}: {last_exc}"
        )

    # -- eval contracts ------------------------------------------------------------

    def run_duel(
        self,
        spec: DuelSpec,
        env: object = None,
        backend_factory: object = None,
        llm_judge: object = None,
        *,
        on_progress: object = None,
    ) -> DuelOutcome:
        """Contract-compatible with :func:`epago.eval.duel.run_duel`; the spec's
        snapshot directories are reverse-resolved to refs and shipped."""
        del env, backend_factory, llm_judge, on_progress  # server-owned
        request = DuelRequest(
            king=self._ref_resolver(spec.king_dir),
            challenger=self._ref_resolver(spec.challenger_dir),
            public_tasks=spec.public_tasks,
            private_tasks=spec.private_tasks,
            block_hash_at_reveal=spec.block_hash_at_reveal,
            author_hotkey=spec.author_hotkey,
            king_acc_ema=spec.king_acc_ema,
            noise_floor=spec.noise_floor,
            round_id=spec.round_id,
        )
        return outcome_from_wire(self._post("/duel", request.to_wire()))

    def run_calibration_duel(
        self,
        king_dir: Path,
        tasks: list[Task],
        env: object = None,
        backend_factory: object = None,
        llm_judge: object = None,
    ) -> float:
        """Contract-compatible with :func:`epago.eval.duel.run_calibration_duel`."""
        del env, backend_factory, llm_judge  # server-owned
        if self._king_ref_resolver is not None:
            king = self._king_ref_resolver()
        else:
            king = self._ref_resolver(king_dir)
        payload = {
            "king": ref_to_wire(king),
            "tasks": [task_to_wire(t) for t in tasks],
        }
        return float(self._post("/calibrate", payload)["rate"])

    def run_probes(self, challenger_dir: Path, king_dir: Path) -> list:
        """Contract-compatible with the composed probe runner; returns
        :class:`~epago.eval.probes.ProbeFailure` items."""
        from epago.eval.probes import ProbeFailure

        payload = {
            "challenger": ref_to_wire(self._ref_resolver(challenger_dir)),
            "king": ref_to_wire(self._ref_resolver(king_dir)),
        }
        body = self._post("/probes", payload)
        return [
            ProbeFailure(code=f["code"], detail=f["detail"])
            for f in body.get("failures", ())
        ]
