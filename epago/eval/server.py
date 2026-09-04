"""Persistent eval server.

A validator runs one of these per GPU box. The invariants that matter:

* exactly one GPU job at a time — a single asyncio.Lock guards the GPU, and a
  second POST /duel (or /calibrate, /probes) while one is running gets 409
  instead of queueing, so the caller (the validator scheduler) owns queue
  policy. One *job* still means one job when the box has eight cards: the job
  itself fans out across them (see :mod:`epago.eval.pool`);
* the king stays warm — backends are cached per model directory, the
  challenger's backend is evicted (closed) after its duel, the king's is kept
  resident for the next challenger. With a GPU pool the pool owns residency
  instead: each device keeps its replica until it is handed work for a
  different checkpoint, so the king survives across duels without an explicit
  eviction policy here;
* models arrive as pinned references — the wire (see
  :mod:`epago.eval.remote`) carries (repo, digest) pairs and the server
  materializes the committed snapshots into its own cache, so weights can
  never be swapped in transit.

Authentication: when ``EPAGO_EVAL_TOKEN`` is set (or a ``token`` is passed to
:func:`create_app`), every POST requires ``Authorization: Bearer <token>``;
GET /health stays open for load balancers.

The duel itself runs in a worker thread (it is synchronous, GPU-bound work);
progress events are marshalled back onto the event loop and fanned out to
GET /progress SSE subscribers.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from epago.core.types import ModelRef
from epago.eval.backend import ModelBackend
from epago.eval.duel import run_calibration_duel, run_duel
from epago.eval.harness import harness_digest
from epago.eval.judge import LlmJudge
from epago.eval.remote import (
    DuelRequest,
    outcome_to_wire,
    ref_from_wire,
    task_from_wire,
)


def _default_materialize(ref: ModelRef, cache_dir: Path) -> Path:
    from epago.model.store import materialize_model  # heavy deps stay late-bound

    return materialize_model(ref, cache_dir)


def create_app(
    env: "object",
    backend_factory: Callable[[Path], ModelBackend],
    *,
    cache_dir: Path | None = None,
    materialize: Callable[[ModelRef, Path], Path] | None = None,
    probe_runner: Callable[[Path, Path], list] | None = None,
    llm_judge: LlmJudge | None = None,
    token: str | None = None,
    pool: "object | None" = None,
) -> FastAPI:
    """Build the eval app around one environment and one backend factory.

    ``env`` is duck-typed to the environment subsystem's ResearchEnvironment
    (``tools_for_task``); the server never imports it so the two subsystems
    deploy independently. ``materialize`` is the snapshot hook (defaults to
    :func:`epago.model.store.materialize_model`; tests inject a stub), and
    ``probe_runner`` composes the pre-duel gate (defaults to the torch-free
    weight-level :func:`epago.eval.probes.run_probes`; the CLI wires the full
    behavioural gate on top). ``token`` defaults to the ``EPAGO_EVAL_TOKEN``
    environment variable.

    ``pool`` is the multi-GPU placement layer (:class:`epago.eval.pool.GpuPool`)
    and defaults to ``None`` — the single-backend path this server has always
    run. It is opt-in rather than auto-detected because only the caller knows
    whether ``backend_factory`` builds real GPU engines at all; ``epago eval
    serve`` supplies :func:`epago.eval.pool.build_pool` for the vllm backend,
    and that helper itself returns ``None`` unless the box has more than one
    usable device.
    """
    app = FastAPI(title="epago-eval", version=harness_digest()[:23])
    duel_lock = asyncio.Lock()
    backends: dict[str, ModelBackend] = {}
    subscribers: set[asyncio.Queue] = set()
    cache = Path(cache_dir) if cache_dir is not None else Path(".epago-eval-cache")
    materialize_fn = materialize or _default_materialize
    auth_token = token if token is not None else os.environ.get("EPAGO_EVAL_TOKEN")

    if probe_runner is None:
        from epago.eval.probes import run_probes as probe_runner  # noqa: PLW0127

    def authorize(request: Request) -> None:
        if not auth_token:
            return
        if request.headers.get("authorization") != f"Bearer {auth_token}":
            raise HTTPException(
                status_code=401, detail="missing or invalid bearer token"
            )

    def cached_factory(model_dir: Path) -> ModelBackend:
        key = str(Path(model_dir).resolve())
        backend = backends.get(key)
        if backend is None:
            backend = backend_factory(model_dir)
            backends[key] = backend
        return backend

    def evict_all_but(keep_key: str) -> None:
        for key in [k for k in backends if k != keep_key]:
            backends.pop(key).close()

    def publish(event: dict) -> None:
        for queue in list(subscribers):
            queue.put_nowait(event)

    def parse_body(payload: dict, parser: Callable[[dict], object]) -> object:
        try:
            return parser(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"bad request body: {exc}")

    @app.get("/health")
    async def health() -> dict:
        resident = sorted(backends)
        body = {
            "status": "ok",
            "busy": duel_lock.locked(),
            "harness_digest": harness_digest(),
            "resident_models": resident,
        }
        if pool is not None:
            by_device = pool.resident()
            body["gpu_devices"] = list(pool.devices)
            body["gpu_residency"] = by_device
            body["resident_models"] = sorted(set(resident) | set(by_device.values()))
        return body

    @app.post("/duel")
    async def duel(request: Request) -> dict:
        authorize(request)
        req: DuelRequest = parse_body(await request.json(), DuelRequest.from_wire)
        if duel_lock.locked():
            raise HTTPException(status_code=409, detail="a duel is already running")
        async with duel_lock:
            loop = asyncio.get_running_loop()

            def on_progress(event: dict) -> None:
                loop.call_soon_threadsafe(publish, event)

            king_dir = await asyncio.to_thread(materialize_fn, req.king, cache)
            challenger_dir = await asyncio.to_thread(
                materialize_fn, req.challenger, cache
            )
            spec = req.to_spec(Path(king_dir), Path(challenger_dir))
            king_key = str(Path(king_dir).resolve())
            try:
                outcome = await asyncio.to_thread(
                    run_duel,
                    spec,
                    env,
                    cached_factory,
                    llm_judge,
                    on_progress=on_progress,
                    pool=pool,
                )
            finally:
                # With a pool, residency is the pool's business: every device
                # holds one replica and drops it only when handed a different
                # checkpoint, so the king stays warm and nothing needs evicting.
                if pool is None:
                    # Low-VRAM boxes cannot keep the king resident between duels:
                    # run_duel already released it, so evict everything.
                    import os

                    if os.environ.get("EPAGO_EVAL_LOW_VRAM", "").lower() in ("1", "true", "yes"):
                        evict_all_but("")
                    else:
                        evict_all_but(king_key)
            publish({"phase": "done", "accepted": outcome.accepted})
            return outcome_to_wire(outcome)

    @app.post("/calibrate")
    async def calibrate(request: Request) -> dict:
        authorize(request)
        payload = await request.json()
        king = parse_body(payload.get("king", {}), ref_from_wire)
        tasks = [
            parse_body(t, task_from_wire) for t in payload.get("tasks", ())
        ]
        if duel_lock.locked():
            raise HTTPException(status_code=409, detail="a duel is already running")
        async with duel_lock:
            king_dir = await asyncio.to_thread(materialize_fn, king, cache)
            # Same judge the /duel path uses: the noise floor has to be measured
            # on the graded path duels actually run, judge included.
            rate = await asyncio.to_thread(
                lambda: run_calibration_duel(
                    Path(king_dir), tasks, env, cached_factory, llm_judge, pool=pool
                )
            )
            return {"rate": rate}

    @app.post("/probes")
    async def probes(request: Request) -> dict:
        authorize(request)
        payload = await request.json()
        challenger = parse_body(payload.get("challenger", {}), ref_from_wire)
        king = parse_body(payload.get("king", {}), ref_from_wire)
        if duel_lock.locked():
            raise HTTPException(status_code=409, detail="a duel is already running")
        async with duel_lock:
            challenger_dir = await asyncio.to_thread(materialize_fn, challenger, cache)
            king_dir = await asyncio.to_thread(materialize_fn, king, cache)
            failures = await asyncio.to_thread(
                probe_runner, Path(challenger_dir), Path(king_dir)
            )
            return {
                "failures": [{"code": f.code, "detail": f.detail} for f in failures]
            }

    @app.get("/progress")
    async def progress() -> StreamingResponse:
        queue: asyncio.Queue = asyncio.Queue()
        subscribers.add(queue)

        async def stream() -> AsyncIterator[str]:
            try:
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event, sort_keys=True)}\n\n"
            finally:
                subscribers.discard(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
