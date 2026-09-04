"""Eval server command line.

``epago eval serve`` runs the persistent eval server on a GPU box: real corpus
environment, real inference backends, model snapshots materialized from their
pinned refs. Heavy dependencies (vllm, uvicorn workers touching the engine)
are import-guarded inside the command so ``--help`` works on any machine with
the base install. The orchestrating CLI mounts this app; nothing here touches
``epago.cli``.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="Persistent eval server tooling.")


@app.command("serve")
def serve(
    corpus: Path = typer.Option(..., help="Pinned corpus snapshot (sqlite)."),
    cache_dir: Path = typer.Option(
        Path(".epago-eval-cache"), help="Local cache for materialized model snapshots."
    ),
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8793, help="Bind port."),
    backend: str = typer.Option("vllm", help="Inference backend kind (vllm|scripted)."),
) -> None:
    """Serve POST /duel, /calibrate, /probes over the local GPU.

    Bearer auth comes from the ``EPAGO_EVAL_TOKEN`` environment variable; the
    LLM judge is enabled by ``EPAGO_ENABLE_LLM_JUDGE`` once a judge snapshot is
    pinned in chain.toml.
    """
    import uvicorn

    from epago import constants
    from epago.config import load_config
    from epago.core.stats import derive_seed
    from epago.environment.corpus import SqliteCorpus
    from epago.environment.services import ResearchEnvironment
    from epago.eval.backend import backend_factory as make_backend
    from epago.eval.judge import load_llm_judge, needs_local_judge_engine
    from epago.eval.pool import place_engines
    from epago.eval.probes import make_probe_runner
    from epago.eval.server import create_app
    from epago.validator.wiring import PROBE_TASK_SEED_LABEL

    cfg = load_config()
    corpus_store = SqliteCorpus(corpus)
    env = ResearchEnvironment(corpus_store)

    def backend_factory(model_dir: Path):
        return make_backend(model_dir, kind=backend)

    def probe_tasks_fn():
        from epago.taskgen.generator import generate_tasks

        # A pool, not the probe set: probes.probe_task_set picks the
        # low-hop, template-balanced FORMAT_PROBE_TASKS out of it.
        seed = derive_seed(cfg.eval.corpus_digest, cfg.chain.name, PROBE_TASK_SEED_LABEL)
        return generate_tasks(
            seed=seed,
            release=cfg.eval.taskgen_release,
            corpus=corpus_store,
            n=constants.FORMAT_PROBE_POOL_TASKS,
            king_probe=None,
        )

    # Multi-GPU: one whole model replica per card, sweeps sharded across them.
    # ``None`` on a single-GPU box (and for the scripted backend, which has no
    # engine to place), which leaves the server on its original path.
    gpu_pool, probe_factory, judge_factory = place_engines(
        backend_factory, kind=backend, judge_on_own_device=needs_local_judge_engine(cfg)
    )

    eval_app = create_app(
        env,
        backend_factory,
        cache_dir=cache_dir,
        probe_runner=make_probe_runner(probe_factory, env, probe_tasks_fn),
        llm_judge=load_llm_judge(cfg, cache_dir, judge_factory),
        pool=gpu_pool,
    )
    uvicorn.run(eval_app, host=host, port=port)
