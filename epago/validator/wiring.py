"""Production composition of the validator-in-a-box.

Everything the service loop needs — eval runners, probe gate, task generation,
the managed private pool — is assembled here from the real subsystems. Tests and soaks build their own :class:`~epago.validator.service.Deps`
with fakes; this module is the one place where the live wiring lives, so a
seam mismatch is a wiring bug in exactly one file.
"""

from __future__ import annotations

import hashlib
import logging
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from epago import constants
from epago.chain.client import ChainClient
from epago.config import EpagoConfig
from epago.core.stats import derive_seed
from epago.validator.service import Deps
from epago.validator.state import ValidatorState

if TYPE_CHECKING:
    from epago.core.types import Task

logger = logging.getLogger(__name__)

PROBE_TASK_SEED_LABEL = b"probe-tasks"
def _task_from_row(row: dict) -> "Task":
    """One minted JSONL row as a :class:`Task`, ignoring mint-only metadata.

    ``meta`` carries the hidden terms and the answer's identity. It is a mint
    and audit record, and must never travel with a file a model is shown.
    """
    from epago.core.types import Task, TaskOrigin

    return Task(
        task_id=row["task_id"],
        question=row["question"],
        answer=row["answer"],
        aliases=tuple(row.get("aliases") or ()),
        evidence_doc_ids=tuple(row.get("evidence_doc_ids") or ()),
        masked_doc_ids=tuple(row.get("masked_doc_ids") or ()),
        origin=TaskOrigin(row.get("origin", TaskOrigin.GENERATED_PRIVATE.value)),
        template=row.get("template", "bridge_intersection"),
        hops=int(row.get("hops", 3)),
    )


PRIVATE_POOL_COMMIT_VERSION = "ep1"


class ManagedPrivatePool:
    """The service-facing private pool: sampling, rotation, delayed transparency.

    Wraps the persisted task pool and its refresh pipeline. On rotation the
    outgoing pool is written in full to ``publish_dir`` (the validator's public
    artifact mirror) and only a compact ``ep1|<epoch>|<digest16>`` commitment
    is returned for the chain — pool content never goes on-chain, its digest
    does, and the published file must hash to the digest committed while the
    pool was active.
    """

    def __init__(
        self,
        state_dir: Path,
        corpus,
        cfg: EpagoConfig,
        ingest_dir: Path | None = None,
        created_block: int = 0,
    ) -> None:
        from epago.taskgen.private_pool import PrivatePool

        self._pool_dir = state_dir / "private_pool"
        self._publish_dir = state_dir / "publications"
        self._publish_dir.mkdir(parents=True, exist_ok=True)
        self._corpus = corpus
        self._cfg = cfg
        self._ingest_dir = ingest_dir  # deprecated manual override; auto source preferred
        try:
            self._pool = PrivatePool.load(self._pool_dir)
        except (FileNotFoundError, ValueError):
            # Stamp the genesis pool with the CURRENT block, never 0. With 0 the
            # pool is older than PRIVATE_POOL_ROTATION_BLOCKS the moment it is
            # built, so a validator starting from an empty state directory
            # rotated it on every single tick: each tick minted a fresh pool
            # (a full private-feed download) and bumped the epoch, while the
            # ``ep1`` commitment for the previous epoch was still rate-limited
            # on chain. ``_ensure_pool_committed`` then never caught up and no
            # round could ever run — a fresh box was permanently unable to duel.
            self._pool = self._build_pool(PrivatePool, epoch=1, created_block=created_block)
            self._pool.save(self._pool_dir)

    # -- PrivatePoolLike -------------------------------------------------------

    @property
    def epoch(self) -> int:
        return self._pool.epoch

    @property
    def digest(self) -> str:
        return self._pool.digest()

    def sample(self, n: int, seed: int) -> list["Task"]:
        rng = np.random.Generator(np.random.PCG64(seed))
        return self._pool.sample(n, rng)

    def rotation_due(self, current_block: int) -> bool:
        return self._pool.due_for_rotation(current_block)

    def commitment(self) -> str:
        """The ``ep1`` commitment for the pool that is active right now."""
        digest16 = self._pool.digest().removeprefix("sha256:")[:16]
        return f"{PRIVATE_POOL_COMMIT_VERSION}|{self._pool.epoch}|{digest16}"

    def rotate(self, current_block: int) -> str | None:
        """Rotate, publish the outgoing pool, and commit the *incoming* one.

        The commitment names the pool that is about to start grading duels, not
        the one that just stopped. Committing on the way out — which is what
        this returned before — chain-stamped a digest roughly six days after
        every verdict that pool had already produced, so the stamp proved
        nothing about what the tasks were while they were live. The spec says
        the commitment is published *before* the pool is used in any duel; this
        is that.
        """
        from epago.taskgen.private_pool import PrivatePool

        outgoing = self._pool
        successor = self._build_pool(
            PrivatePool, epoch=outgoing.epoch + 1, created_block=current_block
        )
        outgoing.rotate(list(successor.tasks), current_block, self._publish_dir)
        self._pool = successor
        self._pool.save(self._pool_dir)
        return self.commitment()

    # -- refresh ----------------------------------------------------------------

    def _build_pool(self, pool_cls, epoch: int, created_block: int):
        """Mint a fresh private task set, autonomously — no human in the loop.

        Preference order: an automated fresh feed (a bounded, secret-seeded slice
        of a large public *dated* dataset — the strongest overfit defense and the
        R2 fix that retires ``--ingest-dir``), falling back to validator-local
        entropy over the pinned corpus when no feed is configured. The seed is
        deliberately a fresh local secret each build: privacy of the *active*
        pool is the point (each validator's holdout is independent and
        unpredictable), and the pool becomes auditable when published at rotation.
        This runs only in the rotation loop, off the duel critical path.
        """
        local_seed = int.from_bytes(
            hashlib.blake2b(os.urandom(32), digest_size=8).digest(), "big"
        )
        tasks: list[Task] = []

        # An audited pool, if this validator has any left unused. These are
        # minted offline and cannot be regenerated from a seed -- a language
        # model wrote the wording, and no promise about temperature survives a
        # provider changing hardware or model version. They do not need to be
        # regenerated: every property that makes them correct is arithmetic
        # over the corpus, re-derivable by anyone with `scripts/verify_pool.py`
        # in about two minutes, and the pool publishes in full at rotation like
        # any other. So the file is the artifact, and the audit is the trust.
        tasks = self._audited_tasks()
        if tasks:
            logger.info("private pool epoch %d: %d audited tasks", epoch, len(tasks))
            return pool_cls(
                epoch=epoch,
                created_block=created_block,
                tasks=tuple(tasks),
                storage_path=self._pool_dir,
            )

        source = self._make_source(local_seed)
        if source is not None:
            from epago.taskgen.ingest import build_private_tasks

            try:
                tasks = build_private_tasks(
                    [source],
                    self._corpus,
                    seed=local_seed,
                    n=constants.N_PRIV_TASKS,
                    release=self._cfg.eval.taskgen_release,
                )
            except Exception as exc:  # noqa: BLE001 - ingest is best-effort supply
                logger.warning("private feed failed, falling back to corpus: %s", exc)
        if not tasks:
            from epago.core.types import TaskOrigin
            from epago.taskgen.generator import generate_tasks

            tasks = [
                _restamp_private(t, TaskOrigin)
                for t in generate_tasks(
                    seed=local_seed,
                    release=self._cfg.eval.taskgen_release,
                    corpus=self._corpus,
                    n=constants.N_PRIV_TASKS,
                    king_probe=None,
                )
            ]
        return pool_cls(
            epoch=epoch,
            created_block=created_block,
            tasks=tuple(tasks),
            storage_path=self._pool_dir,
        )

    def _audited_tasks(self) -> list[Task]:
        """Take the next unused audited pool file, or nothing.

        ``EPAGO_AUDITED_POOL_DIR`` holds pool files minted and verified ahead of
        time. A file is consumed once and then renamed, so a rotation never
        re-serves tasks a previous epoch already published -- republishing a
        retired pool would hand miners a set they have already seen in full.

        Absence is not an error. With no directory configured the pool falls
        back to the generator exactly as before, which is what every existing
        deployment does.
        """
        from epago.core.types import TaskOrigin

        directory = os.environ.get("EPAGO_AUDITED_POOL_DIR", "").strip()
        if not directory:
            return []
        pool_dir = Path(directory)
        if not pool_dir.is_dir():
            logger.warning("EPAGO_AUDITED_POOL_DIR is not a directory: %s", directory)
            return []

        for path in sorted(pool_dir.glob("*.jsonl")):
            try:
                rows = [
                    json.loads(line)
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
            except Exception as exc:  # noqa: BLE001 - a bad file must not stop rotation
                logger.warning("unreadable audited pool %s: %s", path.name, exc)
                continue
            if len(rows) < constants.N_PRIV_TASKS:
                logger.warning(
                    "audited pool %s has %d tasks, need %d — skipping",
                    path.name,
                    len(rows),
                    constants.N_PRIV_TASKS,
                )
                continue
            tasks = [
                _restamp_private(_task_from_row(row), TaskOrigin)
                for row in rows[: constants.N_PRIV_TASKS]
            ]
            # Consumed, not deleted: the file is still needed to publish the
            # pool at rotation and to answer an auditor later.
            path.rename(path.with_suffix(".jsonl.used"))
            return tasks
        return []

    def _make_source(self, seed: int):
        """The private-pool document feed, preferring the automated fresh source.

        A configured ``[private_source]`` (a dated public dataset revision) is the
        default — fully automated, no human. ``ingest_dir`` remains a deprecated
        offline override. Neither → ``None`` → corpus fallback.
        """
        ps = self._cfg.private_source
        if ps.repo:
            from epago.taskgen.ingest import HfSnapshotSource

            return HfSnapshotSource(
                repo=ps.repo,
                revision=ps.revision,
                seed=seed,
                text_column=ps.text_column,
                max_shards=ps.max_shards,
            )
        if self._ingest_dir:
            from epago.taskgen.ingest import LocalDirSource

            return LocalDirSource(self._ingest_dir)
        return None


def _restamp_private(task, task_origin_cls):
    from dataclasses import replace

    return replace(task, origin=task_origin_cls.GENERATED_PRIVATE)


def build_production_deps(
    *,
    cfg: EpagoConfig,
    chain: ChainClient,
    state_dir: Path,
    corpus_path: Path,
    cache_dir: Path,
    wallet_hotkey: str,
    backend_kind: str = "vllm",
    ingest_dir: Path | None = None,
    sign=None,
) -> Deps:
    """Assemble live Deps from the real subsystems. Raises with an actionable
    message when an optional extra (torch/vllm) is missing.

    Eval placement: when the ``EPAGO_EVAL_URL`` environment variable is set,
    duels, calibration duels, and probes are delegated to that remote eval
    server (bearer token from ``EPAGO_EVAL_TOKEN``); model materialization is
    wrapped so every (ref, dir) pair is registered for reverse resolution over
    the wire. Otherwise everything runs in-process on this box.
    """
    from epago.environment.corpus import SqliteCorpus
    from epago.environment.services import ResearchEnvironment
    from epago.environment.sync import verify_corpus
    from epago.eval.backend import backend_factory as make_backend
    from epago.eval.judge import load_llm_judge, needs_local_judge_engine
    from epago.taskgen.generator import generate_tasks, task_ids_digest

    if set(cfg.eval.corpus_digest.removeprefix("sha256:")) != {"0"}:
        verify_corpus(corpus_path, cfg.eval.corpus_digest)
    else:
        logger.warning("corpus digest is the genesis placeholder — integrity check skipped")
    corpus = SqliteCorpus(corpus_path)
    env = ResearchEnvironment(corpus)
    state = ValidatorState.load(state_dir)
    try:
        genesis_pool_block = int(chain.current_block())
    except Exception:  # noqa: BLE001 - a chain read must not stop the validator booting
        logger.warning("could not read the current block for the genesis private pool; using 0")
        genesis_pool_block = 0
    private_pool = ManagedPrivatePool(
        state_dir, corpus, cfg, ingest_dir=ingest_dir, created_block=genesis_pool_block
    )

    def backend_factory(model_dir: Path):
        return make_backend(model_dir, kind=backend_kind)

    def probe_tasks_fn() -> list["Task"]:
        # The POOL the protocol-compliance probe draws from — not the probe
        # set itself: probes.probe_task_set picks the low-hop,
        # template-balanced FORMAT_PROBE_TASKS out of it, so the gate is not
        # handed a random sample of the full exam difficulty. The fixed label
        # keeps the supply deterministic per chain generation without leaking
        # duel seeds.
        seed = derive_seed(cfg.eval.corpus_digest, cfg.chain.name, PROBE_TASK_SEED_LABEL)
        return generate_tasks(
            seed=seed,
            release=cfg.eval.taskgen_release,
            corpus=corpus,
            n=constants.FORMAT_PROBE_POOL_TASKS,
            king_probe=None,
        )

    materialize_dep = None
    # Overridden below when a GPU pool reserves the judge a card of its own.
    judge_factory = backend_factory
    eval_url = os.environ.get("EPAGO_EVAL_URL")
    if eval_url:
        from epago.eval.remote import DirRefIndex, RemoteEvalRunner

        ref_index = DirRefIndex()

        def registering_materialize(ref, dest_cache_dir: Path) -> Path:
            from epago.model.store import materialize_model

            model_dir = materialize_model(ref, dest_cache_dir)
            ref_index.register(ref, model_dir)
            return model_dir

        runner = RemoteEvalRunner(
            eval_url,
            token=os.environ.get("EPAGO_EVAL_TOKEN"),
            ref_resolver=ref_index.resolve,
        )
        run_duel_fn = runner.run_duel
        # The remote eval server has no batch endpoint; the service falls back
        # to one duel per entrant over the round's shared exam.
        run_round_duel_fn = None
        run_calibration_fn = runner.run_calibration_duel
        run_probes_fn = runner.run_probes
        materialize_dep = registering_materialize
        logger.info("eval delegated to remote server at %s", eval_url)
    else:
        import functools

        from epago.eval.duel import run_calibration_duel, run_duel, run_round_duel
        from epago.eval.pool import place_engines
        from epago.eval.probes import make_probe_runner

        # Multi-GPU placement: one whole replica per card, sweeps sharded across
        # them (epago.eval.pool). The pool is ``None`` on a single-GPU box — and
        # on any box not running the real engine — in which case the duel
        # functions are passed through unwrapped and every engine is built by
        # the plain factory, on the exact path they have always taken.
        gpu_pool, gpu_backend_factory, judge_factory = place_engines(
            backend_factory,
            kind=backend_kind,
            judge_on_own_device=needs_local_judge_engine(cfg),
        )
        # Everything else that wants one engine on one card — the probe gate and
        # the observational anchor run — borrows a pool replica instead of
        # building its own on a card the pool has already filled.
        backend_factory = gpu_backend_factory
        run_duel_fn = run_duel
        run_round_duel_fn = run_round_duel
        run_calibration_fn = run_calibration_duel
        if gpu_pool is not None:
            run_duel_fn = functools.partial(run_duel, pool=gpu_pool)
            run_round_duel_fn = functools.partial(run_round_duel, pool=gpu_pool)
            run_calibration_fn = functools.partial(run_calibration_duel, pool=gpu_pool)
        run_probes_fn = make_probe_runner(backend_factory, env, probe_tasks_fn)

    return Deps(
        chain=chain,
        cfg=cfg,
        state=state,
        corpus=corpus,
        env=env,
        backend_factory=backend_factory,
        run_duel=run_duel_fn,
        run_round_duel=run_round_duel_fn,
        run_calibration_duel=run_calibration_fn,
        run_probes=run_probes_fn,
        generate_tasks=generate_tasks,
        task_ids_digest=task_ids_digest,
        private_pool=private_pool,
        wallet_hotkey=wallet_hotkey,
        clock=chain.current_block,
        llm_judge=load_llm_judge(cfg, cache_dir, judge_factory),
        materialize=materialize_dep,
        cache_dir=cache_dir,
        sign=sign,
    )
