"""Full episode transcripts, written out for audit.

A score keeps only whether each answer was right. With ``EPAGO_EVAL_TRANSCRIPT_DIR``
set, every finished episode is also written in full — the task, every model output,
every tool call and its result, and the scored answer — so a run can be read back
step by step. Nothing here changes what an episode sees or how it is scored, and
none of it is part of the harness digest.

Files are grouped per eval job: ``<dir>/<job>-<utc stamp>/<model>.<phase>.jsonl``.
The eval server opens a job for each /duel, /round and /calibrate call; episodes
run outside a job (the pre-duel probes) are not written. One job runs at a time
(the server's GPU lock), so the open job is plain module state.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["DIR_ENV", "begin_job", "end_job", "record"]

DIR_ENV = "EPAGO_EVAL_TRANSCRIPT_DIR"

_lock = threading.Lock()
_job: Path | None = None


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))[:160] or "unnamed"


def begin_job(name: str) -> Path | None:
    """Open the folder this job's episodes are written to, or None when disabled."""
    global _job
    root = os.environ.get(DIR_ENV, "").strip()
    path: Path | None = None
    if root:
        path = Path(root) / f"{_safe(name)}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("episode transcripts off for %s: %s", name, exc)
            path = None
    with _lock:
        _job = path
    return path


def end_job() -> None:
    global _job
    with _lock:
        _job = None


def record(model: str, phase: str, index: int, episode, result) -> None:
    """Append one finished episode to the open job. Never raises."""
    with _lock:
        job = _job
    if job is None:
        return
    task = episode.task
    session = episode.session
    try:
        row = {
            "t": time.time(),
            "model": model,
            "phase": phase,
            "index": index,
            "task_id": task.task_id,
            "question": task.question,
            "gold_answer": task.answer,
            "aliases": list(task.aliases),
            "evidence_doc_ids": list(task.evidence_doc_ids),
            "answer": result.answer,
            "correct": result.correct,
            "judge_tier": result.judge_tier,
            "turns": result.turns,
            "error": result.error,
            "malformed_actions": result.malformed_actions,
            "wall_time_s": result.wall_time_s,
            "search_calls": getattr(session, "search_calls", None),
            "browse_calls": getattr(session, "browse_calls", None),
            "repeated_searches": getattr(session, "repeated_searches", None),
            "budget_closed": getattr(episode, "budget_closed", None),
            "messages": episode.messages,
        }
        line = json.dumps(row, ensure_ascii=False, default=str)
        with _lock:
            with open(job / f"{_safe(model)}.{_safe(phase)}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 - a transcript is a record, never a verdict
        logger.warning("could not write the transcript of %s: %s", task.task_id, exc)
