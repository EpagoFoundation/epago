#!/usr/bin/env python
"""FRAMES anchor run: the pinned model over an external benchmark, our harness.

Runs ``google/frames-benchmark`` items (converted by
``scripts/frames_corpus.py``) through the SHIPPED rollout harness — the same
chat-templated prompts, the same Search/Browse tool surface, the same
programmatic exact-match judge a duel uses — over the local FRAMES Wikipedia
corpus. Nothing about scoring is special-cased for this benchmark; only the
task source changes.

Relationship to :mod:`epago.eval.anchor`: the benchmark file format and its
byte digest come from :func:`epago.eval.anchor.load_benchmark`, and tasks carry
:data:`epago.eval.anchor.ANCHOR_TEMPLATE`, so a run here is the same kind of
object an in-validator anchor run produces. What is NOT reused is
``run_anchor`` itself: it drives one rollout at a time, which for a
300-item multi-hop benchmark is hours of a GPU sitting idle between turns.
This drives the same episodes through :meth:`epago.eval.pool.GpuPool.run_sweeps`
instead — identical per-episode logic (``run_rollouts_batched``), sharded
across every idle card.

Retrieval diagnostics are the point of a first run as much as the score is: if
the gold article never reaches the model, the score says nothing about
reasoning. Two independent measures are recorded:

  * **retrievable** — offline, model-free: does a BM25 search for the question
    text surface any gold chunk in the top k? An upper bound on what the
    simplest possible query can do.
  * **retrieved / browsed** — live: every session is wrapped so the doc ids
    search actually returned and the ids the model actually browsed are
    recorded per task.

Usage:
    EPAGO_EVAL_GPUS=1,2,3,4 .venv-eval/bin/python scripts/frames_eval.py \\
        --model /path/to/model \\
        --corpus data/frames/corpus.db --tasks data/frames/frames_tasks.jsonl \\
        --n 300 --out frames_run.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago.core.types import Task, TaskOrigin  # noqa: E402
from epago.environment.corpus import SqliteCorpus  # noqa: E402
from epago.environment.services import ResearchEnvironment, ToolSession  # noqa: E402
from epago.eval.anchor import ANCHOR_TEMPLATE, load_benchmark  # noqa: E402
from epago.eval.judge import judge_answer, normalize  # noqa: E402
from epago.eval.pool import GpuPool, SweepRequest, resolve_devices  # noqa: E402


class SafeBackend:
    """A replica whose engine errors become wrong answers, not a dead sweep.

    ``run_rollouts_batched`` guards tool calls and judging but not the engine
    call itself, so one request the engine refuses — a prompt over
    ``max_model_len`` is the realistic case on a 15-hop episode — raises out of
    the batch step and kills every episode on that shard. Here a failed batch
    is retried one prompt at a time and only the offending prompt yields ``""``,
    which the harness sees as a generation with no parseable action: that
    episode gets its nudge and then ends unanswered, exactly as
    :func:`epago.eval.anchor.run_anchor` already scores a crashed rollout.
    Nothing about a successful generation changes.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.failures = 0

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        try:
            return self._inner.generate(prompt, max_tokens, stop)
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            print(f"  [engine] generate failed ({type(exc).__name__}); "
                  f"scoring the episode unanswered", flush=True)
            return ""

    def generate_many(self, prompts: list[str], max_tokens: int, stop: list[str]) -> list[str]:
        try:
            return self._inner.generate_many(prompts, max_tokens, stop)
        except Exception as exc:  # noqa: BLE001
            print(f"  [engine] batch of {len(prompts)} failed ({type(exc).__name__}); "
                  f"retrying one at a time", flush=True)
            return [self.generate(p, max_tokens, stop) for p in prompts]

    def close(self) -> None:
        self._inner.close()


class RecordingSession:
    """A :class:`ToolSession` that remembers what the model saw.

    Pure instrumentation: every call is forwarded unchanged, so the episode is
    byte-identical to one run without it. Only the ids are kept, not the text.
    """

    def __init__(self, inner: ToolSession) -> None:
        self._inner = inner
        self.queries: list[str] = []
        self.returned: list[str] = []
        self.browsed: list[str] = []

    def search(self, query: str) -> str:
        out = self._inner.search(query)
        self.queries.append(query)
        for line in out.splitlines():
            if "[" in line and "]" in line:
                self.returned.append(line[line.index("[") + 1 : line.index("]")])
        return out

    def browse(self, doc_id: str) -> str:
        self.browsed.append(doc_id)
        return self._inner.browse(doc_id)

    # v4 native surface: forward and record the web-shaped calls too — without
    # these the harness's tool dispatch hits AttributeError and every episode
    # researches blind (measured: FRAMES EM collapsed to 8%).
    def search_page(self, queries: list[str]) -> str:
        out = self._inner.search_page(queries)
        self.queries.extend(queries)
        for m in re.finditer(r"\(id: ([^)\s]+)\)", out):
            self.returned.append(m.group(1))
        return out

    def visit_pages(self, targets: list[str]) -> str:
        for t in targets:
            resolved = self._inner._resolve(t)
            self.browsed.append(resolved if resolved is not None else t)
        return self._inner.visit_pages(targets)


def select(items: list, n: int) -> list:
    """Deterministic even-stride subset, so a partial run is not front-loaded
    on whatever the benchmark file happens to order first."""
    if n <= 0 or n >= len(items):
        return items
    stride = len(items) / n
    return [items[int(i * stride)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=Path("data/frames/corpus.db"))
    ap.add_argument("--tasks", type=Path, default=Path("data/frames/frames_tasks.jsonl"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("frames_run.json"))
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--search-k", type=int, default=10)
    ap.add_argument("--max-action-tokens", type=int, default=0,
                    help="DIAGNOSTIC: override the harness per-turn generation "
                         "budget (0 = the shipped MAX_ACTION_TOKENS). A run with "
                         "this set is not a shipped-harness measurement.")
    args = ap.parse_args()

    if args.max_action_tokens:
        from epago.eval import harness as _harness

        print(f"DIAGNOSTIC: per-turn budget {_harness.MAX_ACTION_TOKENS} -> "
              f"{args.max_action_tokens} tokens (NOT the shipped harness)", flush=True)
        _harness.MAX_ACTION_TOKENS = int(args.max_action_tokens)

    digest, anchor_tasks = load_benchmark(args.tasks)
    meta = {}
    for line in args.tasks.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            meta[obj["id"]] = obj
    picked = select(anchor_tasks, args.n)
    print(f"benchmark {args.tasks} digest={digest}", flush=True)
    print(f"items: {len(picked)} of {len(anchor_tasks)}", flush=True)

    corpus = SqliteCorpus(args.corpus)
    env = ResearchEnvironment(corpus, search_k=args.search_k)
    print(f"corpus: {corpus.doc_count()} chunks", flush=True)

    tasks: list[Task] = []
    for at in picked:
        info = meta.get(at.task_id, {})
        tasks.append(
            Task(
                task_id=at.task_id,
                question=at.question,
                answer=at.answer,
                aliases=at.aliases,
                evidence_doc_ids=tuple(info.get("evidence_doc_ids", ())),
                masked_doc_ids=(),  # the gold articles ARE the corpus here
                origin=TaskOrigin.GENERATED_PUBLIC,
                template=ANCHOR_TEMPLATE,
                hops=max(len(info.get("articles", ())), 1),
            )
        )

    # --- model-free retrievability -------------------------------------------
    # Two different questions, and conflating them is how a retrieval bug gets
    # reported as a reasoning result:
    #
    #  * BY TITLE — search the gold article's own title. This is the ceiling:
    #    if the article cannot be found by its exact name, it is effectively
    #    not in the corpus and no amount of reasoning reaches it.
    #  * BY QUESTION — paste the whole prompt in as the query, which is the
    #    naive thing a model does on turn one. The gap between these two
    #    numbers is a property of the TOOL, not of the model: while search
    #    AND-ed its terms this was 0/300, because a 40-word question has no
    #    document containing every word.
    docmap = json.loads((args.tasks.parent / "frames_docmap.json").read_text())
    by_title = {t: set(ids) for t, ids in docmap.items()}
    retrievable, title_hit_frac = {}, {}
    for t in tasks:
        gold = set(t.evidence_doc_ids)
        hits = {h.doc_id for h in corpus.search(t.question, k=args.search_k)}
        retrievable[t.task_id] = bool(gold & hits)
        arts = meta.get(t.task_id, {}).get("articles", [])
        found = 0
        for art in arts:
            want = by_title.get(art, set())
            if not want:
                continue
            got = {h.doc_id for h in corpus.search(art, k=args.search_k)}
            found += bool(want & got)
        title_hit_frac[t.task_id] = found / len(arts) if arts else 0.0
    print(f"gold findable by exact title (top-{args.search_k}): "
          f"{statistics.mean(title_hit_frac.values()):.1%} of gold articles; "
          f"all articles findable on "
          f"{sum(v >= 1.0 for v in title_hit_frac.values())}/{len(tasks)} items",
          flush=True)
    print(f"gold surfaced by pasting the whole question as the query: "
          f"{sum(retrievable.values())}/{len(tasks)}", flush=True)

    sessions: dict[str, RecordingSession] = {}
    lock = threading.Lock()

    def session_factory(task: Task) -> RecordingSession:
        s = RecordingSession(env.tools_for_task(task))
        with lock:
            sessions[task.task_id] = s
        return s

    devices = resolve_devices()
    print(f"devices: {devices}", flush=True)

    def engine_factory(model_dir: Path, device: str):
        from epago.eval.worker import WorkerBackend

        return SafeBackend(WorkerBackend(model_dir, device))

    pool = GpuPool(devices, engine_factory=engine_factory)

    done: list = []
    t0 = time.time()

    def on_result(ri: int, phase: str, ti: int, res) -> None:
        with lock:
            done.append(res)
            n = len(done)
            if n % 10 == 0 or n == len(tasks):
                acc = sum(r.correct for r in done) / n
                print(f"  {n:4d}/{len(tasks)}  em={acc:5.1%}  "
                      f"{(time.time()-t0)/n:5.1f}s/task elapsed={(time.time()-t0)/60:.1f}m",
                      flush=True)

    try:
        sweeps = pool.run_sweeps(
            [SweepRequest(model_dir=args.model,
                          phases=(("frames", tuple(tasks)),),
                          label="frames")],
            session_factory=session_factory,
            llm_judge=None,
            concurrency=args.concurrency,
            on_result=on_result,
        )
    finally:
        pool.close()
    wall = time.time() - t0
    sweep = sweeps[0]
    if sweep.error is not None:
        print(f"SWEEP FAILED: {sweep.error}")
        return 1
    results = sweep.phases["frames"]

    records = []
    for t, r in zip(tasks, results):
        s = sessions.get(t.task_id)
        gold = set(t.evidence_doc_ids)
        returned = set(s.returned) if s else set()
        browsed = set(s.browsed) if s else set()
        cand = (r.answer or "")
        # Diagnostics only, never added to the score: a LENIENT match asks
        # whether the gold string is present inside a longer answer, which is
        # what an exact-match grader charges as wrong when a model wraps its
        # answer in a sentence.
        ng, nc = normalize(t.answer), normalize(cand)
        # Both sides must be non-empty: an episode that never answered has
        # nc == "", and "" is a substring of every string, so testing
        # containment without this guard scores every silent episode as a
        # near-miss (it reported 219/300 "lenient" on a run where only 110
        # episodes answered at all).
        lenient = bool(ng) and bool(nc) and (ng in nc or nc in ng) and not r.correct
        records.append({
            "task_id": t.task_id,
            "question": t.question,
            "expected": t.answer,
            "got": r.answer,
            "correct": bool(r.correct),
            "lenient_only": lenient,
            "judge_tier": r.judge_tier,
            "turns": r.turns,
            "error": r.error,
            "malformed_actions": r.malformed_actions,
            "reasoning_types": meta.get(t.task_id, {}).get("reasoning_types", ""),
            "n_articles": len(meta.get(t.task_id, {}).get("articles", ())),
            "missing_articles": meta.get(t.task_id, {}).get("missing_articles", 0),
            "gold_chunks": len(gold),
            "retrievable_offline": retrievable[t.task_id],
            "gold_title_findable_frac": title_hit_frac[t.task_id],
            "gold_returned_by_search": bool(gold & returned),
            "gold_browsed": bool(gold & browsed),
            "n_searches": len(s.queries) if s else 0,
            "n_browses": len(s.browsed) if s else 0,
            "wall_s": r.wall_time_s,
        })

    n = len(records)
    em = sum(r["correct"] for r in records) / max(n, 1)
    print()
    print("=" * 70)
    print(f"FRAMES exact match : {em:6.2%}   ({sum(r['correct'] for r in records)}/{n})")
    print(f"  + lenient-only   : {sum(r['lenient_only'] for r in records)} "
          f"(gold string inside a longer answer; NOT counted above)")
    print(f"RETRIEVAL  gold findable by title : "
          f"{statistics.mean(r['gold_title_findable_frac'] for r in records):6.1%} of articles")
    print(f"           question-as-query works: "
          f"{sum(r['retrievable_offline'] for r in records)/n:6.1%}")
    print(f"           gold returned by search: "
          f"{sum(r['gold_returned_by_search'] for r in records)/n:6.1%}")
    print(f"           gold browsed           : "
          f"{sum(r['gold_browsed'] for r in records)/n:6.1%}")
    seen = [r for r in records if r["gold_browsed"]]
    if seen:
        print(f"           EM | gold browsed      : "
              f"{sum(r['correct'] for r in seen)/len(seen):6.1%} (n={len(seen)})")
    print(f"CORPUS  items missing >=1 article: "
          f"{sum(bool(r['missing_articles']) for r in records)}")
    print(f"ERRORS  {Counter(r['error'] for r in records if r['error'])}")
    print(f"TURNS   mean {statistics.mean(r['turns'] for r in records):.1f}  "
          f"searches {statistics.mean(r['n_searches'] for r in records):.1f}  "
          f"browses {statistics.mean(r['n_browses'] for r in records):.1f}")
    print(f"WALL    {wall/60:.1f} min for {n} items on {len(devices)} device(s) "
          f"({wall/n:.1f}s/item)")
    by_type: dict[str, list] = {}
    for r in records:
        by_type.setdefault(r["reasoning_types"] or "(none)", []).append(r)
    print("BY REASONING TYPE (n>=8):")
    for k, rs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        if len(rs) >= 8:
            print(f"  {k[:52]:54s} {sum(x['correct'] for x in rs)/len(rs):6.1%} (n={len(rs)})")
    print("=" * 70)

    args.out.write_text(json.dumps({
        "benchmark": str(args.tasks), "benchmark_digest": digest,
        "corpus": str(args.corpus), "corpus_chunks": corpus.doc_count(),
        "model": str(args.model), "n": n, "n_total_benchmark": len(anchor_tasks),
        "exact_match": em, "wall_s": wall, "devices": list(devices),
        "search_k": args.search_k, "concurrency": args.concurrency,
        "max_action_tokens": args.max_action_tokens or 2048,
        "records": records,
    }, indent=1, ensure_ascii=False))
    print(f"full record -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
