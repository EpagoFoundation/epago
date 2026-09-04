"""Re-check a published task pool from scratch, with no language model.

This is what makes an LLM-worded exam usable in a trustless setting. The pool
cannot be *reproduced* -- a model wrote the sentences, and no promise about
temperature survives a provider changing hardware, batching or model version.
It does not need to be. Every property that makes a task correct is arithmetic
over the corpus, and this re-derives all of them from the published file plus
the pinned corpus.

So the minter is not trusted. It is checked:

``unique``
    The two hidden terms occur together in exactly one document, and that
    document is the answer.

    Re-derived from the CORPUS, never from the entity index. The index is an
    input the minter supplies, so believing it would leave the one guarantee
    that matters resting on a file the auditor did not build. Every task's two
    terms are re-extracted across all 50,420 documents with the same rule the
    index was built from (:func:`epago.taskgen.entities.extract_bridges`), and
    the index's own claim is then compared against that. A disagreement is
    reported as a failure of the pool, not silently preferred either way.

``concealed``
    Neither hidden term appears in the question. If it did, the reader would be
    handed the key instead of having to go and read it.

``answer_hidden``
    The answer's title is not spelled out by the question's own words.

``route``
    Run against the real search backend: both anchors are reachable from what
    the question says about them, the two terms together surface the answer,
    and the question as a whole does not.

``labels``
    Each "names a specific X" is supported by the text of the paper the reader
    is actually sent to. Without this a solver is told to find a dietary lipid
    level in a semiconductor paper -- true of 65% of one batch before the check
    existed, and invisible to every other test including the oracle.

``identity``
    The task id is the content hash of question, answer and evidence, so a
    renamed or edited copy cannot pass as the original.

Exit status is non-zero if any task fails anything, which makes this usable as
a gate rather than a report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3  # noqa: E402

from epago.environment.corpus import SqliteCorpus  # noqa: E402
from epago.taskgen.chain import (  # noqa: E402
    ANCHOR_FIND_K,
    GOLD_FIND_K,
    GOLD_LEAK_K,
    usable_as_answer,
)
from epago.taskgen.entities import EntityIndex, extract_bridges  # noqa: E402
from epago.taskgen.templates import content_task_id  # noqa: E402
from epago.taskgen.verbalize import _type_supported_by  # noqa: E402

#: Every property this script re-derives. Named so a verdict says what was
#: actually checked, rather than leaving a reader to infer it from the code.
CHECK_NAMES = (
    "index_agrees_with_corpus",
    "answer_unique_to_the_two_terms",
    "anchors_carry_one_term_each",
    "evidence_documents_distinct",
    "terms_concealed_from_question",
    "answer_not_spelled_by_question",
    "answer_usable_and_is_the_gold_title",
    "labels_supported_by_their_own_anchor",
    "answer_reachable_from_the_two_terms",
    "question_does_not_surface_the_answer",
    "anchors_reachable",
    "task_id_matches_content",
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _rank(hits, doc_id: str) -> int | None:
    for i, hit in enumerate(hits, start=1):
        if hit.doc_id == doc_id:
            return i
    return None


def rebuild_postings(
    terms: set[str], docs: dict[str, tuple[str, str]], workers: int
) -> dict[str, frozenset[str]]:
    """Which documents contain each term, computed from the documents.

    One pass over the corpus, extracting bridges per document exactly as the
    minter did, and inverting. This is the step that makes the pool auditable
    without trusting anything the minter shipped: the uniqueness of every
    answer is then a fact about the corpus, checkable by anyone who has it.
    """
    wanted = set(terms)
    postings: dict[str, set[str]] = {t: set() for t in wanted}
    lock = threading.Lock()

    def scan(chunk: list[tuple[str, str]]) -> dict[str, set[str]]:
        local: dict[str, set[str]] = {}
        for doc_id, text in chunk:
            for term in extract_bridges(text) & wanted:
                local.setdefault(term, set()).add(doc_id)
        return local

    # Abstract text only, because that is what the index was built from.
    # Rebuilding from title+text instead produced five spurious disagreements
    # -- terms like "Molecular Docking" that appear in one more title than
    # abstract. The gap is real and is reported separately by
    # `title_only_occurrences`; it is not an index error.
    items = [(doc_id, text) for doc_id, (title, text) in docs.items()]
    size = max(1, len(items) // max(1, workers))
    chunks = [items[i : i + size] for i in range(0, len(items), size)]
    with ThreadPoolExecutor(workers) as pool:
        for local in pool.map(scan, chunks):
            with lock:
                for term, ids in local.items():
                    postings[term].update(ids)
    return {t: frozenset(ids) for t, ids in postings.items()}


def title_only_occurrences(
    terms: set[str], docs: dict[str, tuple[str, str]], workers: int
) -> dict[str, int]:
    """Documents naming a term in their TITLE but not their abstract.

    The uniqueness proof is over abstracts, so such a document is invisible to
    it while being perfectly visible to a person reading the corpus. That is a
    real, if small, source of second answers, and an auditor should be told the
    number rather than left to discover it.
    """
    wanted = set(terms)
    extra: Counter[str] = Counter()
    lock = threading.Lock()

    def scan(chunk):
        local: Counter[str] = Counter()
        for title, text in chunk:
            in_title = extract_bridges(title) & wanted
            if in_title:
                for term in in_title - extract_bridges(text):
                    local[term] += 1
        return local

    values = list(docs.values())
    size = max(1, len(values) // max(1, workers))
    with ThreadPoolExecutor(workers) as pool:
        for local in pool.map(scan, [values[i : i + size] for i in range(0, len(values), size)]):
            with lock:
                extra.update(local)
    return dict(extra)


def check_task(
    task: dict,
    corpus: SqliteCorpus,
    index: EntityIndex,
    docs: dict[str, tuple[str, str]],
    rebuilt: dict[str, frozenset[str]] | None = None,
) -> list[str]:
    """Return the names of every check this task fails; empty means sound."""
    failures: list[str] = []
    meta = task.get("meta") or {}
    x, y = meta.get("bridge_x"), meta.get("bridge_y")
    evidence = task.get("evidence_doc_ids") or []
    if not x or not y or len(evidence) != 3:
        return ["malformed"]
    gold, anchor_a, anchor_b = evidence
    question, answer = task["question"], task["answer"]

    # --- unique: the whole guarantee -----------------------------------------
    # Computed from the corpus when a rebuild is available, and the index's
    # claim checked against it. Falling back to the index is offered only for a
    # quick pass; it is not an audit, and the summary says so.
    if rebuilt is not None:
        docs_x, docs_y = rebuilt.get(x, frozenset()), rebuilt.get(y, frozenset())
        if index.docs_with_bridge(x) != docs_x or index.docs_with_bridge(y) != docs_y:
            failures.append("index_disagrees_with_corpus")
    else:
        docs_x, docs_y = index.docs_with_bridge(x), index.docs_with_bridge(y)

    both = docs_x & docs_y
    if both != frozenset({gold}):
        failures.append(f"not_unique({len(both)} documents)")

    # --- the anchors carry one term each, and are not the answer ------------
    if anchor_a not in docs_x:
        failures.append("anchor_a_lacks_term")
    if anchor_b not in docs_y:
        failures.append("anchor_b_lacks_term")
    if gold in (anchor_a, anchor_b) or anchor_a == anchor_b:
        failures.append("evidence_documents_not_distinct")

    # --- concealed: the reader must go and read, not copy from the question --
    # A term counts as leaked when the question spells ALL of it, not when it
    # happens to share one ordinary word. "First Nations" overlapping the
    # frame's own phrase "named in the first" is not a leak -- the reader still
    # has to discover "Nations" -- and treating it as one condemned sound tasks
    # while hiding the real leaks among them.
    qt = _tokens(question)
    for term in (x, y):
        tokens = _tokens(term)
        if tokens and tokens <= qt:
            failures.append("term_leaked_into_question")
            break
    at = _tokens(answer)
    if at and at <= qt:
        failures.append("answer_spelled_by_question")

    # --- the answer can fairly be demanded back -----------------------------
    if not usable_as_answer(answer):
        failures.append("answer_not_usable")
    if docs.get(gold, ("", ""))[0] != answer:
        failures.append("answer_is_not_the_gold_title")

    # --- labels describe the paper the reader is sent to --------------------
    a_text = " ".join(docs.get(anchor_a, ("", "")))
    b_text = " ".join(docs.get(anchor_b, ("", "")))
    if meta.get("bridge_x_type") and not _type_supported_by(a_text, meta["bridge_x_type"]):
        failures.append("label_x_unsupported_by_anchor")
    if meta.get("bridge_y_type") and not _type_supported_by(b_text, meta["bridge_y_type"]):
        failures.append("label_y_unsupported_by_anchor")

    # --- route: measured against the backend the solver will actually use ---
    if _rank(corpus.search(f"{x} {y}", k=GOLD_FIND_K), gold) is None:
        failures.append("answer_unreachable_from_the_two_terms")
    if _rank(corpus.search(question, k=GOLD_LEAK_K), gold) is not None:
        failures.append("question_surfaces_the_answer")

    # A named anchor is reached by its title, a described one by its
    # description; checking the wrong one would pass a route nobody can walk.
    tier = meta.get("tier", "described_both")
    for which, doc_id, named in (
        ("a", anchor_a, tier in ("named_both", "named_one")),
        ("b", anchor_b, tier == "named_both"),
    ):
        query = docs.get(doc_id, ("", ""))[0] if named else _described_clause(question, which)
        if query and _rank(corpus.search(query, k=ANCHOR_FIND_K), doc_id) is None:
            failures.append(f"anchor_{which}_unreachable")

    # --- identity: an edited copy cannot inherit the original's id ----------
    expected = content_task_id(question, answer, (gold, anchor_a, anchor_b))
    if task.get("task_id") != expected:
        failures.append("task_id_does_not_match_content")

    return failures


def _described_clause(question: str, which: str) -> str:
    """Pull one anchor's description back out of the assembled question.

    The frame is fixed, so this is parsing a known shape rather than guessing.
    """
    lead = "One study in this corpus " if which == "a" else "A second study in this corpus "
    if lead not in question:
        lead = "A study in this corpus "
        if lead not in question:
            return ""
    tail = question.split(lead, 1)[1]
    return tail.split(". That study names", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True, help="the published pool, JSONL")
    ap.add_argument("--corpus", default="data/corpus-science-big/corpus.db")
    ap.add_argument("--index", default="data/corpus-science-big/entities-v1.json")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--show", type=int, default=10, help="failing tasks to print")
    ap.add_argument(
        "--trust-index",
        action="store_true",
        help=(
            "skip rebuilding term postings from the corpus. Faster, and NOT an "
            "audit: uniqueness then rests on a file the minter supplied."
        ),
    )
    ap.add_argument("--expect-corpus-digest", default="", help="fail if the corpus differs")
    args = ap.parse_args()

    tasks = [
        json.loads(line)
        for line in Path(args.tasks).read_text().splitlines()
        if line.strip()
    ]
    index = EntityIndex.read(Path(args.index))
    conn = sqlite3.connect(args.corpus)
    docs = {
        row[0]: (row[1], row[2])
        for row in conn.execute("SELECT doc_id, title, text FROM docs")
    }

    _local = threading.local()

    def corpus() -> SqliteCorpus:
        existing = getattr(_local, "corpus", None)
        if existing is None:
            existing = SqliteCorpus(args.corpus)
            _local.corpus = existing
        return existing

    # The corpus is the ground everything else is measured against, so it is
    # identified by content rather than by path. Two auditors quoting the same
    # verdict must have been looking at the same documents.
    corpus_digest = hashlib.sha256(
        "\n".join(
            f"{doc_id}\t{title}\t{text}" for doc_id, (title, text) in sorted(docs.items())
        ).encode()
    ).hexdigest()

    print(f"pool           {args.tasks}")
    print(f"tasks          {len(tasks)}")
    print(f"corpus         {args.corpus} ({len(docs):,} documents)")
    print(f"corpus digest  sha256:{corpus_digest[:32]}")
    print(f"entity index   {index.digest()[:16]}")
    if args.expect_corpus_digest and not corpus_digest.startswith(
        args.expect_corpus_digest.removeprefix("sha256:")
    ):
        print("CORPUS MISMATCH — this is not the corpus the pool was minted against")
        return 2

    rebuilt = None
    if args.trust_index:
        print("mode           QUICK (index trusted — this is not an audit)")
    else:
        terms = {
            t
            for task in tasks
            for t in ((task.get("meta") or {}).get("bridge_x"), (task.get("meta") or {}).get("bridge_y"))
            if t
        }
        print(f"mode           FULL AUDIT (re-extracting {len(terms)} terms from every document)")
        started = time.time()
        rebuilt = rebuild_postings(terms, docs, args.workers)
        title_only = title_only_occurrences(terms, docs, args.workers)
        print(f"               rebuilt in {time.time() - started:.0f}s")
        if title_only:
            n_terms = len(title_only)
            n_docs = sum(title_only.values())
            print(
                f"               NOTE {n_docs} document(s) across {n_terms} term(s) "
                "name a term only in their title, where the uniqueness proof "
                "cannot see them"
            )
    print()

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(
            pool.map(lambda t: (t, check_task(t, corpus(), index, docs, rebuilt)), tasks)
        )

    failed = [(t, f) for t, f in results if f]
    counts: Counter[str] = Counter()
    for _, f in failed:
        for name in f:
            counts[re.sub(r"\(.*\)", "", name)] += 1

    for task, names in failed[: args.show]:
        print(f"  FAIL {task.get('task_id', '?')[:16]}  {', '.join(names)}")
    if len(failed) > args.show:
        print(f"  ... and {len(failed) - args.show} more")
    if failed:
        print()
        for name, count in counts.most_common():
            print(f"  {name:44s} {count}")

    ok = len(tasks) - len(failed)
    print()
    # A verdict anyone can quote and anyone can reproduce: it is a hash of the
    # pool's contents and the corpus they were checked against, so two auditors
    # agreeing on this string agree on exactly what was verified.
    verdict = hashlib.sha256(
        json.dumps(
            {
                "task_ids": sorted(t.get("task_id", "") for t in tasks),
                "corpus": corpus_digest,
                "checks": sorted(CHECK_NAMES),
                "audit": not args.trust_index,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    print(f"VERIFIED {ok}/{len(tasks)} = {ok / max(1, len(tasks)):.1%}")
    print(f"audit id sha256:{verdict[:32]}")
    if failed:
        print("POOL REJECTED — do not duel on it")
        return 1
    if args.trust_index:
        print("POOL PASSES QUICK CHECK — rerun without --trust-index for an audit")
        return 0
    print(
        "POOL SOUND — every claim re-derived from the corpus itself, no model "
        "and no minter-supplied file trusted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
