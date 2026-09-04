"""Task QA pipeline — the benchmark correctness gate.

Every task, generated or mined, passes :func:`verify_task` before it can
enter a duel; a task that fails QA never scores a model. The checks are
deliberately mechanical (no LLM) so they are deterministic and replayable by
auditors:

(a) **Answer re-derivable** — every evidence document exists, and the answer
    (or an alias) appears literally in at least one evidence document's text.
    Computed answers (aggregation counts, SCI4 named-set counts) are never
    literal, so they are verified by mechanically recomputing from the
    evidence instead (:func:`epago.taskgen.templates.recount_task`).
(b) **Unambiguity (heuristic)** — search the corpus for the question's entity
    and predicate terms and reject when a non-evidence document restates the
    predicate about the same entity with a *different* value of the same type.
    Honest limits: this only catches conflicts that lexical search surfaces in
    the top hits, only for year/number answers (free-text answers have no
    reliable same-type extractor), and it cannot prove global uniqueness — it
    is a duplicate-fact tripwire, not a proof of a single support set.
(c) **Masking sanity** — the task must remain solvable once its masked
    documents are removed: evidence minus masked must be non-empty, unless the
    template is declared mask-exempt
    (:data:`epago.taskgen.templates.MASK_EXEMPT_TEMPLATES`).
(d) **Answer well-formed** — non-empty, at most ``constants.ANSWER_MAX_CHARS``
    characters, single-line, and free of markup characters that could smuggle
    grading instructions into a judge prompt.

Failure strings are machine-readable tokens (``answer_not_in_evidence``,
``evidence_missing:<doc_id>``, ...) so intake pipelines can surface them
without parsing prose.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from epago import constants
from epago.core.types import Task
from epago.taskgen.templates import (
    ENTITY_RE,
    MASK_EXEMPT_TEMPLATES,
    QUANTITY_RE,
    SELF_PINNED_TEMPLATES,
    YEAR_RE,
    recount_task,
)

if TYPE_CHECKING:  # protocol-only coupling to the environment subsystem
    from epago.environment.corpus import CorpusStore

_MARKUP_RE = re.compile(r"[<>{}\[\]`|]")
_WS_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?\s?%?")
_WORD_RE = re.compile(r"[a-z]{4,}")
_AMBIGUITY_SEARCH_K = 5
_MAX_PREDICATE_TERMS = 6

# Function words plus vocabulary injected by the templates' own question
# phrasing — these carry no fact content and would only dilute the
# predicate-overlap signal in the ambiguity check.
_STOPWORDS = frozenset(
    """
    the a an of in on and or to for with was were is are at by from as its
    their this that which what year fill blank using corpus according also
    mentioned document documents titled considering only many distinct
    four-digit appear combined text
    """.split()
)


@dataclass
class QaReport:
    ok: bool
    failures: list[str] = field(default_factory=list)


def normalize_answer(s: str) -> str:
    """Casefold, collapse whitespace, strip surrounding punctuation."""
    s = _WS_RE.sub(" ", s.casefold().strip())
    return s.strip(".,;:!?\"'")


def _norm_text(s: str) -> str:
    return _WS_RE.sub(" ", s.casefold())


def _answer_kind(answer: str) -> str:
    if YEAR_RE.fullmatch(answer.strip()):
        return "year"
    if _NUMBER_RE.fullmatch(answer.strip()):
        return "number"
    return "text"


def _values_of_kind(text: str, kind: str) -> set[str]:
    if kind == "year":
        return set(YEAR_RE.findall(text))
    return {m.replace(",", "").replace(" ", "") for m in QUANTITY_RE.findall(text)}


def _accepted_values(task: Task, kind: str) -> set[str]:
    accepted = {normalize_answer(task.answer), *(normalize_answer(a) for a in task.aliases)}
    if kind == "number":
        accepted |= {a.replace(",", "").replace(" ", "") for a in accepted}
    return accepted


def _predicate_terms(question: str, entity: str) -> list[str]:
    """Content words of the question minus the entity and stopwords, in order."""
    entity_words = {w.lower() for w in entity.split()}
    terms = []
    for word in _WORD_RE.findall(question.lower()):
        if word in _STOPWORDS or word in entity_words or word in terms:
            continue
        terms.append(word)
    return terms[:_MAX_PREDICATE_TERMS]


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def _check_ambiguity(task: Task, corpus: "CorpusStore", failures: list[str]) -> None:
    """Heuristic conflict scan (check b). See module docstring for limits."""
    if task.template in SELF_PINNED_TEMPLATES:
        return  # mint proves uniqueness mechanically / universe pinned by title
    kind = _answer_kind(task.answer)
    if kind == "text":
        return  # no reliable same-type extractor for free text (documented limit)
    entity_match = ENTITY_RE.search(task.question)
    if entity_match is None:
        return
    entity = entity_match.group(0)
    predicate = _predicate_terms(task.question, entity)
    if not predicate:
        return
    accepted = _accepted_values(task, kind)
    known = set(task.evidence_doc_ids) | set(task.masked_doc_ids)
    needed = max(1, math.ceil(len(predicate) / 2))
    query = f"{entity} {' '.join(predicate)}"
    for hit in corpus.search(query, k=_AMBIGUITY_SEARCH_K):
        if hit.doc_id in known:
            continue
        doc = corpus.get(hit.doc_id)
        if doc is None:
            continue
        for sentence in _sentences(doc.text):
            lowered = sentence.lower()
            if entity.lower() not in lowered:
                continue
            if sum(term in lowered for term in predicate) < needed:
                continue
            values = _values_of_kind(sentence, kind)
            if values and not (values & accepted):
                failures.append(f"ambiguous:{hit.doc_id}")
                return


def verify_task(task: Task, corpus: "CorpusStore") -> QaReport:
    """Run all QA checks; a task enters the duel pool only when ``ok``."""
    failures: list[str] = []

    # (d) answer well-formed
    answer = task.answer.strip()
    if not answer:
        failures.append("answer_empty")
    elif len(answer) > constants.ANSWER_MAX_CHARS:
        failures.append("answer_too_long")
    if _MARKUP_RE.search(answer) or "\n" in answer:
        failures.append("answer_markup")
    if not task.question.strip():
        failures.append("question_empty")
    if not task.evidence_doc_ids:
        failures.append("evidence_empty")

    if failures:
        return QaReport(ok=False, failures=failures)

    # (a) evidence exists and the answer is re-derivable from it
    kind = _answer_kind(answer)
    accepted = _accepted_values(task, kind)
    literal_hit = False
    for doc_id in task.evidence_doc_ids:
        doc = corpus.get(doc_id)
        if doc is None:
            failures.append(f"evidence_missing:{doc_id}")
            continue
        text = _norm_text(doc.text)
        if any(a and a in text for a in accepted):
            literal_hit = True
    if not literal_hit and not any(f.startswith("evidence_missing") for f in failures):
        recount = recount_task(task, corpus)
        if recount is None or normalize_answer(recount) not in accepted:
            failures.append("answer_not_in_evidence")

    # (c) masking sanity
    remaining = set(task.evidence_doc_ids) - set(task.masked_doc_ids)
    if not remaining and task.template not in MASK_EXEMPT_TEMPLATES:
        failures.append("mask_unsolvable")

    # (b) unambiguity heuristic — only meaningful once (a) holds
    if not failures:
        _check_ambiguity(task, corpus, failures)

    return QaReport(ok=not failures, failures=failures)
