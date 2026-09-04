"""Rule-based task templates (release R1).

Templates mint corpus-grounded QA tasks with no LLM anywhere in the loop:
every question is derived from literal sentences in corpus documents via
regex/heuristic extraction, and every answer literally appears in (or is
mechanically recomputable from) the evidence documents. That property is what
lets :mod:`epago.taskgen.qa` verify tasks without a model, and what keeps
generation a pure function of ``(seed, release, corpus)`` — the determinism
contract every validator relies on to reproduce each other's holdouts.
LLM-assisted templates can ship in later releases behind new entries in
:data:`RELEASES`; the protocol does not change.

The templates are domain-free by construction — the only field-specific thing
in this module is the word list that decides whether a sentence reports a
result and which quantities are comparable across studies. That list is a
:class:`FindingVocabulary` bound to the template instance (never read from a
global), so one set of mechanics serves a medicine slice (:data:`SCI2`) and all
of scientific literature (:data:`SCI3`) with no divergent code path. A
release's vocabulary is frozen for the same reason its template tuple is: it
would otherwise silently change every validator's holdouts.

Masking rules (recorded per task in ``masked_doc_ids``, enforced by the
rollout environment):

* ``entity_fact`` — single-source; masking the only evidence document would
  make the task unsolvable, so ``masked_doc_ids`` is empty (mask-exempt).
* ``cross_doc_join`` — the linking document (where the entity was found) is
  masked; the fact document stays searchable because it is the only place the
  answer is guaranteed to exist.
* ``aggregation`` — the question pins its universe to explicitly named
  documents, so masking any of them would make the count unanswerable;
  ``masked_doc_ids`` is empty (mask-exempt).

Determinism: templates draw *all* randomness from the caller's
``numpy.random.Generator`` and iterate documents in sorted-doc-id order —
never from wall clock, ``os.urandom``, or hash-seed-dependent set ordering.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Protocol

import numpy as np

from epago import constants
from epago.core.types import Task, TaskOrigin

if TYPE_CHECKING:  # protocol-only coupling to the environment subsystem
    from epago.environment.corpus import CorpusStore, Document

# Four-digit years 1500-2099: old enough to appear in encyclopedic text,
# bounded so page numbers and identifiers rarely false-positive.
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
# Comma-grouped numbers ("12,000") or percentages; bare small integers are
# deliberately excluded — they are too ambiguous to anchor a fact.
QUANTITY_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s?%")
# Multi-word capitalized phrases ("Kelda River"). Single capitalized words are
# skipped here (too often sentence-initial noise); the relation regex below
# still handles single-word subjects.
ENTITY_RE = re.compile(r"\b[A-Z][a-z][\w-]*(?:\s+[A-Z][a-z][\w-]*)+\b")
# "<Entity> was <predicate> in <year>" — the one relation shape common enough
# in encyclopedic prose to phrase as a natural question without an LLM.
_RELATION_RE = re.compile(
    r"(?P<ent>(?:[A-Z][\w-]*\s+){0,4}[A-Z][\w-]*)\s+"
    r"(?P<aux>was|were|is|are)\s+"
    r"(?P<pred>[a-z][^.,;:]{2,70}?)\s+in\s+"
    r"(?P<year>1[5-9]\d{2}|20\d{2})\b"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ARTICLE_RE = re.compile(r"^(The|A|An)\s+")
_MIN_SENTENCE_CHARS = 20
_DOC_TRIES = 8

_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)


class TaskTemplate(Protocol):
    """One rule-based task family.

    ``mint`` returns ``None`` when the documents it sampled cannot support the
    template (no extractable fact, no join partner, ...); the generator treats
    that as a cheap retry, not an error.
    """

    name: str

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None: ...


def content_task_id(question: str, answer: str, evidence_doc_ids: tuple[str, ...]) -> str:
    """Content-addressed task id: same visible content -> same id.

    Aliases, template, and origin are deliberately excluded so a mined copy of
    a generated task (or an alias-tweaked resubmission) collides with the
    original — dedup and single-use enforcement then block it for free.
    """
    payload = json.dumps(
        {"question": question, "answer": answer, "evidence": sorted(evidence_doc_ids)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "tk-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def base_mixture(template_names: list[str], rng: np.random.Generator) -> dict[str, float]:
    """Template mixture weights drawn deterministically from the generation rng.

    The +0.5 floor keeps every enabled template minting (no template starves
    on an unlucky draw); normalization makes the dict directly usable as a
    categorical distribution.
    """
    raw = 0.5 + rng.random(len(template_names))
    total = float(raw.sum())
    return {name: float(w) / total for name, w in zip(template_names, raw)}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if len(s.strip()) >= _MIN_SENTENCE_CHARS]


def _strip_article(entity: str) -> str:
    return _ARTICLE_RE.sub("", entity).strip()


def _lower_article(entity: str) -> str:
    """"The Alto Bridge" -> "the Alto Bridge" so it reads mid-question."""
    m = _ARTICLE_RE.match(entity)
    if m is None:
        return entity
    return m.group(1).lower() + entity[len(m.group(1)):]


def _doc_entities(text: str) -> list[str]:
    """Distinct multi-word entities in document order (deterministic)."""
    seen = dict.fromkeys(_strip_article(e) for e in ENTITY_RE.findall(text))
    return [e for e in seen if len(e) >= 4]


def _quantity_aliases(value: str) -> tuple[str, ...]:
    """Mechanical variants a grader should also accept ("12,000" -> "12000")."""
    stripped = value.replace(",", "").replace(" ", "")
    return (stripped,) if stripped != value else ()


def _sample_doc(corpus: "CorpusStore", rng: np.random.Generator) -> "Document | None":
    """Sample one document uniformly over the sorted doc-id universe.

    Sorting is defensive: the protocol does not promise ``iter_doc_ids``
    ordering, and unsorted iteration would break cross-validator determinism.
    """
    doc_ids = sorted(corpus.iter_doc_ids())
    if not doc_ids:
        return None
    return corpus.get(doc_ids[int(rng.integers(len(doc_ids)))])


def _make_task(
    question: str,
    answer: str,
    aliases: tuple[str, ...],
    evidence_doc_ids: tuple[str, ...],
    masked_doc_ids: tuple[str, ...],
    template: str,
    hops: int,
) -> Task:
    return Task(
        task_id=content_task_id(question, answer, evidence_doc_ids),
        question=question,
        answer=answer,
        aliases=aliases,
        evidence_doc_ids=evidence_doc_ids,
        masked_doc_ids=masked_doc_ids,
        origin=TaskOrigin.GENERATED_PUBLIC,
        template=template,
        hops=hops,
    )


def _fact_from_sentence(sentence: str) -> tuple[str, str, tuple[str, ...], str | None] | None:
    """Extract (question, answer, aliases, entity) from one sentence, or None.

    Prefers the relation shape (natural question phrasing); falls back to a
    cloze blank over a year or quantity when the sentence states a fact in a
    form the relation regex cannot rephrase.
    """
    m = _RELATION_RE.search(sentence)
    if m is not None:
        entity = _strip_article(m.group("ent"))
        if len(entity) >= 3 and YEAR_RE.fullmatch(entity) is None:
            question = f"In which year {m.group('aux')} {_lower_article(m.group('ent'))} {m.group('pred')}?"
            return question, m.group("year"), (), entity
    values = YEAR_RE.findall(sentence) or QUANTITY_RE.findall(sentence)
    entities = _doc_entities(sentence)
    for value in values:
        for entity in entities:
            if value in entity:
                continue
            blanked = sentence.replace(value, "____", 1)
            question = f'Fill in the blank using the corpus: "{blanked}"'
            aliases = _quantity_aliases(value) if YEAR_RE.fullmatch(value) is None else ()
            return question, value, aliases, entity
    return None


class EntityFactTemplate:
    """Single-hop fact extraction: (entity, attribute, value) from one document.

    Masking rule: the sole evidence document is the only guaranteed answer
    source, so masking it would make the task unsolvable — ``masked_doc_ids``
    is always empty (mask-exempt; :mod:`epago.taskgen.qa` knows this).
    """

    name = "entity_fact"

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        for _ in range(_DOC_TRIES):
            doc = _sample_doc(corpus, rng)
            if doc is None:
                return None
            sentences = _sentences(doc.text)
            for idx in rng.permutation(len(sentences)):
                fact = _fact_from_sentence(sentences[int(idx)])
                if fact is None:
                    continue
                question, answer, aliases, _entity = fact
                return _make_task(
                    question=question,
                    answer=answer,
                    aliases=aliases,
                    evidence_doc_ids=(doc.doc_id,),
                    masked_doc_ids=(),
                    template=self.name,
                    hops=1,
                )
        return None


class CrossDocJoinTemplate:
    """Two-hop join: entity found in document A, fact about it stated in B.

    The question names the entity and points at document A as the linking
    context; answering requires locating the fact in B (or an equivalent
    source) via search. Masking rule: document A (the linking document) is
    masked so the pair cannot be co-retrieved trivially; document B stays
    searchable because it is the only guaranteed answer source.
    """

    name = "cross_doc_join"
    _ENTITY_TRIES = 5
    _SEARCH_K = 5

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        for _ in range(_DOC_TRIES):
            doc_a = _sample_doc(corpus, rng)
            if doc_a is None:
                return None
            entities = _doc_entities(doc_a.text)
            if not entities:
                continue
            order = rng.permutation(len(entities))[: self._ENTITY_TRIES]
            for idx in order:
                entity = entities[int(idx)]
                for hit in corpus.search(entity, k=self._SEARCH_K):
                    if hit.doc_id == doc_a.doc_id:
                        continue
                    doc_b = corpus.get(hit.doc_id)
                    if doc_b is None or entity.lower() not in doc_b.text.lower():
                        continue
                    task = self._join_task(doc_a, doc_b, entity, rng)
                    if task is not None:
                        return task
        return None

    def _join_task(
        self, doc_a: "Document", doc_b: "Document", entity: str, rng: np.random.Generator
    ) -> Task | None:
        sentences = [s for s in _sentences(doc_b.text) if entity.lower() in s.lower()]
        for idx in rng.permutation(len(sentences)):
            fact = _fact_from_sentence(sentences[int(idx)])
            if fact is None:
                continue
            question, answer, aliases, fact_entity = fact
            # The fact must be about the join entity, not a bystander in B.
            if fact_entity is None or entity.lower() not in fact_entity.lower():
                continue
            question = (
                f"{question} ({entity} is also mentioned in the document titled "
                f'"{doc_a.title}".)'
            )
            return _make_task(
                question=question,
                answer=answer,
                aliases=aliases,
                evidence_doc_ids=(doc_b.doc_id, doc_a.doc_id),
                masked_doc_ids=(doc_a.doc_id,),
                template=self.name,
                hops=2,
            )
        return None


class AggregationTemplate:
    """Count distinct years across the top documents about one entity.

    The question enumerates its document universe by title, which makes the
    count well-defined without reference to any particular search backend.
    The answer ("3") is not literal text in any evidence document, so
    :func:`recount_aggregation` gives QA a mechanical re-derivation instead.
    Masking rule: every named document is needed for the count, so
    ``masked_doc_ids`` is empty (mask-exempt).
    """

    name = "aggregation"
    _SEARCH_K = 6
    _MIN_DOCS = 2
    _MAX_DOCS = 4

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        for _ in range(_DOC_TRIES):
            seed_doc = _sample_doc(corpus, rng)
            if seed_doc is None:
                return None
            entities = _doc_entities(seed_doc.text)
            if not entities:
                continue
            entity = entities[int(rng.integers(len(entities)))]
            docs: list["Document"] = []
            for hit in corpus.search(entity, k=self._SEARCH_K):
                doc = corpus.get(hit.doc_id)
                if doc is not None and YEAR_RE.search(doc.text):
                    docs.append(doc)
                if len(docs) == self._MAX_DOCS:
                    break
            if len(docs) < self._MIN_DOCS:
                continue
            count = len({y for doc in docs for y in YEAR_RE.findall(doc.text)})
            if count == 0:
                continue
            titles = ", ".join(f'"{doc.title}"' for doc in docs)
            question = (
                f"Considering only the documents titled {titles}, how many distinct "
                f"four-digit years (1500-2099) appear in their combined text?"
            )
            aliases = (_NUMBER_WORDS[count],) if count < len(_NUMBER_WORDS) else ()
            return _make_task(
                question=question,
                answer=str(count),
                aliases=aliases,
                evidence_doc_ids=tuple(doc.doc_id for doc in docs),
                masked_doc_ids=(),
                template=self.name,
                hops=len(docs),
            )
        return None


def recount_aggregation(task: Task, corpus: "CorpusStore") -> str | None:
    """Mechanically re-derive an aggregation answer from its evidence docs.

    Returns ``None`` for non-aggregation tasks or when evidence is missing;
    QA compares the recount against the claimed answer since a count never
    appears literally in the source text.
    """
    if task.template != AggregationTemplate.name:
        return None
    texts = []
    for doc_id in task.evidence_doc_ids:
        doc = corpus.get(doc_id)
        if doc is None:
            return None
        texts.append(doc.text)
    return str(len({y for text in texts for y in YEAR_RE.findall(text)}))


# --- scientific-literature templates ------------------------------------------
#
# The R1 templates were written for a synthetic corpus of invented facts. Over
# real papers they degenerate: `aggregation` asks "how many distinct four-digit
# years appear in these documents", which is counting trivia rather than
# evidence extraction, and it dominated the mixture. These templates ask the
# question the product actually answers — locate the study, report what it
# found — so what the duel measures is what a customer would pay for.

#: Values a scientific abstract reports as findings. Percentages, effect sizes,
#: p-values, sample sizes and confidence bounds — deliberately NOT bare years,
#: which are publication metadata rather than results.
_FINDING_VALUE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?%"                      # 94.37%  12 %
    r"|\bp\s?[<=>]\s?0?\.\d+"                    # p < 0.001
    r"|\b(?:n|N)\s?=\s?\d{1,3}(?:,\d{3})*\b"     # n = 1,284
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"         # 12,000
    r"|\b\d+\.\d+\b"                             # 0.82   3.4
)


@cache
def _compile_result_cues(cues: tuple[str, ...]) -> re.Pattern[str]:
    """``\\b(?:cue|cue|...)\\w*\\b`` — the ``\\w*`` covers plurals/inflections."""
    return re.compile(r"\b(?:" + "|".join(cues) + r")\w*\b", re.IGNORECASE)


@dataclass(frozen=True)
class FindingVocabulary:
    """The domain words a finding-based template reads a corpus through.

    The template *mechanics* — blank a reported value, describe rather than name
    the study, compare a quantity across studies — are domain-free; only the
    word lists that decide "is this sentence a result?" and "is this quantity
    comparable across studies?" carry a field. Binding those lists to the
    template instance instead of reading module globals is what lets one set of
    templates serve a medicine slice and all of scientific literature at once.

    Frozen and tuple-valued so it is hashable: a release binds one vocabulary,
    and the template instances for that vocabulary are built once and reused.
    """

    #: Stable key, recorded in manifests and used to look up template sets.
    name: str
    #: Words marking a sentence as a *result*; matched with a ``\\w*`` suffix.
    result_cues: tuple[str, ...]
    #: Quantities worth comparing across studies (``comparative_finding``).
    #: Fixed per release: a comparison is only meaningful when every matched
    #: sentence talks about the same kind of quantity.
    comparable_cues: tuple[str, ...]

    @property
    def result_cue_re(self) -> re.Pattern[str]:
        return _compile_result_cues(self.result_cues)


#: The medicine-slice vocabulary. These are the exact words R1/SCI1/SCI2 have
#: always used; they are a determinism contract for those releases and must not
#: change — a new word list ships as a new vocabulary bound to a new release.
MEDICAL_VOCABULARY = FindingVocabulary(
    name="medical",
    result_cues=(
        "result", "found", "showed", "reported", "observed", "associated",
        "significant", "increase", "decrease", "reduction", "risk", "rate",
        "prevalence", "incidence", "mortality", "odds", "hazard", "mean", "median",
    ),
    comparable_cues=(
        "mortality",
        "prevalence",
        "incidence",
        "sensitivity",
        "specificity",
        "seropositivity",
        "response rate",
        "success rate",
        "complication",
        "recurrence",
    ),
)

#: All of scientific literature. A superset of the medical cues, not a
#: replacement: life and health sciences are part of "all science", so dropping
#: `mortality` or `prevalence` would blind the release to a whole discipline.
#: What is added is the cross-field vocabulary a physics, CS, chemistry,
#: economics or ecology abstract reports its numbers with.
GENERAL_VOCABULARY = FindingVocabulary(
    name="general",
    result_cues=MEDICAL_VOCABULARY.result_cues
    + (
        # how a result is announced outside the clinical-trial register
        "achieved", "attained", "demonstrated", "measured", "estimated",
        "obtained", "yielded", "exhibited", "improved", "outperformed",
        "revealed", "indicated", "predicted", "detected", "correlated",
        "compared", "averaged", "ranged", "exceeded",
        # what is being reported: quantities that cross every discipline
        "accuracy", "precision", "recall", "efficiency", "efficacy",
        "performance", "throughput", "error", "yield", "correlation",
        "ratio", "improvement", "growth", "conversion", "purity",
        "coverage", "agreement", "uncertainty", "deviation", "variance",
        "average", "threshold", "magnitude", "density", "concentration",
        "capacity", "resolution", "latency", "speedup", "score",
        "proportion", "percentage", "fraction", "probability", "coefficient",
    ),
    comparable_cues=(
        # field-neutral quantities that are conventionally reported as a %
        "accuracy",
        "precision",
        "recall",
        "efficiency",
        "yield",
        "error rate",
        "success rate",
        "conversion",
        "purity",
        "coverage",
        "agreement",
        "growth rate",
        "adherence",
        "survival",
        # ...and the biomedical ones, because biomedicine is still science
        "mortality",
        "prevalence",
        "incidence",
        "sensitivity",
        "specificity",
        "seropositivity",
        "response rate",
        "complication",
        "recurrence",
    ),
)

#: Vocabulary by name, for CLIs and manifests (see scripts/harvest_holdout.py).
VOCABULARIES: dict[str, FindingVocabulary] = {
    v.name: v for v in (MEDICAL_VOCABULARY, GENERAL_VOCABULARY)
}

#: Back-compat: the module-level names the medical vocabulary used to be.
_RESULT_CUE_RE = MEDICAL_VOCABULARY.result_cue_re

_TITLE_MIN_CHARS = 25
_TITLE_MAX_CHARS = 180


def finding_sentences(
    text: str, vocab: FindingVocabulary = MEDICAL_VOCABULARY
) -> list[str]:
    """Sentences that both report a result and carry a checkable number."""
    cue_re = vocab.result_cue_re
    out = []
    for sentence in _sentences(text):
        if len(sentence) > 400:
            continue
        if not cue_re.search(sentence):
            continue
        if _FINDING_VALUE_RE.search(sentence):
            out.append(sentence)
    return out


#: Historical private name; kept because scripts import it. The default
#: vocabulary preserves the exact pre-SCI3 behaviour for one-argument calls.
_finding_sentences = finding_sentences


class StudyFindingTemplate:
    """Locate a named study, then extract a value it reports.

    The title is the locator, not the answer: the model still has to search for
    that paper, open it, and read the right sentence. That is the shape of the
    real job — find the source, report what it says — and unlike a corpus-wide
    cloze it has exactly one defensible answer.

    Masking rule: the study named in the question is the only place the value
    appears, so masking it would make the task unsolvable. Mask-exempt.
    """

    name = "study_finding"

    def __init__(self, vocab: FindingVocabulary = MEDICAL_VOCABULARY) -> None:
        self.vocab = vocab

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        for _ in range(_DOC_TRIES):
            doc = _sample_doc(corpus, rng)
            if doc is None:
                return None
            title = (doc.title or "").strip()
            if not (_TITLE_MIN_CHARS <= len(title) <= _TITLE_MAX_CHARS):
                continue
            candidates = finding_sentences(doc.text, self.vocab)
            if not candidates:
                continue
            for idx in rng.permutation(len(candidates)):
                sentence = candidates[int(idx)]
                values = _FINDING_VALUE_RE.findall(sentence)
                if not values:
                    continue
                # The first reported value is the one the sentence is about;
                # later ones are usually the comparator or the CI.
                value = values[0].strip()
                if len(value) < 2:
                    continue
                blanked = sentence.replace(value, "____", 1)
                if "____" not in blanked:
                    continue
                question = (
                    f'In the study titled "{title}", what value belongs in the blank? '
                    f'"{blanked}"'
                )
                return _make_task(
                    question=question,
                    answer=value,
                    aliases=_quantity_aliases(value),
                    evidence_doc_ids=(doc.doc_id,),
                    masked_doc_ids=(),
                    template=self.name,
                    hops=1,
                )
        return None


class FindTheStudyTemplate:
    """Provenance: given a reported finding, name the study that reports it.

    This is the citation task, and the one nobody else in the market sells. The
    answer is the study's title rather than its internal doc id — a title is
    text a reader can verify, it appears in the corpus, and it is what a
    citation actually is.

    Masking rule: the source document must stay visible or the title cannot be
    found. Mask-exempt.
    """

    name = "find_the_study"

    def __init__(self, vocab: FindingVocabulary = MEDICAL_VOCABULARY) -> None:
        self.vocab = vocab

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        for _ in range(_DOC_TRIES):
            doc = _sample_doc(corpus, rng)
            if doc is None:
                return None
            title = (doc.title or "").strip()
            if not (_TITLE_MIN_CHARS <= len(title) <= constants.ANSWER_MAX_CHARS):
                continue
            candidates = finding_sentences(doc.text, self.vocab)
            if not candidates:
                continue
            idx = int(rng.integers(0, len(candidates)))
            sentence = candidates[idx]
            question = (
                "Which study in the corpus reports the following? "
                f'"{sentence}" Answer with the study title.'
            )
            return _make_task(
                question=question,
                answer=title,
                aliases=(),
                evidence_doc_ids=(doc.doc_id,),
                masked_doc_ids=(),
                template=self.name,
                hops=1,
            )
        return None


# --- SCI2: templates that are hard by construction ----------------------------
#
# Measured on a real 503-paper corpus, the SCI1 templates leak their own
# answers: `study_finding` hands the model the full title (rank-1 retrieval
# 96.9%) and `find_the_study` quotes the finding sentence verbatim (90.9%).
# A verbatim span from a paper is a near-unique string, so full-text search
# lands on the source in one shot no matter how large the corpus grows —
# corpus size does not fix this, only removing the quotes does.
#
# SCI2 rules, applied to every template below:
#   1. never put a full title in a question;
#   2. never quote more than a few words of any sentence;
#   3. prefer locator words that match SEVERAL documents, so the model must
#      open candidates and disambiguate rather than trust rank 1.

#: Longest verbatim fragment a question may quote around a blank. Long enough
#: to make the extraction unambiguous once the right paper is open, short
#: enough to be a poor search key.
_CLAIM_WINDOW_WORDS = 4

#: Back-compat: the comparable quantities are now a field of the release's
#: :class:`FindingVocabulary`; this name is what SCI2 has always used.
_COMPARABLE_CUES = MEDICAL_VOCABULARY.comparable_cues

_PERCENT_RE = re.compile(r"\b(\d{1,2}(?:\.\d+)?)\s?%")
#: "95% CI"/"95% confidence interval" is interval notation, not a finding —
#: on a real corpus it matched constantly and manufactured ties at 95.0.
_CI_RE = re.compile(r"\d{1,2}(?:\.\d+)?\s?%\s?(?:CI\b|confidence)", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
_TITLE_STOP = frozenset(
    "the a an of and or in on for to with by from study review analysis "
    "systematic meta report case series trial randomized controlled".split()
)


def _ambiguous_topic_words(
    title: str, corpus: "CorpusStore", rng: np.random.Generator, want: int = 3
) -> list[str]:
    """Title words shared with MANY other documents — deliberately ambiguous.

    A full title is a unique string and therefore a perfect search key. Even a
    conjunction of rare title words can be one. So the descriptor prefers the
    title's most COMMON topical words: searching them returns a crowd of
    candidates, and telling the described study apart from its neighbours has
    to happen by reading, which is the skill being tested.
    """
    words = []
    seen: set[str] = set()
    for w in _WORD_RE.findall(title):
        lw = w.lower()
        if lw in _TITLE_STOP or lw in seen:
            continue
        seen.add(lw)
        words.append((lw, len(corpus.search(lw, k=12))))
    crowd = [(lw, df) for lw, df in words if df >= 4]
    if len(crowd) < 2:
        crowd = [(lw, df) for lw, df in words if df >= 2]
    # Most common first; title order breaks ties so the pick is deterministic.
    crowd.sort(key=lambda x: -x[1])
    return [lw for lw, _ in crowd[:want]]


_DIGIT_RUN_RE = re.compile(r"\d[\d,\.]*")


def _claim_window(sentence: str, value: str) -> str | None:
    """``<=4 words`` of context either side of the blanked value.

    Every OTHER number in the window is masked to ``N``: numbers are unique
    tokens, so a stray "556" or "390" in the quoted context is a perfect
    search key and re-opens the exact leak this template exists to close.
    The structure survives ("N cases studied, N (____) were males"), which is
    enough to confirm the extraction once the right paper is open.
    """
    blanked = sentence.replace(value, "\x00", 1)
    if "\x00" not in blanked:
        return None
    blanked = _DIGIT_RUN_RE.sub("N", blanked).replace("\x00", "____")
    tokens = blanked.split()
    try:
        at = next(i for i, t in enumerate(tokens) if "____" in t)
    except StopIteration:
        return None
    lo = max(0, at - _CLAIM_WINDOW_WORDS)
    hi = min(len(tokens), at + _CLAIM_WINDOW_WORDS + 1)
    prefix = "... " if lo > 0 else ""
    suffix = " ..." if hi < len(tokens) else ""
    return prefix + " ".join(tokens[lo:hi]) + suffix


class DescribedFindingTemplate:
    """Extraction where the study is described, never named.

    The locator is a handful of topic words each shared with other papers, so
    the search returns a field of candidates; the quoted context around the
    blank is capped at a few words, so it cannot serve as a search key either.
    The model has to open candidates, decide which one is the described study,
    and only then extract the value.

    Mask-exempt: the source document must stay visible to be identifiable.
    """

    name = "described_finding"

    def __init__(self, vocab: FindingVocabulary = MEDICAL_VOCABULARY) -> None:
        self.vocab = vocab

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        for _ in range(_DOC_TRIES):
            doc = _sample_doc(corpus, rng)
            if doc is None:
                return None
            title = (doc.title or "").strip()
            if len(title) < _TITLE_MIN_CHARS:
                continue
            topics = _ambiguous_topic_words(title, corpus, rng)
            if len(topics) < 2:
                continue
            candidates = finding_sentences(doc.text, self.vocab)
            for idx in rng.permutation(len(candidates)):
                sentence = candidates[int(idx)]
                values = _FINDING_VALUE_RE.findall(sentence)
                if not values:
                    continue
                value = values[0].strip()
                window = _claim_window(sentence, value)
                if window is None:
                    continue
                described = ", ".join(topics)
                question = (
                    f"A study in this corpus concerns: {described}. "
                    f'In that study, what value belongs in the blank? "{window}"'
                )
                return _make_task(
                    question=question,
                    answer=value,
                    aliases=_quantity_aliases(value),
                    evidence_doc_ids=(doc.doc_id,),
                    masked_doc_ids=(),
                    template=self.name,
                    hops=1,
                )
        return None


class ComparativeFindingTemplate:
    """Which study reports the highest/lowest rate of X?

    Multi-document by construction: the question quotes nothing from any paper
    — only a cue word like "mortality" — so no single search can answer it.
    The model must find every study reporting that quantity, read each value,
    and compare. This is literally the evidence-synthesis job.

    Determinism and auditability: matching is a fixed rule (a percentage in the
    same sentence as the cue), ties are discarded at mint time, and the answer
    is the winning study's title, which is literal text in its own document so
    the standard QA evidence check holds.

    Mask-exempt: hiding any matched study would change the comparison.

    RETIRED from the live release. It is in SCI2's frozen tuple and nowhere
    else: measured on the pinned all-science corpus, the reference model
    scored 0% on it under every protocol tried, because the agent's tools
    (top-k search, browse-by-id) cannot enumerate the field this mint defines
    by scanning the whole corpus. See the SCI3 entry in :data:`RELEASES` for
    the full reasoning and for the shape a future, answerable version would
    take.
    """

    name = "comparative_finding"

    #: Bounds on the comparison field. Below 3 the task is a coin flip between
    #: two candidates; above 12 the rollout budget goes on reading, not skill.
    _MIN_FIELD = 3
    _MAX_FIELD = 12

    def __init__(self, vocab: FindingVocabulary = MEDICAL_VOCABULARY) -> None:
        self.vocab = vocab

    def _field(self, corpus: "CorpusStore", cue: str) -> list[tuple[str, str, float]]:
        """(doc_id, title, value) for every doc stating a % beside the cue."""
        out: list[tuple[str, str, float]] = []
        for hit in corpus.search(cue, k=50):
            doc = corpus.get(hit.doc_id)
            if doc is None:
                continue
            best: float | None = None
            for sentence in _sentences(doc.text):
                if cue not in sentence.lower():
                    continue
                m = _PERCENT_RE.search(_CI_RE.sub(" ", sentence))
                if m:
                    v = float(m.group(1))
                    best = v if best is None else max(best, v)
            if best is not None and (doc.title or "").strip():
                out.append((doc.doc_id, doc.title.strip(), best))
        return out

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        cues = list(self.vocab.comparable_cues)
        for idx in rng.permutation(len(cues)):
            cue = cues[int(idx)]
            field = self._field(corpus, cue)
            if not (self._MIN_FIELD <= len(field) <= self._MAX_FIELD):
                continue
            values = [v for _, _, v in field]
            top = max(values)
            if values.count(top) != 1:
                continue  # a tie has no single defensible answer
            winner = max(field, key=lambda x: x[2])
            if len(winner[1]) > constants.ANSWER_MAX_CHARS:
                continue
            question = (
                f"Among the studies in this corpus that report {cue} as a "
                f"percentage, which one reports the highest {cue}? "
                "Answer with the study title."
            )
            return _make_task(
                question=question,
                answer=winner[1],
                aliases=(),
                evidence_doc_ids=tuple(d for d, _, _ in field),
                masked_doc_ids=(),
                template=self.name,
                hops=len(field),
            )
        return None


def _template_set(vocab: FindingVocabulary) -> dict[str, TaskTemplate]:
    """One instance of every template family, bound to ``vocab``.

    Templates hold no state beyond their vocabulary, so a set can be built once
    and shared by every release that binds the same words.
    """
    return {
        t.name: t
        for t in (
            EntityFactTemplate(),
            CrossDocJoinTemplate(),
            AggregationTemplate(),
            StudyFindingTemplate(vocab),
            FindTheStudyTemplate(vocab),
            DescribedFindingTemplate(vocab),
            ComparativeFindingTemplate(vocab),
            ConstrainedStudyTemplate(vocab),
            NamedSetSuperlativeTemplate(vocab),
            NamedSetCountTemplate(vocab),
        )
    }


# Release name -> enabled template names. A release is part of the chain
# contract (``eval.taskgen_release`` in chain.toml): changing the tuple for an
# existing release would silently change every validator's holdouts, so new
# template sets always ship as new release names.
RELEASES: dict[str, tuple[str, ...]] = {
    "R1": ("entity_fact", "cross_doc_join", "aggregation"),
    # Scientific literature. `aggregation` is deliberately absent: over real
    # papers it mints "how many distinct years appear in these documents",
    # which measures counting rather than comprehension and crowded out
    # everything else in the R1 mixture.
    "SCI1": ("study_finding", "find_the_study", "cross_doc_join"),
    # SCI2 exists because SCI1 leaks its own answers: naming the study or
    # quoting a full sentence hands full-text search the source at rank 1
    # (measured 96.9% / 90.9% on a real corpus). These describe instead of
    # name, cap quotes at a few words, and add a comparison task no single
    # search can answer.
    "SCI2": ("described_finding", "comparative_finding", "cross_doc_join"),
    # SCI3 is SCI2's task shapes with the corpus widened from a medicine slice
    # to all of scientific literature, minus `comparative_finding`. The
    # vocabulary is what changes, because the mechanics were never medical:
    # what was medical was the word list deciding "is this a result?" and "is
    # this quantity comparable?".
    #
    # `comparative_finding` is NOT in this tuple because it is unanswerable
    # with the tools the agent actually has. It asks "of every study in the
    # corpus reporting X, which reports the highest X?", but the environment
    # offers top-k search plus browse-by-id — there is no way to enumerate a
    # field the mint defined by scanning all 5792 documents offline (it uses
    # k=50 internally; the agent's session returns 10). Measured on the pinned
    # corpus with the reference model: 0% accuracy in all six protocol
    # variants tried (raw completion, chat template at two token budgets, and
    # native tool-calling at 84-87% overall accuracy), on every replicate. A
    # task no protocol can solve contributes no signal to a duel — both models
    # score zero on it, so its paired difference is always 0 — while consuming
    # its share of the rollout budget and depressing the king's measured
    # accuracy. That last part is not cosmetic: delta, the bar a challenger
    # must clear, is DELTA_C * (1 - king_acc_ema), so unsolvable tasks raise
    # the winning bar for every honest challenger.
    #
    # The class stays in this module, and in SCI2's frozen tuple, so history
    # replays exactly. A future release could make the shape answerable rather
    # than retire it, by pinning the comparison universe in the question the
    # way `aggregation` already does ("among documents A, B, C, which reports
    # the highest X?"): that is browsable and needs no corpus-wide
    # enumeration. That is a new template and a new release name, never an
    # edit to a pinned one.
    "SCI3": ("described_finding", "cross_doc_join"),
    # SCI4 replaces the cloze shapes outright: measured on the pinned corpus
    # (2026-08-18), SCI3's question pins its own evidence document (topic
    # words rank it #1 for 88.3% of tasks) and better retrieval leaks the
    # answer into the snippet page for 74.1% — a change proven better on an
    # external benchmark (FRAMES +7.3pp, p=0.002) moved SCI3 by nothing
    # (-1.1pp, p=0.49). SCI4 is the BrowseComp shape — hard to find, easy to
    # check: multi-constraint retrieval where only the conjunction is unique,
    # and computed answers (superlative, count) over a title-pinned set. See
    # the SCI4 section header for the full rules.
    "SCI4": ("constrained_study", "named_set_superlative", "named_set_count"),
}

# Release name -> the vocabulary its templates read with. Part of the same
# determinism contract as RELEASES: an existing release's binding is frozen,
# and a new word list ships as a new release. Releases minted before the
# vocabulary was bindable read the medicine-slice words.
RELEASE_VOCABULARY: dict[str, FindingVocabulary] = {
    "R1": MEDICAL_VOCABULARY,  # no finding template; binding is inert
    "SCI1": MEDICAL_VOCABULARY,
    "SCI2": MEDICAL_VOCABULARY,
    "SCI3": GENERAL_VOCABULARY,
    "SCI4": GENERAL_VOCABULARY,
}

# Templates whose masking rule legitimately leaves masked_doc_ids empty (see
# module docstring); QA's masking-sanity check consults this set.
MASK_EXEMPT_TEMPLATES: frozenset[str] = frozenset(
    {
        "entity_fact",
        "aggregation",
        "study_finding",
        "find_the_study",
        "described_finding",
        "comparative_finding",
        "constrained_study",
        "named_set_superlative",
        "named_set_count",
    }
)


def vocabulary_for_release(release: str) -> FindingVocabulary:
    """The word lists ``release`` reads a corpus with (medicine-slice default)."""
    if release not in RELEASES:
        raise ValueError(f"unknown taskgen release: {release!r} (known: {sorted(RELEASES)})")
    return RELEASE_VOCABULARY.get(release, MEDICAL_VOCABULARY)


def templates_for_release(release: str) -> tuple[TaskTemplate, ...]:
    vocab = vocabulary_for_release(release)  # also validates the release name
    templates = _TEMPLATE_SETS.get(vocab)
    if templates is None:  # a caller-defined vocabulary; build and remember it
        templates = _TEMPLATE_SETS.setdefault(vocab, _template_set(vocab))
    return tuple(templates[name] for name in RELEASES[release])


# --- SCI4: hard to find, easy to check ----------------------------------------
#
# Measured on the pinned all-science corpus (2026-08-18), the
# SCI2/SCI3 shapes are cloze tests: the question itself pins the evidence
# document (topic words alone rank it #1 for 88.3% of tasks) and the answer
# sits verbatim in the search snippet (74.1% under working search), so 28.6%
# of episodes answer after one search with zero documents opened. No tool
# change fixes that; the task shape has to.
#
# SCI4 rules, applied to every template below:
#   1. the question must never contain a string that identifies the evidence
#      document — no titles (unless the universe is deliberately pinned), no
#      verbatim windows, and a mint-time leak self-check that runs the actual
#      question through the actual search backend and discards the task if the
#      gold document surfaces near the top;
#   2. the answer must be computed or located by elimination, never copied
#      from a snippet: either it appears in no document at all (counts,
#      superlatives), or reaching it requires identifying one document among a
#      crowd where every discriminating constraint is only checkable by
#      opening candidates;
#   3. every answer is defined by the frozen extraction rules in
#      :mod:`epago.taskgen.profiles` and is mechanically re-derivable
#      (:func:`recount_task`), so QA still needs no LLM.
#
# The constraint facts (field, largest cue percentage, largest stated n, word
# presence) come from :class:`epago.taskgen.profiles.ProfileIndex`. Field and
# domain labels are corpus metadata the model cannot browse, so they are only
# ever phrased as soft hints and NEVER load-bearing: uniqueness is always
# proven over text-verifiable constraints alone (ranges, mentions, negations).

from epago.taskgen import profiles as _profiles

#: A constraint alone must match at least this many documents (gold included);
#: anything rarer is a fingerprint of the gold document, not a filter.
_SCI4_MIN_ALONE = 5
_SCI4_MIN_CONSTRAINTS = 3
_SCI4_MAX_CONSTRAINTS = 5
#: Leak self-check: the assembled question, pasted into the real search
#: backend, must not surface the gold document in the top 3; each individual
#: constraint phrase must not rank it #1. Pre-registered before measurement.
_SCI4_QUESTION_LEAK_TOP = 3
#: Named-set families: bounds on the pinned universe, and the winner margin
#: (in percentage points) below which a superlative is too close to defend.
_SCI4_SET_MIN = 4
_SCI4_SET_MAX = 6
_SCI4_MARGIN = 0.5
_SCI4_TITLE_MIN = 25
_SCI4_TITLE_MAX = 160
_SCI4_SUBSET_TRIES = 6

#: The extraction rules, stated in the question so the answer is defensible.
_CUE_RULE = (
    "a study's {cue} value means the largest {cue} percentage stated anywhere "
    "in its text (ignoring confidence-interval notation)"
)
_N_RULE = "a study's sample size means the largest n = X it states"


@dataclass(frozen=True)
class _Constraint:
    """One filter over the corpus: a phrase plus exactly the docs satisfying it."""

    kind: str  # "cue_range" | "n_range" | "mention" | "negation"
    key: str  # cue word / mentioned word — dedup key within one task
    phrase: str
    members: frozenset[str]
    #: Range constraints anchor the research feel; at least one is required.
    is_range: bool


def _usable_title(title: str) -> bool:
    return (
        _SCI4_TITLE_MIN <= len(title) <= _SCI4_TITLE_MAX
        and "\n" not in title
        and not re.search(r"[<>{}\[\]`|]", title)
    )


def _fmt_pct(v: float) -> str:
    return f"{v:g}"


def _percent_window(
    values: list[float], gold: float, rng: np.random.Generator
) -> tuple[float, float] | None:
    """An inclusive [lo, hi] around ``gold`` covering >=3 corpus values.

    Bounds are snapped to one decimal just outside the covered neighbours, so
    membership is exactly reproducible from the phrase alone.
    """
    import bisect
    import math as _math

    vs = sorted(values)
    i = bisect.bisect_left(vs, gold)
    if i >= len(vs) or vs[i] != gold:
        return None
    lo_i = max(0, i - int(rng.integers(1, 3)))
    hi_i = min(len(vs) - 1, i + int(rng.integers(1, 3)))
    lo = _math.floor(vs[lo_i] * 10) / 10
    hi = _math.ceil(vs[hi_i] * 10) / 10
    covered = sum(1 for v in vs if lo <= v <= hi)
    if covered < 3 or covered == len(vs):
        return None
    return lo, hi


def _int_window(
    values: list[int], gold: int, rng: np.random.Generator
) -> tuple[int, int] | None:
    import bisect

    vs = sorted(values)
    i = bisect.bisect_left(vs, gold)
    if i >= len(vs) or vs[i] != gold:
        return None
    lo_i = max(0, i - int(rng.integers(1, 3)))
    hi_i = min(len(vs) - 1, i + int(rng.integers(1, 3)))
    lo, hi = vs[lo_i], vs[hi_i]
    covered = sum(1 for v in vs if lo <= v <= hi)
    if covered < 3 or covered == len(vs):
        return None
    return lo, hi


class ConstrainedStudyTemplate:
    """Find the one study satisfying several crowd-sized constraints.

    The BrowseComp shape: each constraint alone matches a crowd
    (>= ``_SCI4_MIN_ALONE`` documents), only the conjunction is unique, and
    every load-bearing constraint is text-verifiable — checking it requires
    opening a candidate, not reading a snippet. Ask modes: name the study
    (answer = its title) or report one of its values that no constraint
    mentions (answer = a number that never appears in the question).

    Mask-exempt: the gold document must stay searchable to be findable.
    """

    name = "constrained_study"

    def __init__(self, vocab: FindingVocabulary = MEDICAL_VOCABULARY) -> None:
        self.vocab = vocab

    # -- candidate constraints -------------------------------------------------

    def _range_constraints(
        self, index: "_profiles.ProfileIndex", gold: "_profiles.DocProfile",
        rng: np.random.Generator,
    ) -> list[_Constraint]:
        out: list[_Constraint] = []
        for cue in sorted(gold.cue_values):
            pairs = index.by_cue.get(cue, ())
            window = _percent_window([v for _, v in pairs], gold.cue_values[cue], rng)
            if window is None:
                continue
            lo, hi = window
            members = frozenset(d for d, v in pairs if lo <= v <= hi)
            if len(members) < _SCI4_MIN_ALONE - 2:  # ranges may run narrower
                continue
            out.append(
                _Constraint(
                    kind="cue_range",
                    key=cue,
                    phrase=(
                        f"its {cue} value is between {_fmt_pct(lo)}% and "
                        f"{_fmt_pct(hi)}% (inclusive)"
                    ),
                    members=members,
                    is_range=True,
                )
            )
        if gold.n_value is not None:
            all_n = [
                (d, p.n_value)
                for d, p in sorted(index.profiles.items())
                if p.n_value is not None
            ]
            window = _int_window([n for _, n in all_n], gold.n_value, rng)
            if window is not None:
                lo, hi = window
                members = frozenset(d for d, n in all_n if lo <= n <= hi)
                if len(members) >= _SCI4_MIN_ALONE - 2:
                    out.append(
                        _Constraint(
                            kind="n_range",
                            key="n",
                            phrase=f"its sample size is between {lo} and {hi} (inclusive)",
                            members=members,
                            is_range=True,
                        )
                    )
        return out

    def _word_constraints(
        self, index: "_profiles.ProfileIndex", gold: "_profiles.DocProfile",
        rng: np.random.Generator,
    ) -> list[_Constraint]:
        out: list[_Constraint] = []
        n_docs = len(index.profiles)
        mention_pool = sorted(
            w
            for w in gold.words
            if _SCI4_MIN_ALONE <= index.word_df[w] <= n_docs // 3
        )
        for idx in rng.permutation(len(mention_pool))[:2]:
            w = mention_pool[int(idx)]
            members = frozenset(
                d for d, p in index.profiles.items() if w in p.words
            )
            out.append(
                _Constraint(
                    kind="mention", key=w,
                    phrase=f'its text mentions the word "{w}"',
                    members=members, is_range=False,
                )
            )
        field_df = index.field_word_df(gold.field) if gold.field else None
        if field_df:
            crowd = len(index.by_field.get(gold.field, ()))
            neg_pool = sorted(
                w
                for w, c in field_df.items()
                if c >= max(3, crowd // 4) and w not in gold.words
            )
            if neg_pool:
                w = neg_pool[int(rng.integers(len(neg_pool)))]
                members = frozenset(
                    d for d, p in index.profiles.items() if w not in p.words
                )
                out.append(
                    _Constraint(
                        kind="negation", key=w,
                        phrase=f'its text never mentions the word "{w}"',
                        members=members, is_range=False,
                    )
                )
        return out

    def _choose(
        self, pool: list[_Constraint], gold_id: str, universe: frozenset[str],
        rng: np.random.Generator,
    ) -> list[_Constraint] | None:
        """Greedy selection to a unique conjunction, one range guaranteed."""
        ranges = [c for c in pool if c.is_range]
        others = [c for c in pool if not c.is_range]
        if not ranges:
            return None
        first = ranges[int(rng.integers(len(ranges)))]
        rest = [c for c in ranges if c is not first] + others
        order = [rest[int(i)] for i in rng.permutation(len(rest))]
        chosen = [first]
        current = universe & first.members
        for c in order:
            if len(chosen) >= _SCI4_MAX_CONSTRAINTS:
                break
            if len(current) == 1 and len(chosen) >= _SCI4_MIN_CONSTRAINTS:
                break
            new = current & c.members
            if gold_id not in new or len(new) == len(current):
                continue
            chosen.append(c)
            current = new
        if len(current) != 1 or not (
            _SCI4_MIN_CONSTRAINTS <= len(chosen) <= _SCI4_MAX_CONSTRAINTS
        ):
            return None
        return chosen

    # -- leak self-check -------------------------------------------------------

    @staticmethod
    def _leaks(
        corpus: "CorpusStore", question: str, phrases: list[str], gold_id: str,
        answer_terms: tuple[str, ...],
    ) -> bool:
        hits = corpus.search(question, k=10)
        if gold_id in [h.doc_id for h in hits[:_SCI4_QUESTION_LEAK_TOP]]:
            return True
        if answer_terms:
            page = " ".join(f"{h.title} {h.snippet}" for h in hits).lower()
            # Digit-boundary guards: a bare "48" must not match inside "1,480",
            # or every small number would falsely count as leaked.
            for term in answer_terms:
                if re.search(
                    rf"(?<![\d.,]){re.escape(term.lower())}(?![\d.,%])", page
                ):
                    return True
        for phrase in phrases:
            top = corpus.search(phrase, k=1)
            if top and top[0].doc_id == gold_id:
                return True
        return False

    # -- mint ------------------------------------------------------------------

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        index = _profiles.profile_index(corpus, self.vocab.comparable_cues)
        universe = frozenset(index.profiles)
        for _ in range(_DOC_TRIES):
            doc = _sample_doc(corpus, rng)
            if doc is None:
                return None
            gold = index.profiles.get(doc.doc_id)
            if gold is None or not _usable_title(gold.title):
                continue
            pool = self._range_constraints(index, gold, rng) + self._word_constraints(
                index, gold, rng
            )
            chosen = self._choose(pool, gold.doc_id, universe, rng)
            if chosen is None:
                continue
            task = self._assemble(corpus, index, gold, chosen, rng)
            if task is not None:
                return task
        return None

    def _assemble(
        self, corpus: "CorpusStore", index: "_profiles.ProfileIndex",
        gold: "_profiles.DocProfile", chosen: list[_Constraint],
        rng: np.random.Generator,
    ) -> Task | None:
        used_cues = {c.key for c in chosen if c.kind == "cue_range"}
        used_n = any(c.kind == "n_range" for c in chosen)
        used_ranges = [
            c for c in chosen if c.kind == "cue_range" or c.kind == "n_range"
        ]

        def _bracketed(v: float) -> bool:
            """True when a chosen range constraint already brackets ``v`` —
            asking for such a value would hand the model a two-bound guess."""
            for c in used_ranges:
                m = re.search(r"between (\d+(?:\.\d+)?)%? and (\d+(?:\.\d+)?)", c.phrase)
                if m and float(m.group(1)) <= v <= float(m.group(2)):
                    return True
            return False

        # Ask mode: report a value no constraint mentions, else name the study.
        # (kind, ask sentence, answer, aliases)
        value_asks: list[tuple[str, str, str, tuple[str, ...]]] = []
        if gold.n_value is not None and not used_n and not _bracketed(gold.n_value):
            n = gold.n_value
            value_asks.append(
                (
                    "n",
                    "Find that study and report its sample size. Answer with the number.",
                    f"{n:,}",
                    (str(n),),
                )
            )
        for cue in sorted(set(gold.cue_values) - used_cues):
            v = gold.cue_values[cue]
            if _bracketed(v):
                continue
            value_asks.append(
                (
                    "cue",
                    f"Find that study and report its {cue} value. "
                    "Answer with the percentage.",
                    f"{_fmt_pct(v)}%",
                    (f"{_fmt_pct(v)} %", _fmt_pct(v)),
                )
            )
        if gold.top_percent is not None and not _bracketed(gold.top_percent):
            v = gold.top_percent
            value_asks.append(
                (
                    "top_pct",
                    "Find that study and report the largest percentage stated "
                    "anywhere in its text (ignoring confidence-interval "
                    "notation). Answer with the percentage.",
                    f"{_fmt_pct(v)}%",
                    (f"{_fmt_pct(v)} %", _fmt_pct(v)),
                )
            )
        ask_kind = "title"
        ask, answer, aliases = "Find that study. Answer with its exact title.", gold.title, ()
        if value_asks and rng.random() < 0.5:
            ask_kind, ask, answer, aliases = value_asks[int(rng.integers(len(value_asks)))]

        rules = []
        if used_cues or ask_kind == "cue":
            rules.append(_CUE_RULE.format(cue="X"))
        if used_n or ask_kind == "n":
            rules.append(_N_RULE)
        rule_text = (" (Here " + "; ".join(rules) + ".)") if rules else ""

        hint = ""
        if gold.field and len(index.by_field.get(gold.field, ())) >= _SCI4_MIN_ALONE:
            if rng.random() < 0.5:
                hint = f" As a hint, it is best described as a {gold.field} study."

        numbered = "; ".join(
            f"({i + 1}) {c.phrase}" for i, c in enumerate(chosen)
        )
        question = (
            f"Exactly one study in this corpus satisfies ALL of the following: "
            f"{numbered}.{rule_text}{hint} {ask}"
        )
        answer_terms = (answer, *aliases) if ask_kind != "title" else ()
        if self._leaks(corpus, question, [c.phrase for c in chosen], gold.doc_id, answer_terms):
            return None
        return _make_task(
            question=question,
            answer=answer,
            aliases=aliases,
            evidence_doc_ids=(gold.doc_id,),
            masked_doc_ids=(),
            template=self.name,
            hops=len(chosen),
        )


def _sample_named_set(
    index: "_profiles.ProfileIndex", cue: str, rng: np.random.Generator
) -> list[tuple[str, str, float]] | None:
    """4-6 distinct usable-title docs reporting ``cue``, deterministically."""
    entries = [
        (d, index.profiles[d].title, v)
        for d, v in index.by_cue.get(cue, ())
        if _usable_title(index.profiles[d].title)
    ]
    # A duplicated title makes "answer with the title" ambiguous; drop dupes.
    seen: dict[str, int] = {}
    for _, title, _ in entries:
        seen[title] = seen.get(title, 0) + 1
    entries = [e for e in entries if seen[e[1]] == 1]
    if len(entries) < _SCI4_SET_MIN:
        return None
    k = int(rng.integers(_SCI4_SET_MIN, min(_SCI4_SET_MAX, len(entries)) + 1))
    picked = [entries[int(i)] for i in rng.permutation(len(entries))[:k]]
    return picked


def _named_set_preamble(picked: list[tuple[str, str, float]], cue: str) -> str:
    titles = "; ".join(f'"{title}"' for _, title, _ in picked)
    rule = _CUE_RULE.format(cue=cue)
    return f"Consider exactly these studies: {titles}. Here {rule}."


class NamedSetSuperlativeTemplate:
    """Which of these named studies reports the highest/lowest X?

    The browsable version of the retired ``comparative_finding``: the
    comparison universe is pinned by title in the question (no corpus-wide
    enumeration needed), but WHICH study wins is stated nowhere — the model
    must open every named study, extract each value under the stated rule,
    and compare. Ties within ``_SCI4_MARGIN`` points are discarded at mint.

    Mask-exempt: every named document is needed for the comparison.
    """

    name = "named_set_superlative"

    def __init__(self, vocab: FindingVocabulary = MEDICAL_VOCABULARY) -> None:
        self.vocab = vocab

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        index = _profiles.profile_index(corpus, self.vocab.comparable_cues)
        cues = sorted(index.by_cue)
        for cue_idx in rng.permutation(len(cues)):
            cue = cues[int(cue_idx)]
            for _ in range(_SCI4_SUBSET_TRIES):
                picked = _sample_named_set(index, cue, rng)
                if picked is None:
                    break
                highest = bool(rng.integers(2))
                values = [v for _, _, v in picked]
                best = max(values) if highest else min(values)
                runner = sorted(values, reverse=highest)[1]
                if abs(best - runner) < _SCI4_MARGIN:
                    continue
                winner = next(e for e in picked if e[2] == best)
                direction = "highest" if highest else "lowest"
                question = (
                    f"{_named_set_preamble(picked, cue)} Which of these studies "
                    f"reports the {direction} {cue} value? "
                    "Answer with the exact study title."
                )
                return _make_task(
                    question=question,
                    answer=winner[1],
                    aliases=(),
                    evidence_doc_ids=tuple(d for d, _, _ in picked),
                    masked_doc_ids=(),
                    template=self.name,
                    hops=len(picked),
                )
        return None


class NamedSetCountTemplate:
    """How many of these named studies report X above a threshold?

    The answer is a count: it appears verbatim in no document and in no search
    snippet, so it must be computed by opening every named study. The
    threshold is an integer placed >= ``_SCI4_MARGIN`` points clear of every
    value, so extraction noise cannot flip the count. Verified mechanically by
    :func:`recount_named_set_count` (the mint asserts the recount round-trips).

    Mask-exempt: every named document is needed for the count.
    """

    name = "named_set_count"

    def __init__(self, vocab: FindingVocabulary = MEDICAL_VOCABULARY) -> None:
        self.vocab = vocab

    def mint(self, corpus: "CorpusStore", rng: np.random.Generator) -> Task | None:
        index = _profiles.profile_index(corpus, self.vocab.comparable_cues)
        cues = sorted(index.by_cue)
        for cue_idx in rng.permutation(len(cues)):
            cue = cues[int(cue_idx)]
            for _ in range(_SCI4_SUBSET_TRIES):
                picked = _sample_named_set(index, cue, rng)
                if picked is None:
                    break
                values = sorted(v for _, _, v in picked)
                split = int(rng.integers(1, len(values)))
                import math as _math

                lo_i = _math.ceil(values[split - 1] + _SCI4_MARGIN)
                hi_i = _math.floor(values[split] - _SCI4_MARGIN)
                if lo_i > hi_i:
                    continue
                threshold = int(lo_i + rng.integers(hi_i - lo_i + 1))
                count = sum(1 for v in values if v > threshold)
                if not 1 <= count <= len(values) - 1:
                    continue
                question = (
                    f"{_named_set_preamble(picked, cue)} How many of these "
                    f"studies report a {cue} value strictly greater than "
                    f"{threshold}%? Answer with a single number."
                )
                aliases = (
                    (_NUMBER_WORDS[count],) if count < len(_NUMBER_WORDS) else ()
                )
                task = _make_task(
                    question=question,
                    answer=str(count),
                    aliases=aliases,
                    evidence_doc_ids=tuple(d for d, _, _ in picked),
                    masked_doc_ids=(),
                    template=self.name,
                    hops=len(picked),
                )
                # The recount must round-trip at mint time or the grammar and
                # the recomputation have drifted — that is a bug, not bad luck.
                if recount_named_set_count(task, corpus) != task.answer:
                    return None
                return task
        return None


_COUNT_QUESTION_RE = re.compile(
    r"report a (?P<cue>.+?) value strictly greater than (?P<threshold>\d+)%"
)


def recount_named_set_count(task: Task, corpus: "CorpusStore") -> str | None:
    """Mechanically re-derive a named-set count from its evidence documents."""
    if task.template != NamedSetCountTemplate.name:
        return None
    m = _COUNT_QUESTION_RE.search(task.question)
    if m is None:
        return None
    cue, threshold = m.group("cue"), int(m.group("threshold"))
    count = 0
    for doc_id in task.evidence_doc_ids:
        doc = corpus.get(doc_id)
        if doc is None:
            return None
        value = _profiles.cue_percentages(doc.text, (cue,)).get(cue)
        if value is not None and value > threshold:
            count += 1
    return str(count)


def recount_task(task: Task, corpus: "CorpusStore") -> str | None:
    """Re-derive any computed (never-literal) answer; None for literal ones."""
    if task.template == AggregationTemplate.name:
        return recount_aggregation(task, corpus)
    if task.template == NamedSetCountTemplate.name:
        return recount_named_set_count(task, corpus)
    return None


#: Templates whose mint mechanically proves its own uniqueness (full profile
#: scan) or pins its universe by title — QA's lexical ambiguity heuristic is
#: strictly weaker than those guarantees and only misfires on them.
SELF_PINNED_TEMPLATES: frozenset[str] = frozenset(
    {
        AggregationTemplate.name,
        ConstrainedStudyTemplate.name,
        NamedSetSuperlativeTemplate.name,
        NamedSetCountTemplate.name,
    }
)


# Built here, after every template class in the module exists (the SCI4
# classes are defined below the registry block that used to build this).
_TEMPLATE_SETS: dict[FindingVocabulary, dict[str, TaskTemplate]] = {
    vocab: _template_set(vocab) for vocab in VOCABULARIES.values()
}

#: Back-compat: the medical set, which is what ``_TEMPLATES`` always held.
_TEMPLATES: dict[str, TaskTemplate] = _TEMPLATE_SETS[MEDICAL_VOCABULARY]
