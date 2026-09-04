"""Document profiles: the mechanical fact layer SCI4 constraints are built on.

A profile is everything a constraint may reference about one document —
its field and domain (parsed from ``category``), the largest percentage it
reports beside each cue word, its largest stated sample size, and its word
set. The extraction rules here are a determinism contract exactly like a
template tuple: a task's answer is *defined* by these rules (``recount`` in
:mod:`epago.taskgen.templates` re-derives answers with them), so changing a
rule for a shipped release would silently change every validator's holdouts.
New rules ship as new functions bound to new release names.

This module deliberately imports nothing from :mod:`epago.taskgen.templates`
(templates import *us*); the cue list arrives as a plain tuple of strings.
"""

from __future__ import annotations

import re
import weakref
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # protocol-only coupling to the environment subsystem
    from epago.environment.corpus import CorpusStore

# Percentages a scientific abstract reports as findings: 0-99.99 with an
# optional decimal part. Three-digit percentages are almost always growth
# multiples or index values, not proportions, so they are excluded.
_PERCENT_RE = re.compile(r"\b(\d{1,2}(?:\.\d+)?)\s?%")
#: "95% CI" / "95% confidence interval" is interval notation, not a finding.
_CI_RE = re.compile(r"\d{1,2}(?:\.\d+)?\s?%\s?(?:CI\b|confidence)", re.IGNORECASE)
#: "n = 1,284" / "N=96" — the stated size of a study group.
_N_RE = re.compile(r"\b[nN]\s?=\s?(\d{1,3}(?:,\d{3})+|\d+)\b")
_WORD_RE = re.compile(r"\b[a-z]{4,}\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class DocProfile:
    """The constraint-referenceable facts of one document."""

    doc_id: str
    title: str
    #: Topical field parsed from ``category`` ("microbiology", "economics", ...).
    field: str
    #: OpenAlex domain ("Life Sciences", "Physical Sciences", ...).
    domain: str
    #: cue word -> the largest percentage reported in a sentence containing it.
    cue_values: dict[str, float]
    #: Largest stated sample size (``n = X``), or None when never stated.
    n_value: int | None
    #: Largest percentage stated anywhere in the text (CI notation removed),
    #: or None. Cue-independent, so it exists for most quantitative papers.
    top_percent: float | None
    #: Lowercased words of length >= 4 (mention / negation constraints).
    words: frozenset[str]


def parse_category(category: str) -> tuple[str, str]:
    """``"openalex:d1:microbiology|Life Sciences"`` -> ``("microbiology", "Life Sciences")``."""
    left, _, domain = category.partition("|")
    field = left.rsplit(":", 1)[-1] if left else ""
    return field.strip(), domain.strip()


def cue_percentages(text: str, cues: tuple[str, ...]) -> dict[str, float]:
    """The largest percentage stated beside each cue word, by fixed rule.

    Rule (frozen): for each cue, over every sentence whose lowercase text
    contains the cue, take the maximum percentage after removing confidence-
    interval notation. "Largest" rather than "first" because abstracts state a
    headline value and then subgroup values; the headline is the largest in
    practice and the max is order-independent, which keeps the rule immune to
    sentence-splitting drift.
    """
    out: dict[str, float] = {}
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        lowered = sentence.lower()
        matched = [cue for cue in cues if cue in lowered]
        if not matched:
            continue
        cleaned = _CI_RE.sub(" ", sentence)
        values = [float(m) for m in _PERCENT_RE.findall(cleaned)]
        if not values:
            continue
        top = max(values)
        for cue in matched:
            if cue not in out or top > out[cue]:
                out[cue] = top
    return out


def top_percentage(text: str) -> float | None:
    """Largest percentage anywhere in the text, CI notation removed (frozen rule)."""
    values = [float(m) for m in _PERCENT_RE.findall(_CI_RE.sub(" ", text))]
    return max(values) if values else None


def sample_size(text: str) -> int | None:
    """Largest ``n = X`` in the document, or None (frozen rule, see above)."""
    sizes = [int(m.replace(",", "")) for m in _N_RE.findall(text)]
    return max(sizes) if sizes else None


def doc_words(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


class ProfileIndex:
    """Profiles for a whole corpus, plus the crowd lookups constraints need.

    Built once per (corpus, cues) pair and cached; iteration is in sorted
    doc-id order everywhere so downstream sampling is deterministic.
    """

    def __init__(self, corpus: "CorpusStore", cues: tuple[str, ...]) -> None:
        self.cues = cues
        self.profiles: dict[str, DocProfile] = {}
        self.by_field: dict[str, list[str]] = {}
        self.by_domain: dict[str, list[str]] = {}
        #: cue -> [(doc_id, value)] in sorted doc-id order.
        self.by_cue: dict[str, list[tuple[str, float]]] = {}
        self.word_df: Counter[str] = Counter()
        self._field_word_df: dict[str, Counter[str]] = {}
        for doc_id in sorted(corpus.iter_doc_ids()):
            doc = corpus.get(doc_id)
            if doc is None:
                continue
            field, domain = parse_category(doc.category)
            profile = DocProfile(
                doc_id=doc_id,
                title=(doc.title or "").strip(),
                field=field,
                domain=domain,
                cue_values=cue_percentages(doc.text, cues),
                n_value=sample_size(doc.text),
                top_percent=top_percentage(doc.text),
                words=doc_words(doc.text),
            )
            self.profiles[doc_id] = profile
            if field:
                self.by_field.setdefault(field, []).append(doc_id)
            if domain:
                self.by_domain.setdefault(domain, []).append(doc_id)
            for cue, value in profile.cue_values.items():
                self.by_cue.setdefault(cue, []).append((doc_id, value))
            self.word_df.update(profile.words)

    def field_word_df(self, field: str) -> Counter[str]:
        """Word document-frequency within one field's crowd (lazy, cached)."""
        cached = self._field_word_df.get(field)
        if cached is None:
            cached = Counter()
            for doc_id in self.by_field.get(field, ()):
                cached.update(self.profiles[doc_id].words)
            self._field_word_df[field] = cached
        return cached


# Keyed weakly on the corpus object so a closed/collected corpus frees its
# index; the inner dict is keyed by the cue tuple (one index per vocabulary).
_INDEX_CACHE: "weakref.WeakKeyDictionary[object, dict[tuple[str, ...], ProfileIndex]]" = (
    weakref.WeakKeyDictionary()
)


def profile_index(corpus: "CorpusStore", cues: tuple[str, ...]) -> ProfileIndex:
    try:
        per_corpus = _INDEX_CACHE.setdefault(corpus, {})
    except TypeError:  # corpus type without weakref support: build uncached
        return ProfileIndex(corpus, cues)
    index = per_corpus.get(cues)
    if index is None:
        index = per_corpus[cues] = ProfileIndex(corpus, cues)
    return index
