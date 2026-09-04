"""Cross-document entity index: the substrate multi-hop chain tasks are built on.

A chain task needs one thing the SCI1-SCI4 shapes never had: a *link* between
two documents that is discoverable only by reading one of them. This module
finds those links.

For every document it extracts two kinds of surface string:

``bridge`` terms
    Named things a paper shares with other papers -- an acronym, a proper
    noun phrase, a hyphenated technical compound. A bridge term is the key to
    the second hop: the question describes paper A without naming the term,
    the solver reads A to learn it, and the term plus a second description
    identifies paper B. Bridges are kept only inside a document-frequency
    band: too rare and the term fingerprints its own document (search finds
    it in one shot), too common and it discriminates nothing.

``topic`` terms
    Ordinary content words used to *describe* a paper without naming it.
    These are deliberately common -- the SCI2 lesson, restated: a conjunction
    of rare title words is itself a perfect search key, so descriptors must
    each match a crowd and only pin the document jointly.

Both are pure functions of the corpus text: no LLM, no network, no clock. The
index is content-addressed by :meth:`EntityIndex.digest` so a minted task can
name the exact index it was minted against, the same way a task names its
corpus digest.

The extraction rules here are a determinism contract exactly like a template
tuple: changing a regex silently changes which chains are mintable, so a new
rule set ships under a new index version, never as an edit to a shipped one.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

INDEX_VERSION = "epago-entity-index-v1"

# --- extraction rules (frozen with INDEX_VERSION) ----------------------------

#: Acronyms: 3-6 capitals, optionally with digits. Two-letter acronyms are
#: excluded outright -- measured on the pinned corpus they are dominated by
#: ambiguous pairs (PR, OS, SE, GA, OC) whose several senses make a chain
#: unreadable even when the set algebra still proves it unique.
_ACRONYM = re.compile(r"\b([A-Z][A-Z0-9]{2,5})\b")

#: Proper-noun phrases: two to four capitalised words. Single capitalised
#: words are excluded because sentence-initial ordinary words look identical.
_PROPER = re.compile(r"\b([A-Z][a-z]{2,}(?:[ -][A-Z][a-z]{2,}){1,3})\b")

#: Hyphenated technical compounds: "dose-dependent", "near-infrared".
_TECHNICAL = re.compile(r"\b([a-z]{4,}(?:-[a-z]{3,}){1,2})\b")

#: Lowercase content words, for topic descriptors.
_WORD = re.compile(r"\b([a-z]{4,})\b")

#: Terms that name how a study was done rather than what it was about. They
#: pass every frequency filter and read like real bridges, but two papers that
#: both searched PubMed share no subject matter, so a chain built on one is a
#: coincidence dressed as a link.
_BOILERPLATE = frozenset(
    {
        "GOOGLE SCHOLAR", "WEB OF SCIENCE", "COCHRANE LIBRARY", "PUBMED",
        "SCOPUS", "EMBASE", "MEDLINE", "SCIENCE DIRECT", "SCIENCEDIRECT",
        "PRISMA", "CINAHL", "PROSPERO", "CROSSREF", "OPENALEX", "SPRINGER",
        "ELSEVIER", "WILEY", "PLOS", "MDPI", "FRONTIERS", "RESEARCH GATE",
        "SUPPLEMENTARY", "ETHICS COMMITTEE", "INFORMED CONSENT",
        "CONFIDENCE INTERVAL", "STANDARD DEVIATION", "STANDARD ERROR",
        "ODDS RATIO", "HAZARD RATIO", "RISK RATIO", "CHI SQUARE",
        "STUDENT T", "MANN WHITNEY", "SHAPIRO WILK", "KRUSKAL WALLIS",
        "ANOVA", "ANCOVA", "MANOVA", "SPSS", "STATA", "GRAPHPAD", "MATLAB",
        "PYTHON", "SAS", "EXCEL", "ORIGIN", "IMAGEJ",
        "THE STUDY", "THIS STUDY", "THE RESULTS", "THE PRESENT",
        "THE AUTHORS", "THE FINDINGS", "THE DATA", "THE PATIENTS",
        "MATERIALS AND METHODS", "RESULTS AND DISCUSSION",
        "CONCLUSION", "BACKGROUND", "OBJECTIVE", "ABSTRACT", "INTRODUCTION",
        "METHODS", "RESULTS", "DISCUSSION", "CONCLUSIONS", "PURPOSE",
        "AIM", "AIMS", "DESIGN", "SETTING", "PARTICIPANTS", "OUTCOMES",
        "CI", "SD", "SE", "IQR", "USA", "UK", "USD", "EUR", "GDP",
        "COVID", "COVID-19", "WHO", "NIH", "FDA", "EPA", "ISO", "IEEE",
        "DOI", "URL", "HTTP", "HTTPS", "PDF", "XML", "HTML", "API",
        "AND", "THE", "FOR", "WITH", "FROM", "THAT", "THIS", "WERE", "WAS",
        "NOT", "ALL", "ONE", "TWO", "NEW", "USE", "USED", "MAY", "CAN",
    }
)

#: Descriptor words that say nothing about subject matter.
_TOPIC_STOP = frozenset(
    {
        "study", "studies", "results", "result", "method", "methods", "data",
        "analysis", "analyses", "research", "paper", "article", "using",
        "used", "based", "between", "among", "these", "those", "which",
        "their", "there", "were", "with", "from", "that", "this", "have",
        "been", "also", "into", "more", "most", "such", "than", "then",
        "when", "what", "will", "would", "could", "should", "both", "each",
        "other", "significant", "significantly", "showed", "shown", "found",
        "however", "therefore", "thus", "while", "during", "after", "before",
        "high", "higher", "highest", "low", "lower", "lowest", "increase",
        "increased", "decrease", "decreased", "compared", "comparison",
        "group", "groups", "control", "total", "mean", "average", "value",
        "values", "level", "levels", "rate", "rates", "effect", "effects",
        "conclusion", "conclusions", "background", "objective", "objectives",
        "abstract", "introduction", "discussion", "findings", "purpose",
        "present", "sample", "samples", "test", "tests", "model", "models",
        "aim", "aims", "aimed", "design", "setting", "participants",
        "outcome", "outcomes", "measure", "measures", "measured", "reported",
        "report", "including", "included", "include", "different", "various",
        "provide", "provides", "provided", "show", "shows", "suggest",
        "suggests", "indicate", "indicates", "associated", "association",
        "potential", "important", "further", "well", "over", "under",
        "within", "across", "through", "against", "about", "only", "some",
        "many", "much", "very", "same", "first", "second", "third",
        "respectively", "additionally", "furthermore", "moreover",
        "performed", "conducted", "obtained", "observed", "determined",
        "evaluated", "assessed", "investigated", "examined", "analyzed",
        "analysed", "developed", "proposed", "applied", "considered",
    }
)


def _norm(term: str) -> str:
    """Canonical form: whitespace collapsed, non-breaking hyphens normalised."""
    return " ".join(term.replace("‐", "-").replace("‑", "-").split())


def extract_bridges(text: str) -> set[str]:
    """Candidate link terms in one document, before any corpus-wide filtering.

    Case is preserved for readability but the boilerplate check is
    case-insensitive, so ``Anova`` and ``ANOVA`` are rejected alike.
    """
    out: set[str] = set()
    for pattern in (_ACRONYM, _PROPER, _TECHNICAL):
        for raw in pattern.findall(text):
            term = _norm(raw)
            if term.upper() in _BOILERPLATE:
                continue
            out.add(term)
    return out


#: A bridge has to be a *name*: something a paper calls by a proper title or an
#: established abbreviation. Hyphenated lowercase compounds pass every
#: frequency filter and look like technical vocabulary, but measured on the
#: pinned corpus they are overwhelmingly descriptors -- "right-sided",
#: "short-chain", "time-lapse", "quasi-spherical". Two papers sharing a
#: descriptor share an adjective, not a referent, and a question asking a
#: solver to find "the same anatomical location" has no unique answer to find.
_NAME_LIKE = re.compile(r"^(?:[A-Z][A-Z0-9]{2,5}|[A-Z][a-z]{2,}(?:[ -][A-Za-z][a-z]{2,}){1,3})$")


#: Words that make a capitalised phrase look like a name without being one.
#: "Factors Associated" and "Conclusion Overall" are sentence fragments that
#: happen to start a sentence, and they pass every shape test a regex can
#: apply. A phrase built only from these is prose, not an entity.
_PROSE_WORDS = frozenset(
    {
        "factors", "associated", "conclusion", "conclusions", "overall",
        "results", "result", "background", "objective", "objectives",
        "methods", "method", "study", "studies", "analysis", "patients",
        "findings", "discussion", "introduction", "purpose", "aim", "aims",
        "significant", "significance", "important", "present", "current",
        "based", "using", "used", "related", "level", "levels", "high",
        "low", "total", "mean", "group", "groups", "control", "effect",
        "effects", "data", "value", "values", "rate", "rates", "risk",
        "clinical", "medical", "health", "care", "treatment", "disease",
        "review", "research", "paper", "article", "report", "case",
        "new", "novel", "recent", "different", "various", "multiple",
        "first", "second", "third", "main", "major", "minor", "general",
        "both", "all", "each", "these", "those", "this", "that", "there",
        "here", "from", "with", "within", "among", "between", "during",
        "after", "before", "under", "over", "through", "across", "into",
    }
)

#: Words that begin a section or a sentence rather than a name. A capitalised
#: phrase that starts with one of these is prose the extractor happened to
#: catch at a boundary: "Abstract Hydrogen", "Conclusion Both", "From October".
#: The prose-word test alone does not catch these, because their *second* word
#: is often a perfectly good noun.
_SENTENCE_STARTERS = frozenset(
    {
        "abstract", "background", "introduction", "methods", "method",
        "results", "result", "conclusion", "conclusions", "discussion",
        "objective", "objectives", "purpose", "aim", "aims", "summary",
        "findings", "significance", "highlights", "keywords", "importance",
        "from", "in", "on", "at", "by", "for", "with", "the", "this",
        "these", "those", "there", "here", "during", "after", "before",
        "between", "among", "within", "across", "over", "under", "both",
        "all", "we", "our", "it", "they", "their", "however", "moreover",
        "furthermore", "therefore", "thus", "overall", "recently", "although",
        "while", "when", "where", "since", "given", "using", "based",
    }
)


def _looks_like_prose_fragment(words: list[str]) -> bool:
    """True when a capitalised phrase is a sentence opening, not a name."""
    return words[0].lower() in _SENTENCE_STARTERS


def is_name_like(term: str) -> bool:
    """True when a term reads as a named entity rather than a description.

    Shape is necessary but not sufficient. An acronym is taken on its shape
    alone; a capitalised phrase must additionally carry at least one word that
    is not ordinary academic prose, or it is a sentence opening rather than a
    name.
    """
    if not _NAME_LIKE.match(term):
        return False
    words = term.replace("-", " ").split()
    if len(words) == 1:
        return True  # an acronym; shape already settled it
    if _looks_like_prose_fragment(words):
        return False
    return any(w.lower() not in _PROSE_WORDS for w in words)


def extract_topics(text: str) -> set[str]:
    """Ordinary subject words usable as a non-naming description."""
    return {w for w in _WORD.findall(text.lower()) if w not in _TOPIC_STOP}


@dataclass(frozen=True, slots=True)
class IndexBands:
    """Document-frequency windows that decide what a term may be used for.

    ``bridge_min`` is 2 because a bridge that appears in one document links
    nothing. ``bridge_max`` bounds how much reading the second hop can cost:
    once a solver knows the bridge term, the papers carrying it are the
    candidate set it must narrow, so an unbounded band turns hop two into an
    enumeration task -- the exact failure that retired ``comparative_finding``.

    ``topic_min`` is the SCI4 crowd rule (a descriptor matching fewer
    documents is a fingerprint, not a filter) and ``topic_max`` keeps
    descriptors from being pure noise.
    """

    bridge_min: int = 2
    bridge_max: int = 60
    topic_min: int = 5
    topic_max: int = 2000


DEFAULT_BANDS = IndexBands()


@dataclass(frozen=True, slots=True)
class EntityIndex:
    """Postings from term to document, for both bridge and topic vocabularies.

    Both directions are materialised: minting walks terms to find a shared
    bridge, and the uniqueness proof walks documents to intersect constraints.
    """

    bridge_docs: dict[str, frozenset[str]]
    doc_bridges: dict[str, frozenset[str]]
    topic_docs: dict[str, frozenset[str]]
    doc_topics: dict[str, frozenset[str]]
    bands: IndexBands
    n_docs: int

    # -- construction ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        docs: Iterable[tuple[str, str]],
        *,
        bands: IndexBands = DEFAULT_BANDS,
    ) -> "EntityIndex":
        """Build from ``(doc_id, text)`` pairs.

        Two passes: count document frequency, then keep only in-band terms.
        Postings are built in sorted doc-id order so the index is a pure
        function of its input and its digest is stable across runs.
        """
        raw_bridge: dict[str, set[str]] = defaultdict(set)
        raw_topic: dict[str, set[str]] = defaultdict(set)
        n = 0
        for doc_id, text in docs:
            n += 1
            for term in extract_bridges(text):
                raw_bridge[term].add(doc_id)
            for term in extract_topics(text):
                raw_topic[term].add(doc_id)

        bridge_docs = {
            t: frozenset(d)
            for t, d in raw_bridge.items()
            if bands.bridge_min <= len(d) <= bands.bridge_max
        }
        topic_docs = {
            t: frozenset(d)
            for t, d in raw_topic.items()
            if bands.topic_min <= len(d) <= bands.topic_max
        }
        doc_bridges: dict[str, set[str]] = defaultdict(set)
        for t, ds in bridge_docs.items():
            for d in ds:
                doc_bridges[d].add(t)
        doc_topics: dict[str, set[str]] = defaultdict(set)
        for t, ds in topic_docs.items():
            for d in ds:
                doc_topics[d].add(t)
        return cls(
            bridge_docs=bridge_docs,
            doc_bridges={d: frozenset(v) for d, v in doc_bridges.items()},
            topic_docs=topic_docs,
            doc_topics={d: frozenset(v) for d, v in doc_topics.items()},
            bands=bands,
            n_docs=n,
        )

    # -- queries --------------------------------------------------------------

    def bridges_of(self, doc_id: str) -> frozenset[str]:
        return self.doc_bridges.get(doc_id, frozenset())

    def topics_of(self, doc_id: str) -> frozenset[str]:
        return self.doc_topics.get(doc_id, frozenset())

    def docs_with_bridge(self, term: str) -> frozenset[str]:
        return self.bridge_docs.get(term, frozenset())

    def docs_with_topic(self, term: str) -> frozenset[str]:
        return self.topic_docs.get(term, frozenset())

    def docs_with_all_topics(self, terms: Iterable[str]) -> frozenset[str]:
        """Intersection over topic postings -- the uniqueness primitive.

        Missing terms yield the empty set rather than being skipped: a
        constraint the index cannot evaluate must never silently widen the
        candidate set a uniqueness proof is about to be read off.
        """
        it = iter(terms)
        try:
            first = next(it)
        except StopIteration:
            return frozenset()
        acc = self.docs_with_topic(first)
        for t in it:
            if not acc:
                break
            acc &= self.docs_with_topic(t)
        return acc

    # -- persistence ----------------------------------------------------------

    def canonical_bytes(self) -> bytes:
        """Stable serialisation; the digest and the file are the same bytes."""
        payload = {
            "format": INDEX_VERSION,
            "n_docs": self.n_docs,
            "bands": {
                "bridge_min": self.bands.bridge_min,
                "bridge_max": self.bands.bridge_max,
                "topic_min": self.bands.topic_min,
                "topic_max": self.bands.topic_max,
            },
            "bridge_docs": {
                t: sorted(d) for t, d in sorted(self.bridge_docs.items())
            },
            "topic_docs": {
                t: sorted(d) for t, d in sorted(self.topic_docs.items())
            },
        }
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def write(self, path: Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = self.canonical_bytes()
        path.write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def read(cls, path: Path) -> "EntityIndex":
        payload = json.loads(Path(path).read_bytes())
        if payload.get("format") != INDEX_VERSION:
            raise ValueError(
                f"entity index format {payload.get('format')!r} != {INDEX_VERSION!r}"
            )
        bands = IndexBands(**payload["bands"])
        bridge_docs = {t: frozenset(d) for t, d in payload["bridge_docs"].items()}
        topic_docs = {t: frozenset(d) for t, d in payload["topic_docs"].items()}
        doc_bridges: dict[str, set[str]] = defaultdict(set)
        for t, ds in bridge_docs.items():
            for d in ds:
                doc_bridges[d].add(t)
        doc_topics: dict[str, set[str]] = defaultdict(set)
        for t, ds in topic_docs.items():
            for d in ds:
                doc_topics[d].add(t)
        return cls(
            bridge_docs=bridge_docs,
            doc_bridges={d: frozenset(v) for d, v in doc_bridges.items()},
            topic_docs=topic_docs,
            doc_topics={d: frozenset(v) for d, v in doc_topics.items()},
            bands=bands,
            n_docs=payload["n_docs"],
        )

    def stats(self) -> dict:
        bl = Counter(len(d) for d in self.bridge_docs.values())
        return {
            "n_docs": self.n_docs,
            "n_bridge_terms": len(self.bridge_docs),
            "n_topic_terms": len(self.topic_docs),
            "docs_with_bridge": len(self.doc_bridges),
            "docs_with_topic": len(self.doc_topics),
            "bridge_df_2": bl.get(2, 0),
            "bridge_df_3_10": sum(v for k, v in bl.items() if 3 <= k <= 10),
            "bridge_df_11_60": sum(v for k, v in bl.items() if 11 <= k <= 60),
        }
