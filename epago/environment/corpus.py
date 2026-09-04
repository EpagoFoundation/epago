"""Corpus store: the frozen document universe rollouts research against.

Every duel runs against one pinned corpus snapshot (``eval.corpus_digest`` in
``chain.toml``), so search results are identical across validators. The store
interface is a small :class:`typing.Protocol`; :class:`SqliteCorpus` is the
reference implementation over SQLite FTS5 (stdlib only). Production vector
backends (e.g. Milvus + BGE embeddings) implement the same protocol and are
import-guarded in their own modules — nothing here imports heavy dependencies.

Source masking is an eval-integrity requirement, not a convenience: a task's
``masked_doc_ids`` must never surface in search results nor be browsable during
that task's rollouts, otherwise the model can look the answer up at the exact
documents the task was minted from instead of re-deriving it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from epago import constants

_SNIPPET_TOKENS = constants.SEARCH_SNIPPET_TOKENS
_SNIPPET_FALLBACK_CHARS = constants.SEARCH_SNIPPET_FALLBACK_CHARS
_MAX_QUERY_TERMS = constants.SEARCH_MAX_QUERY_TERMS
_COMMON_TERM_DF = constants.SEARCH_COMMON_TERM_DF
#: Cap on the per-corpus document-frequency cache. Not a tuning knob: it only
#: bounds memory against model-generated junk terms, and dropping the cache
#: costs one COUNT per term, never a different result.
_DOC_FREQ_CACHE_MAX = 100_000


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    url: str
    title: str
    text: str
    category: str = ""


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked result. ``score`` is negated bm25, so higher is better."""

    doc_id: str
    title: str
    snippet: str
    score: float


class CorpusStore(Protocol):
    def search(
        self, query: str, k: int = 10, mask_doc_ids: frozenset[str] = frozenset()
    ) -> list[SearchHit]: ...

    def get(self, doc_id: str, mask_doc_ids: frozenset[str] = frozenset()) -> Document | None: ...

    def doc_count(self) -> int: ...

    def iter_doc_ids(self) -> Iterator[str]: ...


def _sanitize_terms(query: str) -> list[str]:
    """Reduce arbitrary model output to safe FTS5 phrase terms.

    FTS5 has its own operator syntax (AND/OR/NOT, NEAR, ``-``, ``*``, column
    filters); a raw model-generated query can be a syntax error or worse. Each
    whitespace token is stripped of quote characters and re-wrapped in double
    quotes, which makes it a literal phrase term — operators lose all meaning.
    A token with no indexable characters becomes an empty phrase, which FTS5
    accepts and matches nothing.
    """
    terms = []
    for token in query.split():
        token = token.replace('"', "")
        if token:
            terms.append(f'"{token}"')
        if len(terms) >= _MAX_QUERY_TERMS:
            break
    return terms


class SqliteCorpus:
    """Reference :class:`CorpusStore` over a single-file SQLite FTS5 index.

    The FTS table is external-content over ``docs`` so text is stored once and
    ``snippet()`` still works. Ranking is bm25 with ``doc_id`` as a
    deterministic tie-break, so result order is stable for a given snapshot.

    Query terms are OR-ed, not AND-ed: a research tool must return its best
    partial matches for a natural-language question. Under FTS5's implicit AND
    every term had to appear in a document, so a question-shaped query returned
    nothing and only a lucky 3-4 keyword guess retrieved anything at all.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._doc_count: int | None = None
        self._doc_freq: dict[str, int] = {}
        # check_same_thread=False: the validator runs each tick via asyncio.to_thread,
        # so the corpus (opened on the main thread) is read from worker threads. Access
        # is serialized (one tick at a time), so cross-thread use is safe here.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def create(cls, path: str | Path) -> "SqliteCorpus":
        """Create (or open) a corpus database, building the schema if absent."""
        store = cls(path)
        store._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                doc_id   TEXT PRIMARY KEY,
                url      TEXT NOT NULL,
                title    TEXT NOT NULL,
                text     TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT ''
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                title, text, content='docs', content_rowid='rowid'
            );
            """
        )
        store._conn.commit()
        return store

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteCorpus":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    def add_documents(self, docs: Iterable[Document]) -> None:
        # Corpus statistics are cached for query planning; adding documents
        # invalidates them. A pinned snapshot never grows under a validator,
        # so this only matters while a corpus is being built.
        self._doc_count = None
        self._doc_freq.clear()
        with self._conn:  # one transaction: all-or-nothing
            cur = self._conn.cursor()
            for doc in docs:
                cur.execute(
                    "INSERT INTO docs (doc_id, url, title, text, category) VALUES (?, ?, ?, ?, ?)",
                    (doc.doc_id, doc.url, doc.title, doc.text, doc.category),
                )
                cur.execute(
                    "INSERT INTO docs_fts (rowid, title, text) VALUES (?, ?, ?)",
                    (cur.lastrowid, doc.title, doc.text),
                )

    def _prune_common_terms(self, terms: list[str]) -> list[str]:
        """Drop terms that occur in more than ``_COMMON_TERM_DF`` of the corpus.

        Document frequency is a property of the pinned snapshot, so every
        validator prunes the same terms and ranking stays reproducible. Terms
        are cached because a snapshot is immutable while it is being queried;
        the cache is dropped wholesale when it grows past
        :data:`_DOC_FREQ_CACHE_MAX`, since queries come from a model that can
        emit unbounded novel tokens and this must not grow without limit.
        Pruning everything means the query was all common words; keep them
        rather than return nothing.
        """
        limit = max(int(self.doc_count() * _COMMON_TERM_DF), 1)
        if len(self._doc_freq) > _DOC_FREQ_CACHE_MAX:
            self._doc_freq.clear()
        kept = []
        for term in terms:
            df = self._doc_freq.get(term)
            if df is None:
                df = self._conn.execute(
                    "SELECT COUNT(*) FROM docs_fts WHERE docs_fts MATCH ?", (term,)
                ).fetchone()[0]
                self._doc_freq[term] = df
            if df <= limit:
                kept.append(term)
        return kept or terms

    def search(
        self, query: str, k: int = 10, mask_doc_ids: frozenset[str] = frozenset()
    ) -> list[SearchHit]:
        terms = _sanitize_terms(query)
        if not terms or k <= 0:
            return []
        match_expr = " OR ".join(self._prune_common_terms(terms))
        # Over-fetch by the mask size so post-filtering can never starve the
        # result list below k while masked docs still hold ranked slots.
        rows = self._conn.execute(
            """
            SELECT d.doc_id, d.title,
                   snippet(docs_fts, -1, '', '', '…', ?) AS snip,
                   bm25(docs_fts) AS rank
            FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid
            WHERE docs_fts MATCH ?
            ORDER BY rank, d.doc_id
            LIMIT ?
            """,
            (_SNIPPET_TOKENS, match_expr, k + len(mask_doc_ids)),
        ).fetchall()
        hits: list[SearchHit] = []
        for doc_id, title, snip, rank in rows:
            if doc_id in mask_doc_ids:
                continue
            snippet = snip if snip else title[:_SNIPPET_FALLBACK_CHARS]
            hits.append(SearchHit(doc_id=doc_id, title=title, snippet=snippet, score=-rank))
            if len(hits) == k:
                break
        return hits

    def get(self, doc_id: str, mask_doc_ids: frozenset[str] = frozenset()) -> Document | None:
        if doc_id in mask_doc_ids:
            return None
        row = self._conn.execute(
            "SELECT doc_id, url, title, text, category FROM docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return Document(doc_id=row[0], url=row[1], title=row[2], text=row[3], category=row[4])

    def doc_count(self) -> int:
        if self._doc_count is None:
            self._doc_count = self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        return self._doc_count

    def iter_doc_ids(self) -> Iterator[str]:
        for (doc_id,) in self._conn.execute("SELECT doc_id FROM docs ORDER BY doc_id"):
            yield doc_id
