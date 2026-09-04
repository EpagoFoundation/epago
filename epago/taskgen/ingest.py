"""Autonomous private-task supply.

Validators keep their private pools fresh without any human curation: an
:class:`IngestSource` fetches raw documents, :class:`OverlayCorpus` layers
them over the frozen duel corpus, and :func:`build_private_tasks` mints tasks
*from the ingested documents* through the exact same templates and QA gate as
public generation — only the origin differs (``GENERATED_PRIVATE``).

Determinism scope: ingestion itself is supply, not generation — what a source
fetches may vary between runs. But once a document set is fixed, minting is
the same pure function of ``(seed, release, overlay corpus)`` as everywhere
else, and doc ids are content-derived, so the resulting pool is fully
reproducible from the published rotation file.

The environment package's concrete classes are imported lazily inside methods
(protocol-only coupling at import time); the fully-offline default is
:class:`LocalDirSource`, and :class:`HttpSource` is a skeleton that
import-guards ``httpx``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence

from epago.core.types import Task, TaskOrigin
from epago.taskgen.generator import generate_tasks

if TYPE_CHECKING:  # protocol-only coupling to the environment subsystem
    from epago.environment.corpus import CorpusStore, Document, SearchHit

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 300
_TITLE_MAX_CHARS = 120
_INGEST_CATEGORY = "ingested"
# Documents fetched per requested task; generous because rule-based templates
# only mint from documents with extractable facts.
_DOCS_PER_TASK = 4
_MIN_DOC_BUDGET = 16


class IngestSource(Protocol):
    """Anything that can supply raw documents for private task minting."""

    def fetch_documents(self, budget: int) -> "list[Document]": ...


def _content_doc_id(text: str) -> str:
    """Content-derived id: the same document ingested twice dedups itself."""
    return "ing-" + hashlib.sha256(text.encode()).hexdigest()[:16]


class LocalDirSource:
    """Fully-offline default: reads ``*.txt`` / ``*.md`` files from a directory.

    Files are read in sorted-name order and doc ids derive from content, so a
    fixed drop directory always yields the same document set.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def fetch_documents(self, budget: int) -> "list[Document]":
        from epago.environment.corpus import Document  # deferred: runtime class

        docs: list[Document] = []
        if not self.root.is_dir():
            return docs
        paths = sorted(p for p in self.root.iterdir() if p.suffix in (".txt", ".md"))
        for path in paths[:budget]:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            title = text.splitlines()[0].strip().lstrip("# ")[:_TITLE_MAX_CHARS] or path.stem
            docs.append(
                Document(
                    doc_id=_content_doc_id(text),
                    url=path.resolve().as_uri(),
                    title=title,
                    text=text,
                    category=_INGEST_CATEGORY,
                )
            )
        return docs


class HttpSource:
    """Skeleton HTTP source fetching plain-text documents from fixed URLs.

    ``httpx`` is imported at fetch time and failures degrade to skipped URLs,
    so merely constructing (or importing) this class never requires network
    or optional dependencies. R1 treats each URL body as one plain-text
    document; content negotiation and HTML extraction are later-release work.
    """

    def __init__(self, urls: Sequence[str], timeout_s: float = 30.0) -> None:
        self.urls = tuple(urls)
        self.timeout_s = timeout_s

    def fetch_documents(self, budget: int) -> "list[Document]":
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - guard, not a code path
            raise RuntimeError("HttpSource requires the 'httpx' package") from exc
        from epago.environment.corpus import Document  # deferred: runtime class

        docs: list[Document] = []
        for url in self.urls[:budget]:
            try:
                resp = httpx.get(url, timeout=self.timeout_s, follow_redirects=True)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("ingest: skipping %s (%s)", url, exc)
                continue
            text = resp.text.strip()
            if not text:
                continue
            title = text.splitlines()[0].strip()[:_TITLE_MAX_CHARS] or url
            docs.append(
                Document(
                    doc_id=_content_doc_id(text),
                    url=url,
                    title=title,
                    text=text,
                    category=_INGEST_CATEGORY,
                )
            )
        return docs


def select_shards(shards: Sequence[str], seed: int, k: int) -> list[str]:
    """Deterministically pick ``k`` shards from a manifest under ``seed`` (pure).

    Sorted before the draw so the choice is a function of (manifest, seed) only,
    never of listing order. ``seed`` comes from the drand beacon, so which shards
    are materialised is unpredictable before the challenger commits yet identical
    across every validator that fetches the same beacon round — the non-gameable,
    deterministic slice selection at the heart of the automated fresh pool.
    """
    import numpy as np

    ordered = sorted(shards)
    if not ordered or k <= 0:
        return []
    k = min(k, len(ordered))
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.choice(len(ordered), size=k, replace=False)
    return [ordered[int(i)] for i in sorted(idx)]


class HfSnapshotSource:
    """Fully-automated fresh source: a bounded, revision-pinned, beacon-selected
    slice of a large public dated dataset (e.g. FineWeb-Edu).

    This replaces :class:`LocalDirSource` as the default so the private pool needs
    **no human** — the R2 fix. Determinism: ``(repo, revision, seed)`` fixes
    exactly which shards are materialised, so every validator pinning the same
    dataset revision and fetching the same drand round gets byte-identical docs,
    and the θ-quorum on the resulting verdict holds. Freshness: point ``revision``
    at the newest dated snapshot each rotation. Speed: only the rotation loop
    calls this (off the duel critical path); ``max_shards`` bounds the download so
    supply comfortably exceeds one epoch's demand without materialising the whole
    (billions-of-docs) dataset.

    The HuggingFace listing/download/parquet-read are injectable so the selection
    and packing logic is unit-tested without network or optional deps; the
    defaults use ``huggingface_hub`` + ``pyarrow`` (lazy-imported).

    Note: this is an *external public data source* — a read from the world's
    dated data lake — not Epago-owned storage. It stays on the public dataset
    host on purpose; Epago's own artifacts (models, corpus, validator state)
    live in the object store, but the fresh feed's whole point is
    pulling data that already lives publicly and postdates the model. Swapping
    the source host is a one-method change (``list``/``download``/``read`` are
    injectable) if a different public lake is preferred.
    """

    def __init__(
        self,
        repo: str,
        revision: str,
        seed: int,
        *,
        text_column: str = "text",
        max_shards: int = 4,
        list_shards_fn=None,
        download_shard_fn=None,
        read_texts_fn=None,
    ) -> None:
        self.repo = repo
        self.revision = revision
        self.seed = seed
        self.text_column = text_column
        self.max_shards = max_shards
        self._list = list_shards_fn or _hf_list_parquet
        self._download = download_shard_fn or _hf_download
        self._read = read_texts_fn or _read_parquet_texts

    def fetch_documents(self, budget: int) -> "list[Document]":
        from epago.environment.corpus import Document  # deferred: runtime class

        shards = self._list(self.repo, self.revision)
        picked = select_shards(shards, self.seed, self.max_shards)
        docs: list[Document] = []
        for shard in picked:
            path = self._download(self.repo, self.revision, shard)
            for text in self._read(path, self.text_column):
                text = (text or "").strip()
                if not text:
                    continue
                title = text.splitlines()[0].strip()[:_TITLE_MAX_CHARS] or shard
                docs.append(
                    Document(
                        doc_id=_content_doc_id(text),
                        url=f"hf://{self.repo}@{self.revision}/{shard}",
                        title=title,
                        text=text,
                        category=_INGEST_CATEGORY,
                    )
                )
                if len(docs) >= budget:
                    return docs
        return docs


def _hf_list_parquet(repo: str, revision: str) -> list[str]:
    from huggingface_hub import HfApi  # deferred: optional dep

    files = HfApi().list_repo_files(repo, revision=revision, repo_type="dataset")
    return [f for f in files if f.endswith(".parquet")]


def _hf_download(repo: str, revision: str, shard: str) -> Path:
    from huggingface_hub import hf_hub_download  # deferred: optional dep

    return Path(
        hf_hub_download(repo, filename=shard, revision=revision, repo_type="dataset")
    )


def _read_parquet_texts(path: Path, column: str):
    import pyarrow.parquet as pq  # deferred: optional dep

    table = pq.read_table(path, columns=[column])
    for chunk in table.column(column):
        yield chunk.as_py()


class OverlayCorpus:
    """Validator-local overlay of ingested documents over a base corpus.

    Deliberate protocol deviation, scoped to private minting:
    ``iter_doc_ids`` yields *only* the ingested overlay docs, so templates
    sample their origin documents from fresh material; ``search`` and ``get``
    span base + overlay so cross-document joins and QA can still reach the
    frozen corpus. Overlay search scoring is a simple term-frequency heuristic
    merged with base hits — private tasks are validator-local, so this ranking
    never needs to agree across validators, only with itself.
    """

    def __init__(self, base: "CorpusStore", docs: "Sequence[Document]") -> None:
        self._base = base
        self._docs = {d.doc_id: d for d in docs}

    def search(
        self, query: str, k: int = 10, mask_doc_ids: frozenset[str] = frozenset()
    ) -> "list[SearchHit]":
        from epago.environment.corpus import SearchHit  # deferred: runtime class

        hits = list(self._base.search(query, k, mask_doc_ids))
        terms = [t.strip('"').lower() for t in query.split() if t.strip('"')]
        for doc_id in sorted(self._docs):
            if doc_id in mask_doc_ids:
                continue
            doc = self._docs[doc_id]
            haystack = f"{doc.title}\n{doc.text}".lower()
            score = float(sum(haystack.count(term) for term in terms))
            if score > 0:
                hits.append(
                    SearchHit(
                        doc_id=doc_id,
                        title=doc.title,
                        snippet=doc.text[:_SNIPPET_CHARS],
                        score=score,
                    )
                )
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:k]

    def get(self, doc_id: str, mask_doc_ids: frozenset[str] = frozenset()) -> "Document | None":
        if doc_id in mask_doc_ids:
            return None
        if doc_id in self._docs:
            return self._docs[doc_id]
        return self._base.get(doc_id, mask_doc_ids)

    def doc_count(self) -> int:
        return self._base.doc_count() + len(self._docs)

    def iter_doc_ids(self):
        """Overlay docs only — see class docstring for why."""
        yield from sorted(self._docs)


def build_private_tasks(
    sources: Sequence[IngestSource],
    corpus: "CorpusStore",
    seed: int,
    n: int,
    release: str,
) -> list[Task]:
    """Fetch documents from ``sources`` and mint ``n`` private tasks from them.

    Same templates, same QA, same determinism-per-document-set as public
    generation; only ``origin`` differs. Raises
    :class:`epago.taskgen.generator.GenerationExhausted` when the sources
    cannot supply enough mintable material — a starved private pool must fail
    loudly, not shrink silently.
    """
    budget = max(_DOCS_PER_TASK * n, _MIN_DOC_BUDGET)
    by_id: dict[str, "Document"] = {}
    for source in sources:
        for doc in source.fetch_documents(budget):
            by_id[doc.doc_id] = doc
    docs = [by_id[doc_id] for doc_id in sorted(by_id)]
    logger.info("ingest: %d documents from %d sources", len(docs), len(sources))
    overlay = OverlayCorpus(corpus, docs)
    tasks = generate_tasks(seed, release, overlay, n)
    return [replace(t, origin=TaskOrigin.GENERATED_PRIVATE) for t in tasks]
