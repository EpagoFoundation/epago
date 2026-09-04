"""Tool surface the rollout harness exposes to models.

A rollout interacts with the corpus only through a :class:`ToolSession` bound
to one task: the task's ``masked_doc_ids`` are stripped from both search and
browse for the whole session, so the answer must be re-derived rather than
looked up at its origin documents.

Output formatting is strictly deterministic — stable ordering, no timestamps,
no randomness — because duel rollouts must be reproducible bit-for-bit across
validators for verdicts to agree.
"""

from __future__ import annotations

import re

from epago import constants
from epago.core.types import Task
from epago.environment.corpus import CorpusStore

_DEFAULT_SEARCH_K = constants.SEARCH_K
_DEFAULT_PAGE_CHARS = constants.BROWSE_PAGE_CHARS


def _one_line(text: str) -> str:
    """Collapse all whitespace runs so a snippet renders as a single line."""
    return " ".join(text.split())


class ToolSession:
    """One task's tool bindings: search + browse with the task's mask applied."""

    def __init__(
        self,
        corpus: CorpusStore,
        mask_doc_ids: frozenset[str],
        *,
        search_k: int = _DEFAULT_SEARCH_K,
        page_chars: int = _DEFAULT_PAGE_CHARS,
    ) -> None:
        self._corpus = corpus
        self._mask = mask_doc_ids
        self._search_k = search_k
        self._page_chars = page_chars
        self.search_calls: int = 0
        self.browse_calls: int = 0
        self.repeated_searches: int = 0

    def search(self, query: str) -> str:
        self.search_calls += 1
        hits = self._corpus.search(query, k=self._search_k, mask_doc_ids=self._mask)
        if not hits:
            return "No results found."
        lines = [
            f"{i}. [{hit.doc_id}] {_one_line(hit.title)} — {_one_line(hit.snippet)}"
            for i, hit in enumerate(hits, start=1)
        ]
        return "\n".join(lines)

    def browse(self, doc_id: str) -> str:
        # Tool errors are returned as strings, never raised: a bad doc_id is a
        # model mistake the rollout should observe and recover from.
        self.browse_calls += 1
        doc = self._corpus.get(doc_id, mask_doc_ids=self._mask)
        if doc is None:
            return f"Document not found: {doc_id}"
        text = doc.text
        if len(text) > self._page_chars:
            text = text[: self._page_chars] + f"\n[truncated at {self._page_chars} characters]"
        return f"# {_one_line(doc.title)}\nurl: {doc.url}\ndoc_id: {doc.doc_id}\n\n{text}"


class ResearchEnvironment:
    """Wraps a corpus store and mints per-task tool sessions for the harness."""

    def __init__(
        self,
        corpus: CorpusStore,
        *,
        search_k: int = _DEFAULT_SEARCH_K,
        page_chars: int = _DEFAULT_PAGE_CHARS,
    ) -> None:
        self._corpus = corpus
        self._search_k = search_k
        self._page_chars = page_chars

    @property
    def corpus(self) -> CorpusStore:
        return self._corpus

    def tools_for_task(self, task: Task) -> "WebToolSession":
        # The v4 harness speaks the native web surface; WebToolSession extends
        # ToolSession, so the bespoke methods remain for replay and tests.
        return WebToolSession(
            self._corpus,
            frozenset(task.masked_doc_ids),
            search_k=self._search_k,
            page_chars=self._page_chars,
        )


# --- native web-style surface (tools v3-web) ---------------------------------
#
# The reference model was agentic-RL-trained on real web tooling: a search
# call returning a ranked results page and a visit call opening pages by URL.
# These methods render the SAME corpus through that shape. The bespoke
# ``<search>/<browse>`` surface above measurably put the model out of
# distribution (48% of SCI4 episodes died malformed; tools scored below
# closed-book); it is kept only for replaying old transcripts and for the
# scripted test backend.

_MAX_QUERIES = constants.SEARCH_MAX_QUERIES
_MAX_VISITS = constants.VISIT_MAX_DOCS


class WebToolSession(ToolSession):
    """The v3-web surface: SERP-shaped search, visit by url or id.

    Determinism unchanged: same corpus, same ranking, stable formatting. The
    session remembers every (url -> doc_id) pair it has shown so a model that
    hands back the URL — which is what a web-trained model does — resolves to
    the same document a validator's replay resolves to.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._url_to_id: dict[str, str] = {}
        self._seen_queries: dict[str, int] = {}

    @staticmethod
    def _query_key(query: str) -> str:
        """Two queries that must return identical results share a key.

        Search is deterministic and the term bag is order-free, so
        ``"WPS DMA"`` and ``"DMA WPS"`` are the same query and quoting does not
        change the result set.
        """
        return " ".join(sorted(re.findall(r"[a-z0-9]+", query.lower())))

    def search_page(self, queries: list[str]) -> str:
        """One results page per query, capped at ``SEARCH_MAX_QUERIES``.

        A query already run in this session is answered with a pointer rather
        than a second copy of the same page. Search is deterministic, so the
        repeat could only ever return what the model has already been shown,
        and re-rendering it buys nothing while costing a full page of context.

        That cost is the binding constraint. Measured over 400 episodes on the
        two-hop tasks, 25% of all searches were exact repeats and a third of
        episodes spent over 30% of their searches re-running queries they had
        already run; tool output was 85% of the context in episodes that
        overflowed. One transcript searched ``"whey protein" "DMA" "film"``
        three separate times while the answer sat at rank 1 for the exact
        abbreviation it had already read. Saying so plainly is also a nudge out
        of the loop, which a silently identical page is not.
        """
        blocks: list[str] = []
        for query in queries[:_MAX_QUERIES]:
            self.search_calls += 1
            key = self._query_key(query)
            if key in self._seen_queries:
                self.repeated_searches += 1
                blocks.append(
                    f'You already ran this search: "{_one_line(query)}". It '
                    "returns the same results as before. Try different terms — "
                    "exact names, codes or abbreviations as they are written in "
                    "the documents work better than paraphrases."
                )
                continue
            self._seen_queries[key] = self.search_calls
            hits = self._corpus.search(query, k=self._search_k, mask_doc_ids=self._mask)
            if not hits:
                blocks.append(
                    f'No results found for "{_one_line(query)}". '
                    "Try different, more specific keywords."
                )
                continue
            lines = [f'Search results for "{_one_line(query)}":']
            for i, hit in enumerate(hits, start=1):
                doc = self._corpus.get(hit.doc_id, mask_doc_ids=self._mask)
                url = (doc.url if doc and doc.url else hit.doc_id)
                self._url_to_id[url] = hit.doc_id
                lines.append(
                    f"{i}. {_one_line(hit.title)}\n"
                    f"   url: {url} (id: {hit.doc_id})\n"
                    f"   {_one_line(hit.snippet)}"
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else "No query given."

    def _resolve(self, target: str) -> str | None:
        """A visit target may be a doc id, a URL, or a pasted 'url (id: x)'."""
        target = target.strip()
        m = re.search(r"\(id:\s*([^)\s]+)\)", target)
        if m:
            target = m.group(1)
        if self._corpus.get(target, mask_doc_ids=self._mask) is not None:
            return target
        return self._url_to_id.get(target)

    def visit_pages(self, targets: list[str]) -> str:
        """Full text of up to ``VISIT_MAX_DOCS`` documents."""
        pages: list[str] = []
        for target in targets[:_MAX_VISITS]:
            doc_id = self._resolve(target)
            if doc_id is None:
                pages.append(
                    f"Could not open {_one_line(target)!r}: not a known document "
                    "url or id from these search results."
                )
                continue
            pages.append(self.browse(doc_id))
        return "\n\n".join(pages) if pages else "No url given."


#: Back-compat alias: every environment now mints the web surface.
WebResearchEnvironment = ResearchEnvironment
