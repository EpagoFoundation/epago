"""Environment subsystem tests: corpus search, masking, sync integrity, tool surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from epago.core.types import Task, TaskOrigin
from epago.environment import (
    CorpusIntegrityError,
    ResearchEnvironment,
    SqliteCorpus,
    build_fixture_corpus,
    corpus_digest,
    verify_corpus,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> SqliteCorpus:
    path = tmp_path_factory.mktemp("corpus") / "corpus.db"
    store = build_fixture_corpus(path, n_docs=120, seed=7)
    yield store
    store.close()


def _some_person(corpus: SqliteCorpus) -> tuple[str, str]:
    """Return (doc_id, title) of the first person document."""
    for doc_id in corpus.iter_doc_ids():
        doc = corpus.get(doc_id)
        assert doc is not None
        if doc.category == "person":
            return doc.doc_id, doc.title
    raise AssertionError("fixture corpus has no person documents")


def _make_task(masked: tuple[str, ...]) -> Task:
    return Task(
        task_id="t-0001",
        question="Who built the fixture?",
        answer="nobody",
        aliases=(),
        evidence_doc_ids=(),
        masked_doc_ids=masked,
        origin=TaskOrigin.GENERATED_PUBLIC,
        template="test",
    )


class TestSearch:
    def test_returns_relevant_doc(self, corpus: SqliteCorpus) -> None:
        doc_id, title = _some_person(corpus)
        hits = corpus.search(title, k=10)
        assert hits, "expected at least one hit for an exact title query"
        assert doc_id in {h.doc_id for h in hits}
        assert hits[0].doc_id == doc_id, "exact title match should rank first"

    def test_scores_descend_and_snippets_present(self, corpus: SqliteCorpus) -> None:
        hits = corpus.search("city founded", k=10)
        assert hits
        assert all(h.snippet for h in hits)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_fts_operators_are_neutralized(self, corpus: SqliteCorpus) -> None:
        # Raw model output must never crash the tool: operators become literals.
        for query in ('"unclosed phrase', "NEAR(a b)", "title:* OR -x", "AND", "   "):
            corpus.search(query, k=5)

    def test_k_limit_respected(self, corpus: SqliteCorpus) -> None:
        assert len(corpus.search("the", k=3)) <= 3
        assert corpus.search("the", k=0) == []

    def test_natural_language_question_retrieves(self, corpus: SqliteCorpus) -> None:
        # Terms are OR-ed: a question-shaped query must still retrieve, even
        # though no single document contains every word of the question.
        doc_id, title = _some_person(corpus)
        doc = corpus.get(doc_id)
        assert doc is not None
        question = f"In which city was the person named {title} born, and in what year?"
        hits = corpus.search(question, k=10)
        assert hits, "a full-sentence question must not return zero results"
        assert doc_id in {h.doc_id for h in hits}

    def test_extra_unmatched_terms_do_not_lose_the_document(self, corpus: SqliteCorpus) -> None:
        doc_id, title = _some_person(corpus)
        padded = f"{title} zzzzqqqqxxxx wwwwvvvvuuuu"
        assert doc_id in {h.doc_id for h in corpus.search(padded, k=10)}

    def test_query_of_only_common_words_still_matches(self, corpus: SqliteCorpus) -> None:
        # Pruning high-document-frequency terms must never empty a query.
        assert corpus.search("the a in", k=5)

    def test_common_terms_do_not_drag_in_the_whole_corpus(
        self, corpus: SqliteCorpus
    ) -> None:
        # "the" alone matches nearly every document; paired with a rare name it
        # is pruned, so the result set stays as small as the rare term's own.
        _, title = _some_person(corpus)
        assert len(corpus.search("the", k=10)) == 10
        assert len(corpus.search(f"the {title}", k=10)) == len(corpus.search(title, k=10)) < 10


class TestMasking:
    def test_masked_doc_hidden_from_search(self, corpus: SqliteCorpus) -> None:
        doc_id, title = _some_person(corpus)
        mask = frozenset({doc_id})
        hits = corpus.search(title, k=10, mask_doc_ids=mask)
        assert doc_id not in {h.doc_id for h in hits}

    def test_mask_filter_applied_before_k_limit(self, corpus: SqliteCorpus) -> None:
        # Masking the top hit must not shrink the result list when other
        # matches exist: masked docs may not consume ranked slots.
        hits = corpus.search("city", k=5)
        assert len(hits) == 5
        mask = frozenset({hits[0].doc_id})
        masked_hits = corpus.search("city", k=5, mask_doc_ids=mask)
        assert len(masked_hits) == 5
        assert hits[0].doc_id not in {h.doc_id for h in masked_hits}

    def test_masked_doc_not_gettable(self, corpus: SqliteCorpus) -> None:
        doc_id, _ = _some_person(corpus)
        assert corpus.get(doc_id) is not None
        assert corpus.get(doc_id, mask_doc_ids=frozenset({doc_id})) is None

    def test_masked_doc_not_browsable_via_session(self, corpus: SqliteCorpus) -> None:
        doc_id, title = _some_person(corpus)
        session = ResearchEnvironment(corpus).tools_for_task(_make_task(masked=(doc_id,)))
        assert session.browse(doc_id) == f"Document not found: {doc_id}"
        assert doc_id not in session.search(title)


class TestSyncIntegrity:
    def test_digest_roundtrip(self, tmp_path: Path) -> None:
        store = build_fixture_corpus(tmp_path / "c.db", n_docs=20, seed=1)
        store.close()
        digest = corpus_digest(tmp_path / "c.db")
        assert digest.startswith("sha256:")
        verify_corpus(tmp_path / "c.db", digest)

    def test_tamper_detected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "c.db"
        store = build_fixture_corpus(db_path, n_docs=20, seed=1)
        store.close()
        digest = corpus_digest(db_path)
        with open(db_path, "ab") as fh:
            fh.write(b"\x00")
        with pytest.raises(CorpusIntegrityError, match="mismatch"):
            verify_corpus(db_path, digest)

    def test_bad_digest_format_rejected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "c.db"
        db_path.write_bytes(b"x")
        with pytest.raises(CorpusIntegrityError, match="format"):
            verify_corpus(db_path, "sha256:nothex")


class TestToolSession:
    def test_output_deterministic_across_sessions(self, corpus: SqliteCorpus) -> None:
        env = ResearchEnvironment(corpus)
        _, title = _some_person(corpus)
        task = _make_task(masked=())
        outputs = []
        for _ in range(2):
            session = env.tools_for_task(task)
            search_out = session.search(title)
            first_doc_id = search_out.splitlines()[0].split("[", 1)[1].split("]", 1)[0]
            outputs.append((search_out, session.browse(first_doc_id)))
        assert outputs[0] == outputs[1]

    def test_search_formatting(self, corpus: SqliteCorpus) -> None:
        session = ResearchEnvironment(corpus).tools_for_task(_make_task(masked=()))
        out = session.search("city founded")
        lines = out.splitlines()
        assert lines[0].startswith("1. [fx-")
        assert all("\n" not in line and line[0].isdigit() for line in lines)
        assert session.search("zzzzqqqqxxxx") == "No results found."

    def test_browse_formatting_and_truncation(self, corpus: SqliteCorpus) -> None:
        doc_id = next(corpus.iter_doc_ids())
        doc = corpus.get(doc_id)
        assert doc is not None
        session = ResearchEnvironment(corpus, page_chars=50).tools_for_task(_make_task(masked=()))
        out = session.browse(doc_id)
        assert out.startswith(f"# {doc.title}")
        assert f"doc_id: {doc_id}" in out
        assert "[truncated at 50 characters]" in out
        assert session.browse("no-such-doc") == "Document not found: no-such-doc"

    def test_call_counters(self, corpus: SqliteCorpus) -> None:
        session = ResearchEnvironment(corpus).tools_for_task(_make_task(masked=()))
        session.search("city")
        session.search("city")
        session.browse("no-such-doc")
        assert (session.search_calls, session.browse_calls) == (2, 1)


class TestFixtureDeterminism:
    def test_same_seed_same_digest(self, tmp_path: Path) -> None:
        for name in ("a.db", "b.db"):
            store = build_fixture_corpus(tmp_path / name, n_docs=60, seed=13)
            store.close()
        assert corpus_digest(tmp_path / "a.db") == corpus_digest(tmp_path / "b.db")

    def test_different_seed_different_digest(self, tmp_path: Path) -> None:
        for name, seed in (("a.db", 13), ("b.db", 14)):
            store = build_fixture_corpus(tmp_path / name, n_docs=60, seed=seed)
            store.close()
        assert corpus_digest(tmp_path / "a.db") != corpus_digest(tmp_path / "b.db")

    def test_doc_count(self, corpus: SqliteCorpus) -> None:
        assert corpus.doc_count() == 120
        assert len(list(corpus.iter_doc_ids())) == 120
