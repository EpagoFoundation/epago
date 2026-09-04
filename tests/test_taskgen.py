"""Tests for the taskgen subsystem.

Uses a self-contained dict-backed fake corpus (duck-typed against the
``CorpusStore`` protocol) so these tests do not depend on the environment
package or any storage backend.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from epago import constants
from epago.core.types import Task, TaskOrigin
from epago.taskgen import (
    PrivatePool,
    generate_tasks,
    task_ids_digest,
    verify_task,
)
from epago.taskgen.templates import templates_for_release

# --- fake corpus --------------------------------------------------------------


@dataclass(frozen=True)
class FakeDoc:
    doc_id: str
    url: str
    title: str
    text: str
    category: str = ""


@dataclass(frozen=True)
class FakeHit:
    doc_id: str
    title: str
    snippet: str
    score: float


class FakeCorpus:
    """Dict-backed CorpusStore: term-frequency search, sorted-id iteration."""

    def __init__(self, docs: list[FakeDoc]) -> None:
        self._docs = {d.doc_id: d for d in docs}

    def search(self, query, k=10, mask_doc_ids=frozenset()):
        terms = [t.strip('"').lower() for t in query.split() if t.strip('"')]
        scored = []
        for doc_id in sorted(self._docs):
            if doc_id in mask_doc_ids:
                continue
            doc = self._docs[doc_id]
            haystack = f"{doc.title}\n{doc.text}".lower()
            score = sum(haystack.count(t) for t in terms)
            if score > 0:
                scored.append((-float(score), doc_id))
        scored.sort()
        return [
            FakeHit(doc_id=d, title=self._docs[d].title, snippet=self._docs[d].text[:200], score=-s)
            for s, d in scored[:k]
        ]

    def get(self, doc_id, mask_doc_ids=frozenset()):
        if doc_id in mask_doc_ids:
            return None
        return self._docs.get(doc_id)

    def doc_count(self):
        return len(self._docs)

    def iter_doc_ids(self):
        yield from sorted(self._docs)


def _doc(doc_id: str, title: str, text: str) -> FakeDoc:
    return FakeDoc(doc_id=doc_id, url=f"https://corpus.test/{doc_id}", title=title, text=text)


@pytest.fixture()
def corpus() -> FakeCorpus:
    return FakeCorpus(
        [
            _doc(
                "d01",
                "Alto Bridge",
                "The Alto Bridge was completed in 1932. The Alto Bridge spans the "
                "Kelda River near the town of Varno. Engineers from the Varno "
                "Institute designed the crossing over several seasons.",
            ),
            _doc(
                "d02",
                "Kelda River",
                "The Kelda River was surveyed in 1901. The Kelda River flows south "
                "through the Varno Valley and past Marnex Falls. Fishing "
                "communities settled along its banks long before modern towns.",
            ),
            _doc(
                "d03",
                "Varno Institute",
                "The Varno Institute was founded in 1898. The Varno Institute "
                "trains civil engineers and hydrologists. Its campus sits on the "
                "eastern bank of the Kelda River.",
            ),
            _doc(
                "d04",
                "Marnex Falls",
                "Marnex Falls was mapped in 1874. Marnex Falls drops into the "
                "lower Kelda gorge below the escarpment. Visitors reach the falls "
                "by a trail that begins in Varno.",
            ),
            _doc(
                "d05",
                "Town of Varno",
                "Varno was incorporated in 1856. The town grew around a ferry "
                "crossing on the Kelda River. Varno hosts an annual bridge "
                "festival every autumn.",
            ),
            _doc(
                "d06",
                "Serel Observatory",
                "The Serel Observatory was opened in 1911. The Serel Observatory "
                "studies upper-atmosphere weather patterns. Astronomers at the "
                "site publish a yearly almanac.",
            ),
            _doc(
                "d07",
                "Serel Almanac",
                "The Serel Almanac was first printed in 1913. The Serel Almanac "
                "records rainfall totals for the Varno Valley. Copies are "
                "archived at the Varno Institute.",
            ),
            _doc(
                "d08",
                "Port Denna",
                "Port Denna was established in 1789. Port Denna sits at the mouth "
                "of the Kelda River. Grain from the Varno Valley ships through "
                "Port Denna each harvest.",
            ),
        ]
    )


# --- generation ---------------------------------------------------------------


def test_generation_deterministic(corpus):
    a = generate_tasks(seed=1234, release="R1", corpus=corpus, n=5)
    b = generate_tasks(seed=1234, release="R1", corpus=corpus, n=5)
    assert a == b  # byte-identical tasks, same order
    assert task_ids_digest(a) == task_ids_digest(b)
    assert len(a) == 5
    for task in a:
        assert task.origin is TaskOrigin.GENERATED_PUBLIC
        assert verify_task(task, corpus).ok


def test_generation_seed_sensitivity(corpus):
    a = generate_tasks(seed=1234, release="R1", corpus=corpus, n=5)
    b = generate_tasks(seed=4321, release="R1", corpus=corpus, n=5)
    assert [t.task_id for t in a] != [t.task_id for t in b]


def test_generation_unknown_release(corpus):
    with pytest.raises(ValueError, match="unknown taskgen release"):
        generate_tasks(seed=1, release="R999", corpus=corpus, n=1)


def test_difficulty_band_filter_consumes_no_rng(corpus):
    class BandProbe:
        def solve_rate(self, task, k=4):
            return 0.5  # always inside the band

    with_probe = generate_tasks(seed=77, release="R1", corpus=corpus, n=5, king_probe=BandProbe())
    without = generate_tasks(seed=77, release="R1", corpus=corpus, n=5)
    assert with_probe == without


# --- QA -------------------------------------------------------------------------


def test_qa_rejects_answer_not_in_evidence(corpus):
    task = Task(
        task_id="tk-deadbeefdeadbeef",
        question="In which year was the Alto Bridge completed?",
        answer="9999",
        aliases=(),
        evidence_doc_ids=("d01",),
        masked_doc_ids=(),
        origin=TaskOrigin.GENERATED_PRIVATE,
        template="qa-fixture",
        hops=1,
    )
    report = verify_task(task, corpus)
    assert not report.ok
    assert "answer_not_in_evidence" in report.failures


def test_qa_rejects_missing_evidence_and_bad_mask(corpus):
    base = Task(
        task_id="tk-0000000000000000",
        question="In which year was the Alto Bridge completed?",
        answer="1932",
        aliases=(),
        evidence_doc_ids=("d99",),
        masked_doc_ids=(),
        origin=TaskOrigin.GENERATED_PRIVATE,
        template="qa-fixture",
        hops=1,
    )
    report = verify_task(base, corpus)
    assert not report.ok
    assert "evidence_missing:d99" in report.failures

    fully_masked = replace(base, evidence_doc_ids=("d01",), masked_doc_ids=("d01",))
    report = verify_task(fully_masked, corpus)
    assert not report.ok
    assert "mask_unsolvable" in report.failures


def test_qa_rejects_malformed_answers(corpus):
    good = Task(
        task_id="tk-1111111111111111",
        question="In which year was the Alto Bridge completed?",
        answer="1932",
        aliases=(),
        evidence_doc_ids=("d01",),
        masked_doc_ids=(),
        origin=TaskOrigin.GENERATED_PRIVATE,
        template="qa-fixture",
        hops=1,
    )
    assert verify_task(good, corpus).ok
    assert not verify_task(replace(good, answer=""), corpus).ok
    assert not verify_task(replace(good, answer="<b>1932</b>"), corpus).ok
    assert not verify_task(
        replace(good, answer="x" * (constants.ANSWER_MAX_CHARS + 1)), corpus
    ).ok


# --- private pool ---------------------------------------------------------------


def test_private_pool_rotation_publishes_committed_bytes(corpus, tmp_path: Path):
    tasks = generate_tasks(seed=99, release="R1", corpus=corpus, n=4)
    pool = PrivatePool(epoch=3, created_block=100, tasks=tuple(tasks))
    committed = pool.digest()  # what verdicts would have committed on-chain
    assert committed.startswith("sha256:")

    publish_dir = tmp_path / "published"
    new_pool = pool.rotate(new_tasks=tasks[:2], current_block=500, publish_dir=publish_dir)

    published = list(publish_dir.iterdir())
    assert len(published) == 1
    assert f"epoch{3:06d}" in published[0].name
    assert committed.removeprefix("sha256:")[:16] in published[0].name
    # Delayed transparency: the published bytes hash to the committed digest.
    raw = published[0].read_bytes()
    assert "sha256:" + hashlib.sha256(raw).hexdigest() == committed
    payload = json.loads(raw)
    assert payload["epoch"] == 3
    assert len(payload["tasks"]) == 4
    assert all("answer" in t and t["evidence_doc_ids"] for t in payload["tasks"])

    assert new_pool.epoch == 4
    assert new_pool.created_block == 500
    assert not new_pool.due_for_rotation(500)
    assert new_pool.due_for_rotation(500 + constants.PRIVATE_POOL_ROTATION_BLOCKS)


def test_private_pool_sample_and_persistence(corpus, tmp_path: Path):
    tasks = generate_tasks(seed=7, release="R1", corpus=corpus, n=4)
    pool = PrivatePool(epoch=0, created_block=0, tasks=tuple(tasks), storage_path=tmp_path / "pool")
    pool.save()
    loaded = PrivatePool.load(tmp_path / "pool")
    assert loaded.digest() == pool.digest()

    s1 = pool.sample(3, np.random.Generator(np.random.PCG64(5)))
    s2 = pool.sample(3, np.random.Generator(np.random.PCG64(5)))
    assert s1 == s2
    assert len(s1) == 3


# --- scientific-literature release (SCI1) -------------------------------------


def _paper_corpus(tmp_path):
    """A tiny corpus shaped like real abstracts: title in the text, results
    sentences carrying reported values."""
    from epago.environment.corpus import Document, SqliteCorpus

    papers = [
        (
            "Clinical profile of dengue infection at a teaching hospital",
            "RESULTS: Out of 356 patients with suspected dengue fever enrolled in the "
            "study, 138 (39%) had serologically confirmed dengue infection. The mean "
            "age of admission was 8.7 yrs. Mortality was 1.41% overall.",
        ),
        (
            "Elective case cancellation on the day of surgery: causes and solutions",
            "RESULTS: The rate of cancellations on the day of surgery for elective "
            "procedures during 2012 was 5.19%. Most cancellations were avoidable and "
            "the reduction achieved after intervention reached 2.4%.",
        ),
        (
            "Active travel to school among children aged 11 to 15 years",
            "RESULTS: Overall, 21.4% of children engaged in at least some active "
            "travel to school. Prevalence was significantly associated with distance "
            "and the observed odds ratio was 1.21.",
        ),
    ]
    db = tmp_path / "papers.db"
    corpus = SqliteCorpus.create(db)
    corpus.add_documents(
        Document(
            doc_id=f"ep-paper{i}",
            url=f"https://example.org/{i}",
            title=title,
            # Real corpora lead the text with the title so it is searchable and
            # so a title-answer task can pass the evidence check.
            text=f"{title}\n\n{body}",
        )
        for i, (title, body) in enumerate(papers)
    )
    return corpus


def test_sci1_release_excludes_aggregation():
    """Aggregation mints "how many distinct years appear", which measures
    counting rather than comprehension and crowded out the R1 mixture."""
    names = [t.name for t in templates_for_release("SCI1")]
    assert "aggregation" not in names
    assert "study_finding" in names
    assert "find_the_study" in names


def test_study_finding_extracts_a_reported_value(tmp_path):
    from epago.taskgen.templates import StudyFindingTemplate

    corpus = _paper_corpus(tmp_path)
    template = StudyFindingTemplate()
    minted = []
    for seed in range(25):
        task = template.mint(corpus, np.random.Generator(np.random.PCG64(seed)))
        if task is not None:
            minted.append(task)
    assert minted, "no task minted from paper-shaped abstracts"

    for task in minted:
        # The title locates the study; the model still has to find and read it.
        assert task.question.startswith("In the study titled")
        assert "____" in task.question
        assert verify_task(task, corpus).ok
        # The answer is a reported value, present verbatim in the evidence.
        doc = corpus.get(task.evidence_doc_ids[0])
        assert task.answer in doc.text


def test_find_the_study_answers_with_a_title(tmp_path):
    """The citation task: given a finding, name its source."""
    from epago.taskgen.templates import FindTheStudyTemplate

    corpus = _paper_corpus(tmp_path)
    template = FindTheStudyTemplate()
    titles = {corpus.get(f"ep-paper{i}").title for i in range(3)}
    minted = []
    for seed in range(25):
        task = template.mint(corpus, np.random.Generator(np.random.PCG64(seed)))
        if task is not None:
            minted.append(task)
    assert minted

    for task in minted:
        assert task.answer in titles
        assert "Answer with the study title" in task.question
        assert verify_task(task, corpus).ok


def test_sci1_generation_is_deterministic(tmp_path):
    corpus = _paper_corpus(tmp_path)
    a = generate_tasks(seed=4242, release="SCI1", corpus=corpus, n=6, king_probe=None)
    b = generate_tasks(seed=4242, release="SCI1", corpus=corpus, n=6, king_probe=None)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert all(verify_task(t, corpus).ok for t in a)


# --- SCI2: hard-by-construction templates -------------------------------------


def _overlapping_corpus(tmp_path):
    """Papers sharing topic vocabulary, like a real review corpus. The
    described_finding template deliberately refuses to mint when a title
    shares no words with other papers — nothing to describe by."""
    from epago.environment.corpus import Document, SqliteCorpus

    papers = [
        (
            "Dengue fever outcomes in hospitalized children in South India",
            "RESULTS: Of the enrolled children, 138 (39%) had confirmed dengue "
            "infection and mortality reached 1.41% in this hospitalized cohort.",
        ),
        (
            "Dengue infection severity among children admitted to a teaching hospital",
            "RESULTS: The mean age at admission was 8.7 yrs and 70% of the 556 "
            "admitted children were male in this dengue cohort.",
        ),
        (
            "Clinical predictors of severe dengue in hospitalized children",
            "RESULTS: Severe dengue occurred in 21.4% of hospitalized children "
            "and the observed odds ratio for plasma leakage was 1.21.",
        ),
        (
            "Dengue seroprevalence in children attending a hospital outpatient clinic",
            "RESULTS: Seroprevalence among children was 52.12% and increased "
            "significantly with age across the hospital catchment population.",
        ),
    ]
    db = tmp_path / "overlap.db"
    corpus = SqliteCorpus.create(db)
    corpus.add_documents(
        Document(
            doc_id=f"ep-ov{i}",
            url=f"https://example.org/ov{i}",
            title=title,
            text=f"{title}\n\n{body}",
        )
        for i, (title, body) in enumerate(papers)
    )
    return corpus


def test_sci2_claim_window_masks_all_other_numbers(tmp_path):
    """Numbers are near-unique tokens, so any stray digit in the quoted window
    is a search key that hands the model the source in one query. Measured on a
    real corpus, that leak alone put the source at rank 1 for 89.6% of tasks."""
    from epago.taskgen.templates import DescribedFindingTemplate

    corpus = _overlapping_corpus(tmp_path)
    template = DescribedFindingTemplate()
    minted = [
        t for t in (
            template.mint(corpus, np.random.Generator(np.random.PCG64(seed)))
            for seed in range(40)
        ) if t is not None
    ]
    assert minted, "no described_finding task minted"
    for task in minted:
        quoted = task.question.split('"')[1]
        digits = re.findall(r"\d[\d,\.]*", quoted.replace("____", ""))
        assert not digits, f"window leaks numbers: {quoted!r}"
        assert "____" in quoted
        assert verify_task(task, corpus).ok


def test_sci2_never_names_the_study(tmp_path):
    """A full title is a unique string — quoting it is rank-1 retrieval."""
    from epago.taskgen.templates import DescribedFindingTemplate

    corpus = _overlapping_corpus(tmp_path)
    template = DescribedFindingTemplate()
    titles = {corpus.get(f"ep-ov{i}").title for i in range(4)}
    for seed in range(40):
        task = template.mint(corpus, np.random.Generator(np.random.PCG64(seed)))
        if task is None:
            continue
        for title in titles:
            assert title not in task.question


def test_comparative_finding_needs_the_whole_field(tmp_path):
    """The comparison task quotes nothing: no single search can answer it."""
    from epago.environment.corpus import Document, SqliteCorpus
    from epago.taskgen.templates import ComparativeFindingTemplate

    db = tmp_path / "cmp.db"
    corpus = SqliteCorpus.create(db)
    corpus.add_documents(
        Document(
            doc_id=f"ep-m{i}",
            url=f"https://example.org/m{i}",
            title=f"Cohort study {chr(65 + i)} of postoperative outcomes in adults",
            text=(
                f"Cohort study {chr(65 + i)} of postoperative outcomes in adults\n\n"
                f"RESULTS: In-hospital mortality was {rate}% among the enrolled patients."
            ),
        )
        for i, rate in enumerate((12.5, 3.1, 22.4))
    )
    template = ComparativeFindingTemplate()
    task = None
    for seed in range(20):
        task = template.mint(corpus, np.random.Generator(np.random.PCG64(seed)))
        if task is not None:
            break
    assert task is not None
    assert task.answer == "Cohort study C of postoperative outcomes in adults"  # 22.4% is max
    assert len(task.evidence_doc_ids) == 3          # the whole field is evidence
    assert task.hops == 3
    # The question quotes no document text and no title.
    assert "22.4" not in task.question and "Cohort study" not in task.question
    assert verify_task(task, corpus).ok


def test_comparative_finding_ignores_confidence_intervals(tmp_path):
    """"95% CI" is interval notation, not a finding — on a real corpus it
    manufactured constant ties at 95.0 and the template minted nothing."""
    from epago.environment.corpus import Document, SqliteCorpus
    from epago.taskgen.templates import ComparativeFindingTemplate

    db = tmp_path / "ci.db"
    corpus = SqliteCorpus.create(db)
    corpus.add_documents(
        Document(
            doc_id=f"ep-c{i}",
            url=f"https://example.org/c{i}",
            title=f"Trial {chr(65 + i)} of an intervention in a clinical population",
            text=(
                f"Trial {chr(65 + i)} of an intervention in a clinical population\n\n"
                f"RESULTS: Mortality was {rate}% (95% CI applies to all estimates)."
            ),
        )
        for i, rate in enumerate((8.0, 15.5, 4.2))
    )
    template = ComparativeFindingTemplate()
    task = None
    for seed in range(20):
        task = template.mint(corpus, np.random.Generator(np.random.PCG64(seed)))
        if task is not None:
            break
    assert task is not None
    # 15.5 wins — not the 95 from "95% CI".
    assert task.answer.startswith("Trial B")


# --- SCI3: the same task shapes over all of scientific literature -------------


def _nonmedical_corpus(tmp_path):
    """Papers from disciplines that state findings without a single medical
    word: photovoltaics and machine learning. Titles overlap so the described
    study has ambiguous topic words to be described by."""
    from epago.environment.corpus import Document, SqliteCorpus

    papers = [
        (
            f"Perovskite solar cell stack {chr(65 + i)} with an engineered interface layer",
            "RESULTS: The certified power conversion efficiency of the device achieved "
            f"{rate}% after 500 hours of continuous illumination.",
        )
        for i, rate in enumerate((24.7, 19.2, 31.5))
    ] + [
        (
            f"Sparse attention transformer variant {chr(65 + i)} on long-document retrieval",
            "RESULTS: The model achieved a retrieval accuracy of "
            f"{acc}% on the held-out benchmark split.",
        )
        for i, acc in enumerate((81.4, 76.3, 88.9))
    ]
    db = tmp_path / "nonmed.db"
    corpus = SqliteCorpus.create(db)
    corpus.add_documents(
        Document(
            doc_id=f"ep-nm{i}",
            url=f"https://example.org/nm{i}",
            title=title,
            text=f"{title}\n\n{body}",
        )
        for i, (title, body) in enumerate(papers)
    )
    return corpus


def test_sci2_vocabulary_is_frozen():
    """SCI2's word lists are as much a determinism contract as its template
    tuple: re-binding them would silently change every validator's holdouts."""
    from epago.taskgen.templates import MEDICAL_VOCABULARY, vocabulary_for_release

    assert vocabulary_for_release("SCI2") is MEDICAL_VOCABULARY
    assert vocabulary_for_release("SCI1") is MEDICAL_VOCABULARY
    assert vocabulary_for_release("R1") is MEDICAL_VOCABULARY
    assert MEDICAL_VOCABULARY.result_cues == (
        "result", "found", "showed", "reported", "observed", "associated",
        "significant", "increase", "decrease", "reduction", "risk", "rate",
        "prevalence", "incidence", "mortality", "odds", "hazard", "mean", "median",
    )
    assert MEDICAL_VOCABULARY.comparable_cues == (
        "mortality", "prevalence", "incidence", "sensitivity", "specificity",
        "seropositivity", "response rate", "success rate", "complication", "recurrence",
    )


def test_sci3_is_sci2_shapes_with_a_general_vocabulary():
    """Nothing about the mechanics is medical, so SCI3 changes only the words
    — plus the one shape that was retired for being unanswerable."""
    from epago.taskgen.templates import (
        GENERAL_VOCABULARY,
        RELEASES,
        vocabulary_for_release,
    )

    assert RELEASES["SCI3"] == tuple(
        n for n in RELEASES["SCI2"] if n != "comparative_finding"
    )
    assert vocabulary_for_release("SCI3") is GENERAL_VOCABULARY
    for template in templates_for_release("SCI3"):
        assert getattr(template, "vocab", GENERAL_VOCABULARY) is GENERAL_VOCABULARY
    # ...and binding a vocabulary must not have leaked into the older release.
    for template in templates_for_release("SCI2"):
        assert getattr(template, "vocab", None) is not GENERAL_VOCABULARY


def test_general_vocabulary_keeps_every_medical_word():
    """Life and health sciences are part of "all science": a general vocabulary
    that dropped `mortality` would go blind to a whole discipline."""
    from epago.taskgen.templates import GENERAL_VOCABULARY, MEDICAL_VOCABULARY

    assert set(MEDICAL_VOCABULARY.result_cues) <= set(GENERAL_VOCABULARY.result_cues)
    assert set(MEDICAL_VOCABULARY.comparable_cues) <= set(GENERAL_VOCABULARY.comparable_cues)
    assert len(set(GENERAL_VOCABULARY.comparable_cues)) == len(GENERAL_VOCABULARY.comparable_cues)


def test_general_gate_keeps_findings_the_medical_gate_drops():
    """The mintability gate is what a harvest keeps papers by. Under the medical
    words a physics result is not a result at all."""
    from epago.taskgen.templates import (
        GENERAL_VOCABULARY,
        MEDICAL_VOCABULARY,
        _finding_sentences,
        finding_sentences,
    )

    physics = (
        "The device achieved a power conversion efficiency of 24.7% under "
        "one-sun illumination."
    )
    assert finding_sentences(physics, MEDICAL_VOCABULARY) == []
    assert finding_sentences(physics, GENERAL_VOCABULARY) == [physics]
    # Back-compat: the private name and the bare call still mean "medical".
    assert _finding_sentences is finding_sentences
    assert _finding_sentences(physics) == []


def test_sci3_comparative_finding_mints_where_sci2_starves(tmp_path):
    """The whole point of the generalization: on non-medical text SCI2's fixed
    comparable quantities match nothing, so the multi-document comparison task
    — the one no single search can answer — silently disappears."""
    from epago.taskgen.templates import (
        GENERAL_VOCABULARY,
        MEDICAL_VOCABULARY,
        ComparativeFindingTemplate,
    )

    corpus = _nonmedical_corpus(tmp_path)
    sci2 = ComparativeFindingTemplate(MEDICAL_VOCABULARY)
    sci3 = ComparativeFindingTemplate(GENERAL_VOCABULARY)
    rngs = [np.random.Generator(np.random.PCG64(seed)) for seed in range(12)]

    assert all(sci2.mint(corpus, rng) is None for rng in rngs)

    minted = [
        t for t in (
            sci3.mint(corpus, np.random.Generator(np.random.PCG64(seed)))
            for seed in range(12)
        ) if t is not None
    ]
    assert minted, "general vocabulary minted no comparison task either"
    for task in minted:
        # Still a real comparison: the whole field is evidence, and the answer
        # is the title of the study reporting the highest value.
        assert task.hops == len(task.evidence_doc_ids) >= 3
        assert task.answer in {
            "Perovskite solar cell stack C with an engineered interface layer",
            "Sparse attention transformer variant C on long-document retrieval",
        }
        assert verify_task(task, corpus).ok


def test_sci3_generation_is_deterministic_on_non_medical_text(tmp_path):
    corpus = _nonmedical_corpus(tmp_path)
    a = generate_tasks(seed=8686, release="SCI3", corpus=corpus, n=5, king_probe=None)
    b = generate_tasks(seed=8686, release="SCI3", corpus=corpus, n=5, king_probe=None)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert a == b
    assert all(verify_task(t, corpus).ok for t in a)


def test_sci3_does_not_disturb_sci2_on_the_same_corpus(tmp_path):
    """Adding a release must leave the older one bit-for-bit where it was."""
    corpus = _overlapping_corpus(tmp_path)
    before = generate_tasks(seed=4242, release="SCI2", corpus=corpus, n=3)
    generate_tasks(seed=4242, release="SCI3", corpus=corpus, n=3)
    after = generate_tasks(seed=4242, release="SCI2", corpus=corpus, n=3)
    assert before == after


def test_sci2_still_carries_the_retired_comparison_shape():
    """SCI2 is a frozen determinism contract: retiring `comparative_finding`
    from the live release must not touch it, or every pinned SCI2 holdout is
    silently re-minted."""
    from epago.taskgen.templates import RELEASES, templates_for_release

    assert RELEASES["SCI2"] == ("described_finding", "comparative_finding", "cross_doc_join")
    assert [t.name for t in templates_for_release("SCI2")] == list(RELEASES["SCI2"])
    # The live release drops it: no top-k retrieval protocol can answer it
    # (measured 0% accuracy across six protocol variants).
    assert "comparative_finding" not in RELEASES["SCI3"]


# --- SCI4: hard to find, easy to check ----------------------------------------


def _sci4_corpus(tmp_path):
    """A corpus with crowd structure: two fields, spread cue values, stated
    sample sizes, and shared vocabulary — enough for every constraint kind to
    find a crowd and for the leak gate to have somewhere to hide the gold."""
    from epago.environment.corpus import Document, SqliteCorpus

    fields = [
        ("microbiology", "Life Sciences"),
        ("economics", "Social Sciences"),
    ]
    docs = []
    for i in range(30):
        field, domain = fields[i % 2]
        accuracy = 10 + 2 * i  # 10..68, spaced 2pp so margins always hold
        n = 100 + 17 * i
        extras = []
        if i % 4 == 0:
            extras.append("The cohort was recruited from a regional hospital.")
        if field == "microbiology" and i % 3 == 0:
            extras.append("Samples were collected in tropical wetland sites.")
        title = f"Longitudinal Cohort Study Number {i:02d} on Adaptive Systems"
        body = (
            f"{title}\n\nThis {field} study reports an accuracy of {accuracy}%. "
            f"The enrolled sample includes n = {n} participants overall. "
            + " ".join(extras)
        )
        docs.append(
            Document(
                doc_id=f"ep-sci4-{i:02d}",
                url=f"https://example.org/sci4/{i}",
                title=title,
                text=body,
                category=f"openalex:d1:{field}|{domain}",
            )
        )
    corpus = SqliteCorpus.create(tmp_path / "sci4.db")
    corpus.add_documents(docs)
    return corpus


def test_sci4_release_registered():
    from epago.taskgen.templates import (
        MASK_EXEMPT_TEMPLATES,
        RELEASES,
        SELF_PINNED_TEMPLATES,
        vocabulary_for_release,
    )

    names = RELEASES["SCI4"]
    assert names == ("constrained_study", "named_set_superlative", "named_set_count")
    assert vocabulary_for_release("SCI4").name == "general"
    for name in names:
        assert name in MASK_EXEMPT_TEMPLATES
        assert name in SELF_PINNED_TEMPLATES
    # Pinned releases must be untouched by the SCI4 registration.
    assert RELEASES["SCI3"] == ("described_finding", "cross_doc_join")


def test_profile_extraction_rules():
    from epago.taskgen import profiles

    text = (
        "Intro sentence. The model achieved an accuracy of 81.5% overall. "
        "A subgroup accuracy of 64.2% was seen (95% CI 60.1-68.3). "
        "Enrollment reached n = 1,204 in wave one and n = 96 in wave two. "
        "Specificity was 93.4% in validation."
    )
    cues = ("accuracy", "specificity")
    values = profiles.cue_percentages(text, cues)
    # Rule: the LARGEST percentage beside the cue, CI notation removed — the
    # 95 from "95% CI" must not win, and 81.5 beats the subgroup 64.2.
    assert values["accuracy"] == 81.5
    assert values["specificity"] == 93.4
    assert profiles.sample_size(text) == 1204  # largest stated n
    assert profiles.top_percentage(text) == 93.4  # corpus-wide max, CI removed
    assert profiles.parse_category("openalex:d1:microbiology|Life Sciences") == (
        "microbiology",
        "Life Sciences",
    )
    assert profiles.parse_category("") == ("", "")


def test_constrained_study_question_never_pins_the_gold(tmp_path):
    """The SCI4 contract: pasting the question into the real search backend
    must not surface the evidence document near the top, and the question
    must never contain the gold title or a verbatim text window."""
    corpus = _sci4_corpus(tmp_path)
    templates = {t.name: t for t in templates_for_release("SCI4")}
    rng = np.random.default_rng(11)
    minted = 0
    for _ in range(60):
        task = templates["constrained_study"].mint(corpus, rng)
        if task is None:
            continue
        minted += 1
        gold = task.evidence_doc_ids[0]
        doc = corpus.get(gold)
        if task.answer != doc.title:  # value mode still may not name the study
            assert doc.title not in task.question
        top3 = [h.doc_id for h in corpus.search(task.question, k=3)]
        assert gold not in top3
        assert verify_task(task, corpus).ok
        assert 3 <= task.hops <= 5
    assert minted >= 3  # the tiny corpus still mints; yield is not the point


def test_named_set_superlative_is_recomputable(tmp_path):
    from epago.taskgen import profiles

    corpus = _sci4_corpus(tmp_path)
    templates = {t.name: t for t in templates_for_release("SCI4")}
    rng = np.random.default_rng(5)
    task = templates["named_set_superlative"].mint(corpus, rng)
    assert task is not None
    assert f'"{task.answer}"' in task.question  # universe pinned by title
    # Re-derive the winner under the stated rule and check it matches.
    cue = next(c for c in ("accuracy",) if c in task.question)
    direction_high = "highest" in task.question
    values = {}
    for doc_id in task.evidence_doc_ids:
        doc = corpus.get(doc_id)
        values[doc.title] = profiles.cue_percentages(doc.text, (cue,))[cue]
    expect = max(values, key=values.get) if direction_high else min(values, key=values.get)
    assert task.answer == expect
    assert verify_task(task, corpus).ok


def test_named_set_count_recount_roundtrips(tmp_path):
    from epago.taskgen.templates import recount_named_set_count, recount_task

    corpus = _sci4_corpus(tmp_path)
    templates = {t.name: t for t in templates_for_release("SCI4")}
    rng = np.random.default_rng(9)
    task = templates["named_set_count"].mint(corpus, rng)
    assert task is not None
    # The count appears in no document; QA must verify it by recount.
    assert recount_named_set_count(task, corpus) == task.answer
    assert recount_task(task, corpus) == task.answer
    assert 1 <= int(task.answer) <= len(task.evidence_doc_ids) - 1
    assert verify_task(task, corpus).ok


def test_sci4_generation_is_deterministic(tmp_path):
    corpus = _sci4_corpus(tmp_path)
    a = generate_tasks(seed=777, release="SCI4", corpus=corpus, n=5, king_probe=None)
    b = generate_tasks(seed=777, release="SCI4", corpus=corpus, n=5, king_probe=None)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert all(verify_task(t, corpus).ok for t in a)


# --- the public half from a sealed pool --------------------------------------


def _pool_file(tmp_path, n=50):
    import json

    path = tmp_path / "pool.jsonl"
    rows = [
        {
            "task_id": f"tk-{i:04d}",
            "question": f"question {i}",
            "answer": f"answer {i}",
            "aliases": [],
            "evidence_doc_ids": [f"d{i}"],
            "masked_doc_ids": [],
            "template": "bridge_intersection",
            "hops": 3,
        }
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_a_pool_whose_digest_does_not_match_the_commitment_is_refused(tmp_path):
    """"The pool the validator used" and "the pool it committed to" must be the
    same claim, or the commitment proves nothing."""
    from epago.taskgen.sealed_pool import SealedPoolError, load_pool, pool_digest

    path = _pool_file(tmp_path)
    committed = pool_digest(path.read_bytes())

    assert len(load_pool(path, committed)) == 50

    path.write_text(path.read_text().replace("answer 0", "answer tampered"))
    with pytest.raises(SealedPoolError, match="refusing to duel"):
        load_pool(path, committed)


def test_selection_is_a_function_of_pool_contents_not_file_order(tmp_path):
    """A pool file's line order is an accident of minting.

    If it influenced selection, two byte-identical pools would disagree when
    either was rewritten — and an auditor re-deriving the exam would fail a
    verdict that was perfectly honest.
    """
    import json

    from epago.taskgen.sealed_pool import load_pool, select

    path = _pool_file(tmp_path)
    forward = select(load_pool(path), seed=1234, n=10)

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    path.write_text("\n".join(json.dumps(r) for r in reversed(rows)) + "\n")
    reversed_order = select(load_pool(path), seed=1234, n=10)

    assert [t.task_id for t in forward] == [t.task_id for t in reversed_order]


def test_a_different_block_seed_asks_different_questions(tmp_path):
    """Selection is seeded by a block hash nobody chose, so which tasks get
    asked is unknown even to whoever minted the pool."""
    from epago.taskgen.sealed_pool import load_pool, select

    tasks = load_pool(_pool_file(tmp_path))
    a = [t.task_id for t in select(tasks, seed=1, n=10)]
    b = [t.task_id for t in select(tasks, seed=2, n=10)]
    assert a != b
    assert len(set(a)) == 10, "a draw must not repeat a task"


def test_a_pool_smaller_than_one_exam_is_refused(tmp_path):
    """Otherwise every round asks the same questions and the exam is public."""
    from epago.taskgen.sealed_pool import SealedPoolError, load_pool, select

    tasks = load_pool(_pool_file(tmp_path, n=20))
    with pytest.raises(SealedPoolError, match="more than one exam"):
        select(tasks, seed=1, n=50)


def test_a_release_name_says_how_its_tasks_are_produced(tmp_path):
    """A replay reading only an audit record must be able to tell which
    verification path applies, without being told separately."""
    from epago.taskgen.sealed_pool import is_sealed_release

    assert is_sealed_release("POOL1")
    assert is_sealed_release("pool-chain-2026w36")
    assert not is_sealed_release("SCI4")


# --- sealed pool: manifest, disjoint rounds, publication ------------------------


def _sealed_pool(tmp_path, n=3000):
    """A pool file plus its committed digest, manifest and manifest digest."""
    from epago.taskgen.sealed_pool import Manifest, load_pool, pool_digest, write_manifest

    path = tmp_path / "pool.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": f"tk-{i:05d}",
                    "question": f"question {i}",
                    "answer": f"answer {i}",
                    "aliases": [],
                    "evidence_doc_ids": [f"d{i}"],
                    "masked_doc_ids": [],
                    "template": "bridge_intersection",
                    "hops": 3,
                }
            )
            for i in range(n)
        )
        + "\n"
    )
    digest = pool_digest(path.read_bytes())
    tasks = load_pool(path, digest)
    manifest = Manifest.from_pool(tasks, digest)
    manifest_digest = write_manifest(manifest, tmp_path / "manifest.json")
    return tasks, digest, manifest, manifest_digest


def test_the_manifest_reproduces_selection_without_revealing_the_pool(tmp_path):
    """The whole reason a manifest exists: an auditor proves which tasks a round
    asked while every unasked task stays sealed for later rounds."""
    from epago.taskgen.sealed_pool import select, select_ids

    tasks, _, manifest, _ = _sealed_pool(tmp_path)
    for seed in (1, 42, 2**31):
        by_task = [t.task_id for t in select(tasks, seed, 800)]
        by_id = select_ids(manifest.task_ids, seed, 800)
        assert by_task == by_id


def test_manifest_carries_ids_only_and_no_answers(tmp_path):
    """It publishes while the pool is still in service, so anything beyond an
    opaque id would be a live answer key."""
    _, _, manifest, _ = _sealed_pool(tmp_path, n=50)
    body = (tmp_path / "manifest.json").read_text()
    assert "answer 0" not in body
    assert "question 0" not in body
    assert "tk-00000" in body


def test_manifest_digest_ignores_id_order(tmp_path):
    """Selection uses the id set, so the commitment must too."""
    from epago.taskgen.sealed_pool import Manifest

    _, digest, manifest, _ = _sealed_pool(tmp_path, n=50)
    shuffled = Manifest(
        format=manifest.format,
        pool_digest=digest,
        task_ids=tuple(reversed(manifest.task_ids)),
    )
    assert shuffled.digest() == manifest.digest()


def test_a_swapped_manifest_is_refused(tmp_path):
    """Without the digest check a validator could hand over an id list tailored
    to the tasks it wished it had asked."""
    from epago.taskgen.sealed_pool import (
        Manifest,
        SealedPoolError,
        load_manifest,
        write_manifest,
    )

    _, digest, _, committed = _sealed_pool(tmp_path, n=50)
    write_manifest(
        Manifest(format="eppm1", pool_digest=digest, task_ids=tuple(f"tk-{i:05d}" for i in range(49))),
        tmp_path / "swapped.json",
    )
    with pytest.raises(SealedPoolError, match="refusing to verify"):
        load_manifest(tmp_path / "swapped.json", committed)


def test_published_rounds_are_never_asked_again(tmp_path):
    """A round publishes its tasks in full, which makes them training data. If a
    later round could draw them, a challenger trained after that publication
    would answer part of its exam from memory instead of from research."""
    from epago.taskgen.sealed_pool import select

    tasks, _, _, _ = _sealed_pool(tmp_path)
    served: set[str] = set()
    for seed in (11, 22, 33):
        drawn = {t.task_id for t in select(tasks, seed, 800, exclude=served)}
        assert not (drawn & served)
        assert len(drawn) == 800
        served |= drawn
    assert len(served) == 2400


def test_exclusion_survives_a_pool_rotation(tmp_path):
    """Task ids are content-addressed, so a task re-minted into a fresh pool
    keeps the id it was published under and stays excluded."""
    from epago.taskgen.sealed_pool import select_ids

    _, _, manifest, _ = _sealed_pool(tmp_path, n=100)
    published = set(select_ids(manifest.task_ids, 5, 60))
    # A fresh pool that happens to re-mint the same tasks yields the same ids.
    redrawn = select_ids(manifest.task_ids, 9, 40, exclude=published)
    assert not (set(redrawn) & published)


def test_an_exhausted_pool_says_to_mint_a_new_one(tmp_path):
    """The operator needs to know the pool ran out, not just that a draw failed."""
    from epago.taskgen.sealed_pool import SealedPoolError, select

    tasks, _, manifest, _ = _sealed_pool(tmp_path, n=1000)
    served = {t.task_id for t in select(tasks, 1, 800)}
    with pytest.raises(SealedPoolError, match="mint and commit a fresh pool"):
        select(tasks, 2, 800, exclude=served)


def test_duplicate_task_ids_are_refused_at_load(tmp_path):
    """Selection runs over sorted ids on the validator side and over the
    manifest's ids on the auditor's; duplicates would desynchronise them."""
    from epago.taskgen.sealed_pool import SealedPoolError, load_pool

    path = tmp_path / "dupes.jsonl"
    row = {
        "task_id": "tk-00001",
        "question": "q",
        "answer": "a",
        "aliases": [],
        "evidence_doc_ids": ["d1"],
        "masked_doc_ids": [],
        "template": "bridge_intersection",
        "hops": 3,
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(SealedPoolError, match="repeats"):
        load_pool(path)


def test_a_published_round_carries_only_the_tasks_it_asked(tmp_path):
    """Publishing the whole pool after every round would hand a miner every
    question and answer, and the pool would survive exactly one round."""
    from epago.taskgen.sealed_pool import load_round_file, round_payload, select

    tasks, digest, _, manifest_digest = _sealed_pool(tmp_path)
    drawn = select(tasks, 7, 800)
    payload = round_payload(
        drawn,
        round_no=4,
        task_ids_digest=task_ids_digest(drawn),
        pool_digest_value=digest,
        manifest_digest=manifest_digest,
    )
    path = tmp_path / "round.json"
    path.write_text(payload)
    published = load_round_file(path)

    assert len(published) == 800
    assert {t.task_id for t in published} == {t.task_id for t in drawn}
    # The 2,200 unasked tasks stay sealed.
    assert "question 2999" not in payload or drawn[0].question == "question 2999"
    unasked = {t.task_id for t in tasks} - {t.task_id for t in drawn}
    assert not any(tid in payload for tid in list(unasked)[:50])


def test_a_round_payload_does_not_depend_on_draw_order(tmp_path):
    """The file is a function of which tasks were asked, not the order the
    generator happened to return them in."""
    from epago.taskgen.sealed_pool import round_payload, select

    tasks, digest, _, manifest_digest = _sealed_pool(tmp_path, n=1000)
    drawn = select(tasks, 3, 200)
    kwargs = dict(
        round_no=1,
        task_ids_digest=task_ids_digest(drawn),
        pool_digest_value=digest,
        manifest_digest=manifest_digest,
    )
    assert round_payload(drawn, **kwargs) == round_payload(list(reversed(drawn)), **kwargs)
