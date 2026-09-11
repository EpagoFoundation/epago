"""scripts/build_practice_corpus.py: the practice library miners train on.

It must share no paper with the exam or the private holdout -- by doc id, by
identifier, by title or by abstract -- and it must build the same bytes every
time, so its published digests mean something."""
from __future__ import annotations

import importlib.util
import json
import random
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_practice_corpus.py"
spec = importlib.util.spec_from_file_location("build_practice_corpus_script", _SCRIPT)
bp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bp  # its dataclasses look their module up here
spec.loader.exec_module(bp)

_WORDS = [
    "cohort", "sample", "outcome", "patient", "signal", "protein", "membrane", "lattice",
    "solvent", "catalyst", "enzyme", "receptor", "genome", "neuron", "cortex", "tissue",
    "sediment", "aquifer", "basin", "climate", "rainfall", "drought", "crop", "yield",
    "soil", "nitrogen", "carbon", "polymer", "fiber", "alloy", "crystal", "photon",
    "laser", "spectrum", "quantum", "electron", "magnet", "vortex", "plasma", "orbit",
    "comet", "stellar", "galaxy", "survey", "model", "estimate", "variance", "regression",
    "trial", "dose", "therapy", "vaccine", "infection", "immune", "antibody", "marker",
    "biopsy", "tumor", "lesion", "imaging", "scan",
]


def _words(seed: int, n: int = 90) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def _paper(url: str, title: str, seed: int = 0, body: str | None = None) -> dict:
    body = body if body is not None else _words(seed)
    return {"url": url, "title": title, "text": f"{title}\n\n{body}.", "category": "crossref:test"}


def _corpus(tmp_path: Path, name: str, papers: list[dict]) -> Path:
    src = tmp_path / f"{name}.jsonl"
    src.write_text("".join(json.dumps(p) + "\n" for p in papers))
    out = tmp_path / name / "corpus.db"
    out.parent.mkdir()
    report = bp.build_corpus_script.build_corpus([src], [], out)
    assert report.accepted == len(papers)
    return out


def _titles(corpus: Path) -> list[str]:
    return sorted(r[0] for r in sqlite3.connect(corpus).execute("SELECT title FROM docs"))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A source holding two clean papers and one of every kind of overlap."""
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    shared = _paper("https://doi.org/10.1000/same", "Shared Paper On Aquifers", 3)
    exam = [
        shared,
        _paper("https://doi.org/10.1000/doi-twin", "Exam Title Four", 40),
        _paper("https://openalex.org/W555", "Soil Nitrogen, Crop Yield: A Survey", 50),
    ]
    source = [
        _paper("https://doi.org/10.1000/keep-a", "Kept Paper Alpha", 1),
        _paper("https://doi.org/10.1000/keep-b", "Kept Paper Beta", 2),
        shared,  # the same paper: same doc id
        _paper("https://doi.org/10.1000/DOI-twin", "A Different Title Entirely", 4),  # same DOI
        _paper("https://doi.org/10.1000/other", "soil nitrogen crop yield - a survey", 5),  # title
        _paper("https://pubmed.ncbi.nlm.nih.gov/12345/", "Holdout Paper By Id", 6),  # holdout id
    ]
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pq.write_table(
        pa.table({"url": ["https://pubmed.ncbi.nlm.nih.gov/12345"], "title": ["Other Title"]}),
        holdout / "shard-000.parquet",
    )
    return _corpus(tmp_path, "source", source), _corpus(tmp_path, "exam", exam), holdout


def test_the_library_keeps_no_paper_the_exam_or_holdout_holds(tmp_path):
    source, exam, holdout = _fixture(tmp_path)

    manifest = bp.build([source], [exam], [holdout], tmp_path / "practice")

    assert _titles(tmp_path / "practice" / "corpus.db") == ["Kept Paper Alpha", "Kept Paper Beta"]
    assert manifest["papers"] == 2
    assert manifest["removed"] == {"doc_id": 1, "identifier": 2, "title": 1}
    assert manifest["overlap_with_excluded"] == 0
    assert (tmp_path / "practice" / "entities-v1.json").is_file()
    assert not (tmp_path / "practice" / "papers.jsonl").exists()  # staging is not shipped


def test_one_paper_under_two_records_is_tied_by_its_abstract(tmp_path):
    """A DOI in one index, a PubMed id in the other, and a different title: only
    the abstract ties the two records together. A shared ending counts only
    when one paper on each side carries it; more, and it is boilerplate."""
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    whole, ending, boilerplate, held = _words(20), _words(21), _words(22), _words(28)
    exam = [
        _paper("https://doi.org/10.1000/e1", "Exam One", body=whole),
        _paper("https://doi.org/10.1000/e2", "Exam Two", body=f"{_words(23)} {ending}"),
        _paper("https://doi.org/10.1000/e3", "Exam Three", body=f"{_words(24)} {boilerplate}"),
    ]
    source = [
        _paper("https://pubmed.ncbi.nlm.nih.gov/1", "A Retitled Record", body=whole),
        _paper("https://pubmed.ncbi.nlm.nih.gov/2", "A Shortened Record", body=f"{_words(25)} {ending}"),
        _paper("https://pubmed.ncbi.nlm.nih.gov/3", "A Holdout Record", body=held),
        _paper("https://doi.org/10.1000/k1", "Kept One", body=f"{_words(26)} {boilerplate}"),
        _paper("https://doi.org/10.1000/k2", "Kept Two", body=f"{_words(27)} {boilerplate}"),
    ]
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pq.write_table(
        pa.table({
            "url": ["https://openalex.org/W1"],
            "title": ["Unrelated Title"],
            "text": [f"Unrelated Title\n\n{held}."],
        }),
        holdout / "shard-000.parquet",
    )

    manifest = bp.build(
        [_corpus(tmp_path, "source", source)],
        [_corpus(tmp_path, "exam", exam)],
        [holdout],
        tmp_path / "practice",
    )

    assert _titles(tmp_path / "practice" / "corpus.db") == ["Kept One", "Kept Two"]
    assert manifest["removed"] == {"abstract": 2, "abstract_end": 1}


def test_an_exam_papers_own_second_record_does_not_hide_a_third(tmp_path):
    """Measured on the real build: the exam paper's own record in the source,
    under another link, shared its ending with a third record, so the ending
    looked like boilerplate and the third stayed. Records the other keys
    already remove no longer count toward that."""
    ending = _words(31)
    exam = [
        _paper("https://doi.org/10.1000/e9", "Food Weight Estimation: Phase 2", body=f"{_words(32)} {ending}"),
    ]
    source = [
        # The exam paper itself, listed by PubMed with a looser title: removed by title.
        _paper("https://pubmed.ncbi.nlm.nih.gov/9", "Food weight estimation - phase 2", body=f"{_words(33)} {ending}"),
        # A third record that shares only the ending: removed by the ending.
        _paper("https://doi.org/10.1000/p1", "Food Weight Estimation", body=f"{_words(34)} {ending}"),
        _paper("https://doi.org/10.1000/k1", "Kept One", body=_words(35)),
    ]

    manifest = bp.build(
        [_corpus(tmp_path, "source", source)], [_corpus(tmp_path, "exam", exam)], [], tmp_path / "practice"
    )

    assert _titles(tmp_path / "practice" / "corpus.db") == ["Kept One"]
    assert manifest["removed"] == {"abstract_end": 1, "title": 1}


def test_the_same_inputs_build_the_same_bytes(tmp_path):
    """A published digest is only worth something if anyone can rebuild it."""
    source, exam, holdout = _fixture(tmp_path)

    first = bp.build([source], [exam], [holdout], tmp_path / "a")
    second = bp.build([source], [exam], [holdout], tmp_path / "b")

    assert first == second
    assert first["corpus_digest"].startswith("sha256:")


def test_a_surviving_exam_paper_stops_the_build(tmp_path):
    source, exam, _ = _fixture(tmp_path)
    exclusions = bp.Exclusions()
    exclusions.add_corpus(exam)

    with pytest.raises(bp.PracticeCorpusError, match="survived"):
        bp.check_disjoint(source, exclusions)  # the unfiltered source still holds exam papers


def test_a_build_needs_something_to_exclude(tmp_path):
    source, _, _ = _fixture(tmp_path)
    with pytest.raises(bp.PracticeCorpusError, match="nothing to exclude"):
        bp.build([source], [], [], tmp_path / "practice")


def test_an_earlier_build_is_not_overwritten_by_accident(tmp_path):
    source, exam, holdout = _fixture(tmp_path)
    bp.build([source], [exam], [holdout], tmp_path / "practice")
    with pytest.raises(bp.PracticeCorpusError, match="not empty"):
        bp.build([source], [exam], [holdout], tmp_path / "practice")
    assert bp.build([source], [exam], [holdout], tmp_path / "practice", force=True)["papers"] == 2


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://doi.org/10.1111/EJN.70363", "doi:10.1111/ejn.70363"),
        ("https://openalex.org/W4409721707", "openalex:w4409721707"),
        ("https://pubmed.ncbi.nlm.nih.gov/24317972", "pmid:24317972"),
        ("https://europepmc.org/abstract/MED/24317972", "pmid:24317972"),
        ("https://example.org/paper", None),
    ],
)
def test_identifiers_are_read_from_every_url_form(url, key):
    assert bp.id_key(url) == key


def test_titles_match_across_case_and_punctuation():
    assert bp.title_key("Soil Nitrogen, Crop Yield: A Survey") == bp.title_key(
        "soil nitrogen crop yield - a survey"
    )


@pytest.mark.parametrize(
    ("one", "other"),
    [
        (
            "Distinctions between i Chlorella ohadii /i and i Chlorella sorokiniana /i",
            "Distinctions between Chlorella ohadii and Chlorella sorokiniana",
        ),
        ("b fish- i in /i BOLIVIA: an updated checklist", "fish-inBOLIVIA: an updated checklist"),
        ("Lipid Modulation by <i>Citrus bergamia</i>", "Lipid Modulation by Citrus bergamia"),
        ("Phenolamide-driven &#x3b1;-glucosidase inhibition", "Phenolamide-driven α-glucosidase inhibition"),
        ("Five‐Year Mortality in Patients Aged 75&#x2009;Years", "Five-Year Mortality in Patients Aged 75 Years"),
        ("Longitudinal 1H NMR Metabolomics", "Longitudinal 1 H NMR Metabolomics"),
    ],
)
def test_one_title_under_two_indexes_folds_to_one_key(one, other):
    """Seen in the real overlap: the same paper, titled by two indexes."""
    assert bp.title_key(one) == bp.title_key(other)
