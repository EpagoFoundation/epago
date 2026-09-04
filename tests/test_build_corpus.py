"""Tests for scripts/build_corpus.py: filtering, dedup, determinism, pin lines."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_corpus.py"

spec = importlib.util.spec_from_file_location("build_corpus_script", _SCRIPT)
bc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bc  # dataclass processing requires a registered module
spec.loader.exec_module(bc)

MIN_CHARS = 40


def _long_text(topic: str, mutate_at: int | None = None) -> str:
    """~180 words on one line; optionally mutate one middle word (near-dup)."""
    words = []
    for i in range(20):
        words.extend(["sentence", str(i), "about", "the", topic, "archive", "and", "its", "keepers"])
    if mutate_at is not None:
        words[mutate_at] = "mutated"
    return " ".join(words)


def _jsonl_line(url: str, title: str, text: str, category: str = "") -> str:
    import json

    return json.dumps({"url": url, "title": title, "text": text, "category": category})


@pytest.fixture()
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    """One JSONL file and one text dir with knowable accept/reject outcomes."""
    jsonl = tmp_path / "docs.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                _jsonl_line("https://x/a", "Harbor archive", _long_text("harbor"), "history"),
                _jsonl_line("https://x/b", "Mill archive", _long_text("mill")),
                # exact duplicate of doc A's text (title/url differ; text decides)
                _jsonl_line("https://x/a2", "Harbor again", _long_text("harbor")),
                # near duplicate of doc B: one word changed mid-text
                _jsonl_line("https://x/b2", "Mill archive copy", _long_text("mill", mutate_at=90)),
                # junk: long enough but almost no alphabetic content
                _jsonl_line("https://x/junk", "Numbers", "12345 67890 " * 10),
                # too short
                _jsonl_line("https://x/tiny", "Tiny", "tiny"),
                "not valid json {{{",
            ]
        )
        + "\n"
    )

    text_dir = tmp_path / "notes"
    text_dir.mkdir()
    (text_dir / "lighthouse.txt").write_text(_long_text("lighthouse"))
    (text_dir / "quarry.md").write_text("# Quarry survey\n\n" + _long_text("quarry"))
    (text_dir / "ignored.rst").write_text(_long_text("ignored"))  # wrong suffix: skipped
    return jsonl, text_dir


def _build(inputs: tuple[Path, Path], out: Path, **kwargs):
    jsonl, text_dir = inputs
    return bc.build_corpus([jsonl], [text_dir], out, min_chars=MIN_CHARS, **kwargs)


def test_counts_filters_and_dedup(inputs, tmp_path: Path) -> None:
    report = _build(inputs, tmp_path / "corpus.db")
    # doc A, doc B, lighthouse.txt, quarry.md
    assert report.accepted == 4
    assert report.rejected["exact_duplicate"] == 1
    assert report.rejected["near_duplicate"] == 1
    assert report.rejected["low_alpha"] == 1
    assert report.rejected["too_short"] == 1
    assert report.rejected["invalid_json"] == 1
    assert report.rejected_total == 5

    from epago.environment.corpus import SqliteCorpus

    with SqliteCorpus(tmp_path / "corpus.db") as store:
        assert store.doc_count() == 4
        ids = list(store.iter_doc_ids())
        assert ids == sorted(ids)
        assert all(i.startswith("ep-") for i in ids)
        # the near-duplicate collapsed to one mill doc, searchable
        hits = store.search("mill archive keepers")
        assert len([h for h in hits if "Mill" in h.title]) == 1
        # markdown title came from the heading line
        assert any("Quarry survey" in store.get(i).title for i in ids)


def test_determinism_two_builds_identical_digest(inputs, tmp_path: Path) -> None:
    r1 = _build(inputs, tmp_path / "one.db")
    r2 = _build(inputs, tmp_path / "two.db")
    assert r1.digest == r2.digest
    assert r1.digest.startswith("sha256:")

    from epago.environment.sync import verify_corpus

    verify_corpus(tmp_path / "two.db", r1.digest)  # must not raise


def test_limit_caps_accepted_docs(inputs, tmp_path: Path) -> None:
    report = _build(inputs, tmp_path / "limited.db", limit=2)
    assert report.accepted == 2


def test_stable_content_derived_ids(inputs, tmp_path: Path) -> None:
    text = _long_text("harbor")
    assert bc.content_doc_id("Harbor archive", text) == bc.content_doc_id("Harbor archive", text)
    assert bc.content_doc_id("Harbor archive", text) != bc.content_doc_id("Other", text)


def test_cli_prints_digest_and_pin_lines(inputs, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    jsonl, text_dir = inputs
    out = tmp_path / "cli.db"
    result = CliRunner().invoke(
        bc.app,
        [
            "--jsonl", str(jsonl),
            "--text-dir", str(text_dir),
            "--out", str(out),
            "--min-chars", str(MIN_CHARS),
        ],
    )
    assert result.exit_code == 0, result.output

    from epago.environment.sync import corpus_digest

    digest = corpus_digest(out)
    assert digest in result.output
    assert "[eval]" in result.output
    assert f'corpus_digest   = "{digest}"' in result.output
    assert 'taskgen_release = "R1"' in result.output


def test_cli_refuses_existing_output(inputs, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    jsonl, _ = inputs
    out = tmp_path / "exists.db"
    out.write_text("not a corpus")
    result = CliRunner().invoke(bc.app, ["--jsonl", str(jsonl), "--out", str(out)])
    assert result.exit_code == 2
