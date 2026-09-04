"""Tests for scripts/harvest_holdout.py: domain selection and honest labelling.

The harvest is the only place the subnet's *scope* is decided, so the rules that
matter here are: a slice covers every domain it was asked for, and it is never
labelled broader than the index that produced it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "harvest_holdout.py"

spec = importlib.util.spec_from_file_location("harvest_holdout_script", _SCRIPT)
hh = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hh  # dataclass processing requires a registered module
spec.loader.exec_module(hh)


# --- domain selection ---------------------------------------------------------


def test_parse_domains_accepts_every_documented_form():
    assert hh.parse_domains([]) == hh.ALL_DOMAINS  # default: all of science
    assert hh.parse_domains(["all"]) == hh.ALL_DOMAINS
    assert hh.parse_domains(["1", "3"]) == (1, 3)
    assert hh.parse_domains(["1,3"]) == (1, 3)
    assert hh.parse_domains(["4", "4"]) == (4,)


def test_parse_domains_rejects_unknown_ids():
    with pytest.raises(SystemExit, match="unknown"):
        hh.parse_domains(["9"])
    with pytest.raises(SystemExit, match="not a domain id"):
        hh.parse_domains(["physics"])


def test_domain_ids_match_the_openalex_taxonomy():
    """Verified against https://api.openalex.org/domains — 1 is Life and 3 is
    Physical, which is not the order the numbering suggests."""
    assert {d: name for d, (name, _) in hh.DOMAINS.items()} == {
        1: "Life Sciences",
        2: "Social Sciences",
        3: "Physical Sciences",
        4: "Health Sciences",
    }
    assert hh.BIOMEDICAL_DOMAINS == (1, 4)


def test_repo_slug_never_calls_a_multi_domain_slice_medicine():
    assert hh.domain_slug(hh.ALL_DOMAINS) == "science"
    assert hh.domain_slug((4,)) == "health"
    assert hh.domain_slug((2, 3)) == "social-physical"


# --- the seeded plan ----------------------------------------------------------


def test_themes_cover_every_selected_domain():
    """A four-domain week must not come out as four oncology themes."""
    for seed in range(20):
        plan = hh.plan_harvest(seed, "2026-01-01", "2026-01-08", hh.ALL_DOMAINS)
        covered = {plan["theme_domains"][t] for t in plan["themes"]}
        assert covered == set(hh.ALL_DOMAINS)
        assert all(t in hh.THEMES for t in plan["themes"])


def test_medical_themes_are_still_in_the_pool():
    """Health is part of "all science", not a thing the general pool replaced."""
    assert {"oncology", "cardiology", "epidemiology"} <= set(hh.THEMES_BY_DOMAIN[4])
    assert {"machine learning", "physics", "catalysis"} <= set(hh.THEMES_BY_DOMAIN[3])
    assert set(hh.THEMES_BY_DOMAIN[4]) <= set(hh.THEMES)


def test_plan_is_reproducible_from_its_seed():
    a = hh.plan_harvest(123456789, "2026-01-01", "2026-01-08", hh.ALL_DOMAINS)
    b = hh.plan_harvest(123456789, "2026-01-01", "2026-01-08", hh.ALL_DOMAINS)
    assert a == b
    assert a != hh.plan_harvest(987654321, "2026-01-01", "2026-01-08", hh.ALL_DOMAINS)


def test_biomedical_only_sources_are_dropped_when_no_biomedical_domain():
    """Europe PMC and PubMed index biomedicine; a physics slice drawn from them
    would be mislabelled, so they are not offered at all."""
    for seed in range(20):
        plan = hh.plan_harvest(seed, "2026-01-01", "2026-01-08", (2, 3))
        assert "europepmc" not in plan["sources"]
        assert "pubmed" not in plan["sources"]
        assert plan["sources"], "a plan must still have a source"
        notes = " ".join(plan["scope_notes"])
        assert "europepmc: skipped" in notes and "pubmed: skipped" in notes


def test_biomedical_sources_are_recorded_as_restricted_when_they_are_used():
    plan = next(
        p for p in (
            hh.plan_harvest(s, "2026-01-01", "2026-01-08", hh.ALL_DOMAINS)
            for s in range(20)
        ) if "europepmc" in p["sources"]
    )
    assert plan["source_scope"]["europepmc"] == "biomedical"
    assert any("restricted to the" in n for n in plan["scope_notes"])


# --- execution ----------------------------------------------------------------


def _record_calls(monkeypatch) -> list[tuple[str, str | None]]:
    """Replace every network fetcher with a recorder; returns (source, theme)."""
    calls: list[tuple[str, str | None]] = []

    def fake(name, theme_pos):
        def _f(*args, **kwargs):
            calls.append((name, args[theme_pos]))
            return {}
        return _f

    monkeypatch.setattr(hh, "openalex_fetch", fake("openalex", 2))
    monkeypatch.setattr(hh, "europepmc_fetch", fake("europepmc", 2))
    monkeypatch.setattr(hh, "pubmed_fetch", fake("pubmed", 2))
    monkeypatch.setattr(hh, "crossref_fetch", fake("crossref", 2))
    return calls


def test_a_biomedical_index_never_runs_a_non_biomedical_theme(monkeypatch):
    calls = _record_calls(monkeypatch)
    plan = next(
        p for p in (
            hh.plan_harvest(s, "2026-01-01", "2026-01-08", hh.ALL_DOMAINS)
            for s in range(20)
        ) if {"europepmc", "pubmed"} & set(p["sources"])
    )
    hh.run_plan(plan, 400, "test@example.com", hh.Stats())

    for source, theme in calls:
        if hh.SOURCE_SCOPE[source] != "biomedical" or theme is None:
            continue
        assert plan["theme_domains"][theme] in hh.BIOMEDICAL_DOMAINS, (
            f"{source} was sent the non-biomedical theme {theme!r}"
        )
    # ...and the broad top-up only ever falls back to a full-coverage index.
    assert all(hh.SOURCE_SCOPE[s] == "all" for s, theme in calls if theme is None)


def test_openalex_themed_fetch_is_pinned_to_the_themes_own_domain(monkeypatch):
    seen: list[tuple[str | None, tuple[int, ...]]] = []

    def fake_openalex(fr, to, theme, want, mailto, stats, domains=hh.ALL_DOMAINS):
        seen.append((theme, tuple(domains)))
        return {}

    monkeypatch.setattr(hh, "openalex_fetch", fake_openalex)
    for name in ("europepmc_fetch", "pubmed_fetch", "crossref_fetch"):
        monkeypatch.setattr(hh, name, lambda *a, **k: {})

    plan = next(
        p for p in (
            hh.plan_harvest(s, "2026-01-01", "2026-01-08", hh.ALL_DOMAINS)
            for s in range(20)
        ) if "openalex" in p["sources"]
    )
    hh.run_plan(plan, 400, "test@example.com", hh.Stats())

    themed = [(t, d) for t, d in seen if t is not None]
    assert themed
    for theme, domains in themed:
        assert domains == (plan["theme_domains"][theme],)


# --- publishing: the private-repo rule ---------------------------------------


class _FakeApi:
    """Enough HfApi for :func:`publish`; records what it was asked to do."""

    def __init__(self, private: bool = True) -> None:
        self._private = private
        self.created: list[dict] = []
        self.uploaded: list[str] = []

    def create_repo(self, **kw):
        self.created.append(kw)

    def repo_info(self, repo_id, repo_type):
        return SimpleNamespace(private=self._private)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
        self.uploaded.append(path_in_repo)

    def list_repo_commits(self, repo, repo_type):
        return [SimpleNamespace(commit_id="deadbeef")]


def test_publish_creates_a_private_repo_and_uploads_only_the_shards(tmp_path):
    shards = [tmp_path / "shard-000.parquet", tmp_path / "shard-001.parquet"]
    for path in shards:
        path.write_bytes(b"parquet")
    api = _FakeApi(private=True)

    assert hh.publish("Org/holdout-2026w34", shards, "tok", api=api) == "deadbeef"
    assert api.created == [{"repo_id": "Org/holdout-2026w34", "repo_type": "dataset",
                            "private": True, "exist_ok": True}]
    assert api.uploaded == ["shard-000.parquet", "shard-001.parquet"]


def test_publish_refuses_a_repo_the_hub_reports_as_public(tmp_path):
    """``exist_ok=True`` accepts an existing repo — and an existing PUBLIC repo
    stays public, which would hand every miner the live holdout."""
    shard = tmp_path / "shard-000.parquet"
    shard.write_bytes(b"parquet")
    api = _FakeApi(private=False)

    with pytest.raises(RuntimeError, match="PUBLIC"):
        hh.publish("Org/oops", [shard], "tok", api=api)
    assert api.uploaded == []  # refused before a single byte went up


# --- the write token ----------------------------------------------------------


def test_the_token_comes_from_the_environment_first(monkeypatch):
    """Containers are configured by env alone; nothing is baked into an image."""
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_from_env")
    assert hh.read_token() == "hf_from_env"
    monkeypatch.delenv("HUGGINGFACE_TOKEN")
    monkeypatch.setenv("HF_TOKEN", "hf_alt")
    assert hh.read_token() == "hf_alt"


def test_publishing_without_a_token_fails_before_the_harvest(monkeypatch, tmp_path, capsys):
    """A multi-hour fetch that ends in "no token" has burned the window."""
    monkeypatch.setattr(hh, "read_token", lambda: None)
    monkeypatch.setattr(hh, "run_plan", lambda *a, **k: pytest.fail("harvest must not run"))
    monkeypatch.setattr(sys, "argv", ["harvest_holdout.py", "--publish",
                                      "--out", str(tmp_path / "h")])

    assert hh.main() == 2
    assert "HUGGINGFACE_TOKEN" in capsys.readouterr().err  # actionable, names the variable


# --- the quality gate ---------------------------------------------------------


def test_the_mintability_gate_follows_the_vocabulary():
    """A kept paper must be one the release's templates can mint from; under the
    medical words a photovoltaics result is not a result at all."""
    from epago.taskgen.templates import VOCABULARIES

    title = "Perovskite solar cell stack with an engineered interface layer"
    body = (
        "The certified power conversion efficiency of the device achieved 24.7% "
        "after 500 hours of continuous illumination. " + "Filler prose. " * 20
    )
    assert hh._keep(title, body, hh.Stats(vocab=VOCABULARIES["medical"])) is None
    kept = hh._keep(title, body, hh.Stats(vocab=VOCABULARIES["general"]))
    assert kept is not None and kept.startswith(title)
    assert hh.Stats().vocab is VOCABULARIES["general"]  # the harvester default
