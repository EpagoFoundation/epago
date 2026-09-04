"""Tests for scripts/rotate_holdout.py: the weekly private-holdout rotation.

The rotation is the only automated path that can change what every validator
scores against, so what is tested here is mostly *refusals*: no token, a starved
week, a period that already went out, and a hub that cannot confirm either way.
Plus the two things a scheduler depends on — the JSON report's shape and the
dry-run/apply split.

No network: every fetcher, the publish call and the hub lookup are injected, the
same way ``tests/test_harvest_holdout.py`` does it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass processing requires a registered module
    spec.loader.exec_module(module)
    return module


hh = _load("harvest_holdout_script_rot", "harvest_holdout.py")
rot = _load("rotate_holdout_script", "rotate_holdout.py")

CONTRACT = """# a chain contract
[eval]
corpus_repo = "Org/corpus"

# Private holdout feed: dated, fresh, auditable.
[private_source]
repo        = "Org/epago-holdout-science-2026w33"
revision    = "0000000000000000000000000000000000000000"
text_column = "text"
max_shards  = 4

[quorum]
theta = 0.51
"""


# --- harness ------------------------------------------------------------------


class _Recorder:
    """What the run tried to do, without doing any of it."""

    def __init__(self) -> None:
        self.harvested = 0
        self.published: list[tuple[str, int]] = []


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Replace every side effect: harvest, shard writing, publish, hub lookup."""
    rec = _Recorder()

    def fake_run_plan(plan, target, mailto, stats, *, kept=None):
        rec.harvested += 1
        stats.kept = target if kept is None else kept
        return [{"text": f"t{i}", "title": f"ti{i}", "url": "", "category": "c"}
                for i in range(stats.kept)]

    def fake_write_shards(papers, out_dir, n_shards):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(min(n_shards, max(1, len(papers)))):
            path = out_dir / f"shard-{i:03d}.parquet"
            path.write_bytes(b"parquet")
            paths.append(path)
        return paths

    def fake_publish(repo, shard_paths, token, api=None):
        rec.published.append((repo, len(shard_paths)))
        return "cafe1234cafe1234cafe1234cafe1234cafe1234"

    monkeypatch.setattr(rot, "run_plan", fake_run_plan)
    monkeypatch.setattr(rot, "write_shards", fake_write_shards)
    monkeypatch.setattr(rot, "publish", fake_publish)
    monkeypatch.setattr(rot, "read_token", lambda: "hf_test_token")
    monkeypatch.setattr(rot, "hf_repo_exists", lambda repo, token: False)
    return rec


def _run(argv: list[str], tmp_path: Path, capsys) -> tuple[int, dict]:
    code = rot.main(["--out", str(tmp_path / "holdout"), "--to-date", "2026-08-18",
                     "--target", "1200", *argv])
    return code, json.loads(capsys.readouterr().out)


def _contract(tmp_path: Path) -> Path:
    path = tmp_path / "EPAGO-TEST.toml"
    path.write_text(CONTRACT)
    return path


# --- the refusals -------------------------------------------------------------


def test_a_missing_token_fails_before_the_harvest_runs(stub, monkeypatch, tmp_path, capsys):
    """A multi-hour fetch that ends in "no token" has burned the window."""
    monkeypatch.setattr(rot, "read_token", lambda: None)
    code, report = _run(["--apply"], tmp_path, capsys)

    assert code == rot.EXIT_CONFIG
    assert report["status"] == "error"
    assert "token" in report["reason"].lower()
    assert "HUGGINGFACE_TOKEN" in report["reason"]  # actionable, names the variable
    assert stub.harvested == 0 and stub.published == []


def test_a_dry_run_needs_no_token(stub, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rot, "read_token", lambda: None)
    code, report = _run([], tmp_path, capsys)
    assert code == rot.EXIT_OK and report["status"] == "dry-run"


def test_a_starved_week_is_refused_not_published(stub, monkeypatch, tmp_path, capsys):
    """One throttled harvest must not become the live feed."""
    monkeypatch.setattr(rot, "run_plan", lambda p, t, m, s: (
        setattr(s, "kept", 12) or [{"text": "t", "title": "x", "url": "", "category": "c"}] * 12
    ))
    contract = _contract(tmp_path)
    code, report = _run(["--apply", "--min-papers", "800", "--contract", str(contract)],
                        tmp_path, capsys)

    assert code == rot.EXIT_STARVED
    assert report["status"] == "starved"
    assert report["kept"] == 12 and report["min_papers"] == 800
    assert "refusing to publish" in report["reason"]
    assert stub.published == []
    assert contract.read_text() == CONTRACT  # the live feed is untouched
    # what did come back is kept on disk with its plan, for a human to look at
    assert json.loads(Path(report["manifest"]).read_text())["kept"] == 12


def test_the_minimum_is_a_floor_not_a_warning(stub, tmp_path, capsys):
    code, report = _run(["--apply", "--min-papers", "1199"], tmp_path, capsys)
    assert code == rot.EXIT_OK and report["status"] == "published"
    code, report = _run(["--apply", "--min-papers", "1201", "--force"], tmp_path, capsys)
    assert code == rot.EXIT_STARVED


@pytest.mark.parametrize("already", ["contract", "ledger", "hub"])
def test_an_already_published_period_is_skipped(stub, monkeypatch, tmp_path, capsys, already):
    """Re-running the same week must never publish a second slice over a live feed."""
    repo = rot.default_repo(hh.ALL_DOMAINS, __import__("datetime").date(2026, 8, 18))
    argv = []
    if already == "contract":
        contract = tmp_path / "pinned.toml"
        contract.write_text(CONTRACT.replace("Org/epago-holdout-science-2026w33", repo))
        argv = ["--contract", str(contract)]
    elif already == "ledger":
        ledger = tmp_path / "holdout" / "rotations.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({"period": "2026w34", "repo": repo}) + "\n")
    else:
        monkeypatch.setattr(rot, "hf_repo_exists", lambda r, t: True)

    code, report = _run(["--apply", *argv], tmp_path, capsys)

    assert code == rot.EXIT_OK          # a no-op, not a failure: the loop keeps looping
    assert report["status"] == "skipped"
    assert report["repo"] == repo and repo in report["reason"]
    assert stub.published == [] and stub.harvested == 0


def test_force_overrides_the_duplicate_check(stub, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rot, "hf_repo_exists", lambda r, t: True)
    code, report = _run(["--apply", "--force"], tmp_path, capsys)
    assert code == rot.EXIT_OK and report["status"] == "published"
    assert stub.published


def test_an_unverifiable_hub_refuses_to_publish_blind(stub, monkeypatch, tmp_path, capsys):
    """"Could not check" is not "does not exist" — publishing then clobbers a live feed."""
    monkeypatch.setattr(rot, "hf_repo_exists", lambda r, t: None)
    code, report = _run(["--apply"], tmp_path, capsys)

    assert code == rot.EXIT_CONFIG
    assert "could not reach HuggingFace" in report["reason"]
    assert stub.published == []
    # ...but a dry run is still free to rehearse offline.
    code, report = _run([], tmp_path, capsys)
    assert code == rot.EXIT_OK and report["status"] == "dry-run"


def test_a_failed_publish_is_reported_not_swallowed(stub, monkeypatch, tmp_path, capsys):
    def boom(*a, **k):
        raise RuntimeError("refusing to publish: dataset repo Org/x is PUBLIC")

    monkeypatch.setattr(rot, "publish", boom)
    contract = _contract(tmp_path)
    code, report = _run(["--apply", "--contract", str(contract)], tmp_path, capsys)

    assert code == rot.EXIT_PUBLISH
    assert "is PUBLIC" in report["reason"]
    assert contract.read_text() == CONTRACT  # nothing published => nothing pinned


# --- the machine-readable report ---------------------------------------------


def test_the_json_report_carries_everything_the_next_step_needs(stub, tmp_path, capsys):
    out = tmp_path / "report.json"
    code, report = _run(["--apply", "--json-out", str(out)], tmp_path, capsys)

    assert code == rot.EXIT_OK
    required = {
        "status", "applied", "period", "repo", "revision", "private", "private_source",
        "window", "domains", "vocab", "seed", "kept", "shards", "min_papers",
        "out_dir", "manifest", "block_file", "block", "ledger", "generated_at",
    }
    assert required <= set(report)
    assert report["status"] == "published" and report["applied"] is True
    assert report["private"] is True
    assert report["period"] == "2026w34"
    assert report["repo"].endswith("epago-holdout-science-2026w34")
    assert report["revision"] == "cafe1234cafe1234cafe1234cafe1234cafe1234"
    assert report["private_source"] == {
        "repo": report["repo"], "revision": report["revision"],
        "text_column": "text", "max_shards": 4,
    }
    assert report["window"] == {"from": "2026-08-11", "to": "2026-08-18"}
    assert report["domains"] == [1, 2, 3, 4] and report["vocab"] == "general"
    assert isinstance(report["seed"], int)          # determinism: the seed is recorded
    assert json.loads(Path(report["manifest"]).read_text())["seed"] == report["seed"]
    assert json.loads(out.read_text()) == report    # the file matches stdout exactly
    assert report["block"] == Path(report["block_file"]).read_text()
    assert f'repo        = "{report["repo"]}"' in report["block"]


def test_stdout_is_json_alone_so_a_scheduler_can_parse_it(stub, tmp_path, capsys):
    rot.main(["--out", str(tmp_path / "h"), "--to-date", "2026-08-18", "--target", "900"])
    captured = capsys.readouterr()
    json.loads(captured.out)                        # parses with no prose mixed in
    assert "rotation 2026w34" in captured.err       # progress went to stderr


def test_the_ledger_records_every_applied_rotation(stub, tmp_path, capsys):
    _run(["--apply"], tmp_path, capsys)
    _run(["--apply", "--to-date", "2026-08-25"], tmp_path, capsys)
    lines = [json.loads(ln) for ln in
             (tmp_path / "holdout" / "rotations.jsonl").read_text().splitlines()]
    assert [entry["period"] for entry in lines] == ["2026w34", "2026w35"]
    assert all(entry["revision"] and entry["seed"] and entry["kept"] for entry in lines)


# --- dry-run vs apply ---------------------------------------------------------


def test_dry_run_is_the_default_and_changes_nothing_live(stub, tmp_path, capsys):
    contract = _contract(tmp_path)
    code, report = _run(["--contract", str(contract)], tmp_path, capsys)

    assert code == rot.EXIT_OK
    assert report["status"] == "dry-run" and report["applied"] is False
    assert report["revision"] is None
    assert stub.published == []                      # nothing uploaded
    assert contract.read_text() == CONTRACT          # nothing pinned
    assert not (tmp_path / "holdout" / "rotations.jsonl").exists()
    assert report["contract"]["updated"] is False
    assert "<commit-after-publish>" in report["block"]
    # the harvest itself did run, so the dry run really rehearses the path
    assert stub.harvested == 1 and report["shards"] > 0


def test_apply_publishes_and_pins_keeping_the_old_value(stub, tmp_path, capsys):
    contract = _contract(tmp_path)
    code, report = _run(["--apply", "--contract", str(contract)], tmp_path, capsys)
    text = contract.read_text()

    assert code == rot.EXIT_OK and report["status"] == "published"
    assert stub.published == [(report["repo"], report["shards"])]
    assert f'repo        = "{report["repo"]}"' in text
    assert f'revision    = "{report["revision"]}"' in text
    assert "Org/epago-holdout-science-2026w33" in text      # old pin kept in a comment
    assert "# rotated 2026-" in text
    entry = report["contract"]
    assert entry["updated"] is True
    assert entry["previous"] == {"repo": "Org/epago-holdout-science-2026w33",
                                 "revision": "0" * 40}
    assert Path(entry["backup"]).read_text() == CONTRACT    # the old file survives whole


@pytest.mark.parametrize("contract_name,body", [
    ("no-section.toml", '[eval]\ncorpus_repo = "Org/c"\n'),
    ("missing.toml", None),
])
def test_an_unusable_contract_fails_before_anything_is_published(
    stub, tmp_path, capsys, contract_name, body,
):
    """A rewrite that cannot work must fail now, not after a slice is live."""
    contract = tmp_path / contract_name
    if body is not None:
        contract.write_text(body)
    code, report = _run(["--apply", "--contract", str(contract)], tmp_path, capsys)

    assert code == rot.EXIT_CONFIG
    assert ("[private_source]" in report["reason"]) or ("not found" in report["reason"])
    assert stub.harvested == 0 and stub.published == []


# --- the contract rewrite, as a pure function ---------------------------------


def test_update_private_source_touches_only_the_pin():
    new, previous = rot.update_private_source(CONTRACT, "Org/new-2026w34", "abc123", "2026-08-18")

    assert previous == {"repo": "Org/epago-holdout-science-2026w33", "revision": "0" * 40}
    assert 'repo        = "Org/new-2026w34"' in new
    assert 'revision    = "abc123"' in new
    assert 'text_column = "text"' in new and "max_shards  = 4" in new
    assert "[eval]" in new and 'corpus_repo = "Org/corpus"' in new  # other sections intact
    assert "[quorum]\ntheta = 0.51" in new
    assert "# Private holdout feed: dated, fresh, auditable." in new  # comments intact
    assert "Org/epago-holdout-science-2026w33" in new                # outgoing pin recorded


def test_repeated_rotations_keep_exactly_one_rotation_note():
    once, _ = rot.update_private_source(CONTRACT, "Org/w34", "aaa", "2026-08-18")
    twice, previous = rot.update_private_source(once, "Org/w35", "bbb", "2026-08-25")

    import tomllib

    assert twice.count(rot._ROTATION_COMMENT) == 1
    assert tomllib.loads(twice)["private_source"] == {
        "repo": "Org/w35", "revision": "bbb", "text_column": "text", "max_shards": 4,
    }
    assert previous == {"repo": "Org/w34", "revision": "aaa"}
    assert "Org/w34" in twice and 'repo        = "Org/w35"' in twice
    assert "Org/w33" not in twice.split("[private_source]")[1].split("[quorum]")[0][:400]


def test_update_private_source_refuses_a_contract_without_the_section():
    with pytest.raises(ValueError, match=r"\[private_source\]"):
        rot.update_private_source("[eval]\nx = 1\n", "Org/r", "rev", "2026-08-18")


def test_the_rendered_block_states_the_privacy_rule():
    block = rot.render_block("Org/r", "rev", "text", 4)
    assert block.startswith("[private_source]")
    assert 'repo        = "Org/r"' in block and 'revision    = "rev"' in block
    assert "PRIVATE while live" in block and "rotated out" in block


def test_an_unpublished_block_cannot_be_mistaken_for_a_pin():
    assert '"<commit-after-publish>"' in rot.render_block("Org/r", None, "text", 4)


# --- period naming ------------------------------------------------------------


def test_the_repo_name_is_derived_from_the_period_and_scope():
    import datetime as dt

    assert rot.period_tag(dt.date(2026, 8, 18)) == "2026w34"
    assert rot.default_repo(hh.ALL_DOMAINS, dt.date(2026, 8, 18)) == (
        "EpagoFoundation/epago-holdout-science-2026w34"
    )
    assert rot.default_repo((4,), dt.date(2026, 1, 5), "Org") == "Org/epago-holdout-health-2026w02"


def test_a_hub_that_cannot_answer_is_unknown_not_absent():
    assert rot.hf_repo_exists("Org/r", token=None) is None       # no token: cannot know

    reason, hub_ok = rot.already_rotated(
        "Org/r", None, Path("/nonexistent/ledger.jsonl"), exists_fn=lambda r, t: None
    )
    assert reason is None and hub_ok is False
    reason, hub_ok = rot.already_rotated(
        "Org/r", None, Path("/nonexistent/ledger.jsonl"), exists_fn=lambda r, t: False
    )
    assert reason is None and hub_ok is True
