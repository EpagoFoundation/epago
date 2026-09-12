#!/usr/bin/env python
"""Epago publisher CLI.

Ships a validator's public artifacts to its key namespace in the shared the object store
bucket::

    sync         one publish pass (new/changed files only)
    watch        run sync forever on an interval (systemd-friendly)

Each pass first rebuilds the dashboard into the state directory, so it ships
with the audit files (``--no-dashboard`` skips that).
    mirror-king  upload a mirrored king snapshot content-addressed to the object store

Authentication comes from the object store env config (``EPAGO_S3_BUCKET`` /
``EPAGO_S3_ACCESS_KEY`` / ``EPAGO_S3_SECRET_KEY``). Not registered under the main
``epago`` CLI; run with ``python -m epago.publishing.cli``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from epago.publishing.publisher import (
    PublishReport,
    StatePublisher,
    publish_king_mirror,
    update_mirror_manifest,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=__doc__,
    pretty_exceptions_show_locals=False,
)

_STATE_DIR_OPT = typer.Option(..., "--state-dir", help="validator state directory")
_REPO_ID_OPT = typer.Option(..., "--repo-id", help="key namespace in the bucket (namespace/name)")


def _echo_report(report: PublishReport) -> None:
    typer.echo(
        f"{report.repo_id}: {len(report.uploaded)} uploaded, "
        f"{len(report.skipped)} unchanged, {len(report.errors)} error(s)"
    )
    for path in report.uploaded:
        typer.echo(f"  + {path}")
    for path, err in report.errors:
        typer.echo(f"  ! {path}: {err}", err=True)


def _refresh_dashboard(state_dir: Path, chain_toml: Path | None) -> None:
    """Rebuild ``dashboard.json`` and the page into the state dir for this pass."""
    from epago.config import load_config
    from epago.dashboard.export import write_dashboard

    try:
        write_dashboard(state_dir, state_dir / "dashboard", load_config(chain_toml))
    except FileNotFoundError:
        typer.echo("dashboard: no validator state yet", err=True)
    except Exception as exc:  # noqa: BLE001 - a dashboard hiccup never stops publishing
        typer.echo(f"dashboard: {type(exc).__name__}: {exc}", err=True)


_DASHBOARD_OPT = typer.Option(
    True, "--dashboard/--no-dashboard", help="rebuild the dashboard before each pass"
)
_CHAIN_TOML_OPT = typer.Option(
    None, "--chain-toml", help="chain.toml for the dashboard (default: EPAGO_CHAIN_TOML)"
)


@app.command()
def sync(
    state_dir: Path = _STATE_DIR_OPT,
    repo_id: str = _REPO_ID_OPT,
    dashboard: bool = _DASHBOARD_OPT,
    chain_toml: Path | None = _CHAIN_TOML_OPT,
) -> None:
    """One publish pass: upload new/changed public artifacts. Exit 1 on any error."""
    if dashboard:
        _refresh_dashboard(state_dir, chain_toml)
    report = StatePublisher(state_dir, repo_id).sync()
    _echo_report(report)
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def watch(
    state_dir: Path = _STATE_DIR_OPT,
    repo_id: str = _REPO_ID_OPT,
    interval_s: float = typer.Option(300.0, "--interval-s", help="seconds between sync passes"),
    dashboard: bool = _DASHBOARD_OPT,
    chain_toml: Path | None = _CHAIN_TOML_OPT,
) -> None:
    """Sync in a loop. Errors are reported and retried next pass, never fatal."""
    publisher = StatePublisher(state_dir, repo_id)
    typer.echo(f"watching {state_dir} -> {repo_id} every {interval_s:g}s (ctrl-c to stop)")
    while True:
        if dashboard:
            _refresh_dashboard(state_dir, chain_toml)
        _echo_report(publisher.sync())
        time.sleep(interval_s)


@app.command("mirror-king")
def mirror_king(
    king_dir: Path = typer.Option(..., "--king-dir", help="materialized king snapshot folder"),
    digest: str = typer.Option(..., "--digest", help="the king's original committed digest"),
    repo_id: str = _REPO_ID_OPT,
    state_dir: Optional[Path] = typer.Option(
        None,
        "--state-dir",
        help="validator state dir; records the mirror in publications/mirrors.json",
    ),
) -> None:
    """Mirror a king snapshot content-addressed to the object store and record the pin."""
    try:
        mirror_ref = publish_king_mirror(king_dir.expanduser(), digest, repo_id)
    except Exception as exc:  # noqa: BLE001 - surface upload errors cleanly
        typer.echo(f"error: mirror upload failed: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"mirrored {digest} -> {mirror_ref.repo}@{mirror_ref.digest}")
    if state_dir is not None:
        path = update_mirror_manifest(state_dir, digest, mirror_ref)
        typer.echo(f"recorded in {path} (next sync publishes it)")


if __name__ == "__main__":
    app()
