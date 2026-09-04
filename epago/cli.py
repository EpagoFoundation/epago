"""Epago command-line interface.

All production wiring lives here; ``neurons/validator.py`` is a thin shim over
this module. Heavy dependencies (bittensor, the eval harness, taskgen) are
import-guarded inside the commands so ``epago validator sla`` / ``state`` work
on any machine with only the base install.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="Epago subnet tooling.")
validator_app = typer.Typer(no_args_is_help=True, help="Validator-in-a-box.")
app.add_typer(validator_app, name="validator")


def _register_role_clis() -> None:
    """Attach the role sub-apps; all are base-install friendly (heavy imports
    stay inside their commands)."""
    from epago.chain.cli import app as chain_app
    from epago.eval.cli import app as eval_app
    from epago.miner.cli import app as miner_app
    from epago.publishing.cli import app as publishing_app

    app.add_typer(miner_app, name="miner", help="Miner tooling.")
    app.add_typer(chain_app, name="chain", help="Chain preflight and debugging.")
    app.add_typer(eval_app, name="eval", help="Eval server (GPU box).")
    app.add_typer(publishing_app, name="publish", help="Public state mirroring.")


dashboard_app = typer.Typer(no_args_is_help=True, help="Static dashboard tooling.")
app.add_typer(dashboard_app, name="dashboard")


def _chain_for_dashboard(netuid: "Optional[int]", network: str, wallet_name: str, wallet_hotkey: str):
    """Read-only chain client for the canonical (``--from-chain``) dashboard."""
    if netuid is None:
        raise typer.BadParameter("--from-chain requires --netuid")
    import bittensor as bt

    from epago.chain.client import BittensorChainClient

    wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
    return BittensorChainClient(wallet, netuid=netuid, network=network)


@dashboard_app.command("export")
def dashboard_export(
    out: Path = typer.Option(Path("dashboard"), help="Output directory for dashboard.json + index.html."),
    state_dir: "Optional[Path]" = typer.Option(
        None, help="Validator state directory (or a published mirror). Omit when using --from-chain."
    ),
    from_chain: bool = typer.Option(
        False, "--from-chain", help="Build the canonical dashboard from chain state alone — no local files."
    ),
    netuid: "Optional[int]" = typer.Option(None, help="Subnet netuid (required with --from-chain)."),
    network: str = typer.Option("finney", help="Subtensor network (with --from-chain)."),
    wallet_name: str = typer.Option("default", help="Wallet name for read-only chain access."),
    wallet_hotkey: str = typer.Option("default", help="Wallet hotkey for read-only chain access."),
    chain_toml: "Optional[Path]" = typer.Option(None, help="Override chain.toml path."),
) -> None:
    """Render dashboard.json + the static site.

    Two sources: a validator's published state directory (default), or — with
    ``--from-chain`` — the chain itself, which yields the canonical consensus
    view that every observer reproduces bit-identically.
    """
    from epago.config import load_config
    from epago.dashboard.export import write_dashboard, write_dashboard_from_chain

    cfg = load_config(chain_toml)
    if from_chain:
        chain = _chain_for_dashboard(netuid, network, wallet_name, wallet_hotkey)
        path = write_dashboard_from_chain(chain, out, cfg)
    else:
        if state_dir is None:
            raise typer.BadParameter("provide --state-dir or --from-chain")
        path = write_dashboard(state_dir, out, cfg)
    console.print(f"[green]dashboard[/green] written: {path.parent}")


@dashboard_app.command("watch")
def dashboard_watch(
    out: Path = typer.Option(Path("dashboard"), help="Output directory."),
    state_dir: "Optional[Path]" = typer.Option(None, help="Validator state directory (omit with --from-chain)."),
    from_chain: bool = typer.Option(False, "--from-chain", help="Continuously re-export the canonical chain view."),
    netuid: "Optional[int]" = typer.Option(None, help="Subnet netuid (required with --from-chain)."),
    network: str = typer.Option("finney", help="Subtensor network (with --from-chain)."),
    wallet_name: str = typer.Option("default", help="Wallet name for read-only chain access."),
    wallet_hotkey: str = typer.Option("default", help="Wallet hotkey for read-only chain access."),
    interval_s: float = typer.Option(60.0, help="Seconds between re-exports."),
    chain_toml: "Optional[Path]" = typer.Option(None, help="Override chain.toml path."),
) -> None:
    """Continuously re-export the dashboard (from local state, or --from-chain)."""
    import time

    from epago.config import load_config
    from epago.dashboard.export import write_dashboard, write_dashboard_from_chain

    cfg = load_config(chain_toml)
    chain = _chain_for_dashboard(netuid, network, wallet_name, wallet_hotkey) if from_chain else None
    src = "chain" if from_chain else str(state_dir)
    console.print(f"[green]dashboard watch[/green] {src} -> {out} every {interval_s}s")
    while True:
        try:
            if from_chain:
                write_dashboard_from_chain(chain, out, cfg)
            elif state_dir is not None:
                write_dashboard(state_dir, out, cfg)
            else:
                raise typer.BadParameter("provide --state-dir or --from-chain")
        except FileNotFoundError:
            console.print("[yellow]state.json not present yet — waiting[/yellow]")
        time.sleep(interval_s)


_register_role_clis()

console = Console()

_DEFAULT_STATE_DIR = Path.home() / ".epago" / "validator"
_DEFAULT_CACHE_DIR = Path.home() / ".epago" / "models"


def _fail(message: str) -> "typer.Exit":
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code=1)


def _restore_epago_log_levels() -> None:
    """Undo bittensor's blanket silencing of every logger that already exists.

    Importing bittensor walks the logger registry and sets third-party loggers
    to CRITICAL ("Enabling default logging (Warning level)"). Every
    ``epago.*`` logger is already registered by then, so the validator's own
    mechanism lines go with them — not just INFO, but the ``set_weights``
    retry warnings, the round-field cap, the burn-hotkey fallback and the
    ``tick failed`` traceback that ``run_forever`` logs before degrading. A
    live validator then wrote essentially nothing to its log while it ran.
    Resetting our tree to NOTSET hands it back to whatever the operator
    configured on the root logger, and touches no logger outside ``epago``.
    """
    import logging

    for name, logger_obj in list(logging.root.manager.loggerDict.items()):
        if (name == "epago" or name.startswith("epago.")) and isinstance(
            logger_obj, logging.Logger
        ):
            logger_obj.setLevel(logging.NOTSET)


def _load_bittensor():
    try:
        import bittensor as bt  # noqa: PLC0415
    except ImportError:
        raise _fail(
            "bittensor is not installed — the validator loop needs the chain extra: "
            r"pip install 'epago\[chain]'"
        ) from None
    _restore_epago_log_levels()
    return bt


def _load_eval():
    try:
        from epago.eval import duel as duel_mod  # noqa: PLC0415
        from epago.eval import probes as probes_mod  # noqa: PLC0415
    except ImportError:
        raise _fail(
            r"the eval subsystem is not installed — pip install 'epago\[eval]' "
            "(torch/vllm) and make sure epago.eval is present"
        ) from None
    return duel_mod, probes_mod


@validator_app.command("run")
def validator_run(
    netuid: int = typer.Option(..., help="Subnet netuid."),
    network: str = typer.Option("finney", help="Subtensor network."),
    state_dir: Path = typer.Option(_DEFAULT_STATE_DIR, help="Durable validator state directory."),
    corpus: Path = typer.Option(..., help="Path to the pinned SQLite corpus snapshot."),
    cache_dir: Path = typer.Option(_DEFAULT_CACHE_DIR, help="Model snapshot cache."),
    wallet_name: str = typer.Option("default", help="Bittensor wallet name."),
    wallet_hotkey: str = typer.Option("default", help="Bittensor wallet hotkey."),
    chain_toml: Optional[Path] = typer.Option(None, help="Override chain.toml path."),
    poll_interval_s: float = typer.Option(30.0, help="Seconds between ticks."),
    ingest_dir: Optional[Path] = typer.Option(
        None,
        help="DEPRECATED manual override — prefer the automated [private_source] feed "
        "in chain.toml. A local drop directory of post-snapshot private-task documents.",
    ),
) -> None:
    """Run the validator loop against the live chain."""
    from epago.config import load_config
    from epago.validator.service import ValidatorService
    from epago.validator.wiring import build_production_deps

    cfg = load_config(chain_toml)
    bt = _load_bittensor()
    _load_eval()

    from epago.chain.client import BittensorChainClient

    wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
    chain = BittensorChainClient(wallet, netuid=netuid, network=network)

    def _sign(data: bytes) -> str:
        return wallet.hotkey.sign(data).hex()

    deps = build_production_deps(
        cfg=cfg,
        chain=chain,
        state_dir=state_dir,
        corpus_path=corpus,
        cache_dir=cache_dir,
        wallet_hotkey=wallet.hotkey.ss58_address,
        ingest_dir=ingest_dir,
        sign=_sign,
    )
    service = ValidatorService(deps)
    console.print(f"[green]epago validator[/green] netuid={netuid} network={network}")
    asyncio.run(service.run_forever(poll_interval_s=poll_interval_s))


@validator_app.command("sla")
def validator_sla(
    state_dir: Path = typer.Option(_DEFAULT_STATE_DIR, help="Validator state directory."),
) -> None:
    """Print the reveal-to-verdict latency report (p50/p95 in blocks)."""
    from epago.validator.service import compute_sla_report
    from epago.validator.state import ValidatorState

    state = ValidatorState.load(state_dir)
    console.print_json(json.dumps(compute_sla_report(state.sla)))


@validator_app.command("state")
def validator_state(
    state_dir: Path = typer.Option(_DEFAULT_STATE_DIR, help="Validator state directory."),
) -> None:
    """Print a machine-readable summary of the durable validator state."""
    from epago.validator.state import ValidatorState, _king_to_dict

    state = ValidatorState.load(state_dir)
    summary = {
        "king": _king_to_dict(state.king),
        "king_acc_ema": state.king_acc_ema,
        "queue_depth": len(state.queue),
        "queued_digests": [q.digest for q in state.queue],
        "candidates": sorted(state.candidates),
        "clean_duels": state.clean_duels,
        "organic_dethrones": state.organic_dethrones,
        "genesis_block": state.genesis_block,
        "arena_entries": len(state.arena),
        "burned_bonds": state.burned_bonds,
        "noise_floor_samples": state.noise_floor_samples,
        "failure_memory_size": len(state.failure_memory),
        "last_weights_block": state.last_weights_block,
        "tick_count": state.tick_count,
        "pending_mirror": state.pending_mirror,
        "last_error": state.last_error,
    }
    console.print_json(json.dumps(summary))


if __name__ == "__main__":
    app()
