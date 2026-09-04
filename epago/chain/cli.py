"""Chain preflight CLI.

``epago chain check`` runs the doctor (see :mod:`epago.chain.doctor`) and
renders one PASS/FAIL/WARN/SKIP row per check; exit code 1 iff any check
FAILed. ``epago chain watch-reveals`` tails the subnet's timelock-reveal
channel — the fastest way to see e2/er1/ev3/ep1/ek1 payloads land on testnet.
"""

from __future__ import annotations

import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Chain preflight and debugging tools.")
console = Console()

_STATUS_STYLE = {
    "PASS": "green",
    "FAIL": "bold red",
    "WARN": "yellow",
    "SKIP": "dim",
}


@app.command()
def check(
    netuid: int = typer.Option(..., help="Subnet netuid to check."),
    network: str = typer.Option("finney", help="Subtensor network."),
    wallet_name: Optional[str] = typer.Option(
        None, help="Bittensor wallet name (enables the wallet checks)."
    ),
    wallet_hotkey: Optional[str] = typer.Option(
        None, help="Bittensor wallet hotkey (default: 'default')."
    ),
    probe_writes: bool = typer.Option(
        False,
        "--probe-writes",
        help="ALSO write test payloads through both on-chain channels (status slot + "
        "timelock reveal) and read them back. Costs a transaction fee; testnet-friendly.",
    ),
) -> None:
    """Preflight a subtensor + netuid (+ optionally your wallet) for Epago."""
    from epago.chain.doctor import run_doctor

    results = run_doctor(
        network=network,
        netuid=netuid,
        wallet_name=wallet_name,
        wallet_hotkey=wallet_hotkey,
        probe_writes=probe_writes,
    )

    table = Table(title=f"epago chain doctor — {network!r} netuid {netuid}", show_lines=False)
    table.add_column("status", justify="center")
    table.add_column("check", style="bold")
    table.add_column("detail", overflow="fold")
    for r in results:
        table.add_row(f"[{_STATUS_STYLE.get(r.status, '')}]{r.status}[/]", r.name, r.detail)
    console.print(table)

    failed = [r for r in results if r.status == "FAIL"]
    warned = [r for r in results if r.status == "WARN"]
    if failed:
        console.print(f"[bold red]{len(failed)} check(s) FAILED[/] — fix the details above.")
        raise typer.Exit(code=1)
    if warned:
        console.print(f"[yellow]all checks pass ({len(warned)} warning(s))[/]")
    else:
        console.print("[green]all checks pass[/]")


@app.command("watch-reveals")
def watch_reveals(
    netuid: int = typer.Option(..., help="Subnet netuid to watch."),
    network: str = typer.Option("finney", help="Subtensor network."),
    prefix: str = typer.Option(
        "", help="Only show payloads with this prefix (e.g. 'ev3|', 'er1|'); empty = all."
    ),
    since_block: int = typer.Option(0, help="Ignore reveals older than this block."),
    interval_s: float = typer.Option(12.0, help="Seconds between polls (~one block)."),
    once: bool = typer.Option(False, "--once", help="Print the current backlog and exit."),
) -> None:
    """Stream newly revealed timelock payloads (testnet debugging aid)."""
    try:
        import bittensor as bt
    except ImportError:
        console.print(r"[red]error:[/red] bittensor is not installed — pip install 'epago\[chain]'")
        raise typer.Exit(code=1)

    subtensor = bt.Subtensor(network=network)
    console.print(
        f"[green]watching reveals[/green] network={network!r} netuid={netuid} "
        f"prefix={prefix!r} since_block={since_block}"
    )
    seen: set[tuple[str, int, str]] = set()
    while True:
        try:
            revealed = subtensor.get_all_revealed_commitments(netuid) or {}
        except Exception as exc:  # noqa: BLE001 - a watch loop must survive RPC blips
            console.print(f"[yellow]poll failed:[/yellow] {type(exc).__name__}: {exc}")
            revealed = {}
        fresh: list[tuple[int, str, str]] = []
        for hotkey, entries in revealed.items():
            for block, payload in entries:
                key = (str(hotkey), int(block), str(payload))
                if key in seen or int(block) < since_block:
                    continue
                seen.add(key)
                if prefix and not str(payload).startswith(prefix):
                    continue
                fresh.append((int(block), str(hotkey), str(payload)))
        for block, hotkey, payload in sorted(fresh):
            console.print(f"[cyan]block {block}[/cyan] [bold]{hotkey}[/bold] {payload}")
        if once:
            return
        time.sleep(interval_s)


if __name__ == "__main__":
    app()


@app.command("start-round")
def start_round(
    config: Optional[str] = typer.Option(None, "--config", help="Path to chain.toml."),
    wallet_name: Optional[str] = typer.Option(None, help="bittensor wallet name."),
    wallet_hotkey: Optional[str] = typer.Option(None, help="bittensor wallet hotkey name."),
    round_no: Optional[int] = typer.Option(
        None, "--round", help="Round number; defaults to the last round on chain plus one."
    ),
    force: bool = typer.Option(
        False, "--force", help="Publish even if the local interval check would refuse."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the er1 payload and exit."),
    mock: bool = typer.Option(False, "--mock", help="Publish to an in-memory mock chain."),
) -> None:
    """Open a competition round (round authority only).

    Nothing is evaluated on this subnet until this command runs: validators
    queue submissions continuously but only duel when a round opens. The
    authority is the wallet hotkey, which must match
    ``[chain] round_authority_hotkey`` — the signature IS the credential, so
    there is no shared secret to leak or rotate separately.

    The chain re-checks both rules validators enforce (a strictly increasing
    round number, and at least ``ROUND_MIN_INTERVAL_BLOCKS`` since the previous
    round), so a payload published too early is simply ignored by every
    validator. ``--force`` skips only the *local* pre-check; it cannot make
    validators accept an early round.
    """
    from epago import constants
    from epago.config import load_config
    from epago.core.reveal import build_round_start
    from epago.miner import workflow

    cfg = load_config(config)
    try:
        chain = workflow.make_chain_client(cfg, wallet_name, wallet_hotkey, mock=mock)
    except RuntimeError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    authority = cfg.chain.round_authority_hotkey
    if not authority:
        console.print(
            "[red]error:[/red] [chain] round_authority_hotkey is empty in "
            f"{cfg.path} — no round can be started until it names a hotkey"
        )
        raise typer.Exit(code=1)

    previous = None if dry_run and mock else chain.latest_round(authority)
    if round_no is None:
        round_no = 1 if previous is None else previous.round + 1

    if previous is not None and not force:
        if round_no <= previous.round:
            console.print(
                f"[red]error:[/red] round {round_no} is not newer than the last "
                f"round on chain ({previous.round})"
            )
            raise typer.Exit(code=1)
        elapsed = chain.current_block() - previous.block
        if elapsed < constants.ROUND_MIN_INTERVAL_BLOCKS:
            remaining = constants.ROUND_MIN_INTERVAL_BLOCKS - elapsed
            console.print(
                f"[red]error:[/red] only {elapsed} blocks since round {previous.round}; "
                f"minimum is {constants.ROUND_MIN_INTERVAL_BLOCKS} "
                f"(~{remaining * 12 / 3600:.1f}h to go). Validators would ignore this round."
            )
            raise typer.Exit(code=1)

    payload = build_round_start(round_no)
    if dry_run:
        console.print(payload)
        return

    if not chain.publish_reveal(payload, constants.VERDICT_REVEAL_BLOCKS):
        console.print(
            "[red]error:[/red] the chain did not accept the commit (rate limited?) — retry"
        )
        raise typer.Exit(code=1)
    console.print(
        f"[green]round {round_no} opened[/green] — payload {payload!r} reveals in "
        f"{constants.VERDICT_REVEAL_BLOCKS} blocks; the block hash there mints the exam"
    )


@app.command("request-round")
def request_round(
    url: str = typer.Option("http://127.0.0.1:8799", help="Validator round-trigger base URL."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="Round API key; falls back to EPAGO_ROUND_API_KEY."
    ),
) -> None:
    """Open a competition round via the validator's API-key trigger.

    The key is the credential — no wallet, no on-chain signature. The validator
    seeds the round's exam from the chain block current when the request lands,
    so the trigger cannot leak the questions. A request that arrives before the
    minimum interval has elapsed is accepted by the endpoint but ignored by the
    validator; check its logs to confirm the round actually opened.
    """
    import os
    import urllib.error
    import urllib.request

    key = api_key or os.environ.get("EPAGO_ROUND_API_KEY", "")
    if not key:
        console.print("[red]error:[/red] no API key (pass --api-key or set EPAGO_ROUND_API_KEY)")
        raise typer.Exit(code=1)
    req = urllib.request.Request(
        url.rstrip("/") + "/round/start",
        method="POST",
        headers={"X-Epago-Round-Key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            console.print(f"[green]{resp.status}[/green] {resp.read().decode()[:120]}")
    except urllib.error.HTTPError as exc:
        console.print(f"[red]{exc.code}[/red] {exc.read().decode()[:120]}")
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)
