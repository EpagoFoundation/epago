#!/usr/bin/env python
"""Epago model-miner CLI.

Thin wrapper over :mod:`epago.miner.workflow` — every command delegates to the
reference pipeline. The lifecycle is::

    prepare   fetch the current king snapshot and copy it as your challenger
    (train)   your business entirely — see epago/miner/train_example.py
    preflight run the exact validator intake checks locally, before submitting
    submit    build the e2 payload and commit it via timelock commit-reveal
    status    look up your last submission in the public dashboard state

Chain access needs the Bittensor SDK (``pip install 'epago[chain]'``) and a
local wallet; ``--mock`` / ``--dry-run`` work without either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from epago.config import load_config
from epago.core.types import ModelRef
from epago.miner import workflow

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=__doc__,
    pretty_exceptions_show_locals=False,
)

_CONFIG_OPT = typer.Option(None, "--config", help="path to chain.toml (defaults to the packaged one)")


def _fail(message: str) -> "typer.Exit":
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


@app.command()
def prepare(
    out_dir: Path = typer.Argument(..., help="challenger folder to create (your training start)"),
    state_dir: Path = typer.Option(
        Path("~/.epago/miner"), "--state-dir", help="local cache for downloaded snapshots"
    ),
    king_repo: Optional[str] = typer.Option(None, help="override: king repo (skip chain resolution)"),
    king_digest: Optional[str] = typer.Option(None, help="override: king digest (hf:.. or sha256:..)"),
    king_dir: Optional[Path] = typer.Option(
        None, help="override: already-materialized king snapshot folder (skip download)"
    ),
    config: Optional[Path] = _CONFIG_OPT,
    wallet_name: Optional[str] = typer.Option(None, help="bittensor wallet name (chain resolution)"),
    wallet_hotkey: Optional[str] = typer.Option(None, help="bittensor wallet hotkey name"),
    mock: bool = typer.Option(False, help="use an in-memory mock chain (local rehearsal)"),
) -> None:
    """Fetch the current king and copy it into OUT_DIR as your challenger."""
    cfg = load_config(config)
    if king_dir is not None:
        king_path = king_dir.expanduser()
        typer.echo(f"using local king snapshot: {king_path}")
    else:
        if (king_repo is None) != (king_digest is None):
            raise _fail("--king-repo and --king-digest must be given together")
        if king_repo is not None and king_digest is not None:
            ref = ModelRef(repo=king_repo, digest=king_digest)
        else:
            try:
                chain = workflow.make_chain_client(cfg, wallet_name, wallet_hotkey, mock=mock)
                ref = workflow.resolve_king_from_chain(chain, cfg)
            except (RuntimeError, workflow.NoKingError) as exc:
                raise _fail(str(exc))
        typer.echo(f"king: {ref.repo} @ {ref.digest}")
        try:
            ref, king_path = workflow.fetch_king(cfg, state_dir, ref)
        except Exception as exc:  # noqa: BLE001 - surface download errors cleanly
            raise _fail(f"could not materialize king snapshot: {exc}")
    out = workflow.prepare_challenger(king_path, out_dir.expanduser())
    typer.echo(f"challenger prepared at {out}")
    typer.echo("train it however you like, then run `preflight` before `submit`")


@app.command()
def preflight(
    challenger_dir: Path = typer.Argument(..., help="your trained challenger folder"),
    king_dir: Path = typer.Argument(..., help="the king snapshot you trained against"),
    repo: str = typer.Option(..., help="the repo you will publish the challenger to"),
    hotkey: str = typer.Option(..., help="your hotkey ss58 (repo-name policy check)"),
    config: Optional[Path] = _CONFIG_OPT,
    format_probe: bool = typer.Option(
        False, help="also run a local format probe (needs the eval extra installed)"
    ),
) -> None:
    """Run the exact validator intake checks locally. Exit 1 on any failure."""
    cfg = load_config(config)
    problems = workflow.preflight(
        challenger_dir.expanduser(),
        king_dir.expanduser(),
        repo,
        hotkey,
        cfg,
        format_probe=format_probe,
    )
    if problems:
        typer.echo(f"PREFLIGHT FAIL ({len(problems)} problem(s)):")
        for p in problems:
            typer.echo(f"  - {p}")
        typer.echo("fix these before submitting — a validator would reject this at intake")
        raise typer.Exit(code=1)
    typer.echo("PREFLIGHT PASS: this challenger would clear validator intake")


@app.command()
def submit(
    repo: str = typer.Option(..., help="published challenger repo"),
    digest: str = typer.Option(..., help="pinned challenger digest (hf:<rev> or sha256:<hex>)"),
    king_digest: str = typer.Option(..., help="digest of the king this challenger was trained against"),
    hotkey: Optional[str] = typer.Option(
        None, help="author hotkey ss58 embedded in the payload (defaults to the wallet hotkey)"
    ),
    config: Optional[Path] = _CONFIG_OPT,
    wallet_name: Optional[str] = typer.Option(None, help="bittensor wallet name"),
    wallet_hotkey: Optional[str] = typer.Option(None, help="bittensor wallet hotkey name"),
    dry_run: bool = typer.Option(False, help="print the e2 payload without touching any chain"),
    mock: bool = typer.Option(False, help="submit to an in-memory mock chain (local rehearsal)"),
) -> None:
    """Build the e2 reveal payload and submit it via timelock commit-reveal."""
    from epago import constants
    from epago.core.reveal import build_reveal

    challenger = ModelRef(repo=repo, digest=digest)
    if dry_run:
        payload = build_reveal(king_digest, challenger)
        typer.echo(payload)
        return
    cfg = load_config(config)
    try:
        chain = workflow.make_chain_client(cfg, wallet_name, wallet_hotkey, mock=mock)
    except RuntimeError as exc:
        raise _fail(str(exc))
    if hotkey is None:
        wallet = getattr(chain, "wallet", None)
        hotkey = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
        if hotkey is None:
            raise _fail("no --hotkey given and the chain client exposes no wallet hotkey")
    payload = workflow.submit(chain, king_digest, challenger, hotkey)
    typer.echo(f"submitted: {payload}")
    typer.echo(f"reveals in ~{constants.BLOCKS_UNTIL_REVEAL} blocks; watch `status` afterwards")


@app.command()
def status(
    hotkey: str = typer.Argument(..., help="your hotkey ss58"),
    state: str = typer.Option(
        ..., help="public dashboard state: local JSON path, http(s) URL, or s3://<repo>[/<path>]"
    ),
) -> None:
    """Show your last submission's state from the public dashboard/audit state."""
    try:
        doc = workflow.load_public_state(state)
    except Exception as exc:  # noqa: BLE001 - bad path/URL/JSON all end the same way
        raise _fail(f"could not load state from {state!r}: {exc}")
    entries = workflow.find_hotkey_entries(doc, hotkey)
    if not entries:
        typer.echo(f"no entries mentioning {hotkey} in {state}")
        raise typer.Exit(code=1)
    typer.echo(f"{len(entries)} entr(ies) for {hotkey}:")
    for entry in entries:
        typer.echo(json.dumps(entry, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    app()


# --- private upload: fetch credentials, upload, submit -----------------------
#
# A challenger uploads into the validator's own object store rather than to a
# public repository, so a model that loses is never handed to the rivals that
# beat it. The two commands below are that path: `auth` opens the envelope
# sealed to this hotkey, `upload` pushes the checkpoint into the one prefix
# those credentials can write to.
#
# Public submission still works exactly as before — push to a Hugging Face
# repo you own and call `submit` with `hf:<revision>`. Uploading privately is
# a choice about who can read your weights before you win, not a different
# protocol.


def _load_hotkey_seed(wallet_name: str, wallet_hotkey: str) -> bytes:
    """The Ed25519 seed for this hotkey, straight from the local wallet.

    Never leaves the machine and is never written anywhere by this CLI: it is
    used once, in memory, to open an envelope.
    """
    try:
        from bittensor_wallet import Wallet
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise _fail("this command needs the bittensor wallet: pip install 'epago[chain]'") from exc

    wallet = Wallet(name=wallet_name, hotkey=wallet_hotkey)
    keypair = wallet.hotkey
    seed = getattr(keypair, "private_key", None)
    if not seed:
        raise _fail(
            f"could not read a private key for hotkey {wallet_hotkey!r}. "
            "Submitting privately needs an Ed25519 hotkey: create one with "
            "`btcli wallet new-hotkey --key-type ed25519`."
        )
    # substrate keypairs carry a 64-byte expanded key; the seed is the first 32.
    return bytes(seed)[:32]


@app.command()
def auth(
    mailbox: str = typer.Option(..., help="mailbox URL or local path published by the validator"),
    wallet_name: str = typer.Option(..., help="bittensor wallet name"),
    wallet_hotkey: str = typer.Option(..., help="bittensor wallet hotkey name"),
    out: Path = typer.Option(Path("upload-auth.json"), help="where to write the opened credentials"),
) -> None:
    """Open this hotkey's envelope from the public mailbox.

    The mailbox is public and holds one envelope per miner; only the holder of
    a hotkey's private key can open that hotkey's entry. Nothing here is
    secret because of *where* it lives — the sealing is the protection.
    """
    import json as _json
    import stat
    import urllib.request

    from epago.chain.envelope import EnvelopeError, open_envelope
    from epago.chain.mailbox import Mailbox

    try:
        if mailbox.startswith(("http://", "https://")):
            with urllib.request.urlopen(mailbox, timeout=60) as response:
                raw = response.read()
        else:
            raw = Path(mailbox).expanduser().read_bytes()
    except Exception as exc:  # noqa: BLE001
        raise _fail(f"could not read the mailbox at {mailbox}: {exc}")

    try:
        box = Mailbox.from_json(raw)
    except EnvelopeError as exc:
        raise _fail(str(exc))

    if box.is_expired():
        typer.echo(
            "warning: this mailbox has expired; the validator republishes on a "
            "cadence, so fetch it again before uploading",
            err=True,
        )

    from bittensor_wallet import Wallet

    hotkey_ss58 = Wallet(name=wallet_name, hotkey=wallet_hotkey).hotkey.ss58_address
    envelope = box.for_hotkey(hotkey_ss58)
    if envelope is None:
        raise _fail(
            f"no envelope for {hotkey_ss58} in this mailbox. Register the hotkey on "
            "the subnet and wait for the next mailbox publication."
        )

    try:
        credentials = open_envelope(envelope, _load_hotkey_seed(wallet_name, wallet_hotkey))
    except EnvelopeError as exc:
        raise _fail(str(exc))

    out = Path(out).expanduser()
    out.write_text(_json.dumps(credentials, indent=2))
    out.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600: never share or commit this file
    typer.echo(f"credentials written to {out} (mode 0600 — do not share or commit)")
    typer.echo(f"you may write to: {credentials.get('prefix', '?')}")


@app.command()
def upload(
    folder: Path = typer.Option(..., help="the checkpoint folder to upload"),
    auth_file: Path = typer.Option(Path("upload-auth.json"), help="credentials from `auth`"),
    name: str = typer.Option("model", help="a short name for this submission"),
) -> None:
    """Upload a checkpoint into this hotkey's prefix and print its pinned ref.

    The printed ``repo`` and ``sha256:`` digest are what `submit` then commits
    to chain. The digest is computed from the uploaded bytes, so the
    commitment pins exactly what was stored.
    """
    import json as _json

    from epago.model.objectstore import ObjectStore

    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise _fail(f"not a directory: {folder}")

    try:
        credentials = _json.loads(Path(auth_file).expanduser().read_text())
    except Exception as exc:  # noqa: BLE001
        raise _fail(f"could not read {auth_file}: {exc}. Run `auth` first.")

    prefix = str(credentials.get("prefix", "")).rstrip("/")
    if not prefix:
        raise _fail("these credentials carry no prefix; re-run `auth`")

    store = ObjectStore(
        bucket=credentials.get("bucket"),
        endpoint=credentials.get("endpoint"),
        region=credentials.get("region"),
        access_key=credentials.get("access_key"),
        secret_key=credentials.get("secret_key"),
    )
    repo = f"{prefix}/{name}"
    try:
        digest = store.upload_snapshot(repo, folder)
    except Exception as exc:  # noqa: BLE001
        raise _fail(f"upload failed: {exc}")

    typer.echo(f"uploaded to {repo}")
    typer.echo(f"digest: {digest}")
    typer.echo("")
    typer.echo("submit it with:")
    typer.echo(f"  epago-miner submit --repo {repo} --digest {digest} --king-digest <king>")
