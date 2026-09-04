#!/usr/bin/env python
"""Genesis utility: compute and print the chain.toml pins for a new generation.

Given a local seed model folder and a corpus snapshot file, this computes the
content-addressed digests and prints the exact ``[seed]`` / ``[eval]`` lines
to paste into ``chain.toml``. With ``--upload`` the seed folder is first
pushed to the Hugging Face Hub and the printed digest pins the returned
revision (``hf:<commit>``), which is what validators will materialize.

Credentials: no secrets live in code or in chain.toml. Uploads authenticate
through the standard Hugging Face environment — set ``HF_TOKEN`` (or run
``huggingface-cli login`` beforehand). Chain registration itself (netuid,
wallet) is a separate operator step and is not touched here.

Usage:
    .venv/bin/python scripts/seed_genesis.py SEED_DIR CORPUS_DB \\
        --seed-repo Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \\
        --corpus-repo epago-ai/epago-corpus-medicine-1 [--upload]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    seed_dir: Path = typer.Argument(..., help="local seed model folder (canonical safetensors layout)"),
    corpus_db: Path = typer.Argument(..., help="corpus snapshot file (corpus.db)"),
    seed_repo: str = typer.Option(
        # Current generation: keep in sync with chains/EPAGO-DR-30B.toml [chain].seed_repo.
        "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        help="repo the seed model is (or will be) published to",
    ),
    corpus_repo: str = typer.Option(
        "epago-ai/epago-corpus-medicine-1", help="repo the corpus snapshot is published to"
    ),
    tokenizer_repo: Optional[str] = typer.Option(
        None, help="tokenizer repo pin (defaults to the seed repo itself)"
    ),
    taskgen_release: str = typer.Option("SCI2", help="taskgen release tag to pin"),
    upload: bool = typer.Option(
        False, help="upload SEED_DIR to --seed-repo first and pin the returned hf revision"
    ),
) -> None:
    """Compute genesis digests and print the exact chain.toml pin lines."""
    from epago.environment.sync import corpus_digest
    from epago.model.store import snapshot_digest, upload_model_folder

    seed_dir = seed_dir.expanduser()
    corpus_db = corpus_db.expanduser()
    if not (seed_dir / "config.json").exists():
        typer.echo(f"error: {seed_dir} does not look like a model snapshot (no config.json)", err=True)
        raise typer.Exit(code=1)
    if not any(seed_dir.rglob("*.safetensors")):
        typer.echo(f"error: {seed_dir} contains no .safetensors files", err=True)
        raise typer.Exit(code=1)
    if not corpus_db.is_file():
        typer.echo(f"error: corpus snapshot {corpus_db} is not a file", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"hashing seed folder {seed_dir} ...")
    folder_digest = snapshot_digest(seed_dir)
    typer.echo(f"  folder digest: {folder_digest}")

    if upload:
        typer.echo(f"uploading to {seed_repo} (auth via HF_TOKEN / huggingface-cli login) ...")
        try:
            ref = upload_model_folder(seed_dir, seed_repo)
        except Exception as exc:  # noqa: BLE001 - hub errors surface as-is
            typer.echo(f"error: upload failed: {exc}", err=True)
            raise typer.Exit(code=1)
        seed_digest = ref.digest
        typer.echo(f"  uploaded, pinned revision: {seed_digest}")
    else:
        seed_digest = folder_digest
        typer.echo(
            "  (no --upload: pinning the folder sha256; with the hf backend you will "
            "usually re-run with --upload to pin the hub revision instead)"
        )

    typer.echo(f"hashing corpus snapshot {corpus_db} ...")
    c_digest = corpus_digest(corpus_db)
    typer.echo(f"  corpus digest: {c_digest}")

    typer.echo("")
    typer.echo("# ---- paste into chain.toml ------------------------------------")
    typer.echo("[seed]")
    typer.echo(f'tokenizer_repo = "{tokenizer_repo or seed_repo}"')
    typer.echo('repo_backend   = "hf"')
    typer.echo(f'seed_digest    = "{seed_digest}"')
    typer.echo("")
    typer.echo("[eval]")
    typer.echo(f'corpus_repo     = "{corpus_repo}"')
    typer.echo(f'corpus_digest   = "{c_digest}"')
    typer.echo(f'taskgen_release = "{taskgen_release}"')
    typer.echo("# judge_repo / judge_digest: pin separately once the judge model is published")
    typer.echo("# also set chain.seed_repo to match [seed] above:")
    typer.echo(f'# seed_repo    = "{seed_repo}"')


if __name__ == "__main__":
    app()
