#!/usr/bin/env python
"""One full private-holdout rotation: harvest -> publish (PRIVATE) -> pin.

The anti-overfit guarantee is *freshness*: the private half of every duel is
minted from papers published after the miner's training cutoff. That only holds
while the feed keeps moving, so the rotation is a loop, not a command someone
remembers to run — ``docker/docker-compose.yml``'s ``holdout-rotator`` service
(profile ``auto-holdout``) runs this script every ``EPAGO_HOLDOUT_INTERVAL_S``
seconds (default 7 days).

One cycle:

1. harvest a fresh dated window with :mod:`scripts.harvest_holdout` (all four
   OpenAlex domains and the ``general`` vocabulary by default — the SCI3 release
   the live contract runs);
2. publish the shards to a PRIVATE, dated dataset repo whose name states its
   scope (``…/epago-holdout-science-2026w34``);
3. emit the new ``[private_source]`` repo + revision as JSON on stdout (and to
   ``--json-out``), plus the ready-to-paste TOML block, so the next step needs no
   human to read a log;
4. optionally rewrite the ``[private_source]`` block of a chain contract, keeping
   the old pin in a comment and a timestamped backup beside it.

Nothing above happens for real unless ``--apply`` is given. The default is
``--dry-run``: harvest, write the shards and the block locally, report what
*would* be published. A rotation changes what every validator scores against, so
going live is an explicit act.

Safety rules, all of them refusals rather than warnings:

* **No token, no start.** The HF write token is read from ``HUGGINGFACE_TOKEN`` /
  ``HF_TOKEN`` (or a repo-root ``.env``) *before* the harvest — a multi-hour fetch
  must never end in "no token".
* **No starved week.** Sources rate-limit; OpenAlex has hard-throttled a harvest
  before. A slice below ``--min-papers`` usable papers is written locally for
  inspection and refused for publication (exit 3) — a thin week would leave
  validators short of private tasks, and the old feed is better than a bad one.
* **No duplicate, no clobber.** A period whose repo is already pinned in the
  contract, already in the rotation ledger, or already present on the hub is
  skipped (exit 0). If the hub cannot be reached to check, ``--apply`` refuses
  rather than risk pushing a second slice over a live feed.
* **Private, always.** :func:`scripts.harvest_holdout.publish` reads the repo's
  visibility back from the hub and aborts if it is public.
* **Determinism untouched.** The plan seed (OS-random unless ``--seed``) and the
  whole plan are recorded in ``manifest.json`` and in the JSON report, so any
  rotation can be rebuilt exactly.

Delayed transparency is unchanged: the live slice stays private, and a slice is
revealed only once it has rotated out of service.

Usage:
    # rehearse (publishes nothing, touches no contract)
    .venv/bin/python scripts/rotate_holdout.py --dry-run

    # rotate for real and pin it into the active contract
    .venv/bin/python scripts/rotate_holdout.py --apply \\
        --contract chains/EPAGO-DR-30B.toml --json-out holdout/last-rotation.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.harvest_holdout import (  # noqa: E402
    DEFAULT_ORG,
    DOMAINS,
    TOKEN_HELP,
    VOCABULARIES,
    Stats,
    default_repo,
    parse_domains,
    period_tag,
    plan_harvest,
    publish,
    read_token,
    run_plan,
    write_shards,
)

#: Fewest usable papers a slice may carry and still go live. A validator mints
#: ``N_PRIV_TASKS`` (200) private tasks per rotation from a 4-documents-per-task
#: budget, so a published slice below 800 papers cannot fill one rotation even if
#: every shard is read — and ``max_shards`` means only a few are. The harvest
#: target (2500) leaves the intended headroom; this is the floor below which the
#: week is a failed harvest, not a small one.
DEFAULT_MIN_PAPERS = 800

#: Exit codes, so the scheduler (and a human) can tell the failures apart.
EXIT_OK = 0
EXIT_CONFIG = 2      # missing token / unusable contract / unverifiable hub
EXIT_STARVED = 3     # harvest came back below the minimum; nothing published
EXIT_PUBLISH = 4     # the publish itself failed

_ROTATION_COMMENT = "# rotated "


def log(msg: str) -> None:
    """Human progress goes to stderr; stdout carries the JSON report alone."""
    print(msg, file=sys.stderr, flush=True)


# --- the [private_source] block ----------------------------------------------


def render_block(repo: str, revision: str | None, text_column: str, max_shards: int) -> str:
    """The contract block for this rotation, ready to paste or apply."""
    return "\n".join(
        [
            "[private_source]",
            f'repo        = "{repo}"',
            f'revision    = "{revision or "<commit-after-publish>"}"',
            f'text_column = "{text_column}"',
            f"max_shards  = {max_shards}",
            "# PRIVATE while live; reveal this repo only after it has rotated out",
            "",
        ]
    )


def read_pinned_repo(contract: Path) -> str:
    """The ``[private_source].repo`` a contract currently pins ("" if none)."""
    try:
        import tomllib

        with contract.open("rb") as fh:
            return str((tomllib.load(fh).get("private_source") or {}).get("repo") or "")
    except (OSError, ValueError):
        return ""


def update_private_source(
    text: str, repo: str, revision: str, when: str, text_column: str = "text",
    max_shards: int | None = None,
) -> tuple[str, dict]:
    """Rewrite a contract's ``[private_source]`` pin, returning (text, previous).

    Surgical on purpose: only ``repo`` and ``revision`` (and ``max_shards`` when
    asked) change, every comment and every other key survives, and the pin being
    replaced is preserved in a ``# rotated …`` comment — the outgoing slice is
    the one that becomes revealable, so its name has to stay written down. One
    such comment is kept, not a growing stack.
    """
    lines = text.splitlines()
    try:
        head = next(i for i, ln in enumerate(lines) if ln.strip() == "[private_source]")
    except StopIteration:
        raise ValueError(
            "contract has no [private_source] section to update; add one (see "
            "chains/EPAGO-DR-30B.toml) or drop --contract and apply the block by hand"
        ) from None
    end = next(
        (i for i in range(head + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )

    previous = {"repo": "", "revision": ""}
    body: list[str] = []
    seen: set[str] = set()
    for line in lines[head + 1 : end]:
        if line.strip().startswith(_ROTATION_COMMENT):
            continue  # keep exactly one rotation note, not a growing stack
        replaced = False
        for key, value in (("repo", repo), ("revision", revision),
                           ("max_shards", max_shards), ("text_column", text_column)):
            if value is None:
                continue
            quoted = key != "max_shards"
            pattern = (
                rf'^(\s*{key}\s*=\s*)"(?P<old>[^"]*)"(?P<post>.*)$' if quoted
                else rf"^(\s*{key}\s*=\s*)(?P<old>\S+)(?P<post>.*)$"
            )
            m = re.match(pattern, line)
            if not m:
                continue
            if key in previous:
                previous[key] = m.group("old")
            body.append(f'{m.group(1)}"{value}"{m.group("post")}' if quoted
                        else f"{m.group(1)}{value}{m.group('post')}")
            seen.add(key)
            replaced = True
            break
        if not replaced:
            body.append(line)

    for key, value in (("revision", revision), ("repo", repo)):  # section was incomplete
        if key not in seen:
            body.insert(0, f'{key:<11} = "{value}"')

    was = f'{previous["repo"] or "(none)"} @ {previous["revision"][:12] or "(none)"}'
    note = (
        f"{_ROTATION_COMMENT}{when}: previous pin was {was} — that slice is out of "
        "service and may now be revealed"
    )
    return "\n".join(lines[: head + 1] + [note] + body + lines[end:]) + "\n", previous


def apply_contract(
    contract: Path, repo: str, revision: str, when: str, max_shards: int | None,
) -> tuple[dict, Path]:
    """Update the contract in place, leaving a timestamped backup beside it."""
    text = contract.read_text()
    updated, previous = update_private_source(
        text, repo, revision, when, max_shards=max_shards
    )
    backup = contract.with_suffix(contract.suffix + f".bak-{when}")
    shutil.copy2(contract, backup)
    contract.write_text(updated)
    return previous, backup


# --- "has this period already gone out?" -------------------------------------


def hf_repo_exists(repo: str, token: str | None) -> bool | None:
    """True / False / None when the hub could not answer (network, auth).

    The three-valued answer is the point: "I could not check" must not be read as
    "it does not exist", because publishing then uploads a second slice into a
    repo validators may be reading right now.
    """
    if not token:
        return None
    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    try:
        HfApi(token=token).repo_info(repo_id=repo, repo_type="dataset")
        return True
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404 or type(exc).__name__ == "RepositoryNotFoundError":
            return False
        return None  # unreachable hub, bad token, anything else: unknown


def ledger_has(ledger: Path, repo: str) -> bool:
    """Did a previous run of this script already publish that repo?"""
    if not ledger.is_file():
        return False
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if json.loads(line).get("repo") == repo:
                return True
        except json.JSONDecodeError:
            continue
    return False


def already_rotated(
    repo: str, contract: Path | None, ledger: Path, exists_fn=None,
    token: str | None = None,
) -> tuple[str | None, bool]:
    """Return (reason this period is already out, hub-was-reachable).

    Three independent checks because each covers a different way a rerun can
    happen: the contract already pins it (the feed is live), our own ledger
    recorded it (the loop restarted), or the hub already holds it (someone else
    ran the rotation). Any hit means skip — re-publishing would push a second
    slice into a repo validators are reading right now.
    """
    if contract is not None and contract.is_file() and read_pinned_repo(contract) == repo:
        return f"{contract} already pins {repo}", True
    if ledger_has(ledger, repo):
        return f"{ledger} already records a rotation for {repo}", True
    exists = (exists_fn or hf_repo_exists)(repo, token)
    if exists:
        return f"dataset repo {repo} already exists on HuggingFace", True
    return None, exists is not None


# --- report -------------------------------------------------------------------


def write_build(papers: list[dict], out_dir: Path, n_shards: int, plan: dict,
                stats, repo: str, period: str) -> tuple[list[Path], Path]:
    """Write the shards and the manifest that documents them.

    Always together: shards on disk without their plan would be an artifact nobody
    can reproduce, and the seed in that manifest is the whole determinism story.
    """
    shard_paths = write_shards(papers, out_dir, n_shards)
    manifest = {**plan, "kept": stats.kept, "per_source": stats.per_source,
                "per_domain": stats.per_domain, "vocab": stats.vocab.name,
                "shards": len(shard_paths), "repo": repo, "period": period}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return shard_paths, manifest_path


def emit(report: dict, json_out: Path | None) -> None:
    """The machine-readable half: JSON on stdout, and optionally to a file."""
    text = json.dumps(report, indent=2, sort_keys=True)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(text + "\n")
    print(text, flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--days", type=int, default=7, help="Window length (default 7).")
    ap.add_argument("--to-date", default=None, help="ISO end date (default: today).")
    ap.add_argument("--from-date", default=None)
    ap.add_argument("--target", type=int, default=2500)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument(
        "--min-papers", type=int, default=DEFAULT_MIN_PAPERS,
        help=f"Refuse to publish a slice below this many usable papers "
             f"(default {DEFAULT_MIN_PAPERS}).",
    )
    ap.add_argument("--out", type=Path, default=Path("holdout"),
                    help="Root for per-period build directories (default: holdout/).")
    ap.add_argument("--org", default=DEFAULT_ORG)
    ap.add_argument("--repo", default=None, help="Override the dated repo id.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Plan seed (default: OS random, recorded in the manifest).")
    ap.add_argument(
        "--domain", action="append", default=[], metavar="ID",
        help="OpenAlex domain ids to harvest; 'all' (default) takes every domain.",
    )
    ap.add_argument("--vocab", default="general", choices=sorted(VOCABULARIES))
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--max-shards", type=int, default=4,
                    help="max_shards to write into the [private_source] block.")
    ap.add_argument("--mailto", default="epago@example.com")
    ap.add_argument("--contract", type=Path, default=None,
                    help="Chain contract whose [private_source] block to update. "
                         "Rewritten only with --apply (backup + old pin kept in a comment).")
    ap.add_argument("--json-out", type=Path, default=None, help="Write the JSON report here too.")
    ap.add_argument("--ledger", type=Path, default=None,
                    help="Rotation ledger JSONL (default: <out>/rotations.jsonl).")
    ap.add_argument("--force", action="store_true",
                    help="Rotate even if this period already looks published.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true",
                      help="Actually publish and (with --contract) pin. Off by default.")
    mode.add_argument("--dry-run", action="store_true", default=False,
                      help="Default: harvest and report, publish nothing.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply = bool(args.apply)

    domains = parse_domains(args.domain)
    to_date = args.to_date or dt.date.today().isoformat()
    end = dt.date.fromisoformat(to_date)
    from_date = args.from_date or (end - dt.timedelta(days=args.days)).isoformat()
    period = period_tag(end)
    repo = args.repo or default_repo(domains, end, args.org)
    out_dir = args.out / period
    ledger = args.ledger or (args.out / "rotations.jsonl")

    report: dict = {
        "status": "error",
        "applied": apply,
        "period": period,
        "repo": repo,
        "revision": None,
        "private": True,
        "window": {"from": from_date, "to": to_date},
        "domains": list(domains),
        "domain_names": [DOMAINS[d][0] for d in domains],
        "vocab": args.vocab,
        "min_papers": args.min_papers,
        "kept": 0,
        "shards": 0,
        "seed": None,
        "out_dir": str(out_dir),
        "manifest": None,
        "block_file": None,
        "ledger": str(ledger),
        "contract": None,
        "private_source": {
            "repo": repo, "revision": None,
            "text_column": args.text_column, "max_shards": args.max_shards,
        },
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    def fail(code: int, reason: str, status: str = "error") -> int:
        report["status"] = status
        report["reason"] = reason
        log(f"error: {reason}")
        emit(report, args.json_out)
        return code

    log(f"rotation {period}: {repo}  ({'APPLY — publishes' if apply else 'dry-run'})")

    # 1. Token first. A harvest that ends in "no token" has burned the window.
    token = read_token()
    if apply and not token:
        return fail(EXIT_CONFIG, TOKEN_HELP)

    # ...and the same for the contract: a rewrite that cannot work must fail now,
    # not after a fresh slice has already been published.
    if args.contract is not None:
        if not args.contract.is_file():
            return fail(EXIT_CONFIG, f"contract {args.contract} not found")
        try:
            update_private_source(args.contract.read_text(), "probe", "probe", "probe")
        except ValueError as exc:
            return fail(EXIT_CONFIG, str(exc))

    # 2. Never publish a period twice, never overwrite a live feed.
    reason, hub_ok = already_rotated(repo, args.contract, ledger, token=token)
    if reason and not args.force:
        report["status"] = "skipped"
        report["reason"] = reason
        log(f"skipped: {reason} (pass --force to rotate anyway)")
        emit(report, args.json_out)
        return EXIT_OK
    if apply and not hub_ok and not args.force:
        return fail(
            EXIT_CONFIG,
            f"could not reach HuggingFace to check whether {repo} already exists; "
            "refusing to publish blind over a possibly-live feed. Re-run when the hub "
            "is reachable, or pass --force.",
        )

    # 3. Harvest the fresh window. The seed is recorded, so this is reproducible.
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    plan = plan_harvest(seed, from_date, to_date, domains)
    report["seed"] = seed
    log(f"  seed={seed} window {from_date}..{to_date} sources={plan['weights']}")
    log(f"  themes={plan['themes']}")

    stats = Stats(vocab=VOCABULARIES[args.vocab])
    papers = run_plan(plan, args.target, args.mailto, stats)
    report["kept"] = stats.kept
    report["api_failures"] = stats.api_failures
    report["per_source"] = stats.per_source
    log(f"  kept {stats.kept}/{args.target}  per-source {stats.per_source}  "
        f"api failures {stats.api_failures}")

    # 4. A throttled week is a failed harvest, not a small one.
    if stats.kept < args.min_papers:
        if papers:  # keep the evidence: what did come back is on disk to inspect
            _, manifest_path = write_build(papers, out_dir, args.shards, plan, stats,
                                           repo, period)
            report["manifest"] = str(manifest_path)
        return fail(
            EXIT_STARVED,
            f"harvest yielded {stats.kept} usable papers, below the {args.min_papers} "
            f"minimum ({stats.api_failures} API failures) — refusing to publish a "
            "starved slice; the current feed stays live until the next run.",
            status="starved",
        )

    shard_paths, manifest_path = write_build(papers, out_dir, args.shards, plan, stats,
                                             repo, period)
    report["shards"] = len(shard_paths)
    report["manifest"] = str(manifest_path)
    log(f"  wrote {len(shard_paths)} shards + manifest.json to {out_dir}")

    # 5. Publish — private, verified private by publish() itself.
    revision = None
    if apply:
        try:
            revision = publish(repo, shard_paths, token)
        except Exception as exc:  # network, auth, or the public-repo refusal
            return fail(EXIT_PUBLISH, f"publish failed: {exc}")
        report["revision"] = revision
        report["private_source"]["revision"] = revision
        log(f"  published PRIVATE dataset {repo} @ {revision}")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as fh:
            fh.write(json.dumps({"period": period, "repo": repo, "revision": revision,
                                 "kept": stats.kept, "seed": seed,
                                 "published_at": report["generated_at"]}) + "\n")

    # 6. The block: always written to disk, so an operator can apply it by hand.
    block = render_block(repo, revision, args.text_column, args.max_shards)
    block_path = out_dir / "private_source.toml"
    block_path.write_text(block)
    report["block_file"] = str(block_path)
    report["block"] = block

    # 7. The contract. Rewritten only under --apply; otherwise just reported.
    if args.contract is not None:
        entry = {"path": str(args.contract), "updated": False, "backup": None,
                 "previous": None}
        if apply:
            previous, backup = apply_contract(   # pre-flighted above
                args.contract, repo, revision or "", report["generated_at"][:10],
                args.max_shards,
            )
            entry.update(updated=True, backup=str(backup), previous=previous)
            log(f"  pinned into {args.contract} (backup {backup})")
            log("  commit the contract so every validator reads the new pin")
        else:
            entry["previous"] = {"repo": read_pinned_repo(args.contract), "revision": ""}
            log(f"  would pin into {args.contract} (dry-run: not written)")
        report["contract"] = entry

    report["status"] = "published" if apply else "dry-run"
    emit(report, args.json_out)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
