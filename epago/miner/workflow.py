"""Reference miner pipeline as composable steps.

The pipeline is: resolve the current king -> materialize its snapshot ->
prepare a challenger folder -> (train, which is entirely the miner's business)
-> preflight the exact validator intake checks locally -> submit via timelock
commit-reveal.

Nothing here trains a model. The protocol never inspects *how* an improvement
was produced — any recipe is legal as long as the checkpoint passes intake and
wins the duel. What this module guarantees is that a miner never burns a
submission (and its bond) on a rejection that was locally preventable.
"""

from __future__ import annotations

import importlib
import json
import logging
import shutil
from pathlib import Path

from epago import constants
from epago.chain.client import ChainClient
from epago.config import EpagoConfig
from epago.core.quorum import evaluate_quorum
from epago.core.reveal import build_reveal
from epago.core.types import EvaluatorInfo, ModelRef, VerdictDecision
from epago.model.store import materialize_model
from epago.model.validation import (
    exact_copy_of_king,
    validate_challenger_folder,
    validate_repo_name,
)

logger = logging.getLogger(__name__)


class NoKingError(RuntimeError):
    """No king could be resolved from chain state or the genesis pin."""


def _is_placeholder_digest(digest: str) -> bool:
    """True for the unpinned all-zeros digests shipped in a template chain.toml."""
    hexpart = digest.split(":", 1)[-1]
    return set(hexpart) == {"0"}


def resolve_king_from_chain(chain: ChainClient, cfg: EpagoConfig) -> ModelRef:
    """Derive the current king from on-chain ``ev3``/coronation state.

    This is a miner-side convenience scan: it applies the same quorum rule as
    :mod:`epago.core.quorum` over all published verdicts, treating every
    validator-permit neuron's stake as active. The authoritative coronation
    derivation lives validator-side; miners who want the fully resolved view
    (including reign history and queue state) can instead read the public
    dashboard state file that validators publish, via
    :func:`load_public_state`.

    Falls back to the genesis seed pinned in ``chain.toml`` when no challenger
    has ever been crowned. Raises :class:`NoKingError` when there is no
    coronation on-chain *and* the seed digest is still an unpinned placeholder.
    """
    # The authority's ``ek1`` pointer is the direct answer when one is
    # configured — it names the king the validators are actually dueling
    # against, without replaying every verdict.
    authority = cfg.chain.king_authority_hotkey
    if authority:
        pointer = chain.read_king_pointer(authority)
        if pointer is not None:
            return pointer.ref

    verdicts = chain.read_verdicts()
    evaluators = [
        EvaluatorInfo(hotkey=n.hotkey, stake=n.stake, last_verdict_block=chain.current_block())
        for n in chain.neurons()
        if n.validator_permit
    ]
    accepted_digests = {v.challenger_digest for v in verdicts if v.decision is VerdictDecision.ACCEPT}
    # Newest accepted digest first: the latest coronation is the current king.
    candidates = sorted(
        accepted_digests,
        key=lambda d: max(v.block for v in verdicts if v.challenger_digest == d),
        reverse=True,
    )
    revealed = {r.challenger.digest: r.challenger for r in chain.read_revealed_submissions(0)}
    for digest in candidates:
        status = evaluate_quorum(digest, verdicts, evaluators, cfg.quorum)
        if not status.reached:
            continue
        ref = revealed.get(digest)
        if ref is None:
            logger.warning(
                "crowned digest %s has no matching e2 reveal in current chain state; "
                "skipping (read the public dashboard state file for the resolved king)",
                digest,
            )
            continue
        return ref

    seed_digest = cfg.seed.seed_digest
    if _is_placeholder_digest(seed_digest):
        raise NoKingError(
            "no coronation found on-chain and the [seed] digest in "
            f"{cfg.path} is an unpinned placeholder ({seed_digest!r}). "
            "Either the subnet has not run genesis yet (operators: pin the seed with "
            "scripts/seed_genesis.py), or you are pointed at the wrong netuid/network. "
            "Miners can also read the current king from the public dashboard state file."
        )
    return ModelRef(repo=cfg.chain.seed_repo, digest=seed_digest)


def fetch_king(cfg: EpagoConfig, state_dir: Path, king_ref: ModelRef) -> tuple[ModelRef, Path]:
    """Materialize the king's pinned snapshot into the local miner cache.

    ``king_ref`` is injected rather than resolved here so callers choose their
    source of truth: :func:`resolve_king_from_chain`, the public dashboard
    state file, or an explicit ``--king-repo/--king-digest`` override.
    Idempotent: an already-verified snapshot returns without network access.
    """
    cache_dir = Path(state_dir).expanduser() / "models"
    path = materialize_model(king_ref, cache_dir)
    return king_ref, path


def prepare_challenger(king_dir: Path, out_dir: Path) -> Path:
    """Copy the king snapshot into ``out_dir`` as the training starting point.

    Prepares and validates only — training is the miner's business. Internal
    cache markers and dotfiles are not copied. Note that submitting the
    unmodified copy is pointless: byte-identical challengers are rejected at
    intake (see :func:`preflight`).
    """
    king_dir = Path(king_dir)
    out_dir = Path(out_dir)
    if not (king_dir / "config.json").exists():
        raise FileNotFoundError(f"{king_dir} does not look like a model snapshot (no config.json)")
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(king_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(king_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return out_dir


def preflight(
    challenger_dir: Path,
    king_dir: Path,
    repo: str,
    hotkey: str,
    cfg: EpagoConfig,
    *,
    format_probe: bool = False,
) -> list[str]:
    """Run the EXACT intake checks a validator will run, before submitting.

    Returns a list of human-readable ``code: detail`` problems; empty means the
    submission would clear validator intake. Checks mirror validator intake
    one-for-one: repo-name policy, file hygiene, config lock, size cap, and the
    exact-copy-of-king rejection. With ``format_probe=True`` a local format
    probe additionally runs when :mod:`epago.eval` is importable (optional —
    it needs the eval extra installed).
    """
    problems: list[str] = []
    failure = validate_repo_name(repo, hotkey, cfg)
    if failure is not None:
        problems.append(f"{failure.code}: {failure.detail}")
    for f in validate_challenger_folder(Path(challenger_dir), Path(king_dir), cfg):
        problems.append(f"{f.code}: {f.detail}")
    if exact_copy_of_king(Path(challenger_dir), Path(king_dir)):
        problems.append(
            "exact_copy: challenger safetensors are byte-identical to the king; "
            "copies are rejected at intake"
        )
    if format_probe:
        problems.extend(_run_format_probe(Path(challenger_dir)))
    return problems


def _run_format_probe(challenger_dir: Path) -> list[str]:
    """Optional local format probe via a late import of :mod:`epago.eval`.

    Silently skipped when the eval extra (torch/vllm) is not installed — the
    static intake checks above never require heavy dependencies.
    """
    try:
        eval_mod = importlib.import_module("epago.eval")
    except ImportError as exc:
        logger.info("epago.eval not importable (%s); skipping local format probe", exc)
        return []
    probe = None
    for name in ("format_probe", "run_format_probe", "probe_format"):
        probe = getattr(eval_mod, name, None)
        if callable(probe):
            break
    if not callable(probe):
        logger.info("epago.eval exposes no format probe entry point; skipping")
        return []
    try:
        result = probe(challenger_dir)
    except Exception as exc:  # noqa: BLE001 - a crashing probe is itself a finding
        return [f"format_probe: probe raised {type(exc).__name__}: {exc}"]
    if result is None or result is True:
        return []
    if result is False:
        return ["format_probe: challenger failed the local format probe"]
    return [f"format_probe: {item}" for item in result]


def submit(
    chain: ChainClient, king_digest: str, challenger: ModelRef, hotkey: str | None = None
) -> str:
    """Build the ``e2`` reveal payload and submit it via timelock commit-reveal.

    The payload binds the challenger to the king generation it was trained
    against (``king_digest``): if the king changes before the duel runs, the
    submission goes stale rather than dueling the wrong parent. Returns the
    payload string that will appear on-chain at reveal.

    ``hotkey`` is accepted and ignored. Authorship is now the hotkey that signs
    the commitment, which the chain records; carrying it in the payload let
    anyone submit under someone else's identity, and intake keys cooldowns and
    emission off authorship.
    """
    del hotkey
    payload = build_reveal(king_digest, challenger)
    chain.publish_reveal(payload, constants.BLOCKS_UNTIL_REVEAL)
    return payload


# --- public state helpers (dashboard file published by validators) -----------

_DEFAULT_PUBLIC_STATE_FILE = "dashboard/dashboard.json"


def fetch_public_state(
    publisher_repo: str,
    filename: str = _DEFAULT_PUBLIC_STATE_FILE,
    store: object = None,
) -> dict | list | None:
    """Fetch one published JSON file from a validator's the object store key namespace.

    ``publisher_repo`` is the key prefix a validator syncs its state under in the
    shared bucket (see :mod:`epago.publishing`); ``filename`` is a path within
    it, e.g. ``dashboard/dashboard.json`` or ``audit/audit.jsonl``'s released
    siblings. Returns the parsed document, or ``None`` when the object is not
    reachable. boto3 (the ``epago[chain]`` extra) is import-guarded.
    """
    import tempfile

    try:
        from epago.model.objectstore import ObjectStore
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is not installed; `pip install 'epago[chain]'` to read a "
            "validator's published state from the object store"
        ) from exc
    store = store if store is not None else ObjectStore()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "state.json"
            store.get_object(f"{publisher_repo}/{filename}", dest)
            return json.loads(dest.read_text())
    except Exception as exc:  # noqa: BLE001 - offline/missing publishers end the same way
        logger.info("could not fetch %s from publisher %s: %s", filename, publisher_repo, exc)
        return None


def load_public_state(source: str) -> dict:
    """Load the public dashboard state file.

    ``source`` is a local JSON path, an http(s) URL, or a validator publisher
    key namespace as ``s3://<namespace>/<name>[/<path>]`` (the path
    defaults to ``dashboard/dashboard.json``) — so ``miner status
    --state s3://some-validator/epago-state`` reads the public mirror
    directly.
    """
    if source.startswith("s3://"):
        parts = source.removeprefix("s3://").strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"expected s3://<namespace>/<name>[/<path>], got {source!r}")
        repo = "/".join(parts[:2])
        filename = "/".join(parts[2:]) or _DEFAULT_PUBLIC_STATE_FILE
        doc = fetch_public_state(repo, filename)
        if doc is None:
            raise FileNotFoundError(f"{filename} not available from publisher {repo}")
        return doc
    if source.startswith(("http://", "https://")):
        import httpx

        resp = httpx.get(source, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    return json.loads(Path(source).expanduser().read_text())


def find_hotkey_entries(state: object, hotkey: str) -> list[dict]:
    """Collect every dict in a nested state document that mentions ``hotkey``.

    The dashboard state schema is validator-owned and may evolve; this scan is
    deliberately schema-agnostic so ``miner status`` keeps working across
    releases.
    """
    found: list[dict] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if any(v == hotkey for v in node.values() if isinstance(v, str)):
                found.append(node)
            else:
                for v in node.values():
                    visit(v)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(state)
    return found


def make_chain_client(
    cfg: EpagoConfig,
    wallet_name: str | None = None,
    wallet_hotkey: str | None = None,
    *,
    mock: bool = False,
) -> ChainClient:
    """Construct a chain client, import-guarding the Bittensor SDK.

    ``mock=True`` returns an in-memory :class:`MockChainClient` for local
    rehearsal; otherwise the Bittensor SDK and a local wallet are required.
    """
    if mock:
        from epago.chain.client import MockChainClient

        return MockChainClient()
    try:
        import bittensor as bt
    except ImportError as exc:
        raise RuntimeError(
            "the bittensor SDK is not installed; install the chain extra with "
            "`pip install 'epago[chain]'` (or pass --mock for a local dry run)"
        ) from exc
    from epago.chain.client import BittensorChainClient

    wallet = bt.Wallet(name=wallet_name or "default", hotkey=wallet_hotkey or "default")
    return BittensorChainClient(wallet, netuid=cfg.chain.netuid, network=cfg.chain.network)
