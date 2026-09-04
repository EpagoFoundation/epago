"""Chain-contract loader.

``chain.toml`` is the single source of truth for a chain generation. All
components read it through :func:`load_config`; environment variables prefixed
``EPAGO_`` override scalar fields (e.g. ``EPAGO_NETUID=42``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import MISSING, dataclass, field
from pathlib import Path

#: Filename of the genesis chain contract, inside whichever ``chains``
#: directory is in force.
_GENESIS_CONTRACT = "EPAGO-DR-30B.toml"


def _default_config_path() -> Path:
    """Locate the genesis contract in an installed wheel or a source checkout.

    The wheel ships ``chains/`` inside the package, so an installed validator
    finds its own contract; a source checkout has it one level above the
    package instead. Resolving the installed location first means a checkout
    never shadows the contract a wheel was built with.

    Returning a path that may not exist is deliberate: a missing contract must
    surface where it is loaded, naming the file, rather than as an import-time
    failure with no context.
    """
    packaged = Path(__file__).resolve().parent / "chains" / _GENESIS_CONTRACT
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parent.parent / "chains" / _GENESIS_CONTRACT


DEFAULT_CONFIG_PATH = _default_config_path()


@dataclass(frozen=True)
class ChainSection:
    name: str
    netuid: int
    network: str
    seed_repo: str
    repo_pattern: str
    #: Hotkey whose ``ek1`` king pointers every validator follows. In the
    #: current topology one validator scores and the rest audit and mirror its
    #: weight vector, so the throne has to be nameable from chain state alone —
    #: otherwise a validator with an empty state directory can only fall back to
    #: the genesis seed, treats every live challenge as ``stale_parent``, and
    #: never catches up. Empty disables pointer following (single-box testnets).
    king_authority_hotkey: str = ""
    #: Hotkey allowed to publish ``er1`` round starts on chain. When set, any
    #: validator can independently verify round timing from chain data. Empty
    #: disables the on-chain trigger.
    round_authority_hotkey: str = ""
    #: Local API-key round trigger, as an alternative to the on-chain hotkey.
    #: The scoring validator serves ``POST /round/start`` on this address and
    #: opens a round when a request carries the key (``EPAGO_ROUND_API_KEY``).
    #: Trade-off: the exam is still seeded from a chain block hash the owner
    #: cannot choose, so freshness holds — but round *timing* is no longer
    #: chain-visible, so auditors cannot independently check the cadence. Empty
    #: address disables the local trigger.
    round_api_bind: str = ""
    #: Where unallocatable emission goes. Previously the lowest-UID neuron stood
    #: in for a burn address, which is not a burn: UID 0 is an ordinary
    #: registered neuron, and during Phase A that is the entire subnet emission.
    #: Empty falls back to the old behaviour with a startup warning.
    burn_hotkey: str = ""


@dataclass(frozen=True)
class ArchSection:
    module: str
    extra_lock_keys: tuple[str, ...]


@dataclass(frozen=True)
class SeedSection:
    tokenizer_repo: str
    repo_backend: str
    seed_digest: str


@dataclass(frozen=True)
class EvalSection:
    corpus_repo: str
    corpus_digest: str
    taskgen_release: str
    judge_repo: str
    judge_digest: str
    #: A sealed public pool, used when ``taskgen_release`` names one (see
    #: :func:`epago.taskgen.sealed_pool.is_sealed_release`). The digest is the
    #: commitment: it is fixed in the contract before a round opens, so the
    #: exam existed before any challenger's weights were frozen and cannot be
    #: swapped afterwards. Empty for generator-served releases.
    public_pool_path: str = ""
    public_pool_digest: str = ""
    #: The pool's task-id manifest. Pinned here as well as the pool itself
    #: because the manifest is what an auditor actually verifies a round
    #: against while the pool is still in service: without a commitment fixed
    #: before the round, a validator could hand over an id list tailored to the
    #: tasks it wished it had asked. Empty for generator-served releases.
    public_pool_manifest_path: str = ""
    public_pool_manifest_digest: str = ""


@dataclass(frozen=True)
class QuorumSection:
    theta: float
    active_window_duels: int
    verdict_timeout_blocks: int
    bootstrap_min_evaluators: int


@dataclass(frozen=True)
class EmissionsSection:
    king_share: float
    arena_share: float
    reign_halflife_blocks: int
    coronation_bonus_cap: float
    coronation_bonus_blocks: int
    #: The king's share never falls below this. An unchallenged reign bleeds
    #: from ``king_share`` down to here and then stops, so an incumbent always
    #: keeps a defined majority and the arena's share is bounded too.
    king_share_floor: float = 0.85
    #: How long the bleed takes. Reaching the floor in days rather than decaying
    #: asymptotically over a month makes the schedule something a miner can
    #: reason about.
    reign_decay_blocks: int = 21_600  # ~3 days


@dataclass(frozen=True)
class PrivateSourceSection:
    """Automated fresh-document feed for the private pool — replaces the manual
    ``--ingest-dir`` (the R2 fix). An empty ``repo`` disables it and the pool
    falls back to the pinned corpus, so this whole section is optional.

    ``repo`` + ``revision`` pin a large public *dated* dataset (e.g. FineWeb-Edu)
    by an immutable commit hash so a rotation's slice is fresh and later
    auditable; ``max_shards`` bounds the per-rotation download so supply exceeds
    one epoch's demand without materialising the whole dataset.
    """

    repo: str = ""            # e.g. "HuggingFaceFW/fineweb-edu"
    revision: str = ""        # dated snapshot commit hash (freshness + audit pin)
    text_column: str = "text"
    max_shards: int = 4


@dataclass(frozen=True)
class EpagoConfig:
    chain: ChainSection
    arch: ArchSection
    seed: SeedSection
    eval: EvalSection
    quorum: QuorumSection
    emissions: EmissionsSection
    private_source: PrivateSourceSection = field(default_factory=PrivateSourceSection)
    path: Path = field(compare=False, default=DEFAULT_CONFIG_PATH)


def _env_override(section: str, key: str, current):
    raw = os.environ.get(f"EPAGO_{key.upper()}")
    if raw is None:
        raw = os.environ.get(f"EPAGO_{section.upper()}_{key.upper()}")
    if raw is None:
        return current
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _section(cls, name: str, data: dict):
    """Parse a required section. Fields carrying a default may be omitted.

    Keeping defaults optional means adding a field (e.g. ``king_authority_hotkey``)
    does not invalidate every existing ``chain.toml`` on disk.
    """
    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        if f.name in data:
            value = data[f.name]
        elif f.default is not MISSING:
            value = f.default
        else:
            raise KeyError(f"[{name}] is missing required key {f.name!r}")
        if isinstance(value, list):
            value = tuple(value)
        else:
            value = _env_override(name, f.name, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _optional_section(cls, name: str, data: dict):
    """Parse a section whose keys are all optional (fields keep their defaults).

    Used for ``[private_source]``: a chain.toml without it still loads, and each
    field can be set in the toml or via ``EPAGO_<SECTION>_<KEY>`` env for testnets.
    """
    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        current = data[f.name] if f.name in data else f.default
        if isinstance(current, list):
            current = tuple(current)
        else:
            current = _env_override(name, f.name, current)
        kwargs[f.name] = current
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> EpagoConfig:
    cfg_path = Path(path) if path else Path(os.environ.get("EPAGO_CHAIN_TOML", DEFAULT_CONFIG_PATH))
    with open(cfg_path, "rb") as fh:
        raw = tomllib.load(fh)
    cfg = EpagoConfig(
        chain=_section(ChainSection, "chain", raw["chain"]),
        arch=_section(ArchSection, "arch", raw["arch"]),
        seed=_section(SeedSection, "seed", raw["seed"]),
        eval=_section(EvalSection, "eval", raw["eval"]),
        quorum=_section(QuorumSection, "quorum", raw["quorum"]),
        emissions=_section(EmissionsSection, "emissions", raw["emissions"]),
        private_source=_optional_section(
            PrivateSourceSection, "private_source", raw.get("private_source", {})
        ),
        path=cfg_path,
    )
    shares = cfg.emissions
    total = shares.king_share + shares.arena_share
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"emission shares must sum to 1.0, got {total}")
    if not 0.0 < cfg.quorum.theta <= 1.0:
        raise ValueError(f"quorum theta must be in (0, 1], got {cfg.quorum.theta}")
    return cfg
