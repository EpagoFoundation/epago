"""The public half, served from a sealed pool instead of a generator.

The public task set has always been a pure function of ``(seed, release,
corpus)`` so every validator regenerates it byte-identically and an auditor can
rebuild the exam years later. That works only for tasks a program can write.
The task families worth asking are worded by a language model, and no promise
about temperature survives a provider changing hardware or model version, so
they cannot be a pure function of a seed.

They do not need to be. What the protocol actually requires is narrower than
"regenerable":

1. **The exam must not be knowable before a model is frozen.** Otherwise a
   miner trains on it.
2. **A verdict must be checkable afterwards.** Otherwise a validator is trusted
   rather than audited.

A generator satisfies both by construction. A *sealed pool* satisfies both by
sequencing, and it does so with three artifacts rather than one:

**The pool** is every minted task with its answer. Its digest is pinned in the
contract before any duel uses it, so the exam was fixed before the challenger's
weights were. The file itself stays sealed while the pool is in service.

**The manifest** is the pool's task ids and nothing else — no questions, no
answers. Its digest is pinned in the contract too, and the file publishes
immediately, because a list of opaque ids tells a miner nothing it could train
on. This is what makes the pool auditable while still sealed: selection is a
function of the sorted id list alone, so an auditor holding only the manifest
can recompute exactly which tasks a round asked.

**The round file** is the tasks a single round actually asked, published in
full after the verdict. That is what an auditor re-grades against.

Publishing the whole pool after every round would be the obvious design and it
is the wrong one: after one round a miner would hold every question and answer,
and each later round would draw from a set it had memorised. A pool would
survive exactly one round. Splitting ids from contents lets the unasked
remainder stay sealed, so one pool serves many rounds while every round stays
independently checkable.

The security property that changes against a generator: an auditor can no
longer rebuild the exam from a seed alone; it needs the published manifest and
round file. Both are pinned by digests fixed before the round, so a validator
cannot swap them retroactively — but an auditor who cannot obtain them cannot
complete the check, where before it needed nothing but the corpus. That is a
real reduction in independence and is the price of asking questions a program
cannot write.

Selection is deterministic given ``(sorted task ids, seed, n)``. Sorting first
matters — a pool file's line order is an accident of minting, and letting it
affect selection would make two byte-identical pools disagree if either were
ever rewritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epago.core.types import Task, TaskOrigin

#: Wire format tag for a manifest file.
MANIFEST_FORMAT = "eppm1"


class SealedPoolError(RuntimeError):
    """The pool is missing, malformed, or not the one that was committed."""


def pool_digest(raw: bytes) -> str:
    """Digest of the pool file's exact bytes.

    Over the raw file rather than over parsed contents: the commitment must
    cover what was published, not a normalization of it, or two different files
    could satisfy the same commitment.
    """
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _task_from_row(row: dict) -> Task:
    return Task(
        task_id=row["task_id"],
        question=row["question"],
        answer=row["answer"],
        aliases=tuple(row.get("aliases") or ()),
        evidence_doc_ids=tuple(row.get("evidence_doc_ids") or ()),
        masked_doc_ids=tuple(row.get("masked_doc_ids") or ()),
        origin=TaskOrigin.GENERATED_PUBLIC,
        template=row.get("template", "sealed"),
        hops=int(row.get("hops", 1)),
    )


def _task_to_row(task: Task) -> dict:
    return {
        "task_id": task.task_id,
        "question": task.question,
        "answer": task.answer,
        "aliases": list(task.aliases),
        "evidence_doc_ids": list(task.evidence_doc_ids),
        "masked_doc_ids": list(task.masked_doc_ids),
        "template": task.template,
        "hops": int(task.hops),
    }


def load_pool(path: str | Path, expected_digest: str = "") -> tuple[Task, ...]:
    """Read a sealed pool, verifying it against its committed digest.

    An empty ``expected_digest`` skips verification and is meant only for
    minting and local inspection. A duel must always pass the committed digest:
    without it, "the pool the validator used" and "the pool the validator
    committed to" are two different claims.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SealedPoolError(f"cannot read the sealed pool at {path}: {exc}") from exc

    if expected_digest:
        actual = pool_digest(raw)
        if actual.removeprefix("sha256:") != expected_digest.removeprefix("sha256:"):
            raise SealedPoolError(
                f"sealed pool at {path} has digest {actual}, but {expected_digest} "
                "was committed on chain — refusing to duel on it"
            )

    tasks: list[Task] = []
    for line_no, line in enumerate(raw.decode().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            tasks.append(_task_from_row(json.loads(line)))
        except (KeyError, ValueError, TypeError) as exc:
            raise SealedPoolError(f"{path} line {line_no} is not a task: {exc}") from exc

    if not tasks:
        raise SealedPoolError(f"{path} contains no tasks")

    # Selection runs over sorted task ids, and an auditor reproduces it from
    # the manifest's id list alone. Duplicate ids would make those two orderings
    # disagree, so the auditor's recomputation would diverge from the
    # validator's for reasons having nothing to do with honesty.
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise SealedPoolError(
            f"{path} repeats {len(dupes)} task id(s) (e.g. {dupes[0]!r}); ids must be "
            "unique or selection is not reproducible from the manifest"
        )
    return tuple(tasks)


# -- manifest ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Manifest:
    """A pool's task ids, published while the pool itself stays sealed."""

    format: str
    pool_digest: str
    task_ids: tuple[str, ...]

    def canonical_bytes(self) -> bytes:
        """The exact bytes the manifest digest commits to.

        Ids are sorted here rather than trusted from the file so the digest is
        a function of the id *set*, matching how selection uses them.
        """
        return json.dumps(
            {
                "format": self.format,
                "pool_digest": self.pool_digest,
                "task_ids": sorted(self.task_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()

    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_pool(cls, tasks: tuple[Task, ...], digest: str) -> "Manifest":
        return cls(
            format=MANIFEST_FORMAT,
            pool_digest=digest,
            task_ids=tuple(sorted(t.task_id for t in tasks)),
        )


def load_manifest(path: str | Path, expected_digest: str = "") -> Manifest:
    """Read a manifest, verifying it against its committed digest.

    The digest check is what makes a manifest evidence rather than a claim: the
    id list was pinned in the contract before the round opened, so a validator
    cannot hand an auditor a list tailored to the tasks it wishes it had asked.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SealedPoolError(f"cannot read the pool manifest at {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SealedPoolError(f"{path} is not valid JSON: {exc}") from exc
    if str(data.get("format", "")) != MANIFEST_FORMAT:
        raise SealedPoolError(f"{path} is not a {MANIFEST_FORMAT} manifest")
    ids = tuple(str(i) for i in data.get("task_ids") or ())
    if not ids:
        raise SealedPoolError(f"{path} lists no task ids")
    if len(set(ids)) != len(ids):
        raise SealedPoolError(f"{path} repeats task ids; selection would not be reproducible")
    manifest = Manifest(
        format=MANIFEST_FORMAT,
        pool_digest=str(data.get("pool_digest", "")),
        task_ids=ids,
    )
    if expected_digest:
        actual = manifest.digest()
        if actual.removeprefix("sha256:") != expected_digest.removeprefix("sha256:"):
            raise SealedPoolError(
                f"manifest at {path} has digest {actual}, but {expected_digest} was "
                "committed — refusing to verify a round against it"
            )
    return manifest


def write_manifest(manifest: Manifest, path: str | Path) -> str:
    """Write a manifest atomically and return its digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(manifest.canonical_bytes())
    tmp.replace(path)
    return manifest.digest()


# -- selection ---------------------------------------------------------------


def _draw(total: int, seed: int, n: int) -> list[int]:
    """The index draw, in one place.

    Both selection paths route through this: a validator drawing from the full
    pool and an auditor drawing from the manifest's id list must agree exactly,
    and the surest way to guarantee that is to leave them no separate arithmetic
    to drift apart.
    """
    if n <= 0:
        raise SealedPoolError(f"cannot draw {n} tasks")
    if n > total:
        raise SealedPoolError(
            f"only {total} unserved tasks remain, exam needs {n}. A pool must hold "
            "more than one exam, or every round asks the same questions — mint and "
            "commit a fresh pool."
        )
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.choice(total, size=n, replace=False)]


def _eligible(task_ids, exclude) -> list[str]:
    """Sorted ids a round may still draw, with already-served ones removed.

    Rounds are disjoint on purpose. A round publishes its tasks in full so they
    can be re-graded, which makes them training data the moment they land; if a
    later round could draw them again, a challenger trained after that
    publication would answer part of its exam from memory rather than from
    research. Removing served ids costs nothing and closes that gap.

    The exclusion set is not something an auditor has to take on trust: it is
    the union of the ids in the previously published round files, each of them
    pinned by its own verdict's ``public_task_ids_digest``.
    """
    spent = set(exclude or ())
    return sorted(i for i in task_ids if i not in spent)


def select(
    tasks: tuple[Task, ...],
    seed: int,
    n: int,
    exclude: set[str] | frozenset[str] | None = None,
) -> list[Task]:
    """Draw ``n`` tasks, deterministically, from a block-hash seed.

    Sorted by task id before drawing: a pool file's line order is an accident
    of minting, so letting it influence selection would make two byte-identical
    pools disagree if either were rewritten. Sorting makes the draw a function
    of the pool's *contents*.

    The seed comes from a block hash nobody chose, so which tasks are asked is
    unknown even to whoever minted the pool — the property that stops a miner
    training on the exam. ``exclude`` carries the ids earlier rounds already
    published, keeping rounds disjoint (see :func:`_eligible`).
    """
    by_id = {t.task_id: t for t in tasks}
    eligible = _eligible(by_id, exclude)
    return [by_id[eligible[i]] for i in _draw(len(eligible), seed, n)]


def select_ids(
    task_ids: tuple[str, ...] | list[str],
    seed: int,
    n: int,
    exclude: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """The ids :func:`select` would draw, computed without the tasks.

    This is the whole reason a manifest exists. An auditor holding only the id
    list can establish exactly which tasks a round asked and check that against
    the verdict's ``public_task_ids_digest`` — without ever seeing a question
    or answer the pool has not finished serving.
    """
    eligible = _eligible(task_ids, exclude)
    return [eligible[i] for i in _draw(len(eligible), seed, n)]


def is_sealed_release(release: str) -> bool:
    """True when a release name means "served from a sealed pool".

    A naming convention rather than a config flag, so a release name alone says
    how its tasks are produced — a replay reading only an audit record can tell
    which verification path applies without being told separately.
    """
    return release.upper().startswith("POOL")


# -- per-round publication ---------------------------------------------------

#: Wire format tag for a published round file.
ROUND_FORMAT = "eppr1"


def round_stage_name(round_no: int, task_ids_digest: str) -> str:
    """The name a round's published tasks are staged under.

    Carries the round number and the digest the verdict already records, so an
    auditor locates the file from the audit record alone — the same addressing
    the private pool uses for its rotations.
    """
    short = task_ids_digest.removeprefix("sha256:")[:16]
    return f"publicpool-round{int(round_no):06d}-{short}"


def round_payload(
    tasks: list[Task] | tuple[Task, ...],
    *,
    round_no: int,
    task_ids_digest: str,
    pool_digest_value: str,
    manifest_digest: str,
) -> str:
    """The full text of the tasks one round asked, for auditors to re-grade.

    Only the asked tasks. The rest of the pool stays sealed so it can serve
    later rounds — publishing the whole pool after every round would hand a
    miner every question and answer, and the pool would survive exactly one
    round.

    The pool and manifest digests travel with the tasks so a released file is
    self-locating: an auditor can tell which committed pool it came from
    without cross-referencing a contract revision from the same era.

    Sorted by task id, so the file is a function of *which* tasks were asked
    rather than of the order they happened to be drawn in.
    """
    return json.dumps(
        {
            "format": ROUND_FORMAT,
            "round": int(round_no),
            "public_task_ids_digest": task_ids_digest,
            "pool_digest": pool_digest_value,
            "manifest_digest": manifest_digest,
            "tasks": [_task_to_row(t) for t in sorted(tasks, key=lambda t: t.task_id)],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def load_round_file(path: str | Path) -> tuple[Task, ...]:
    """Read a released round file back into tasks.

    Deliberately not digest-checked here: a round file's authority is the
    ``public_task_ids_digest`` already recorded in the verdict, which the
    replay compares against ids recomputed from the manifest. Re-checking a
    digest the file carries about itself would prove nothing.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedPoolError(f"cannot read the round file at {path}: {exc}") from exc
    if str(data.get("format", "")) != ROUND_FORMAT:
        raise SealedPoolError(f"{path} is not a {ROUND_FORMAT} round file")
    try:
        tasks = tuple(_task_from_row(row) for row in data.get("tasks") or ())
    except (KeyError, ValueError, TypeError) as exc:
        raise SealedPoolError(f"{path} holds a malformed task: {exc}") from exc
    if not tasks:
        raise SealedPoolError(f"{path} lists no tasks")
    return tasks
