#!/usr/bin/env python
"""Genesis corpus builder: local document collections -> pinned corpus.db.

Takes local inputs only (JSONL files and/or directories of .txt/.md files),
runs a deterministic cleaning pipeline, and writes a fresh
:class:`epago.environment.corpus.SqliteCorpus` snapshot ready for
``scripts/seed_genesis.py`` / chain.toml pinning:

  1. **normalize** — unicode NFKC on title and text, whitespace canonicalized
     (line endings to ``\\n``, horizontal runs to one space, blank-line runs to
     one blank line);
  2. **filter** — length bounds, language-agnostic junk heuristics (too little
     alphabetic content, dominant repeated lines = boilerplate);
  3. **dedup** — exact (sha256 of normalized text) then near-duplicate via a
     documented pure-python minhash-lite (5-word shingles, 64 blake2b min-hashes,
     LSH banding 16x4, signature similarity >= threshold rejects);
  4. **doc_ids** — stable and content-derived (``ep-`` + sha256 prefix of
     title+text), so identical inputs always yield identical ids;
  5. **insert** — documents sorted by doc_id, one transaction, fresh database.

Determinism: no wall clock, no randomness, sorted directory iteration, fixed
hash salts. Byte-identical inputs produce a byte-identical corpus.db and
therefore the same ``corpus_digest`` — which is the whole point, since that
digest is the chain-pinned commitment every validator verifies against.

Usage:
    .venv/bin/python scripts/build_corpus.py \\
        --jsonl dump_a.jsonl --jsonl dump_b.jsonl --text-dir ./notes \\
        --out corpus.db [--limit 500]

JSONL lines are objects with ``url``, ``title``, ``text`` and an optional
``category``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago.environment.corpus import Document, SqliteCorpus  # noqa: E402
from epago.environment.sync import corpus_digest  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# --- filter defaults ---------------------------------------------------------
DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 100_000
# Minimum fraction of alphabetic characters among non-whitespace characters.
# Unicode isalpha() keeps this language-agnostic (CJK, Cyrillic, ... all count).
MIN_ALPHA_FRACTION = 0.50
# A document whose most frequent non-empty line covers more than this fraction
# of its non-empty lines is treated as boilerplate (nav bars, cookie banners,
# templated listings).
MAX_TOP_LINE_FRACTION = 0.30

# --- minhash-lite (near dedup) ----------------------------------------------
# Shingle = 5 consecutive lowercase words. 64 independent min-hashes, each a
# keyed 8-byte blake2b; the fraction of equal signature positions estimates
# the Jaccard similarity of the two shingle sets. LSH banding (16 bands of 4
# rows) selects candidate pairs so the pass stays near-linear.
SHINGLE_WORDS = 5
NUM_HASHES = 64
LSH_BANDS = 16
LSH_ROWS = NUM_HASHES // LSH_BANDS
DEFAULT_NEAR_DUP_SIMILARITY = 0.85


@dataclass(frozen=True, slots=True)
class RawDoc:
    """One input document before the pipeline (source kept for diagnostics)."""

    source: str
    url: str
    title: str
    text: str
    category: str = ""


@dataclass(slots=True)
class BuildReport:
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    digest: str = ""
    out_path: Path = Path()

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())


# --- normalization ------------------------------------------------------------


def normalize_text(text: str) -> str:
    """NFKC + canonical whitespace: one space inside lines, one blank line max."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue  # collapse blank-line runs; drop leading blanks
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def normalize_title(title: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", title).split())


# --- junk heuristics -----------------------------------------------------------


def junk_reason(text: str, min_chars: int, max_chars: int) -> Optional[str]:
    """Return a rejection reason for normalized text, or None if it passes."""
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return "too_short"
    alpha = sum(1 for ch in non_space if ch.isalpha())
    if alpha / len(non_space) < MIN_ALPHA_FRACTION:
        return "low_alpha"
    lines = [line for line in text.split("\n") if line]
    if len(lines) >= 4:
        top = Counter(lines).most_common(1)[0][1]
        if top / len(lines) > MAX_TOP_LINE_FRACTION:
            return "boilerplate"
    return None


# --- content ids and hashing ----------------------------------------------------


def content_doc_id(title: str, text: str) -> str:
    """Stable content-derived doc id: same title+text -> same id, always."""
    h = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()
    return f"ep-{h[:16]}"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _shingles(text: str) -> list[bytes]:
    words = text.lower().split()
    if len(words) < SHINGLE_WORDS:
        return [" ".join(words).encode()]
    return [
        " ".join(words[i : i + SHINGLE_WORDS]).encode()
        for i in range(len(words) - SHINGLE_WORDS + 1)
    ]


def minhash_signature(text: str) -> tuple[int, ...]:
    """64 min-hashes over the shingle set, fixed salts, pure python."""
    shingle_set = set(_shingles(text))
    sig: list[int] = []
    for i in range(NUM_HASHES):
        salt = b"epmh" + i.to_bytes(2, "big")
        sig.append(
            min(
                int.from_bytes(hashlib.blake2b(sh, digest_size=8, salt=salt).digest(), "big")
                for sh in shingle_set
            )
        )
    return tuple(sig)


def signature_similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / NUM_HASHES


# --- input readers ---------------------------------------------------------------


def iter_jsonl(path: Path, rejected: Counter) -> Iterator[RawDoc]:
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rejected["invalid_json"] += 1
                continue
            if not isinstance(obj, dict) or not all(k in obj for k in ("url", "title", "text")):
                rejected["missing_fields"] += 1
                continue
            yield RawDoc(
                source=f"{path}:{lineno}",
                url=str(obj["url"]),
                title=str(obj["title"]),
                text=str(obj["text"]),
                category=str(obj.get("category", "")),
            )


def iter_text_dir(root: Path, category: str) -> Iterator[RawDoc]:
    """All .txt/.md files under root, in sorted relative-path order.

    Title is the first non-empty line (markdown heading markers stripped); the
    url records the relative path so the mapping back to inputs is auditable.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".txt", ".md"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        title = ""
        for line in text.split("\n"):
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                title = stripped
                break
        yield RawDoc(
            source=str(path),
            url=f"local://{rel}",
            title=title,
            text=text,
            category=category,
        )


# --- the pipeline ------------------------------------------------------------------


def build_corpus(
    jsonl_files: Iterable[Path],
    text_dirs: Iterable[Path],
    out_path: Path,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    near_dup_similarity: float = DEFAULT_NEAR_DUP_SIMILARITY,
    limit: Optional[int] = None,
    text_dir_category: str = "",
) -> BuildReport:
    """Run the full pipeline and write a fresh corpus snapshot at ``out_path``."""
    report = BuildReport(out_path=out_path)
    seen_text_hashes: set[str] = set()
    seen_doc_ids: set[str] = set()
    lsh_buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    signatures: list[tuple[int, ...]] = []
    accepted: list[Document] = []

    def sources() -> Iterator[RawDoc]:
        for path in jsonl_files:
            yield from iter_jsonl(Path(path), report.rejected)
        for root in text_dirs:
            yield from iter_text_dir(Path(root), text_dir_category)

    for raw in sources():
        if limit is not None and len(accepted) >= limit:
            break
        text = normalize_text(raw.text)
        title = normalize_title(raw.title)
        if not title:  # fall back to the first 80 chars of content
            title = text.replace("\n", " ")[:80].strip()

        reason = junk_reason(text, min_chars, max_chars)
        if reason is not None:
            report.rejected[reason] += 1
            continue

        text_hash = _text_hash(text)
        if text_hash in seen_text_hashes:
            report.rejected["exact_duplicate"] += 1
            continue

        sig = minhash_signature(text)
        candidates: set[int] = set()
        bands = [
            (b, sig[b * LSH_ROWS : (b + 1) * LSH_ROWS])
            for b in range(LSH_BANDS)
        ]
        for band_key in bands:
            candidates.update(lsh_buckets.get(band_key, ()))
        if any(
            signature_similarity(sig, signatures[i]) >= near_dup_similarity
            for i in sorted(candidates)
        ):
            report.rejected["near_duplicate"] += 1
            continue

        doc_id = content_doc_id(title, text)
        if doc_id in seen_doc_ids:  # same title+text with different metadata
            report.rejected["exact_duplicate"] += 1
            continue

        seen_text_hashes.add(text_hash)
        seen_doc_ids.add(doc_id)
        doc_index = len(signatures)
        signatures.append(sig)
        for band_key in bands:
            lsh_buckets.setdefault(band_key, []).append(doc_index)
        accepted.append(
            Document(doc_id=doc_id, url=raw.url, title=title, text=text, category=raw.category)
        )

    accepted.sort(key=lambda d: d.doc_id)
    store = SqliteCorpus.create(out_path)
    store.add_documents(accepted)
    store.close()

    report.accepted = len(accepted)
    report.digest = corpus_digest(out_path)
    return report


# --- CLI ----------------------------------------------------------------------------


@app.command()
def main(
    jsonl: List[Path] = typer.Option(
        [], "--jsonl", help="JSONL file of {url,title,text[,category]} objects (repeatable)."
    ),
    text_dir: List[Path] = typer.Option(
        [], "--text-dir", help="Directory of .txt/.md documents (repeatable, recursive)."
    ),
    out: Path = typer.Option(Path("corpus.db"), help="Output corpus snapshot path."),
    min_chars: int = typer.Option(DEFAULT_MIN_CHARS, help="Minimum normalized text length."),
    max_chars: int = typer.Option(DEFAULT_MAX_CHARS, help="Maximum normalized text length."),
    near_dup_similarity: float = typer.Option(
        DEFAULT_NEAR_DUP_SIMILARITY,
        help="Minhash signature similarity at or above which a doc is a near-duplicate.",
    ),
    limit: Optional[int] = typer.Option(None, help="Stop after N accepted documents (smoke runs)."),
    category: str = typer.Option("", help="Category label applied to --text-dir documents."),
    corpus_repo: str = typer.Option(
        "epago-ai/epago-corpus-snapshot-1", help="Repo the snapshot will be published to."
    ),
    taskgen_release: str = typer.Option("R1", help="Taskgen release tag for the pin lines."),
    force: bool = typer.Option(False, help="Overwrite an existing output file."),
) -> None:
    """Build a fresh, pinned corpus snapshot from local document collections."""
    if not jsonl and not text_dir:
        typer.echo("error: provide at least one --jsonl file or --text-dir directory", err=True)
        raise typer.Exit(code=2)
    for path in jsonl:
        if not path.is_file():
            typer.echo(f"error: --jsonl {path} is not a file", err=True)
            raise typer.Exit(code=2)
    for path in text_dir:
        if not path.is_dir():
            typer.echo(f"error: --text-dir {path} is not a directory", err=True)
            raise typer.Exit(code=2)
    if out.exists():
        if not force:
            typer.echo(f"error: {out} exists; a snapshot must be fresh (use --force to replace)", err=True)
            raise typer.Exit(code=2)
        out.unlink()

    report = build_corpus(
        jsonl,
        text_dir,
        out,
        min_chars=min_chars,
        max_chars=max_chars,
        near_dup_similarity=near_dup_similarity,
        limit=limit,
        text_dir_category=category,
    )

    typer.echo(f"accepted documents: {report.accepted}")
    typer.echo(f"rejected documents: {report.rejected_total}")
    for reason in sorted(report.rejected):
        typer.echo(f"  {reason}: {report.rejected[reason]}")
    typer.echo(f"snapshot: {out}")
    typer.echo(f"corpus_digest: {report.digest}")
    typer.echo("")
    typer.echo("# ---- paste into chain.toml ------------------------------------")
    typer.echo("[eval]")
    typer.echo(f'corpus_repo     = "{corpus_repo}"')
    typer.echo(f'corpus_digest   = "{report.digest}"')
    typer.echo(f'taskgen_release = "{taskgen_release}"')
    typer.echo("# judge_repo / judge_digest: pin separately (see scripts/seed_genesis.py)")


if __name__ == "__main__":
    app()
