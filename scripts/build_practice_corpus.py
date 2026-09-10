#!/usr/bin/env python
"""Practice corpus builder: a public training library that shares no paper with the exam.

Miners need a library to learn the research procedure on -- search, open, read,
answer -- with the same tools the validator gives a model. The exam corpus
cannot be that library: with the exam's own papers and the public minter, a
miner could mint much of the sealed pool ahead of time and learn answers instead
of research. So the practice library is built from a separate, larger source
with every paper the exam could use removed.

One paper often reaches two indexes as two records: a DOI in one, a PubMed id
in the other, and a title that differs only by leftover markup ("i Citrus
bergamia /i"), an HTML entity ("&#x3b1;" for "α"), spacing ("1H" against
"1 H") or a dropped subtitle. So a paper is removed when an excluded source
holds any of:

  doc_id          the same doc id (same title and text)
  identifier      the same DOI, OpenAlex work id or PubMed id, read from the url
  title           the same title, once markup, entities, case, accents,
                  punctuation and spacing are folded away
  abstract        the same abstract, folded the same way
  abstract_start  the same first 160 folded characters of the abstract
  abstract_end    the same last 160

A shared opening or ending counts only when exactly one excluded paper carries
it and no other source paper does. Anything more common is boilerplate -- a
journal's evidence-level note, an author-roles statement -- and matching on it
would remove unrelated papers. Source papers the first four keys already remove
are not counted: an exam paper's own second record in the source would otherwise
make the stretch it shares with a third record look like boilerplate. Past
that, every check errs toward removing: too much costs a few practice papers,
too little leaks an exam paper.

The kept papers then go through scripts/build_corpus.py -- the builder the exam
corpus came from -- so the format, the search index and task minting behave the
same. The build re-reads what it wrote and refuses to finish if any excluded
paper survived.

Writes into --out:

  corpus.db          the library, in the exam corpus's format
  entities-v1.json   the entity index scripts/mint_intersections.py mints from
  manifest.json      counts, digests, sources, and what was removed and why

Deterministic: the same inputs give byte-identical outputs and digests.

Usage:

    python scripts/build_practice_corpus.py \\
        --source corpus-science-500k/corpus.db \\
        --exclude-corpus data/corpus-science-big/corpus.db \\
        --exclude-parquet <private holdout snapshot directory> \\
        --out practice-v1
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epago.taskgen.entities import EntityIndex

MANIFEST_FORMAT = "epago-practice-1"
ENTITIES = "entities-v1.json"
#: Abstracts shorter than this, once folded, are too short to identify a paper.
MIN_ABSTRACT_CHARS = 200
#: How much of a folded abstract's opening or ending identifies a paper.
EDGE_CHARS = 160

# The exam corpus's own builder, loaded from beside this file. It is registered
# before it runs because its dataclasses look their module up in sys.modules.
_spec = importlib.util.spec_from_file_location(
    "build_corpus_script", Path(__file__).with_name("build_corpus.py")
)
build_corpus_script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = build_corpus_script
_spec.loader.exec_module(build_corpus_script)

_DOI = re.compile(r"(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)
_OPENALEX = re.compile(r"openalex\.org/(w\d+)", re.IGNORECASE)
_PMID = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|europepmc\.org/(?:abstract|article)/med/)(\d+)",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]*>")
# What an inline tag leaves once an index strips its angle brackets:
# "<i>Citrus</i>" arrives as "i Citrus /i".
_TAG_REMNANT = re.compile(r"(?<![^\W_])/?(?:i|b|u|em|sub|sup|sc|scp)(?![^\W_])", re.IGNORECASE)


class PracticeCorpusError(RuntimeError):
    pass


def id_key(url: str) -> str | None:
    """The paper's identifier from its url: a DOI, an OpenAlex work id or a PubMed id."""
    text = (url or "").strip()
    if match := _DOI.search(text):
        return "doi:" + match.group(1).lower().rstrip(".,;/")
    if match := _OPENALEX.search(text):
        return "openalex:" + match.group(1).lower()
    if match := _PMID.search(text):
        return "pmid:" + match.group(1)
    return None


def _plain(text: str) -> str:
    """Text without markup: entities decoded, tags and their stripped remnants gone."""
    return _TAG_REMNANT.sub(" ", _TAG.sub(" ", html.unescape(text or "")))


def _fold(text: str) -> str:
    """Letters and digits only, without accents or case."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return re.sub(r"[\W_]+", "", text)


def title_key(title: str) -> str:
    """A title with markup, entities, case, accents, punctuation and spacing folded away."""
    return _fold(_plain(title))


def abstract_key(text: str) -> str:
    """The abstract -- a paper's text after its title -- folded like a title; empty if short."""
    body = text.split("\n\n", 1)[1] if "\n\n" in (text or "") else ""
    folded = _fold(_plain(body))
    return folded if len(folded) >= MIN_ABSTRACT_CHARS else ""


def _digest(tag: str, text: str) -> bytes:
    return hashlib.blake2b(f"{tag}:{text}".encode(), digest_size=16).digest()


def _edges(abstract: str) -> tuple[bytes, bytes]:
    return _digest("start", abstract[:EDGE_CHARS]), _digest("end", abstract[-EDGE_CHARS:])


def iter_corpus(path: Path) -> Iterator[tuple[str, str, str, str, str]]:
    """``(doc_id, url, title, text, category)`` for every paper, in doc-id order."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        yield from db.execute("SELECT doc_id, url, title, text, category FROM docs ORDER BY doc_id")
    finally:
        db.close()


def count_edges(corpora: list[Path], exclusions: Exclusions | None = None) -> Counter[bytes]:
    """How many papers across ``corpora`` carry each abstract opening and ending.

    With ``exclusions``, papers its first four keys already remove are left
    out: an exam paper's own second record in the source must not make the
    stretch it shares with a third record look like boilerplate. Measured on
    the real build, that let four such records through.
    """
    counts: Counter[bytes] = Counter()
    for corpus in corpora:
        for doc_id, url, title, text, _category in iter_corpus(corpus):
            abstract = abstract_key(text)
            if not abstract:
                continue
            if exclusions is not None and exclusions.basic_reason(doc_id, url, title, abstract):
                continue
            counts.update(_edges(abstract))
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def _label(path: Path) -> str:
    """A short name for a source, without this machine's directory layout."""
    path = Path(path)
    return path.name if path.is_dir() else f"{path.parent.name}/{path.name}"


@dataclass
class Exclusions:
    """Every key an excluded paper can be recognised by."""

    doc_ids: set[str] = field(default_factory=set)
    ids: set[str] = field(default_factory=set)
    titles: set[str] = field(default_factory=set)
    abstracts: set[bytes] = field(default_factory=set)
    #: How many excluded papers carry each abstract opening and ending.
    edges: Counter[bytes] = field(default_factory=Counter)
    sources: list[dict] = field(default_factory=list)

    def add(self, doc_id: str | None, url: str, title: str, text: str = "") -> None:
        if doc_id:
            self.doc_ids.add(doc_id)
        if key := id_key(url):
            self.ids.add(key)
        if key := title_key(title):
            self.titles.add(key)
        if abstract := abstract_key(text):
            self.abstracts.add(_digest("abstract", abstract))
            self.edges.update(_edges(abstract))

    def basic_reason(self, doc_id: str, url: str, title: str, abstract: str) -> str | None:
        """Removal by doc id, identifier, title or whole abstract (``abstract`` already folded)."""
        if doc_id in self.doc_ids:
            return "doc_id"
        key = id_key(url)
        if key and key in self.ids:
            return "identifier"
        key = title_key(title)
        if key and key in self.titles:
            return "title"
        if abstract and _digest("abstract", abstract) in self.abstracts:
            return "abstract"
        return None

    def reason(
        self, doc_id: str, url: str, title: str, text: str, source_edges: Counter[bytes]
    ) -> str | None:
        """Why a paper must be removed, or None when no excluded source holds it.

        ``source_edges`` comes from :func:`count_edges` over the papers being
        filtered, so a stretch they share with each other reads as boilerplate
        rather than as this paper's fingerprint.
        """
        abstract = abstract_key(text)
        if why := self.basic_reason(doc_id, url, title, abstract):
            return why
        if not abstract:
            return None
        for why, edge in zip(("abstract_start", "abstract_end"), _edges(abstract)):
            if self.edges.get(edge) == 1 and source_edges.get(edge) == 1:
                return why
        return None

    def add_corpus(self, path: Path) -> None:
        count = 0
        for doc_id, url, title, text, _category in iter_corpus(path):
            self.add(doc_id, url, title, text)
            count += 1
        self.sources.append(
            {"kind": "corpus", "name": _label(path), "papers": count,
             "sha256": build_corpus_script.corpus_digest(Path(path))}
        )

    def add_parquet(self, path: Path) -> None:
        import pyarrow.parquet as pq

        path = Path(path)
        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        if not files:
            raise PracticeCorpusError(f"no parquet files at {path}")
        combined = hashlib.sha256()
        count = 0
        for shard in files:
            combined.update(shard.name.encode())
            combined.update(bytes.fromhex(_sha256(shard)))
            columns = [c for c in ("url", "title", "text") if c in pq.read_schema(shard).names]
            for row in pq.read_table(shard, columns=columns).to_pylist():
                self.add(None, row.get("url") or "", row.get("title") or "", row.get("text") or "")
                count += 1
        self.sources.append(
            {"kind": "parquet", "name": _label(path), "papers": count,
             "sha256": "sha256:" + combined.hexdigest()}
        )


def write_kept(
    sources: list[Path], exclusions: Exclusions, jsonl: Path, source_edges: Counter[bytes]
) -> Counter[str]:
    """Write every source paper no excluded source holds; count the rest by reason."""
    removed: Counter[str] = Counter()
    with jsonl.open("w") as out:
        for source in sources:
            for doc_id, url, title, text, category in iter_corpus(source):
                if why := exclusions.reason(doc_id, url, title, text, source_edges):
                    removed[why] += 1
                    continue
                out.write(
                    json.dumps(
                        {"url": url, "title": title, "text": text, "category": category},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return removed


def check_disjoint(
    corpus: Path, exclusions: Exclusions, source_edges: Counter[bytes] | None = None
) -> None:
    """Refuse a library that still holds an excluded paper.

    ``source_edges`` should be the counts the filter used; without them the
    corpus's own are taken.
    """
    if source_edges is None:
        source_edges = count_edges([corpus], exclusions)
    survivors = [
        (doc_id, why)
        for doc_id, url, title, text, _category in iter_corpus(corpus)
        if (why := exclusions.reason(doc_id, url, title, text, source_edges))
    ]
    if survivors:
        raise PracticeCorpusError(
            f"{len(survivors)} excluded papers survived the build, e.g. {survivors[:3]}"
        )


def build(
    sources: list[Path],
    exclude_corpora: list[Path],
    exclude_parquet: list[Path],
    out: Path,
    *,
    force: bool = False,
) -> dict:
    """Build the library into ``out`` and return its manifest."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not force:
            raise PracticeCorpusError(f"{out} is not empty; a build must be fresh (use --force)")
        for name in ("corpus.db", ENTITIES, "manifest.json", "papers.jsonl"):
            (out / name).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    exclusions = Exclusions()
    for path in exclude_corpora:
        exclusions.add_corpus(Path(path))
    for path in exclude_parquet:
        exclusions.add_parquet(Path(path))
    if not exclusions.sources:
        raise PracticeCorpusError("nothing to exclude: pass the exam corpus with --exclude-corpus")

    sources = [Path(s) for s in sources]
    source_edges = count_edges(sources, exclusions)
    staged = out / "papers.jsonl"
    removed = write_kept(sources, exclusions, staged, source_edges)
    report = build_corpus_script.build_corpus([staged], [], out / "corpus.db")
    staged.unlink()
    check_disjoint(out / "corpus.db", exclusions, source_edges)

    index = EntityIndex.build(
        (doc_id, text) for doc_id, _url, _title, text, _category in iter_corpus(out / "corpus.db")
    )
    entities_digest = "sha256:" + index.write(out / ENTITIES)
    by_source = Counter(
        category.split(":")[0] for *_rest, category in iter_corpus(out / "corpus.db")
    )

    manifest = {
        "format": MANIFEST_FORMAT,
        "papers": report.accepted,
        "corpus_digest": report.digest,
        "entities_digest": entities_digest,
        "sources": [
            {"name": _label(s), "sha256": build_corpus_script.corpus_digest(s)} for s in sources
        ],
        "excluded": exclusions.sources,
        "removed": dict(sorted(removed.items())),
        "rejected_by_builder": dict(sorted(report.rejected.items())),
        "by_source": dict(sorted(by_source.items())),
        "overlap_with_excluded": 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, action="append", required=True,
                    help="a corpus.db to draw papers from (repeatable)")
    ap.add_argument("--exclude-corpus", type=Path, action="append", default=[],
                    help="a corpus.db whose papers must not appear, e.g. the exam corpus (repeatable)")
    ap.add_argument("--exclude-parquet", type=Path, action="append", default=[],
                    help="parquet file or shard directory with url/title (and optional text) "
                         "columns, e.g. the private holdout (repeatable)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--force", action="store_true", help="replace an earlier build in --out")
    args = ap.parse_args(argv)
    try:
        manifest = build(args.source, args.exclude_corpus, args.exclude_parquet, args.out,
                         force=args.force)
    except PracticeCorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {k: manifest[k] for k in ("papers", "corpus_digest", "entities_digest", "removed")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
