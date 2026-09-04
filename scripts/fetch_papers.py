#!/usr/bin/env python
"""Corpus source builder: systematic reviews -> the papers they cite.

Reads a Webis-SR4ALL-26 dump, picks reviews that can carry an evaluation
corpus, resolves their reference lists through OpenAlex, and writes two files:

  1. ``--out-papers``   JSONL of ``{url, title, text, category}`` — exactly what
     ``scripts/build_corpus.py`` consumes to produce a pinned ``corpus.db``.
  2. ``--out-reviews``  JSONL of one record per review: its objective, its
     eligibility criteria, and the doc ids of the papers it included. This is
     the label file for task generation — a review *is* a labelled screening
     decision over its own reference list.

Why reviews rather than a random paper dump: a review's reference list is a
curated, topically coherent collection assembled by a domain expert. That makes
every corpus a real search problem with a known answer set, instead of a pile
of unrelated abstracts where retrieval is trivially easy or hopeless.

Determinism: reviews are processed in sorted id order and papers are emitted in
sorted id order, so the same input dump and the same OpenAlex responses produce
a byte-identical papers file — which is what lets ``build_corpus.py`` produce a
stable ``corpus_digest``. OpenAlex content can change under us; the cache makes
a run reproducible once fetched, and ``--cache`` should be kept for any corpus
you intend to pin.

Usage:
    .venv/bin/python scripts/fetch_papers.py \\
        --sr4all sr4all_full.jsonl \\
        --out-papers papers.jsonl --out-reviews reviews.jsonl \\
        --field Medicine --max-reviews 500 --mailto you@example.com
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterator

OPENALEX = "https://api.openalex.org/works"
#: OpenAlex accepts 50 ids per filter expression.
BATCH = 50
#: Polite-pool courtesy delay between batches.
DELAY_S = 0.25
MAX_RETRIES = 4

#: A review only earns a corpus if enough of its references actually resolve to
#: text. Below this a "search the collection" task is not a search task.
MIN_REFS_WITH_ABSTRACT = 10
MIN_ABSTRACT_RATIO = 0.5
#: Abstracts shorter than this are stubs ("No abstract available.") and add
#: noise without adding anything findable.
MIN_ABSTRACT_CHARS = 200


@dataclass
class Stats:
    reviews_seen: int = 0
    reviews_kept: int = 0
    reviews_rejected: dict[str, int] = dc_field(default_factory=dict)
    papers_requested: int = 0
    papers_resolved: int = 0
    papers_no_abstract: int = 0
    papers_too_short: int = 0
    papers_unique: int = 0
    api_calls: int = 0
    api_failures: int = 0

    def reject(self, reason: str) -> None:
        self.reviews_rejected[reason] = self.reviews_rejected.get(reason, 0) + 1


def short_id(openalex_id: str) -> str:
    """``https://openalex.org/W123`` -> ``W123``."""
    return openalex_id.rsplit("/", 1)[-1]


#: OpenAlex titles carry publisher markup (``<scp>``, ``<i>``, ``&amp;``).
#: It has to go: markup in an answer trips the QA gate, and it is not something
#: a reader would ever be asked to reproduce.
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|#\d+);")


def clean(text: str) -> str:
    """Strip publisher markup and collapse whitespace."""
    text = _TAG_RE.sub("", text or "")
    text = _ENTITY_RE.sub(" ", text)
    return " ".join(text.split())


def inverted_to_text(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as a word -> positions index; rebuild the prose."""
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, places in inverted.items():
        for p in places:
            positions[p] = word
    return " ".join(positions[i] for i in sorted(positions))


def viable(review: dict) -> str | None:
    """Return a rejection reason, or None when the review can carry a corpus."""
    cov = review.get("references_abstract_coverage") or {}
    if (cov.get("valid_refs") or 0) < MIN_REFS_WITH_ABSTRACT:
        return "too_few_references_with_abstracts"
    if (cov.get("ratio") or 0.0) < MIN_ABSTRACT_RATIO:
        return "abstract_coverage_too_low"
    if not review.get("referenced_works"):
        return "no_reference_list"
    # Screening labels need the eligibility criteria the review actually used.
    if not any(
        isinstance(c, str) and len(c.strip()) >= 20
        for c in (review.get("inclusion_criteria") or [])
    ):
        return "no_usable_inclusion_criteria"
    if not isinstance(review.get("objective"), str) or len(review["objective"].strip()) < 30:
        return "no_usable_objective"
    return None


class OpenAlexClient:
    """Batched OpenAlex reader with an on-disk cache.

    The cache is the difference between a reproducible corpus and one that
    quietly drifts: OpenAlex is a living index, so a re-run months later would
    otherwise resolve different text and change the pinned digest.
    """

    def __init__(self, cache_dir: Path, mailto: str, stats: Stats) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mailto = mailto
        self.stats = stats

    def _cache_path(self, work_id: str) -> Path:
        # Two-level fan-out; a flat directory of millions of files is miserable.
        return self.cache_dir / work_id[1:4] / f"{work_id}.json"

    def _cached(self, work_id: str) -> dict | None:
        p = self._cache_path(work_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _store(self, work_id: str, record: dict) -> None:
        p = self._cache_path(work_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, sort_keys=True))

    def _get(self, url: str) -> dict | None:
        for attempt in range(MAX_RETRIES):
            try:
                self.stats.api_calls += 1
                with urllib.request.urlopen(url, timeout=90) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(2**attempt)
                    continue
                self.stats.api_failures += 1
                return None
            except Exception:
                time.sleep(2**attempt)
        self.stats.api_failures += 1
        return None

    def works(self, work_ids: list[str]) -> dict[str, dict]:
        """Resolve ids to ``{id: {title, abstract}}``, using the cache first."""
        out: dict[str, dict] = {}
        missing: list[str] = []
        for wid in work_ids:
            hit = self._cached(wid)
            if hit is None:
                missing.append(wid)
            elif hit:  # empty dict = known-unresolvable, cached as a negative
                out[wid] = hit

        for i in range(0, len(missing), BATCH):
            chunk = missing[i : i + BATCH]
            params = urllib.parse.urlencode(
                {
                    "filter": "openalex_id:" + "|".join(chunk),
                    "per-page": str(BATCH),
                    "mailto": self.mailto,
                    "select": "id,title,abstract_inverted_index",
                }
            )
            data = self._get(f"{OPENALEX}?{params}")
            time.sleep(DELAY_S)
            found: set[str] = set()
            for work in (data or {}).get("results", []):
                wid = short_id(work.get("id") or "")
                if not wid:
                    continue
                found.add(wid)
                record = {
                    "title": clean(work.get("title") or ""),
                    "abstract": clean(inverted_to_text(work.get("abstract_inverted_index"))),
                }
                self._store(wid, record)
                out[wid] = record
            # Cache the misses too, so a re-run does not re-ask for records
            # OpenAlex does not have.
            for wid in chunk:
                if wid not in found:
                    self._store(wid, {})
        return out


def iter_reviews(path: Path, field: str | None, limit: int | None) -> Iterator[dict]:
    """Stream reviews from the dump, filtered, in a deterministic order.

    The dump is ~1.6 GB, so it is streamed rather than loaded. Sorting happens
    on the kept subset only.
    """
    kept: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if field and rec.get("field") != field:
                continue
            kept.append(rec)
            if limit and len(kept) >= limit * 4:  # oversample; many will reject
                break
    kept.sort(key=lambda r: r.get("id") or "")
    yield from kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sr4all", type=Path, required=True, help="sr4all_full.jsonl")
    ap.add_argument("--out-papers", type=Path, required=True, help="JSONL for build_corpus.py")
    ap.add_argument("--out-reviews", type=Path, required=True, help="JSONL of review labels")
    ap.add_argument("--cache", type=Path, default=Path(".openalex-cache"))
    ap.add_argument("--field", default="Medicine", help="OpenAlex field; empty for all")
    ap.add_argument("--max-reviews", type=int, default=200)
    ap.add_argument("--mailto", default="epago@example.com", help="OpenAlex polite pool contact")
    args = ap.parse_args()

    stats = Stats()
    client = OpenAlexClient(args.cache, args.mailto, stats)

    papers: dict[str, dict] = {}   # doc_id -> paper record, deduped across reviews
    reviews_out: list[dict] = []

    for review in iter_reviews(args.sr4all, args.field or None, args.max_reviews):
        if len(reviews_out) >= args.max_reviews:
            break
        stats.reviews_seen += 1
        reason = viable(review)
        if reason:
            stats.reject(reason)
            continue

        ref_ids = [short_id(w) for w in (review.get("referenced_works") or [])]
        stats.papers_requested += len(ref_ids)
        resolved = client.works(ref_ids)

        included: list[str] = []
        for wid in sorted(resolved):
            rec = resolved[wid]
            # Cleaned here rather than at fetch time so records already in the
            # cache are covered too — the cache holds whatever OpenAlex sent.
            abstract = clean(rec.get("abstract") or "")
            if not abstract:
                stats.papers_no_abstract += 1
                continue
            if len(abstract) < MIN_ABSTRACT_CHARS:
                stats.papers_too_short += 1
                continue
            stats.papers_resolved += 1
            included.append(wid)
            if wid not in papers:
                title = clean(rec.get("title") or "") or wid
                papers[wid] = {
                    "url": f"https://openalex.org/{wid}",
                    "title": title,
                    # The title leads the text as well as living in its own
                    # column: a reader sees the title as part of the paper, it
                    # has to be full-text searchable to be findable, and a task
                    # whose answer is a title needs that title present in the
                    # evidence or the QA gate rejects it.
                    "text": f"{title}\n\n{abstract}",
                    "category": review.get("field") or "",
                }

        if len(included) < MIN_REFS_WITH_ABSTRACT:
            stats.reject("too_few_resolved_after_fetch")
            continue

        reviews_out.append(
            {
                "review_id": short_id(review.get("id") or ""),
                "title": review.get("title") or "",
                "field": review.get("field") or "",
                "subfield": review.get("subfield") or "",
                "year": review.get("year"),
                "objective": review.get("objective") or "",
                "research_questions": review.get("research_questions") or [],
                "inclusion_criteria": review.get("inclusion_criteria") or [],
                "exclusion_criteria": review.get("exclusion_criteria") or [],
                "n_studies_final": review.get("n_studies_final"),
                # The label: these papers passed this review's screening.
                "included_work_ids": included,
            }
        )
        stats.reviews_kept += 1
        if stats.reviews_kept % 10 == 0:
            print(
                f"  {stats.reviews_kept:5d} reviews  {len(papers):7d} unique papers  "
                f"{stats.api_calls:5d} api calls",
                file=sys.stderr,
                flush=True,
            )

    stats.papers_unique = len(papers)

    args.out_papers.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_papers, "w", encoding="utf-8") as fh:
        for doc_id in sorted(papers):
            fh.write(json.dumps(papers[doc_id], sort_keys=True) + "\n")
    with open(args.out_reviews, "w", encoding="utf-8") as fh:
        for r in sorted(reviews_out, key=lambda x: x["review_id"]):
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    print(f"\nreviews seen      {stats.reviews_seen}")
    print(f"reviews kept      {stats.reviews_kept}")
    for reason, n in sorted(stats.reviews_rejected.items(), key=lambda x: -x[1]):
        print(f"    rejected: {reason:36s} {n}")
    print(f"papers requested  {stats.papers_requested}")
    print(f"papers usable     {stats.papers_resolved}")
    print(f"    no abstract   {stats.papers_no_abstract}")
    print(f"    too short     {stats.papers_too_short}")
    print(f"unique papers     {stats.papers_unique}")
    print(f"api calls         {stats.api_calls}  (failures: {stats.api_failures})")
    print(f"\nwrote {args.out_papers} and {args.out_reviews}")
    print("next: scripts/build_corpus.py --jsonl", args.out_papers, "--out corpus.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
