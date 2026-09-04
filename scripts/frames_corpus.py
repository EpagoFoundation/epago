#!/usr/bin/env python
"""FRAMES -> local Epago corpus + task file.

FRAMES (``google/frames-benchmark``, 824 multi-hop questions) ships the exact
Wikipedia articles each question needs. That is what makes it runnable without
live web: fetch those articles once, index them the way the pinned corpora are
indexed (SQLite FTS5, ``docs`` table, content-addressed doc ids), and the
shipped Search/Browse tools work unchanged.

Three stages, each cached on disk so a re-run is cheap:

  1. **fetch** — MediaWiki ``action=query&prop=revisions&rvprop=content`` in
     batches of 50 titles, wikitext gzipped per title under ``cache/``. Batched
     because the per-article ``action=parse`` path needs ~2,500 requests and
     Wikipedia 429s an anonymous client long before that; 50 titles per request
     turns the whole fetch into ~51 requests, which is both fast and polite.
  2. **render** — wikitext to plain text with TABLES PRESERVED as ``a | b | c``
     rows and infobox fields kept as ``key: value``. ~30% of FRAMES items are
     tagged "Tabular reasoning" and many others need an infobox field, so the
     usual ``prop=extracts`` path (which drops both) would make those items
     unanswerable from the corpus no matter how good the model is.
  3. **chunk + index** — articles are split into ~3.5k-char passages at
     paragraph boundaries, each carrying the article title. The shipped
     ``ToolSession`` truncates a browse to 6000 chars, so a 400 KB list
     article indexed whole would show the model only its lede; chunking keeps
     every passage fully browsable and matches the ~1.8k-char document shape
     the existing pinned corpora have.

Doc ids are content-derived exactly as ``scripts/build_corpus.py`` does it
(``ep-`` + sha256 prefix of title+text), so the same input always yields the
same corpus digest.

Usage:
    .venv/bin/python scripts/frames_corpus.py --out-dir data/frames
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import html
import json
import re
import sys
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago.environment.corpus import Document, SqliteCorpus  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
UA = "EpagoBenchmarkAnchor/0.1 (research eval; contact: epago-foundation)"

CHUNK_CHARS = 3500
# Sections that carry no answerable content but a lot of bytes.
DROP_SECTIONS = {
    "references", "external links", "further reading", "notes", "citations",
    "bibliography", "sources", "see also", "footnotes", "works cited",
}


# --- line finalization --------------------------------------------------------


def finalize_lines(text: str) -> str:
    """Canonical plain text: NFKC, one line per block, dropped tail sections."""
    text = unicodedata.normalize("NFKC", text)
    lines: list[str] = []
    dropping = False
    for line in text.split("\n"):
        line = " ".join(line.split())
        line = re.sub(r"(\s*\|\s*)+", " | ", line).strip(" |").strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if line.startswith("==") and line.endswith("=="):
            head = line.strip("= ").strip().casefold()
            dropping = head in DROP_SECTIONS
            if dropping:
                continue
        if dropping:
            continue
        if line.startswith("[") and line.endswith("]") and len(line) < 12:
            continue  # stray citation marker
        lines.append(line)
    out: list[str] = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    return "\n".join(out).strip()


# --- fetch --------------------------------------------------------------------

_lock = threading.Lock()
MIN_INTERVAL_S = 0.4  # between batched requests; ~51 requests for the whole set
BATCH_TITLES = 50     # the anonymous ``titles=`` ceiling


def _title_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    title = path.rsplit("/", 1)[-1]
    return urllib.parse.unquote(title).replace("_", " ")


def _cache_path(cache_dir: Path, title: str) -> Path:
    key = hashlib.sha256(title.encode()).hexdigest()[:24]
    return cache_dir / f"{key}.wiki.gz"


def _cached(cache_dir: Path, title: str) -> str | None:
    path = _cache_path(cache_dir, title)
    if not path.exists():
        return None
    try:
        return gzip.decompress(path.read_bytes()).decode("utf-8")
    except Exception:  # noqa: BLE001 - a corrupt cache entry is refetched
        return None


def _api_post(params: dict, retries: int = 5) -> dict | None:
    data = urllib.parse.urlencode(params).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API, data=data, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 - 429/5xx both back off
            time.sleep(min(60.0, 3.0 * (2 ** attempt)))
            if attempt == retries - 1:
                print(f"  api failure: {type(exc).__name__}: {exc}", flush=True)
    return None


def fetch_batch(titles: list[str], cache_dir: Path) -> dict[str, str]:
    """Wikitext for up to :data:`BATCH_TITLES` articles in one request.

    Redirects are resolved server-side and mapped back to the requested title,
    so a task's link list keeps pointing at content even when the article has
    since been renamed.
    """
    out: dict[str, str] = {}
    want = [t for t in titles if _cached(cache_dir, t) is None]
    for t in titles:
        hit = _cached(cache_dir, t)
        if hit is not None:
            out[t] = hit
    if not want:
        return out
    data = _api_post({
        "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "titles": "|".join(want), "redirects": "1", "format": "json",
        "formatversion": "2", "maxlag": "5",
    })
    if not data or "query" not in data:
        return out
    query = data["query"]
    # requested title -> resolved title, through normalization and redirects
    alias: dict[str, str] = {}
    for kind in ("normalized", "redirects"):
        for entry in query.get(kind, ()):  # from -> to, applied in order
            alias[entry["from"]] = entry["to"]
    by_title: dict[str, str] = {}
    for page in query.get("pages", ()):
        try:
            by_title[page["title"]] = page["revisions"][0]["slots"]["main"]["content"]
        except Exception:  # noqa: BLE001 - missing page or no readable slot
            continue
    for title in want:
        resolved = title
        for _ in range(4):  # normalization then redirect, possibly chained
            if resolved in by_title:
                break
            if resolved in alias:
                resolved = alias[resolved]
            else:
                break
        body = by_title.get(resolved)
        if body is None:
            continue
        _cache_path(cache_dir, title).write_bytes(gzip.compress(body.encode("utf-8")))
        out[title] = body
    return out


# --- wikitext -> text ---------------------------------------------------------

_DROP_TEMPLATE_PREFIXES = (
    "cite", "citation", "sfn", "harv", "refn", "efn", "notelist", "reflist",
    "short description", "use ", "about", "main", "see also", "further",
    "hatnote", "redirect", "commons", "wikiquote", "wikisource", "portal",
    "navbox", "authority control", "div col", "colend", "col-", "clear",
    "toc", "good article", "featured article", "pp-", "italic title",
    "defaultsort", "sister", "subst:", "citation needed", "cn", "who",
    "when", "clarify", "dead link", "webarchive", "isbn", "issn", "doi",
    "legend", "sfnp", "rp", "nbsp", "spaced ndash", "\u2013", "small",
    "flagicon", "flag", "sort", "center", "align", "anchor", "coord missing",
)
_DROP_PARAM_NAMES = frozenset({
    "image", "image_size", "imagesize", "caption", "alt", "logo", "signature",
    "image_upright", "width", "align", "class", "style", "url", "archive-url",
    "archiveurl", "ref", "footnotes", "module", "embed",
})
_INNER_TPL_RE = re.compile(r"\{\{([^{}]*)\}\}")
_CELL_ATTR_RE = re.compile(
    r"""^\s*(?:[A-Za-z-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s|]+)\s*)+\|(?!\|)"""
)


def _render_template(body: str) -> str:
    parts = body.split("|")
    name = parts[0].strip()
    low = name.casefold()
    if not low or low.startswith(_DROP_TEMPLATE_PREFIXES):
        return " "
    # {{convert|1776|ft|m|0|abbr=values}} -> "1776 ft". Special-cased because
    # it is by far the most common fact-bearing template in the list and
    # infobox articles FRAMES draws on, and the generic parameter dump turns
    # every measurement into "1776 ft m 0 abbr: values".
    if low in ("convert", "cvt") and len(parts) >= 3:
        return f" {parts[1].strip()} {parts[2].strip()} "
    chunks: list[str] = []
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            key, value = key.strip(), value.strip()
            if not value or key.casefold() in _DROP_PARAM_NAMES:
                continue
            chunks.append(f"{key}: {value}")
        else:
            value = part.strip()
            if value:
                chunks.append(value)
    if not chunks:
        return f" {name} " if len(name) < 40 else " "
    return " " + " ".join(chunks) + " "


def _expand_templates(text: str, rounds: int = 12) -> str:
    """Flatten templates innermost-first, keeping their parameter VALUES.

    Wikitext is unexpanded source, so an infobox arrives as
    ``{{Infobox book | dewey = 823.8 | oclc = 3163777}}``. Dropping templates
    wholesale (what a generic wikitext stripper does) would throw away exactly
    the structured facts FRAMES asks for; keeping ``key: value`` preserves them
    in a form BM25 indexes and a reader can follow.
    """
    for _ in range(rounds):
        text, n = _INNER_TPL_RE.subn(lambda m: _render_template(m.group(1)), text)
        if not n:
            break
    return text.replace("{{", " ").replace("}}", " ")


def _cell_text(cell: str) -> str:
    cell = _CELL_ATTR_RE.sub("", cell, count=1)
    return cell.strip()


def _convert_tables(text: str) -> str:
    """``{| ... |}`` blocks to one pipe-joined line per row."""
    out: list[str] = []
    row: list[str] = []
    depth = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("{|"):
            depth += 1
            continue
        if depth == 0:
            out.append(line)
            continue
        if stripped.startswith("|}"):
            depth -= 1
            if depth == 0 and row:
                out.append(" | ".join(row))
                row = []
            continue
        if stripped.startswith("|-"):
            if row:
                out.append(" | ".join(row))
                row = []
            continue
        if stripped.startswith("|+"):
            out.append(_cell_text(stripped[2:]))
            continue
        if stripped[:1] in ("|", "!"):
            for cell in re.split(r"\|\||!!", stripped[1:]):
                value = _cell_text(cell)
                if value:
                    row.append(value)
            continue
        if row:  # a cell whose content wrapped onto the next line
            row[-1] = f"{row[-1]} {stripped}".strip()
        elif stripped:
            out.append(stripped)
    if row:
        out.append(" | ".join(row))
    return "\n".join(out)


def wikitext_to_text(raw: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", raw, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", " ", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S | re.I)
    text = re.sub(
        r"<(gallery|imagemap|timeline|score|syntaxhighlight|source|maplink|mapframe)"
        r"[^>]*>.*?</\1>", " ", text, flags=re.S | re.I,
    )
    # Links FIRST: ``[[One World Trade Center|One WTC]]`` carries a pipe, and a
    # pipe is also the template-parameter and table-cell separator. Resolving
    # links to their display text before either of those passes is what keeps
    # a piped link from cutting an infobox field or a table row in half.
    text = re.sub(r"\[\[(?:File|Image|Category|Media)\s*:[^\[\]]*\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[([^\]|]*)\|([^\]|]*)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]|]*)\]\]", r"\1", text)
    text = re.sub(r"\[(?:https?:)?//\S+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[(?:https?:)?//\S+\]", " ", text)
    text = _convert_tables(text)
    text = _expand_templates(text)
    text = re.sub(r"\[\[([^\]|]*)\|([^\]|]*)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"<[^>]{1,200}>", " ", text)
    text = text.replace(", ).replace(", "").replace("''", "")
    text = re.sub(r"^[\*#:;]+\s*", "", text, flags=re.M)
    text = re.sub(r"^-{4,}\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*(={2,6})\s*(.*?)\s*\1\s*$", r"== \2 ==", text, flags=re.M)
    return finalize_lines(html.unescape(text))


# --- frames rows --------------------------------------------------------------


def article_urls(row: dict) -> list[str]:
    urls: list[str] = []
    try:
        urls.extend(str(u) for u in ast.literal_eval(row.get("wiki_links") or "[]"))
    except Exception:  # noqa: BLE001
        pass
    for i in range(1, 11):
        v = (row.get(f"wikipedia_link_{i}") or "").strip()
        if v:
            urls.append(v)
    tail = (row.get("wikipedia_link_11+") or "").strip()
    for part in re.split(r"[\s,]+", tail):
        if part.startswith("http"):
            urls.append(part)
    seen: list[str] = []
    for u in urls:
        u = u.strip().strip("'\"")
        if u.startswith("http") and u not in seen:
            seen.append(u)
    return seen


# --- chunking + indexing ------------------------------------------------------


def content_doc_id(title: str, text: str) -> str:
    h = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()
    return f"ep-{h[:16]}"


def chunk_article(title: str, text: str, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    """Split at paragraph boundaries into <= chunk_chars passages."""
    paras = [p for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paras:
        if size + len(para) + 1 > chunk_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        if len(para) > chunk_chars:  # a single monster row/paragraph
            for i in range(0, len(para), chunk_chars):
                chunks.append(para[i : i + chunk_chars])
            continue
        buf.append(para)
        size += len(para) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks or [text[:chunk_chars]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", type=Path, default=Path("data/frames/test.tsv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    args = ap.parse_args()

    out_dir = args.out_dir
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8"), delimiter="\t"))
    print(f"frames rows: {len(rows)}", flush=True)

    url_by_title: dict[str, str] = {}
    for row in rows:
        for url in article_urls(row):
            url_by_title.setdefault(_title_from_url(url), url)
    titles = sorted(url_by_title)
    print(f"unique articles: {len(titles)}", flush=True)

    t0 = time.time()
    articles: dict[str, str] = {}
    for i in range(0, len(titles), BATCH_TITLES):
        batch = titles[i : i + BATCH_TITLES]
        articles.update(fetch_batch(batch, cache_dir))
        print(f"  fetched {min(i+BATCH_TITLES, len(titles))}/{len(titles)} "
              f"({len(articles)} ok) in {time.time()-t0:.0f}s", flush=True)
        time.sleep(MIN_INTERVAL_S)
    failed = [t for t in titles if t not in articles]
    print(f"fetched {len(articles)} articles, {len(failed)} failed, "
          f"{time.time()-t0:.0f}s", flush=True)
    if failed:
        (out_dir / "fetch_failures.json").write_text(json.dumps(failed, indent=1))

    # render + chunk + index
    db_path = out_dir / "corpus.db"
    if db_path.exists():
        db_path.unlink()
    corpus = SqliteCorpus.create(db_path)
    docs: list[Document] = []
    doc_ids_by_title: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    empty: list[str] = []
    for title in sorted(articles):
        text = wikitext_to_text(articles[title])
        if len(text) < 200:
            empty.append(title)
            continue
        url = url_by_title[title]
        ids: list[str] = []
        chunks = chunk_article(title, text, args.chunk_chars)
        for i, chunk in enumerate(chunks):
            body = f"{title}\n\n{chunk}"
            doc_title = title if len(chunks) == 1 else f"{title} (part {i+1}/{len(chunks)})"
            doc_id = content_doc_id(doc_title, body)
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            docs.append(Document(doc_id=doc_id, url=url, title=doc_title, text=body,
                                 category="wikipedia"))
            ids.append(doc_id)
        doc_ids_by_title[title] = ids
    docs.sort(key=lambda d: d.doc_id)
    corpus.add_documents(docs)
    print(f"corpus: {corpus.doc_count()} chunks from {len(doc_ids_by_title)} articles "
          f"({len(empty)} rendered empty)", flush=True)
    corpus.close()

    # tasks: anchor-shaped JSONL + an article map for retrieval diagnostics
    tasks_path = out_dir / "frames_tasks.jsonl"
    map_path = out_dir / "frames_docmap.json"
    n_full = 0
    with open(tasks_path, "w", encoding="utf-8") as fh:
        for row in rows:
            titles_for_row = [_title_from_url(u) for u in article_urls(row)]
            evidence: list[str] = []
            missing = 0
            for t in titles_for_row:
                ids = doc_ids_by_title.get(t)
                if ids:
                    evidence.extend(ids)
                else:
                    missing += 1
            n_full += int(missing == 0)
            fh.write(json.dumps({
                "id": f"frames-{int(row['']):04d}",
                "question": row["Prompt"].strip(),
                "answer": row["Answer"].strip(),
                "aliases": [],
                "reasoning_types": row.get("reasoning_types", ""),
                "articles": titles_for_row,
                "missing_articles": missing,
                "evidence_doc_ids": evidence,
            }, ensure_ascii=False) + "\n")
    (map_path).write_text(json.dumps(doc_ids_by_title))
    print(f"tasks -> {tasks_path} ({n_full}/{len(rows)} items have every article present)")
    print(f"docmap -> {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
