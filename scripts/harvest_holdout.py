#!/usr/bin/env python
"""Weekly private-holdout builder: many fresh sources -> one dated parquet dataset.

Harvests papers published in a recent window from several independent sources,
keeps only the ones the task templates can mint from, and writes a sharded
parquet dataset in exactly the shape
:class:`epago.taskgen.ingest.HfSnapshotSource` consumes — parquet shards with a
``text`` column whose first line is the paper title. Optionally publishes it to
a PRIVATE, dated HuggingFace dataset repo and prints the ``[private_source]``
pin lines for the chain contract.

Scope is a flag, not a hardcode. ``--domain`` selects OpenAlex domains (default:
all four), and the theme pool spans every discipline, so the same builder
produces a medicine-only slice, a physics + CS slice, or all of scientific
literature. The task templates never needed a field — see
:class:`epago.taskgen.templates.FindingVocabulary` — so widening the corpus is
the only change the subnet needs to cover all of science.

Threat model — why this is randomized and multi-source. The holdout's real
protection is *freshness*: a paper published this week cannot be in a miner's
training data, so a model can only score by genuinely searching and extracting.
A sophisticated miner counters freshness by continuously ingesting *every* new
paper and re-training. This pipeline widens what "every" has to mean:

  * multiple independent sources (OpenAlex; Europe PMC spanning MEDLINE, PMC and
    preprints; PubMed; and Crossref's broad cross-publisher index), so covering
    one index is not enough;
  * a seeded random *theme* each week drawn across the selected domains, so the
    slice is e.g. "recent catalysis + nephrology + robotics", not "all science";
  * a random source mix and sub-window.

To be safe a miner must therefore continuously cover all sources, all subfields
and all recent windows — far more expensive than mirroring one feed. Widening
from one domain to four multiplies that cost again. The repo is also PRIVATE
while live, so the exact slice is never visible until it rotates.

Source scope is recorded, never faked. Europe PMC and PubMed index biomedicine
only; they are given biomedical themes only, and are dropped from the plan
entirely when no biomedical domain is selected. The manifest records each
source's scope and which themes it actually ran, so a slice is never labelled
broader than the index that produced it.

Determinism is untouched: the randomness lives entirely in this off-chain
*builder*. The published parquet is a fixed, digest-pinned artifact; validators
read it deterministically through ``HfSnapshotSource``. The plan (seed, domains,
sources, themes, counts) is recorded in ``manifest.json`` locally for our own
audit and is NOT uploaded while the feed is live.

Operational rule: keep each week's repo PRIVATE while it is the live holdout, and
reveal it (name + contents) only AFTER it rotates out — matching the
delayed-transparency audit model in :mod:`epago.taskgen.private_pool`.

This module is the *builder*. One scheduled rotation — build, publish, emit the new
``[private_source]`` pin as JSON, optionally rewrite a contract — is
``scripts/rotate_holdout.py``, which the ``auto-holdout`` compose profile runs
weekly. Use this script directly for one-off or experimental slices.

Usage:
    .venv/bin/python scripts/harvest_holdout.py \\
        --days 7 --target 2500 --shards 12 --publish
    # one discipline only (OpenAlex domain ids; see DOMAINS below):
    .venv/bin/python scripts/harvest_holdout.py --domain 4 --target 2500
    # reproduce a past build exactly:
    .venv/bin/python scripts/harvest_holdout.py --seed 123456789 ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field as dc_field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fetch_papers import clean, inverted_to_text  # noqa: E402
from epago.taskgen.templates import (  # noqa: E402
    VOCABULARIES,
    FindingVocabulary,
    finding_sentences,
)

DELAY_S = 0.25
MAX_RETRIES = 4

#: HuggingFace org the dated holdout repos are published under.
DEFAULT_ORG = "EpagoFoundation"

MIN_TITLE_CHARS = 25
MAX_TITLE_CHARS = 180
MIN_ABSTRACT_CHARS = 200

#: OpenAlex top-level domains, id -> (display name, repo slug). Verified against
#: https://api.openalex.org/domains — note 1 is Life and 3 is Physical, which is
#: the opposite of what the ordering suggests, so read these, do not guess.
DOMAINS: dict[int, tuple[str, str]] = {
    1: ("Life Sciences", "life"),
    2: ("Social Sciences", "social"),
    3: ("Physical Sciences", "physical"),
    4: ("Health Sciences", "health"),
}
ALL_DOMAINS = tuple(sorted(DOMAINS))

#: Domains Europe PMC and PubMed actually index. A slice drawn from those two
#: cannot honestly be called physics or economics, so they are confined to these.
BIOMEDICAL_DOMAINS = (1, 4)

#: Themes the weekly slice is randomly drawn from, grouped by the domain they
#: belong to. Each is a plain keyword every source API understands (title /
#: abstract match), so the same theme narrows every index to the same subfield.
#: The medical themes are still here — health is part of "all science", not a
#: thing the general pool replaced.
THEMES_BY_DOMAIN: dict[int, list[str]] = {
    1: [
        "genomics", "molecular biology", "ecology", "evolution", "microbiology",
        "biochemistry", "plant biology", "cell biology", "biodiversity",
        "marine biology", "agriculture", "structural biology", "bioinformatics",
        "protein folding", "gene expression",
        "virology", "mycology", "entomology", "zoology", "botany",
        "neuroscience", "genetics", "epigenetics", "proteomics", "metabolomics",
        "synthetic biology", "conservation biology", "fisheries", "forestry",
        "soil science", "food science", "veterinary science", "parasitology",
        "toxicology", "developmental biology",
    ],
    2: [
        "economics", "education", "psychology", "sociology", "political science",
        "linguistics", "urban planning", "finance", "criminology", "demography",
        "anthropology", "labor market", "public policy", "human geography",
        "behavioral economics",
        "archaeology", "accounting", "marketing", "management",
        "international relations", "social work", "communication", "journalism",
        "tourism", "gender studies", "econometrics", "game theory",
        "cognitive science", "migration", "consumer behavior",
        # business & industry
        "branding", "advertising", "retail", "e-commerce", "supply chain",
        "logistics", "entrepreneurship", "innovation", "human resources",
        "corporate governance", "banking", "insurance", "real estate",
        "taxation", "auditing", "investment", "hospitality",
        "small business", "manufacturing industry", "labor economics",
    ],
    3: [
        "physics", "materials science", "machine learning", "chemistry",
        "astronomy", "robotics", "renewable energy", "climate", "geology",
        "computer vision", "quantum computing", "catalysis", "semiconductors",
        "fluid dynamics", "cryptography", "optics", "batteries", "photovoltaics",
        "natural language processing", "atmospheric science", "seismology",
        "polymer science", "nanotechnology", "hydrology",
        "astrophysics", "particle physics", "condensed matter", "spectroscopy",
        "electrochemistry", "organic chemistry", "inorganic chemistry",
        "analytical chemistry", "geochemistry", "mineralogy", "oceanography",
        "meteorology", "aerospace", "civil engineering",
        "mechanical engineering", "electrical engineering",
        "software engineering", "cybersecurity", "data mining",
        "signal processing", "control systems", "thermodynamics", "photonics",
        "superconductivity", "statistics", "operations research",
        "remote sensing", "wireless networks", "combustion", "metallurgy",
    ],
    4: [
        "oncology", "cardiology", "infectious disease", "neurology", "pediatrics",
        "immunology", "endocrinology", "psychiatry", "nephrology", "pulmonology",
        "gastroenterology", "dermatology", "hematology", "rheumatology",
        "ophthalmology", "public health", "surgery", "radiology", "pharmacology",
        "epidemiology",
        "anesthesiology", "orthopedics", "urology", "gynecology", "obstetrics",
        "nursing", "dentistry", "otolaryngology", "pathology", "geriatrics",
        "palliative care", "sports medicine", "occupational health",
        "nutrition", "physiotherapy", "vaccinology", "telemedicine",
        "medical imaging", "emergency medicine", "transplantation",
    ],
}

#: The whole pool, flat — every discipline in one list.
THEMES = [t for d in ALL_DOMAINS for t in THEMES_BY_DOMAIN[d]]

#: What each index actually covers. "all" sources can serve any domain;
#: "biomedical" ones are restricted to :data:`BIOMEDICAL_DOMAINS` themes and
#: dropped when no biomedical domain is selected.
SOURCE_SCOPE = {
    "openalex": "all",
    "crossref": "all",
    "europepmc": "biomedical",
    "pubmed": "biomedical",
}

#: Europe PMC source buckets to randomly mix: journals (MED), full-text (PMC),
#: preprints (PPR). Mixing these changes the journal/preprint balance week to week.
EPMC_SRC_CHOICES = ["MED", "MED OR PPR", "MED OR PMC", "MED OR PMC OR PPR"]


def parse_domains(values: list[str]) -> tuple[int, ...]:
    """``["all"]`` / ``["1","3"]`` / ``["1,3"]`` -> a sorted tuple of domain ids."""
    if not values:
        return ALL_DOMAINS
    out: set[int] = set()
    for value in values:
        for part in str(value).replace(" ", ",").split(","):
            if not part:
                continue
            if part.lower() == "all":
                out.update(ALL_DOMAINS)
                continue
            try:
                domain = int(part)
            except ValueError:
                raise SystemExit(f"error: --domain {part!r} is not a domain id or 'all'")
            if domain not in DOMAINS:
                raise SystemExit(
                    f"error: --domain {domain} unknown; ids are "
                    + ", ".join(f"{d}={DOMAINS[d][0]}" for d in ALL_DOMAINS)
                )
            out.add(domain)
    return tuple(sorted(out))


def domain_slug(domains: tuple[int, ...]) -> str:
    """Repo-name fragment: 'science' for the lot, else the domain slugs joined."""
    if tuple(domains) == ALL_DOMAINS:
        return "science"
    return "-".join(DOMAINS[d][1] for d in domains)


def period_tag(end: dt.date) -> str:
    """The rotation period a window ending on ``end`` belongs to: ``2026w34``."""
    iso = end.isocalendar()
    return f"{iso.year}w{iso.week:02d}"


def default_repo(domains: tuple[int, ...], end: dt.date, org: str = DEFAULT_ORG) -> str:
    """The dated, scope-honest repo id for one rotation period.

    Deterministic from (domains, period), which is what makes a rotation
    re-runnable: the same week always names the same repo, so a second run is
    detectable as a duplicate instead of publishing a second slice.
    """
    return f"{org}/epago-holdout-{domain_slug(domains)}-{period_tag(end)}"


@dataclass
class Stats:
    seen: int = 0
    kept: int = 0
    rej_title: int = 0
    rej_abstract: int = 0
    rej_no_finding: int = 0
    rej_duplicate: int = 0
    api_failures: int = 0
    per_source: dict = dc_field(default_factory=dict)
    per_domain: dict = dc_field(default_factory=dict)
    #: Word list the "is this mintable?" gate reads with — the harvest must
    #: keep what the release it feeds can actually mint from.
    vocab: FindingVocabulary = VOCABULARIES["general"]


def _get_bytes(url: str, stats: Stats, headers: dict | None = None) -> bytes | None:
    req = urllib.request.Request(url, headers=headers or {})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            stats.api_failures += 1
            return None
        except Exception:
            time.sleep(2**attempt)
    stats.api_failures += 1
    return None


def _get_json(url: str, stats: Stats, headers: dict | None = None) -> dict | None:
    raw = _get_bytes(url, stats, headers)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        stats.api_failures += 1
        return None


def _content_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _keep(title: str, abstract: str, stats: Stats) -> str | None:
    """Return the finished text if the paper passes every gate, else None.

    The gate is unchanged in kind — a kept paper must be one the templates can
    actually mint a task from — but it now reads with the release's vocabulary
    (:data:`Stats.vocab`), so a physics abstract reporting an efficiency is kept
    for the same reason a trial reporting a mortality always was.
    """
    if not (MIN_TITLE_CHARS <= len(title) <= MAX_TITLE_CHARS):
        stats.rej_title += 1
        return None
    if len(abstract) < MIN_ABSTRACT_CHARS:
        stats.rej_abstract += 1
        return None
    text = f"{title}\n\n{abstract}"
    if not finding_sentences(text, stats.vocab):  # the quality gate: must be mintable
        stats.rej_no_finding += 1
        return None
    return text


# --- sources -----------------------------------------------------------------


def openalex_fetch(
    from_date: str,
    to_date: str,
    theme: str | None,
    want: int,
    mailto: str,
    stats: Stats,
    domains: tuple[int, ...] = ALL_DOMAINS,
) -> dict[str, dict]:
    """Works in the window from the given OpenAlex domains, optionally themed.

    ``domains`` becomes an OR filter over ``primary_topic.domain.id``, so the
    domain label on every record here is the index's own classification rather
    than an inference from the search term.
    """
    base = "https://api.openalex.org/works"
    domain_filter = "|".join(f"https://openalex.org/domains/{d}" for d in domains)
    dom_tag = "-".join(str(d) for d in domains)
    filt = ",".join(
        [
            f"from_publication_date:{from_date}",
            f"to_publication_date:{to_date}",
            f"primary_topic.domain.id:{domain_filter}",
            "has_abstract:true",
            "language:languages/en",
            "type:article",
        ]
    )
    out: dict[str, dict] = {}
    cursor = "*"
    while len(out) < want:
        params = {
            "filter": filt,
            "per-page": "200",
            "cursor": cursor,
            "select": "id,title,abstract_inverted_index,primary_topic",
            "mailto": mailto,
        }
        if theme:
            params["search"] = theme
        data = _get_json(f"{base}?{urllib.parse.urlencode(params)}", stats)
        if data is None:
            break
        results = data.get("results", [])
        for w in results:
            stats.seen += 1
            title = clean(w.get("title") or "")
            abstract = clean(inverted_to_text(w.get("abstract_inverted_index")))
            text = _keep(title, abstract, stats)
            if text is None:
                continue
            cid = _content_id(text)
            if cid in out:
                stats.rej_duplicate += 1
                continue
            domain = ((w.get("primary_topic") or {}).get("domain") or {}).get("display_name", "")
            out[cid] = {"url": w.get("id") or "", "title": title, "text": text,
                        "category": f"openalex:d{dom_tag}:{theme or 'all'}",
                        "domain": domain}
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor or not results:
            break
        time.sleep(DELAY_S)
    stats.per_source["openalex"] = stats.per_source.get("openalex", 0) + len(out)
    return out


def europepmc_fetch(
    from_date: str, to_date: str, theme: str | None, src: str, want: int, stats: Stats
) -> dict[str, dict]:
    """Europe PMC core records (MEDLINE/PMC/preprints) in the window, with abstracts."""
    base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    terms = [f"(SRC:{src})", f"(FIRST_PDATE:[{from_date} TO {to_date}])",
             "HAS_ABSTRACT:Y", "LANG:eng"]
    if theme:
        terms.append(f'"{theme}"')
    query = " AND ".join(terms)
    out: dict[str, dict] = {}
    cursor = "*"
    seen_cursors: set[str] = set()
    while len(out) < want:
        params = {"query": query, "format": "json", "resultType": "core",
                  "pageSize": "100", "cursorMark": cursor}
        data = _get_json(f"{base}?{urllib.parse.urlencode(params)}", stats,
                         headers={"User-Agent": "epago-holdout/1.0"})
        if data is None:
            break
        for r in data.get("resultList", {}).get("result", []):
            stats.seen += 1
            title = clean(r.get("title") or "")
            abstract = clean(r.get("abstractText") or "")
            text = _keep(title, abstract, stats)
            if text is None:
                continue
            cid = _content_id(text)
            if cid in out:
                stats.rej_duplicate += 1
                continue
            url = r.get("doi") and f"https://doi.org/{r['doi']}" or \
                f"https://europepmc.org/article/{r.get('source','MED')}/{r.get('id','')}"
            out[cid] = {"url": url, "title": title, "text": text,
                        "category": f"epmc:{src}:{theme or 'all'}"}
        cursor = data.get("nextCursorMark") or ""
        if not cursor or cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
        time.sleep(DELAY_S)
    stats.per_source["europepmc"] = stats.per_source.get("europepmc", 0) + len(out)
    return out


def pubmed_fetch(
    from_date: str, to_date: str, theme: str | None, want: int, stats: Stats
) -> dict[str, dict]:
    """NCBI PubMed (MEDLINE) via E-utilities: clean structured abstracts."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    hdr = {"User-Agent": "epago-holdout/1.0"}
    term = (f"{from_date.replace('-', '/')}:{to_date.replace('-', '/')}[pdat] "
            "AND hasabstract[filt] AND English[lang]")
    if theme:
        term = f"({theme}[tiab]) AND " + term
    es = _get_json(f"{base}/esearch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(min(max(want * 3, 100), 1000))}),
        stats, headers=hdr)
    out: dict[str, dict] = {}
    if es is None:
        return out
    ids = es.get("esearchresult", {}).get("idlist", [])
    for i in range(0, len(ids), 200):
        if len(out) >= want:
            break
        xml = _get_bytes(f"{base}/efetch.fcgi?" + urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(ids[i:i + 200]), "retmode": "xml"}), stats, headers=hdr)
        if xml is None:
            continue
        for art in re.split(r"<PubmedArticle>", xml.decode("utf-8", "replace"))[1:]:
            stats.seen += 1
            ti = re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", art, re.S)
            if not ti:
                continue
            title = clean(ti.group(1))
            abstract = clean(" ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.S)))
            text = _keep(title, abstract, stats)
            if text is None:
                continue
            cid = _content_id(text)
            if cid in out:
                stats.rej_duplicate += 1
                continue
            pmid = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
            out[cid] = {"url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid.group(1) if pmid else ''}",
                        "title": title, "text": text, "category": f"pubmed:{theme or 'all'}"}
        time.sleep(DELAY_S)
    stats.per_source["pubmed"] = stats.per_source.get("pubmed", 0) + len(out)
    return out


def crossref_fetch(
    from_date: str, to_date: str, theme: str | None, want: int, mailto: str, stats: Stats
) -> dict[str, dict]:
    """Crossref: the broad cross-publisher index — journals the others miss."""
    base = "https://api.crossref.org/works"
    out: dict[str, dict] = {}
    cursor = "*"
    while len(out) < want:
        params = {
            "filter": (f"from-pub-date:{from_date},until-pub-date:{to_date},"
                       "type:journal-article,has-abstract:true"),
            "rows": "100", "cursor": cursor, "select": "title,abstract,DOI", "mailto": mailto,
        }
        if theme:
            params["query"] = theme
        d = _get_json(f"{base}?" + urllib.parse.urlencode(params), stats,
                      headers={"User-Agent": "epago-holdout/1.0"})
        if d is None:
            break
        items = d.get("message", {}).get("items", [])
        for it in items:
            stats.seen += 1
            raw_title = it.get("title") or []
            title = clean(" ".join(raw_title) if isinstance(raw_title, list) else str(raw_title))
            abstract = clean(it.get("abstract", ""))
            text = _keep(title, abstract, stats)
            if text is None:
                continue
            cid = _content_id(text)
            if cid in out:
                stats.rej_duplicate += 1
                continue
            doi = it.get("DOI")
            out[cid] = {"url": f"https://doi.org/{doi}" if doi else "", "title": title,
                        "text": text, "category": f"crossref:{theme or 'all'}"}
        cursor = d.get("message", {}).get("next-cursor")
        if not cursor or not items:
            break
        time.sleep(DELAY_S)
    stats.per_source["crossref"] = stats.per_source.get("crossref", 0) + len(out)
    return out


#: Every source the planner may draw from.
ALL_SOURCES = ["openalex", "europepmc", "pubmed", "crossref"]


# --- weekly plan (seeded random) ---------------------------------------------


def plan_harvest(
    seed: int, from_date: str, to_date: str, domains: tuple[int, ...] = ALL_DOMAINS,
    n_themes: int | None = None,
) -> dict:
    """Deterministically derive this week's harvest plan from ``seed``.

    Random axes: which sources, their target mix, and 3-6 themes drawn *across*
    the selected domains — one per domain first so a four-domain week is never
    accidentally four oncology themes, then the remainder from the whole pool.
    Recorded in the manifest so any build can be reproduced with ``--seed``.
    """
    rng = random.Random(seed)
    domains = tuple(sorted(domains))
    pool = [(d, t) for d in domains for t in THEMES_BY_DOMAIN[d]]
    # The 3-6 draw sizes a weekly holdout slice; a corpus-scale build passes
    # ``n_themes`` explicitly to sweep most of the pool in one run.
    drawn = max(rng.randint(3, 6), len(domains))
    n_themes = drawn if n_themes is None else max(n_themes, len(domains))
    # Guarantee coverage: one theme per selected domain, then fill from the rest.
    picked = [(d, rng.choice(THEMES_BY_DOMAIN[d])) for d in domains]
    rest = [p for p in pool if p not in picked]
    picked += rng.sample(rest, k=min(n_themes - len(picked), len(rest)))
    rng.shuffle(picked)
    themes = [t for _, t in picked]
    theme_domains = {t: d for d, t in picked}

    # Only offer sources that can honestly serve the selected domains: Europe PMC
    # and PubMed index biomedicine, so they are out entirely unless a biomedical
    # domain was asked for.
    biomedical = [d for d in domains if d in BIOMEDICAL_DOMAINS]
    available = sorted(
        s for s in ALL_SOURCES if SOURCE_SCOPE[s] == "all" or biomedical
    )
    # Pick a random subset of the available sources (at least 2 where possible),
    # with random normalized weights — so both which indexes appear and their
    # mix vary weekly.
    k = rng.randint(min(2, len(available)), len(available))
    chosen = sorted(rng.sample(available, k))
    raw = {s: rng.uniform(0.5, 1.5) for s in chosen}
    total = sum(raw.values())
    weights = {s: round(v / total, 3) for s, v in raw.items()}

    notes = []
    for source in ALL_SOURCES:
        if SOURCE_SCOPE[source] != "biomedical":
            continue
        if not biomedical:
            notes.append(
                f"{source}: skipped — indexes biomedicine only and no biomedical "
                f"domain ({', '.join(str(d) for d in BIOMEDICAL_DOMAINS)}) was selected"
            )
        elif source in chosen:
            notes.append(
                f"{source}: biomedical index — restricted to the "
                f"{', '.join(DOMAINS[d][0] for d in biomedical)} share of the themes"
            )
    return {
        "seed": seed,
        "from_date": from_date,
        "to_date": to_date,
        "domains": list(domains),
        "domain_names": [DOMAINS[d][0] for d in domains],
        "themes": themes,
        "theme_domains": theme_domains,
        "sources": chosen,
        "source_scope": {s: SOURCE_SCOPE[s] for s in chosen},
        "weights": weights,
        "epmc_src": rng.choice(EPMC_SRC_CHOICES),
        "scope_notes": notes,
    }


def run_plan(plan: dict, target: int, mailto: str, stats: Stats) -> list[dict]:
    """Execute the plan: fetch themed slices from each chosen source, top up broad."""
    themes = plan["themes"]
    theme_domains = plan.get("theme_domains", {})
    domains = tuple(plan.get("domains") or ALL_DOMAINS)
    fr, to = plan["from_date"], plan["to_date"]
    merged: dict[str, dict] = {}
    plan.setdefault("themes_run", {})

    def spread(want: int) -> list[int]:
        if not themes:
            return [want]
        base = want // len(themes)
        return [base + (1 if i < want % len(themes) else 0) for i in range(len(themes))]

    def fetch(name: str, theme: str | None, want: int) -> dict[str, dict]:
        if want <= 0:
            return {}
        # A biomedical index never runs a non-biomedical theme: the records it
        # would return are not from the domain the theme claims.
        theme_domain = theme_domains.get(theme) if theme else None
        if (
            SOURCE_SCOPE[name] == "biomedical"
            and theme is not None
            and theme_domain not in BIOMEDICAL_DOMAINS
        ):
            return {}
        if theme is not None:
            plan["themes_run"].setdefault(name, []).append(theme)
        if name == "openalex":
            # Themed slices are pinned to the theme's own domain; the broad
            # top-up spans everything selected.
            doms = (theme_domain,) if theme_domain in DOMAINS else domains
            return openalex_fetch(fr, to, theme, want, mailto, stats, doms)
        if name == "europepmc":
            return europepmc_fetch(fr, to, theme, plan["epmc_src"], want, stats)
        if name == "pubmed":
            return pubmed_fetch(fr, to, theme, want, stats)
        if name == "crossref":
            return crossref_fetch(fr, to, theme, want, mailto, stats)
        return {}

    for name in plan["sources"]:
        s_target = int(target * plan["weights"].get(name, 0.0))
        for theme, w in zip(themes, spread(s_target)):
            merged.update(fetch(name, theme, w))

    # Fallback: if themed slices starved, top up broad (no theme) from a source
    # that can cover every selected domain — prefer OpenAlex, else any "all"
    # source, else give up rather than backfill a biomedical index into a
    # slice labelled physics.
    if len(merged) < target:
        broad = next(
            (s for s in ("openalex", "crossref") if s in plan["sources"]),
            next((s for s in plan["sources"] if SOURCE_SCOPE[s] == "all"), None),
        )
        if broad is not None:
            merged.update(fetch(broad, None, target - len(merged)))

    for rec in merged.values():
        key = rec.get("domain") or "unclassified"
        stats.per_domain[key] = stats.per_domain.get(key, 0) + 1
    stats.kept = len(merged)
    return [merged[cid] for cid in sorted(merged)]


# --- output ------------------------------------------------------------------


def write_shards(papers: list[dict], out_dir: Path, n_shards: int) -> list[Path]:
    """Split papers into ``n_shards`` contiguous parquet files with a ``text`` column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, min(n_shards, len(papers)))
    size = (len(papers) + n_shards - 1) // n_shards
    paths: list[Path] = []
    for i in range(n_shards):
        chunk = papers[i * size : (i + 1) * size]
        if not chunk:
            continue
        # ``domain`` is the index's own classification and is only populated
        # where an index provides one (OpenAlex); empty means unclassified, not
        # "some default domain".
        table = pa.table({
            "text": [p["text"] for p in chunk],
            "title": [p["title"] for p in chunk],
            "url": [p["url"] for p in chunk],
            "category": [p["category"] for p in chunk],
            "domain": [p.get("domain", "") for p in chunk],
        })
        path = out_dir / f"shard-{i:03d}.parquet"
        pq.write_table(table, path)
        paths.append(path)
    return paths


def publish(repo: str, shard_paths: list[Path], token: str, api=None) -> str:
    """Create a PRIVATE dataset repo and upload only the parquet shards.

    The private check is not decoration: ``exist_ok=True`` silently accepts a repo
    that already exists, and an existing PUBLIC repo would stay public — which
    would hand every miner the live holdout. So the repo's visibility is read back
    from the hub and a public one aborts before a single shard is uploaded.
    ``api`` is injectable so the rule is testable without a network.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo, repo_type="dataset")
    if not getattr(info, "private", False):
        raise RuntimeError(
            f"refusing to publish: dataset repo {repo} is PUBLIC. A live holdout must "
            "be private — make it private on the hub (or pick a fresh repo id) and re-run."
        )
    for path in shard_paths:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                        repo_id=repo, repo_type="dataset")
    return api.list_repo_commits(repo, repo_type="dataset")[0].commit_id


#: Where a HuggingFace write token may come from, in order. Env first so a
#: container is configured by env alone and no token is ever baked into an image.
TOKEN_ENV_VARS = ("HUGGINGFACE_TOKEN", "HF_TOKEN")

TOKEN_HELP = (
    "no HuggingFace write token found. Set HUGGINGFACE_TOKEN (or HF_TOKEN) in the "
    "environment, or put HUGGINGFACE_TOKEN=<token> in a repo-root .env file. The token "
    "needs write access to the holdout org; publishing cannot proceed without it."
)


def read_token() -> str | None:
    """The HF write token from the environment, else from a repo-root ``.env``."""
    for var in TOKEN_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.is_file():
        return None
    for line in env.read_text().splitlines():
        if line.startswith("HUGGINGFACE_TOKEN"):
            return line.split("=", 1)[1].strip()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--to-date", default=None, help="ISO end date (default: today).")
    ap.add_argument("--from-date", default=None)
    ap.add_argument("--target", type=int, default=2500)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("holdout"))
    ap.add_argument("--repo", default=None, help="HF dataset repo id (default: dated).")
    ap.add_argument("--seed", type=int, default=None, help="Plan seed (default: OS random, recorded).")
    ap.add_argument(
        "--domain", action="append", default=[], metavar="ID",
        help="OpenAlex domain id to harvest, repeatable or comma-separated; "
             "'all' (the default) takes every domain. Ids: "
             + ", ".join(f"{d}={DOMAINS[d][0]}" for d in ALL_DOMAINS),
    )
    ap.add_argument(
        "--vocab", default="general", choices=sorted(VOCABULARIES),
        help="Finding vocabulary the mintability gate reads with; must match the "
             "taskgen release this holdout feeds (SCI2 -> medical, SCI3 -> general).",
    )
    ap.add_argument("--themes", type=int, default=None,
                    help="How many themes to draw (default: the weekly 3-6). "
                         "Corpus-scale builds want most of the pool, e.g. 120.")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--mailto", default="epago@example.com")
    args = ap.parse_args()

    domains = parse_domains(args.domain)
    to_date = args.to_date or dt.date.today().isoformat()
    end = dt.date.fromisoformat(to_date)
    from_date = args.from_date or (end - dt.timedelta(days=args.days)).isoformat()
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    # The repo name states the scope: never call a four-domain slice "medicine".
    repo = args.repo or default_repo(domains, end)

    # Fail before the harvest, not after it: a multi-hour fetch that ends in
    # "no token" has burned the window and the API budget for nothing.
    token = None
    if args.publish:
        token = read_token()
        if not token:
            print(f"error: {TOKEN_HELP}", file=sys.stderr)
            return 2

    plan = plan_harvest(seed, from_date, to_date, domains, n_themes=args.themes)
    print(f"plan seed={seed}  window {from_date}..{to_date}")
    print(f"  domains={[f'{d}:{DOMAINS[d][0]}' for d in domains]}  vocab={args.vocab}")
    print(f"  sources={plan['weights']}  epmc_src='{plan['epmc_src']}'")
    themed = [f"{t} (d{plan['theme_domains'][t]})" for t in plan["themes"]]
    print(f"  themes={themed}")
    for note in plan["scope_notes"]:
        print(f"  scope: {note}")

    stats = Stats(vocab=VOCABULARIES[args.vocab])
    papers = run_plan(plan, args.target, args.mailto, stats)

    print(f"\nrecords seen   {stats.seen}")
    print(f"kept (usable)  {stats.kept}   per-source {stats.per_source}")
    print(f"  per-domain (index-classified) {stats.per_domain}")
    print(f"  rej title {stats.rej_title}  abstract {stats.rej_abstract}  "
          f"no-finding {stats.rej_no_finding}  dup {stats.rej_duplicate}")
    print(f"api failures   {stats.api_failures}")
    if not papers:
        print("no usable papers; aborting", file=sys.stderr)
        return 1

    shard_paths = write_shards(papers, args.out, args.shards)
    # Manifest: local audit only, NOT uploaded while the feed is live.
    manifest = {**plan, "kept": stats.kept, "per_source": stats.per_source,
                "per_domain": stats.per_domain, "vocab": stats.vocab.name,
                "shards": len(shard_paths), "repo": repo}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nwrote {len(shard_paths)} shards + manifest.json to {args.out}")

    revision = None
    if args.publish:
        # token was read (and required) before the harvest ran
        revision = publish(repo, shard_paths, token)
        print(f"published PRIVATE dataset: {repo} @ {revision}")

    print("\n# ---- paste into the chain contract for this week ----------------")
    print("[private_source]")
    print(f'repo        = "{repo}"')
    print(f'revision    = "{revision or "<commit-after-publish>"}"')
    print('text_column = "text"')
    print("max_shards  = 4")
    print("# keep this repo PRIVATE while live; reveal only after it rotates out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
