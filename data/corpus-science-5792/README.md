# corpus-science-5792 — all-science duel corpus

5,792 paper abstracts spanning **all four OpenAlex domains**, balanced 1,450 per
domain before dedup (Life, Social, Physical, Health Sciences). This is the
corpus the SCI3 release reads: the templates are field-neutral, so widening the
corpus is the only change needed to cover all of scientific literature.

    corpus_digest: sha256:ceee9ec5abd5a755bc7524ddcd4036c9d8f90121f5f6fb562a3614b55ddc4043

Rebuild (byte-identical):

    .venv/bin/python scripts/build_corpus.py \
        --jsonl data/corpus-science-5792/papers.jsonl --out corpus.db

## Provenance

Harvested with `scripts/harvest_holdout.py`'s fetchers over the window
**2025-01-01 .. 2026-06-30**, deliberately ending before the weekly private
holdout window so the frozen duel corpus and the fresh holdout never overlap.
Every theme in `THEMES_BY_DOMAIN` was queried and the mintability gate ran with
the SCI3 (`general`) finding vocabulary.

  * **OpenAlex** (1,683 kept) — domain-pinned queries, so the domain label on
    each record is the index's own `primary_topic.domain` classification.
  * **Crossref** (4,109 kept) — the broad cross-publisher index. Crossref carries
    no domain field, so the domain label on those records is *theme-derived*
    (the queried theme's own domain), not an index classification.

The `category` column records `source:theme|Domain` for every document, so the
two labelling regimes stay distinguishable.

Replaces the medicine-only `corpus-med-503` / `corpus-med-10k` slices for
SCI3 rounds.
