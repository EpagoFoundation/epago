# corpus-med-503 — first real evaluation corpus

503 medical paper abstracts (dengue, surgery outcomes, pediatric health, ...),
built from 19 systematic reviews' reference lists via OpenAlex.

    corpus_digest: sha256:2b90ed6eb8aaac0d09d4d531b0e1100228132e0213ccf49ef7694d052d326725

| file | what |
|---|---|
| corpus.db      | pinned SqliteCorpus snapshot (FTS5) — what validators search |
| papers.jsonl   | the documents that built it ({url,title,text,category}) |
| reviews.jsonl  | the 19 source reviews: objective, eligibility criteria, and which papers they included (screening labels) |

Rebuild (byte-identical, digest included):

    .venv/bin/python scripts/build_corpus.py --jsonl data/corpus-med-503/papers.jsonl --out corpus.db

Regenerate papers.jsonl from scratch (needs network; the OpenAlex cache makes
it reproducible):

    .venv/bin/python scripts/fetch_papers.py --sr4all <sr4all_full.jsonl> \
        --out-papers papers.jsonl --out-reviews reviews.jsonl \
        --field Medicine --max-reviews 40

This is a TEST slice: production is the same pipeline over 50k-200k papers.
