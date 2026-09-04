# corpus-med-10k — 7.4k-paper medical evaluation corpus

7,331 medical paper abstracts from 450 systematic reviews' reference lists
(Webis-SR4ALL-26 Medicine slice) resolved via OpenAlex.

    corpus_digest: sha256:eb34c45084c26eeebd783e85ec2d44ff36072e84404a2bea86f60928264ae9dc

Rebuild (byte-identical):

    .venv/bin/python scripts/build_corpus.py --jsonl data/corpus-med-10k/papers.jsonl --out corpus.db

Larger than corpus-med-503: retrieval is harder (more lookalike papers per
query), which pushes solve rates down toward the duel band. This is the corpus
to pin for the first live rounds.
