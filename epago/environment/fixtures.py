"""Deterministic synthetic corpus for tests and soaks.

Documents describe a closed world of fictional people and cities with simple,
machine-checkable relations (birth city, birth year, invention, founding year),
so taskgen and eval tests can mint tasks whose answers are verifiable against
the corpus itself. Generation is pure-seeded — no wall clock, no filesystem
state — so the same ``(n_docs, seed)`` always yields the same documents and
the same snapshot digest.
"""

from __future__ import annotations

import random
from pathlib import Path

from epago.environment.corpus import Document, SqliteCorpus

_SYLLABLES = (
    "bar", "cor", "del", "fen", "gal", "hol", "jas", "kel",
    "lor", "mar", "nev", "oli", "pra", "quil", "ras", "sel",
    "tam", "ulm", "vor", "wes", "yal", "zeb",
)
_CITY_SUFFIXES = ("burg", "ford", "haven", "mont", "port", "stad", "vale", "wick")
_PROFESSIONS = (
    "astronomer", "botanist", "cartographer", "composer",
    "engineer", "historian", "mathematician", "physician",
)
_INVENTIONS = (
    "chronometric loom", "dual-lens astrolabe", "glass harmonium",
    "pendulum seismograph", "rotary printing frame", "spring-driven orrery",
    "tidal mill governor", "vacuum bell furnace",
)
_LANDMARKS = ("clock tower", "grand aqueduct", "iron bridge", "observatory", "stone amphitheatre")
_REGIONS = ("northern highlands", "eastern marshes", "southern coast", "western plateau")


def _unique_name(rng: random.Random, n_syllables: int, taken: set[str]) -> str:
    while True:
        name = "".join(rng.choice(_SYLLABLES) for _ in range(n_syllables)).capitalize()
        if name not in taken:
            taken.add(name)
            return name


def build_fixture_corpus(path: Path, n_docs: int = 200, seed: int = 7) -> SqliteCorpus:
    """Build a synthetic corpus of ``n_docs`` documents at ``path``."""
    if n_docs < 8:
        raise ValueError(f"fixture corpus needs at least 8 docs, got {n_docs}")
    rng = random.Random(seed)
    taken: set[str] = set()
    n_cities = max(4, n_docs // 10)
    n_people = n_docs - n_cities

    cities: list[tuple[str, int, str, str]] = []  # (name, founded, region, landmark)
    for _ in range(n_cities):
        cities.append(
            (
                _unique_name(rng, 2, taken) + rng.choice(_CITY_SUFFIXES),
                rng.randint(1400, 1800),
                rng.choice(_REGIONS),
                rng.choice(_LANDMARKS),
            )
        )

    docs: list[Document] = []
    for i, (city, founded, region, landmark) in enumerate(cities):
        doc_id = f"fx-{i:05d}"
        docs.append(
            Document(
                doc_id=doc_id,
                url=f"https://corpus.epago.local/{doc_id}",
                title=f"{city} (city)",
                text=(
                    f"{city} is a city in the {region}, founded in {founded}. "
                    f"Its best-known landmark is the {landmark}, and the city has long "
                    f"served as a regional centre for trade and scholarship."
                ),
                category="city",
            )
        )

    for j in range(n_people):
        doc_id = f"fx-{n_cities + j:05d}"
        first = _unique_name(rng, 2, taken)
        last = _unique_name(rng, 2, taken)
        name = f"{first} {last}"
        birth_city = rng.choice(cities)[0]
        birth_year = rng.randint(1500, 1900)
        profession = rng.choice(_PROFESSIONS)
        invention = rng.choice(_INVENTIONS)
        invention_year = birth_year + rng.randint(25, 55)
        docs.append(
            Document(
                doc_id=doc_id,
                url=f"https://corpus.epago.local/{doc_id}",
                title=name,
                text=(
                    f"{name} was a {profession} born in {birth_city} in {birth_year}. "
                    f"{first} is best remembered for the {invention}, completed in "
                    f"{invention_year}, which drew scholars from across the {rng.choice(_REGIONS)}. "
                    f"{last}'s notebooks are preserved in the {birth_city} archive."
                ),
                category="person",
            )
        )

    corpus = SqliteCorpus.create(path)
    corpus.add_documents(docs)
    return corpus
