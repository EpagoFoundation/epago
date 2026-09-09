"""Word-sense consistency for bridge terms.

The uniqueness proof is set algebra over postings: ``docs(X) ∩ docs(Y) == {gold}``.
That is a statement about *strings*. It cannot see that ``STC`` is Standard Test
Conditions in a photovoltaics paper and slow-transit constipation in a
gastroenterology paper -- and when an anchor and the gold disagree on what an
acronym means, the task is unanswerable by anyone who understands it. Measured
on a 400-task exam drawn from the pinned corpus: 13% of tasks were confirmed
collisions, and the RL checkpoints scored 0.0% on them while the base model
scored 7.7% by lexical luck. 86% of the colliding bridges were three letters.

Scientific abstracts define an acronym on first use as ``Full Name (ACR)``.
That convention makes the check mechanical: read the expansion beside the
acronym in each paper and compare them. No model and no judgment -- an auditor
re-derives it from the corpus exactly as they re-derive uniqueness.

Three verdicts, and what each licenses:

``SAME``
    Both papers expand the acronym and the expansions share a content word.
    The bridge refers to one thing.

``COLLISION``
    Both papers expand it and the expansions share nothing. The task is
    nonsense: it asks for a paper about two things that are not in it.

``UNVERIFIED``
    At least one paper never spells the acronym out. The check cannot decide.

The two callers treat ``UNVERIFIED`` differently, on purpose. The **auditor**
fails only ``COLLISION`` -- it rejects what it can prove wrong. On the measured
exam the unverified bucket scored indistinguishably from confirmed-clean tasks
(24.5% against 23.3%), so failing all of it would discard sound tasks by the
thousand for no measured gain. The **minter** additionally refuses
``UNVERIFIED`` bridges of :data:`SHORT_ACRONYM_MAX` letters or fewer -- it
avoids what it cannot prove right, in the one class where collisions
concentrate. An auditor needs proof to condemn; a minter needs proof to admit.
"""

from __future__ import annotations

import re
from enum import Enum

#: Acronyms this short or shorter must be expansion-verified before the minter
#: will build a task on them. Three-letter acronyms were 86% of measured
#: collisions; the space of three-letter strings is small enough that two
#: fields reusing one is the norm rather than the exception.
SHORT_ACRONYM_MAX = 3

#: Function words that may sit inside an expansion without carrying meaning:
#: "Efficient Channel Attention" and "an Efficient Channel Attention" are one
#: name.
_STOP = frozenset(
    "the of and a an in for to with on by or its as at from into over under".split()
)

_WORD = re.compile(r"[A-Za-z][\w\-']*")
_CONTENT = re.compile(r"[a-z]{4,}")


class Sense(str, Enum):
    SAME = "same"
    COLLISION = "collision"
    UNVERIFIED = "unverified"
    #: A multi-word or long term is a name, not an abbreviation; two papers
    #: sharing one share a referent. The check does not apply.
    NOT_ACRONYM = "not_acronym"


def is_acronym(term: str) -> bool:
    """True for a short, single-token term -- the class that collides.

    Six alphanumerics or fewer with no space. Bridges that long or longer, or
    with a space in them, are names rather than abbreviations; the entity
    extractor's proper-phrase and technical-compound patterns only produce
    terms of four or more letters, so anything this short came from the
    acronym pattern regardless of how it was normalised.
    """
    core = re.sub(r"[^A-Za-z0-9]", "", term)
    return " " not in term.strip() and 2 <= len(core) <= 6


def _spells(letters: str, content: list[str]) -> bool:
    """Do these words spell the acronym?

    Two ways a paper spells one. By initials -- ``Efficient Channel Attention``
    → ``ECA`` -- with hyphenated words contributing each half, so
    ``Fenna-Matthews-Olson`` spells ``FMO``. Or from inside the words:
    ``trimethylamine N-oxide`` → ``TMAO`` and ``chronic thromboembolic pulmonary
    hypertension`` → ``CTEPH`` take letters that are not initials. The second
    form is accepted when the acronym's letters appear in order across the
    words *and* its first letter opens the first word: that keeps ``silicon
    carbide (SiC)`` while still refusing "the effect of its concentration
    (SiC)", where no word starts with an s.
    """
    initials = "".join(w[0].lower() for w in content)
    hyphenated = "".join(part[0].lower() for w in content for part in re.split(r"[-/]", w) if part)
    if letters in (initials, hyphenated):
        return True
    if content[0][0].lower() != letters[0]:
        return False
    stream = iter("".join(content).lower())
    return all(ch in stream for ch in letters)


def expansions(text: str, acronym: str) -> set[str]:
    """Every phrase ``text`` defines ``acronym`` as, by the ``Name (ACR)`` convention.

    A candidate is the run of words immediately before ``(ACR)`` that spells the
    acronym (see :func:`_spells`), tried from one word upward so a single
    hyphenated name is found, and allowing up to two function words inside.
    Requiring the words to actually spell the acronym is what separates a
    definition from a parenthetical that merely follows some words.

    Matching is case-insensitive and tolerates a plural ``(ACRs)``, since both
    are how papers actually write them.
    """
    letters = re.sub(r"[^a-z0-9]", "", acronym.lower())
    if not letters:
        return set()
    found: set[str] = set()
    for hit in re.finditer(r"\(\s*" + re.escape(acronym) + r"s?\s*\)", text, re.I):
        window = _WORD.findall(text[max(0, hit.start() - 160) : hit.start()])
        for width in range(1, min(len(window), len(letters) + 2) + 1):
            candidate = window[-width:]
            content = [w for w in candidate if w.lower() not in _STOP]
            if content and _spells(letters, content):
                found.add(" ".join(candidate).lower())
                break
    return found


def _content_words(expansion: str) -> set[str]:
    return {w for w in _CONTENT.findall(expansion) if w not in _STOP}


def sense_of(anchor_text: str, gold_text: str, term: str) -> Sense:
    """Does ``term`` mean the same thing in the anchor as in the gold?

    Two expansions agree when they share at least one content word: "gut
    microbiota-derived TMAO" and "trimethylamine N-oxide (TMAO)" are written
    differently but a shared "trimethylamine" settles it, while "standard test
    conditions" and "slow-transit constipation" share nothing.
    """
    if not is_acronym(term):
        return Sense.NOT_ACRONYM
    a = expansions(anchor_text, term)
    g = expansions(gold_text, term)
    if not a or not g:
        return Sense.UNVERIFIED
    for left in a:
        for right in g:
            if _content_words(left) & _content_words(right):
                return Sense.SAME
    return Sense.COLLISION


def minter_rejects(verdict: Sense, term: str) -> str | None:
    """The minter's rule: the reason to refuse a bridge, or None to admit it.

    Collisions are refused at any length. Unverified acronyms are refused only
    when short enough to sit in the class where collisions concentrate; a
    longer unverified acronym is admitted, because the measured exam showed the
    unverified bucket scoring like clean tasks and the danger is specifically
    the three-letter space.
    """
    if verdict is Sense.COLLISION:
        return "bridge_sense_collision"
    core = re.sub(r"[^A-Za-z0-9]", "", term)
    if verdict is Sense.UNVERIFIED and len(core) <= SHORT_ACRONYM_MAX:
        return "bridge_sense_unverified_short"
    return None
