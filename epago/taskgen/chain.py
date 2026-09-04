"""Two-hop chain tasks: the key to the second hop exists only inside a document.

Every SCI1-SCI4 shape hands the solver all of its keys up front. The question
states the constraints; the work is applying them. Measured consequences are in
:mod:`epago.taskgen.templates` -- SCI1 named the study, SCI3's topic words
ranked it first for 88.3% of tasks, and SCI4 closed the leak by making the
remaining difficulty procedural: follow a stated extraction rule over a named
set, carefully.

A chain task withholds a key instead. The question describes an *anchor* paper
without naming it and describes the *gold* paper without naming it, and the one
string that connects them -- a method, an instrument, a place, a named scale --
appears in neither description. It is written only inside the anchor's text. So
the route is: find the anchor from its description, read it, learn the bridge,
then use the bridge to cut the gold's crowd down to one. A solver that never
opens a document cannot guess the bridge, because nothing in the question
implies it.

That is the BrowseComp construction (start from the answer, invert into clues
that only jointly identify it) with the second hop's key hidden inside the
corpus rather than merely being obscure. The shape is chosen against measured
evidence about what separates browsing agents: handed the gold documents,
frontier and mid-size models score within about ten points of each other;
made to find those documents, they differ by twenty-five. Finding, not reading,
is the skill worth paying for.

Every guarantee below is proved by set algebra over
:class:`~epago.taskgen.entities.EntityIndex` before a task is emitted, and then
re-checked against the *real* search backend. No LLM is required to mint or to
grade; an LLM may later rewrite the question for fluency, but it rewrites a
structure that was already proved.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from epago import constants
from epago.taskgen.entities import EntityIndex, is_name_like

TEMPLATE_NAME = "bridge_chain"

# --- mint parameters (frozen with the release that ships them) ---------------

#: The gold's own description must leave at least this many candidates. If the
#: description alone identified the gold, the first hop would be decorative and
#: the task would collapse to the SCI3 shape it exists to replace.
GOLD_CROWD_MIN = 5

#: ...and at most this many, so the second hop is a narrowing step rather than
#: an enumeration over an unbounded field -- the failure that retired
#: ``comparative_finding``.
GOLD_CROWD_MAX = 400

#: Descriptions are built from this many topic terms, at most. Common words
#: narrow slowly by design -- that is the point of preferring them -- so the
#: budget has to be large enough for a conjunction of ordinary subject words
#: to reach a single document. Five was too few and rejected most candidates
#: at the narrowing step rather than on any property worth enforcing.
MAX_CLUES = 8

#: A bridge shared by more than this many documents makes the second hop an
#: enumeration. The floor matters just as much and for the opposite reason: a
#: bridge occurring in three documents nearly answers the question by itself,
#: so once the solver has it the gold's description does no work and the task
#: is a one-step lookup wearing a chain's clothes. Enforced by the index bands too,
#: repeated here because minting may run against a differently banded index.
BRIDGE_DF_MIN = 6
BRIDGE_DF_MAX = 40

#: Anchor and gold must share at least this many topic terms beyond the
#: bridge itself. A shared string is not a shared subject: "Particle System"
#: occurs in a game-engine paper and in a microfluidics paper with no relation
#: between them, and a chain built on that coincidence asks the solver to
#: follow a link that does not exist. Requiring real topical overlap is the
#: cheap programmatic proxy for a link an expert would recognise.
MIN_SUBJECT_OVERLAP = 6

#: Descriptions shorter than this read as a single keyword rather than as a
#: characterisation, and a one-clue description cannot satisfy necessity in
#: any interesting way.
MIN_CLUES = 2

#: How many of the gold's neighbours to carry through to the wording step. A
#: handful is enough to show what the description has to hold in common; more
#: makes the prompt long without changing the answer.
CROWD_SAMPLE = 5

_TOKEN = re.compile(r"[a-z0-9]+")

#: Bridges that name how a study was run rather than what it was about. The
#: verbalizer is told to reject these and mostly does, but it accepted
#: "Key Informant Interviews" linking a Lagos maternal-health study to a
#: Bangladeshi land-use study -- two papers that share an interview format and
#: nothing else. A model asked for judgement will sometimes exercise it
#: differently; a list costs nothing and does not drift.
_METHOD_BRIDGES = frozenset(
    {
        "key informant interviews", "focus group discussion",
        "focus group discussions", "semi structured interview",
        "semi-structured interview", "in depth interview",
        "in-depth interview", "likert scale", "chi square", "chi-square",
        "cross sectional", "cross-sectional", "randomized controlled trial",
        "systematic review", "meta analysis", "meta-analysis",
        "content analysis", "thematic analysis", "grounded theory",
        "purposive sampling", "snowball sampling", "convenience sampling",
        "descriptive statistics", "informed consent", "ethical approval",
        "questionnaire survey", "online survey", "pilot study",
        "data collection", "data analysis", "statistical analysis",
        "literature review", "case study", "case report", "case series",
        "quality control", "quality assurance", "sensitivity analysis",
        "principal component analysis", "regression analysis",
        "logistic regression", "linear regression", "machine learning",
        "deep learning", "artificial intelligence", "neural network",
        "neural networks",
    }
)

#: Residual markup from the ingest ("i Toxoplasma gondii /i"), non-Latin
#: scripts, and words fused by a lost space all make an exact-match answer
#: unfair in a way the solver cannot fix: the title it reads in the corpus is
#: not a title anyone would type. Measured on the pinned corpus, 1.87% of
#: titles carry markup residue, 0.48% have a fused word and 0.16% are not
#: Latin script. Those documents stay in the corpus and stay usable as
#: anchors; they are simply never the answer.
_MARKUP_RESIDUE = re.compile(r"(?:^|\s)/?(?:i|b|sub|sup|em|strong|scp|it)(?:\s|$)")
_FUSED_WORD = re.compile(r"[a-z]{3}[A-Z][a-z]{3}")

#: Titles outside this length are unusable as an exact-match answer for the
#: opposite reasons: too short is not a title, too long cannot be retyped. The
#: ceiling is the protocol's own answer cap rather than a number chosen here,
#: because a task whose answer exceeds it is rejected by the task-QA form check
#: after minting -- which would discard the work silently and late.
TITLE_MIN_CHARS = 20
TITLE_MAX_CHARS = constants.ANSWER_MAX_CHARS


def _latin_fraction(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(
        1 for ch in letters if "LATIN" in unicodedata.name(ch, "")
    ) / len(letters)


def _title_is_clean(title: str) -> bool:
    """Shared hygiene: no ingest markup, no fused words, mostly Latin script."""
    if _MARKUP_RESIDUE.search(title):
        return False
    if _FUSED_WORD.search(title):
        return False
    return _latin_fraction(title) >= 0.9


def usable_as_answer(title: str) -> bool:
    """True when a title can fairly be demanded back, character for character."""
    if not title or not (TITLE_MIN_CHARS <= len(title) <= TITLE_MAX_CHARS):
        return False
    return _title_is_clean(title)


def printable_in_a_question(title: str) -> bool:
    """True when a title can be quoted in a question.

    Deliberately not :func:`usable_as_answer`. That function's minimum length
    exists so a solver is not asked to reproduce a title too short to be
    distinctive -- an answerability rule. An anchor title is printed, never
    typed back, so the minimum does not apply; what still applies is that it be
    clean and not so long it swamps the question.
    """
    return bool(title) and len(title) <= TITLE_MAX_CHARS and _title_is_clean(title)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


@dataclass(frozen=True, slots=True)
class ChainSkeleton:
    """A proved chain, before it is worded.

    Wording is deliberately a separate step: the structure carries the
    guarantees, and any verbalizer -- template or LLM -- may only rephrase
    what is already proved, never widen it.
    """

    anchor_doc_id: str
    gold_doc_id: str
    bridge: str
    anchor_clues: tuple[str, ...]
    gold_clues: tuple[str, ...]
    gold_crowd: int
    #: A sample of the papers the gold's description must fit *besides* the
    #: gold. These are the documents the second hop is choosing between, and
    #: they are carried on the skeleton because the wording step needs them:
    #: a description written from the gold alone inevitably describes only the
    #: gold, and then search finds it without the bridge. Measured on a full
    #: mint, that single failure accounted for 56% of everything the wording
    #: step produced.
    gold_crowd_sample: tuple[str, ...]
    bridge_df: int
    #: Every check that passed, so a rejected variant can be diagnosed and an
    #: emitted task can be audited without re-deriving the proof.
    proof: dict


class ChainMinter:
    """Mints :class:`ChainSkeleton` values that satisfy every structural rule.

    The rules are the ones that recur across every published hard-task
    pipeline, stated here as operations on postings lists:

    uniqueness
        The anchor's clues intersect to exactly the anchor. The gold's clues
        intersected with the bridge's postings give exactly the gold.

    necessity
        Dropping any single clue leaves more than one candidate, so no proper
        subset of the question identifies its target. Without this a solver
        can ignore most of the question -- the "single-clue selectivity"
        shortcut.

    crowd
        Each clue on its own matches many documents. A clue that matches few
        is a fingerprint of its own document, not a filter.

    hop necessity
        The gold's own clues leave a crowd of at least ``GOLD_CROWD_MIN``.
        This is what makes the first hop load-bearing.

    disjointness
        The anchor does not satisfy the gold's description and vice versa, so
        the two halves of the question cannot be answered by one document.

    concealment
        No token of the bridge appears anywhere in the question, and no token
        of the answer does either.
    """

    def __init__(
        self,
        index: EntityIndex,
        titles: dict[str, str],
        *,
        rng: np.random.Generator,
        min_subject_overlap: int = MIN_SUBJECT_OVERLAP,
    ) -> None:
        self._idx = index
        self._titles = titles
        self._rng = rng
        self._min_overlap = min_subject_overlap
        self._title_terms: dict[str, frozenset[str]] = {}

    def _title_topics(self, doc_id: str) -> frozenset[str]:
        """Topic terms that appear in the document's own title.

        A title states what a paper is about in the author's own words, so
        title terms make descriptions that read like a subject rather than
        like a word list. Abstract-only terms are still usable, but they rank
        below these.
        """
        cached = self._title_terms.get(doc_id)
        if cached is None:
            title_tokens = _tokens(self._titles.get(doc_id, ""))
            cached = frozenset(t for t in self._idx.topics_of(doc_id) if t in title_tokens)
            self._title_terms[doc_id] = cached
        return cached

    # -- clue selection -------------------------------------------------------

    def _choose_clues(
        self,
        target: str,
        *,
        universe: frozenset[str] | None = None,
        want_singleton: bool,
        floor: int = 1,
    ) -> tuple[tuple[str, ...], frozenset[str]] | None:
        """Greedily narrow ``universe`` toward ``target`` using topic terms.

        Terms are added most-discriminating first, which reaches a singleton
        in the fewest clues; then every clue is tested for necessity and the
        redundant ones are dropped. Greedy-then-prune is used rather than an
        exhaustive search because the postings are large and the pruning pass
        restores the property an exhaustive search would have given.

        ``floor`` stops the narrowing early: for the gold half we deliberately
        want a crowd left over, not a singleton, so the bridge has work to do.
        """
        terms = sorted(self._idx.topics_of(target))
        if not terms:
            return None
        # Ordering is the whole difference between a description and a word
        # list. Title terms first, then commonest first -- the SCI2 rule that
        # a descriptor must match a crowd, so that only the conjunction is
        # unique. Ordering by rarity instead picks whichever odd word the
        # abstract happened to use ("aerated", "mecca"), which narrows in one
        # step and reads like nothing a researcher would ask.
        in_title = self._title_topics(target)
        terms.sort(
            key=lambda t: (
                0 if t in in_title else 1,
                -len(self._idx.docs_with_topic(t)),
                t,
            )
        )

        acc = universe if universe is not None else None
        chosen: list[str] = []
        for term in terms:
            if len(chosen) >= MAX_CLUES:
                break
            posting = self._idx.docs_with_topic(term)
            nxt = posting if acc is None else (acc & posting)
            if target not in nxt:
                continue  # cannot happen for the target's own terms; defensive
            if len(nxt) < floor:
                continue  # would narrow past the crowd we must keep
            acc = nxt
            chosen.append(term)
            if want_singleton and len(acc) == 1:
                break
            if not want_singleton and len(acc) <= floor:
                break
        if acc is None or not chosen:
            return None
        if want_singleton and acc != frozenset({target}):
            return None

        # Necessity: drop any clue whose removal does not widen the result.
        pruned = list(chosen)
        for term in list(chosen):
            trial = [t for t in pruned if t != term]
            if len(trial) < MIN_CLUES:
                continue
            got = self._idx.docs_with_all_topics(trial)
            if universe is not None:
                got = got & universe
            if got == acc:
                pruned = trial  # redundant clue: the answer set is unchanged
        return tuple(pruned), acc

    # -- the proof ------------------------------------------------------------

    def build(
        self,
        anchor: str,
        gold: str,
        bridge: str,
        *,
        reasons: "Counter[str] | None" = None,
    ) -> ChainSkeleton | None:
        """Prove one candidate chain, or return ``None``.

        Ordering is cheapest-check-first so a rejected candidate costs little;
        minting rejects far more candidates than it emits. Pass ``reasons`` to
        accumulate why candidates died: a mint whose yield falls has either
        got stricter about something real or started failing on a mechanical
        step, and only the breakdown tells the two apart.
        """
        def _no(reason: str) -> None:
            if reasons is not None:
                reasons[reason] += 1
            return None

        idx = self._idx
        if anchor == gold:
            return _no("same_doc")
        if not is_name_like(bridge):
            return _no("bridge_not_a_name")
        if bridge.lower() in _METHOD_BRIDGES:
            return _no("bridge_is_methodology")
        bridge_posting = idx.docs_with_bridge(bridge)
        if not (BRIDGE_DF_MIN <= len(bridge_posting) <= BRIDGE_DF_MAX):
            return _no("bridge_df")
        if anchor not in bridge_posting or gold not in bridge_posting:
            return _no("bridge_missing")

        # A shared string is not a shared subject. Reject coincidental bridges
        # before doing any of the expensive narrowing work.
        shared_subject = self._idx.topics_of(anchor) & self._idx.topics_of(gold)
        if len(shared_subject) < self._min_overlap:
            return _no("subject_overlap")

        # Anchor half: its clues must identify it outright, or the solver
        # cannot know which document to read for the bridge.
        got = self._choose_clues(anchor, want_singleton=True)
        if got is None:
            return _no("anchor_not_unique")
        anchor_clues, anchor_set = got
        if len(anchor_clues) < MIN_CLUES:
            return _no("anchor_too_few_clues")

        # Gold half. The clues are chosen *under the bridge*, because that is
        # the state the solver is in when it uses them: it has read the anchor
        # and knows the bridge, and now has to pick the gold out of the
        # bridge's crowd. Choosing them without the bridge and then testing
        # necessity against it is the same mistake as pricing a filter against
        # the wrong population -- nearly every clue looks redundant, and the
        # candidate is thrown away for a property it was never asked for.
        got = self._choose_clues(gold, universe=bridge_posting, want_singleton=True)
        if got is None:
            return _no("chain_not_unique")
        gold_clues, resolved = got
        if resolved != frozenset({gold}):
            return _no("chain_not_unique")

        # Hop necessity, and the single most important check in this class:
        # the gold's own description, without the bridge, must still leave a
        # crowd. If it does not, the solver can skip the anchor entirely and
        # the task has silently degraded into the one-hop SCI3 shape.
        gold_crowd_set = idx.docs_with_all_topics(gold_clues)
        if len(gold_crowd_set) < GOLD_CROWD_MIN:
            return _no("gold_findable_without_bridge")
        if len(gold_crowd_set) > GOLD_CROWD_MAX:
            return _no("gold_crowd_band")

        # Disjointness: neither half may be satisfied by the other's document.
        if gold in anchor_set or anchor in gold_crowd_set:
            return _no("halves_overlap")

        # Concealment: the bridge must not be derivable from the question's
        # own words, and the answer must not appear in them.
        question_tokens = _tokens(" ".join(anchor_clues) + " " + " ".join(gold_clues))
        if _tokens(bridge) & question_tokens:
            return _no("bridge_exposed")
        answer = self._titles.get(gold, "")
        if not answer:
            return _no("no_title")
        if not usable_as_answer(answer):
            return _no("title_not_answerable")
        anchor_title = self._titles.get(anchor, "")
        # The anchor's title is not in the question either -- naming it would
        # turn hop one into a lookup, which is the SCI1 leak.
        if _tokens(anchor_title) and _tokens(anchor_title) <= question_tokens:
            return _no("anchor_title_exposed")

        # Neighbours the gold shares its description with, excluding anything
        # that also carries the bridge: those are second-hop candidates, not
        # crowd, and showing them to the wording step would invite a
        # description that fails to discriminate where it must.
        crowd_sample = tuple(
            sorted(gold_crowd_set - bridge_posting - {gold})[:CROWD_SAMPLE]
        )

        return ChainSkeleton(
            anchor_doc_id=anchor,
            gold_doc_id=gold,
            bridge=bridge,
            anchor_clues=anchor_clues,
            gold_clues=gold_clues,
            gold_crowd=len(gold_crowd_set),
            gold_crowd_sample=crowd_sample,
            bridge_df=len(bridge_posting),
            proof={
                "anchor_unique": True,
                "gold_crowd_before_bridge": len(gold_crowd_set),
                "gold_unique_after_bridge": True,
                "clues_necessary": True,
                "bridge_concealed": True,
                "halves_disjoint": True,
                "subject_overlap": len(shared_subject),
            },
        )

    # -- candidate generation -------------------------------------------------

    def candidates(self, n: int) -> list[tuple[str, str, str]]:
        """Draw ``n`` ``(anchor, gold, bridge)`` triples to attempt.

        Bridges are drawn first and their document pairs second, so the yield
        is spread across bridge terms instead of concentrating on whichever
        documents happen to be term-rich.
        """
        idx = self._idx
        usable = [
            t
            for t, d in idx.bridge_docs.items()
            if BRIDGE_DF_MIN <= len(d) <= BRIDGE_DF_MAX and is_name_like(t)
        ]
        usable.sort()
        if not usable:
            return []
        out: list[tuple[str, str, str]] = []
        picks = self._rng.integers(0, len(usable), size=n)
        for i in picks:
            bridge = usable[int(i)]
            docs = sorted(idx.docs_with_bridge(bridge))
            if len(docs) < 2:
                continue
            a, b = self._rng.choice(len(docs), size=2, replace=False)
            out.append((docs[int(a)], docs[int(b)], bridge))
        return out


# --- route verification against the real search backend ----------------------
#
# Everything above is proved over exact postings lists. The solver does not get
# postings lists; it gets BM25 over an OR-bag of query terms. Those two are not
# the same object, and a proof in one is not a promise in the other: a
# conjunction that intersects to a single document can still fail to rank that
# document anywhere near the top.
#
# SCI4 was shipped on a leak measurement alone -- nobody checked whether its
# gold documents were reachable with the tools the agent actually has, and
# 35.1% of the resulting exam turned out to be solvable by no model at all.
# Those dead tasks are not neutral: the coronation bar is a function of the
# king's accuracy, so every unsolvable task raises the margin an honest
# challenger has to clear. The checks below are the fix, and they run at mint.

#: Where the anchor must appear when its own description is searched. This is
#: the solvability floor: if the first hop cannot be taken, nothing downstream
#: matters.
ANCHOR_FIND_K = 10

#: Where the gold must appear once the bridge is known. The second hop has to
#: be takeable too.
GOLD_FIND_K = 10

#: Where the gold must *not* appear for the question as written. Pasting the
#: whole question into search is the first thing a solver tries; if that works,
#: the chain was decorative.
GOLD_LEAK_K = 3

#: Where the gold must not appear for its own description alone -- the check
#: that decides whether the task is genuinely two-hop.
#:
#: This was 10, and 10 is the wrong number. It rejected 56% of everything the
#: wording step produced, and the only way for a description to satisfy it was
#: to stop describing: "examines respiratory health and outcomes in pediatric
#: and adult populations" clears a top-10 rule and identifies nothing, which
#: makes the task unanswerable rather than hard.
#:
#: What the guard is actually for is the case where a solver can skip the
#: first hop and still be right. Measured on a batch where 99.2% of golds sat
#: in the top 10 for their own description, the model visited the anchor in
#: 86.7% of episodes and only 11.2% of its correct answers came without
#: reading it. So being *retrievable* alongside similar papers is not the
#: failure; being retrievable *first*, so that a greedy solver needs no second
#: thought, is. Rank 1 is therefore the line, and it is paired with a semantic
#: check that the description leaves a real choice open -- see
#: :mod:`epago.taskgen.ambiguity`. Rank alone was never the right question.
GOLD_SHORTCUT_K = 1


@dataclass(frozen=True, slots=True)
class RouteReport:
    """Where each document ranked, and whether the intended route survives."""

    anchor_rank: int | None
    gold_rank_via_bridge: int | None
    gold_rank_on_question: int | None
    gold_rank_on_own_clues: int | None
    ok: bool
    failure: str | None


def _rank_of(hits: Sequence, doc_id: str) -> int | None:
    for i, hit in enumerate(hits, start=1):
        if hit.doc_id == doc_id:
            return i
    return None


def verify_route(
    skeleton: ChainSkeleton,
    corpus,
    question: str,
    *,
    anchor_query: str | None = None,
    gold_query: str | None = None,
) -> RouteReport:
    """Check the intended route is takeable and every shortcut is closed.

    ``corpus`` is any object exposing the harness's ``search(query, k)``. The
    same backend the rollout will use is passed deliberately: a mint-time check
    against a different retriever than the agent gets would prove nothing.

    ``anchor_query`` and ``gold_query`` must be the descriptions *as the solver
    will read them*. This matters more than it looks. The skeleton's clue terms
    are what the uniqueness proof was written over, but they are never shown to
    anyone; the question carries a paraphrase instead. Checking the shortcut
    against the clue terms therefore measures a string the solver cannot type,
    and a paraphrase that restates the gold's title sails through -- observed
    on a minted batch, where a description reading "investigated how aflatoxin
    B1 impairs chemoembolization efficacy through downregulated carbonic
    anhydrase 2" was accepted against a paper of almost exactly that name. The
    fallback to clue terms exists only for the pre-verification pass, which
    runs before any wording exists.
    """
    anchor_q = anchor_query or " ".join(skeleton.anchor_clues)
    gold_q = gold_query or " ".join(skeleton.gold_clues)
    bridge_q = f"{skeleton.bridge} {gold_q}"

    a_rank = _rank_of(corpus.search(anchor_q, k=ANCHOR_FIND_K), skeleton.anchor_doc_id)
    g_bridge = _rank_of(corpus.search(bridge_q, k=GOLD_FIND_K), skeleton.gold_doc_id)
    g_question = _rank_of(corpus.search(question, k=GOLD_LEAK_K), skeleton.gold_doc_id)
    g_own = _rank_of(corpus.search(gold_q, k=GOLD_SHORTCUT_K), skeleton.gold_doc_id)

    failure = None
    if a_rank is None:
        failure = "anchor_unreachable"
    elif g_bridge is None:
        failure = "gold_unreachable_via_bridge"
    elif g_question is not None:
        failure = "gold_leaks_from_question"
    # `gold_rank_on_own_clues` is recorded but no longer decides anything. It
    # was a rank rule standing in for a semantic question -- can a solver
    # decide from the description alone? -- and it answered that question
    # badly in both directions. It is kept as a diagnostic because the rank is
    # worth knowing when a batch is audited.
    return RouteReport(
        anchor_rank=a_rank,
        gold_rank_via_bridge=g_bridge,
        gold_rank_on_question=g_question,
        gold_rank_on_own_clues=g_own,
        ok=failure is None,
        failure=failure,
    )


# --- the intersection shape --------------------------------------------------
#
# The chain above describes the gold and lets a bridge narrow that description
# to one paper. Measured across three full mints, that is not achievable: a
# description specific enough for a reader to decide on is specific enough for
# BM25 to retrieve, and 52-58% of candidates died on exactly that, in every
# phrasing tried. Pushing the wording more general produced sentences like
# "examines respiratory health and outcomes in pediatric and adult
# populations", which pass every leak check because they identify nothing at
# all. The two demands are not both satisfiable by one sentence.
#
# So this shape stops describing the gold. The question describes two *other*
# papers, each of which names something, and asks for the single paper that
# involves both. Nothing in the question is a query for the answer, because the
# answer is never characterised -- it is defined by an intersection the solver
# can only compute after reading both anchors. Uniqueness is then not a
# judgement about wording at all; it is `|docs(X) & docs(Y)| == 1`, decided on
# postings and provable at mint.
#
# This is the set-theoretic construction the formal synthesis work argues for:
# state the task as an intersection of relations and let the structure carry
# the guarantee, rather than collecting information first and hoping the
# question that describes it happens to be well posed.

INTERSECTION_TEMPLATE_NAME = "bridge_intersection"

#: Each bridge must leave a real candidate set. The pair intersects to one
#: paper by construction, so the work is in discovering *which* two things to
#: intersect, not in the intersection itself.
PAIR_DF_MIN = 4
PAIR_DF_MAX = 40


@dataclass(frozen=True, slots=True)
class IntersectionSkeleton:
    """Two anchors, two bridges, and the one paper carrying both."""

    anchor_a_doc_id: str
    anchor_b_doc_id: str
    gold_doc_id: str
    bridge_x: str
    bridge_y: str
    anchor_a_clues: tuple[str, ...]
    anchor_b_clues: tuple[str, ...]
    bridge_x_df: int
    bridge_y_df: int
    proof: dict


class IntersectionMinter(ChainMinter):
    """Mints :class:`IntersectionSkeleton` values.

    Inherits the clue chooser and the answer-eligibility rules from
    :class:`ChainMinter`; what changes is what has to be proved.
    """

    def build_intersection(
        self,
        gold: str,
        bridge_x: str,
        bridge_y: str,
        *,
        reasons: "Counter[str] | None" = None,
    ) -> IntersectionSkeleton | None:
        def _no(reason: str) -> None:
            if reasons is not None:
                reasons[reason] += 1
            return None

        idx = self._idx
        if bridge_x == bridge_y:
            return _no("same_bridge")
        for term in (bridge_x, bridge_y):
            if not is_name_like(term):
                return _no("bridge_not_a_name")
            if term.lower() in _METHOD_BRIDGES:
                return _no("bridge_is_methodology")

        docs_x = idx.docs_with_bridge(bridge_x)
        docs_y = idx.docs_with_bridge(bridge_y)
        if not (PAIR_DF_MIN <= len(docs_x) <= PAIR_DF_MAX):
            return _no("bridge_x_df")
        if not (PAIR_DF_MIN <= len(docs_y) <= PAIR_DF_MAX):
            return _no("bridge_y_df")

        # The whole uniqueness guarantee, in one line.
        if (docs_x & docs_y) != frozenset({gold}):
            return _no("intersection_not_unique")

        answer = self._titles.get(gold, "")
        if not answer:
            return _no("no_title")
        if not usable_as_answer(answer):
            return _no("title_not_answerable")

        # Anchors: one paper carrying each bridge, neither of them the answer.
        # An anchor that carries *both* bridges would hand the solver the whole
        # intersection in a single read, which is one hop, not two.
        cand_a = sorted(docs_x - {gold} - docs_y)
        cand_b = sorted(docs_y - {gold} - docs_x)
        if not cand_a or not cand_b:
            return _no("no_disjoint_anchors")

        chosen_a = chosen_b = None
        clues_a: tuple[str, ...] = ()
        clues_b: tuple[str, ...] = ()
        for doc in cand_a:
            got = self._choose_clues(doc, want_singleton=True)
            if got and len(got[0]) >= MIN_CLUES:
                chosen_a, clues_a = doc, got[0]
                break
        if chosen_a is None:
            return _no("anchor_a_not_unique")
        for doc in cand_b:
            if doc == chosen_a:
                continue
            got = self._choose_clues(doc, want_singleton=True)
            if got and len(got[0]) >= MIN_CLUES:
                chosen_b, clues_b = doc, got[0]
                break
        if chosen_b is None:
            return _no("anchor_b_not_unique")

        # Concealment. Neither bridge may be spelled by the clues, and the
        # answer's own title must not be assembled out of them either.
        question_tokens = _tokens(" ".join(clues_a) + " " + " ".join(clues_b))
        if _tokens(bridge_x) & question_tokens or _tokens(bridge_y) & question_tokens:
            return _no("bridge_exposed")
        if _tokens(answer) and _tokens(answer) <= question_tokens:
            return _no("answer_exposed")

        return IntersectionSkeleton(
            anchor_a_doc_id=chosen_a,
            anchor_b_doc_id=chosen_b,
            gold_doc_id=gold,
            bridge_x=bridge_x,
            bridge_y=bridge_y,
            anchor_a_clues=clues_a,
            anchor_b_clues=clues_b,
            bridge_x_df=len(docs_x),
            bridge_y_df=len(docs_y),
            proof={
                "intersection_size": 1,
                "bridge_x_df": len(docs_x),
                "bridge_y_df": len(docs_y),
                "anchors_disjoint": True,
                "bridges_concealed": True,
            },
        )

    def intersection_candidates(self, n: int) -> list[tuple[str, str, str]]:
        """Draw ``(gold, bridge_x, bridge_y)`` triples worth attempting.

        Golds are drawn from papers carrying at least two usable bridges, which
        is 19,234 of the pinned corpus -- a larger mintable pool than the SCI4
        templates ever had.
        """
        idx = self._idx
        by_doc: dict[str, list[str]] = {}
        for term, docs in idx.bridge_docs.items():
            if not (PAIR_DF_MIN <= len(docs) <= PAIR_DF_MAX):
                continue
            if not is_name_like(term) or term.lower() in _METHOD_BRIDGES:
                continue
            for doc in docs:
                by_doc.setdefault(doc, []).append(term)
        eligible = sorted(d for d, terms in by_doc.items() if len(terms) >= 2)
        if not eligible:
            return []
        # Distinct triples only. Golds are drawn with replacement -- a paper
        # with many bridges should get many chances -- but the same
        # (gold, x, y) pair drawn twice would prove, word and emit twice, and
        # two copies of one task in a pool is worse than a wasted draw: a pool
        # is rejected outright for repeating a task id, because selection has
        # to stay reproducible from the manifest's id list.
        #
        # Oversampling covers the collisions rather than looping forever: the
        # caller asks for a multiple of what it needs, so returning slightly
        # fewer than ``n`` on a saturated corpus is correct behaviour, not a
        # failure.
        out: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        picks = self._rng.integers(0, len(eligible), size=n * 2)
        for i in picks:
            if len(out) >= n:
                break
            gold = eligible[int(i)]
            terms = sorted(by_doc[gold])
            a, b = self._rng.choice(len(terms), size=2, replace=False)
            triple = (gold, terms[int(a)], terms[int(b)])
            if triple in seen:
                continue
            seen.add(triple)
            out.append(triple)
        return out


def verify_intersection_route(
    skeleton: IntersectionSkeleton,
    corpus,
    question: str,
    *,
    anchor_a_query: str,
    anchor_b_query: str,
) -> RouteReport:
    """Both anchors must be findable, and the question must not find the answer.

    There is no shortcut check here, and that absence is the point: the
    question contains no description of the gold, so there is no query a solver
    can lift out of it. What replaced a leak check is a structural property.
    """
    a_rank = _rank_of(corpus.search(anchor_a_query, k=ANCHOR_FIND_K), skeleton.anchor_a_doc_id)
    b_rank = _rank_of(corpus.search(anchor_b_query, k=ANCHOR_FIND_K), skeleton.anchor_b_doc_id)
    pair_q = f"{skeleton.bridge_x} {skeleton.bridge_y}"
    g_pair = _rank_of(corpus.search(pair_q, k=GOLD_FIND_K), skeleton.gold_doc_id)
    g_question = _rank_of(corpus.search(question, k=GOLD_LEAK_K), skeleton.gold_doc_id)

    failure = None
    if a_rank is None:
        failure = "anchor_a_unreachable"
    elif b_rank is None:
        failure = "anchor_b_unreachable"
    elif g_pair is None:
        failure = "gold_unreachable_from_pair"
    elif g_question is not None:
        failure = "gold_leaks_from_question"
    return RouteReport(
        anchor_rank=a_rank,
        gold_rank_via_bridge=g_pair,
        gold_rank_on_question=g_question,
        gold_rank_on_own_clues=b_rank,
        ok=failure is None,
        failure=failure,
    )


# --- difficulty tiers --------------------------------------------------------
#
# One shape, three lengths. Measured on 394 tasks of the longest form, the base
# model answered 39% of episodes and was right 51.6% of those -- but 78% of
# tasks went unsolved, and rl5 beat base by 0.8% against a noise band of about
# 2%. An exam whose items are all beyond the models cannot rank them: an item
# nobody solves contributes a paired difference of exactly zero to every duel,
# forever, exactly like an item everybody solves.
#
# So the anchors can be *named* instead of described. Naming an anchor removes
# a find-and-identify step while leaving the answer exactly as hidden as
# before, because an anchor is never the answer. This is not SCI1's leak, which
# named the study being asked for.
#
#   named_both     both anchors named. Two reads, then the intersection.
#   named_one      one named, one described. Three steps.
#   described_both neither named. The full form measured above.
#
# The mixture is what makes the exam informative across a range of ability
# rather than at one point on it.
INTERSECTION_TIERS = ("named_both", "named_one", "described_both")


def can_name_anchor(skeleton: "IntersectionSkeleton", which: str, titles: dict[str, str]) -> bool:
    """True when an anchor's title can be quoted without giving anything away.

    A title is only safe to print if it spells neither bridge -- a paper called
    "WPS treatment of fresh-cut potatoes" hands over the very term the reader is
    supposed to go and find -- and does not spell the answer.
    """
    doc_id = skeleton.anchor_a_doc_id if which == "a" else skeleton.anchor_b_doc_id
    title = titles.get(doc_id, "")
    if not printable_in_a_question(title):
        return False
    tokens = _tokens(title)
    if _tokens(skeleton.bridge_x) & tokens or _tokens(skeleton.bridge_y) & tokens:
        return False
    answer_tokens = _tokens(titles.get(skeleton.gold_doc_id, ""))
    return not (answer_tokens and answer_tokens <= tokens)


def nameable_anchors(
    skeleton: "IntersectionSkeleton", titles: dict[str, str]
) -> tuple[bool, bool]:
    """Which of the two anchors may have its title printed."""
    return (
        can_name_anchor(skeleton, "a", titles),
        can_name_anchor(skeleton, "b", titles),
    )


def tiers_available(skeleton: "IntersectionSkeleton", titles: dict[str, str]) -> tuple[str, ...]:
    """Which tiers this skeleton can support, hardest last."""
    a_ok, b_ok = nameable_anchors(skeleton, titles)
    out = ["described_both"]
    if a_ok or b_ok:
        out.append("named_one")
    if a_ok and b_ok:
        out.append("named_both")
    return tuple(reversed(out))
