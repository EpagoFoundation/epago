"""Does the question have exactly one answer once a reader, not a regex, reads it?

The mint proves uniqueness over *clue terms*: the gold's topic words,
intersected with the bridge's postings, give one document. The solver never
sees those terms. It sees a paraphrase, and a paraphrase can drift -- widen
into a sentence that fits three papers, or narrow onto a detail the gold does
not actually claim. Nothing in the set algebra notices, because the set algebra
was applied to different words.

So this asks the question the way the solver will meet it. The bridge's
document set is small and known, so the whole candidate space of the second hop
can be laid out and re-solved directly: given the description and the titles of
every paper carrying the bridge, which one is meant? Three outcomes matter and
they mean different things.

*It picks the gold.* The description discriminates, and the second hop is
takeable by something other than luck.

*It picks another paper, or several.* The question has more than one defensible
answer. That is not a hard task, it is a broken one, and a solver that answers
it correctly by the key's lights did so by guessing which reading was intended.

*It picks nothing.* The description does not fit its own paper well enough to be
recognised, so the task is unfair rather than hard.

This is the re-solve check that human benchmarks run with a second annotator
and that ASearcher runs with a tool-less model: every candidate answer a strong
reader produces is tested against the constraints, and anything that survives
alongside the gold makes the item ambiguous. Here it is cheap, because the
chain bounds the candidate set to at most a few dozen papers instead of the
whole corpus.

Ambiguity is a property of a question, not of the model that happened to read
it, so a task rejected here is discarded rather than repaired.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from epago.taskgen.verbalize import _post

AMBIGUITY_CHECK_VERSION = "epago-chain-ambiguity-v1"

#: Laying out more candidates than this makes the prompt long and the judgement
#: worse, and a bridge that broad was already rejected at mint.
MAX_CANDIDATES = 40

#: Candidates are shown with an opening slice of their abstract, not by title
#: alone. A title is terse enough that a faithful description can fail to match
#: it: "No Association Between Autism Spectrum Disorder and Mitochondrial DNA
#: Variants" was judged not to match "investigates the association between
#: mitochondrial DNA variations and autism", which is exactly what the paper
#: did. That is a false rejection -- it costs yield rather than correctness,
#: but 1 in 6 of them is too many to accept for the sake of a shorter prompt.
SNIPPET_CHARS = 260

_SYSTEM = (
    "You match a description of a scientific paper against a list of candidate "
    "papers. You are strict: you pick a paper only when the description clearly "
    "fits it and does not fit the others. You return only JSON."
)

_PROMPT = """Here is a description of one scientific paper.

DESCRIPTION: {description}

Here are candidate papers, each with the opening of its abstract. Exactly one
of them is meant, or none of them is.

{candidates}

Which candidates does the DESCRIPTION fit? Judge only on whether the
description is a fair characterisation of what the paper investigated. A
description that states the paper's research question fits it even if the
paper's finding was negative, and even if it uses different wording from the
title. Do not try to guess which one was intended, and do not break a tie: if
two papers both fit, say both.

Return JSON:
"matches": a list of the candidate numbers the description fits. Empty if none.
"confident": true if you would defend the answer to a colleague.

Return only the JSON object."""


#: How many search rivals to show beside the bridge's set. These stand in for
#: what a solver sees if it ignores the chain and searches the description.
MAX_RIVALS = 8


@dataclass(frozen=True, slots=True)
class TwoHopReport:
    """Which candidates the description fit, split by whether they carry the bridge."""

    #: Matches among the papers carrying the bridge. Exactly one -- the gold --
    #: means the second hop is decidable.
    matched_in_bridge_set: tuple[int, ...]
    #: Matches among search rivals. At least one means the first hop is needed.
    matched_outside: tuple[int, ...]
    ok: bool
    failure: str | None


@dataclass(frozen=True, slots=True)
class AmbiguityReport:
    matched: tuple[int, ...]
    gold_index: int
    n_candidates: int
    unique: bool
    failure: str | None


def _snippet(snippets: dict[str, str] | None, doc_id: str) -> str:
    text = (snippets or {}).get(doc_id, "")
    return " ".join(text[:SNIPPET_CHARS].split())


def check_two_hop(
    description: str,
    gold_doc_id: str,
    bridge_doc_ids: list[str],
    rival_doc_ids: list[str],
    titles: dict[str, str],
    *,
    model: str,
    snippets: dict[str, str] | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> TwoHopReport:
    """Ask, in one pass, whether the second hop is both necessary and decidable.

    Two conditions define a working two-hop task, and they pull against each
    other, which is why they have to be checked together:

    *necessary* -- the description on its own must NOT pick out the gold. It is
    shown alongside ``rival_doc_ids``, the papers a solver would actually be
    handed if it simply searched the description, and at least one of those
    must fit the description too. Otherwise a solver reads the description,
    searches it, takes the top hit and is right without ever following the
    chain.

    *decidable* -- among ``bridge_doc_ids``, the papers that carry the bridge,
    exactly the gold must fit. Otherwise the solver does everything right and
    still cannot tell which paper was meant.

    Checking these by retrieval rank instead was the earlier approach and it
    does not work. A faithful description of a paper is an excellent query for
    that paper, so the gold lands at rank 1 about half the time no matter how
    the description is phrased; demanding otherwise rejected 52% of candidates
    and pushed the wording toward sentences like "examines respiratory health
    in various populations", which clear any rank rule precisely because they
    describe nothing. Rank was never the question. Whether a reader could
    *decide* is.

    One call covers both, because the two candidate pools are shown together
    and the verdict is read off which side the matches fall on.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return TwoHopReport((), (), False, "no_api_key")

    bridge_set = sorted(set(bridge_doc_ids) | {gold_doc_id})
    rivals = [d for d in sorted(set(rival_doc_ids)) if d not in set(bridge_set)]
    if len(bridge_set) > MAX_CANDIDATES:
        return TwoHopReport((), (), False, "too_many_candidates")
    ordered = bridge_set + rivals[:MAX_RIVALS]
    gold_index = ordered.index(gold_doc_id)

    listing = "\n\n".join(
        (
            f"{i}. {titles.get(doc_id, '')}\n   {_snippet(snippets, doc_id)}"
            if snippets
            else f"{i}. {titles.get(doc_id, '')}"
        )
        for i, doc_id in enumerate(ordered)
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _PROMPT.format(description=description, candidates=listing),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    try:
        data = _post(body, api_key, timeout)
        content = data["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}")
        payload = json.loads(content[start : end + 1])
    except Exception as exc:  # noqa: BLE001 - any failure is a non-verdict
        return TwoHopReport((), (), False, f"error:{type(exc).__name__}")

    raw = payload.get("matches")
    if not isinstance(raw, list):
        return TwoHopReport((), (), False, "malformed")
    matched = {m for m in raw if isinstance(m, int) and 0 <= m < len(ordered)}

    n_bridge = len(bridge_set)
    in_bridge = tuple(sorted(m for m in matched if m < n_bridge))
    outside = tuple(sorted(m for m in matched if m >= n_bridge))

    if gold_index not in matched:
        return TwoHopReport(in_bridge, outside, False, "gold_unrecognisable")
    if len(in_bridge) > 1:
        # More than one bridge-carrying paper fits: the solver follows the
        # chain correctly and still cannot choose.
        return TwoHopReport(in_bridge, outside, False, "undecidable_after_bridge")
    if not outside:
        # Nothing else fits, so the description alone identifies the gold and
        # the first hop is scenery.
        return TwoHopReport(in_bridge, outside, False, "decidable_without_bridge")
    return TwoHopReport(in_bridge, outside, True, None)


def check_gold_is_unique(
    description: str,
    gold_doc_id: str,
    candidate_doc_ids: list[str],
    titles: dict[str, str],
    *,
    model: str,
    snippets: dict[str, str] | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> AmbiguityReport:
    """Re-solve the second hop and report whether the answer is the only one.

    ``candidate_doc_ids`` is the bridge's document set: precisely the papers a
    solver holding the bridge would be choosing between.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return AmbiguityReport((), -1, 0, False, "no_api_key")

    ordered = sorted(set(candidate_doc_ids) | {gold_doc_id})
    if len(ordered) > MAX_CANDIDATES:
        return AmbiguityReport((), -1, len(ordered), False, "too_many_candidates")
    gold_index = ordered.index(gold_doc_id)

    listing = "\n\n".join(
        f"{i}. {titles.get(doc_id, '')}\n   {_snippet(snippets, doc_id)}"
        if snippets
        else f"{i}. {titles.get(doc_id, '')}"
        for i, doc_id in enumerate(ordered)
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _PROMPT.format(description=description, candidates=listing),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    try:
        data = _post(body, api_key, timeout)
        content = data["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}")
        payload = json.loads(content[start : end + 1])
    except Exception as exc:  # noqa: BLE001 - any failure is a non-verdict
        return AmbiguityReport((), gold_index, len(ordered), False, f"error:{type(exc).__name__}")

    raw = payload.get("matches")
    if not isinstance(raw, list):
        return AmbiguityReport((), gold_index, len(ordered), False, "malformed")
    matched = tuple(
        sorted({m for m in raw if isinstance(m, int) and 0 <= m < len(ordered)})
    )

    if matched == (gold_index,):
        return AmbiguityReport(matched, gold_index, len(ordered), True, None)
    if not matched:
        return AmbiguityReport(matched, gold_index, len(ordered), False, "gold_unrecognisable")
    if gold_index in matched:
        return AmbiguityReport(matched, gold_index, len(ordered), False, "multiple_answers")
    return AmbiguityReport(matched, gold_index, len(ordered), False, "describes_another_paper")
