"""Turn a proved chain skeleton into a question a researcher would recognise.

The structure is proved before this module runs and is not negotiable here:
which documents, which bridge, and the fact that the chain resolves to exactly
one paper are all settled by :mod:`epago.taskgen.chain`. What is missing is
everything a language model is actually good at, and a regex is not:

*naming the bridge's kind.*
    The set algebra proves the bridge identifies the gold. It cannot make the
    question *well posed*. "A second study shares a specific named item with
    the first" leaves the solver to guess which of an abstract's dozens of
    proper nouns was meant. Saying "the same sequencing platform" or "the same
    screening questionnaire" turns a guessing game into a research task, and is
    the difference between a task with a unique answer and a task with a
    findable one.

*rejecting coincidences.*
    A shared string is not a shared subject. "time-lapse" occurs in glacier
    photography and in electrical resistivity tomography with no relation
    between them; "Key Informant Interviews" links any two papers that ran
    interviews. Topical overlap filters some of this; only a reader catches the
    rest.

*breaking the lexical trail.*
    Descriptions built from a paper's own words are found by BM25 whether or
    not the solver follows the chain -- measured at 81% of structurally valid
    candidates. A paraphrase keeps the meaning and loses the term overlap,
    which is what makes the first hop load-bearing rather than decorative.

The model writes *parts*, never the question. The question is assembled from a
fixed frame here, so the logical form is identical across every task and is
auditable without reading a model's prose. Everything the model returns is then
re-checked mechanically, and the assembled question is put back through the
real search backend: nothing this module produces is trusted on the strength of
having come from a language model.

Generation is offline and one-way. It happens before any miner sees a task,
consumes no miner-controlled input, and its output is graded by exact match
against a title the corpus already contains -- so there is no path by which a
competitor can influence the model that writes the exam.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from epago.taskgen.chain import ChainSkeleton

VERBALIZER_VERSION = "epago-chain-verbalizer-v1"

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

#: Descriptions longer than this stop being descriptions and start being
#: summaries that carry the answer.
MAX_DESCRIPTION_CHARS = 220
MAX_TYPE_CHARS = 60

#: Guard against a description that is the title with the punctuation removed.
#: It is deliberately loose. A faithful one-sentence description of a paper
#: called "Graph Neural Network-Based Optimization for Renewable Energy Layout"
#: has to say graph neural network, optimization and renewable energy -- that
#: is the description doing its job, not a leak. Whether the description
#: actually finds the paper is not a question about word overlap at all, and it
#: is answered directly by running the assembled question through the real
#: search backend in :func:`epago.taskgen.chain.verify_route`. A tight
#: threshold here just rejects good tasks for failing a proxy while the real
#: measurement stands next to it unused.
MAX_TITLE_OVERLAP = 0.85

#: Untrusted-text hygiene. Model output is pasted into a prompt another model
#: reads, so anything that could read as markup or as an instruction boundary
#: is refused rather than escaped. Ordinary prose punctuation is allowed; the
#: class below is deliberately narrow so that rejecting is rare and meaningful.
_FORBIDDEN = re.compile(r"""[<>{}\[\]\\|`"\x00-\x1f]""")

_TOKEN = re.compile(r"[a-z0-9]+")

#: The model is asked for a clause that continues "One study in this corpus
#: ...", and it reliably supplies a whole sentence instead about a fifth of the
#: time. Stripping the opener is better than rejecting the task: the content is
#: right, only the grammar of the join is wrong, and re-rolling the call would
#: cost a request to fix a leading article.
_FRAME_ECHO = re.compile(
    r"^(?:one|another|a|the|this)\s+(?:study|paper|article|work|research)"
    r"(?:\s+in\s+this\s+corpus)?\s*(?:,)?\s*",
    re.IGNORECASE,
)


def _strip_frame_echo(text: str) -> str:
    """Drop a leading 'One study in this corpus' the frame already supplies."""
    stripped = _FRAME_ECHO.sub("", text, count=1).lstrip()
    return stripped or text

_SYSTEM = (
    "You write questions for a benchmark that tests whether an AI research "
    "agent can follow evidence between scientific papers. You are precise, you "
    "never invent facts, and you return only JSON."
)

_PROMPT = """You are given two papers from a fixed corpus and one term that appears in both.

TERM: {bridge}

PAPER ONE
title: {anchor_title}
abstract: {anchor_text}

PAPER TWO
title: {gold_title}
abstract: {gold_text}

NEIGHBOURS OF PAPER TWO
These are other papers in the same corpus on the same subject. They do NOT
involve the TERM. Paper two must not be distinguishable from them by its
description alone.
{crowd}

A benchmark question will be built like this: the solver is told what PAPER ONE
is about, must find and read it, must notice the TERM in it, and must then use
the TERM together with a description of PAPER TWO to identify PAPER TWO. The
question never states the TERM.

Decide first whether that is a fair question, then write the parts.

Return JSON with exactly these keys:

"link_is_real": true only if BOTH of these hold.
  (a) The TERM names a specific thing -- a named method, instrument, platform,
      material, gene, organism, place, cohort, scale, model or dataset -- and
      not a property, adjective or category. "Right-sided" and "single-atom"
      describe things; they do not name one.
  (b) It refers to the SAME specific thing in both papers, and that shared
      thing is substantive enough that a researcher would call it a real
      connection between the two studies.
  Set false for terms two unrelated papers could share by routine: statistical
  software, a statistics test, a study design, an interview or survey format, a
  funding body, a journal, a reporting checklist, a generic outcome measure.
  When in doubt, set false. A rejected pair costs nothing; a bad pair produces
  a question with no findable answer.

"bridge_type": a short, general noun phrase naming what kind of thing the TERM
  is, as a solver would need to hear it: sequencing platform, screening
  questionnaire, gene, climate scenario set, catalyst class. It must NOT
  contain the TERM or any word from it, and must not be so specific that it
  identifies the TERM by itself.

"anchor_description": one sentence, under 200 characters, describing what PAPER
  ONE studied. Paraphrase: use ordinary scientific wording, and avoid reusing
  distinctive words from its title where a plain synonym exists. Do NOT mention
  the TERM. Do not name the paper. Write it so it continues the sentence 'One
  study in this corpus ...', for example 'examines how drought affects wheat
  yield in semi-arid regions'.

"gold_description": one sentence, under 200 characters, saying what PAPER TWO
  investigated. It must be specific enough that a reader who had the paper in
  front of them would recognise it, and general enough that it does not
  single paper two out from the NEIGHBOURS listed above at a glance.
  The way to do both: state the research question accurately, but use the
  general category rather than the one distinctive name only paper two used --
  say the class of material rather than the exact compound, the type of cohort
  rather than the city, the family of model rather than its name. Keep the
  substance; drop the fingerprint.
  Do NOT write a vague filler sentence. "examines respiratory health in various
  populations" is useless and will be rejected: it describes nothing and gives
  the solver no way to choose. Aim for the level of "compares two ventilation
  strategies in preterm infants".
  Do NOT mention the TERM, and do not reveal paper two's title.

"reject_reason": null if usable, otherwise a short string saying why not.

Do not use quotation marks, brackets or backslashes inside any string value.
Return only the JSON object."""


@dataclass(frozen=True, slots=True)
class Verbalization:
    bridge_type: str
    anchor_description: str
    gold_description: str
    model: str
    ok: bool
    reject_reason: str | None


def assemble_question(v: Verbalization) -> str:
    """Build the question from a fixed frame.

    The frame is deliberately identical for every task in the release: the
    solver learns the format once, and any difficulty difference between two
    tasks comes from the papers rather than from how the question was worded.
    """
    return (
        f"One study in this corpus {v.anchor_description.rstrip('.')}. "
        f"That study names a specific {v.bridge_type}. "
        f"A different study in this corpus {v.gold_description.rstrip('.')}, "
        f"and it involves that same {v.bridge_type}. "
        f"Give the exact title of that second study."
    )


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _clean(value, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > limit:
        return None
    if _FORBIDDEN.search(text):
        return None
    return text


def check_verbalization(
    skeleton: ChainSkeleton,
    payload: dict,
    titles: dict[str, str],
    *,
    model: str,
) -> Verbalization:
    """Re-derive every property the prompt asked for, mechanically.

    A model asked not to mention the bridge will sometimes mention the bridge.
    Each rule below is checked rather than trusted, and a violation rejects the
    task instead of repairing it -- a repaired question is one nobody proved
    anything about.
    """

    def bad(reason: str) -> Verbalization:
        return Verbalization("", "", "", model, False, reason)

    if not isinstance(payload, dict):
        return bad("not_json")
    if payload.get("link_is_real") is not True:
        return bad("llm_rejected")

    btype = _clean(payload.get("bridge_type"), MAX_TYPE_CHARS)
    adesc = _clean(payload.get("anchor_description"), MAX_DESCRIPTION_CHARS)
    gdesc = _clean(payload.get("gold_description"), MAX_DESCRIPTION_CHARS)
    if not btype or not adesc or not gdesc:
        return bad("malformed_fields")
    adesc = _strip_frame_echo(adesc)
    gdesc = _strip_frame_echo(gdesc)

    bridge_tokens = _tokens(skeleton.bridge)
    written = _tokens(f"{btype} {adesc} {gdesc}")
    if bridge_tokens & written:
        return bad("bridge_leaked")

    # The answer must not be recoverable by reading the question.
    gold_title_tokens = _tokens(titles.get(skeleton.gold_doc_id, ""))
    if gold_title_tokens:
        overlap = len(gold_title_tokens & _tokens(gdesc)) / len(gold_title_tokens)
        if overlap > MAX_TITLE_OVERLAP:
            return bad("gold_title_echoed")
    anchor_title_tokens = _tokens(titles.get(skeleton.anchor_doc_id, ""))
    if anchor_title_tokens:
        overlap = len(anchor_title_tokens & _tokens(adesc)) / len(anchor_title_tokens)
        if overlap > MAX_TITLE_OVERLAP:
            return bad("anchor_title_echoed")

    return Verbalization(btype, adesc, gdesc, model, True, None)


def _post(body: dict, api_key: str, timeout: float) -> dict:
    request = urllib.request.Request(
        _ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def verbalize(
    skeleton: ChainSkeleton,
    docs: dict[str, tuple[str, str]],
    *,
    model: str,
    api_key: str | None = None,
    max_text_chars: int = 2400,
    timeout: float = 90.0,
    retries: int = 3,
) -> Verbalization:
    """One model call for one skeleton, with the mechanical re-checks applied.

    Network faults retry with a backoff; a refusal or a malformed answer does
    not. The distinction matters: retrying a transport error costs a second,
    while retrying a judgement re-rolls the dice on a task the model already
    declined, which is how a rejection filter quietly stops filtering.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return Verbalization("", "", "", model, False, "no_api_key")

    anchor_title, anchor_text = docs[skeleton.anchor_doc_id]
    gold_title, gold_text = docs[skeleton.gold_doc_id]
    crowd = "\n".join(
        f"- {docs[doc_id][0]}"
        for doc_id in skeleton.gold_crowd_sample
        if doc_id in docs
    ) or "- (none found)"
    prompt = _PROMPT.format(
        bridge=skeleton.bridge,
        anchor_title=anchor_title,
        anchor_text=anchor_text[:max_text_chars],
        gold_title=gold_title,
        gold_text=gold_text[:max_text_chars],
        crowd=crowd,
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }

    last = "unknown"
    for attempt in range(retries):
        try:
            data = _post(body, api_key, timeout)
            content = data["choices"][0]["message"]["content"]
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                return Verbalization("", "", "", model, False, "no_json")
            payload = json.loads(content[start : end + 1])
            titles = {k: v[0] for k, v in docs.items()}
            return check_verbalization(skeleton, payload, titles, model=model)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = f"transport:{type(exc).__name__}"
            time.sleep(2**attempt)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            return Verbalization(
                "", "", "", model, False, f"bad_response:{type(exc).__name__}"
            )
    return Verbalization("", "", "", model, False, last)


# --- the intersection shape --------------------------------------------------
#
# This wording step is strictly easier than the one above, and the reason is
# structural rather than a matter of prompting. There, the model had to write a
# description of the answer that a reader could decide on but a search engine
# could not resolve, and those two demands cannot both be met by one sentence:
# measured across three mints, 52-58% of candidates failed on it whatever the
# phrasing, and pushing toward generality produced sentences that identified
# nothing. Here the answer is never described. The model characterises two
# *other* papers, and there is no leak to trade against, so the descriptions
# can simply be accurate.
#
# What remains is the judgement a regex cannot make, and it turned out to be
# two judgements rather than one. The first is what kind of thing each term
# is, without which the question reads "it names a specific item" and asks the
# solver to guess which of an abstract's proper nouns was meant. The second is
# whether the intersection means anything at all: a set intersection over
# postings is blind to word sense, and abbreviations collide constantly across
# fields. A first mint produced a question whose two anchors established that
# FOV was a camera setting and SEP an annotation method, and whose answer was a
# paper about Solar Energetic Particles. The set algebra was correct and the
# task was nonsense. Only a reader holding all three papers can see that, so
# the answer paper is shown here even though it is never described.

_INTERSECTION_PROMPT = """You are given two papers from a fixed corpus, and one term from each.

TERM ONE: {bridge_x}
PAPER ONE
title: {anchor_a_title}
abstract: {anchor_a_text}

TERM TWO: {bridge_y}
PAPER TWO
title: {anchor_b_title}
abstract: {anchor_b_text}

THE ANSWER PAPER
This is the third paper. It is the only paper in the corpus containing both
terms, and it is what the question will ask for.
title: {gold_title}
abstract: {gold_text}

A benchmark question will be built like this: the solver is told what PAPER ONE
is about, finds and reads it, and notices TERM ONE. It does the same for PAPER
TWO and TERM TWO. It then has to find the single further paper in the corpus
that involves both terms. The question never states either term.

Decide whether that is a fair question, then write the parts.

Return JSON with exactly these keys:

"link_is_real": true only if ALL THREE of these hold.
  (a) Both terms name a specific thing -- a named method, instrument,
      platform, material, gene, organism, place, cohort, scale, dataset,
      model or standard -- rather than a property, an adjective, or a
      fragment of a sentence.
  (b) Neither term is generic research vocabulary that two unrelated papers
      could share by routine: a statistics test, a study design, an interview
      format, a reporting checklist.
  (c) MOST IMPORTANT. TERM ONE means the same thing in THE ANSWER PAPER as it
      does in PAPER ONE, and TERM TWO means the same thing in THE ANSWER PAPER
      as it does in PAPER TWO. Judge this from PAPER ONE and PAPER TWO
      outward. If you find yourself describing a term the way the ANSWER paper
      uses it, the senses differ and the answer is false. Abbreviations collide constantly across fields:
      SEP is an annotation scheme in one paper and Solar Energetic Particles in
      another; RIS is a radiology information system in one and a
      Reconfigurable Intelligent Surface in another; a place name can be a
      river in one paper and a Brazilian state in another. If either term
      refers to a different thing in the answer paper, set false. A solver
      told to look for "the annotation method" would never accept a solar
      physics paper, and the question would have no findable answer.
  When in doubt, set false. A rejected triple costs nothing.

"bridge_x_type": a short, general noun phrase naming what kind of thing TERM
  ONE is **as PAPER ONE uses it** -- not as the answer paper uses it. This is
  the most common way to get this wrong. A reader will open PAPER ONE and look
  for the thing you name here, so it must be findable there. If TERM ONE is a
  crystal structure in a semiconductor paper, say crystal structure, even if
  the answer paper uses the same letters for a dietary lipid level. If it is a
  nursing degree in a nursing paper, say nursing degree, even if the answer
  paper means a nanoparticle material. Examples of good types: sequencing
  platform, screening questionnaire, gene, study region, catalyst class. It
  must NOT contain TERM ONE or any word from it.

"bridge_y_type": the same for TERM TWO, as PAPER TWO uses it. It must NOT
  contain TERM TWO or any word from it. If the two types would read identically, make them distinct
  enough that a solver can tell which paper to look in for which.

"anchor_a_description": one sentence, under 200 characters, saying what PAPER
  ONE investigated. Be accurate and specific -- this paper is not the answer,
  so there is nothing to hide. Write it to continue the sentence 'One study in
  this corpus ...', for example 'examines how drought affects wheat yield in
  semi-arid regions'. Do NOT mention TERM ONE. Do not name the paper.

"anchor_b_description": the same for PAPER TWO, and do NOT mention TERM TWO.

"reject_reason": null if usable, otherwise a short string saying why not.

Do not use quotation marks, brackets or backslashes inside any string value.
Return only the JSON object."""


@dataclass(frozen=True, slots=True)
class IntersectionVerbalization:
    bridge_x_type: str
    bridge_y_type: str
    anchor_a_description: str
    anchor_b_description: str
    model: str
    ok: bool
    reject_reason: str | None


def assemble_intersection_question(
    v: IntersectionVerbalization,
    *,
    tier: str = "described_both",
    anchor_a_title: str = "",
    anchor_b_title: str = "",
) -> str:
    """Build the question from a fixed frame, at the requested length.

    Naming an anchor replaces "find the study that ..." with "the study titled
    ...", which removes a find-and-identify step. The answer is untouched by
    this: an anchor is never the answer, and the study being asked for is still
    described nowhere. Only the route to the two keys gets shorter.

    The closing sentence always states that exactly one study qualifies,
    because a solver that does not know the answer is unique cannot tell when
    it is finished.
    """
    # A named anchor is a noun phrase and takes the verb directly; a described
    # one is a whole clause and needs its own sentence, or the description runs
    # straight into "names a specific ..." as one ungrammatical breath.
    def named(title: str, kind: str) -> str:
        return f'The study titled "{title}" names a specific {kind}.'

    def described(lead: str, desc: str, kind: str) -> str:
        return f"{lead} {desc.rstrip('.')}. That study names a specific {kind}."

    if tier == "named_both":
        first = named(anchor_a_title, v.bridge_x_type)
        second = named(anchor_b_title, v.bridge_y_type)
    elif tier == "named_one":
        # Which anchor is named is decided by which one is SAFE to name, not by
        # position. Naming A unconditionally leaked a hidden term into 25 of
        # 400 questions: a paper called "GRID CONNECTED HYBRID RENEWABLE ENERGY
        # SYSTEM" prints the term HYBRID that the reader was supposed to go and
        # find. The tier was offered whenever EITHER anchor was nameable, so
        # `named_one` was reached on B's eligibility and then printed A anyway.
        if anchor_a_title:
            first = named(anchor_a_title, v.bridge_x_type)
            second = described("A study in this corpus", v.anchor_b_description, v.bridge_y_type)
        else:
            first = described("One study in this corpus", v.anchor_a_description, v.bridge_x_type)
            second = named(anchor_b_title, v.bridge_y_type)
    else:
        first = described("One study in this corpus", v.anchor_a_description, v.bridge_x_type)
        second = described("A second study in this corpus", v.anchor_b_description, v.bridge_y_type)

    return (
        f"{first} {second} "
        f"Exactly one other study in this corpus involves both the "
        f"{v.bridge_x_type} named in the first and the {v.bridge_y_type} named "
        f"in the second. Give the exact title of that study."
    )


_TYPE_STOP = frozenset(
    {
        "specific", "type", "kind", "method", "system", "index", "level",
        "component", "organization", "technique", "material", "group",
        "score", "tool", "model", "unit", "class", "scale", "measure",
        "instrument", "platform", "device", "process", "approach", "based",
    }
)


def _type_supported_by(text: str, type_phrase: str) -> bool:
    """Does the paper the reader will open actually talk about this kind of thing?

    The check that was missing, and its absence broke 65% of a 400-task batch.
    The wording step sees all three papers, and when an abbreviation means
    different things in different fields it would label the term with the
    *answer* paper's sense: a semiconductor paper was said to name "a specific
    dietary lipid level" (L12 is a crystal structure there) and a nursing paper
    "a specific nanoparticle component" (BSN is a nursing degree there). A
    reader following those instructions is looking for something that is not in
    the document.

    The oracle test cannot catch this -- it is handed all three papers and
    works backwards, so it passed the broken tasks at 98.1%, higher than the
    sound ones. Only a check tied to the anchor alone sees it.

    Deliberately lenient: one content word in common is enough, because a fair
    type is a paraphrase rather than a quotation.
    """
    content = {
        w for w in re.findall(r"[a-z]{4,}", type_phrase.lower()) if w not in _TYPE_STOP
    }
    if not content:
        return False
    haystack = text.lower()
    return any(w in haystack for w in content)


def check_intersection(
    skeleton,
    payload: dict,
    titles: dict[str, str],
    *,
    model: str,
    anchor_a_text: str = "",
    anchor_b_text: str = "",
) -> IntersectionVerbalization:
    """Re-derive every property the prompt asked for, mechanically."""

    def bad(reason: str) -> IntersectionVerbalization:
        return IntersectionVerbalization("", "", "", "", model, False, reason)

    if not isinstance(payload, dict):
        return bad("not_json")
    if payload.get("link_is_real") is not True:
        return bad("llm_rejected")

    x_type = _clean(payload.get("bridge_x_type"), MAX_TYPE_CHARS)
    y_type = _clean(payload.get("bridge_y_type"), MAX_TYPE_CHARS)
    a_desc = _clean(payload.get("anchor_a_description"), MAX_DESCRIPTION_CHARS)
    b_desc = _clean(payload.get("anchor_b_description"), MAX_DESCRIPTION_CHARS)
    if not x_type or not y_type or not a_desc or not b_desc:
        return bad("malformed_fields")
    a_desc = _strip_frame_echo(a_desc)
    b_desc = _strip_frame_echo(b_desc)

    written = _tokens(f"{x_type} {y_type} {a_desc} {b_desc}")
    if _tokens(skeleton.bridge_x) & written or _tokens(skeleton.bridge_y) & written:
        return bad("bridge_leaked")

    # The two types must be distinguishable, or the question cannot say which
    # paper to look in for which term.
    if normalize_type(x_type) == normalize_type(y_type):
        return bad("types_indistinguishable")


    # Each type must describe the paper the reader is sent to, not the answer.
    # Without this, an abbreviation that means different things in different
    # fields gets labelled with the answer's sense, and the reader is hunting
    # something the document does not contain.
    if anchor_a_text and not _type_supported_by(anchor_a_text, x_type):
        return bad("type_x_not_in_anchor")
    if anchor_b_text and not _type_supported_by(anchor_b_text, y_type):
        return bad("type_y_not_in_anchor")

    # The answer must not be assembled out of the question's own words.
    gold_tokens = _tokens(titles.get(skeleton.gold_doc_id, ""))
    if gold_tokens and gold_tokens <= written:
        return bad("answer_exposed")

    return IntersectionVerbalization(x_type, y_type, a_desc, b_desc, model, True, None)


def normalize_type(text: str) -> str:
    return " ".join(sorted(_TOKEN.findall(text.lower())))


def verbalize_intersection(
    skeleton,
    docs: dict[str, tuple[str, str]],
    *,
    model: str,
    api_key: str | None = None,
    max_text_chars: int = 2400,
    timeout: float = 90.0,
    retries: int = 3,
) -> IntersectionVerbalization:
    """One model call for one intersection skeleton, with the re-checks applied."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return IntersectionVerbalization("", "", "", "", model, False, "no_api_key")

    a_title, a_text = docs[skeleton.anchor_a_doc_id]
    b_title, b_text = docs[skeleton.anchor_b_doc_id]
    gold_title, gold_text = docs[skeleton.gold_doc_id]
    prompt = _INTERSECTION_PROMPT.format(
        bridge_x=skeleton.bridge_x,
        bridge_y=skeleton.bridge_y,
        anchor_a_title=a_title,
        anchor_a_text=a_text[:max_text_chars],
        anchor_b_title=b_title,
        anchor_b_text=b_text[:max_text_chars],
        gold_title=gold_title,
        gold_text=gold_text[:max_text_chars],
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }

    last = "unknown"
    for attempt in range(retries):
        try:
            data = _post(body, api_key, timeout)
            content = data["choices"][0]["message"]["content"]
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                return IntersectionVerbalization("", "", "", "", model, False, "no_json")
            payload = json.loads(content[start : end + 1])
            titles = {k: v[0] for k, v in docs.items()}
            return check_intersection(
                skeleton,
                payload,
                titles,
                model=model,
                anchor_a_text=f"{a_title} {a_text}",
                anchor_b_text=f"{b_title} {b_text}",
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = f"transport:{type(exc).__name__}"
            time.sleep(2**attempt)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            return IntersectionVerbalization(
                "", "", "", "", model, False, f"bad_response:{type(exc).__name__}"
            )
    return IntersectionVerbalization("", "", "", "", model, False, last)
