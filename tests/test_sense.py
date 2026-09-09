"""The uniqueness proof is over strings; this is the check on what they mean."""

import pytest

from epago.taskgen.sense import (
    SHORT_ACRONYM_MAX,
    Sense,
    expansions,
    is_acronym,
    minter_rejects,
    sense_of,
)


# --- finding the definition beside an acronym ------------------------------------


@pytest.mark.parametrize(
    "text, acr, expected",
    [
        ("incorporates the Efficient Channel Attention (ECA) mechanism", "eca", "efficient channel attention"),
        # function words inside the name are allowed and kept
        ("the Supplemental Nutrition Assistance Program (SNAP) Online", "snap", "supplemental nutrition assistance program"),
        # a single hyphenated word spells a three-letter acronym
        ("the Fenna-Matthews-Olson (FMO) complex", "fmo", "fenna-matthews-olson"),
        # letters taken from inside the words, first letter opening the first word
        ("gut-derived trimethylamine N-oxide (TMAO) levels", "tmao", "trimethylamine n-oxide"),
        ("chronic thromboembolic pulmonary hypertension (CTEPH) is", "cteph", "chronic thromboembolic pulmonary hypertension"),
        ("wide-bandgap silicon carbide (SiC) devices", "sic", "silicon carbide"),
        # plural form and case are how papers actually write them
        ("optimized antireflection coatings (ARCs) reduce", "arc", "antireflection coatings"),
    ],
)
def test_a_definition_is_read_from_the_words_before_the_parenthesis(text, acr, expected):
    assert expected in expansions(text, acr)


@pytest.mark.parametrize(
    "text, acr",
    [
        # words precede the parenthesis but do not spell the acronym
        ("the effect of its concentration (SiC) on yield", "sic"),
        ("modules were measured at 25 °C (STC)", "stc"),
        # the acronym appears without ever being defined
        ("a validated 99mTc-MAA preparation was applied", "maa"),
    ],
)
def test_a_parenthetical_that_is_not_a_definition_is_ignored(text, acr):
    assert expansions(text, acr) == set()


# --- the three verdicts ----------------------------------------------------------


def test_the_same_name_in_both_papers_is_the_same_thing():
    anchor = "Association between gut microbiota-derived trimethylamine N-oxide (TMAO) and outcomes."
    gold = "Serum trimethylamine-N-oxide (TMAO) fell with treatment."
    assert sense_of(anchor, gold, "tmao") is Sense.SAME


def test_two_different_names_sharing_letters_are_a_collision():
    """The measured case: photovoltaics on one side, gastroenterology on the other."""
    anchor = "Modules were rated under Standard Test Conditions (STC)."
    gold = "CC was classified as normal transit constipation (NTC) or slow-transit constipation (STC)."
    assert sense_of(anchor, gold, "stc") is Sense.COLLISION


@pytest.mark.parametrize(
    "anchor, gold, acr",
    [
        ("Lung cancer screening (LCS) uptake", "a low-carbon scenario (LCS) was modelled", "lcs"),
        ("the Magnetospheric Multiscale (MMS) mission", "Mohs micrographic surgery (MMS) outcomes", "mms"),
        ("the Fenna-Matthews-Olson (FMO) complex", "fluorescence minus one (FMO) controls", "fmo"),
    ],
)
def test_measured_collisions_are_caught(anchor, gold, acr):
    assert sense_of(anchor, gold, acr) is Sense.COLLISION


def test_an_acronym_one_paper_never_spells_out_cannot_be_judged():
    """MAA: a drug metabolite in the anchor, an imaging tracer in the gold -- a
    real collision the check cannot see, because the gold never expands it."""
    anchor = "metabolites 4-methylaminoantipyrine (MAA) and 4-aminoantipyrine (AA)"
    gold = "A single-patient preparation of 99mTc-MAA enabled daily imaging."
    assert sense_of(anchor, gold, "maa") is Sense.UNVERIFIED


def test_a_multi_word_or_long_term_is_a_name_and_needs_no_check():
    assert sense_of("x", "y", "global horizontal irradiance") is Sense.NOT_ACRONYM
    assert sense_of("x", "y", "mann-kendall") is Sense.NOT_ACRONYM


@pytest.mark.parametrize("term, acro", [("stc", True), ("cteph", True), ("ghi", True), ("sc-2", True),
                                        ("mann-kendall", False), ("global horizontal irradiance", False)])
def test_what_counts_as_an_acronym(term, acro):
    assert is_acronym(term) is acro


# --- the minter's rule: proof to admit -------------------------------------------


def test_a_collision_is_refused_at_any_length():
    assert minter_rejects(Sense.COLLISION, "cteph") == "bridge_sense_collision"
    assert minter_rejects(Sense.COLLISION, "stc") == "bridge_sense_collision"


def test_a_short_unverified_acronym_is_refused_but_a_longer_one_is_admitted():
    """Three-letter acronyms were 86% of measured collisions; the unverified
    bucket as a whole scored like clean tasks. The rule is drawn at that line."""
    assert minter_rejects(Sense.UNVERIFIED, "maa") == "bridge_sense_unverified_short"
    assert len("maa") <= SHORT_ACRONYM_MAX
    assert minter_rejects(Sense.UNVERIFIED, "tmao") is None
    assert minter_rejects(Sense.UNVERIFIED, "cteph") is None


def test_same_and_names_are_always_admitted():
    assert minter_rejects(Sense.SAME, "stc") is None
    assert minter_rejects(Sense.NOT_ACRONYM, "global horizontal irradiance") is None
