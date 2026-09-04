"""Tests for epago.core.reveal — the e2/ev2/ek1 on-chain wire formats."""

from __future__ import annotations

import pytest

from epago.core.reveal import (
    WireFormatError,
    build_king_pointer,
    build_reveal,
    build_verdict,
    parse_king_pointer,
    parse_reveal,
    parse_verdict,
)
from epago.core.types import KingPointer, ModelRef, Verdict, VerdictDecision

KING_DIGEST = "sha256:" + "a" * 64
CHALL_DIGEST = "hf:" + "b" * 40
HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
OTHER_HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
AUDIT16 = "0123456789abcdef"


def make_verdict(**overrides) -> Verdict:
    kwargs = dict(
        challenger_digest=CHALL_DIGEST,
        decision=VerdictDecision.ACCEPT,
        lcb_pub_e6=31_500,
        mu_priv_e6=-2_000,
        delta_e6=25_000,
        round=3,
        private_pool_epoch=7,
        audit_digest=AUDIT16,
    )
    kwargs.update(overrides)
    return Verdict(**kwargs)


def make_pointer(**overrides) -> KingPointer:
    kwargs = dict(
        repo="team/EPAGO-DR-4B-run1",
        digest=CHALL_DIGEST,
        author_hotkey=HOTKEY,
        crowned_block=9_000,
        reign_started_block=8_000,
        coronation_lcb_e6=31_500,
        coronation_delta_e6=25_000,
    )
    kwargs.update(overrides)
    return KingPointer(**kwargs)


class TestRevealRoundtrip:
    def test_roundtrip(self):
        challenger = ModelRef(repo="team/EPAGO-DR-4B-run1", digest=CHALL_DIGEST)
        raw = build_reveal(KING_DIGEST, challenger)
        assert raw == f"e2|{KING_DIGEST}|team/EPAGO-DR-4B-run1|{CHALL_DIGEST}"
        parsed = parse_reveal(
            raw, reveal_block=1000, block_hash_at_reveal="0xdeadbeef", author_hotkey=HOTKEY
        )
        assert parsed.king_digest == KING_DIGEST
        assert parsed.challenger == challenger
        assert parsed.author_hotkey == HOTKEY
        assert parsed.reveal_block == 1000
        assert parsed.block_hash_at_reveal == "0xdeadbeef"

    def test_author_is_never_read_from_the_payload(self):
        """The whole point of e2: authorship comes from the chain, not the wire.

        Under e1 the author was a payload field nothing verified, so a hotkey
        could submit a deliberately losing checkpoint under a rival's identity
        and collect the rival an escalating intake cooldown.
        """
        raw = build_reveal(KING_DIGEST, ModelRef(repo="r/x", digest=CHALL_DIGEST))
        assert HOTKEY not in raw and OTHER_HOTKEY not in raw
        parsed = parse_reveal(raw, 1, "0x00", author_hotkey=OTHER_HOTKEY)
        assert parsed.author_hotkey == OTHER_HOTKEY

    def test_build_rejects_bad_king_digest(self):
        challenger = ModelRef(repo="r/x", digest=CHALL_DIGEST)
        with pytest.raises(WireFormatError):
            build_reveal("sha256:tooshort", challenger)

    def test_build_rejects_repo_with_separator(self):
        challenger = ModelRef(repo="r|x", digest=CHALL_DIGEST)
        with pytest.raises(WireFormatError):
            build_reveal(KING_DIGEST, challenger)

    @pytest.mark.parametrize(
        "raw",
        [
            "e3|x|y|z",                                                   # unknown version
            f"e1|{KING_DIGEST}|repo|{CHALL_DIGEST}|{HOTKEY}",             # retired format
            f"e2|{KING_DIGEST}|repo",                                     # too few fields
            f"e2|{KING_DIGEST}|repo|{CHALL_DIGEST}|extra",                # too many fields
            "",                                                            # empty
            "not a reveal at all",
        ],
    )
    def test_parse_rejects_malformed(self, raw):
        with pytest.raises(WireFormatError):
            parse_reveal(raw, 1, "0x00", author_hotkey=HOTKEY)

    def test_parse_rejects_bad_king_digest(self):
        raw = f"e2|sha256:nothex|repo|{CHALL_DIGEST}"
        with pytest.raises(WireFormatError):
            parse_reveal(raw, 1, "0x00", author_hotkey=HOTKEY)

    def test_parse_rejects_bad_challenger_digest_as_wire_error(self):
        """A bad digest must surface as WireFormatError, not a bare ValueError.

        ``read_revealed_submissions`` only catches WireFormatError, so a
        ModelRef ValueError escaping here would take out the whole intake scan.
        """
        raw = f"e2|{KING_DIGEST}|repo|hf:short"
        with pytest.raises(WireFormatError):
            parse_reveal(raw, 1, "0x00", author_hotkey=HOTKEY)


class TestVerdictRoundtrip:
    def test_roundtrip(self):
        v = make_verdict()
        raw = build_verdict(v)
        assert raw == f"ev3|{CHALL_DIGEST}|A|31500|-2000|25000|3|7|{AUDIT16}"
        parsed = parse_verdict(raw, validator_hotkey="val-1", block=555)
        assert parsed.challenger_digest == CHALL_DIGEST
        assert parsed.decision is VerdictDecision.ACCEPT
        assert parsed.lcb_pub_e6 == 31_500
        assert parsed.mu_priv_e6 == -2_000
        assert parsed.delta_e6 == 25_000
        assert parsed.round == 3
        assert parsed.private_pool_epoch == 7
        assert parsed.audit_digest == AUDIT16
        assert parsed.validator_hotkey == "val-1"
        assert parsed.block == 555

    def test_reject_decision_roundtrip(self):
        raw = build_verdict(make_verdict(decision=VerdictDecision.REJECT))
        assert parse_verdict(raw, "v", 1).decision is VerdictDecision.REJECT

    def test_near_miss_is_derivable_from_the_wire(self):
        """Arena credit must be computable by a validator that ran no duel."""

        def near_miss(**kw):
            return parse_verdict(build_verdict(make_verdict(**kw)), "v", 1).is_near_miss

        # Probably better, not provably: 0 < lcb <= delta with a positive
        # private half.
        assert near_miss(
            decision=VerdictDecision.REJECT, lcb_pub_e6=10_000, mu_priv_e6=500, delta_e6=25_000
        )
        # Exactly at the floor still counts.
        assert near_miss(
            decision=VerdictDecision.REJECT, lcb_pub_e6=25_000, mu_priv_e6=500, delta_e6=25_000
        )
        # A round runner-up: it beat the king outright but someone beat it.
        # Rejected, yet it must not be treated as a loss.
        assert near_miss(
            decision=VerdictDecision.REJECT, lcb_pub_e6=90_000, mu_priv_e6=500, delta_e6=25_000
        )
        # A plain loss.
        assert not near_miss(
            decision=VerdictDecision.REJECT, lcb_pub_e6=-1, mu_priv_e6=500, delta_e6=25_000
        )
        # Overfit: cleared the public floor, lost the private half. Not a
        # near-miss — this is the generator-overfit attack and it pays.
        assert not near_miss(
            decision=VerdictDecision.REJECT, lcb_pub_e6=90_000, mu_priv_e6=-1, delta_e6=25_000
        )
        # The winner is an ACCEPT, never a near-miss.
        assert not near_miss(mu_priv_e6=500)

    def test_build_rejects_bad_audit16(self):
        for bad in ("ABCDEF0123456789", "0123", "0123456789abcdefg", "zzzzzzzzzzzzzzzz"):
            with pytest.raises(WireFormatError):
                build_verdict(make_verdict(audit_digest=bad))

    def test_build_rejects_bad_challenger_digest(self):
        v = make_verdict()
        object.__setattr__(v, "challenger_digest", "hf:nope")
        with pytest.raises(WireFormatError):
            build_verdict(v)

    @pytest.mark.parametrize(
        "raw",
        [
            f"ev3|{CHALL_DIGEST}|A|1|1|1|1|{AUDIT16}",         # unknown version
            f"ev1|{CHALL_DIGEST}|A|1|1|1|{AUDIT16}",           # retired format
            f"ev2|{CHALL_DIGEST}|A|1|1|1|{AUDIT16}",           # too few fields
            f"ev2|{CHALL_DIGEST}|A|1|1|1|1|{AUDIT16}|x",       # too many fields
            f"ev2|{CHALL_DIGEST}|X|1|1|1|1|{AUDIT16}",         # bad decision
            f"ev2|{CHALL_DIGEST}|a|1|1|1|1|{AUDIT16}",         # lowercase decision
            f"ev2|sha256:bad|A|1|1|1|1|{AUDIT16}",             # bad digest
            f"ev2|{CHALL_DIGEST}|A|1|1|1|1|nothexdigest!",     # bad audit16
            f"ev2|{CHALL_DIGEST}|A|1|1|1|1|{AUDIT16.upper()}", # uppercase audit16
            f"ev2|{CHALL_DIGEST}|A|1|1|-1|1|{AUDIT16}",        # negative delta
            f"ev2|{CHALL_DIGEST}|A|1|1|1|-1|{AUDIT16}",        # negative pool epoch
        ],
    )
    def test_parse_rejects_malformed(self, raw):
        with pytest.raises(WireFormatError):
            parse_verdict(raw, "v", 1)

    @pytest.mark.parametrize(
        "lcb",
        ["1_0", " 10", "10 ", "+10", "010", "١٢"],
    )
    def test_parse_rejects_non_canonical_integers(self, lcb):
        """int() would accept all of these, so two payloads could differ in
        bytes yet parse to the same verdict."""
        with pytest.raises(WireFormatError):
            parse_verdict(f"ev2|{CHALL_DIGEST}|A|{lcb}|1|1|1|{AUDIT16}", "v", 1)


class TestVerdictFloatProperties:
    def test_lcb_pub(self):
        assert make_verdict(lcb_pub_e6=31_500).lcb_pub == pytest.approx(0.0315)

    def test_mu_priv_signed(self):
        assert make_verdict(mu_priv_e6=-2_000).mu_priv == pytest.approx(-0.002)

    def test_delta(self):
        assert make_verdict(delta_e6=25_000).delta == pytest.approx(0.025)

    def test_zero(self):
        v = make_verdict(lcb_pub_e6=0, mu_priv_e6=0)
        assert v.lcb_pub == 0.0
        assert v.mu_priv == 0.0


class TestKingPointerRoundtrip:
    def test_roundtrip(self):
        raw = build_king_pointer(make_pointer())
        assert raw == (
            f"ek1|team/EPAGO-DR-4B-run1|{CHALL_DIGEST}|{HOTKEY}|9000|8000|31500|25000"
        )
        parsed = parse_king_pointer(raw, publisher_hotkey="authority", block=9_100)
        assert parsed.ref == ModelRef(repo="team/EPAGO-DR-4B-run1", digest=CHALL_DIGEST)
        assert parsed.author_hotkey == HOTKEY
        assert parsed.crowned_block == 9_000
        assert parsed.reign_started_block == 8_000
        assert parsed.coronation_lcb == pytest.approx(0.0315)
        assert parsed.coronation_delta == pytest.approx(0.025)
        assert parsed.publisher_hotkey == "authority"
        assert parsed.block == 9_100

    def test_inherited_reign_survives_the_roundtrip(self):
        """A self-dethrone carries the old reign clock; adopting validators must
        see the same clock or the incumbent gets a fresh decay curve."""
        parsed = parse_king_pointer(
            build_king_pointer(make_pointer(crowned_block=50_000, reign_started_block=1_000)),
            "authority",
            50_010,
        )
        assert parsed.reign_started_block == 1_000

    @pytest.mark.parametrize(
        "raw",
        [
            f"ek2|r|{CHALL_DIGEST}|{HOTKEY}|1|1|0|0",          # unknown version
            f"ek1|r|{CHALL_DIGEST}|{HOTKEY}|1|1|0",            # too few fields
            f"ek1|r|sha256:bad|{HOTKEY}|1|1|0|0",              # bad digest
            f"ek1|r|{CHALL_DIGEST}|{HOTKEY}|-1|1|0|0",         # negative block
            f"ek1|r|{CHALL_DIGEST}|{HOTKEY}|10|20|0|0",        # reign starts after coronation
            f"ek1|r|{CHALL_DIGEST}|{HOTKEY}|10|1|0|-5",        # negative delta
        ],
    )
    def test_parse_rejects_malformed(self, raw):
        with pytest.raises(WireFormatError):
            parse_king_pointer(raw, "authority", 1)
