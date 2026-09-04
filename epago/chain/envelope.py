"""Sealed envelopes: encrypt a secret to a miner's hotkey, in the open.

A miner uploads its checkpoint into the validator's own object store rather
than to a public repository, so competitors cannot read a challenger that has
not won. To do that the miner needs write credentials, and those credentials
have to travel over a public channel — there is no private channel between a
validator and a miner that does not reintroduce an operator.

So they travel encrypted. The validator seals each miner's credentials to that
miner's **hotkey public key**, publishes every envelope in one public mailbox,
and only the holder of the corresponding private key can open one. Anybody can
read the mailbox; nobody can read anyone else's envelope.

**Why Ed25519 hotkeys are required.** Bittensor hotkeys default to sr25519,
which signs but has no defined encryption. Ed25519 does: its public key is a
point on a twisted Edwards curve that maps birationally onto Curve25519, so an
Ed25519 signing key can be converted to an X25519 key-exchange key and used
with a sealed box. sr25519 has no such mapping in general use, which is why a
miner must register an Ed25519 hotkey to submit. The same requirement, for the
same reason, appears in other king-of-the-hill subnets that keep challengers
private.

**What a sealed box gives.** Anonymous public-key encryption: the sender
generates an ephemeral keypair per message, derives a shared secret with the
recipient's X25519 key, and discards its own private key. The recipient needs
only its own key to open the envelope. There is no shared secret to distribute,
no session to establish, and nothing the validator must remember per miner.

**What it does not give, and what we add.** A sealed box is anonymous: the
recipient learns nothing about who sealed it, because anyone holding a public
key can seal to it. Left there, a forged envelope would hand a miner
credentials that quietly fail, or point it at an attacker's bucket. So the
payload is **signed by the validator before it is sealed**, and the miner
verifies that signature after opening. Confidentiality comes from the sealing;
authenticity comes from the signature. Neither substitutes for the other.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

#: Envelope format tag. A miner's client refuses anything it does not know,
#: rather than guessing at a payload it cannot verify.
ENVELOPE_VERSION = "epenv1"


class EnvelopeError(RuntimeError):
    """Sealing or opening failed. Never carries key material in its message."""


def _nacl():
    """PyNaCl, imported lazily with an actionable error.

    Kept out of module import so a validator that never issues credentials, and
    every replay tool, run without the dependency.
    """
    try:
        import nacl.public  # noqa: F401
        import nacl.signing  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise EnvelopeError(
            "sealed envelopes need PyNaCl: pip install 'epago[chain]'"
        ) from exc
    import nacl.public
    import nacl.signing

    return nacl.public, nacl.signing


def x25519_public_from_ed25519(public_key: bytes) -> bytes:
    """Convert an Ed25519 *public* key to its X25519 counterpart.

    The two curves are birationally equivalent, so this is a coordinate
    transform rather than a re-derivation: the same secret opens both.
    """
    if len(public_key) != 32:
        raise EnvelopeError(f"an Ed25519 public key is 32 bytes, got {len(public_key)}")
    _, signing = _nacl()
    try:
        return bytes(signing.VerifyKey(public_key).to_curve25519_public_key())
    except Exception as exc:  # noqa: BLE001 - a bad point is a bad key, not a crash
        raise EnvelopeError("public key is not a valid Ed25519 point") from exc


def x25519_private_from_ed25519(private_key: bytes) -> bytes:
    """Convert an Ed25519 *private* key (32-byte seed) to X25519."""
    if len(private_key) != 32:
        raise EnvelopeError(f"an Ed25519 seed is 32 bytes, got {len(private_key)}")
    _, signing = _nacl()
    try:
        return bytes(signing.SigningKey(private_key).to_curve25519_private_key())
    except Exception as exc:  # noqa: BLE001
        raise EnvelopeError("seed is not a valid Ed25519 private key") from exc


@dataclass(frozen=True, slots=True)
class Envelope:
    """One sealed payload addressed to one hotkey."""

    version: str
    hotkey: str
    ciphertext: str  # base64

    def to_dict(self) -> dict:
        return {"version": self.version, "hotkey": self.hotkey, "ciphertext": self.ciphertext}

    @classmethod
    def from_dict(cls, data: dict) -> "Envelope":
        version = str(data.get("version", ""))
        if version != ENVELOPE_VERSION:
            raise EnvelopeError(f"unknown envelope version {version!r}")
        hotkey = str(data.get("hotkey", ""))
        ciphertext = str(data.get("ciphertext", ""))
        if not hotkey or not ciphertext:
            raise EnvelopeError("envelope is missing hotkey or ciphertext")
        return cls(version=version, hotkey=hotkey, ciphertext=ciphertext)


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def seal(
    payload: dict,
    recipient_ed25519_public: bytes,
    hotkey: str,
    *,
    signer_seed: bytes | None = None,
) -> Envelope:
    """Seal ``payload`` so only the holder of ``hotkey``'s key can read it.

    When ``signer_seed`` is given the payload is signed first and the signature
    travels inside the envelope, so the recipient can confirm the validator
    wrote it. Without a signature a miner could be handed a forged envelope
    pointing at an attacker's bucket — sealing proves nobody else *read* it,
    never that the right party *wrote* it.

    The payload is canonical JSON, so an envelope is a pure function of its
    contents up to the ephemeral key.
    """
    public, signing = _nacl()
    body = dict(payload)
    if signer_seed is not None:
        key = signing.SigningKey(signer_seed)
        body["signature"] = base64.b64encode(
            key.sign(_canonical(payload)).signature
        ).decode()
        body["signer"] = base64.b64encode(bytes(key.verify_key)).decode()
    recipient = public.PublicKey(x25519_public_from_ed25519(recipient_ed25519_public))
    raw = _canonical(body)
    try:
        ciphertext = public.SealedBox(recipient).encrypt(raw)
    except Exception as exc:  # noqa: BLE001
        raise EnvelopeError("sealing failed") from exc
    return Envelope(
        version=ENVELOPE_VERSION,
        hotkey=hotkey,
        ciphertext=base64.b64encode(ciphertext).decode(),
    )


def open_envelope(
    envelope: Envelope,
    recipient_ed25519_seed: bytes,
    *,
    expect_signer: bytes | None = None,
) -> dict:
    """Open an envelope addressed to us, optionally verifying who wrote it.

    ``expect_signer`` is the validator's Ed25519 public key. Supplying it turns
    "somebody sealed this to me" into "the validator sealed this to me", which
    is the difference between usable credentials and a plausible forgery.
    """
    public, _ = _nacl()
    secret = public.PrivateKey(x25519_private_from_ed25519(recipient_ed25519_seed))
    try:
        raw = public.SealedBox(secret).decrypt(base64.b64decode(envelope.ciphertext))
    except Exception as exc:  # noqa: BLE001 - wrong key and corrupt bytes look alike
        raise EnvelopeError(
            "could not open this envelope: it is addressed to a different hotkey, "
            "or the mailbox entry is corrupt"
        ) from exc
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EnvelopeError("envelope opened but its contents are not JSON") from exc
    if expect_signer is not None:
        verify_payload(body, expect_signer)
    return body


def verify_payload(body: dict, signer_public: bytes) -> None:
    """Confirm an opened payload was signed by ``signer_public``.

    Raises rather than returning a flag: a caller that forgets to check a
    boolean would use unverified credentials, which is precisely the failure
    this exists to prevent.
    """
    _, signing = _nacl()
    signature = body.get("signature")
    if not signature:
        raise EnvelopeError("envelope carries no signature; refusing to trust it")
    payload = {k: v for k, v in body.items() if k not in ("signature", "signer")}
    try:
        signing.VerifyKey(signer_public).verify(
            _canonical(payload), base64.b64decode(signature)
        )
    except Exception as exc:  # noqa: BLE001
        raise EnvelopeError(
            "envelope signature does not verify against the expected validator key"
        ) from exc
