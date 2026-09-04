"""The credential mailbox: one public file, one sealed envelope per miner.

A miner uploads its checkpoint into the validator's own bucket rather than to a
public repository, so a challenger that has not won stays unreadable by its
rivals. That requires the miner to hold write credentials, and there is no
private channel to deliver them over — any such channel would reintroduce an
operator, which the design forbids.

So the channel is public and the payload is not. The validator writes one file
containing an envelope per registered miner, each sealed to that miner's hotkey
(:mod:`epago.chain.envelope`), and publishes it in the object store beside
everything else it publishes. Every miner reads the same file; each can open
exactly one entry.

Three properties this file must have, and why:

*Scoped.* Each envelope carries a prefix the miner may write to and nothing
else. A credential that could write anywhere would let one miner overwrite
another's submission, which is worse than the public-repo model it replaces.

*Expiring.* Credentials carry an expiry and the mailbox is republished on a
cadence. A leaked key is then bounded in time rather than permanently valid,
and a miner who never submits stops holding live credentials.

*Not a secret store.* The mailbox is written to a public path on purpose. Its
security is the sealing, not its location — treating the location as the secret
would mean one misconfigured bucket policy exposed every credential at once.

The mailbox is deliberately not on chain. Envelopes are kilobytes and rotate;
chain commitments are small and permanent. What goes on chain is the digest, so
a miner can tell whether the file it fetched is the one the validator published.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from epago.chain.envelope import Envelope, EnvelopeError, seal

#: Wire format tag for the whole file, separate from the per-envelope version so
#: the container can change without invalidating envelopes inside it.
MAILBOX_VERSION = "epmbx1"

#: Where the mailbox lives inside a validator's published namespace.
MAILBOX_KEY = "mailbox/credentials.json"

#: How long issued credentials stay valid. Long enough to upload a large
#: checkpoint on a slow link, short enough that a leaked key is not a standing
#: liability. The validator republishes well inside this window.
DEFAULT_TTL_SECONDS = 6 * 3600


def submission_prefix(hotkey: str) -> str:
    """The one key prefix a given miner may write to.

    Derived from the hotkey rather than assigned, so two validators computing a
    miner's prefix agree without coordinating, and a miner can predict its own
    prefix before it ever reads the mailbox.
    """
    if not hotkey or "/" in hotkey or ".." in hotkey:
        raise ValueError(f"unusable hotkey for a key prefix: {hotkey!r}")
    return f"submissions/{hotkey}/"


@dataclass(frozen=True, slots=True)
class Mailbox:
    """A published set of sealed credential envelopes."""

    version: str
    issued_at: int
    expires_at: int
    envelopes: tuple[Envelope, ...] = field(default_factory=tuple)

    def to_json(self) -> str:
        """Canonical JSON, so the digest is a function of contents alone."""
        return json.dumps(
            {
                "version": self.version,
                "issued_at": self.issued_at,
                "expires_at": self.expires_at,
                # Sorted by hotkey: a republished mailbox with the same members
                # differs only where the ciphertext differs, which makes an
                # unexpected diff worth looking at.
                "envelopes": [e.to_dict() for e in sorted(self.envelopes, key=lambda x: x.hotkey)],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_json().encode()).hexdigest()

    def for_hotkey(self, hotkey: str) -> Envelope | None:
        for envelope in self.envelopes:
            if envelope.hotkey == hotkey:
                return envelope
        return None

    def is_expired(self, now: int | None = None) -> bool:
        return (now if now is not None else int(time.time())) >= self.expires_at

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Mailbox":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EnvelopeError("mailbox is not valid JSON") from exc
        version = str(data.get("version", ""))
        if version != MAILBOX_VERSION:
            raise EnvelopeError(f"unknown mailbox version {version!r}")
        return cls(
            version=version,
            issued_at=int(data.get("issued_at", 0)),
            expires_at=int(data.get("expires_at", 0)),
            envelopes=tuple(Envelope.from_dict(e) for e in data.get("envelopes", ())),
        )


def build_mailbox(
    recipients: dict[str, bytes],
    issue: "callable",
    *,
    signer_seed: bytes | None = None,
    now: int | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> Mailbox:
    """Seal one credential set per recipient.

    ``recipients`` maps hotkey to its **Ed25519 public key**; ``issue`` is
    called as ``issue(hotkey, prefix, expires_at)`` and returns the credential
    dict to seal. Issuance is injected rather than performed here so this
    module never touches a cloud provider's API, stays unit-testable without
    credentials, and cannot log a secret it never sees.

    ``signer_seed`` is the validator's Ed25519 seed. Supplying it signs every
    payload before sealing, which is what lets a miner tell a real credential
    from a forged envelope pointing at somebody else's bucket. Sealing alone
    proves nobody else read it, never that the right party wrote it.

    A recipient whose key cannot be sealed to is skipped rather than aborting
    the whole mailbox: one miner registering an unusable hotkey must not stop
    every other miner from receiving credentials.
    """
    now = int(time.time()) if now is None else now
    expires_at = now + int(ttl_seconds)
    envelopes: list[Envelope] = []
    for hotkey in sorted(recipients):
        try:
            prefix = submission_prefix(hotkey)
            payload = dict(issue(hotkey, prefix, expires_at))
            # The prefix is stated in the payload as well as enforced by the
            # credential, so a miner can check where it is allowed to write
            # without having to provoke a permission error to find out.
            payload.setdefault("prefix", prefix)
            payload.setdefault("expires_at", expires_at)
            envelopes.append(
                seal(payload, recipients[hotkey], hotkey, signer_seed=signer_seed)
            )
        except Exception:  # noqa: BLE001 - one bad recipient is not a bad mailbox
            continue
    return Mailbox(
        version=MAILBOX_VERSION,
        issued_at=now,
        expires_at=expires_at,
        envelopes=tuple(envelopes),
    )


def write_mailbox(mailbox: Mailbox, path: Path) -> str:
    """Write the mailbox atomically and return its digest.

    Atomic because a miner may fetch at any moment: a half-written mailbox
    would parse as corrupt and send an honest miner chasing a fault that does
    not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(mailbox.to_json())
    tmp.replace(path)
    return mailbox.digest()
