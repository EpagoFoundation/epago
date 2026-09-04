"""Prefix-scoped, expiring upload credentials, minted locally.

A miner uploads its checkpoint into the validator's bucket so a challenger that
has not won stays unreadable by its rivals. That needs a credential, and the
credential has to be narrow: one miner must not be able to read, overwrite or
delete another's submission.

R2 mints these without a network round trip. The validator holds one **parent
token** for the bucket, and derives a per-miner credential by signing a JWT with
the parent secret: the token names the bucket, the one prefix it may touch, an
expiry, and an explicit list of permitted actions. The gateway validates that
signature itself, so issuing a credential is pure computation — no API call, no
rate limit, no failure mode between deciding to issue and being able to.

**The action list is write-only on purpose.** Upload and multipart operations
only: no ``GetObject``, no ``ListObjectsV2``, not even within the miner's own
prefix. A miner does not need to read back what it just wrote, and a credential
that cannot read is a credential that cannot exfiltrate — a leaked one can only
add bytes to one prefix until it expires.

Nothing here talks to Cloudflare, so it is testable without an account and
cannot log a secret it never fetched.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib.parse import urlparse

#: R2's own ceiling on a temporary credential's lifetime (7 days).
MAX_TTL_SECONDS = 604_800

#: Exactly what a miner needs to upload a checkpoint, and nothing else.
#: Reading is absent deliberately: see the module docstring.
UPLOAD_ACTIONS = (
    "PutObject",
    "CreateMultipartUpload",
    "UploadPart",
    "CompleteMultipartUpload",
    "AbortMultipartUpload",
    "ListParts",
)

#: The scope name R2 expects for prefix-restricted object credentials.
PREFIX_SCOPE = "object-read-write"


@dataclass(frozen=True, slots=True)
class TemporaryCredentials:
    """One miner's upload credential. Valid until ``expires_at_unix``."""

    access_key_id: str
    secret_access_key: str
    session_token: str
    expires_at_unix: int

    def as_payload(self, *, endpoint: str, bucket: str, prefix: str) -> dict:
        """The dict that gets sealed into a miner's envelope.

        Carries everything an S3 client needs and nothing more. The prefix is
        included so a miner can see where it may write without having to
        provoke a permission error to find out.
        """
        return {
            "endpoint": endpoint,
            "bucket": bucket,
            "prefix": prefix,
            "access_key": self.access_key_id,
            "secret_key": self.secret_access_key,
            "session_token": self.session_token,
            "expires_at": self.expires_at_unix,
        }


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def mint_upload_credentials(
    *,
    endpoint: str,
    account_id: str,
    parent_access_key_id: str,
    parent_secret_access_key: str,
    bucket: str,
    prefix: str,
    ttl_seconds: int,
    actions: tuple[str, ...] = UPLOAD_ACTIONS,
    issued_at_unix: int | None = None,
) -> TemporaryCredentials:
    """Derive a credential that may only write under ``prefix``.

    Every argument is validated before anything is signed. A malformed prefix
    is the dangerous case: an empty or absolute one would widen the credential
    to the whole bucket, which is the single mistake this module exists to
    prevent, and it would do so silently.
    """
    if not endpoint or not account_id:
        raise ValueError("endpoint and account_id are required")
    if not parent_access_key_id or not parent_secret_access_key:
        raise ValueError("a parent token is required to mint credentials")
    if not bucket:
        raise ValueError("bucket is required")
    if not prefix or prefix.startswith("/") or ".." in prefix:
        raise ValueError(f"unusable prefix {prefix!r}: it must be relative and non-empty")
    if not prefix.endswith("/"):
        # A prefix that does not end in a separator matches sibling names by
        # accident: "submissions/hk-a" would also cover "submissions/hk-abc".
        raise ValueError(f"prefix {prefix!r} must end with '/' so it cannot match a sibling")
    if not 1 <= ttl_seconds <= MAX_TTL_SECONDS:
        raise ValueError(f"ttl must be between 1 and {MAX_TTL_SECONDS} seconds")
    if not actions:
        raise ValueError("an empty action list would grant nothing")

    issued_at = int(time.time()) if issued_at_unix is None else int(issued_at_unix)
    expires_at = issued_at + int(ttl_seconds)

    header = {"alg": "HS256", "typ": "JWT"}
    claims = {
        "aud": urlparse(endpoint).netloc,
        "bucket": bucket,
        "exp": expires_at,
        "iat": issued_at,
        "iss": parent_access_key_id,
        "paths": {"objectPaths": [], "prefixPaths": [prefix]},
        "scope": PREFIX_SCOPE,
        "sub": account_id,
        "actions": list(actions),
    }
    signing_input = (
        _b64url(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
    )
    signature = _b64url(
        hmac.new(
            parent_secret_access_key.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
    )
    jwt = f"{signing_input}.{signature}"
    return TemporaryCredentials(
        access_key_id=parent_access_key_id,
        secret_access_key=hashlib.sha256(jwt.encode()).hexdigest(),
        session_token=base64.b64encode(f"jwt/{jwt}".encode()).decode(),
        expires_at_unix=expires_at,
    )
