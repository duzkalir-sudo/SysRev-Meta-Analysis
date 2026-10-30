"""Reviewer credential primitives (pure functions; no I/O).

A reviewer token has the form ``rvw_<reviewer_id>_<secret>``. The 256-bit ``secret``
is shown once to the human at enrollment; only a salted scrypt hash is stored.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

TOKEN_PREFIX = "rvw"
_TOKEN_RE = re.compile(r"^rvw_([0-9a-f]{32})_([A-Za-z0-9_-]{43,})$")

# scrypt parameters (RFC 7914). n=2**14, r=8, p=1 is the interactive-login setting
# suggested in Percival's scrypt paper; the token's 256-bit entropy is the main defence.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _DKLEN = 2**14, 8, 1, 32


def generate_secret() -> str:
    return secrets.token_urlsafe(32)


def new_salt() -> str:
    return secrets.token_hex(16)


def hash_secret(secret: str, salt_hex: str) -> str:
    digest = hashlib.scrypt(
        secret.encode(),
        salt=bytes.fromhex(salt_hex),
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DKLEN,
    )
    return digest.hex()


def verify_secret(secret: str, salt_hex: str, expected_hash_hex: str) -> bool:
    return hmac.compare_digest(hash_secret(secret, salt_hex), expected_hash_hex)


def format_token(reviewer_id: str, secret: str) -> str:
    return f"{TOKEN_PREFIX}_{reviewer_id}_{secret}"


def parse_token(token: str) -> tuple[str, str] | None:
    """Return ``(reviewer_id, secret)`` or ``None`` for a malformed token."""
    match = _TOKEN_RE.match(token.strip())
    return (match.group(1), match.group(2)) if match else None
