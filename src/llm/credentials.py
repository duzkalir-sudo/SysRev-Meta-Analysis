"""API credentials for the LLM providers, and secret redaction.

Keys are never written to the project database, configuration files, exports or logs.
Resolution order for a provider key:

1. a key entered in the running UI session (process memory only, gone on restart);
2. the environment variable (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``), or the same
   name in the git-ignored ``.env`` file;
3. the operating-system credential store via the optional ``keyring`` package
   (Windows Credential Manager, macOS Keychain, Secret Service) — only if the user
   explicitly saved the key there.

Billing note: an API key bills the provider *API* account that issued it (Anthropic
Console / OpenAI Platform). Claude subscription or Claude Code cloud development credits,
and ChatGPT subscriptions, cannot pay for these API calls.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import threading
from types import ModuleType
from typing import Literal

from pydantic import SecretStr

from src.llm.provider import ProviderAccessError

KEYRING_SERVICE = "sysrev-meta-analysis"
ENV_VARS: dict[str, str] = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
BILLING_NOTE = (
    "API calls are billed to the API account that owns the key (Anthropic Console / "
    "OpenAI Platform). Claude subscription or Claude Code cloud development credits and "
    "ChatGPT subscriptions do not pay for them."
)
CredentialSource = Literal["session", "environment", "os-keyring"]

_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"rvw_[0-9a-f]{32}_[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(x-api-key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-]{8,}"),
)
_known_secrets: set[str] = set()
_lock = threading.Lock()


def register_secret(value: str) -> None:
    """Remember a secret so :func:`redact` removes it even if it matches no pattern."""
    if len(value) >= 8:
        with _lock:
            _known_secrets.add(value)


def redact(text: str) -> str:
    with _lock:
        known = sorted(_known_secrets, key=len, reverse=True)
    for secret in known:
        text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


class RedactingFilter(logging.Filter):
    """Redacts secrets from log records (message and arguments)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


def install_log_redaction() -> None:
    """Attach :class:`RedactingFilter` to the root handlers (idempotent)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO)
    for handler in root.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _keyring() -> ModuleType | None:
    try:
        return importlib.import_module("keyring")
    except ImportError:
        return None


class CredentialResolver:
    def __init__(self, *, use_keyring: bool = True, use_dotenv: bool = True) -> None:
        self._session: dict[str, SecretStr] = {}
        self._use_keyring = use_keyring
        self._use_dotenv = use_dotenv

    def set_session_key(self, provider: str, key: str) -> None:
        key = key.strip()
        if not key:
            raise ValueError("empty API key")
        register_secret(key)
        self._session[provider] = SecretStr(key)

    def clear_session_key(self, provider: str) -> None:
        self._session.pop(provider, None)

    def _lookup(self, provider: str) -> tuple[SecretStr, CredentialSource] | None:
        if provider in self._session:
            return self._session[provider], "session"
        env = os.environ.get(ENV_VARS.get(provider, ""), "").strip()
        if env:
            return SecretStr(env), "environment"
        if self._use_dotenv:
            from src.core.config import get_settings

            dotenv = getattr(get_settings(), f"{provider}_api_key", None)
            if isinstance(dotenv, SecretStr) and dotenv.get_secret_value().strip():
                return SecretStr(dotenv.get_secret_value().strip()), "environment"
        if self._use_keyring and (kr := _keyring()) is not None:
            try:
                stored = kr.get_password(KEYRING_SERVICE, provider)
            except Exception:  # backend unavailable: treated as "not stored"
                stored = None
            if stored:
                return SecretStr(str(stored)), "os-keyring"
        return None

    def source(self, provider: str) -> CredentialSource | None:
        found = self._lookup(provider)
        return found[1] if found else None

    def get(self, provider: str) -> SecretStr:
        found = self._lookup(provider)
        if found is None:
            raise ProviderAccessError(
                f"no API key for {provider}: enter one in the UI, set "
                f"{ENV_VARS.get(provider, 'the provider key variable')}, or save it in the OS "
                f"credential store. {BILLING_NOTE}"
            )
        register_secret(found[0].get_secret_value())
        return found[0]

    @staticmethod
    def keyring_available() -> bool:
        return _keyring() is not None

    def save_to_keyring(self, provider: str, key: str) -> None:
        kr = _keyring()
        if kr is None:
            raise RuntimeError("the optional 'keyring' package is not installed")
        register_secret(key.strip())
        kr.set_password(KEYRING_SERVICE, provider, key.strip())

    def delete_from_keyring(self, provider: str) -> None:
        kr = _keyring()
        if kr is not None:
            try:
                kr.delete_password(KEYRING_SERVICE, provider)
            except Exception:  # not stored
                return
