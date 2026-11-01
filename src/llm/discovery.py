"""Model discovery through the providers' official model-list endpoints ("Refresh Models").

Only ``GET /v1/models`` is ever requested: discovery cannot start an inference call and is
not billed as one. The API key is read through :class:`CredentialResolver`, sent only in the
request header, registered for redaction and never stored.

What a listing establishes — and what it does not:

* an ID returned for this key means the provider *listed* it for this account at that time
  (``account availability: listed``). It does **not** establish that the model works with
  this software's adapter, which efforts it accepts, its output limit or its price;
* Anthropic's endpoint also reports lifecycle stage, retirement date, ``max_tokens`` and a
  ``capabilities`` object (effort levels, thinking types, structured outputs). These are kept
  as *provider-reported metadata* to help a reviewer verify the entry; they are never used
  to make a model executable on their own;
* OpenAI's endpoint reports only ``id``, ``created``, ``owned_by`` and an optional
  ``shutdown_date``; capabilities and prices must come from the model's documentation page.

Provider facts (official API references, retrieved 2026-10-31):

* Anthropic ``GET https://api.anthropic.com/v1/models`` — headers ``x-api-key`` (documented
  as the still-supported alternative to ``Authorization: Bearer``) and ``anthropic-version``;
  cursor pagination ``limit`` (1-1000, default 20), ``after_id``; response ``data``,
  ``has_more``, ``last_id``; items ``id``, ``display_name``, ``created_at``, ``lifecycle``
  (``active`` | ``deprecated`` | ``retired``; retired models are omitted unless requested),
  ``deprecated_at``, ``retires_at``, ``max_tokens``, ``max_input_tokens``, ``capabilities``.
* OpenAI ``GET https://api.openai.com/v1/models`` — ``Authorization: Bearer``; no documented
  pagination; items ``id``, ``created`` (Unix seconds), ``owned_by``, optional
  ``shutdown_date`` ("the date when the model will shut down, or null if not announced").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

import httpx

from src.llm.adapters import ANTHROPIC_VERSION
from src.llm.credentials import CredentialResolver, redact, register_secret
from src.llm.provider import ProviderAccessError

ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
MAX_PAGES = 20  # Anthropic pagination bound (1000 per page)

DiscoveryStatus = Literal["ok", "access_error", "unavailable", "no_credentials", "bad_response"]


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    provider: str
    model_id: str
    display_name: str | None = None
    created_at: str | None = None
    lifecycle: Literal["active", "deprecated", "retired"] | None = None
    retires_at: date | None = None
    reported: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Outcome of one provider listing. On failure ``models`` is empty and the previous
    catalogue is left exactly as it was."""

    provider: str
    status: DiscoveryStatus
    endpoint: str
    fetched_at: datetime
    models: tuple[DiscoveredModel, ...] = ()
    error: str | None = None
    requests: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def listed_ids(self) -> list[str]:
        return sorted(m.model_id for m in self.models)


def _date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None


def _obj(node: Any) -> dict[str, Any]:
    return node if isinstance(node, dict) else {}


def _supported(node: Any) -> bool | None:
    if isinstance(node, dict) and isinstance(node.get("supported"), bool):
        return bool(node["supported"])
    return None


def anthropic_reported(item: dict[str, Any]) -> dict[str, Any]:
    """The subset of Anthropic's model metadata relevant to this software (as reported)."""
    caps = _obj(item.get("capabilities"))
    effort = _obj(caps.get("effort"))
    thinking = _obj(caps.get("thinking"))
    types = _obj(thinking.get("types"))
    # Levels are reported only when the provider returned an effort object: absent metadata
    # is "not reported", never "no effort levels".
    efforts = (
        [lvl for lvl in ("low", "medium", "high", "xhigh", "max")
         if _supported(effort.get(lvl)) is True]
        if _supported(effort) is not None
        else None
    )  # fmt: skip
    out: dict[str, Any] = {
        "max_tokens": item.get("max_tokens"),
        "max_input_tokens": item.get("max_input_tokens"),
        "effort_supported": _supported(effort),
        "effort_levels": efforts,
        "thinking_adaptive": _supported(types.get("adaptive")),
        "structured_outputs": _supported(caps.get("structured_outputs")),
        "batch": _supported(caps.get("batch")),
        "deprecated_at": item.get("deprecated_at"),
        "line": item.get("line"),
    }
    return {k: v for k, v in out.items() if v is not None}


class ModelLister:
    """Calls the two official list endpoints with an injectable ``httpx.Client``."""

    def __init__(
        self,
        credentials: CredentialResolver,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = 30.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._credentials = credentials
        self._client = client or httpx.Client()
        self._timeout = timeout_seconds
        self._clock = clock

    def refresh(
        self, providers: tuple[str, ...] = ("anthropic", "openai")
    ) -> list[DiscoveryResult]:
        return [self.list_models(p) for p in providers]

    def list_models(self, provider: str) -> DiscoveryResult:
        endpoint = ANTHROPIC_MODELS_URL if provider == "anthropic" else OPENAI_MODELS_URL
        now = self._clock()
        if provider not in ("anthropic", "openai"):
            return DiscoveryResult(provider, "bad_response", endpoint, now,
                                   error=f"unknown provider {provider!r}")  # fmt: skip
        try:
            key = self._credentials.get(provider).get_secret_value()
        except ProviderAccessError as exc:
            return DiscoveryResult(provider, "no_credentials", endpoint, now,
                                   error=redact(str(exc))[:500])  # fmt: skip
        register_secret(key)
        try:
            if provider == "anthropic":
                models, requests = self._anthropic(key)
            else:
                models, requests = self._openai(key)
        except _ListingError as exc:
            return DiscoveryResult(provider, exc.status, endpoint, now, error=exc.message,
                                   requests=exc.requests)  # fmt: skip
        return DiscoveryResult(provider, "ok", endpoint, now, tuple(models), requests=requests)

    # ------------------------------------------------------------------ providers
    def _get(self, url: str, headers: dict[str, str], params: dict[str, Any], n: int) -> Any:
        try:
            resp = self._client.get(url, headers=headers, params=params, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise _ListingError(
                "unavailable", f"network error ({type(exc).__name__}); nothing changed", n
            ) from None
        if resp.status_code in (401, 403):
            raise _ListingError(
                "access_error",
                f"HTTP {resp.status_code}: the key was rejected or may not list models "
                f"({redact(resp.text[:200])})",
                n,
            )
        if resp.status_code != 200:
            raise _ListingError(
                "unavailable",
                f"HTTP {resp.status_code}: provider unavailable ({redact(resp.text[:200])}); "
                "nothing changed",
                n,
            )
        try:
            return resp.json()
        except ValueError:
            raise _ListingError("bad_response", "response body is not JSON", n) from None

    def _anthropic(self, key: str) -> tuple[list[DiscoveredModel], int]:
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        params: dict[str, Any] = {"limit": 1000}
        models: list[DiscoveredModel] = []
        for page in range(1, MAX_PAGES + 1):
            data = self._get(ANTHROPIC_MODELS_URL, headers, params, page)
            items = data.get("data") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise _ListingError("bad_response", "no 'data' list in the response", page)
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    raise _ListingError("bad_response", "a listed model has no id", page)
                lifecycle = item.get("lifecycle")
                models.append(
                    DiscoveredModel(
                        provider="anthropic",
                        model_id=item["id"],
                        display_name=str(item.get("display_name") or "") or None,
                        created_at=str(item.get("created_at") or "") or None,
                        lifecycle=lifecycle
                        if lifecycle in ("active", "deprecated", "retired")
                        else None,
                        retires_at=_date(item.get("retires_at")),
                        reported=anthropic_reported(item),
                    )
                )
            if not data.get("has_more"):
                return models, page
            last = data.get("last_id")
            if not isinstance(last, str) or not last:
                raise _ListingError("bad_response", "has_more without last_id", page)
            params = {"limit": 1000, "after_id": last}
        raise _ListingError("bad_response", f"more than {MAX_PAGES} pages; refused", MAX_PAGES)

    def _openai(self, key: str) -> tuple[list[DiscoveredModel], int]:
        data = self._get(OPENAI_MODELS_URL, {"Authorization": f"Bearer {key}"}, {}, 1)
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise _ListingError("bad_response", "no 'data' list in the response", 1)
        models = []
        today = self._clock().date()
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise _ListingError("bad_response", "a listed model has no id", 1)
            shutdown = _date(item.get("shutdown_date"))
            lifecycle: Literal["active", "deprecated", "retired"] | None = None
            if shutdown is not None:  # an announced shutdown; past date = no longer usable
                lifecycle = "retired" if shutdown <= today else "deprecated"
            created = item.get("created")
            models.append(
                DiscoveredModel(
                    provider="openai",
                    model_id=item["id"],
                    created_at=(
                        datetime.fromtimestamp(created, UTC).isoformat()
                        if isinstance(created, int)
                        else None
                    ),
                    lifecycle=lifecycle,
                    retires_at=shutdown,
                    reported={
                        k: item[k] for k in ("owned_by", "shutdown_date") if item.get(k) is not None
                    },
                )
            )
        return models, 1


class _ListingError(Exception):
    def __init__(self, status: DiscoveryStatus, message: str, requests: int) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.requests = requests
