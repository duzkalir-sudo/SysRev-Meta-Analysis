from __future__ import annotations

import pytest

from src.core.config import Settings, get_settings


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSREV_NCBI_EMAIL", "me@example.org")
    monkeypatch.setenv("SYSREV_DEDUP_TITLE_THRESHOLD", "97")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.ncbi_email == "me@example.org"
    assert s.dedup_title_threshold == 97
    assert s.ncbi_requests_per_second == 3.0


def test_ncbi_rate_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSREV_NCBI_API_KEY", "secret")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.ncbi_requests_per_second == 10.0
    assert "secret" not in repr(s)  # SecretStr is masked


def test_settings_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSREV_HTTP_MAX_RETRIES", "99")
    with pytest.raises(ValueError, match="http_max_retries"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
