"""ASGI entry point configured from environment settings."""

from __future__ import annotations

from src.api_server.app import create_app
from src.core.config import get_settings

_settings = get_settings()
if _settings.database_url.startswith("sqlite:///./"):
    _settings.data_dir.mkdir(parents=True, exist_ok=True)

app = create_app(_settings.database_url)
