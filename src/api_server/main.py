"""ASGI entry point configured from environment settings.

Bind to loopback only (uvicorn's default ``--host 127.0.0.1``); reviewer tokens are sent
as bearer headers and must not cross a network without TLS.
"""

from __future__ import annotations

from src.api_server.app import create_app
from src.core.config import get_settings
from src.state.db import prepare_sqlite_path

_settings = get_settings()
prepare_sqlite_path(_settings.database_url)

app = create_app(_settings.database_url)
