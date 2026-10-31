"""Refresh Models: list provider models and store a candidate catalogue snapshot.

Runs in the process that holds the API keys (the local UI), like the AI steps; it is not an
HTTP endpoint, so keys never travel over HTTP. It makes **no inference request**: only the
official ``GET /v1/models`` endpoints are called (:class:`src.llm.discovery.ModelLister`).
The candidate is stored but not active; an enrolled reviewer previews and activates it.
"""

from __future__ import annotations

from typing import Any

from src.llm.catalogue import ModelCatalogue
from src.llm.discovery import ModelLister
from src.llm.model_registry import CatalogueSnapshot, diff, refresh_snapshot
from src.state.repository import Repository


def refresh_catalogue(
    repo: Repository,
    base: ModelCatalogue,
    lister: ModelLister,
    *,
    created_by: str = "local UI (Refresh Models)",
    providers: tuple[str, ...] = ("anthropic", "openai"),
) -> tuple[CatalogueSnapshot, dict[str, Any]]:
    """Store a candidate snapshot and return it with its preview against the active one."""
    active = repo.active_catalogue(base)
    results = lister.refresh(providers)
    snapshot = refresh_snapshot(active, results, created_by=created_by)
    repo.save_catalogue_snapshot(snapshot)
    return snapshot, diff(active, snapshot.catalogue())
