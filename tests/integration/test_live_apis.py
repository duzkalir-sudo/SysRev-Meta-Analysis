"""Opt-in smoke tests against the real public APIs: ``pytest -m live``.

Excluded from the default run (network access is not guaranteed in CI or
sandboxes). They assert only on response *shape* and on the stable PRISMA
2020 statement record (Page MJ et al. BMJ 2021;372:n71, doi:10.1136/bmj.n71),
never on hit counts, which change daily.
"""

from __future__ import annotations

import os

import pytest

from src.apis import CrossrefClient, EuropePMCClient, OpenAlexClient, PubMedClient, UnpaywallClient

pytestmark = pytest.mark.live

CONTACT = os.environ.get("SYSREV_NCBI_EMAIL", "")
PRISMA_DOI = "10.1136/bmj.n71"

needs_email = pytest.mark.skipif(not CONTACT, reason="set SYSREV_NCBI_EMAIL to run live tests")


@needs_email
async def test_live_pubmed_doi_lookup() -> None:
    async with PubMedClient(email=CONTACT) as c:
        result, records = await c.search(f"{PRISMA_DOI}[doi]")
    assert result.count >= 1
    assert any(r.doi == PRISMA_DOI for r in records)


@needs_email
async def test_live_pubmed_mesh_check() -> None:
    async with PubMedClient(email=CONTACT) as c:
        assert await c.mesh_descriptor_exists("Diabetes Mellitus")


async def test_live_europepmc() -> None:
    async with EuropePMCClient() as c:
        _, records = await c.search(f'DOI:"{PRISMA_DOI}"', max_records=5)
    assert any(r.doi == PRISMA_DOI for r in records)


async def test_live_openalex() -> None:
    async with OpenAlexClient(mailto=CONTACT or None) as c:
        record = await c.get_work_by_doi(PRISMA_DOI)
    assert record is not None and record.doi == PRISMA_DOI


async def test_live_crossref() -> None:
    async with CrossrefClient(mailto=CONTACT or None) as c:
        work = await c.get_work(PRISMA_DOI)
    assert work is not None and work.year == 2021


@needs_email
async def test_live_unpaywall() -> None:
    async with UnpaywallClient(email=CONTACT) as c:
        result = await c.lookup(PRISMA_DOI)
    assert result is not None and result.doi == PRISMA_DOI
