"""Deterministic, rate-limited async clients for literature databases.

These clients are the *only* entry point for bibliographic data. Records are
built exclusively from fields present in the API responses.
"""

from src.apis.crossref import CrossrefClient
from src.apis.europepmc import EuropePMCClient
from src.apis.openalex import OpenAlexClient
from src.apis.pubmed import PubMedClient
from src.apis.rate_limit import AsyncRateLimiter
from src.apis.unpaywall import UnpaywallClient

__all__ = [
    "AsyncRateLimiter",
    "CrossrefClient",
    "EuropePMCClient",
    "OpenAlexClient",
    "PubMedClient",
    "UnpaywallClient",
]
