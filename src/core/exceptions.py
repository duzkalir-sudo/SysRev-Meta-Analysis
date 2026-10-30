"""Domain exception hierarchy."""

from __future__ import annotations


class SysRevError(Exception):
    """Base class for all domain errors."""


class InvalidIdentifierError(SysRevError, ValueError):
    """A DOI, PMID, PMCID or registry identifier failed validation."""


class StateTransitionError(SysRevError):
    """A requested pipeline transition is not permitted."""


class GuardViolationError(StateTransitionError):
    """A transition is structurally allowed but a precondition (guard) failed."""

    def __init__(self, message: str, violations: list[str]) -> None:
        super().__init__(message)
        self.violations = violations


class PRISMAInconsistencyError(SysRevError):
    """PRISMA flow counts violate an arithmetic invariant."""


class APIClientError(SysRevError):
    """A literature API returned an unrecoverable error."""

    def __init__(self, message: str, *, status_code: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class SearchResultTooLargeError(APIClientError):
    """The query matched more records than the API can page through.

    PubMed ESearch cannot page beyond the first 10,000 records; the search must
    be partitioned (e.g. by publication date) rather than silently truncated.
    """


class SearchIncompleteError(APIClientError):
    """Retrieval returned fewer distinct records than the database reported."""


class StatisticsInputError(SysRevError):
    """Stage 5 refuses its inputs (e.g. unapproved data, a plan that contradicts the protocol)."""
