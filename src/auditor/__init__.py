"""Independent Auditor Agent (Stage 6) — implementation pending.

Contract: produce an ``src.core.schemas.AuditReport`` in which every number in
the manuscript is matched to a key in ``stats_output.json`` (``NumericClaimCheck``)
and every citation to a PMID/DOI already in the project database, re-verified
via ``src.apis.CrossrefClient`` / ``PubMedClient`` (``CitationCheck``). The
state machine only allows ``STAGE6_AUDIT -> COMPLETED`` when ``AuditReport.passed``.
"""
