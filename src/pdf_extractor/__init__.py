"""Source-grounded PDF extraction (Stage 4) — implementation pending.

Contract: every extracted number must be returned as an
``src.core.schemas.ExtractedValue`` whose ``reported_text`` occurs verbatim in a
``SourceEvidence.verbatim_quote`` taken from a specific PDF page. Values that
cannot be grounded are left empty and surfaced to the reviewer at Gate 3.
"""
