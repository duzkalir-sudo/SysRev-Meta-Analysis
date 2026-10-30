"""SysRev Meta-Analysis Agent.

A PRISMA 2020-compliant, human-in-the-loop systematic review and meta-analysis
pipeline. Two non-negotiable guardrails shape every module:

1. Zero fabrication: literature and study data enter the system only through
   deterministic API clients (``src.apis``) or user-uploaded PDFs, and every
   extracted number carries a verbatim ``SourceEvidence`` record.
2. Zero LLM arithmetic: all statistics are computed by ``src.stats_engine``.
"""

__version__ = "0.1.0"
