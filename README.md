# SysRev-Meta-Analysis

A PRISMA 2020-compliant, **zero-fabrication**, human-in-the-loop agent for systematic
reviews and meta-analyses.

* Literature enters the system only through deterministic API clients (PubMed
  E-utilities, Europe PMC, OpenAlex, Crossref, Unpaywall). An LLM never writes a citation.
* Every extracted number must be a literal copy of a quoted passage on a specific PDF
  page (`SourceEvidence`); computed numbers are rejected at schema and database level.
* All statistics are computed by deterministic code, never by an LLM.
* A 6-stage state machine pauses at four HITL gates that only an authenticated, enrolled
  human reviewer can release; transition guards fail closed on missing data.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the state machine, data contracts,
algorithms, verification workflow and the current implementation status. Stages 4–6
(PDF extraction, the statistics engine, the Auditor) are defined as contracts but are
not implemented yet.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # set SYSREV_NCBI_EMAIL, SYSREV_UNPAYWALL_EMAIL, ...
pytest
python -m src.api_server.reviewer_cli enroll "Dr Jane Doe"   # human only; token shown once
uvicorn src.api_server.main:app --reload
```

Gate decisions are submitted with `Authorization: Bearer <token>`. Keep the token out of
`.env`, the repository and any location an automated agent can read (see
ARCHITECTURE.md §6 for the threat model).

Requires Python ≥ 3.11.
