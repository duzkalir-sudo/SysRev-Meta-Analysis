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
algorithms, verification workflow and the current implementation status.

**Implemented:** Stages 1–4 — protocol contracts, search clients, deduplication, PRISMA
tracking, two automated (LLM) screening passes with human adjudication, open-access
full-text retrieval and the missing-PDF manifest, PDF parsing, source-grounded
extraction into a human-reviewed Master Extraction Table, and risk-of-bias drafts that
only a human can approve. **Not implemented yet:** the statistics engine, plots, GRADE,
manuscript writer, Auditor and a graphical UI.

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

Requires Python ≥ 3.11 (Linux, macOS or Windows; PDF handling uses pure-Python
pdfplumber and pypdf).

## Stages 2–4 in practice (Session 2)

LLM-backed steps run from Python; no provider is built in or chosen by default. Register
one explicitly (any object with `provider_name`, `model_id` and
`complete(LLMRequest) -> LLMResponse`), otherwise `ProviderNotConfiguredError` is raised:

```python
from src.llm import LLMConfig, default_registry
default_registry().register("my-provider", make_my_provider)   # your adapter
cfg1 = LLMConfig(provider="my-provider", model_id="model-x", label="llm-pass-1")
cfg2 = LLMConfig(provider="my-provider", model_id="model-y", label="llm-pass-2")

from src.core.enums import ReviewerRole
from src.screening.llm_screening import DualScreeningRunner, LLMScreener, title_abstract_source
reg = default_registry()
runner = DualScreeningRunner(repo, project_id, approved_protocol, [
    LLMScreener(reg.build(cfg1), cfg1, ReviewerRole.REVIEWER_1),
    LLMScreener(reg.build(cfg2), cfg2, ReviewerRole.REVIEWER_2),
])
runner.run([title_abstract_source(r) for r in repo.load_records(project_id, unique_only=True)])
```

Re-running resumes where it stopped. Full-text retrieval works the same way
(`src.fulltext.FullTextRetriever(...).run()`, async). Every human step — adjudication,
PDF upload / identity confirmation, "unobtainable", accepting/rejecting/correcting
extracted values, building approved arms, linking reports to studies, approving risk of
bias — is an authenticated API call, e.g.:

```bash
curl -X PUT -H "Authorization: Bearer $TOKEN" -H "X-Filename: smith2019.pdf" \
     --data-binary @smith2019.pdf http://127.0.0.1:8000/projects/$PID/fulltext/$RECORD/pdf
curl "http://127.0.0.1:8000/projects/$PID/fulltext/manifest?format=csv"
curl "http://127.0.0.1:8000/projects/$PID/extraction/export?format=csv"
```

Two automated screening passes are **not** two human reviewers; the generated methods
text says so (`src.screening.reporting`). LLM output is always a proposal that is
checked against the source text or parsed PDF page before a human can accept it.
