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

**Implemented:** Stages 1–5 — protocol contracts, search clients, deduplication, PRISMA
tracking, two automated (LLM) screening passes with human adjudication, open-access
full-text retrieval and the missing-PDF manifest, PDF parsing, source-grounded
extraction into a human-reviewed Master Extraction Table, risk-of-bias drafts that
only a human can approve, and (Session 3) a deterministic meta-analysis engine validated
against R `metafor` with forest plots and a hashed `stats_output.json`. Session 4 adds
Anthropic and OpenAI API adapters with per-task model/effort selection, a usage and cost
audit trail, and a first local graphical interface. **Not implemented yet:** GRADE,
manuscript writer and Auditor.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,stats]"      # "stats" adds matplotlib for forest plots
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

LLM-backed steps run from Python or the local UI (Session 4, below); no provider or model
is chosen by default. The Anthropic/OpenAI adapters are registered with
`src.llm.tasks.register_api_providers(...)`; any other adapter can be
registered explicitly (any object with `provider_name`, `model_id` and
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

## Stage 5 in practice (Session 3)

A reviewer approves an analysis plan (authenticated; one entry per analysis):

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  http://127.0.0.1:8000/projects/$PID/stats/plan -d '{"analyses": [
    {"analysis_id": "control_5y", "outcome_id": "local_control", "effect_measure": "KM_PROBABILITY",
     "timepoints": [{"kind": "fixed", "value": 5, "unit": "years"}], "reported_ci_scale": "log_log"},
    {"analysis_id": "necrosis", "outcome_id": "radionecrosis", "effect_measure": "PROPORTION",
     "timepoints": [{"kind": "last_follow_up"}]}]}'
```

Then, in Stage 5, run the engine from Python and advance to Stage 6:

```python
from pathlib import Path
from src.stats_engine.stage5 import run_stage5
artifacts = run_stage5(repo, project_id, output_dir=Path("output/stats"))
# output/stats/stats_output.json, report.md, tables/*.md|csv, plots/*.svg, manifest.json
```

Every analysis ends with an explicit status (`pooled`, `single_study`, `descriptive_only`,
`not_estimable`, `unsupported`, `no_data`) and lists excluded studies with reasons. Methods,
defaults and limitations: ARCHITECTURE.md §9.5. Numerical validation against R metafor:
[validation/VALIDATION_REPORT.md](validation/VALIDATION_REPORT.md).

## Session 4: local UI and real AI providers

### Windows 11 (PowerShell), Python 3.11+

```powershell
cd C:\path\to\SysRev-Meta-Analysis
py -m venv .venv                   # any installed Python 3.11, 3.12 or 3.13
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[stats,ui]"
Copy-Item .env.example .env        # optional: edit e-mail addresses for the literature APIs
python -m src.api_server.reviewer_cli enroll "Dr Jane Doe"   # shows the reviewer token once
python -m src.ui                    # opens http://127.0.0.1:8501
```

If `Activate.ps1` is blocked, run once
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or call
`.\.venv\Scripts\python.exe` directly. To run the tests: `python -m pip install -e
".[dev,stats,ui]"` then `python -m pytest`. (The UI has been tested headlessly on Linux; it
has not yet been run on a Windows machine.)

Linux/macOS: the same commands with `python3 -m venv .venv && source .venv/bin/activate`.

### Using it

1. **Projects** — create or open a project.
2. **AI models & keys** — enter the Anthropic and/or OpenAI API key (kept in memory for
   the browser session; optionally saved in the OS credential store; or set
   `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`). Choose, *per task* (screening pass 1 and 2,
   extraction, advisory verification, risk-of-bias drafts), the provider, the exact API
   model ID and the effort, which is sent unchanged ("provider default" sends none).
   Optional warning and stop thresholds apply locally.
3. **Pipeline status** — current phase and blocking reasons; transitions and gate
   decisions need the reviewer token (sidebar).
4. **Run AI steps** — shows readiness per task; each run makes paid API calls and only
   creates proposals plus an audit-trail row per call.
5. **Screening / Full text & extraction / Risk of bias** — human review queues; every
   decision is an authenticated API call.
6. **Statistics** — deterministic Stage 5 engine and forest plots (no LLM).
7. **Usage & costs** — tokens and list-price cost estimates per call.

**Billing.** API keys bill the provider *API* account (Anthropic Console / OpenAI
Platform). Claude subscription or Claude Code cloud development credits and ChatGPT
subscriptions cannot pay for these calls. Costs shown are estimates from list prices in
`src/llm/model_catalogue.json`; the provider invoice is authoritative. The local stop
threshold is not a hard cap — set a spending limit in the provider console as well.

**Models.** Only models in the capability catalogue can be selected. The Anthropic entries
come from Anthropic's API reference (cached 2026-10-06). The OpenAI entries
(`gpt-5.6-sol`, `gpt-6.1-sol`, `gpt-6-luna`, `gpt-6-astra`) were checked against the
official OpenAI model pages on 2026-10-30, including Standard prices and the >272K-input
long-context tier. The `gpt-5.6` alias, which routes to `gpt-5.6-sol`, is deliberately not
offered, so the exact model is recorded. GPT-5.6 Sol's listed price is promotional (at
least through 2026-11-21). Prices change: check the provider documentation and correct
entries in `<SYSREV_DATA_DIR>/model_catalogue.json` (same format).

**Validation status.** Both adapters are tested against mocked HTTP responses only. No
live API call has been made; run a small screening batch with your own key and inspect
**Usage & costs** before relying on them.
