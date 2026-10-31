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
audit trail, and a first local graphical interface. Session 5 adds the Stage 1–2 front end
of a review in the same interface: a protocol editor with NLM MeSH verification and
authenticated approval, database selection, reproducible PubMed / Europe PMC / OpenAlex
searches, RIS / BibTeX / CSV imports for subscription databases and registers, deterministic
deduplication with human adjudication of ambiguous pairs, a PRISMA-S search log and
derived PRISMA counts. Session 6 adds evidence-grounded manuscript drafting (deterministic,
with optional model-written prose that may not type numbers or citations), a reference
registry with live PubMed/Crossref verification, an independent completeness-aware
Auditor, manuscript versioning, human acknowledgement of unresolved findings and a human
sign-off before a project can be completed. Session 7 adds deterministic PRISMA 2020 flow
diagrams (SVG/PNG/JSON source), the PRISMA 2020 and PRISMA 2020 for Abstracts checklists
with authenticated item-level decisions, human-supervised GRADE certainty assessments with
Summary of Findings and single-group evidence-summary tables, structured synthesis without
meta-analysis with SWiM reporting where a reviewer judges it applicable, and a new
`reporting` scope of the Auditor. **Not implemented yet:** the "Refresh Models" catalogue
update (issue #9).

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

## Session 5: protocol, database selection, searches and imports

### Windows 11 (PowerShell): install and start

```powershell
cd C:\path\to\SysRev-Meta-Analysis
git pull
py -m venv .venv                     # skip if .venv already exists (Python 3.11-3.13)
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[stats,ui]"
Copy-Item .env.example .env           # only if you have no .env yet
notepad .env                          # set SYSREV_NCBI_EMAIL to YOUR address (NCBI policy)
python -m src.api_server.reviewer_cli enroll "Dr Jane Doe"   # once; shows the token once
python -m src.ui                      # opens http://127.0.0.1:8501
```

`SYSREV_NCBI_EMAIL` is required for PubMed requests and MeSH lookups; replace the
placeholder from `.env.example` with your own address. `SYSREV_NCBI_API_KEY` (optional)
raises the NCBI limit from 3 to 10 requests per second. Linux/macOS: the same commands with
`python3 -m venv .venv && source .venv/bin/activate`. The new pages were tested headlessly
(Streamlit AppTest) on Linux; they have **not** been run on a Windows machine.

### Workflow in the interface

1. **Projects** — create or open a project; paste the reviewer token in the sidebar.
2. **Protocol** (Stage 1) — enter the review question and design, PICO, outcomes (data
   type, comparison, effect measure, timepoints such as `5 years; 24-36 months; last
   follow-up`), eligibility criteria and study designs, MeSH terms, the databases to search
   and one database-specific search string per database (with optional date limits and
   filters). *Save draft* validates the whole protocol on the server.
   * *Verify MeSH terms* checks each descriptor in NLM MeSH through the PubMed client.
   * *Preview count* (PubMed, Europe PMC, OpenAlex) runs a count-only search, shows
     PubMed's query translation and errors/warnings, and marks the strategy
     syntax-validated only if PubMed reported no query error.
   * Subscription databases and registers cannot be checked by the system: a reviewer
     *attests* that the strategy ran in the vendor interface.
   * *Approve protocol* is authenticated and refused while anything is unverified.
   Any later change starts a new draft version, resets verification of changed items and
   voids the approval; every save, check and approval is kept in an append-only history.
3. **Pipeline status** — advance to Gate 1 and approve the gate (authenticated). The Gate 1
   guard now also requires that the current protocol version was approved by an enrolled
   reviewer. After Gate 1 the protocol is locked.
4. **Searches & import** (Stage 2) — for each approved strategy:
   * **PubMed / Europe PMC / OpenAlex**: *Execute approved search*. The exact query, the
     date parameters, the time, the reported count, the number retrieved and every raw API
     response (SHA-256) are recorded. PubMed searches above 10,000 records are split by
     publication date and the union is checked against PubMed's total; if they differ,
     nothing is saved. Fewer records than reported is shown as a count mismatch that a
     reviewer must acknowledge.
   * **Embase, Scopus, Web of Science, Cochrane CENTRAL, ClinicalTrials.gov**: run the
     strategy in the vendor interface, export all records (RIS recommended; BibTeX or CSV
     also accepted), *Check file*, then *Import* with the date of the search and the
     **total the interface reported for the whole search**. The file is preserved with its
     SHA-256, each raw entry is kept, a file cannot be imported twice, and a file with an
     untitled entry is refused as a whole. Large exports may be imported as several files;
     the combined number of imported records is reconciled with that total.
     * If the interface showed no total, tick that explicitly: the total is then recorded
       as *unknown* (never replaced by the number imported), and screening stays blocked
       until a reviewer acknowledges it with a reason. A shortfall (e.g. 100 of 150) or
       conflicting totals need the same acknowledgement; an acknowledgement is bound to the
       counts and becomes invalid when another file is imported.
     * Tick that the protocol's date limits, language limits and filters were applied in
       the vendor interface (optionally saying how). Otherwise the reports say the
       restrictions were **not confirmed** as applied; imported records with publication
       years outside the date limits are flagged.
     * A search that ran but found nothing is recorded as a *zero-result search* (with an
       attestation), so it is reported as searched with 0 records.
   * A database that could not be searched is recorded as **not searched** with a reason.
     Every declaration stays in the log; the *current* status is derived from the sequence
     of events: a later successful execution or import supersedes the declaration, a later
     declaration replaces an earlier one, and a failed attempt never counts as a search.
     The strategy table, database lists, Markdown and CSV exports all use this status.
   * *Deduplicate and prepare for screening* runs the existing deterministic deduplicator
     (DOI/PMID/PMCID/OpenAlex ID, then near-identical titles with year and first-author
     checks). In this workflow a shared identifier on clearly different titles is **not**
     merged but listed for a human decision, like other conflicting pairs. Re-running the
     preparation (e.g. after another import) keeps and re-applies earlier human decisions.
   * PRISMA counts are derived from the stored records and decisions (*Recompute PRISMA*
     after screening decisions change); ClinicalTrials.gov records are counted in the
     PRISMA "registers" box. The PRISMA-S search summary can be downloaded as Markdown.
5. **Run AI steps / Screening** — the unique records are the screening pool. AI screening
   refuses to start while a strategy has no search, import or "not searched" entry, a count
   discrepancy is unacknowledged, or records were added after the last deduplication.
   Searches and imports are refused once screening has started.

### What is and is not validated

* All literature-API behaviour (E-utilities, Europe PMC, OpenAlex), MeSH lookups and the
  date partitioning were tested with **mocked HTTP responses only**. No live database call
  was made while developing Session 5.
* The import parsers were tested on synthetic files that imitate common RIS, BibTeX and
  CSV exports. They were **not** tested on real Embase, Scopus, Web of Science, Cochrane
  Library or ClinicalTrials.gov exports; vendor formats change. Check the column mapping
  shown by *Check file* before importing a CSV.
* No subscription-database API (Elsevier, Clarivate, Wiley/Cochrane) is implemented. A
  direct ClinicalTrials.gov API search is not implemented either (CSV/RIS import only).
* RIS has no standard PMID tag, so RIS imports carry DOIs but no PMIDs; PMIDs are read
  only from explicitly labelled BibTeX (`pmid`) or CSV (`PMID`, `PubMed ID`) fields.
  Accession numbers (e.g. Embase `AN`) are kept as source identifiers and never
  reinterpreted.
* The deduplication title floor for shared identifiers (token-sort similarity 60) is an
  engineering heuristic, not validated against a labelled corpus.

## Session 6: evidence-grounded manuscript and independent audit

Stage 6 starts once Stage 5 has recorded a verified `stats_output.json`. In the UI, open
**Manuscript**:

1. **Create draft from the evidence package** — a deterministic draft (no AI provider
   needed). Every number and citation is a marker (`{{stats.nec.pooled_estimate}}`,
   `[@study-…]`) resolved from approved records; content only an author can supply is left
   as `[[AUTHOR INPUT: …]]`, which the audit reports as missing.
2. **Edit** sections (each save is a new version; earlier versions are kept) or, after
   authorising paid calls, let the configured `manuscript_drafting` model rewrite prose.
   A model reply that types a digit, cites an unknown reference, names authors without a
   registry citation or includes a DOI/URL is discarded.
3. **References** — import background literature by PMID/DOI (live lookup), run live
   verification, and record supporting excerpts. A failed lookup is shown as
   *unavailable*, never as verified.
4. **Audit** — the deterministic Auditor checks the stored version against the current
   evidence (see ARCHITECTURE.md §9.8). Blocking findings must be fixed; findings that
   require human review must be acknowledged by an authenticated reviewer with a reason.
5. **Sign-off** — in the audit phase, an authenticated human signs off the exact version
   and audit; only then does *Pipeline status → completed* succeed. Any later change to the
   text or to the underlying evidence invalidates the audit and the sign-off.

Exports: Markdown, editable Word (`pip install -e ".[manuscript]"` for `python-docx`), a
provenance note and the audit report. Forest plots are the Stage 5 SVG files, referenced by
the figure legends and submitted separately.

### What is and is not validated (Session 6)

* Tested with synthetic projects, mocked model replies (respx-mocked Anthropic HTTP) and
  in-memory PubMed/Crossref fakes. **No live model call and no live PubMed/Crossref call
  was made** during development; real-world manuscript quality is unvalidated.
* The Auditor's numerical checks of inserted values are exact (with stated rounding
  tolerances). Its checks of free prose (typed numbers, labels, claims, methods
  descriptions) are rule-based and incomplete; unresolved statements are routed to human
  review, never marked verified. Every author-written sentence (anything that is not text
  the template derived from the evidence package) requires an authenticated human's
  assessment before sign-off; several can be acknowledged together with one stated reason.
* Included reports without DOI/PMID cannot be looked up; their identity is attested by a
  reviewer against the identity-confirmed PDF (*References → attestation*). Whether a background reference supports a sentence is
  checked only lexically against human-entered excerpts.

## Session 7: PRISMA 2020, GRADE and synthesis without meta-analysis

Three new pages in the UI (each decision needs the reviewer token, exactly like approvals):

1. **PRISMA flow & checklists** — the PRISMA 2020 flow diagram for *new reviews with
   searches of databases and registers*, generated from the stored PRISMA snapshot
   (download SVG, PNG or the JSON source it is regenerated from). Inconsistent or non-final
   snapshots are drawn with a visible red warning. Variants needing data the project does
   not record (other methods; updated reviews) are refused with the list of unsupported
   fields. Below it, the PRISMA 2020 (42 entries) and PRISMA 2020 for Abstracts (12)
   checklists: the system only *suggests* candidate locations; you record for each item
   *reported* (with section and precise location), *partially reported*, *missing* or
   *not applicable* (with a justification). Exports: CSV, DOCX, XLSX.
2. **GRADE & Summary of Findings** — per analysis: decision support (k, participants,
   estimate and interval, I², risk-of-bias distribution; never pre-filled judgements), the
   approaches that apply to the evidence, and a form for the five domains with rationales,
   rating-up factors (non-randomised evidence only), final certainty, and optional assumed
   comparator risks *with their source* for absolute effects. Single-group outcomes can be
   rated only with the GRADE approach for overall prognosis and a documented rationale;
   otherwise they stay *not assessed*. Preview and download the Summary of Findings
   (Markdown/DOCX) with a source-key appendix.
3. **Synthesis without MA & SWiM** — groups of studies synthesised without meta-analysis:
   grouping and rationale, why not pooled, metric, method (structured tabulation or vote
   counting on the *direction* of effect), source-linked study findings (approved arms /
   accepted extraction items), directions with their basis, narrative and limitations. Once
   a synthesis is approved, record whether SWiM applies; if it does, complete its checklist.

The Stage 6 Auditor (version 1.3.0) checks all of this before sign-off: diagram counts
against the persisted PRISMA facts, current human decisions for every checklist item,
current GRADE records (recomputed absolute effects; certainty words typed in the text),
current syntheses and the SWiM decision. Any relevant change to the manuscript sections or
to the evidence makes the affected decisions stale; they are re-confirmed, never repaired.
Each decision is tied to the screen you saw: if the manuscript, the evidence or the record
changed after the page was loaded, the decision is refused and nothing is saved — reload,
review and decide again. The pages keep the screen you loaded until you press "Reload the
current state", so a submission always carries what you saw; a warning appears when the stored
state has changed meanwhile. A synthesis is saved in two steps (preview the linked evidence,
then save exactly what you previewed).

### What is and is not validated (Session 7)

* Checklist and SWiM item wording was reproduced from the primary publications (BMJ
  2021;372:n71 and BMJ 2020;368:l6890, CC BY 4.0) retrieved from PubMed Central. The PRISMA
  and Cochrane websites could not be reached from the build environment.
* Tested on synthetic projects only. No real review, real literature or live API was used.
* A complete checklist record means every item has a current human decision; the system
  never claims that a review is PRISMA-compliant, and a typed compliance claim is routed to
  human review (or blocked while the checklist is incomplete).
* GRADE judgements are never computed from statistics. The tool enforces the structural
  rules listed in ARCHITECTURE.md §9.9; it cannot judge whether a rationale is correct.

