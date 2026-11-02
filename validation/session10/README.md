# Session 10 — repeating the real-world pilot on Windows

The status of every step (prepared / executed / blocked) and the findings are in
[`SESSION10_VALIDATION_TR.md`](../../SESSION10_VALIDATION_TR.md). Nothing here replaces a
human decision: the scripts only install, start, check and collect read-only evidence.

| Tool | Purpose | Writes |
|---|---|---|
| `Run-Session10.ps1` | install in a path with spaces/Turkish characters, full test suite with default Windows behaviour (no `PYTHONUTF8`), preflight | `session10_results\windows_<UTC>\` |
| `preflight.py` | database creation + owner-only ACL, API and UI startup, restart, UTF-8 export, (`--network`) service reachability without credentials | `preflight.json/.md` |
| `collect_pilot_evidence.py` | saves every read-only API response of the pilot project with its SHA-256 (no token, GET only) | `evidence\*.json/.csv/.md/.docx`, `index.json` |
| `check_pilot_stats.py` + `pilot_reference.R` | recomputes every reported model of the pilot's `stats_output.json` with R metafor and compares estimates, CIs, τ², k, study ids, participants/events, weights, Q/I² | `pilot_stats_check.json/.md` |

`session10_results/` is in `.gitignore`: it contains bibliographic data, verbatim quotations
and file names from the pilot. Never commit it, `.env`, PDFs or reviewer tokens.

## 1. Installation, tests and startup (PowerShell, about 20 minutes)

```powershell
cd "C:\path\to\SysRev-Meta-Analysis"
git fetch origin
git checkout claude/practical-darwin-abm38r
git pull
powershell -ExecutionPolicy Bypass -File validation\session10\Run-Session10.ps1 -Network -CompareUtf8
```

* `-Network` checks reachability of PubMed, Europe PMC, OpenAlex, Crossref, Unpaywall (needs
  `SYSREV_UNPAYWALL_EMAIL` in the environment) and the Anthropic/OpenAI model-list endpoints
  **without keys** (HTTP 401 means reachable; nothing is billed).
* `-CompareUtf8` adds a second test run with `PYTHONUTF8=1`. Only the default run counts as
  Windows validation; the second one shows whether a failure is an encoding issue.
* Use `-PythonVersion 3.11` or `3.13` to test another interpreter (`py -0` lists them).

The script exits **0 only if** every test run has a readable JUnit report with at least one
test and no failure or error, and the preflight passed; **1** if validation failed; **2** if the
script could not complete (e.g. installation). The verdict and its reasons are printed and
written to `summary.json` (`verdict`, `verdict_problems`). `-VerdictOnly <summary.json>`
re-evaluates an existing summary. The run includes the Windows-only process-level controls in
`tests/unit/test_session10_powershell.py`.

Expected on this branch: no failures in the default run, the R tests skipped if R is not
installed, six deselected `live` tests. A preflight network check that fails once and succeeds
on retry is reported as `warn` (it does not fail the run). Send back the results folder.

## 2. The pilot through the normal interface

Start the application from the same virtual environment (it is in the `-Work` folder):

```powershell
& "$env:USERPROFILE\SysRev Doğrulama ş10 İ\venv\Scripts\Activate.ps1"
notepad .env      # SYSREV_NCBI_EMAIL / SYSREV_UNPAYWALL_EMAIL / SYSREV_CROSSREF_MAILTO: your address
python -m src.api_server.reviewer_cli enroll "Dr <your name>"   # once; the token is shown once
python -m uvicorn src.api_server.main:app       # window 1 (API, 127.0.0.1:8000)
python -m src.ui                                # window 2 (browser UI, 127.0.0.1:8501)
```

The pilot uses the user's own Gamma Knife vestibular schwannoma protocol (its contents and the
required human decisions are established before the pilot starts). Then follow the plan in
`SESSION10_VALIDATION_TR.md` §5: protocol → Gate 1 → PubMed
search → deduplication → AI screening (small batch) → your adjudication → full texts (open
access + your PDFs, identity confirmation) → AI extraction proposals → your acceptance and
corrections (including one awkward page) → arms → risk of bias → Gate 3 → analysis plan →
Stage 5 → GRADE, checklists → manuscript → audit → acknowledgements → sign-off. Stop and
restart both windows at least once in the middle (restart/resume).

## 3. Evidence and independent statistics check

```powershell
python -m validation.session10.collect_pilot_evidence <project_id>
python -m validation.session10.check_pilot_stats "data\output\<project_id>\stats"
```

The checker's exit status is its verdict: **0 `validated`** (bound to the manifest, every
reported model reproduced), 1 `failed` (a mismatch, or `stats_output_sha256` /
`files["stats_output.json"]` missing or not equal to the file's hash), 2 `error` (R could not
compute a reference), 3 `inconclusive` (something was not compared, e.g. an unsupported HR
outcome). The statistics folder is the one the UI writes (`<SYSREV_DATA_DIR>\output\<project_id>\stats`,
by default `data\output\...`). The second command needs R with `metafor`, `lme4` and `jsonlite` (`Rscript` on `PATH`, or
`--rscript "C:\Program Files\R\R-4.x.x\bin\Rscript.exe"`). Usage and costs: page
**Usage & costs** (estimates from list prices) and the provider consoles (measured).

R refuses a session temporary directory whose path contains a space ("'R_TempDir' contains
space"; observed on Linux in Session 10, not yet checked on Windows). On Linux/macOS the
checker then gives R `/tmp`; on Windows, if the message appears, run it from a window where
`TMP` and `TEMP` point to a folder without spaces (e.g. `$env:TMP = "C:\Temp"; $env:TEMP = "C:\Temp"`).
