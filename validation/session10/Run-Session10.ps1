<#
.SYNOPSIS
  Session 10: Windows installation, test and startup validation of SysRev-Meta-Analysis.

.DESCRIPTION
  Run from a PowerShell window in the repository root:

      powershell -ExecutionPolicy Bypass -File validation\session10\Run-Session10.ps1

  The script
    1. records the exact commit, the working-tree state and the Windows/Python versions;
    2. creates a NEW virtual environment in a folder whose path contains spaces and Turkish
       characters (default: %USERPROFILE%\SysRev Dogrulama <s-cedilla>10 <I-dot>\venv) and
       installs the project with the dev, stats, ui and manuscript extras;
    3. runs the full test suite with the DEFAULT Windows behaviour: PYTHONUTF8 is removed,
       and TEMP/TMP point to a folder with spaces and Turkish characters, so every test that
       writes files does so in such a path (JUnit XML: pytest_default.xml);
    4. optionally (-CompareUtf8) repeats the run with PYTHONUTF8=1, only for comparison;
    5. runs validation.session10.preflight (database creation and its Windows ACL, API and
       UI startup, restart, UTF-8 export, and -with -Network- reachability of the services);
    6. writes summary.json with an overall verdict.

  Exit status (review of PR #15, P2): 0 only if every recorded test run has a readable JUnit
  report with at least one test, no failure or error and exit code 0, AND the preflight
  exited 0; 1 if validation failed; 2 if the script itself could not complete (for example
  the installation failed). The verdict and its reasons are in summary.json.

  -VerdictOnly <summary.json> only evaluates an existing summary and exits with its status
  (used by the process-level controls in tests/unit/test_session10_powershell.py).

  It never enrolls a reviewer, never reads or stores API keys or reviewer tokens, and never
  touches your own data\ directory or database. Send back the results folder (it contains
  no secrets; check it before sharing).

  This file is ASCII-only on purpose: Windows PowerShell 5.1 reads a script without a BOM in
  the ANSI code page, so the non-ASCII folder names are built from character codes.
#>
[CmdletBinding()]
param(
    [string]$Work = "",
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.12",
    [switch]$CompareUtf8,
    [switch]$Network,
    [switch]$SkipInstall,
    [string]$VerdictOnly = ""
)
# "Continue": in Windows PowerShell 5.1, "Stop" turns any stderr line of a native command
# (pip and git warnings) into a terminating error; exit codes are checked explicitly instead.
$ErrorActionPreference = "Continue"

function Get-Verdict($summary) {
    # Every condition below must hold for a pass; anything missing counts as a failure.
    $problems = @()
    if ($null -eq $summary) { return [ordered]@{ verdict = "failed"; exit_code = 2; problems = @("no summary") } }
    if ($summary.script_error) { $problems += "script error: $($summary.script_error)" }
    $runs = @($summary.test_runs | Where-Object { $null -ne $_ })
    if ($runs.Count -eq 0) { $problems += "no test run was recorded" }
    foreach ($r in $runs) {
        $name = $r.run
        if ($r.report_error) { $problems += "test run '$name': $($r.report_error)" }
        if ($null -eq $r.exit_code -or [int]$r.exit_code -ne 0) { $problems += "test run '$name': pytest exit code $($r.exit_code)" }
        if ($null -eq $r.tests -or [int]$r.tests -le 0) { $problems += "test run '$name': the report lists no tests" }
        if ([int]$r.failures -gt 0 -or [int]$r.errors -gt 0) { $problems += "test run '$name': $($r.failures) failure(s), $($r.errors) error(s)" }
    }
    if ($null -eq $summary.preflight_exit_code) { $problems += "the preflight did not run" }
    elseif ([int]$summary.preflight_exit_code -ne 0) { $problems += "preflight exit code $($summary.preflight_exit_code)" }
    if ($problems.Count -eq 0) { return [ordered]@{ verdict = "passed"; exit_code = 0; problems = @() } }
    $code = 1
    if ($summary.script_error) { $code = 2 }
    return [ordered]@{ verdict = "failed"; exit_code = $code; problems = $problems }
}

if ($VerdictOnly) {
    try {
        $summary = Get-Content -Raw -Encoding UTF8 -LiteralPath $VerdictOnly -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
    } catch {
        Write-Host "verdict: failed (summary unreadable: $($_.Exception.Message))"
        exit 2
    }
    $v = Get-Verdict $summary
    Write-Host "verdict: $($v.verdict)"
    $v.problems | ForEach-Object { Write-Host " - $_" }
    exit $v.exit_code
}

$s = [char]0x015F; $g = [char]0x011F; $u = [char]0x00FC; $I = [char]0x0130; $o = [char]0x00F6
if (-not $Work) { $Work = Join-Path $env:USERPROFILE ("SysRev Do" + $g + "rulama " + $s + "10 " + $I) }
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$Out = Join-Path $Repo ("session10_results\windows_" + $Stamp)
$Tmp = Join-Path $Work ("tmp dizin " + $g + $u + $s + $I + " " + $o)
New-Item -ItemType Directory -Force -Path $Out, $Tmp | Out-Null
Start-Transcript -Path (Join-Path $Out "transcript.txt") | Out-Null

function Write-Step($text) { Write-Host ""; Write-Host "=== $text ===" -ForegroundColor Cyan }

function Invoke-Tests($name, [hashtable]$extraEnv) {
    $saved = @{}
    foreach ($k in @("PYTHONUTF8", "PYTHONIOENCODING", "TEMP", "TMP")) { $saved[$k] = [Environment]::GetEnvironmentVariable($k, "Process") }
    [Environment]::SetEnvironmentVariable("PYTHONUTF8", $null, "Process")
    [Environment]::SetEnvironmentVariable("PYTHONIOENCODING", $null, "Process")
    [Environment]::SetEnvironmentVariable("TEMP", $Tmp, "Process")
    [Environment]::SetEnvironmentVariable("TMP", $Tmp, "Process")
    foreach ($k in $extraEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $extraEnv[$k], "Process") }
    $xml = Join-Path $Out "pytest_$name.xml"
    $log = Join-Path $Out "pytest_$name.txt"
    $result = [ordered]@{ run = $name; exit_code = $null; tests = $null; failures = $null;
        errors = $null; skipped = $null; failed_tests = @(); report_error = $null }
    try {
        $p = Start-Process -FilePath $Py -ArgumentList @("-m", "pytest", "-p", "no:cacheprovider", "-rA",
                "--junitxml", "`"$xml`"") -WorkingDirectory $Repo -NoNewWindow -Wait -PassThru `
                -RedirectStandardOutput $log -RedirectStandardError ($log + ".stderr") -ErrorAction Stop
        $result.exit_code = $p.ExitCode
    } catch {
        $result.report_error = "pytest could not be started: $($_.Exception.Message)"
    } finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], "Process") }
    }
    if (-not $result.report_error) {
        try {
            $r = [xml](Get-Content -Raw -Encoding UTF8 -LiteralPath $xml -ErrorAction Stop)
            $suite = @($r.testsuites.testsuite)[0]
            if ($null -eq $suite -or $null -eq $suite.tests) { throw "no <testsuite> element" }
            $result.tests = [int]$suite.tests; $result.failures = [int]$suite.failures
            $result.errors = [int]$suite.errors; $result.skipped = [int]$suite.skipped
            $result.failed_tests = @($r.SelectNodes("//testcase[failure or error]") | ForEach-Object { "$($_.classname)::$($_.name)" })
        } catch {
            $result.report_error = "JUnit report missing or malformed: $($_.Exception.Message)"
        }
    }
    return $result
}

$envInfo = [ordered]@{ recorded_at_utc = $Stamp }
$results = @()
$preflightExit = $null
$scriptError = $null
Push-Location $Repo
try {
    Write-Step "1. Environment"
    $commit = (git rev-parse HEAD)
    if ($LASTEXITCODE -ne 0 -or -not $commit) { throw "not a git checkout: $Repo" }
    $branch = (git rev-parse --abbrev-ref HEAD)
    $changes = git status --porcelain
    $os = Get-CimInstance Win32_OperatingSystem
    $envInfo = [ordered]@{
        recorded_at_utc = $Stamp
        commit = $commit.Trim()
        branch = "$branch".Trim()
        working_tree_clean = [string]::IsNullOrEmpty(($changes -join ""))
        windows = "$($os.Caption) $($os.Version) build $($os.BuildNumber)"
        culture = (Get-Culture).Name
        ansi_code_page = [System.Text.Encoding]::Default.CodePage
        oem_code_page = [Console]::OutputEncoding.CodePage
        repo_path = $Repo
        work_path = $Work
        temp_path = $Tmp
        python_launcher = "$PythonLauncher -$PythonVersion"
    }
    $envInfo | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Out "environment.json")
    $envInfo | Format-List

    $Venv = Join-Path $Work "venv"
    $Py = Join-Path $Venv "Scripts\python.exe"
    if (-not $SkipInstall) {
        Write-Step "2. Virtual environment and installation in a path with spaces/non-ASCII"
        if (Test-Path $Venv) { throw "$Venv already exists: remove it or use -SkipInstall" }
        & $PythonLauncher "-$PythonVersion" -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
        & $Py -m pip install --upgrade pip 2>&1 | Out-File -Encoding utf8 (Join-Path $Out "pip.txt")
        if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE): see pip.txt" }
        & $Py -m pip install -e "$Repo[dev,stats,ui,manuscript]" 2>&1 |
            Out-File -Encoding utf8 -Append (Join-Path $Out "pip.txt")
        if ($LASTEXITCODE -ne 0) { throw "installation failed (exit $LASTEXITCODE): see pip.txt" }
    }
    if (-not (Test-Path -LiteralPath $Py)) { throw "no Python in the virtual environment: $Py" }
    & $Py -m pip freeze | Out-File -Encoding utf8 (Join-Path $Out "pip_freeze.txt")
    & $Py -c "import sys, locale; print(sys.version); print('utf8_mode', sys.flags.utf8_mode, 'locale', locale.getencoding())"

    Write-Step "3. Test suite with default Windows behaviour (no PYTHONUTF8)"
    $results = @(Invoke-Tests "default" @{})
    $results[0] | Format-List
    if ($CompareUtf8) {
        Write-Step "4. Comparison run with PYTHONUTF8=1 (a workaround, reported separately)"
        $results += Invoke-Tests "utf8" @{ PYTHONUTF8 = "1" }
        $results[1] | Format-List
    }

    Write-Step "5. Preflight (database ACL, API/UI startup, restart, export, network)"
    $pf = @("-m", "validation.session10.preflight", "--out", $Out, "--workdir", $Work)
    if ($Network) { $pf += "--network" }
    & $Py @pf
    $preflightExit = $LASTEXITCODE
} catch {
    $scriptError = $_.Exception.Message
    Write-Host "ERROR: $scriptError" -ForegroundColor Red
} finally {
    Pop-Location
}

Write-Step "6. Summary"
$summary = [ordered]@{ environment = $envInfo; test_runs = $results;
    preflight_exit_code = $preflightExit; script_error = $scriptError }
$v = Get-Verdict ([pscustomobject]$summary)
$summary.verdict = $v.verdict
$summary.verdict_problems = $v.problems
$summary.exit_code = $v.exit_code
$summary | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Out "summary.json")
$summary | ConvertTo-Json -Depth 5 | Write-Host
Stop-Transcript | Out-Null
Write-Host ""
if ($v.verdict -eq "passed") {
    Write-Host "VERDICT: passed" -ForegroundColor Green
} else {
    Write-Host "VERDICT: failed (exit $($v.exit_code))" -ForegroundColor Red
    $v.problems | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
}
Write-Host "Results: $Out"
Write-Host "Check the folder for anything private before sharing it."
exit $v.exit_code
