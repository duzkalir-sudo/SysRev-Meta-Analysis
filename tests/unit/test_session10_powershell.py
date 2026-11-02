"""Session 10 review of PR #15, P2: Run-Session10.ps1 must report failure in its exit code.

On Windows the unchanged script recorded pytest exit code 1 and preflight exit code 1, yet
``powershell.exe -File`` exited 0. These are process-level controls: the real
``powershell.exe`` runs the real script, once per prepared ``summary.json`` (``-VerdictOnly``)
and once through the main path with a broken environment. Windows only (Windows PowerShell
5.1 is part of every Windows installation); skipped elsewhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "validation" / "session10" / "Run-Session10.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(
    os.name != "nt" or POWERSHELL is None, reason="Windows PowerShell process-level controls"
)


def _run(*args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


GOOD_RUN = {
    "run": "default",
    "exit_code": 0,
    "tests": 996,
    "failures": 0,
    "errors": 0,
    "skipped": 5,
    "failed_tests": [],
    "report_error": None,
}


def _summary(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "environment": {},
        "test_runs": [dict(GOOD_RUN)],
        "preflight_exit_code": 0,
        "script_error": None,
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (_summary(), 0),  # the successful control
        # the observed Windows result: pytest 1 failure, preflight exit 1
        (
            _summary(test_runs=[GOOD_RUN | {"exit_code": 1, "failures": 1}], preflight_exit_code=1),
            1,
        ),
        (_summary(test_runs=[GOOD_RUN | {"exit_code": 1, "failures": 1}]), 1),
        (_summary(preflight_exit_code=1), 1),
        (_summary(preflight_exit_code=None), 1),  # preflight never ran
        (_summary(test_runs=[]), 1),  # no test run
        (_summary(test_runs=[GOOD_RUN | {"report_error": "JUnit report missing or malformed"}]), 1),
        (_summary(test_runs=[GOOD_RUN | {"tests": 0}]), 1),  # an empty report is not a pass
        (_summary(test_runs=[GOOD_RUN | {"exit_code": None}]), 1),
        (_summary(test_runs=[GOOD_RUN, GOOD_RUN | {"run": "utf8", "errors": 2}]), 1),
        (_summary(script_error="installation failed (exit 1): see pip.txt"), 2),
    ],
)
def test_verdict_exit_codes(tmp_path: Path, summary: dict[str, Any], expected: int) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    proc = _run("-VerdictOnly", str(path))
    assert proc.returncode == expected, proc.stdout + proc.stderr


@pytest.mark.parametrize("content", [None, "{not json"])
def test_missing_or_malformed_summary_is_an_error(tmp_path: Path, content: str | None) -> None:
    path = tmp_path / "summary.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert _run("-VerdictOnly", str(path)).returncode == 2


def test_main_path_fails_when_the_environment_is_unusable(tmp_path: Path) -> None:
    """-SkipInstall with no virtual environment: nothing can run, so the script must exit
    non-zero (it previously turned such failures into recorded zeros and exit 0)."""
    work = tmp_path / "no venv here"
    work.mkdir()
    proc = _run("-Work", str(work), "-SkipInstall")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "VERDICT: failed" in proc.stdout
