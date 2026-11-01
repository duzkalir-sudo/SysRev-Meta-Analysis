"""Session 10 preflight: installation and startup checks for a real-world (Windows) pilot.

Usage (from the repository root, inside the project's virtual environment):

    python -m validation.session10.preflight --out session10_results [--network]

Everything runs in a *scratch* directory whose path contains spaces and non-ASCII characters
(default: ``<temp>/SysRev Doğrulama ş10 İ``). The user's own database, data directory, `.env`
and reviewer tokens are never opened or changed; the only fact read about the configured
database location is whether its directory is accessible to other accounts (read-only).

Checks (each recorded as ``pass``/``FAIL``/``skip`` with details):

1. environment: OS, Python, UTF-8 mode, locale/filesystem encodings, package versions, the
   exact git commit and whether the working tree is clean; whether API keys and the NCBI
   e-mail are *set* (true/false only - values are never read into the report);
2. database creation in the scratch path, owner-only protection of its directory (POSIX
   mode 0700, or the Windows ACL), reopening after the engine is closed (restart/resume);
3. API server startup (``uvicorn src.api_server.main:app`` on 127.0.0.1), ``/health``,
   project creation over HTTP, server restart and the project still present;
4. UI startup (``python -m src.ui`` headless) and Streamlit's health endpoint;
5. a UTF-8 text export and reading it back from a non-ASCII path;
6. (``--network``) reachability of the literature APIs and of the providers' model-list
   endpoints *without credentials* (an HTTP 401/403 proves reachability; nothing is billed).

Writes ``preflight.json`` and ``preflight.md``. Exit status 1 if any check failed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as md
import json
import locale
import os
import platform
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_NAME = "SysRev Doğrulama ş10 İ"  # "SysRev Doğrulama ş10 İ"
PACKAGES = (
    "pydantic",
    "SQLAlchemy",
    "fastapi",
    "uvicorn",
    "httpx",
    "numpy",
    "scipy",
    "pdfplumber",
    "pypdf",
    "matplotlib",
    "streamlit",
    "keyring",
    "python-docx",
    "openpyxl",
    "pytest",
)
NETWORK = {
    "pubmed_eutils": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pubmed&retmode=json",
    "europe_pmc": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMID:1&format=json&pageSize=1",
    "openalex": "https://api.openalex.org/works?per_page=1",
    "crossref": "https://api.crossref.org/works?rows=1",
    "unpaywall": "https://api.unpaywall.org/v2/10.1038/nature12373",
    "anthropic_models_no_key": "https://api.anthropic.com/v1/models",
    "openai_models_no_key": "https://api.openai.com/v1/models",
}


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: Any = "") -> None:
        self.checks.append({"check": name, "status": status, "detail": detail})
        print(f"[{status:>4}] {name}: {detail if isinstance(detail, str) else json.dumps(detail)}")

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[dict[str, Any]]:
        info: dict[str, Any] = {}
        try:
            yield info
        except Exception as exc:  # a preflight records every failure and continues
            self.add(name, "FAIL", f"{type(exc).__name__}: {exc}")
        else:
            self.add(name, info.pop("_status", "pass"), info)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable ({type(exc).__name__})"


def environment() -> dict[str, Any]:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    dirty = _git("status", "--porcelain")
    return {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "windows": os.name == "nt",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "utf8_mode": bool(sys.flags.utf8_mode),
        "PYTHONUTF8": os.environ.get("PYTHONUTF8"),
        "locale_encoding": locale.getencoding(),
        "preferred_encoding": locale.getpreferredencoding(False),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "stdout_encoding": getattr(sys.stdout, "encoding", None),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_clean": dirty == "",
        "git_changes": dirty.splitlines()[:20],
        "packages": versions,
        # booleans only: secrets are never written to the report
        "anthropic_api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "ncbi_email_set_in_environment": bool(os.environ.get("SYSREV_NCBI_EMAIL")),
        "dotenv_present": (ROOT / ".env").is_file(),
    }


def directory_protection(path: Path) -> dict[str, Any]:
    """Who can access ``path``: POSIX mode bits, or the Windows ACL entries (read-only)."""
    if os.name != "nt":
        mode = stat.S_IMODE(path.stat().st_mode)
        return {"mode": oct(mode), "owner_only": mode & 0o077 == 0}
    from src.state.db import _windows_tool

    out = subprocess.run(
        [_windows_tool("icacls.exe"), path.name],
        cwd=path.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout
    lines = [x.strip() for x in out.splitlines()]
    entries = [lines[0][len(path.name) :].strip()]
    for x in lines[1:]:
        if not x:
            break
        entries.append(x)
    entries = [e for e in entries if e]
    return {
        "acl_entries": len(entries),
        "inherited_entries": sum("(I)" in e for e in entries),
        # principal names are not reported (they may contain the user's name)
        "owner_only": len(entries) == 1 and "(I)" not in entries[0],
    }


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_http(url: str, timeout: float = 60.0) -> Any:
    import httpx

    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=5.0, trust_env=False)
            if r.status_code == 200:
                return r
        except httpx.HTTPError as exc:
            last = exc
        time.sleep(0.5)
    raise TimeoutError(f"{url} did not answer within {timeout:.0f} s ({last})")


@contextlib.contextmanager
def server(cmd: list[str], env: dict[str, str], cwd: Path, log: Path) -> Iterator[None]:
    with log.open("ab") as fh:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
        try:
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def scratch_env(scratch: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SYSREV_")}
    env.pop("ANTHROPIC_API_KEY", None)  # no provider is needed for startup checks
    env.pop("OPENAI_API_KEY", None)
    env["SYSREV_DATABASE_URL"] = f"sqlite:///{(scratch / 'db' / 'sysrev.db').as_posix()}"
    env["SYSREV_DATA_DIR"] = str(scratch / "data")
    env["PYTHONPATH"] = str(ROOT)
    return env


def run(args: argparse.Namespace) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # never fail on a narrow console code page
        sys.stdout.reconfigure(errors="backslashreplace")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    env_info = environment()
    rep.add(
        "environment",
        "pass",
        {
            k: env_info[k]
            for k in ("os", "python", "utf8_mode", "locale_encoding", "git_commit", "git_clean")
        },
    )
    base = args.workdir or Path(tempfile.gettempdir())
    scratch = base / SCRATCH_NAME / datetime.now(UTC).strftime("run %Y%m%dT%H%M%SZ")
    scratch.mkdir(parents=True)
    rep.add("scratch directory", "pass", str(scratch))
    env = scratch_env(scratch)
    url = env["SYSREV_DATABASE_URL"]

    with rep.step("database: create, protect, reopen") as info:
        from sqlalchemy import select

        from src.state.db import (
            ReviewerORM,
            init_db,
            make_engine,
            make_session_factory,
            prepare_sqlite_path,
        )
        from src.state.repository import Repository

        prepare_sqlite_path(url)
        engine = make_engine(url)
        init_db(engine)
        with make_session_factory(engine)() as s:
            Repository(s).create_project("preflight", "Preflight project (synthetic)")
            s.commit()
        engine.dispose()
        engine = make_engine(url)  # restart: a new engine on the same file
        init_db(engine)
        with make_session_factory(engine)() as s:
            phase = Repository(s).get_project("preflight").phase
            reviewers = len(s.scalars(select(ReviewerORM)).all())
        engine.dispose()
        prot = directory_protection(scratch / "db")
        info.update(phase_after_reopen=phase, reviewers=reviewers, protection=prot)
        if not prot["owner_only"]:
            info["_status"] = "FAIL"

    with rep.step("configured database directory (read-only)") as info:
        from src.core.config import get_settings

        cfg = get_settings().database_url
        if cfg.startswith("sqlite:///") and ":memory:" not in cfg:
            d = Path(cfg.removeprefix("sqlite:///")).parent
            if not d.is_absolute():
                d = (ROOT / d).resolve()
            info["exists"] = d.is_dir()
            if d.is_dir():
                info["protection"] = directory_protection(d)
                if not info["protection"]["owner_only"]:
                    info["_status"] = "FAIL"
                    info["note"] = (
                        "existing directories are not changed automatically; see "
                        "SESSION10_VALIDATION_TR.md for how to restrict it"
                    )
        else:
            info.update(_status="skip", reason="not a file-based SQLite database")

    with rep.step("API server: startup, /health, create project, restart") as info:
        import httpx

        port = free_port()
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api_server.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        base_url = f"http://127.0.0.1:{port}"
        with server(cmd, env, scratch, out / "preflight_api.log"):
            info["health"] = wait_http(f"{base_url}/health").json()
            r = httpx.post(
                f"{base_url}/projects",
                json={"title": "Ön kontrol projesi ğüşİ"},
                timeout=10,
                trust_env=False,
            )
            r.raise_for_status()
            pid = r.json()["project_id"]
        with server(cmd, env, scratch, out / "preflight_api.log"):
            wait_http(f"{base_url}/health")
            titles = {
                p["project_id"]: p["title"]
                for p in httpx.get(f"{base_url}/projects", timeout=10, trust_env=False).json()
            }
        info["project_after_restart"] = titles.get(pid)
        if titles.get(pid) != "Ön kontrol projesi ğüşİ":
            info["_status"] = "FAIL"

    with rep.step("UI: streamlit startup and health") as info:
        port = free_port()
        cmd = [
            sys.executable,
            "-m",
            "src.ui",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ]
        with server(cmd, env, scratch, out / "preflight_ui.log"):
            r = wait_http(f"http://127.0.0.1:{port}/_stcore/health", timeout=90)
            info["health"] = r.text.strip()
        info["note"] = "browser use (pages, uploads, downloads) is checked manually"

    with rep.step("UTF-8 export to a non-ASCII path") as info:
        target = scratch / "dışa aktarım" / "özet ğüşİ.md"
        target.parent.mkdir()
        text = "Duyma korunumu: %72,4 — τ² = 0,013; I² = 12 %\n"
        target.write_text(text, encoding="utf-8")
        info["round_trip"] = target.read_text(encoding="utf-8") == text
        if not info["round_trip"]:
            info["_status"] = "FAIL"

    if args.network:
        import httpx

        for name, u in NETWORK.items():
            with rep.step(f"network: {name}") as info:
                params = {}
                if name == "unpaywall":
                    email = os.environ.get("SYSREV_UNPAYWALL_EMAIL")
                    if not email:
                        info.update(_status="skip", reason="SYSREV_UNPAYWALL_EMAIL not set")
                        continue
                    params["email"] = email
                r = httpx.get(u, params=params, timeout=20)  # honours the system proxy
                info["http_status"] = r.status_code
                ok = r.status_code in ((401, 403) if name.endswith("no_key") else (200,))
                info["_status"] = "pass" if ok else "FAIL"
    else:
        rep.add("network", "skip", "run with --network to check the live services")

    failed = [c for c in rep.checks if c["status"] == "FAIL"]
    result = {
        "environment": env_info,
        "scratch": str(scratch),
        "checks": rep.checks,
        "failed": len(failed),
    }
    (out / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Session 10 preflight",
        "",
        f"* commit `{env_info['git_commit']}` "
        f"(clean: {env_info['git_clean']}); {env_info['os']}; Python {env_info['python']}; "
        f"UTF-8 mode {env_info['utf8_mode']}; locale encoding {env_info['locale_encoding']}",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    lines += [
        f"| {c['check']} | {c['status']} | {json.dumps(c['detail'], ensure_ascii=False)} |"
        for c in rep.checks
    ]
    (out / "preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(failed)} failed check(s); report in {out}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=Path("session10_results"))
    ap.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="parent of the scratch directory (default: the temp directory)",
    )
    ap.add_argument("--network", action="store_true", help="check live service reachability")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
