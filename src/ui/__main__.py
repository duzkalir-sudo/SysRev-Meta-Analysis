"""Launch the local UI: ``python -m src.ui`` (works the same in Windows PowerShell).

Binds to 127.0.0.1 only: reviewer tokens and API keys must not cross a network.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app = Path(__file__).with_name("app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false",
        "--client.toolbarMode", "minimal",
        *sys.argv[1:],
    ]  # fmt: skip
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
