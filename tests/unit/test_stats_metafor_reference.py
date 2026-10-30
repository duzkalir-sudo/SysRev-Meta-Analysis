"""Independent numerical validation of src.stats_engine against R metafor (and lme4).

``tests/fixtures/stats/metafor_reference.json`` was produced by
``validation/r/metafor_reference.R`` (R 4.3.3, metafor 4.4-0, lme4 1.1-35.1; versions are
stored in the file). The committed values let this suite run without R; when ``Rscript``
with metafor, lme4 and jsonlite is available, ``test_live_r_matches`` regenerates the
references and compares again (the dedicated CI job installs R for this).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from src.stats_engine import effect_sizes as es
from src.stats_engine.numerics import heterogeneity, random_effects
from validation.compare_metafor import DEFAULT_REFERENCE, comparisons, main, within

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads(DEFAULT_REFERENCE.read_text())


def test_reference_metadata_records_software_versions() -> None:
    meta = REFERENCE["meta"]
    assert meta["metafor"] == "4.4.0" and meta["lme4"].startswith("1.1")
    assert meta["R"].startswith("R version 4.")
    ids = {c["id"] for c in REFERENCE["cases"]}
    assert {"pritz1997", "bcg", "hine1989", "normand1999"} <= ids  # real metadat datasets


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["id"])
def test_matches_metafor(case: dict[str, object]) -> None:
    sub = {"meta": REFERENCE["meta"], "cases": [case]}
    failures = [
        f"{cid} {q}: python={a!r} R={b!r} ({cls})"
        for cid, q, a, b, cls in comparisons(sub)
        if not within(a, b, cls)
    ]
    assert not failures, "\n".join(failures)


def test_comparison_count_is_substantial() -> None:
    assert sum(1 for _ in comparisons(REFERENCE)) > 1400


def test_bcg_matches_published_values() -> None:
    """Viechtbauer W. J Stat Softw 2010;36(3):1-48, BCG example (log RR, REML)."""
    case = next(c for c in REFERENCE["cases"] if c["id"] == "bcg")
    d = case["data"]
    ests = [es.log_ratio("RR", *t) for t in zip(d["ai"], d["n1i"], d["ci"], d["n2i"], strict=True)]
    y, v = [e.yi for e in ests], [e.vi for e in ests]
    re = random_effects(y, v, tau2_method="REML", ci_method="wald")
    assert round(re.pooled.estimate, 4) == -0.7145
    assert round(re.pooled.se, 4) == 0.1798
    assert round(re.tau2, 4) == 0.3132
    assert round(re.i2_model, 2) == 92.22
    assert round(heterogeneity(y, v).q, 4) == 152.2330


def _r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        [
            "Rscript",
            "-e",
            "for (p in c('metafor','lme4','jsonlite')) library(p, character.only=TRUE)",
        ],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _r_available(), reason="Rscript with metafor/lme4/jsonlite not installed")
def test_live_r_matches(tmp_path: Path) -> None:
    out = tmp_path / "ref.json"
    subprocess.run(
        [
            "Rscript",
            str(ROOT / "validation/r/metafor_reference.R"),
            str(ROOT / "validation/r/synthetic_cases.json"),
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    assert main(out) == 0
    fresh = json.loads(out.read_text())
    # the committed reference is reproducible from the script (up to build-level noise)
    committed = {c["id"]: c for c in REFERENCE["cases"]}
    assert set(committed) == {c["id"] for c in fresh["cases"]}
    for case in fresh["cases"]:
        assert case["data"] == committed[case["id"]]["data"]
        _assert_close(case["expected"], committed[case["id"]]["expected"], case["id"])


def _assert_close(a: object, b: object, path: str) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float | int) and isinstance(b, float | int):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), f"{path}: {a} != {b}"
    else:
        assert a == b, path
