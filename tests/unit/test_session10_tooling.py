"""Session 10 validation tooling, exercised on synthetic data only (no real review).

``validation.session10.check_pilot_stats`` must reproduce, with R metafor, every model the
engine reports for the pilot's analyses. Here it is run on synthetic outputs whose metafor
agreement is already established (validation/VALIDATION_REPORT.md), so a failure means a
defect of the checker, not of the engine."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from src.core.schemas.stats import AnalysisSpec
from src.stats_engine.engine import run_analysis
from tests.stats_helpers import LAST_FU, OUTCOMES, inputs, necrosis_arms
from tests.unit.test_session9_review_followup import TRIALS, _two, _workflow
from validation.session10 import check_pilot_stats as cps


def _has_r() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = 'for (p in c("metafor","lme4","jsonlite")) library(p, character.only = TRUE)'
    return (
        subprocess.run(["Rscript", "-e", probe], capture_output=True, check=False).returncode == 0
    )


# In the CI job that installs R (SYSREV_REQUIRE_R=1) a missing R fails instead of skipping.
needs_r = pytest.mark.skipif(
    not _has_r() and not os.environ.get("SYSREV_REQUIRE_R"),
    reason="Rscript with metafor/lme4/jsonlite missing",
)


def _write(tmp: Path, analyses: list[Any]) -> Path:
    d = tmp / "stats"
    d.mkdir()
    body = json.dumps({"engine_version": "synthetic", "analyses": [
        a.model_dump(mode="json") for a in analyses]})  # fmt: skip
    (d / "stats_output.json").write_text(body, encoding="utf-8")
    sha = hashlib.sha256(body.encode()).hexdigest()
    (d / "manifest.json").write_text(json.dumps({"stats_output_sha256": sha}), encoding="utf-8")
    return d


def _cases() -> list[Any]:
    arms = [arm for t in TRIALS for arm in _two(*t)]
    out = []
    for primary in ("random_effects", "common_effect"):
        for measure in ("RR", "OR"):
            spec = AnalysisSpec(analysis_id=f"{measure.lower()}_{primary}",
                                outcome_id="complication", effect_measure=measure,
                                primary_model=primary)  # type: ignore[arg-type]  # fmt: skip
            out.append(run_analysis(spec, OUTCOMES["complication"], inputs(arms)))
    for method in ("glmm", "logit_inverse_variance"):
        spec = AnalysisSpec(analysis_id=f"nec_{method}", outcome_id="necrosis",
                            effect_measure="PROPORTION", proportion_method=method,
                            timepoints=[LAST_FU])  # type: ignore[arg-type]  # fmt: skip
        out.append(run_analysis(spec, OUTCOMES["necrosis"], inputs(necrosis_arms())))
    return out


def test_reference_mapping_never_passes_an_unknown_model() -> None:
    a = {"settings": {"tau2_estimator": "REML", "random_effects_ci": "hksj"}}
    assert cps.reference_key(a, {"method": "inverse variance", "model": "common_effect"}) == (
        "iv_common", "closed")  # fmt: skip
    re_ = "inverse variance, REML τ², Hartung-Knapp-Sidik-Jonkman t(k−1) CI"
    assert cps.reference_key(a, {"method": re_, "model": "random_effects"})[0] == "iv_random"  # type: ignore[index]
    wald = "inverse variance, REML τ², Wald (normal) CI"
    assert cps.reference_key(a, {"method": wald, "model": "random_effects"}) is None
    assert cps.reference_key(a, {"method": "something new", "model": "common_effect"}) is None


@needs_r
def test_checker_reproduces_every_model_on_synthetic_outputs(tmp_path: Path) -> None:
    stats_dir = _write(tmp_path, _cases())
    out = tmp_path / "out"
    assert cps.main([str(stats_dir), "--out", str(out)]) == 0
    res = json.loads((out / "pilot_stats_check.json").read_text(encoding="utf-8"))
    assert res["failures"] == 0 and res["listed_in_manifest"]
    rows = res["rows"]
    compared = {(r["analysis_id"], r["quantity"].split(" [")[0]) for r in rows
                if r["status"] == "pass"}  # fmt: skip
    # every pooled model of every analysis was compared, including the MH RR with the
    # double-all-event table (k = 3) and the GLMM
    for aid in ("rr_random_effects", "rr_common_effect", "or_random_effects", "or_common_effect",
                "nec_glmm", "nec_logit_inverse_variance"):  # fmt: skip
        assert (aid, "common_effect") in compared and (aid, "random_effects") in compared, aid
    mh_k = next(r for r in rows if r["analysis_id"] == "rr_common_effect"
                and r["quantity"].endswith("] k") and "Mantel" in r["quantity"])  # fmt: skip
    assert mh_k["engine"] == mh_k["reference"] == 3
    assert res["comparisons"] > 150


@needs_r
def test_checker_detects_a_changed_value(tmp_path: Path) -> None:
    analyses = _cases()
    d = _write(tmp_path, analyses)
    data = json.loads((d / "stats_output.json").read_text(encoding="utf-8"))
    data["analyses"][0]["random_effects"]["ci_upper_natural"] *= 1.001
    data["analyses"][0]["studies"][0]["data"]["n_control"] += 1  # population no longer matches
    (d / "stats_output.json").write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "out"
    assert cps.main([str(d), "--out", str(out)]) == 1
    res = json.loads((out / "pilot_stats_check.json").read_text(encoding="utf-8"))
    assert not res["listed_in_manifest"]
    failed = {r["quantity"] for r in res["rows"] if r["status"] == "FAIL"}
    assert any(q.endswith("ci_upper_natural") for q in failed)
    assert any(q.endswith("participants") for q in failed)


@needs_r
@pytest.mark.parametrize("primary", ["random_effects", "common_effect"])
def test_checker_on_real_stage5_artifacts(session: Session, tmp_path: Path, primary: str) -> None:
    """The artifact directory written by run_stage5 in a guarded workflow is accepted as is."""
    _workflow(session, primary, tmp_path)
    out = tmp_path / "out"
    assert cps.main([str(tmp_path / "stats"), "--out", str(out)]) == 0
