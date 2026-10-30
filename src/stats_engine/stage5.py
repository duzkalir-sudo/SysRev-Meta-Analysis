"""Stage 5 orchestration: approved plan + approved data → verified ``stats_output.json``.

``run_stage5`` is the only way a statistics artifact enters the pipeline. It refuses to run
outside Stage 5, without an approved analysis plan, with unapproved arms, or with arms for
reports not included at full-text assessment. The artifact is recorded in the database
(append-only) before any file is written; Stage 6 can only start once
``Repository.stats_output_problems`` finds nothing wrong (see the Stage 5 → 6 guard).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.core.enums import Phase, ScreeningStage
from src.core.exceptions import StatisticsInputError
from src.core.schemas.base import utc_now
from src.core.schemas.stats import StatsOutput, canonical_json
from src.state.repository import Repository
from src.stats_engine.engine import run_plan
from src.stats_engine.report import analysis_csv, analysis_markdown, forest_plot, report_markdown


@dataclass(frozen=True)
class Stage5Artifacts:
    output: StatsOutput
    sha256: str
    directory: Path
    files: dict[str, str]  # relative path -> SHA-256


def run_stage5(
    repo: Repository,
    project_id: str,
    *,
    output_dir: Path,
    generated_at: datetime | None = None,
    render_plots: bool = True,
) -> Stage5Artifacts:
    project = repo.get_project(project_id)
    if project.phase != Phase.STAGE5_META_ANALYSIS.value:
        raise StatisticsInputError(f"Stage 5 cannot run in phase {project.phase}")
    plan = repo.latest_analysis_plan(project_id)
    if plan is None:
        raise StatisticsInputError("no human-approved analysis plan")
    inputs = repo.load_stage5_inputs(project_id)
    included = repo.included_record_ids(project_id, ScreeningStage.FULL_TEXT)
    stray = sorted({a.record_id for a in inputs.arms} - included)
    if stray:
        raise StatisticsInputError(f"arms exist for reports not included at full text: {stray}")
    output = run_plan(plan, inputs, project_id=project_id, generated_at=generated_at or utc_now())
    sha = repo.record_stats_output(project_id, output)
    files = write_artifacts(output, output_dir, render_plots=render_plots)
    return Stage5Artifacts(output=output, sha256=sha, directory=output_dir, files=files)


def write_artifacts(output: StatsOutput, directory: Path, *, render_plots: bool) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def put(rel: str, text: str) -> None:
        path = directory / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode("utf-8")
        path.write_bytes(data)
        written[rel] = hashlib.sha256(data).hexdigest()

    put("stats_output.json", output.to_canonical_json())
    put("report.md", report_markdown(output))
    for a in output.analyses:
        put(f"tables/{a.analysis_id}.md", analysis_markdown(a))
        put(f"tables/{a.analysis_id}.csv", analysis_csv(a))
        if render_plots:
            svg = forest_plot(a, directory / "plots" / f"{a.analysis_id}.svg")
            if svg is not None:
                written[f"plots/{a.analysis_id}.svg"] = hashlib.sha256(svg.read_bytes()).hexdigest()
    put("manifest.json", canonical_json({"stats_output_sha256": output.sha256(), "files": written}))
    return written
