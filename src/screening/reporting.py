"""Accurate reporting of who screened: automated passes are never presented as humans.

``screening_reporting_facts`` derives the facts from stored decisions; ``methods_statement``
renders them as a deterministic sentence for the manuscript writer (Session 4+); and
``check_reviewer_claims`` flags manuscript text that describes automated passes as
independent human reviewers (for the Auditor).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.enums import ReviewerRole, ScreeningStage
from src.screening.llm_screening import agreement_summary
from src.state.repository import Repository


@dataclass(frozen=True, slots=True)
class ScreeningReportingFacts:
    stage: ScreeningStage
    automated_passes: tuple[str, ...]  # "provider/model (label)"
    human_screeners: tuple[str, ...]  # humans acting as reviewer_1/reviewer_2
    human_adjudicators: tuple[str, ...]
    paired_records: int
    kappa: float | None
    conflicts: int
    same_model_both_passes: bool

    @property
    def human_independent_reviewers(self) -> int:
        return len(self.human_screeners)


def screening_reporting_facts(
    repo: Repository, project_id: str, stage: ScreeningStage
) -> ScreeningReportingFacts:
    automated: dict[ReviewerRole, set[str]] = {}
    humans: set[str] = set()
    adjudicators: set[str] = set()
    models: set[tuple[str, str]] = set()
    for d in repo.load_screening_decisions(project_id):
        if d.stage != stage:
            continue
        if d.reviewer == ReviewerRole.ADJUDICATOR:
            adjudicators.add(d.reviewer_identity)
        elif d.is_human:
            humans.add(d.reviewer_identity)
        else:
            automated.setdefault(d.reviewer, set()).add(d.reviewer_identity)
            if d.automation:
                models.add((d.automation.provider, d.automation.model_id))
    agreement = agreement_summary(repo, project_id, stage)
    passes = tuple(sorted({i for ids in automated.values() for i in ids}))
    return ScreeningReportingFacts(
        stage=stage,
        automated_passes=passes,
        human_screeners=tuple(sorted(humans)),
        human_adjudicators=tuple(sorted(adjudicators)),
        paired_records=agreement.paired_records,
        kappa=agreement.statistics.kappa if agreement.statistics else None,
        conflicts=agreement.conflicts,
        same_model_both_passes=len(automated) == 2 and len(models) == 1,
    )


def methods_statement(f: ScreeningReportingFacts) -> str:
    stage = "Title and abstract" if f.stage == ScreeningStage.TITLE_ABSTRACT else "Full-text"
    parts: list[str] = []
    if f.automated_passes and not f.human_screeners:
        same = " (two runs of the same model)" if f.same_model_both_passes else ""
        parts.append(
            f"{stage} screening was performed by two automated screening passes using large "
            f"language models{same} [{'; '.join(f.automated_passes)}], applied to the "
            "pre-specified eligibility criteria. These automated passes are not equivalent "
            "to independent human reviewers."
        )
    elif f.automated_passes:
        parts.append(
            f"{stage} screening combined {len(f.human_screeners)} human reviewer(s) with "
            f"automated large-language-model screening [{'; '.join(f.automated_passes)}]."
        )
    else:
        parts.append(
            f"{stage} screening was performed by {len(f.human_screeners)} human reviewer(s)."
        )
    if f.kappa is not None:
        parts.append(
            f"Agreement between the two screening passes was Cohen's kappa = {f.kappa:.2f} "
            f"over {f.paired_records} records."
        )
    parts.append(
        f"{f.conflicts} record(s) with disagreement or uncertainty were adjudicated by "
        f"{len(f.human_adjudicators)} human reviewer(s)."
    )
    return " ".join(parts)


_HUMAN_CLAIM = re.compile(
    r"\b(two|2|both|independent(ly)?)\b[^.]{0,40}\b(human\s+)?"
    r"(reviewers?|investigators?|authors?|researchers?|raters?|screeners?)\b",
    re.IGNORECASE,
)


def check_reviewer_claims(text: str, facts: ScreeningReportingFacts) -> list[str]:
    """Sentences that would misrepresent automated passes as two human reviewers."""
    if facts.human_independent_reviewers >= 2:
        return []
    problems: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if _HUMAN_CLAIM.search(sentence) and not re.search(
            r"\b(automated|language model|LLM)\b", sentence, re.IGNORECASE
        ):
            problems.append(
                f"claims human dual screening, but only {facts.human_independent_reviewers} "
                f"human screener(s) are recorded: {sentence.strip()!r}"
            )
    return problems
