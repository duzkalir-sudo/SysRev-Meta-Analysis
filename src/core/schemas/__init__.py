"""Pydantic v2 data contracts for every stage of the pipeline."""

from src.core.schemas.audit import AuditReport, CitationCheck, NumericClaimCheck
from src.core.schemas.cohort import CohortOverlapWarning, OverlapSignal
from src.core.schemas.extraction import (
    DomainJudgement,
    ExtractedStudyArm,
    ExtractedValue,
    FullTextItem,
    MasterExtractionTable,
    RiskOfBiasAssessment,
    SourceEvidence,
    parse_reported_number,
)
from src.core.schemas.pipeline import GateDecision, PRISMACounts, ProjectState, StateTransition
from src.core.schemas.protocol import MeSHTerm, OutcomeDefinition, PICOProtocol, SearchStrategy
from src.core.schemas.screening import AgreementStatistics, ScreeningConflict, ScreeningDecision
from src.core.schemas.study import Author, RecruitmentPeriod, SearchProvenance, StudyRecord

__all__ = [
    "AgreementStatistics",
    "AuditReport",
    "Author",
    "CitationCheck",
    "CohortOverlapWarning",
    "DomainJudgement",
    "ExtractedStudyArm",
    "ExtractedValue",
    "FullTextItem",
    "GateDecision",
    "MasterExtractionTable",
    "MeSHTerm",
    "NumericClaimCheck",
    "OutcomeDefinition",
    "OverlapSignal",
    "PICOProtocol",
    "PRISMACounts",
    "ProjectState",
    "RecruitmentPeriod",
    "RiskOfBiasAssessment",
    "ScreeningConflict",
    "ScreeningDecision",
    "SearchProvenance",
    "SearchStrategy",
    "SourceEvidence",
    "StateTransition",
    "StudyRecord",
    "parse_reported_number",
]
