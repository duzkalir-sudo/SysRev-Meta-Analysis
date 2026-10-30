"""Controlled vocabularies shared by schemas, persistence and the state machine."""

from __future__ import annotations

from enum import StrEnum


class Phase(StrEnum):
    """Pipeline phases. Gate phases are pauses that only a human can release."""

    STAGE1_PROTOCOL = "stage1_protocol"
    GATE1_SEARCH_STRATEGY = "gate1_search_strategy"
    STAGE2_SEARCH_SCREENING = "stage2_search_screening"
    GATE1_5_SCREENING_CONFLICTS = "gate1_5_screening_conflicts"
    STAGE3_FULLTEXT_RETRIEVAL = "stage3_fulltext_retrieval"
    GATE2_MISSING_FULLTEXT = "gate2_missing_fulltext"
    STAGE4_EXTRACTION_ROB = "stage4_extraction_rob"
    GATE3_EXTRACTION_APPROVAL = "gate3_extraction_approval"
    STAGE5_META_ANALYSIS = "stage5_meta_analysis"
    STAGE6_MANUSCRIPT = "stage6_manuscript"
    STAGE6_AUDIT = "stage6_audit"
    COMPLETED = "completed"


class GateId(StrEnum):
    GATE_1 = "gate_1"
    GATE_1_5 = "gate_1_5"
    GATE_2 = "gate_2"
    GATE_3 = "gate_3"


class GateVerdict(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class ActorType(StrEnum):
    SYSTEM = "system"
    AGENT = "agent"
    HUMAN = "human"


class SourceDatabase(StrEnum):
    PUBMED = "pubmed"
    EUROPE_PMC = "europe_pmc"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    MANUAL = "manual"


class ReviewDesign(StrEnum):
    COMPARATIVE = "comparative"
    SINGLE_ARM = "single_arm"


class StudyDesign(StrEnum):
    RCT = "rct"
    CLUSTER_RCT = "cluster_rct"
    CROSSOVER_RCT = "crossover_rct"
    NON_RANDOMISED_TRIAL = "non_randomised_trial"
    PROSPECTIVE_COHORT = "prospective_cohort"
    RETROSPECTIVE_COHORT = "retrospective_cohort"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    SINGLE_ARM_TRIAL = "single_arm_trial"
    CASE_SERIES = "case_series"


class ComparisonType(StrEnum):
    COMPARATIVE = "comparative"
    SINGLE_GROUP = "single_group"


class EffectMeasure(StrEnum):
    """Summary measure an outcome is analysed on. Defines the target only; computes nothing."""

    # Comparative (between groups)
    RR = "RR"
    OR = "OR"
    RD = "RD"
    MD = "MD"
    SMD = "SMD"
    HR = "HR"
    IRR = "IRR"
    # Single group
    PROPORTION = "PROPORTION"  # crude events / n, ignores censoring
    KM_PROBABILITY = "KM_PROBABILITY"  # Kaplan-Meier (or actuarial) probability at time t
    CUMULATIVE_INCIDENCE = "CUMULATIVE_INCIDENCE"  # competing-risks cumulative incidence at t
    MEDIAN_TIME_TO_EVENT = "MEDIAN_TIME_TO_EVENT"
    INCIDENCE_RATE = "INCIDENCE_RATE"  # events per person-time
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"


class ClinicalOutcomeType(StrEnum):
    TUMOR_CONTROL = "tumor_control"
    AVM_OBLITERATION = "avm_obliteration"
    PAIN_RELIEF = "pain_relief"
    HEARING_PRESERVATION = "hearing_preservation"
    ADVERSE_EVENT = "adverse_event"
    OVERALL_SURVIVAL = "overall_survival"
    PROGRESSION_FREE_SURVIVAL = "progression_free_survival"
    FUNCTIONAL_OUTCOME = "functional_outcome"
    OTHER = "other"


class TimepointType(StrEnum):
    FIXED = "fixed"  # e.g. at 5 years
    WINDOW = "window"  # e.g. between 24 and 36 months
    LAST_FOLLOW_UP = "last_follow_up"  # at each patient's last follow-up (variable)


class EstimateType(StrEnum):
    """How a study-reported arm-level estimate was derived (as stated by the authors)."""

    CRUDE_PROPORTION = "crude_proportion"
    KAPLAN_MEIER = "kaplan_meier"
    ACTUARIAL_LIFE_TABLE = "actuarial_life_table"
    CUMULATIVE_INCIDENCE = "cumulative_incidence"  # competing-risks estimator
    INCIDENCE_RATE = "incidence_rate"
    MEDIAN_TIME_TO_EVENT = "median_time_to_event"


class ScreeningStage(StrEnum):
    TITLE_ABSTRACT = "title_abstract"
    FULL_TEXT = "full_text"


class ScreeningVerdict(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"


class ReviewerRole(StrEnum):
    REVIEWER_1 = "reviewer_1"
    REVIEWER_2 = "reviewer_2"
    ADJUDICATOR = "adjudicator"


class ArmRole(StrEnum):
    INTERVENTION = "intervention"
    CONTROL = "control"
    SINGLE_GROUP = "single_group"
    OTHER = "other"


class EvidenceLocationType(StrEnum):
    TABLE = "table"
    FIGURE = "figure"
    PARAGRAPH = "paragraph"
    SUPPLEMENT = "supplement"


class RoBTool(StrEnum):
    ROB2 = "rob2"
    ROBINS_I = "robins_i"
    NOS = "nos"


class FullTextStatus(StrEnum):
    PENDING = "pending"
    OA_DOWNLOADED = "oa_downloaded"
    MISSING = "missing"
    USER_UPLOADED = "user_uploaded"
    UNOBTAINABLE = "unobtainable"


class OverlapSeverity(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class OverlapResolution(StrEnum):
    UNRESOLVED = "unresolved"
    DISTINCT_COHORTS = "distinct_cohorts"
    OVERLAPPING_KEEP_A = "overlapping_keep_a"
    OVERLAPPING_KEEP_B = "overlapping_keep_b"
    OVERLAPPING_DIFFERENT_OUTCOMES = "overlapping_different_outcomes"


class CheckStatus(StrEnum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    NOT_FOUND = "not_found"
