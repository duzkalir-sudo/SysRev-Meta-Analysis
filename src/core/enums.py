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
    # Session 5: subscription or interface-only sources, entered through file imports.
    EMBASE = "embase"
    SCOPUS = "scopus"
    WEB_OF_SCIENCE = "web_of_science"
    COCHRANE_CENTRAL = "cochrane_central"
    CLINICALTRIALS_GOV = "clinicaltrials_gov"  # a trial register (PRISMA "registers" box)


class SearchRunKind(StrEnum):
    """Entries of the append-only search log (PRISMA-S)."""

    PREVIEW = "preview"  # count only; identifies nothing
    EXECUTION = "execution"  # records retrieved through an API client
    IMPORT = "import"  # records imported from a file exported by the user
    NOT_SEARCHED = "not_searched"  # a protocol database a human declares was not searched
    ACKNOWLEDGEMENT = "acknowledgement"  # a human acknowledges a count discrepancy
    DEDUPLICATION = "deduplication"  # records prepared for screening


class SearchRunStatus(StrEnum):
    COMPLETE = "complete"
    COUNT_MISMATCH = "count_mismatch"  # retrieved != reported; needs human acknowledgement
    FAILED = "failed"  # attempted but nothing identified (kept for the audit trail)


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
    # An established instrument not implemented here (e.g. MINORS, JBI case-series
    # checklist), named in ``tool_rationale`` and recorded by a human only.
    EXTERNAL = "external"


class FullTextStatus(StrEnum):
    PENDING = "pending"
    OA_DOWNLOADED = "oa_downloaded"
    MISSING = "missing"
    USER_UPLOADED = "user_uploaded"
    UNOBTAINABLE = "unobtainable"
    # Open-access structured full text (e.g. PMC JATS XML) retrieved, but no PDF yet.
    OA_STRUCTURED_TEXT = "oa_structured_text"


class UploadMatchStatus(StrEnum):
    """Does a PDF belong to the study record it was linked to? (deterministic heuristics)"""

    CONFIRMED = "confirmed"  # strong deterministic evidence (e.g. DOI printed on page 1)
    UNCERTAIN = "uncertain"  # needs a human to confirm before Gate 2
    MISMATCH = "mismatch"  # evidence that the PDF is a different report
    HUMAN_CONFIRMED = "human_confirmed"  # an authenticated reviewer confirmed the link


class DataStatus(StrEnum):
    """What the source says about one requested datum."""

    REPORTED = "reported"  # printed in the document, verbatim evidence attached
    NOT_REPORTED = "not_reported"  # the document does not report it (never stored as zero)
    AMBIGUOUS = "ambiguous"  # reported, but unclear (e.g. denominator or timepoint unclear)
    CONTRADICTORY = "contradictory"  # the document reports conflicting values
    INACCESSIBLE = "inaccessible"  # e.g. only in a figure or an image-only page
    DERIVED = "derived"  # computed from reported data — not authorised in Stage 4


class EvidenceCheckStatus(StrEnum):
    """Result of checking a SourceEvidence against the parsed source document."""

    VERIFIED = "verified"
    HUMAN_ATTESTED = "human_attested"  # not machine-verifiable; a named human vouches for it
    NOT_CHECKED = "not_checked"
    HASH_MISMATCH = "hash_mismatch"
    FILE_MISMATCH = "file_mismatch"
    PAGE_OUT_OF_RANGE = "page_out_of_range"
    WRONG_PAGE = "wrong_page"
    QUOTE_NOT_FOUND = "quote_not_found"
    NUMBER_NOT_IN_QUOTE = "number_not_in_quote"
    LOCATION_MISMATCH = "location_mismatch"
    PAGE_UNREADABLE = "page_unreadable"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"  # replaced by a human correction (a new item version)


class ExtractionDomain(StrEnum):
    STUDY_CHARACTERISTICS = "study_characteristics"  # design, institution, recruitment, n
    BASELINE = "baseline"
    INTERVENTION = "intervention"
    OUTCOME_DEFINITION = "outcome_definition"
    OUTCOME_DATA = "outcome_data"
    FOLLOW_UP = "follow_up"
    ADVERSE_EVENT = "adverse_event"


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
