"""The deterministic evidence package: every fact a manuscript may state, with its source.

The package is built from persisted, human-approved records only — the approved protocol,
the search log, screening decisions, the PRISMA snapshot, approved extraction data and
risk-of-bias assessments, the latest approved analysis plan and the verified
``stats_output.json`` — before any language model is involved. Each :class:`Fact` has a
stable key (used as ``{{key}}`` in the manuscript source), a journal-ready text, the
full-precision numbers behind that text, and semantic attributes the Auditor uses (pooled
or single-study, estimator, model, null value, count unit).

The package also enumerates the *expected reportable outputs* (coverage list) and a
fingerprint of every input, so a manuscript version or audit can be tied to the exact
evidence state it used and invalidated when any of it changes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.core.enums import (
    ActorType,
    EffectMeasure,
    ReviewDesign,
    ReviewStatus,
    RoBTool,
    ScreeningStage,
    SearchRunKind,
    SearchRunStatus,
    SourceDatabase,
    TimepointType,
)
from src.core.exceptions import SysRevError
from src.core.schemas.extraction import RiskOfBiasAssessment
from src.core.schemas.manuscript import (
    EvidenceExcerpt,
    ReferenceEntry,
    ReferenceVerification,
    fingerprint_sha256,
)
from src.core.schemas.pipeline import PRISMACounts
from src.core.schemas.protocol import FollowUpTimepoint, PICOProtocol
from src.core.schemas.search import SearchRun
from src.core.schemas.stats import AnalysisResult, PooledResult, StatsOutput, canonical_sha256
from src.core.schemas.study import Author, StudyRecord
from src.manuscript.formatting import (
    DisplayNumber,
    find_numbers,
    fmt,
    fmt_count,
    fmt_p,
    numbers_in_text,
)
from src.reporting.state import ReportingState, add_reporting_outputs, build_reporting_state
from src.search.databases import DATABASES
from src.state.repository import Repository

PROBABILITY = frozenset(
    {EffectMeasure.PROPORTION, EffectMeasure.KM_PROBABILITY, EffectMeasure.CUMULATIVE_INCIDENCE}
)
RATIO = frozenset({EffectMeasure.RR, EffectMeasure.OR})
DIFFERENCE = frozenset({EffectMeasure.MD, EffectMeasure.SMD})
MEASURE_LABEL = {
    EffectMeasure.PROPORTION: "crude proportion",
    EffectMeasure.KM_PROBABILITY: "Kaplan–Meier probability",
    EffectMeasure.CUMULATIVE_INCIDENCE: "cumulative incidence (competing-risks estimator)",
    EffectMeasure.RR: "risk ratio",
    EffectMeasure.OR: "odds ratio",
    EffectMeasure.MD: "mean difference",
    EffectMeasure.SMD: "standardised mean difference (Hedges' g)",
}
ESTIMATOR = {
    EffectMeasure.PROPORTION: "crude_proportion",
    EffectMeasure.KM_PROBABILITY: "kaplan_meier",
    EffectMeasure.CUMULATIVE_INCIDENCE: "cumulative_incidence",
}
ROB_LABEL = {
    RoBTool.ROB2: "the Cochrane RoB 2 tool",
    RoBTool.ROBINS_I: "ROBINS-I",
    RoBTool.NOS: "the Newcastle–Ottawa Scale",
    RoBTool.EXTERNAL: "an external instrument",
}
# Methods the engine never performs (Session 3 scope). Mentioning them as performed is an
# unsupported claim.
NEVER_PERFORMED = (
    "meta_regression",
    "subgroup",
    "publication_bias_test",
    "trim_and_fill",
    "leave_one_out",
    "bayesian",
    "network",
    "grade",
    "mean_sd_conversion",
    "hazard_ratio_pooling",
    "ipd_reconstruction",
)
AUTOMATED_SOURCES = frozenset({SourceDatabase.PUBMED})
# Short wording of the engine's exclusion codes (full reasons are in the report ledger).
EXCLUSION_TEXT = {
    "outcome_data_not_extracted_or_unverified": "no approved, verified data extracted for this "
    "outcome; this does not mean the outcome was unreported",
    "confirmed_not_reported": "outcome not reported, as confirmed by a reviewer",
    "recorded_unavailable": "data recorded by a reviewer as unusable for this outcome",
    "ci_inconsistent": "reported confidence interval inconsistent with the declared scale",
    "ci_bound_not_transformable": "a confidence limit at 0% or 100%, where the declared "
    "transformation is undefined",
    "timepoint_mismatch": "no estimate at the analysed timepoint",
    "timepoint_missing": "follow-up timepoint not reported",
    "incompatible_estimator": "estimate of a different type (e.g. Kaplan–Meier vs crude)",
    "overlapping_cohort": "overlapping cohort already represented",
    "boundary_estimate": "estimate at 0% or 100% without a usable interval",
    "non_informative": "no information for this measure (e.g. no events in either arm)",
}


@dataclass(frozen=True, slots=True)
class Fact:
    key: str
    text: str
    description: str
    source: str
    category: str
    numbers: tuple[DisplayNumber, ...] = ()
    role: str = "text"
    analysis_id: str | None = None
    record_ids: tuple[str, ...] = ()
    measure: str | None = None
    estimator: str | None = None
    model: str | None = None
    pooled: bool | None = None
    null_value: float | None = None
    ci: tuple[float, float] | None = None  # natural-scale CI of an estimate
    count_unit: str | None = None

    def sha256(self) -> str:
        return canonical_sha256(
            {
                "key": self.key,
                "text": self.text,
                "numbers": [[n.role, repr(n.value), n.decimals, n.relation] for n in self.numbers],
                "source": self.source,
            }
        )

    @property
    def ci_excludes_null(self) -> bool | None:
        if self.ci is None or self.null_value is None:
            return None
        lo, hi = self.ci
        return lo > self.null_value or hi < self.null_value


@dataclass(frozen=True, slots=True)
class ExpectedOutput:
    """An output the manuscript must report (coverage), with the sections that may carry it.
    ``any_of`` lists alternative keys: one of them must be bound."""

    key: str
    description: str
    sections: tuple[str, ...]
    any_of: tuple[str, ...] = ()

    @property
    def keys(self) -> tuple[str, ...]:
        return self.any_of or (self.key,)


@dataclass(frozen=True, slots=True)
class ReferenceState:
    entry: ReferenceEntry
    # latest live lookup per source, *for the current metadata* (older metadata ignored)
    verifications: tuple[ReferenceVerification, ...]
    provenance_basis: str | None  # identity established by retrieval from a bibliographic API
    excerpts: tuple[EvidenceExcerpt, ...]
    # human attestations no longer backed by the report's current confirmed PDF (ignored)
    invalid_attestations: tuple[str, ...] = ()
    # current PDF identity of an included report ("<sha256>:<match status>"), fingerprinted
    pdf_identity: str | None = None

    @property
    def identity_status(self) -> str:
        """verified / mismatch / not_found / unavailable / not_checked (worst source wins)."""
        decisive = {v.status for v in self.verifications} - {"unavailable"}
        for status in ("mismatch", "not_found", "verified"):
            if status in decisive:
                return status
        if self.provenance_basis is not None:
            return "verified"
        return "unavailable" if self.verifications else "not_checked"

    @property
    def identity_basis(self) -> str:
        if any(v.status != "unavailable" for v in self.verifications):
            return "; ".join(
                f"{v.source} {v.status} ({v.checked_at.date()})" for v in self.verifications
            )
        if self.provenance_basis:
            return self.provenance_basis
        notes = [v.message for v in self.verifications] + list(self.invalid_attestations)
        return "; ".join(notes) or "never verified"


@dataclass
class EvidencePackage:
    project_id: str
    review_design: ReviewDesign
    protocol: PICOProtocol
    stats: StatsOutput
    stats_sha256: str
    facts: dict[str, Fact]
    templates: dict[str, str]  # table/figure name -> Markdown source with markers
    references: dict[str, ReferenceState]
    included_ref_keys: dict[str, str]  # record_id -> ref_key (full-text included reports)
    study_of: dict[str, str]  # record_id -> study id
    searched_databases: frozenset[str]
    rob_tools: frozenset[RoBTool]
    performed_methods: frozenset[str]
    automated_screening: dict[str, tuple[int, int]]  # stage -> (human screeners, automated)
    expected: list[ExpectedOutput]
    problems: list[str]
    fingerprint: dict[str, str] = field(default_factory=dict)
    # Session 7: the stored PRISMA snapshot and the reporting state (flow diagram, GRADE,
    # syntheses without meta-analysis, SWiM, checklist decisions)
    prisma: PRISMACounts | None = None
    reporting: ReportingState | None = None

    @property
    def fingerprint_sha256(self) -> str:
        return fingerprint_sha256(self.fingerprint)

    def analysis(self, analysis_id: str) -> AnalysisResult | None:
        return next((a for a in self.stats.analyses if a.analysis_id == analysis_id), None)

    def facts_for_analysis(self, analysis_id: str) -> list[Fact]:
        return [f for f in self.facts.values() if f.analysis_id == analysis_id]


# --------------------------------------------------------------------------- helpers


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def vancouver_author(a: Author) -> str:
    if a.last_name:
        initials = a.initials or "".join(
            p[0] for p in (a.fore_name or "").replace("-", " ").split() if p
        )
        return f"{a.last_name} {initials}".strip()
    return a.collective_name or a.full_name or ""


def study_label(rec: StudyRecord) -> str:
    author = rec.authors[0].last_name if rec.authors and rec.authors[0].last_name else None
    if author is None and rec.authors:
        author = rec.authors[0].collective_name or rec.authors[0].full_name
    parts = [p for p in (author, str(rec.publication_year) if rec.publication_year else None) if p]
    return " ".join(parts) or rec.title[:40]


def record_reference(rec: StudyRecord) -> ReferenceEntry:
    return ReferenceEntry(
        ref_key=f"study-{rec.record_id}"[:64],
        origin="included_study_record",
        record_id=rec.record_id,
        pmid=rec.pmid,
        doi=rec.doi,
        title=rec.title,
        authors=[x for x in (vancouver_author(a) for a in rec.authors) if x],
        journal=rec.journal,
        year=rec.publication_year,
        metadata_source=f"stored_record:{rec.source_database.value}",
        retrieved_at=rec.retrieved_at,
        added_by="system (stored bibliographic record)",
    )


def provenance_basis(rec: StudyRecord) -> str | None:
    """Identity is established when the record itself was retrieved from PubMed E-utilities
    by the deterministic client (its metadata *is* PubMed metadata)."""
    p = rec.provenance
    if (
        p is not None
        and p.database in AUTOMATED_SOURCES
        and "efetch" in p.endpoint
        and rec.pmid is not None
    ):
        return f"retrieved from PubMed E-utilities ({p.endpoint}) on {p.executed_at.date()}"
    return None


def _ci_level(a: AnalysisResult) -> float:
    return float(a.settings.get("confidence_level", 0.95)) * 100


def _decimals(a: AnalysisResult) -> int:
    return int(a.settings.get("display_decimals", 2))


def _estimate_text(
    a: AnalysisResult, est: float, lo: float | None, hi: float | None, role: str
) -> tuple[str, tuple[DisplayNumber, ...]]:
    """'8.1% (95% CI 3.9% to 16.0%)' / '0.72 (95% CI 0.55 to 0.94)'."""
    pct = a.effect_measure in PROBABILITY
    d = 1 if pct else _decimals(a)
    m = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    level = _ci_level(a)
    nums = [DisplayNumber(role, est * m, d)]
    text = f"{fmt(est * m, d)}{unit}"
    if lo is not None and hi is not None:
        nums += [
            DisplayNumber("ci_level", level, 0),
            DisplayNumber("ci_lower", lo * m, d),
            DisplayNumber("ci_upper", hi * m, d),
        ]
        text += f" ({fmt(level, 0)}% CI {fmt(lo * m, d)}{unit} to {fmt(hi * m, d)}{unit})"
    return text, tuple(nums)


def _null(a: AnalysisResult) -> float | None:
    if a.effect_measure in RATIO:
        return 1.0
    if a.effect_measure in DIFFERENCE:
        return 0.0
    return None


def _timepoints(a: AnalysisResult) -> str:
    return _join(list(a.timepoints)) if a.timepoints else ""


def _measure_label(a: AnalysisResult) -> str:
    label = MEASURE_LABEL.get(a.effect_measure, str(a.effect_measure))
    if a.effect_measure == EffectMeasure.KM_PROBABILITY and a.settings.get(
        "combine_km_and_actuarial"
    ):
        label = "Kaplan–Meier or actuarial (life-table) probability"
    tp = _timepoints(a)
    return f"{label} at {tp}" if tp else label


def _model_label(r: PooledResult) -> str:
    kind = "random-effects" if r.model == "random_effects" else "common-effect"
    return f"{kind} model ({r.method})"


# --------------------------------------------------------------------------- builder


class _Builder:
    def __init__(self) -> None:
        self.facts: dict[str, Fact] = {}

    def add(
        self,
        key: str,
        text: str,
        description: str,
        source: str,
        category: str,
        numbers: tuple[DisplayNumber, ...] | None = None,
        **kw: object,
    ) -> Fact:
        if key in self.facts:
            raise SysRevError(f"duplicate evidence key {key}")
        if numbers is not None and len(numbers) != len(find_numbers(text)):
            raise SysRevError(f"{key}: every number in the text must have a source value")
        fact = Fact(
            key=key,
            text=text,
            description=description,
            source=source,
            category=category,
            numbers=numbers if numbers is not None else numbers_in_text(text),
            **kw,  # type: ignore[arg-type]
        )
        self.facts[key] = fact
        return fact

    def count(
        self,
        key: str,
        n: int,
        description: str,
        source: str,
        category: str,
        unit: str,
        **kw: object,
    ) -> Fact:
        text, num = fmt_count(n)
        return self.add(
            key, text, description, source, category, (num,), role="count", count_unit=unit, **kw
        )


def build_evidence_package(repo: Repository, project_id: str) -> EvidencePackage:
    """Assemble the package. Raises if the statistics artifact is not a valid basis."""
    problems = list(repo.stats_output_problems(project_id))
    stats = repo.current_stats_output(project_id)
    if stats is None:
        raise SysRevError("no verified stats_output.json: complete Stage 5 first")
    if problems:
        raise SysRevError("stats_output.json is not a valid basis: " + "; ".join(problems))
    protocol = repo.latest_protocol(project_id)
    if protocol is None or protocol.status != "approved":
        raise SysRevError("an approved protocol is required")
    prisma = repo.load_prisma(project_id)
    if prisma is None:
        raise SysRevError("no PRISMA snapshot recorded")
    problems += prisma_consistency_problems(repo, project_id, prisma)
    plan = repo.latest_analysis_plan(project_id)
    if plan is None:
        raise SysRevError("no approved analysis plan")

    b = _Builder()
    expected: list[ExpectedOutput] = []

    # ---- protocol
    src = f"protocol v{protocol.version} (approved)"
    b.add("protocol.title", protocol.review_title, "review title", src, "protocol")
    b.add("protocol.question", protocol.research_question, "research question", src, "protocol")
    b.add("protocol.population", protocol.population, "population", src, "protocol")
    b.add("protocol.intervention", protocol.intervention, "intervention", src, "protocol")
    if protocol.comparator:
        b.add("protocol.comparator", protocol.comparator, "comparator", src, "protocol")
    outcomes = "; ".join(
        f"{o.name} ({o.kind}{', ' + o.definition if o.definition else ''})"
        for o in protocol.outcomes
    )
    b.add("protocol.outcomes", outcomes, "pre-specified outcomes", src, "protocol")
    designs = _join([d.value.replace("_", " ") for d in protocol.study_designs])
    b.add("protocol.designs", designs, "eligible study designs", src, "protocol")
    b.add(
        "protocol.inclusion",
        "; ".join(protocol.inclusion_criteria),
        "inclusion criteria",
        src,
        "protocol",
    )
    b.add(
        "protocol.exclusion",
        "; ".join(protocol.exclusion_criteria),
        "exclusion criteria",
        src,
        "protocol",
    )
    if protocol.prospero_id:
        b.add(
            "protocol.prospero_id",
            protocol.prospero_id,
            "PROSPERO registration number",
            src,
            "protocol",
        )

    # ---- searches (PRISMA-S log)
    runs = repo.load_search_runs(project_id)
    searched_runs = [
        r
        for r in runs
        if r.kind in (SearchRunKind.EXECUTION, SearchRunKind.IMPORT)
        and r.status != SearchRunStatus.FAILED
    ]
    searched = sorted({r.database.value for r in searched_runs if r.database})
    labels = [
        DATABASES[SourceDatabase(d)].label for d in searched if SourceDatabase(d) in DATABASES
    ]
    dates = [r.searched_on or r.executed_at.date() for r in searched_runs]
    if searched_runs:
        b.add(
            "search.databases",
            _join(labels),
            "databases/registers searched",
            "search log",
            "search",
        )
        last = max(dates)
        b.add("search.last_date", _date(last), "date of the last search", "search log", "search")
        b.add(
            "search.statement",
            _search_statement(protocol, searched_runs, runs),
            "information sources, platforms, dates and restrictions",
            "search log",
            "search",
        )
        expected.append(
            ExpectedOutput(
                "search.statement",
                "information sources and search dates",
                ("methods",),
                ("search.statement",),
            )
        )

    # ---- screening (who screened; automated passes are never called humans)
    from src.screening.reporting import methods_statement, screening_reporting_facts

    automated: dict[str, tuple[int, int]] = {}
    for stage, key in (
        (ScreeningStage.TITLE_ABSTRACT, "screening.title_abstract"),
        (ScreeningStage.FULL_TEXT, "screening.full_text"),
    ):
        facts = screening_reporting_facts(repo, project_id, stage)
        automated[stage.value] = (facts.human_independent_reviewers, len(facts.automated_passes))
        b.add(
            key,
            methods_statement(facts),
            f"{stage.value} screening procedure",
            "screening decisions",
            "screening",
        )
        expected.append(ExpectedOutput(key, f"{stage.value} screening procedure", ("methods",)))

    # ---- extraction
    b.add(
        "extraction.methods",
        _extraction_statement(repo, project_id),
        "data extraction procedure",
        "extraction items and approved arms",
        "extraction",
    )
    expected.append(ExpectedOutput("extraction.methods", "data extraction", ("methods",)))

    # ---- risk of bias
    approved_rob = [a for a in repo.load_rob(project_id) if a.approved_by_human]
    tools = frozenset(a.tool for a in approved_rob)
    if approved_rob:
        b.add(
            "rob.methods",
            _rob_methods(approved_rob),
            "risk-of-bias tools and procedure",
            "approved risk-of-bias assessments",
            "rob",
        )
        b.add(
            "rob.summary",
            _rob_summary(approved_rob),
            "risk-of-bias judgements",
            "approved risk-of-bias assessments",
            "rob",
        )
        expected.append(ExpectedOutput("rob.methods", "risk-of-bias methods", ("methods",)))
        expected.append(ExpectedOutput("rob.summary", "risk-of-bias results", ("results",)))

    # ---- PRISMA
    psrc = "PRISMA snapshot (derived from stored records)"
    excluded_reports = sum(prisma.reports_excluded_by_reason.values())
    prisma_counts = [
        ("prisma.identified", prisma.total_identified, "records identified", "records"),
        (
            "prisma.duplicates_removed",
            prisma.duplicates_removed,
            "duplicate records removed",
            "records",
        ),
        ("prisma.records_screened", prisma.records_screened, "records screened", "records"),
        (
            "prisma.records_excluded",
            prisma.records_excluded,
            "records excluded at title/abstract",
            "records",
        ),
        ("prisma.reports_sought", prisma.reports_sought, "reports sought for retrieval", "reports"),
        (
            "prisma.reports_not_retrieved",
            prisma.reports_not_retrieved,
            "reports not retrieved",
            "reports",
        ),
        (
            "prisma.reports_assessed",
            prisma.reports_assessed,
            "reports assessed for eligibility",
            "reports",
        ),
        ("prisma.reports_excluded", excluded_reports, "reports excluded at full text", "reports"),
        ("prisma.studies_included", prisma.studies_included, "studies included", "studies"),
        (
            "prisma.reports_included",
            prisma.reports_included,
            "reports of included studies",
            "reports",
        ),
    ]
    for key, n, desc, unit in prisma_counts:
        b.count(key, n, desc, psrc, "prisma", unit)
        expected.append(ExpectedOutput(key, desc, ("results",)))
    if prisma.reports_excluded_by_reason:
        reasons = "; ".join(
            f"{r} (n = {n})"
            for r, n in sorted(
                prisma.reports_excluded_by_reason.items(), key=lambda x: (-x[1], x[0])
            )
        )
        b.add("prisma.exclusion_reasons", reasons, "full-text exclusion reasons", psrc, "prisma")
        expected.append(
            ExpectedOutput("prisma.exclusion_reasons", "reasons for exclusion", ("results",))
        )

    # ---- included studies and references
    included = sorted(repo.included_record_ids(project_id, ScreeningStage.FULL_TEXT))
    study_of = repo.study_ids(project_id, set(included))
    records = {rid: repo.get_record(project_id, rid) for rid in included}
    references: dict[str, ReferenceState] = {}
    included_ref_keys: dict[str, str] = {}
    for rid, rec in records.items():
        entry = record_reference(rec)
        included_ref_keys[rid] = entry.ref_key
        references[entry.ref_key] = ReferenceState(entry, (), provenance_basis(rec), ())
    for entry in repo.load_reference_entries(project_id):
        references[entry.ref_key] = ReferenceState(entry, (), None, ())
    verifications = repo.load_reference_verifications(project_id)
    excerpts = repo.load_evidence_excerpts(project_id)
    for key, state in list(references.items()):
        meta = state.entry.metadata_sha256()
        item = (
            repo.get_fulltext_item(project_id, state.entry.record_id)
            if state.entry.origin == "included_study_record" and state.entry.record_id
            else None
        )
        confirmed_pdf = item.pdf_sha256 if item is not None and item.has_usable_pdf else None
        latest: dict[str, ReferenceVerification] = {}
        invalid: list[str] = []
        for v in verifications:  # chronological: later lookups replace earlier ones
            if v.ref_key != key or v.metadata_sha256 != meta:
                continue
            if v.source == "human_attestation" and (
                confirmed_pdf is None or v.pdf_sha256 != confirmed_pdf
            ):
                # The attestation relied on a PDF that is no longer the report's
                # identity-confirmed PDF (replaced, changed or confirmation revoked).
                invalid.append(
                    f"attestation by {v.checked_by} ({v.checked_at.date()}) no longer valid: it "
                    f"relied on PDF {str(v.pdf_sha256)[:12]}…, the report's confirmed PDF is now "
                    f"{confirmed_pdf[:12] + '…' if confirmed_pdf else 'none'}; attest again"
                )
                continue
            prior = latest.get(v.source)
            if v.status == "unavailable" and prior is not None and prior.status != "unavailable":
                continue  # a failed lookup does not erase an earlier live result
            latest[v.source] = v
        references[key] = ReferenceState(
            state.entry,
            tuple(latest.values()),
            state.provenance_basis,
            tuple(e for e in excerpts if e.ref_key == key),
            tuple(invalid),
            None if item is None else f"{item.pdf_sha256}:{item.match_status}",
        )

    # ---- statistics
    performed: set[str] = set()
    ssrc = f"stats_output.json {repo.get_project(project_id).stats_output_sha256}"
    method_lines: list[str] = []
    templates: dict[str, str] = {}
    outcome_kind = {o.outcome_id: o.kind for o in protocol.outcomes}
    for a in stats.analyses:
        _analysis_facts(
            b, a, ssrc, expected, performed, outcome_kind.get(a.outcome_id) == "primary"
        )
        method_lines.append(
            f"For {a.outcome_name} ({_measure_label(a)}): " + " ".join(a.methods)
            if a.methods
            else f"For {a.outcome_name} ({_measure_label(a)}): {a.status_reason}."
        )
        if a.status in ("pooled", "single_study"):
            templates[f"figure:forest:{a.analysis_id}"] = _figure_template(a)
    b.add(
        "stats.methods",
        " ".join(method_lines),
        "statistical methods per analysis",
        ssrc,
        "statistics",
    )
    software = "; ".join(f"{k} {v}" for k, v in sorted(stats.software.items()))
    b.add(
        "stats.software",
        f"All statistics were computed by the project's deterministic engine "
        f"(src.stats_engine {stats.engine_version}; {software}); {stats.numeric_precision}",
        "statistical software",
        ssrc,
        "statistics",
    )
    expected.append(ExpectedOutput("stats.methods", "statistical methods", ("methods",)))
    templates["table:results_summary"] = _results_table(stats)
    templates["table:study_characteristics"] = _characteristics_table(
        b, repo, project_id, records, included_ref_keys, study_of, approved_rob
    )

    # ---- Session 7 reporting: PRISMA flow diagram, GRADE / SoF, synthesis without MA, SWiM
    reporting = build_reporting_state(
        repo,
        project_id,
        protocol=protocol,
        stats=stats,
        prisma=prisma,
        prisma_problems=problems,
        searched_sources=frozenset(searched),
        approved_rob=approved_rob,
    )
    add_reporting_outputs(
        b,
        reporting,
        protocol=protocol,
        stats=stats,
        expected=expected,
        templates=templates,
        included_ref_keys=included_ref_keys,
        study_of=study_of,
        performed=performed,
    )

    package = EvidencePackage(
        project_id=project_id,
        review_design=protocol.review_design,
        protocol=protocol,
        stats=stats,
        stats_sha256=stats.sha256(),
        facts=b.facts,
        templates=templates,
        references=references,
        included_ref_keys=included_ref_keys,
        study_of=study_of,
        searched_databases=frozenset(searched),
        rob_tools=tools,
        performed_methods=frozenset(performed),
        automated_screening=automated,
        expected=expected,
        problems=problems,
        prisma=prisma,
        reporting=reporting,
    )
    package.fingerprint = _fingerprint(
        repo, project_id, package, plan.sha256(), prisma, runs, approved_rob
    )
    return package


PRISMA_FIELDS = (
    "identified_by_database",
    "identified_from_registers",
    "duplicates_removed",
    "marked_ineligible_by_automation",
    "removed_other_reasons",
    "records_screened",
    "records_excluded",
    "records_awaiting_screening",
    "reports_sought",
    "reports_not_retrieved",
    "reports_awaiting_retrieval",
    "reports_assessed",
    "reports_excluded_by_reason",
    "reports_awaiting_assessment",
    "reports_included",
    "studies_included",
)


def prisma_consistency_problems(
    repo: Repository, project_id: str, stored: PRISMACounts
) -> list[str]:
    """Compare the stored PRISMA snapshot with counts derived independently from the stored
    records, decisions, manifest and study links. Differences are reported, never repaired:
    the manuscript keeps showing the stored values and the audit blocks."""
    from src.search.prisma_sync import derive_prisma

    try:
        derived = derive_prisma(repo, project_id)
    except Exception as exc:  # inconsistent records: report, do not guess
        return [f"PRISMA cannot be derived from the stored records: {exc}"]
    problems = [f"PRISMA: {v}" for v in stored.invariant_violations()]
    if not stored.is_final:
        problems.append("PRISMA: the flow is not final (records/reports still awaiting a decision)")
    # Studies: exactly the number of distinct studies documented by authenticated
    # report-to-study links (an unlinked report is its own study). A cohort-overlap decision
    # says that participants overlap, not that two reports describe one study, so it never
    # changes this count; reports of one study must be linked (``link_reports_to_study``).
    included = repo.included_record_ids(project_id, ScreeningStage.FULL_TEXT)
    studies = repo.study_ids(project_id, set(included))
    exact = len(set(studies.values()))
    if stored.studies_included != exact:
        linked = sorted({sid for rid, sid in studies.items() if sid != rid})
        problems.append(
            f"PRISMA studies_included is {stored.studies_included} in the stored snapshot but "
            f"the documented report-to-study links give exactly {exact} distinct studies from "
            f"{len(included)} included reports (linked studies: {linked or 'none'}); link the "
            "reports of one study, or correct the snapshot"
        )
    for name in PRISMA_FIELDS:
        if name == "studies_included":
            continue
        mine, theirs = getattr(stored, name), getattr(derived, name)
        if isinstance(mine, dict):
            mine = {str(k): v for k, v in mine.items() if v}
            theirs = {str(k): v for k, v in theirs.items() if v}
        if mine != theirs:
            problems.append(
                f"PRISMA {name} is {mine} in the stored snapshot but the stored records imply "
                f"{theirs}"
            )
    return problems


def _date(d: date) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _search_statement(
    protocol: PICOProtocol, searched_runs: list[SearchRun], runs: list[SearchRun]
) -> str:
    from src.search.reporting import APPLIED_TEXT

    parts: list[str] = []
    for s in protocol.search_strategies:
        mine = [r for r in searched_runs if r.strategy_sha256 == s.sha256]
        info = DATABASES.get(s.database)
        label = info.label if info else s.database.value
        if mine:
            when = max(r.searched_on or r.executed_at.date() for r in mine)
            how = (
                "searched through its API"
                if any(r.kind == SearchRunKind.EXECUTION for r in mine)
                else "searched with records imported from an export file"
            )
            platform = f" via {s.platform}" if s.platform else ""
            limits = []
            if s.date_from or s.date_to:
                limits.append(
                    f"publication dates {s.date_from or 'inception'} to "
                    f"{s.date_to or 'the search date'}"
                )
            if s.languages:
                limits.append("languages: " + ", ".join(s.languages))
            if s.filters:
                limits.append("filters: " + "; ".join(s.filters))
            applied = {APPLIED_TEXT.get(r.restrictions_applied, "not recorded") for r in mine}
            restr = (
                f" Restrictions ({'; '.join(limits)}) were {'; '.join(sorted(applied))}."
                if limits
                else " No date, language or filter restrictions were defined."
            )
            parts.append(f"{label}{platform} was {how} on {_date(when)}.{restr}")
        else:
            declared = [
                r
                for r in runs
                if r.kind == SearchRunKind.NOT_SEARCHED and r.strategy_sha256 == s.sha256
            ]
            reason = (
                f" (recorded reason: {declared[-1].note})" if declared and declared[-1].note else ""
            )
            parts.append(f"{label} was selected in the protocol but not searched{reason}.")
    return " ".join(parts)


def _extraction_statement(repo: Repository, project_id: str) -> str:
    items = repo.load_extraction_items(project_id)
    arms = repo.load_extracted_arms(project_id)
    approved = sum(1 for a in arms if a.approved_by_human)
    text = (
        "Each extracted number was recorded as printed, together with a verbatim quotation "
        "and the page of the source PDF on which it appears."
    )
    if items:
        by = Counter(i.proposed_by for i in items)
        accepted = sum(1 for i in items if i.review_status == ReviewStatus.ACCEPTED)
        ai = by.get(ActorType.AGENT, 0)
        human = by.get(ActorType.HUMAN, 0)
        text += (
            f" Of {len(items)} extraction items, {ai} were proposed by a large language model "
            f"and {human} by human reviewers; {accepted} were accepted by an authenticated "
            "human reviewer, and no automated proposal entered the analyses without that "
            "acceptance."
        )
    text += (
        f" {approved} study arm(s) used in the analyses were approved by an authenticated "
        "human reviewer. Extraction was not performed in duplicate by two independent human "
        "reviewers."
    )
    return text


def _rob_methods(assessments: list[RiskOfBiasAssessment]) -> str:
    tools = Counter(a.tool for a in assessments)
    parts = []
    for tool, n in sorted(tools.items(), key=lambda x: x[0].value):
        name = ROB_LABEL[tool]
        if tool == RoBTool.EXTERNAL:
            named = sorted(
                {a.tool_rationale for a in assessments if a.tool == tool and a.tool_rationale}
            )
            if named:
                name += f" (as recorded: {'; '.join(named)})"
        parts.append(f"{name} ({n} assessment(s))")
    drafted = Counter(a.assessed_by for a in assessments)
    ai = drafted.get(ActorType.AGENT, 0)
    drafted_note = (
        f"{ai} assessment(s) were first drafted by a large language model; every assessment"
        if ai
        else "Every assessment"
    )
    return (
        f"Risk of bias was assessed with {_join(parts)}. {drafted_note}, including the overall "
        "judgement, was made or approved by an authenticated human reviewer."
    )


def _rob_summary(assessments: list[RiskOfBiasAssessment]) -> str:
    parts = []
    for tool in sorted({a.tool for a in assessments}, key=lambda t: t.value):
        overall = Counter(a.overall or "not stated" for a in assessments if a.tool == tool)
        detail = _join([f"{k} in {v}" for k, v in sorted(overall.items())])
        parts.append(
            f"with {ROB_LABEL[tool]}, overall risk of bias was judged {detail} assessment(s)"
        )
    text = "; ".join(parts)
    return text[0].upper() + text[1:]


def _analysis_facts(
    b: _Builder,
    a: AnalysisResult,
    src: str,
    expected: list[ExpectedOutput],
    performed: set[str],
    primary_outcome: bool,
) -> None:
    p = f"stats.{a.analysis_id}"
    common: dict[str, Any] = {
        "analysis_id": a.analysis_id,
        "measure": a.effect_measure.value,
        "estimator": ESTIMATOR.get(a.effect_measure),
    }
    # Session 9 review: counts shown with the pooled estimate describe the primary model.
    # When every reported model uses every listed study (the usual case, and every engine
    # 1.0.0 output) the facts and their wording are unchanged.
    pop = a.population()
    model_specific = model_specific_populations(a)
    rids = pop.record_ids
    primary_label = a.primary.label if a.primary is not None else None
    of_primary = f" of the primary {primary_label} model" if model_specific else ""
    b.add(f"{p}.outcome", a.outcome_name, "outcome name", src, "statistics", **common)
    b.add(
        f"{p}.measure_label",
        _measure_label(a),
        "summary measure and timepoint",
        src,
        "statistics",
        **common,
    )
    b.count(
        f"{p}.k",
        pop.k,
        f"number of studies contributing to the analysis{of_primary}",
        src,
        "statistics",
        "studies",
        record_ids=rids,
        **common,
    )
    if pop.total_participants is not None:
        b.count(
            f"{p}.participants",
            pop.total_participants,
            f"participants in contributing studies{of_primary}",
            src,
            "statistics",
            "participants",
            **common,
        )
    if pop.total_events is not None:
        b.count(
            f"{p}.events",
            pop.total_events,
            f"events in contributing studies{of_primary}",
            src,
            "statistics",
            "events",
            **common,
        )
    for aux_model, aux in a.auxiliary_populations():  # labelled, never as the primary's
        key = _model_key(a, aux_model)
        what = f"auxiliary {aux_model.label} model"
        aux_rids = aux.record_ids
        b.count(f"{p}.{key}.k", aux.k, f"number of studies in the {what}", src, "statistics",
                "studies", record_ids=aux_rids, model=aux_model.model, **common)  # fmt: skip
        if aux.total_participants is not None:
            b.count(f"{p}.{key}.participants", aux.total_participants,
                    f"participants in the {what}", src, "statistics", "participants",
                    model=aux_model.model, **common)  # fmt: skip
        if aux.total_events is not None:
            b.count(f"{p}.{key}.events", aux.total_events, f"events in the {what}", src,
                    "statistics", "events", model=aux_model.model, **common)  # fmt: skip
    status_text = _status_text(a)
    b.add(
        f"{p}.status",
        status_text,
        "what was (and was not) estimated",
        src,
        "statistics",
        role="status",
        **common,
    )
    null = _null(a)
    comparative = a.effect_measure in RATIO | DIFFERENCE
    main_key = f"{p}.status"
    if a.status == "pooled" and a.primary is not None:
        r = a.primary
        text, nums = _estimate_text(
            a, r.estimate_natural, r.ci_lower_natural, r.ci_upper_natural, "pooled_estimate"
        )
        b.add(
            f"{p}.pooled_estimate",
            text,
            f"pooled {_measure_label(a)} (primary {_model_label(r)})",
            src + f" analyses[{a.analysis_id}].primary",
            "statistics",
            nums,
            role="pooled_estimate",
            pooled=True,
            model=r.model,
            null_value=null,
            ci=(r.ci_lower_natural, r.ci_upper_natural),
            record_ids=rids,
            **common,
        )
        b.add(
            f"{p}.model",
            _model_label(r),
            "primary model",
            src,
            "statistics",
            model=r.model,
            **common,
        )
        main_key = f"{p}.pooled_estimate"
        for name in ("common_effect", "random_effects"):
            res: PooledResult | None = getattr(a, name)
            if res is None:
                continue
            performed.add(name)
            t, n = _estimate_text(
                a,
                res.estimate_natural,
                res.ci_lower_natural,
                res.ci_upper_natural,
                "pooled_estimate",
            )
            label = _model_label(res)
            b.add(
                f"{p}.{name}",
                f"{t}; {label}",
                f"{name.replace('_', '-')} estimate",
                src + f" analyses[{a.analysis_id}].{name}",
                "statistics",
                n + numbers_in_text(label),
                role="pooled_estimate",
                pooled=True,
                model=res.model,
                null_value=null,
                ci=(res.ci_lower_natural, res.ci_upper_natural),
                record_ids=a.population(res).record_ids,
                **common,
            )
            if comparative:
                pt, pn = fmt_p(res.p_value)
                b.add(
                    f"{p}.{name}.p_value",
                    pt,
                    f"p-value ({name.replace('_', '-')} model; test of no effect)",
                    src,
                    "statistics",
                    (pn,),
                    role="p_value",
                    model=res.model,
                    **common,
                )
        if comparative:
            pt, pn = fmt_p(r.p_value)
            b.add(
                f"{p}.p_value",
                pt,
                "p-value of the primary model (test of no effect)",
                src,
                "statistics",
                (pn,),
                role="p_value",
                model=r.model,
                **common,
            )
        for i, sres in enumerate(a.sensitivity):
            performed.add("sensitivity")
            t, n = _estimate_text(
                a,
                sres.estimate_natural,
                sres.ci_lower_natural,
                sres.ci_upper_natural,
                "sensitivity_estimate",
            )
            label = _model_label(sres)
            b.add(
                f"{p}.sensitivity.{i + 1}",
                f"{t}; {label}",
                "sensitivity analysis estimate",
                src,
                "statistics",
                n + numbers_in_text(label),
                role="sensitivity_estimate",
                pooled=True,
                model=sres.model,
                null_value=null,
                ci=(sres.ci_lower_natural, sres.ci_upper_natural),
                **common,
            )
        re_ = a.random_effects
        if re_ is not None and re_.prediction_interval_natural is not None:
            performed.add("prediction_interval")
            lo, hi = re_.prediction_interval_natural
            pct = a.effect_measure in PROBABILITY
            d, m, u = (1, 100.0, "%") if pct else (_decimals(a), 1.0, "")
            lvl = _ci_level(a)
            b.add(
                f"{p}.prediction_interval",
                f"{fmt(lvl, 0)}% prediction interval ranged from {fmt(lo * m, d)}{u} to "
                f"{fmt(hi * m, d)}{u}",
                "prediction interval of the random-effects model",
                src,
                "statistics",
                (
                    DisplayNumber("ci_level", lvl, 0),
                    DisplayNumber("pi_lower", lo * m, d),
                    DisplayNumber("pi_upper", hi * m, d),
                ),
                role="prediction_interval",
                model="random_effects",
                **common,
            )
        if re_ is not None and re_.tau2 is not None:
            tau = f"τ² = {fmt(re_.tau2, 4)}"
            tnums = [DisplayNumber("tau2", re_.tau2, 4)]
            if re_.tau2_ci is not None:
                lvl = _ci_level(a)
                tau += f" ({fmt(lvl, 0)}% CI {fmt(re_.tau2_ci[0], 4)} to {fmt(re_.tau2_ci[1], 4)})"
                tnums += [
                    DisplayNumber("ci_level", lvl, 0),
                    DisplayNumber("ci_lower", re_.tau2_ci[0], 4),
                    DisplayNumber("ci_upper", re_.tau2_ci[1], 4),
                ]
            b.add(
                f"{p}.tau2",
                tau,
                "between-study variance τ²",
                src,
                "statistics",
                tuple(tnums),
                role="tau2",
                **common,
            )
    elif a.status == "single_study" and a.studies:
        s = a.studies[0]
        if s.estimate_natural is not None:
            text, nums = _estimate_text(
                a,
                s.estimate_natural,
                s.ci_lower_natural,
                s.ci_upper_natural,
                "single_study_estimate",
            )
            b.add(
                f"{p}.single_study_estimate",
                text,
                f"{_measure_label(a)} from the only "
                f"contributing study ({s.label}); not a pooled estimate",
                src,
                "statistics",
                nums,
                role="single_study_estimate",
                pooled=False,
                null_value=null,
                ci=None
                if s.ci_lower_natural is None or s.ci_upper_natural is None
                else (s.ci_lower_natural, s.ci_upper_natural),
                record_ids=tuple(s.record_ids),
                **common,
            )
            main_key = f"{p}.single_study_estimate"
    if a.heterogeneity is not None:
        h = a.heterogeneity
        i2 = DisplayNumber("i2", h.i2, 0)
        b.add(
            f"{p}.i2",
            f"I² = {fmt(h.i2, 0)}%",
            "I² heterogeneity statistic",
            src,
            "statistics",
            (i2,),
            role="i2",
            **common,
        )
        pt, pn = fmt_p(h.p_value)
        d = _decimals(a)
        b.add(
            f"{p}.heterogeneity",
            f"Q = {fmt(h.q, d)} (df = {h.df}; {pt}); I² = {fmt(h.i2, 0)}%",
            f"heterogeneity ({h.basis})",
            src,
            "statistics",
            (DisplayNumber("q", h.q, d), DisplayNumber("df", float(h.df), 0), pn, i2),
            role="heterogeneity",
            **common,
        )
    for s in a.studies:
        if s.estimate_natural is None:
            continue
        text, nums = _estimate_text(
            a, s.estimate_natural, s.ci_lower_natural, s.ci_upper_natural, "study_estimate"
        )
        b.add(
            f"{p}.study.{s.study_id}",
            text,
            f"{_measure_label(a)} reported for {s.label} (single study)",
            src,
            "statistics",
            nums,
            role="study_estimate",
            pooled=False,
            null_value=null,
            record_ids=tuple(s.record_ids),
            **common,
        )
        b.add(
            f"{p}.study.{s.study_id}.label",
            s.label,
            "study label",
            src,
            "statistics",
            record_ids=tuple(s.record_ids),
            **common,
        )
    if a.excluded:
        groups: dict[str, list[str]] = {}
        for e in a.excluded:
            groups.setdefault(
                EXCLUSION_TEXT.get(e.reason_code, e.reason_code.replace("_", " ")), []
            ).append(e.label)
        ex = "; ".join(f"{_join(labels)} ({why})" for why, labels in groups.items())
        b.add(
            f"{p}.excluded",
            ex,
            "studies not contributing, with reasons",
            src,
            "statistics",
            **common,
        )
    caveats = [*a.warnings, *a.limitations]
    if caveats:
        b.add(
            f"{p}.limitations",
            f"For {a.outcome_name}: "
            + " ".join(c if c.endswith(".") else c + "." for c in caveats),
            "warnings and limitations reported by the statistics engine",
            src,
            "statistics",
            **common,
        )
        expected.append(
            ExpectedOutput(
                f"{p}.limitations",
                f"engine warnings/limitations for {a.outcome_name}",
                ("limitations", "discussion", "results"),
            )
        )
    # Coverage: the main result (or the statement of why there is none) must be reported.
    expected.append(ExpectedOutput(main_key, f"main result for {a.outcome_name}", ("results",)))
    if a.status == "pooled":
        expected.append(
            ExpectedOutput(f"{p}.k", f"number of studies for {a.outcome_name}", ("results",))
        )
        if a.heterogeneity is not None:
            expected.append(
                ExpectedOutput(
                    f"{p}.i2",
                    f"heterogeneity for {a.outcome_name}",
                    ("results",),
                    (f"{p}.i2", f"{p}.heterogeneity"),
                )
            )
    if primary_outcome:
        expected.append(
            ExpectedOutput(
                main_key, f"primary outcome {a.outcome_name} in the abstract", ("abstract",)
            )
        )
    if a.status in ("pooled", "single_study"):
        expected.append(
            ExpectedOutput(
                f"figure:forest:{a.analysis_id}",
                f"forest plot legend for {a.outcome_name}",
                ("figures",),
            )
        )


def model_specific_populations(a: AnalysisResult) -> bool:
    """Whether the reported models of ``a`` use different studies or totals (Session 9
    review), so that every count must name its model."""
    return bool(a.auxiliary_populations()) or a.population().k != a.k


def _model_key(a: AnalysisResult, r: PooledResult) -> str:
    """Fact-key segment of a non-primary model: ``common_effect``/``random_effects`` or
    ``sensitivity.<i>`` (as for the estimate facts)."""
    for name in ("common_effect", "random_effects"):
        if getattr(a, name) == r:
            return name
    return f"sensitivity.{a.sensitivity.index(r) + 1}"


def _status_text(a: AnalysisResult) -> str:
    what = f"{a.outcome_name} ({_measure_label(a)})"
    reason = a.status_reason.rstrip(".")
    if a.status == "pooled" and a.primary is not None and model_specific_populations(a):
        pop = a.population()
        n = "study" if pop.k == 1 else "studies"
        text = f"{what}: {pop.k} {n} contributed to the pooled estimate of the primary "
        text += f"{a.primary.label} model."
        for r, aux in a.auxiliary_populations():
            text += f" The auxiliary {r.label} model used {aux.k} studies."
        return text
    if a.status == "pooled":
        n = "study" if a.k == 1 else "studies"
        return f"{what}: {a.k} {n} contributed to a pooled estimate."
    if a.status == "single_study":
        return f"{what}: only one study contributed data, so no pooled estimate was calculated."
    if a.status == "no_data":
        return f"{what}: no eligible study provided usable data; no estimate was calculated."
    if a.status == "unsupported":
        return (
            f"{what}: this analysis is not supported by the statistics engine and was not "
            f"performed ({reason})."
        )
    if a.status == "not_estimable":
        return f"{what}: the primary model could not be estimated ({reason})."
    return f"{what}: the data could not be pooled defensibly ({reason})."


def _figure_template(a: AnalysisResult) -> str:
    p = f"stats.{a.analysis_id}"
    if a.status == "pooled" and model_specific_populations(a):
        aux = "; ".join(
            f"the {r.label} model used {{{{{p}.{_model_key(a, r)}.k}}}} studies"
            for r, _ in a.auxiliary_populations()
        )
        return (
            f"Forest plot of {{{{{p}.outcome}}}} ({{{{{p}.measure_label}}}}): the primary "
            f"{{{{{p}.model}}}} used {{{{{p}.k}}}} studies" + (f"; {aux}" if aux else "") + ". "
            "Squares show study estimates (size proportional to the primary model's weight) "
            "with confidence intervals; a row without a square is a study without a "
            "study-level estimate that contributes only to the model named on that row; the "
            "diamonds show the pooled estimates. "
            f"File: plots/{a.analysis_id}.svg (stats artifact)."
        )
    if a.status == "pooled":
        return (
            f"Forest plot of {{{{{p}.outcome}}}} ({{{{{p}.measure_label}}}}), "
            f"{{{{{p}.k}}}} studies. Squares show study estimates (size proportional to "
            f"weight) with confidence intervals; the diamond shows the pooled estimate from the "
            f"{{{{{p}.model}}}}. File: plots/{a.analysis_id}.svg (stats artifact)."
        )
    return (
        f"Forest plot of {{{{{p}.outcome}}}} ({{{{{p}.measure_label}}}}): estimate of the single "
        f"contributing study; no pooled estimate. File: plots/{a.analysis_id}.svg "
        "(stats artifact)."
    )


def _results_table(stats: StatsOutput) -> str:
    rows = [
        "| Outcome | Measure | Studies | Estimate | Heterogeneity | Status |",
        "|---|---|---|---|---|---|",
    ]
    for a in stats.analyses:
        p = f"stats.{a.analysis_id}"
        if a.status == "pooled":
            est = f"{{{{{p}.pooled_estimate}}}}"
        elif a.status == "single_study" and a.studies and a.studies[0].estimate_natural is not None:
            est = f"{{{{{p}.single_study_estimate}}}} (single study)"
        else:
            est = "not estimated"
        het = f"{{{{{p}.i2}}}}" if a.heterogeneity is not None else "not applicable"
        rows.append(
            f"| {{{{{p}.outcome}}}} | {{{{{p}.measure_label}}}} | {{{{{p}.k}}}} | {est} "
            f"| {het} | {a.status.replace('_', ' ')} |"
        )
    return "\n".join(rows)


def _characteristics_table(
    b: _Builder,
    repo: Repository,
    project_id: str,
    records: dict[str, StudyRecord],
    ref_keys: dict[str, str],
    study_of: dict[str, str],
    rob: list[RiskOfBiasAssessment],
) -> str:
    arms = [a for a in repo.load_extracted_arms(project_id) if a.approved_by_human]
    rob_of = {a.record_id: a for a in rob}
    rows = [
        "| Study (reports) | Outcome data extracted | Arm: n analysed (follow-up) | "
        "Risk of bias (overall) |",
        "|---|---|---|---|",
    ]
    by_study: dict[str, list[str]] = {}
    for rid in sorted(records):
        by_study.setdefault(study_of.get(rid, rid), []).append(rid)
    for sid, rids in sorted(by_study.items()):
        label_key = f"study.{sid}.label"
        b.add(
            label_key,
            study_label(records[rids[0]]),
            "first author and year of the study's first report",
            "stored bibliographic record",
            "study",
            record_ids=tuple(rids),
        )
        cites = "; ".join(f"@{ref_keys[r]}" for r in rids)
        outcomes, cells = set(), []
        for arm in sorted(
            (x for x in arms if x.record_id in rids),
            key=lambda x: (x.outcome_id, x.arm_label, x.arm_id),
        ):
            outcomes.add(arm.outcome_id)
            if arm.n_analyzed is None:
                continue
            key = f"extraction.{arm.arm_id}.n_analyzed"
            if key not in b.facts:
                v = arm.n_analyzed
                b.add(
                    key,
                    v.reported_text,
                    f"n analysed, {arm.arm_label} ({arm.outcome_id}), as printed in the source",
                    f"approved arm {arm.arm_id} (page "
                    f"{v.evidence.page_number}: {v.evidence.verbatim_quote[:80]!r})",
                    "extraction",
                    record_ids=(arm.record_id,),
                    count_unit="participants",
                    role="count",
                )
            arm_key = f"extraction.{arm.arm_id}.arm"
            b.add(
                arm_key,
                f"{arm.arm_label} ({arm.outcome_id})",
                "arm label and outcome",
                f"approved arm {arm.arm_id}",
                "extraction",
                record_ids=(arm.record_id,),
            )
            cell = f"{{{{{arm_key}}}}}: {{{{{key}}}}}"
            if arm.follow_up is not None:
                fu_key = f"extraction.{arm.arm_id}.follow_up"
                text, numbers = _follow_up(arm.follow_up)
                b.add(
                    fu_key,
                    text,
                    f"follow-up timepoint of {arm.arm_label} ({arm.outcome_id}), "
                    f"structured ({arm.follow_up.kind.value})",
                    f"approved arm {arm.arm_id}: structured follow-up timepoint",
                    "extraction",
                    numbers,
                    role="follow_up",
                    record_ids=(arm.record_id,),
                )
                cell += f", {{{{{fu_key}}}}}"
            cells.append(cell)
        judgements = [rob_of[r] for r in rids if r in rob_of]
        rob_text = (
            "; ".join(f"{ROB_LABEL[j.tool]}: {j.overall or 'not stated'}" for j in judgements)
            or "not assessed"
        )
        b.add(
            f"study.{sid}.outcomes",
            ", ".join(sorted(outcomes)) or "none",
            "outcomes with approved extracted data",
            "approved arms",
            "study",
            record_ids=tuple(rids),
        )
        b.add(
            f"study.{sid}.rob",
            rob_text,
            "overall risk-of-bias judgement(s)",
            "approved risk-of-bias assessments",
            "study",
            record_ids=tuple(rids),
        )
        rows.append(
            f"| {{{{{label_key}}}}} [{cites}] | {{{{study.{sid}.outcomes}}}} "
            f"| {'<br>'.join(cells) or 'no arm-level data'} | {{{{study.{sid}.rob}}}} |"
        )
    return "\n".join(rows)


def _num(v: float) -> tuple[str, DisplayNumber]:
    text = f"{v:g}"
    decimals = len(text.split(".")[1]) if "." in text else 0
    return text, DisplayNumber("follow_up", v, decimals)


def _follow_up(tp: FollowUpTimepoint) -> tuple[str, tuple[DisplayNumber, ...]]:
    """A structured follow-up timepoint as text, with its source numbers and unit (no
    conversion between units: "24 months" is never shown as "2 years")."""
    if tp.kind == TimepointType.FIXED and tp.value is not None:
        t, n = _num(tp.value)
        return f"{t} {tp.unit}", (n,)
    if tp.kind == TimepointType.WINDOW and tp.window_min is not None and tp.window_max is not None:
        lo, nlo = _num(tp.window_min)
        hi, nhi = _num(tp.window_max)
        return f"{lo} to {hi} {tp.unit}", (nlo, nhi)
    return "last follow-up", ()


def _fingerprint(
    repo: Repository,
    project_id: str,
    package: EvidencePackage,
    plan_sha: str,
    prisma: object,
    runs: Iterable[object],
    rob: list[RiskOfBiasAssessment],
) -> dict[str, str]:
    from src.core.schemas.pipeline import PRISMACounts

    assert isinstance(prisma, PRISMACounts)
    refs = {
        k: {
            "meta": s.entry.metadata_sha256(),
            "verifications": [v.verification_id for v in s.verifications],
            "basis": s.provenance_basis,
            "excerpts": [e.excerpt_id for e in s.excerpts],
            # a replaced PDF or a revoked identity confirmation changes the evidence state
            "pdf_identity": s.pdf_identity,
        }
        for k, s in sorted(package.references.items())
    }
    return {
        "stats_output_sha256": package.stats_sha256,
        "analysis_plan_sha256": plan_sha,
        "extraction_input_sha256": repo.load_stage5_inputs(project_id).sha256(),
        "protocol_sha256": canonical_sha256(package.protocol.model_dump(mode="json")),
        "prisma_sha256": canonical_sha256(prisma.model_dump(mode="json")),
        "search_log_sha256": canonical_sha256(
            [r.model_dump(mode="json") for r in runs]  # type: ignore[attr-defined]
        ),
        "rob_sha256": canonical_sha256([a.model_dump(mode="json") for a in rob]),
        "references_sha256": canonical_sha256(refs),
        "facts_sha256": canonical_sha256({k: f.sha256() for k, f in sorted(package.facts.items())}),
        "templates_sha256": _sha_text(json.dumps(package.templates, sort_keys=True)),
        # Session 7
        "reporting_sha256": package.reporting.content_sha256 if package.reporting else "",
        "reporting_decisions_sha256": (
            package.reporting.decisions_sha256 if package.reporting else ""
        ),
    }


# Fingerprint components that never change the rendered manuscript (reporting-checklist
# decisions). A manuscript version is not made stale by them, but an audit or sign-off made
# before they changed is (the audit records the full fingerprint).
AUDIT_ONLY_FINGERPRINT_KEYS = frozenset({"reporting_decisions_sha256"})


def content_fingerprint(fingerprint: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in fingerprint.items() if k not in AUDIT_ONLY_FINGERPRINT_KEYS}


def screening_is_automated(package: EvidencePackage) -> bool:
    return any(humans < 2 and auto for humans, auto in package.automated_screening.values())


def comparative(package: EvidencePackage) -> bool:
    return package.review_design == ReviewDesign.COMPARATIVE


__all__ = [
    "AUDIT_ONLY_FINGERPRINT_KEYS",
    "NEVER_PERFORMED",
    "EvidencePackage",
    "ExpectedOutput",
    "Fact",
    "ReferenceState",
    "build_evidence_package",
]
