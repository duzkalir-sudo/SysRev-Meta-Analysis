"""Stage 6 Auditor: deterministic, completeness-aware checks of a persisted manuscript.

The Auditor re-renders the stored manuscript version (with the values frozen into it) against
the *current* evidence package and checks it scope by scope (see
``src.core.schemas.audit.REQUIRED_SCOPES``). It never asks a language model whether the
manuscript is correct.

What it can establish deterministically:

* every number inserted from the evidence package still equals its approved source value
  and is a correct rounding of it (explicit tolerance per field);
* every expected reportable output (PRISMA counts, each analysis' main result or the reason
  there is none, heterogeneity, engine warnings, methods facts, tables, figures, citations
  of included studies) appears in the required section;
* citations resolve to registry entries, are numbered in order of first appearance, are not
  duplicated, and have a live or provenance-based identity check;
* typed numbers, labels and method descriptions that contradict the evidence (wrong
  estimator, pooled vs single study, model, scale, significance, record/report/study counts,
  unperformed analyses, invented registration or databases, human-reviewer claims).

What it cannot establish, and therefore leaves as ``REQUIRES_HUMAN_REVIEW`` or reports as
``UNSUPPORTED`` rather than verifying: the meaning of arbitrary prose, whether a cited
background paper supports a sentence (only a lexical check against human-entered excerpts
is made), and the clinical soundness of interpretations. Sentence segmentation and cue
detection are heuristic; their limitations are listed in every report.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass, field

from src.core.enums import CheckStatus, ReviewDesign, RoBTool, SourceDatabase
from src.core.identifiers import normalize_title
from src.core.schemas.audit import (
    AuditFinding,
    AuditReport,
    CitationCheck,
    CoverageItem,
    NumericClaimCheck,
)
from src.core.schemas.manuscript import REQUIRED_SECTIONS, ManuscriptVersion
from src.manuscript.document import AUTHOR_INPUT_RE, Rendered, RenderedSection, Span, render
from src.manuscript.drafting import ALLOWED_LITERALS, AUTHOR_YEAR_RE, MANUAL_CITATION_RE
from src.manuscript.evidence import (
    EvidencePackage,
    Fact,
    screening_is_automated,
)
from src.manuscript.formatting import DisplayNumber, FoundNumber, find_numbers, parse_number

AUDITOR_VERSION = "stage6-auditor-1.0.0"
ESTIMATE_ROLES = frozenset(
    {"pooled_estimate", "single_study_estimate", "study_estimate", "sensitivity_estimate"}
)
INTERPRETIVE = frozenset({"abstract", "introduction", "discussion", "limitations", "conclusions"})
NEGATION = re.compile(
    r"\b(not|no|neither|nor|without|cannot|could not|were not|was not|did not|unable|none)\b",
    re.IGNORECASE,
)
LIMITATIONS = [
    "Prose claims are segmented into sentences and matched with fixed cue lists; the Auditor "
    "does not understand arbitrary prose. Unmatched or ambiguous statements are reported for "
    "human review, never verified.",
    "Whether a cited background reference supports a sentence is checked only lexically "
    "against excerpts entered by a human (numbers present, content-word overlap); semantic "
    "support always requires human confirmation.",
    "A typed number is verified only through a {{key}} binding. Typed numbers that happen to "
    "equal an approved value are reported for human review, not verified.",
    "Clinical interpretation, plausibility of mechanisms and appropriateness of "
    "recommendations are not assessed automatically; cue words route them to human review.",
]

DATABASE_CUES: list[tuple[re.Pattern[str], SourceDatabase | None]] = [
    (re.compile(r"\b(PubMed|MEDLINE)\b"), SourceDatabase.PUBMED),
    (re.compile(r"\bEmbase\b", re.I), SourceDatabase.EMBASE),
    (re.compile(r"\bScopus\b", re.I), SourceDatabase.SCOPUS),
    (re.compile(r"\bWeb of Science\b", re.I), SourceDatabase.WEB_OF_SCIENCE),
    (
        re.compile(r"\b(CENTRAL|Cochrane (Library|Central Register))\b"),
        SourceDatabase.COCHRANE_CENTRAL,
    ),
    (re.compile(r"\bEurope PMC\b", re.I), SourceDatabase.EUROPE_PMC),
    (re.compile(r"\bOpenAlex\b", re.I), SourceDatabase.OPENALEX),
    (re.compile(r"\bClinicalTrials\.gov\b", re.I), SourceDatabase.CLINICALTRIALS_GOV),
    (re.compile(r"\b(Google Scholar|CINAHL|PsycINFO|LILACS|Web of Knowledge)\b", re.I), None),
]
METHOD_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("meta_regression", re.compile(r"meta-?regression", re.I)),
    ("subgroup", re.compile(r"\bsubgroups?\b", re.I)),
    (
        "publication_bias_test",
        re.compile(r"\bEgger|\bBegg|funnel plot|small-study effects?|publication bias", re.I),
    ),
    ("trim_and_fill", re.compile(r"trim[- ]and[- ]fill", re.I)),
    ("leave_one_out", re.compile(r"leave[- ]one[- ]out|influence analys", re.I)),
    ("bayesian", re.compile(r"\bbayesian\b", re.I)),
    ("network", re.compile(r"network meta", re.I)),
    ("grade", re.compile(r"\bGRADE\b|certainty of (the )?evidence")),
    (
        "mean_sd_conversion",
        re.compile(
            r"\bHozo\b|\bWan et al|\bLuo et al|estimated (the )?means?|convert\w* (the )?medians?",
            re.I,
        ),
    ),
    ("hazard_ratio_pooling", re.compile(r"hazard ratios?", re.I)),
    (
        "ipd_reconstruction",
        re.compile(r"individual (patient|participant) data|reconstruct\w* .*curves?", re.I),
    ),
    ("sensitivity", re.compile(r"sensitivity analys", re.I)),
    ("random_effects", re.compile(r"random[- ]effects?", re.I)),
    ("common_effect", re.compile(r"(common|fixed)[- ]effects?", re.I)),
    ("prediction_interval", re.compile(r"prediction interval", re.I)),
    ("reml", re.compile(r"\bREML\b|restricted maximum", re.I)),
    ("dl", re.compile(r"DerSimonian|\bDL\b")),
    ("hksj", re.compile(r"Hartung|Knapp|\bHKSJ\b")),
    ("mantel_haenszel", re.compile(r"Mantel", re.I)),
    ("glmm", re.compile(r"\bGLMM\b|generali[sz]ed linear mixed", re.I)),
]
ROB_CUES: list[tuple[re.Pattern[str], RoBTool | str]] = [
    (re.compile(r"\bRoB[\s-]?2\b|risk of bias 2", re.I), RoBTool.ROB2),
    (re.compile(r"\bROBINS-I\b", re.I), RoBTool.ROBINS_I),
    (re.compile(r"Newcastle", re.I), RoBTool.NOS),
    (re.compile(r"\bMINORS\b"), "MINORS"),
    (re.compile(r"\bJBI\b|Joanna Briggs", re.I), "JBI"),
]
REVIEWERS_RE = re.compile(
    r"\b(two|2|both|independent(ly)?|pairs? of|duplicate)\b[^.]{0,60}\b(human\s+)?"
    r"(reviewers?|investigators?|authors?|researchers?|raters?|screeners?|assessors?)\b",
    re.IGNORECASE,
)
AUTOMATION_RE = re.compile(r"\b(automated|language model|LLM|artificial intelligence|AI)\b", re.I)
REGISTRATION_RE = re.compile(
    r"PROSPERO|\bCRD\d+|\bpre-?regist|\bregistered\b|\bregistration\b", re.IGNORECASE
)
CRD_RE = re.compile(r"CRD\d{11}")
OVERCLAIM_RE = re.compile(
    r"\b(prove[sdn]?|proof|conclusively|definitively|undoubtedly|beyond (any )?doubt|"
    r"establish(es|ed)? that|confirm(s|ed)? that|clearly demonstrates?)\b",
    re.I,
)
CAUSAL_RE = re.compile(
    r"\b(caus(e|es|ed|al|ally)|leads? to|led to|results? in|resulted in|due to|because of|"
    r"attributable to|prevents?|prevented|protects? against|induces?|induced)\b",
    re.I,
)
RECOMMEND_RE = re.compile(
    r"\b(should (be|not)|we recommend|(is|are) recommended|must be|standard of care|"
    r"first[- ]line|clinicians should|ought to|should consider)\b",
    re.I,
)
MECHANISM_RE = re.compile(r"\b(mechanism|mediated by|pathophysiolog\w*|biologically plausib)", re.I)
SUPERIORITY_RE = re.compile(
    r"\b(superior|inferior|more effective|less effective|better than|worse than|outperform\w*|"
    r"non-?inferior)\b",
    re.I,
)
POOLED_CUE = re.compile(
    r"\b(pooled|meta-analy\w*|summary estimate|combined estimate|"
    r"overall estimate)\b",
    re.I,
)
ESTIMATOR_CUES = [
    ("kaplan_meier", re.compile(r"kaplan|\bKM\b|actuarial|life[- ]table", re.I)),
    ("cumulative_incidence", re.compile(r"cumulative incidence|competing[- ]risk", re.I)),
    ("crude_proportion", re.compile(r"\bcrude\b|raw proportion|simple proportion", re.I)),
]
MODEL_CUES = [
    ("random_effects", re.compile(r"random[- ]effects?", re.I)),
    ("common_effect", re.compile(r"(common|fixed)[- ]effects?", re.I)),
]
SCALE_CUE = re.compile(r"\blogit\b|log[- ]odds|on the log|log scale|log-transformed", re.I)
ABSOLUTE_CUE = re.compile(
    r"absolute (risk )?(difference|reduction|increase)|percentage points|"
    r"risk difference",
    re.I,
)
SIGNIF_RE = re.compile(r"\b(non-?|in)?significan\w*", re.I)
NULL_CLAIM_RE = re.compile(
    r"\bno (significant )?(difference|effect|association)|did not differ|"
    r"\bequivalent\b|\bcomparable\b|\bsimilar\b",
    re.I,
)
CLINICAL_RE = re.compile(r"clinically (important|meaningful|relevant|significant)", re.I)
UNIT_CUES = [
    ("studies", re.compile(r"^\s*(?:\w+\s+){0,2}?(studies|study|trials?|cohorts?)\b", re.I)),
    ("reports", re.compile(r"^\s*(?:\w+\s+){0,2}?(reports?|full[- ]text articles?)\b", re.I)),
    (
        "records",
        re.compile(r"^\s*(?:\w+\s+){0,2}?(records?|citations?|titles?|abstracts?)\b", re.I),
    ),
    (
        "participants",
        re.compile(
            r"^\s*(?:\w+\s+){0,2}?(patients?|participants?|subjects?|"
            r"individuals?|people|persons)\b",
            re.I,
        ),
    ),
    ("events", re.compile(r"^\s*(?:\w+\s+){0,1}?(events?)\b", re.I)),
]
TIME_RE = re.compile(r"^\s*-?\s*(years?|months?|weeks?|days?)\b", re.I)
STOP = frozenset(
    [
        "this",
        "that",
        "with",
        "were",
        "from",
        "have",
        "been",
        "their",
        "there",
        "which",
        "these",
        "those",
        "than",
        "into",
        "also",
        "among",
        "after",
        "before",
        "about",
        "over",
        "under",
        "such",
        "more",
        "less",
        "most",
        "other",
        "each",
        "both",
        "only",
        "same",
    ]
)


@dataclass
class Sentence:
    section_id: str
    index: int
    text: str
    start: int
    spans: list[Span]
    literal: str  # text with spans and markers blanked out

    @property
    def location(self) -> str:
        return f"{self.section_id} sentence {self.index + 1}"

    def facts(self, package: EvidencePackage) -> list[Fact]:
        return [
            package.facts[s.key] for s in self.spans if s.kind == "fact" and s.key in package.facts
        ]

    def refs(self) -> list[str]:
        return [k for s in self.spans if s.kind == "citation" for k in s.key.split(";")]


@dataclass
class _Audit:
    version: ManuscriptVersion
    package: EvidencePackage
    rendered: Rendered
    findings: list[AuditFinding] = field(default_factory=list)
    numeric: list[NumericClaimCheck] = field(default_factory=list)
    citations: list[CitationCheck] = field(default_factory=list)
    coverage: list[CoverageItem] = field(default_factory=list)
    scopes: dict[str, int] = field(default_factory=dict)

    def find(
        self,
        scope: str,
        status: CheckStatus,
        location: str,
        message: str,
        action: str,
        excerpt: str = "",
        key: str | None = None,
        value: str | None = None,
    ) -> None:
        digest = hashlib.sha256(
            "|".join([scope, status, location, key or "", message, excerpt]).encode()
        ).hexdigest()[:20]
        if any(f.finding_id == digest for f in self.findings):
            return
        self.findings.append(
            AuditFinding(
                finding_id=digest,
                scope=scope,
                status=status,
                location=location,
                excerpt=excerpt[:400],
                source_key=key,
                source_value=value,
                message=message,
                action=action,
            )
        )

    def tick(self, scope: str, n: int = 1) -> None:
        self.scopes[scope] = self.scopes.get(scope, 0) + n


# --------------------------------------------------------------------------- segmentation


def _split(sec: RenderedSection) -> list[Sentence]:
    text = sec.text
    inside = [(s.start, s.end) for s in sec.spans]

    def in_span(i: int) -> bool:
        return any(a <= i < b for a, b in inside)

    cuts = [0]
    for m in re.finditer(r"\n+|(?<=[.!?])\s+(?=[A-Z\[*(“\"])", text):
        if not in_span(m.start()) and not re.search(
            r"\b(et al|e\.g|i\.e|vs|Fig)\.$", text[max(0, m.start() - 8) : m.start()]
        ):
            cuts.append(m.end())
    cuts.append(len(text))
    out: list[Sentence] = []
    for a, b in itertools.pairwise(cuts):
        raw = text[a:b]
        if not raw.strip():
            continue
        spans = [
            Span(s.start - a, s.end - a, s.kind, s.key, s.binding_index)
            for s in sec.spans
            if a <= s.start < b
        ]
        chars = list(raw)
        for s in spans:
            for i in range(s.start, min(s.end, len(chars))):
                chars[i] = " "
        literal = AUTHOR_INPUT_RE.sub(lambda m: " " * len(m.group(0)), "".join(chars))
        out.append(Sentence(sec.section_id, len(out), raw.strip(), a, spans, literal))
    return out


def _negated(text: str) -> bool:
    return bool(NEGATION.search(text))


# --------------------------------------------------------------------------- main entry


def audit_manuscript(version: ManuscriptVersion, package: EvidencePackage) -> AuditReport:
    rendered = render(version.sections, package, frozen=version.bindings)
    a = _Audit(version, package, rendered)
    sentences = [s for sec in rendered.sections for s in _split(sec)]
    _provenance(a)
    _structure(a, sentences)
    _bindings(a)
    for s in sentences:
        _literal_numbers(a, s)
        _labels(a, s)
        _methods(a, s)
        _claims(a, s)
    a.tick("literal_numbers", 0)
    a.tick("semantic_labels", 0)
    a.tick("methods", 0)
    a.tick("claims", 0)
    _coverage(a)
    _consistency(a)
    _citations(a, sentences)
    _claim_support(a, sentences)
    return AuditReport(
        project_id=package.project_id,
        manuscript_version=version.version,
        stats_output_sha256=package.stats_sha256,
        manuscript_sha256=version.content_sha256,
        evidence_fingerprint=dict(package.fingerprint),
        auditor_version=AUDITOR_VERSION,
        scopes_checked=dict(sorted(a.scopes.items())),
        coverage=a.coverage,
        findings=a.findings,
        numeric_checks=a.numeric,
        citation_checks=a.citations,
        limitations=LIMITATIONS,
    )


# --------------------------------------------------------------------------- scopes


def _provenance(a: _Audit) -> None:
    v, p = a.version, a.package
    a.tick("provenance", len(p.fingerprint))
    for name, current in sorted(p.fingerprint.items()):
        old = v.evidence_fingerprint.get(name)
        if old != current:
            a.find(
                "provenance",
                CheckStatus.MISMATCH,
                "whole manuscript",
                f"{name.replace('_sha256', '').replace('_', ' ')} changed since version "
                f"{v.version} was generated",
                "Re-render the manuscript against the current "
                "evidence (new version) and audit again.",
                key=name,
                value=f"version {old}, current {current}",
            )
    for msg in a.rendered.drift:
        a.find(
            "provenance",
            CheckStatus.MISMATCH,
            msg.split(":")[0],
            msg,
            "Re-render the manuscript: its tables/markers no longer align with the stored values.",
        )
    if a.rendered.markdown != v.rendered_markdown:
        a.find(
            "provenance",
            CheckStatus.MISMATCH,
            "whole manuscript",
            "the stored rendered text does not equal the re-rendering of the stored source "
            "and frozen values (references, templates or numbering changed, or the stored "
            "text was altered)",
            "Re-render the manuscript and audit the new version.",
        )


def _structure(a: _Audit, sentences: list[Sentence]) -> None:
    v = a.version
    for sid in REQUIRED_SECTIONS:
        a.tick("structure")
        msec = v.section(sid)
        if msec is None or not msec.source.strip():
            a.find(
                "structure",
                CheckStatus.MISSING,
                sid,
                f"required section '{sid}' is missing or empty",
                f"Write the {sid} section.",
            )
    for rsec in a.rendered.sections:
        for m in AUTHOR_INPUT_RE.finditer(rsec.text):
            a.find(
                "structure",
                CheckStatus.MISSING,
                rsec.section_id,
                f"author input required: {m.group(1).strip()}",
                "Replace the [[AUTHOR INPUT]] marker with author-written, evidence-supported text.",
                excerpt=m.group(0),
            )
        for key in rsec.unknown_facts:
            a.find(
                "structure",
                CheckStatus.NOT_FOUND,
                rsec.section_id,
                f"marker {{{{{key}}}}} names no approved fact",
                "Use a key listed in the evidence package.",
                key=key,
            )
    abstract = next((s for s in a.rendered.sections if s.section_id == "abstract"), None)
    if abstract is not None:
        wanted = v.journal.abstract_headings or [
            "Background|Introduction",
            "Methods",
            "Results",
            "Conclusions?",
        ]
        for h in wanted:
            a.tick("structure")
            if not re.search(rf"\b({h})\s*:", abstract.text, re.I):
                a.find(
                    "structure",
                    CheckStatus.MISSING,
                    "abstract",
                    f"structured abstract heading '{h.split('|')[0].rstrip('?')}' is missing",
                    "Add the heading followed by a colon.",
                )
    _journal_limits(a)
    # Table / figure numbering: every reference must resolve, every item must be cited.
    defined = {"Table": set(), "Figure": set()}  # type: dict[str, set[str]]
    cited = {"Table": set(), "Figure": set()}  # type: dict[str, set[str]]
    for rsec in a.rendered.sections:
        for m in re.finditer(r"\*\*(Table|Figure) (\d+)\.\*\*", rsec.text):
            if rsec.section_id in ("tables", "figures"):
                defined[m.group(1)].add(m.group(2))
    for s in sentences:
        if s.section_id in ("tables", "figures"):
            continue
        for m in re.finditer(r"\b(Table|Figure)s? (\d+)\b", s.literal):
            cited[m.group(1)].add(m.group(2))
            a.tick("structure")
            if m.group(2) not in defined[m.group(1)]:
                a.find(
                    "structure",
                    CheckStatus.NOT_FOUND,
                    s.location,
                    f"{m.group(1)} {m.group(2)} is referred to but not defined",
                    "Correct the reference or add the table/figure.",
                    s.text,
                )
    for kind in ("Table", "Figure"):
        for n in sorted(defined[kind] - cited[kind]):
            a.find(
                "structure",
                CheckStatus.MISSING,
                f"{kind.lower()}s",
                f"{kind} {n} is never referred to in the text",
                f"Cite {kind} {n} in the text.",
            )


def _journal_limits(a: _Audit) -> None:
    j = a.version.journal
    words = {s.section_id: len(re.findall(r"\b\w+\b", s.text)) for s in a.rendered.sections}
    src = f"journal requirements ({j.requirements_source}, entered by {j.entered_by})"
    if j.abstract_word_limit is not None:
        a.tick("structure")
        if words.get("abstract", 0) > j.abstract_word_limit:
            a.find(
                "structure",
                CheckStatus.MISMATCH,
                "abstract",
                f"abstract has {words['abstract']} words; limit {j.abstract_word_limit} per {src}",
                "Shorten the abstract.",
            )
    if j.main_text_word_limit is not None:
        a.tick("structure")
        main = sum(
            words.get(s, 0)
            for s in (
                "introduction",
                "methods",
                "results",
                "discussion",
                "limitations",
                "conclusions",
            )
        )
        if main > j.main_text_word_limit:
            a.find(
                "structure",
                CheckStatus.MISMATCH,
                "main text",
                f"main text has {main} words; limit {j.main_text_word_limit} per {src}",
                "Shorten the main text.",
            )
    if j.reference_limit is not None:
        a.tick("structure")
        n = len(a.rendered.bibliography)
        if n > j.reference_limit:
            a.find(
                "structure",
                CheckStatus.MISMATCH,
                "references",
                f"{n} references; limit {j.reference_limit} per {src}",
                "Reduce the number of references.",
            )


def _bindings(a: _Audit) -> None:
    p = a.package
    for b in a.version.bindings:
        a.tick("numeric_bindings")
        loc = f"{b.section_id} marker {b.index + 1}"
        fact = p.facts.get(b.key)
        if fact is None:
            a.find(
                "numeric_bindings",
                CheckStatus.NOT_FOUND,
                loc,
                f"{b.key} no longer exists in the evidence package",
                "Remove or replace the marker and re-render.",
                b.rendered_text,
                key=b.key,
            )
            continue
        if b.fact_sha256 != fact.sha256() or b.rendered_text != fact.text:
            a.find(
                "numeric_bindings",
                CheckStatus.MISMATCH,
                loc,
                f"the value shown for {b.key} differs from the current approved source",
                "Re-render the manuscript so "
                "the current value is shown, then review the surrounding text.",
                b.rendered_text,
                key=b.key,
                value=f"{fact.text} ({fact.source})",
            )
        shown = find_numbers(b.rendered_text)
        if len(shown) != len(fact.numbers):
            a.find(
                "numeric_bindings",
                CheckStatus.MISMATCH,
                loc,
                f"the number of values shown for {b.key} does not match its source",
                "Re-render the manuscript.",
                b.rendered_text,
                key=b.key,
                value=fact.text,
            )
            continue
        for found, dn in zip(shown, fact.numbers, strict=True):
            _check_number(a, loc, b.rendered_text, b.key, found, dn)


def _check_number(
    a: _Audit, loc: str, text: str, key: str, found: FoundNumber, dn: DisplayNumber
) -> None:
    if dn.relation == "eq":
        ok = dn.accepts(found.value)
        a.numeric.append(
            NumericClaimCheck(
                claim_text=text[:300] or found.text,
                manuscript_location=loc,
                reported_value=found.value,
                stats_key=f"{key}#{dn.role}",
                expected_value=dn.value,
                tolerance=dn.tolerance,
                status=CheckStatus.VERIFIED if ok else CheckStatus.MISMATCH,
            )
        )
        if not ok:
            a.find(
                "numeric_bindings",
                CheckStatus.MISMATCH,
                loc,
                f"{dn.role} shown as "
                f"{found.text} but the source value is {dn.value!r} (allowed rounding ±"
                f"{dn.tolerance:.2g})",
                "Re-render the manuscript.",
                text,
                key=key,
                value=repr(dn.value),
            )
        return
    ok = dn.accepts(found.value)
    sign = "<" if dn.relation == "lt" else ">"
    a.find(
        "numeric_bindings",
        CheckStatus.VERIFIED if ok else CheckStatus.MISMATCH,
        loc,
        f"{dn.role} shown as {sign} {found.text}; source value {dn.value!r}",
        "Re-render the manuscript." if not ok else "None.",
        text,
        key=key,
        value=repr(dn.value),
    )


# --------------------------------------------------------------------------- typed numbers


def _analysis_context(a: _Audit, s: Sentence) -> str | None:
    ids = {f.analysis_id for f in s.facts(a.package) if f.analysis_id}
    if len(ids) == 1:
        return ids.pop()
    if ids:
        return None
    named = [
        x.analysis_id for x in a.package.stats.analyses if x.outcome_name.lower() in s.text.lower()
    ]
    return named[0] if len(set(named)) == 1 else None


def _role_of(literal: str, f: FoundNumber) -> tuple[str | None, str | None]:
    """(role, unit) of a typed number from the words around it."""
    before = literal[max(0, f.start - 25) : f.start]
    after = literal[f.end : f.end + 40]
    if re.search(r"^\s*%?\s*(CI|confidence|credible|prediction)", after, re.I) and f.value in (
        80,
        90,
        95,
        99,
    ):
        return "ci_level", None
    if re.search(r"I\s*(²|2|\^2)\s*(=|of|was|:)?\s*$", before):
        return "i2", None
    if re.search(r"\b[pP]\s*(=|<|>|≤|≥|value of|-value of|:)\s*$", before):
        return "p_value", None
    if re.search(r"(τ²|tau\s*(²|2|\^2|squared))\s*(=|of|was)?\s*$", before, re.I):
        return "tau2", None
    if re.search(r"(CI|interval)[^)]{0,12}$", before) or re.search(r"\bto\s*$", before):
        return "ci_bound", None
    if re.search(r"^\s*%?\s*\(\s*\d{2}\s*%\s*(CI|confidence)", after, re.I):
        return "estimate", None
    if TIME_RE.search(after):
        return "timepoint", TIME_RE.search(after).group(1).lower().rstrip("s")  # type: ignore[union-attr]
    for unit, cue in UNIT_CUES:
        if cue.search(after):
            return "count", unit
    return None, None


def _matching_keys(a: _Audit, value: float, facts: list[Fact] | None = None) -> list[str]:
    pool = facts if facts is not None else list(a.package.facts.values())
    return [
        f.key for f in pool if any(dn.relation == "eq" and dn.accepts(value) for dn in f.numbers)
    ]


def _literal_numbers(a: _Audit, s: Sentence) -> None:
    p = a.package
    if s.section_id in ("title",):
        return
    literal = ALLOWED_LITERALS.sub(lambda m: " " * len(m.group(0)), s.literal)
    found = find_numbers(literal)
    if not found:
        return
    analysis_id = _analysis_context(a, s)
    analysis = p.analysis(analysis_id) if analysis_id else None
    cited = bool(s.refs())
    _percent_arithmetic(a, s, literal)
    est_fact = None
    if analysis_id:
        est_fact = next(
            (
                f
                for f in p.facts_for_analysis(analysis_id)
                if f.role in ("pooled_estimate", "single_study_estimate")
                and f.model
                in (None, getattr(analysis.primary, "model", None) if analysis else None)
            ),
            None,
        )
    for f in found:
        a.tick("literal_numbers")
        role, unit = _role_of(literal, f)
        excerpt = s.text
        if role == "ci_level":
            levels = {
                dn.value for x in p.facts.values() for dn in x.numbers if dn.role == "ci_level"
            }
            if levels and f.value not in levels:
                a.find(
                    "literal_numbers",
                    CheckStatus.MISMATCH,
                    s.location,
                    f"{f.text}% interval stated, but the analyses use {sorted(levels)}%",
                    "Correct the level.",
                    excerpt,
                    value=str(sorted(levels)),
                )
            continue
        if (
            analysis is not None
            and f.value == 0
            and analysis.status in ("no_data", "unsupported", "not_estimable")
        ):
            a.find(
                "literal_numbers",
                CheckStatus.MISMATCH,
                s.location,
                f"zero reported for "
                f"{analysis.outcome_name}, but the analysis status is {analysis.status}: an "
                "unavailable result is not zero",
                "State that no estimate is available.",
                excerpt,
                key=f"stats.{analysis.analysis_id}.status",
                value=analysis.status,
            )
            continue
        target: list[Fact] | None = None
        roles: tuple[str, ...] = ()
        if role in ("i2", "p_value", "tau2") and analysis_id:
            roles = (role,)
            target = [
                x
                for x in p.facts_for_analysis(analysis_id)
                if any(dn.role == role for dn in x.numbers)
            ]
        elif role in ("estimate", "ci_bound") and est_fact is not None:
            roles = (
                ("pooled_estimate", "single_study_estimate")
                if role == "estimate"
                else ("ci_lower", "ci_upper")
            )
            target = [est_fact]
        elif role == "count" and unit:
            target = [
                x
                for x in p.facts.values()
                if x.count_unit == unit
                and (analysis_id is None or x.analysis_id in (None, analysis_id))
            ]
            roles = ("count",)
        elif role == "timepoint" and analysis is not None:
            tp_numbers = [n.value for t in analysis.timepoints for n in find_numbers(t)]
            unit_ok = any(unit and unit in t for t in analysis.timepoints)
            if f.value in tp_numbers and unit_ok:
                a.find(
                    "literal_numbers",
                    CheckStatus.REQUIRES_HUMAN_REVIEW,
                    s.location,
                    f"typed follow-up '{f.text} {unit}' matches the timepoint of "
                    f"{analysis.analysis_id}",
                    "Confirm, or use the measure_label marker.",
                    excerpt,
                    key=f"stats.{analysis.analysis_id}.measure_label",
                )
            else:
                a.find(
                    "literal_numbers",
                    CheckStatus.MISMATCH,
                    s.location,
                    f"follow-up "
                    f"'{f.text} {unit}' does not match the analysed timepoint(s) "
                    f"{analysis.timepoints or ['none (comparative)']}",
                    "Correct the follow-up period.",
                    excerpt,
                    key=f"stats.{analysis.analysis_id}.measure_label",
                    value=", ".join(analysis.timepoints),
                )
            continue
        if target is not None:
            nums = [dn for x in target for dn in x.numbers if dn.role in roles]
            if nums and any(dn.accepts(f.value) for dn in nums):
                a.find(
                    "literal_numbers",
                    CheckStatus.REQUIRES_HUMAN_REVIEW,
                    s.location,
                    f"typed number {f.text} equals an approved value but is not bound to it",
                    "Replace it with the {{key}} marker so it can be verified.",
                    excerpt,
                    key=", ".join(x.key for x in target)[:200],
                )
                continue
            if nums or role == "count":
                expected = "; ".join(f"{x.description}: {x.text}" for x in target)[:300]
                hint = ""
                if unit == "studies" and any(
                    x.count_unit == "reports" and any(dn.accepts(f.value) for dn in x.numbers)
                    for x in p.facts.values()
                ):
                    hint = " (it equals a count of reports: reports are not studies)"
                a.find(
                    "literal_numbers",
                    CheckStatus.MISMATCH,
                    s.location,
                    f"typed {(role or 'value').replace('_', ' ')} {f.text}"
                    f"{' ' + unit if unit else ''} does not match the approved source{hint}",
                    "Use the {{key}} marker for the correct value.",
                    excerpt,
                    key=", ".join(x.key for x in target)[:200],
                    value=expected or "none",
                )
                continue
        keys = _matching_keys(a, f.value)
        if keys:
            a.find(
                "literal_numbers",
                CheckStatus.REQUIRES_HUMAN_REVIEW,
                s.location,
                f"typed number {f.text} is not bound to a source; it equals {keys[:3]}",
                "Replace it with the matching {{key}} marker, or confirm it is not a finding.",
                excerpt,
                key=", ".join(keys[:5]),
            )
        elif cited and s.section_id in INTERPRETIVE:
            a.find(
                "literal_numbers",
                CheckStatus.REQUIRES_HUMAN_REVIEW,
                s.location,
                f"number {f.text} is attributed to cited literature and cannot be checked "
                "against this review's data",
                "Verify the number against the cited source.",
                excerpt,
            )
        else:
            a.find(
                "literal_numbers",
                CheckStatus.UNSUPPORTED,
                s.location,
                f"number {f.text} is not traceable to any approved source",
                "Remove it or use a {{key}} marker for an approved value.",
                excerpt,
            )


def _percent_arithmetic(a: _Audit, s: Sentence, literal: str) -> None:
    pat = re.compile(
        r"(\d[\d,]*)\s*(?:of|/|out of)\s*(\d[\d,]*)[^()\d]{0,15}\(\s*(\d+(?:[.·]\d+)?)\s*%"
    )
    for m in pat.finditer(literal):
        x, n = parse_number(m.group(1)), parse_number(m.group(2))
        shown = m.group(3)
        if n == 0:
            continue
        d = len(re.split(r"[.·]", shown)[1]) if re.search(r"[.·]", shown) else 0
        pct = 100.0 * x / n
        dn = DisplayNumber("percentage", pct, d)
        if not dn.accepts(parse_number(shown)):
            a.find(
                "literal_numbers",
                CheckStatus.MISMATCH,
                s.location,
                f"{shown}% is not {m.group(1)}/{m.group(2)} (= {pct:.{d + 1}f}%)",
                "Correct the percentage.",
                s.text,
                value=f"{pct:.{d + 1}f}%",
            )


# --------------------------------------------------------------------------- labels


def _window(s: Sentence, span: Span, width: int = 90) -> str:
    return s.literal[max(0, span.start - width) : span.start]


def _cue(pattern: re.Pattern[str], text: str) -> bool:
    m = pattern.search(text)
    if not m:
        return False
    before = text[max(0, m.start() - 25) : m.start()]
    return not re.search(
        r"\b(not|no|non|without|rather than|instead of|unlike)\b[\s-]*\w*\s*$", before, re.I
    )


def _labels(a: _Audit, s: Sentence) -> None:
    p = a.package
    for span in s.spans:
        if span.kind != "fact" or span.key not in p.facts:
            continue
        f = p.facts[span.key]
        if f.role not in ESTIMATE_ROLES:
            continue
        a.tick("semantic_labels")
        win = _window(s, span)
        if f.pooled is False and _cue(POOLED_CUE, win):
            a.find(
                "semantic_labels",
                CheckStatus.MISMATCH,
                s.location,
                f"{f.key} is a single-study result but is described as pooled",
                "Describe it as the result of a single study.",
                s.text,
                key=f.key,
                value=f.description,
            )
        for est, cue in ESTIMATOR_CUES:
            if f.estimator and est != f.estimator and _cue(cue, win):
                a.find(
                    "semantic_labels",
                    CheckStatus.MISMATCH,
                    s.location,
                    f"{f.key} is a "
                    f"{f.estimator.replace('_', ' ')} estimate but is labelled as "
                    f"{est.replace('_', ' ')}",
                    "Use the correct estimator label.",
                    s.text,
                    key=f.key,
                    value=f.description,
                )
        for model, cue in MODEL_CUES:
            if f.model and model != f.model and _cue(cue, win):
                a.find(
                    "semantic_labels",
                    CheckStatus.MISMATCH,
                    s.location,
                    f"{f.key} comes from "
                    f"the {f.model.replace('_', '-')} model but is described as "
                    f"{model.replace('_', '-')}",
                    "Correct the model description.",
                    s.text,
                    key=f.key,
                    value=f.description,
                )
        if SCALE_CUE.search(win):
            a.find(
                "semantic_labels",
                CheckStatus.MISMATCH,
                s.location,
                f"{f.key} is on the "
                "natural scale but the text describes a transformed (log/logit) scale",
                "Remove the scale description or report the natural-scale value.",
                s.text,
                key=f.key,
            )
        if f.measure in ("RR", "OR") and _cue(ABSOLUTE_CUE, win):
            a.find(
                "semantic_labels",
                CheckStatus.MISMATCH,
                s.location,
                f"{f.key} is a relative effect but is described as an absolute difference",
                "Describe it as a ratio.",
                s.text,
                key=f.key,
            )
        if f.measure == "RR" and re.search(r"\bodds\b", win, re.I):
            a.find(
                "semantic_labels",
                CheckStatus.MISMATCH,
                s.location,
                f"{f.key} is a risk ratio but is described as odds",
                "Call it a risk ratio.",
                s.text,
                key=f.key,
            )
        if f.measure == "OR" and re.search(r"risk ratio|relative risk", win, re.I):
            a.find(
                "semantic_labels",
                CheckStatus.MISMATCH,
                s.location,
                f"{f.key} is an odds ratio but is described as a risk ratio",
                "Call it an odds ratio.",
                s.text,
                key=f.key,
            )
        if f.measure == "MD" and re.search(r"standardi[sz]ed", win, re.I):
            a.find(
                "semantic_labels",
                CheckStatus.MISMATCH,
                s.location,
                f"{f.key} is a mean difference, not a standardised one",
                "Correct the label.",
                s.text,
                key=f.key,
            )
        _significance(a, s, f)
    if not any(
        sp.kind == "fact"
        and a.package.facts.get(sp.key, None) is not None
        and a.package.facts[sp.key].role in ESTIMATE_ROLES
        for sp in s.spans
    ):
        aid = _analysis_context(a, s)
        if aid:
            main = next(
                (
                    f
                    for f in p.facts_for_analysis(aid)
                    if f.role in ("pooled_estimate", "single_study_estimate")
                ),
                None,
            )
            if main is not None:
                a.tick("semantic_labels")
                _significance(a, s, main)


def _significance(a: _Audit, s: Sentence, f: Fact) -> None:
    excl = f.ci_excludes_null
    lit = s.literal
    for m in SIGNIF_RE.finditer(lit):
        neg = bool(m.group(1)) or bool(
            re.search(
                r"\b(not|no)\s+(statistically\s+)?$", lit[max(0, m.start() - 20) : m.start()], re.I
            )
        )
        if CLINICAL_RE.search(lit[max(0, m.start() - 12) : m.end()]):
            continue
        if excl is None:
            if f.null_value is None:
                a.find(
                    "semantic_labels",
                    CheckStatus.UNSUPPORTED,
                    s.location,
                    "statistical "
                    "significance is claimed for a single-group estimate, which has no "
                    "null hypothesis in this analysis",
                    "Remove the significance claim.",
                    s.text,
                    key=f.key,
                )
            continue
        if neg and excl:
            a.find(
                "semantic_labels",
                CheckStatus.MISMATCH,
                s.location,
                f"described as not significant, but the CI of {f.key} excludes the null value",
                "Correct the statement.",
                s.text,
                key=f.key,
                value=f.text,
            )
        elif not neg and not excl:
            a.find(
                "semantic_labels",
                CheckStatus.MISMATCH,
                s.location,
                f"described as significant, but the CI of {f.key} includes the null value",
                "Correct the statement.",
                s.text,
                key=f.key,
                value=f.text,
            )
    if excl is False and NULL_CLAIM_RE.search(lit):
        a.find(
            "semantic_labels",
            CheckStatus.REQUIRES_HUMAN_REVIEW,
            s.location,
            "a "
            "non-significant result is described as no difference/equivalence; absence of "
            "evidence is not evidence of absence",
            "Rephrase, or confirm that an equivalence margin justifies it.",
            s.text,
            key=f.key,
            value=f.text,
        )
    if CLINICAL_RE.search(lit):
        a.find(
            "semantic_labels",
            CheckStatus.REQUIRES_HUMAN_REVIEW,
            s.location,
            "clinical importance is asserted; statistical results do not establish it",
            "Confirm the minimal important difference and its source, or rephrase.",
            s.text,
            key=f.key,
        )


# --------------------------------------------------------------------------- methods


def _methods(a: _Audit, s: Sentence) -> None:
    p = a.package
    if s.section_id in ("title", "tables", "figures"):
        return
    lit = s.literal
    a.tick("methods")
    neg = _negated(lit)
    # who screened / extracted
    if REVIEWERS_RE.search(lit) and not AUTOMATION_RE.search(lit) and not neg:
        if re.search(r"extract", lit, re.I):
            a.find(
                "methods",
                CheckStatus.MISMATCH,
                s.location,
                "claims duplicate/independent "
                "human data extraction; extraction was not performed in duplicate by two "
                "independent humans",
                "Describe extraction as recorded (see {{extraction.methods}}).",
                s.text,
                key="extraction.methods",
            )
        elif screening_is_automated(p):
            a.find(
                "methods",
                CheckStatus.MISMATCH,
                s.location,
                "describes independent human "
                "reviewers, but screening used automated passes (not equivalent to two "
                "independent human reviewers)",
                "Use the recorded screening statement.",
                s.text,
                key="screening.title_abstract",
            )
    # registration
    m = REGISTRATION_RE.search(lit)
    if m and not re.search(r"trial regist|registries|registers?\b", lit, re.I):
        crd = CRD_RE.search(lit)
        actual = p.protocol.prospero_id
        if crd and crd.group(0) != actual:
            a.find(
                "methods",
                CheckStatus.MISMATCH,
                s.location,
                f"registration {crd.group(0)} "
                f"does not match the protocol ({actual or 'none recorded'})",
                "Use {{protocol.prospero_id}}.",
                s.text,
                key="protocol.prospero_id",
                value=actual or "none",
            )
        elif actual is None and not neg:
            a.find(
                "methods",
                CheckStatus.UNSUPPORTED,
                s.location,
                "protocol registration is claimed but no registration identifier is recorded",
                "Remove the claim, or record the registration in the protocol.",
                s.text,
                key="protocol.prospero_id",
            )
    # databases
    for cue, db in DATABASE_CUES:
        if s.section_id not in ("methods", "abstract"):
            break
        hit = cue.search(lit)
        if hit and not neg and (db is None or db.value not in p.searched_databases):
            a.find(
                "methods",
                CheckStatus.UNSUPPORTED,
                s.location,
                f"'{hit.group(0)}' is described as searched, but the search log has no search "
                "of it",
                "Use {{search.statement}}.",
                s.text,
                key="search.statement",
            )
    # analyses and estimators
    if s.section_id in ("methods", "results", "abstract", "discussion", "limitations"):
        for name, cue in METHOD_CUES:
            if not cue.search(lit) or neg:
                continue
            if name == "hazard_ratio_pooling" and s.section_id not in ("methods", "results"):
                continue
            if name not in _performed(p):
                a.find(
                    "methods",
                    CheckStatus.UNSUPPORTED,
                    s.location,
                    f"describes "
                    f"{name.replace('_', ' ')}, which was not performed in the recorded "
                    "analyses",
                    "Remove the statement, or state that it was not performed.",
                    s.text,
                    key="stats.methods",
                )
    # risk-of-bias tools
    raw = s.text
    for cue, tool in ROB_CUES:
        hit = cue.search(raw)
        if not hit or _in_span(s, hit.start()) or neg:
            continue
        if isinstance(tool, RoBTool):
            ok = tool in p.rob_tools
        else:
            ok = (
                RoBTool.EXTERNAL in p.rob_tools
                and tool.lower() in p.facts.get("rob.methods", _EMPTY).text.lower()
            )
        if not ok:
            a.find(
                "methods",
                CheckStatus.UNSUPPORTED,
                s.location,
                f"risk-of-bias tool "
                f"'{hit.group(0)}' is mentioned but no approved assessment used it",
                "Use {{rob.methods}}.",
                s.text,
                key="rob.methods",
            )


@dataclass(frozen=True)
class _Empty:
    text: str = ""


_EMPTY = _Empty()


def _in_span(s: Sentence, pos: int) -> bool:
    return any(sp.start <= pos < sp.end for sp in s.spans)


def _performed(p: EvidencePackage) -> set[str]:
    done = set(p.performed_methods)
    for a in p.stats.analyses:
        if a.random_effects is not None:
            done.add(str(a.settings.get("tau2_estimator", "")).lower())
            ci = str(a.settings.get("random_effects_ci", ""))
            if ci.startswith("hksj"):
                done.add("hksj")
        for r in (a.common_effect, a.random_effects, *a.sensitivity):
            if r is None:
                continue
            if "Mantel" in r.method:
                done.add("mantel_haenszel")
            if "GLMM" in r.method:
                done.add("glmm")
    return done


# --------------------------------------------------------------------------- claims


def _claims(a: _Audit, s: Sentence) -> None:
    if s.section_id in ("title", "methods", "tables", "figures", "supplementary"):
        return
    a.tick("claims")
    lit = s.literal
    if OVERCLAIM_RE.search(lit):
        a.find(
            "claims",
            CheckStatus.UNSUPPORTED,
            s.location,
            "overstated certainty ("
            f"'{OVERCLAIM_RE.search(lit).group(0)}'); a systematic review estimates, it "  # type: ignore[union-attr]
            "does not prove",
            "Rephrase with appropriate uncertainty.",
            s.text,
        )
    if a.package.review_design == ReviewDesign.SINGLE_ARM and SUPERIORITY_RE.search(lit):
        a.find(
            "claims",
            CheckStatus.UNSUPPORTED,
            s.location,
            "comparative claim in a single-arm review, which has no comparator",
            "Remove the comparative claim.",
            s.text,
        )
    if s.section_id == "results":
        return
    if CAUSAL_RE.search(lit):
        a.find(
            "claims",
            CheckStatus.REQUIRES_HUMAN_REVIEW,
            s.location,
            "causal language "
            f"('{CAUSAL_RE.search(lit).group(0)}'); check that the study designs and "  # type: ignore[union-attr]
            "evidence support causation",
            "Confirm or rephrase as association.",
            s.text,
        )
    if RECOMMEND_RE.search(lit):
        a.find(
            "claims",
            CheckStatus.REQUIRES_HUMAN_REVIEW,
            s.location,
            "clinical recommendation; check that it stays within the evidence",
            "Confirm or remove the recommendation.",
            s.text,
        )
    if MECHANISM_RE.search(lit):
        a.find(
            "claims",
            CheckStatus.REQUIRES_HUMAN_REVIEW,
            s.location,
            "mechanistic statement; not established by this review",
            "Confirm it is cited and labelled as hypothesis.",
            s.text,
        )


# --------------------------------------------------------------------------- coverage


def _bound_where(a: _Audit) -> dict[str, set[str]]:
    where: dict[str, set[str]] = {}
    for b in a.version.bindings:
        where.setdefault(b.key, set()).add(b.section_id)
    for sid, key in a.rendered.templates_used:
        where.setdefault(key, set()).add(sid)
    return where


def _coverage(a: _Audit) -> None:
    where = _bound_where(a)
    items = list(a.package.expected)
    from src.manuscript.evidence import ExpectedOutput

    items += [
        ExpectedOutput(
            "table:study_characteristics", "table of included-study characteristics", ("tables",)
        ),
        ExpectedOutput("table:results_summary", "summary table of all analyses", ("tables",)),
    ]
    for e in items:
        a.tick("coverage")
        found = sorted({sec for k in e.keys for sec in where.get(k, set())})
        ok = any(sec in e.sections for sec in found)
        a.coverage.append(
            CoverageItem(
                key=" | ".join(e.keys),
                description=e.description,
                required_in=list(e.sections),
                status=CheckStatus.VERIFIED if ok else CheckStatus.MISSING,
                found_in=found,
            )
        )
        if not ok:
            where_txt = f" (found only in {found})" if found else ""
            a.find(
                "coverage",
                CheckStatus.MISSING,
                "/".join(e.sections),
                f"expected output not reported: {e.description}{where_txt}",
                f"Add {{{{{e.keys[0]}}}}} to the {e.sections[0]} section.",
                key=e.keys[0],
            )
    cited = set(a.rendered.citation_order)
    for rid, key in sorted(a.package.included_ref_keys.items()):
        a.tick("coverage")
        ok = key in cited
        a.coverage.append(
            CoverageItem(
                key=f"cite:{key}",
                description=f"citation of included report {rid}",
                required_in=["any"],
                status=CheckStatus.VERIFIED if ok else CheckStatus.MISSING,
            )
        )
        if not ok:
            a.find(
                "coverage",
                CheckStatus.MISSING,
                "references",
                f"included report {rid} is never cited",
                f"Cite [@{key}] (e.g. in the Results or Table 1).",
                key=key,
            )


def _consistency(a: _Audit) -> None:
    where = _bound_where(a)
    for key, sections in sorted(where.items()):
        if not key.startswith("stats.") or key.endswith((".outcome", ".measure_label")):
            continue
        for sec in ("abstract", "conclusions"):
            if sec in sections:
                a.tick("consistency")
                if "results" not in sections and "tables" not in sections:
                    a.find(
                        "consistency",
                        CheckStatus.MISSING,
                        sec,
                        f"{key} is reported in the {sec} but not in the Results",
                        f"Report it in the Results, or remove it from the {sec}.",
                        key=key,
                    )
    a.tick("consistency", len(a.rendered.templates_used))


# --------------------------------------------------------------------------- citations


def _citations(a: _Audit, sentences: list[Sentence]) -> None:
    p = a.package
    for sec in a.rendered.sections:
        for k in sec.unknown_refs:
            a.tick("citations")
            a.find(
                "citations",
                CheckStatus.NOT_FOUND,
                sec.section_id,
                f"[@{k}] is not in the reference registry",
                "Import the reference by PMID/DOI, or remove the citation.",
                key=k,
            )
    for s in sentences:
        if MANUAL_CITATION_RE.search(s.literal):
            a.tick("citations")
            a.find(
                "citations",
                CheckStatus.UNSUPPORTED,
                s.location,
                "a typed citation number is not linked to the reference registry",
                "Use [@ref] markers.",
                s.text,
            )
        if AUTHOR_YEAR_RE.search(s.literal) and not s.refs():
            a.tick("citations")
            a.find(
                "citations",
                CheckStatus.UNSUPPORTED,
                s.location,
                "work attributed to named authors without a registry citation",
                "Add the [@ref] marker of the registered reference.",
                s.text,
            )
    seen: dict[str, str] = {}
    for n, key in enumerate(a.rendered.citation_order, start=1):
        a.tick("citations")
        state = p.references[key]
        e = state.entry
        status = {
            "verified": CheckStatus.VERIFIED,
            "mismatch": CheckStatus.MISMATCH,
            "not_found": CheckStatus.NOT_FOUND,
        }.get(state.identity_status, CheckStatus.NOT_CHECKED)
        if status == CheckStatus.VERIFIED and not (e.pmid or e.doi):
            status = CheckStatus.NOT_CHECKED
        a.citations.append(
            CitationCheck(
                citation_text=f"[{n}] {e.title}"[:300],
                manuscript_location=f"reference {n}",
                pmid=e.pmid,
                doi=e.doi,
                matched_record_id=(e.record_id or e.ref_key)
                if status == CheckStatus.VERIFIED
                else None,
                status=status,
            )
        )
        if status != CheckStatus.VERIFIED:
            a.find(
                "citations",
                status,
                f"reference {n}",
                f"identity of reference {n} "
                f"({key}) is {state.identity_status}: {state.identity_basis}",
                "Run live reference verification (PubMed/Crossref) and correct the entry."
                if status == CheckStatus.NOT_CHECKED
                else "Correct or replace the reference.",
                key=key,
            )
        for ident in filter(
            None,
            (
                f"doi:{e.doi}" if e.doi else None,
                f"pmid:{e.pmid}" if e.pmid else None,
                f"title:{normalize_title(e.title)}:{e.year}",
            ),
        ):
            if ident in seen and seen[ident] != key:
                a.find(
                    "citations",
                    CheckStatus.MISMATCH,
                    f"reference {n}",
                    f"duplicate "
                    f"reference: {key} and {seen[ident]} are the same work ({ident[:60]})",
                    "Cite one registry entry only.",
                    key=key,
                )
            seen.setdefault(ident, key)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def _claim_support(a: _Audit, sentences: list[Sentence]) -> None:
    p = a.package
    for s in sentences:
        refs = [k for k in dict.fromkeys(s.refs()) if k in p.references]
        if not refs:
            continue
        facts = s.facts(p)
        study_facts = [
            f
            for f in facts
            if f.record_ids
            and (
                f.role in ("study_estimate", "single_study_estimate")
                or f.key.startswith(("study.", "extraction."))
            )
        ]
        listing = s.section_id == "tables" or any(
            f.key in ("prisma.studies_included", "prisma.reports_included")
            or (f.role == "count" and f.count_unit == "studies")
            for f in facts
        )
        for key in refs:
            a.tick("claim_support")
            state = p.references[key]
            e = state.entry
            loc = f"{s.location} [@{key}]"
            if e.origin == "included_study_record":
                if study_facts:
                    if any(e.record_id in f.record_ids for f in study_facts):
                        a.find(
                            "claim_support",
                            CheckStatus.VERIFIED,
                            loc,
                            "study-level values in this sentence come from the cited report",
                            "None.",
                            s.text,
                            key=key,
                        )
                    else:
                        a.find(
                            "claim_support",
                            CheckStatus.MISMATCH,
                            loc,
                            "the study-level "
                            "values in this sentence belong to a different study than the "
                            "one cited",
                            "Cite the study the values come from.",
                            s.text,
                            key=key,
                            value=", ".join(sorted({r for f in study_facts for r in f.record_ids})),
                        )
                elif listing and e.record_id in p.included_ref_keys:
                    a.find(
                        "claim_support",
                        CheckStatus.VERIFIED,
                        loc,
                        "cited as an included study",
                        "None.",
                        s.text,
                        key=key,
                    )
                else:
                    a.find(
                        "claim_support",
                        CheckStatus.REQUIRES_HUMAN_REVIEW,
                        loc,
                        "statement "
                        "attributed to an included study is not tied to its extracted data",
                        "Confirm against the study report, or bind the study's values.",
                        s.text,
                        key=key,
                    )
                continue
            if not state.excerpts:
                a.find(
                    "claim_support",
                    CheckStatus.REQUIRES_HUMAN_REVIEW,
                    loc,
                    "claim-level support is unverified: no excerpt of the cited source is recorded",
                    "Add the supporting passage as an excerpt, or confirm support after "
                    "reading the source.",
                    s.text,
                    key=key,
                )
                continue
            excerpt_text = " ".join(x.text for x in state.excerpts)
            exc_nums = [n.value for n in find_numbers(excerpt_text)]
            lit = ALLOWED_LITERALS.sub(" ", s.literal)
            missing = [n.text for n in find_numbers(lit) if n.value not in exc_nums]
            words = _content_words(lit)
            overlap = len(words & _content_words(excerpt_text)) / max(1, len(words))
            if missing:
                a.find(
                    "claim_support",
                    CheckStatus.UNSUPPORTED,
                    loc,
                    f"number(s) {missing} do "
                    "not appear in the recorded excerpt(s) of the cited source",
                    "Correct the number or cite the source that reports it.",
                    s.text,
                    key=key,
                    value=excerpt_text[:300],
                )
            elif overlap < 0.25:
                a.find(
                    "claim_support",
                    CheckStatus.UNSUPPORTED,
                    loc,
                    f"the recorded excerpt(s) do "
                    f"not appear to support this sentence (content-word overlap "
                    f"{overlap:.0%})",
                    "Cite a source that supports the statement, or add the relevant excerpt.",
                    s.text,
                    key=key,
                    value=excerpt_text[:300],
                )
            else:
                a.find(
                    "claim_support",
                    CheckStatus.REQUIRES_HUMAN_REVIEW,
                    loc,
                    f"lexically "
                    f"consistent with the recorded excerpt (overlap {overlap:.0%}); semantic "
                    "support needs human confirmation",
                    "Confirm that the source supports the statement.",
                    s.text,
                    key=key,
                    value=excerpt_text[:300],
                )
