"""Stage 6 contracts: manuscript versions, the reference registry, acknowledgements, sign-off.

A manuscript is stored as *source* Markdown per section. Numbers and citations are not typed
into the source; they are markers resolved deterministically from the evidence package:

* ``{{KEY}}`` — an approved fact (statistics, PRISMA, protocol, search, screening, RoB);
* ``{{table:NAME}}`` / ``{{figure:forest:ANALYSIS}}`` — tables and figure legends;
* ``[@REF]`` or ``[@REF1; @REF2]`` — citations of entries in the reference registry;
* ``[[AUTHOR INPUT: ...]]`` — content that only an author can supply (blocks the audit).

Each version freezes what was rendered (the bindings and the rendered Markdown), so an audit
can tell whether the numbers in *this* version still match the current approved sources.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, computed_field, model_validator

from src.core.enums import ActorType
from src.core.schemas.base import (
    DOI,
    PMID,
    AwareDatetime,
    FrozenModel,
    Identifier,
    NonEmptyStr,
    StrictModel,
    new_id,
    utc_now,
)

SectionId = Literal[
    "title",
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "limitations",
    "conclusions",
    "tables",
    "figures",
    "supplementary",
]
SECTION_ORDER: tuple[str, ...] = (
    "title", "abstract", "introduction", "methods", "results", "discussion", "limitations",
    "conclusions", "tables", "figures", "supplementary",
)  # fmt: skip
REQUIRED_SECTIONS: tuple[str, ...] = (
    "title", "abstract", "introduction", "methods", "results", "discussion", "limitations",
    "conclusions",
)  # fmt: skip


def _sha(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ManuscriptSection(StrictModel):
    section_id: SectionId
    heading: NonEmptyStr
    source: str = Field(description="Markdown with {{fact}} and [@ref] markers")
    origin: Literal["template", "llm", "human"]


class StyleSettings(StrictModel):
    """Writing preferences passed to the drafting model. They never change numbers."""

    spelling: Literal["british", "american"] = "british"
    person: Literal["first_plural", "impersonal"] = "first_plural"
    notes: str | None = Field(default=None, max_length=2000)


class JournalSettings(StrictModel):
    """Target-journal requirements. Limits apply only when an author supplied them together
    with where they come from (author guidelines URL or document) — nothing is assumed."""

    name: str | None = None
    requirements_source: str | None = None
    entered_by: str | None = None
    abstract_word_limit: int | None = Field(default=None, ge=1)
    main_text_word_limit: int | None = Field(default=None, ge=1)
    reference_limit: int | None = Field(default=None, ge=1)
    abstract_headings: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sourced(self) -> JournalSettings:
        has_rules = any(
            (self.abstract_word_limit, self.main_text_word_limit, self.reference_limit,
             self.abstract_headings)
        )  # fmt: skip
        if has_rules and not (self.requirements_source and self.entered_by):
            raise ValueError(
                "journal limits/headings need requirements_source and entered_by: they are "
                "applied only when the requirements are explicitly provided"
            )
        return self


class NumericBinding(FrozenModel):
    """One marker as rendered in a version: the key, the text and the source values."""

    section_id: SectionId
    index: int = Field(ge=0, description="position of the marker within the section")
    key: NonEmptyStr
    rendered_text: str
    fact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GenerationProvenance(FrozenModel):
    method: Literal["deterministic_template", "llm", "human_edit", "rerender"]
    sections: list[str] = Field(default_factory=list)
    provider: str | None = None
    model_id: str | None = None
    effort: str | None = None
    llm_call_ids: list[str] = Field(default_factory=list)
    prompt_sha256: list[str] = Field(default_factory=list)
    response_sha256: list[str] = Field(default_factory=list)
    note: str | None = None


class ManuscriptVersion(StrictModel):
    project_id: Identifier
    version: int = Field(ge=1)
    parent_version: int | None = Field(default=None, ge=1)
    sections: list[ManuscriptSection] = Field(min_length=1)
    style: StyleSettings = Field(default_factory=StyleSettings)
    journal: JournalSettings = Field(default_factory=JournalSettings)
    bindings: list[NumericBinding] = Field(default_factory=list)
    rendered_markdown: str
    evidence_fingerprint: dict[str, str]
    created_by: NonEmptyStr
    actor_type: ActorType
    reviewer_id: Identifier | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    change_summary: NonEmptyStr
    provenance: list[GenerationProvenance] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> ManuscriptVersion:
        ids = [s.section_id for s in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError("each section appears once")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("parent_version must precede version")
        if self.actor_type == ActorType.HUMAN and self.reviewer_id is None:
            raise ValueError("a human edit records the authenticated reviewer_id")
        return self

    def section(self, section_id: str) -> ManuscriptSection | None:
        return next((s for s in self.sections if s.section_id == section_id), None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_sha256(self) -> str:
        """Hash of what the manuscript says: sources, frozen bindings, rendered text, style."""
        return _sha(
            {
                "sections": [s.model_dump(mode="json") for s in self.sections],
                "bindings": [b.model_dump(mode="json") for b in self.bindings],
                "rendered": self.rendered_markdown,
                "style": self.style.model_dump(mode="json"),
                "journal": self.journal.model_dump(mode="json"),
            }
        )


# --------------------------------------------------------------------------- references


class ReferenceEntry(StrictModel):
    """A bibliographic reference that may be cited. Every field comes from a stored record
    or an API response — never from a language model."""

    ref_key: Identifier
    origin: Literal["included_study_record", "imported_literature"]
    record_id: Identifier | None = None
    pmid: PMID = None
    doi: DOI = None
    title: NonEmptyStr
    authors: list[NonEmptyStr] = Field(default_factory=list, description="'Surname AB' form")
    journal: str | None = None
    year: int | None = Field(default=None, ge=1800, le=2100)
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    metadata_source: NonEmptyStr = Field(description="e.g. 'pubmed_efetch', 'crossref'")
    retrieved_at: AwareDatetime
    added_by: NonEmptyStr
    reviewer_id: Identifier | None = None

    @model_validator(mode="after")
    def _origin(self) -> ReferenceEntry:
        if self.origin == "included_study_record" and self.record_id is None:
            raise ValueError("an included-study reference names its record_id")
        if self.origin == "imported_literature" and not (self.pmid or self.doi):
            raise ValueError("imported literature must carry a PMID or DOI it was fetched by")
        return self

    def metadata_sha256(self) -> str:
        return _sha(
            self.model_dump(mode="json", exclude={"retrieved_at", "added_by", "reviewer_id"})
        )


VerificationStatus = Literal["verified", "mismatch", "not_found", "unavailable"]


class ReferenceVerification(FrozenModel):
    """Result of one *live* metadata lookup. ``unavailable`` means the lookup failed (no
    network, API error) — it is never reported as a verification.

    ``human_attestation`` exists only for included-study references that have neither a DOI
    nor a PMID (nothing to look up): an authenticated reviewer compares the stored metadata
    with the report's identity-confirmed PDF and states the basis. The repository enforces
    those preconditions."""

    verification_id: Identifier = Field(default_factory=new_id)
    ref_key: Identifier
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: Literal["pubmed", "crossref", "human_attestation"]
    status: VerificationStatus
    field_results: dict[str, str] = Field(default_factory=dict)
    message: str = ""
    checked_at: AwareDatetime = Field(default_factory=utc_now)
    checked_by: NonEmptyStr
    reviewer_id: Identifier | None = None

    @model_validator(mode="after")
    def _attestation(self) -> ReferenceVerification:
        if self.source == "human_attestation":
            if self.reviewer_id is None or len(self.message.strip()) < 10:
                raise ValueError("an identity attestation needs the reviewer and its basis")
            if self.status not in ("verified", "mismatch"):
                raise ValueError("an attestation either confirms or contradicts the identity")
        elif self.reviewer_id is not None:
            raise ValueError("live lookups are not human attestations")
        return self


class EvidenceExcerpt(FrozenModel):
    """A passage from a cited source, entered or confirmed by a human, used to check whether
    the source supports the sentence citing it."""

    excerpt_id: Identifier = Field(default_factory=new_id)
    ref_key: Identifier
    text: str = Field(min_length=20, max_length=5000)
    location: NonEmptyStr = Field(description="page, section, table or figure")
    added_by: NonEmptyStr
    reviewer_id: Identifier
    added_at: AwareDatetime = Field(default_factory=utc_now)


# --------------------------------------------------------------------------- human review


class FindingAcknowledgement(FrozenModel):
    """A human's assessment of a REQUIRES_HUMAN_REVIEW finding for one exact manuscript
    content and evidence state. Blocking findings cannot be acknowledged."""

    ack_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    audit_report_id: Identifier
    finding_id: NonEmptyStr
    manuscript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["confirmed_supported", "accepted_as_author_statement"]
    note: str = Field(min_length=10)
    reviewer_id: Identifier
    reviewer_name: NonEmptyStr
    acknowledged_at: AwareDatetime = Field(default_factory=utc_now)


class ManuscriptApproval(FrozenModel):
    """Final human sign-off of one manuscript version on the basis of one audit report."""

    approval_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    manuscript_version: int = Field(ge=1)
    manuscript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_report_id: Identifier
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: Identifier
    reviewer_name: NonEmptyStr
    statement: str = Field(min_length=10)
    approved_at: AwareDatetime = Field(default_factory=utc_now)


class ClaimReview(FrozenModel):
    """An *advisory* model opinion on whether an excerpt supports a sentence. It never
    changes an audit finding and never counts as a human assessment."""

    review_id: Identifier = Field(default_factory=new_id)
    project_id: Identifier
    audit_report_id: Identifier
    finding_id: NonEmptyStr
    verdict: Literal["supports", "does_not_support", "unclear"]
    rationale: str = Field(min_length=1, max_length=4000)
    provider: NonEmptyStr
    model_id: NonEmptyStr
    effort: str | None = None
    call_id: str | None = None
    prompt_sha256: str
    response_sha256: str
    is_human: Literal[False] = False
    created_at: AwareDatetime = Field(default_factory=utc_now)


def fingerprint_sha256(fingerprint: dict[str, str]) -> str:
    return _sha(fingerprint)
