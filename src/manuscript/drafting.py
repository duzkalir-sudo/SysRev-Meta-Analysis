"""Stage 6 drafting: a deterministic template draft, and optional model-assisted prose.

**Deterministic path (no AI provider needed).** :func:`template_sections` writes every
section from the evidence package. Numbers and citations are markers resolved by the
renderer; content only an author can supply (background literature, interpretation,
conclusions) is left as ``[[AUTHOR INPUT: ...]]``, which the Auditor reports as missing.

**Model-assisted path.** :class:`ManuscriptDrafter` asks a configured model to write or
improve one section in academic English. The model sees the facts (key, meaning and the
already-rendered text) and the reference registry, and must express every number as a
``{{key}}`` marker and every citation as ``[@ref]``. :func:`validate_section_text` rejects a
reply that types a digit, uses an unknown key or reference, cites by number or by
author/year, or includes a URL — so a model can neither compute a result nor supply a
reference from memory. A rejected reply saves nothing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from src.core.schemas.manuscript import ManuscriptSection, StyleSettings
from src.llm.provider import (
    LLMConfig,
    LLMProvider,
    LLMRequest,
    LLMResponseError,
    parse_json_object,
)
from src.manuscript.document import AUTHOR_INPUT_RE, CITE_RE, FACT_RE, vancouver
from src.manuscript.evidence import EvidencePackage

PROMPT_VERSION = "manuscript-draft-v1"
HEADINGS = {
    "title": "Title",
    "abstract": "Abstract",
    "introduction": "Introduction",
    "methods": "Methods",
    "results": "Results",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusions": "Conclusions",
    "tables": "Tables",
    "figures": "Figure legends",
    "supplementary": "Supplementary material",
}
# Literal tokens with digits that are names, not findings (also accepted by the Auditor).
ALLOWED_LITERALS = re.compile(
    r"PRISMA[\s-]2020|PRISMA-S|RoB[\s-]2|ROBINS-I|"
    r"\b(?:Supplementary\s+)?(?:Table|Figure|File)\s+S?\d+\b",
    re.IGNORECASE,
)
MANUAL_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]|<sup>\s*\d")
AUTHOR_YEAR_RE = re.compile(r"\bet al\b\.?", re.IGNORECASE)
URL_RE = re.compile(r"https?://|www\.|doi:\s*10\.|\bPMID\b", re.IGNORECASE)


def _abstract(p: EvidencePackage) -> str:
    comp = "protocol.comparator" in p.facts
    lines = [
        "**Background:** [[AUTHOR INPUT: one or two sentences of clinical background]]",
        "",
        "**Objective:** To answer the pre-specified review question: {{protocol.question}}",
        "",
        "**Methods:** "
        + (
            "We searched {{search.databases}} up to {{search.last_date}} for "
            if "search.databases" in p.facts
            else "[[AUTHOR INPUT: no search log is recorded in this project; state the sources "
            "searched and the search dates]] We sought "
        )
        + "{{protocol.designs}} studies of {{protocol.intervention}}"
        + (" compared with {{protocol.comparator}}" if comp else "")
        + " in {{protocol.population}}. All statistics were computed by a deterministic "
        "statistics engine from human-approved extraction data.",
        "",
        "**Results:** {{prisma.studies_included}} studies ({{prisma.reports_included}} reports) "
        "were included.",
    ]
    for o in p.protocol.outcomes:
        if o.kind != "primary":
            continue
        for a in p.stats.analyses:
            if a.outcome_id != o.outcome_id:
                continue
            k = f"stats.{a.analysis_id}"
            if a.status == "pooled":
                lines[-1] += (
                    f" For {{{{{k}.outcome}}}} ({{{{{k}.measure_label}}}}), the pooled "
                    f"estimate from {{{{{k}.k}}}} studies was {{{{{k}.pooled_estimate}}}}."
                )
            elif f"{k}.single_study_estimate" in p.facts:
                lines[-1] += (
                    f" For {{{{{k}.outcome}}}}, only one study contributed data "
                    f"({{{{{k}.single_study_estimate}}}}; not pooled)."
                )
            else:
                lines[-1] += f" {{{{{k}.status}}}}"
    lines += [
        "",
        "**Conclusions:** [[AUTHOR INPUT: conclusions strictly supported by the "
        "results above, including their limitations]]",
    ]
    return "\n".join(lines)


def _methods(p: EvidencePackage) -> str:
    reg = (
        "The protocol was registered in PROSPERO ({{protocol.prospero_id}})."
        if "protocol.prospero_id" in p.facts
        else "[[AUTHOR INPUT: state the registration status of the protocol; no registration "
        "identifier is recorded in this project]]"
    )
    return "\n\n".join(
        [
            "**Protocol and reporting.** The review followed a protocol approved before the "
            "searches and is reported according to the PRISMA 2020 statement. " + reg,
            "**Eligibility criteria.** We included {{protocol.designs}} studies of "
            "{{protocol.population}} receiving {{protocol.intervention}}"
            + (" compared with {{protocol.comparator}}" if "protocol.comparator" in p.facts else "")
            + ". Inclusion criteria: {{protocol.inclusion}}. Exclusion criteria: "
            "{{protocol.exclusion}}. Outcomes: {{protocol.outcomes}}.",
            "**Information sources and search.** "
            + (
                "{{search.statement}}"
                if "search.statement" in p.facts
                else "[[AUTHOR INPUT: no search log is recorded in this project; describe the "
                "information sources, platforms, search dates and restrictions]]"
            ),
            "**Study selection.** {{screening.title_abstract}} {{screening.full_text}}",
            "**Data extraction.** {{extraction.methods}}",
            (
                "**Risk of bias.** {{rob.methods}}"
                if "rob.methods" in p.facts
                else "**Risk of bias.** [[AUTHOR INPUT: no approved risk-of-bias assessment is "
                "recorded]]"
            ),
            "**Synthesis.** {{stats.methods}} {{stats.software}}",
        ]
    )


def _results(p: EvidencePackage) -> str:
    included = "; ".join(f"@{k}" for k in sorted(p.included_ref_keys.values()))
    paras = [
        "**Study selection.** The searches identified {{prisma.identified}} records, of which "
        "{{prisma.duplicates_removed}} were duplicates. Of {{prisma.records_screened}} records "
        "screened, {{prisma.records_excluded}} were excluded on title and abstract. "
        "{{prisma.reports_sought}} reports were sought, {{prisma.reports_not_retrieved}} could "
        "not be retrieved and {{prisma.reports_assessed}} were assessed in full text; "
        "{{prisma.reports_excluded}} were excluded"
        + (" ({{prisma.exclusion_reasons}})" if "prisma.exclusion_reasons" in p.facts else "")
        + ". {{prisma.studies_included}} studies, described in {{prisma.reports_included}} "
        f"reports, were included [{included}] (Table 1).",
    ]
    if "rob.summary" in p.facts:
        paras.append("**Risk of bias.** {{rob.summary}}.")
    fig = 0
    for a in p.stats.analyses:
        k = f"stats.{a.analysis_id}"
        if a.status == "pooled":
            fig += 1
            s = (
                f"**{{{{{k}.outcome}}}}.** {{{{{k}.k}}}} studies contributed to the analysis of "
                f"{{{{{k}.outcome}}}} ({{{{{k}.measure_label}}}})"
            )
            if f"{k}.participants" in p.facts:
                s += f", including {{{{{k}.participants}}}} participants"
            s += (
                f". The pooled estimate was {{{{{k}.pooled_estimate}}}} ({{{{{k}.model}}}}"
                + (f"; {{{{{k}.p_value}}}}" if f"{k}.p_value" in p.facts else "")
                + f") (Figure {fig})."
            )
            if f"{k}.heterogeneity" in p.facts:
                s += f" Heterogeneity: {{{{{k}.heterogeneity}}}}" + (
                    f"; {{{{{k}.tau2}}}}." if f"{k}.tau2" in p.facts else "."
                )
            if f"{k}.prediction_interval" in p.facts:
                s += f" The {{{{{k}.prediction_interval}}}}."
            sens = [x for x in p.facts if x.startswith(f"{k}.sensitivity.")]
            if sens:
                s += " Sensitivity analysis: " + "; ".join(f"{{{{{x}}}}}" for x in sens) + "."
        elif f"{k}.single_study_estimate" in p.facts:
            fig += 1
            rids = p.facts[f"{k}.single_study_estimate"].record_ids
            cites = "; ".join(
                f"@{p.included_ref_keys[r]}" for r in rids if r in p.included_ref_keys
            )
            s = (
                f"**{{{{{k}.outcome}}}}.** {{{{{k}.status}}}} The only contributing study "
                + (f"[{cites}] " if cites else "")
                + f"reported {{{{{k}.single_study_estimate}}}} (Figure {fig})."
            )
        else:
            s = f"**{{{{{k}.outcome}}}}.** {{{{{k}.status}}}}"
        if f"{k}.excluded" in p.facts:
            s += f" Studies not contributing: {{{{{k}.excluded}}}}."
        paras.append(s)
    paras.append("A summary of all analyses is given in Table 2.")
    return "\n\n".join(paras)


def _discussion(p: EvidencePackage) -> str:
    principal = []
    for a in p.stats.analyses:
        k = f"stats.{a.analysis_id}"
        if a.status == "pooled":
            principal.append(
                f"the pooled {{{{{k}.measure_label}}}} of {{{{{k}.outcome}}}} was "
                f"{{{{{k}.pooled_estimate}}}}"
            )
    first = (
        "**Principal findings.** In this review, " + "; ".join(principal) + "."
        if principal
        else "**Principal findings.** No outcome could be pooled; the "
        "results are reported study by study."
    )
    return "\n\n".join(
        [
            first,
            "**Comparison with existing literature.** [[AUTHOR INPUT: compare these findings with "
            "verified, cited literature from the reference registry]]",
            "**Interpretation.** [[AUTHOR INPUT: clinical and methodological interpretation, "
            "clearly separating what the review demonstrated from hypotheses]]",
            "**Strengths.** Every extracted number was linked to a verbatim, page-located "
            "quotation and approved by an authenticated human reviewer, and all statistics were "
            "computed deterministically from a pre-specified, human-approved analysis plan.",
            "**Implications and future research.** [[AUTHOR INPUT: implications within the limits "
            "of the evidence, and research needs]]",
        ]
    )


def _limitations(p: EvidencePackage) -> str:
    items = [
        f"{{{{{k}}}}}" for k in p.facts if k.endswith(".limitations") and k.startswith("stats.")
    ]
    text = [
        "The statistics engine reported the following warnings and limitations: " + " ".join(items)
        if items
        else "The statistics engine reported no warnings or limitations for these analyses."
    ]
    from src.manuscript.evidence import screening_is_automated

    if screening_is_automated(p):
        text.append(
            "Screening relied on automated large-language-model passes with human "
            "adjudication of disagreements; these passes are not equivalent to two "
            "independent human reviewers."
        )
    text.append("[[AUTHOR INPUT: further limitations of the evidence and of the review]]")
    return "\n\n".join(text)


def _tables(p: EvidencePackage) -> str:
    return (
        "**Table 1.** Characteristics of the included studies (reports of the same study "
        "are grouped).\n\n{{table:study_characteristics}}\n\n**Table 2.** Summary of "
        "analyses.\n\n{{table:results_summary}}"
    )


def _figures(p: EvidencePackage) -> str:
    keys = [k for k in p.templates if k.startswith("figure:forest:")]
    order = [a.analysis_id for a in p.stats.analyses]
    keys.sort(key=lambda k: order.index(k.rsplit(":", 1)[1]))
    return (
        "\n\n".join(f"**Figure {i + 1}.** {{{{{k}}}}}" for i, k in enumerate(keys))
        or "No forest plots: no analysis produced a study-level or pooled estimate."
    )


def _supplementary(p: EvidencePackage) -> str:
    return (
        "Supplementary File 1: PRISMA-S search report (search log export). Supplementary "
        "File 2: statistics report, per-analysis tables and report ledgers "
        "(stats_output.json and report.md)."
    )


def template_sections(p: EvidencePackage) -> list[ManuscriptSection]:
    t = "{{protocol.title}}"
    builders: dict[str, Callable[[], str]] = {
        "title": lambda: t,
        "abstract": lambda: _abstract(p),
        "introduction": lambda: (
            "[[AUTHOR INPUT: background, burden of the condition and rationale, citing "
            "verified references from the reference registry]] The objective of this "
            "systematic review was to answer the following question: {{protocol.question}}"
        ),
        "methods": lambda: _methods(p),
        "results": lambda: _results(p),
        "discussion": lambda: _discussion(p),
        "limitations": lambda: _limitations(p),
        "conclusions": lambda: (
            "[[AUTHOR INPUT: conclusions supported by the results, without causal claims "
            "or recommendations the evidence does not support]]"
        ),
        "tables": lambda: _tables(p),
        "figures": lambda: _figures(p),
        "supplementary": lambda: _supplementary(p),
    }
    return [
        ManuscriptSection.model_validate(
            {"section_id": sid, "heading": HEADINGS[sid], "source": fn(), "origin": "template"}
        )
        for sid, fn in builders.items()
    ]


# --------------------------------------------------------------------------- validation


def strip_markers(text: str) -> str:
    text = FACT_RE.sub(" ", text)
    text = CITE_RE.sub(" ", text)
    return AUTHOR_INPUT_RE.sub(" ", text)


def validate_section_text(text: str, package: EvidencePackage) -> list[str]:
    """Hard rules for model-written source. Any problem rejects the reply."""
    problems: list[str] = []
    for key in FACT_RE.findall(text):
        if key not in package.facts and key not in package.templates:
            problems.append(f"unknown fact key {{{{{key}}}}}")
    for group in CITE_RE.findall(text):
        for k in (x.strip().lstrip("@") for x in group.split(";")):
            if k not in package.references:
                problems.append(
                    f"unknown reference [@{k}] (references come only from the registry)"
                )
    bare = ALLOWED_LITERALS.sub(" ", strip_markers(text))
    if MANUAL_CITATION_RE.search(bare):
        problems.append("citations must use [@ref] markers, not typed numbers")
    if re.search(r"\d", bare):
        snippet = re.search(r".{0,30}\d.{0,30}", bare)
        problems.append(
            "a number was typed instead of using a {{key}} marker: "
            f"{snippet.group(0).strip() if snippet else ''!r}"
        )
    if URL_RE.search(bare):
        problems.append("URLs, DOIs and PMIDs may not be typed (use the reference registry)")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if AUTHOR_YEAR_RE.search(sentence) and not CITE_RE.search(sentence):
            problems.append(f"author citation without a registry reference: {sentence[:80]!r}")
    return problems


# --------------------------------------------------------------------------- model drafting

SYSTEM = """You are drafting one section of a systematic review manuscript for a \
peer-reviewed medical journal. Write fluent, precise academic English; avoid generic, \
repetitive or promotional phrasing. Evidence integrity takes priority over style.

Hard rules (a reply that breaks any of them is discarded):
1. Never type a digit. Every number (counts, estimates, intervals, p-values, dates, \
percentages, years) must be a {{key}} marker copied exactly from FACTS. Do not calculate, \
round, combine or re-express any number; do not infer values that are not listed.
2. Cite only with [@ref_key] markers from REFERENCES. Never name authors with "et al.", never \
write DOIs, PMIDs, URLs or bibliographic details, and never cite from memory.
3. State only what FACTS and the cited REFERENCES support. Distinguish pooled estimates from \
single-study results, crude proportions from Kaplan-Meier or cumulative-incidence estimates, \
relative from absolute effects, and random-effects from common-effect results exactly as \
the fact descriptions state. Do not equate statistical significance with clinical \
importance, and do not present absence of evidence as evidence of absence.
4. Never invent mechanisms, causal relationships, study conclusions, recommendations, \
protocol registration, searches, reviewers or methods. Where the section needs content \
the evidence does not provide, write [[AUTHOR INPUT: what is needed]] instead.
5. Distinguish what this review demonstrated from interpretation or hypothesis.
Reply with one JSON object: {"text": "<section source>", "author_input_needed": ["..."]}."""

PURPOSE = {
    "introduction": "Background and rationale (cite only registry references), then the "
    "objective using {{protocol.question}}.",
    "discussion": "Structure: (1) principal findings; (2) comparison with verified existing "
    "literature (registry references only); (3) plausible clinical and methodological "
    "interpretation, labelled as interpretation; (4) strengths and limitations; (5) clinical "
    "implications within the evidence; (6) directions for future research.",
    "conclusions": "Conclusions strictly supported by the results; no recommendations beyond "
    "the evidence.",
    "abstract": "Structured abstract with the headings Background, Objective, Methods, "
    "Results and Conclusions (bold, followed by a colon).",
}


@dataclass(frozen=True, slots=True)
class DraftResult:
    section: ManuscriptSection
    prompt_sha256: str
    response_sha256: str


class ManuscriptDrafter:
    def __init__(self, provider: LLMProvider, config: LLMConfig) -> None:
        self.provider = provider
        self.config = config

    @property
    def identity(self) -> str:
        return f"{self.config.provider}/{self.config.model_id} ({PROMPT_VERSION})"

    def request(
        self,
        section_id: str,
        current: str,
        package: EvidencePackage,
        style: StyleSettings,
        instructions: str | None,
    ) -> LLMRequest:
        facts = [
            {"key": f.key, "meaning": f.description, "renders_as": f.text}
            for f in package.facts.values()
        ]
        facts += [
            {"key": k, "meaning": "table/figure (rendered from facts)"} for k in package.templates
        ]
        refs = [
            {
                "ref_key": k,
                "reference": vancouver(s.entry),
                "role": "included study"
                if s.entry.origin == "included_study_record"
                else "background literature",
                "excerpts": [e.text[:1500] for e in s.excerpts],
            }
            for k, s in package.references.items()
        ]
        prompt = json.dumps(
            {
                "section": section_id,
                "heading": HEADINGS.get(section_id, section_id),
                "purpose": PURPOSE.get(section_id, "Improve the prose without changing meaning."),
                "review_design": package.review_design.value,
                "style": style.model_dump(mode="json"),
                "author_instructions": instructions or "",
                "current_source": current,
                "FACTS": facts,
                "REFERENCES": refs,
            },
            ensure_ascii=False,
            indent=1,
        )
        return LLMRequest(
            system=SYSTEM,
            prompt=prompt,
            response_schema={
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "author_input_needed": {"type": "array"},
                },
            },
            temperature=None,
            max_output_tokens=self.config.max_output_tokens,
        )

    def draft(
        self,
        section_id: str,
        current: str,
        package: EvidencePackage,
        style: StyleSettings,
        instructions: str | None = None,
    ) -> DraftResult:
        req = self.request(section_id, current, package, style, instructions)
        resp = self.provider.complete(req)
        data = parse_json_object(resp.text)
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("the model returned no section text")
        needed = data.get("author_input_needed") or []
        if not isinstance(needed, list):
            raise LLMResponseError("author_input_needed must be a list")
        for item in needed:
            if isinstance(item, str) and item.strip() and item.strip() not in text:
                text += f" [[AUTHOR INPUT: {item.strip()}]]"
        problems = validate_section_text(text, package)
        if problems:
            raise LLMResponseError("draft rejected (nothing saved): " + "; ".join(problems[:5]))
        section = ManuscriptSection(
            section_id=section_id,  # type: ignore[arg-type]
            heading=HEADINGS.get(section_id, section_id),
            source=text.strip(),
            origin="llm",
        )
        return DraftResult(section, req.sha256, resp.sha256)
