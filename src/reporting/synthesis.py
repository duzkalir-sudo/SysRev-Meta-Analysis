"""Structured synthesis without meta-analysis: deterministic validation of human input.

The system never synthesises: groups, study-level findings, directions of effect and the
narrative summary are written and approved by an authenticated reviewer. This module checks
what can be checked mechanically and refuses what the evidence cannot support:

* every number in a study-level finding must appear in the evidence it links (accepted
  extraction items — reported text and verbatim quote — or approved arm values);
* every number in the narrative summary must appear in the study-level findings;
* a direction of effect needs a comparison (it is refused for single-group findings);
* the summary may not present a pooled, overall or combined estimate, may not count studies
  by statistical significance (vote counting based on significance is not an acceptable
  synthesis method — Cochrane Handbook ch. 12; SWiM item 3 lists vote counting based on the
  *direction* of effect), and may not claim a direction, or a consistent direction, that the
  approved study-level directions do not show; "most/majority of studies" statements need the
  vote-counting method.

These are lexical checks of human prose. They prevent the listed unsupported statements;
they do not establish that the narrative is a correct interpretation, which remains the
reviewer's responsibility (and the Auditor routes the text to human review).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from src.core.schemas.reporting import (
    DIRECTION_LABEL,
    METHOD_LABEL,
    VOTE_DIRECTIONS,
    SynthesisFinding,
    SynthesisGroup,
)
from src.manuscript.formatting import find_numbers, mask_identifiers

NEGATION = re.compile(r"\b(not|no|neither|nor|without|cannot|could not|unable|none)\b", re.I)
POOLED_RE = re.compile(
    r"\bpooled\b|\b(overall|summary|combined|average) (effect|estimate|result)s?\b|"
    r"\bmeta-?analys[ie]s (showed|found|demonstrated|indicated|suggested)\b",
    re.IGNORECASE,
)
SIGNIF_RE = re.compile(
    r"\b(statistically\s+)?(non-?|in)?significan\w*|\bp\s*[<=>≤≥]\s*\.?\d", re.IGNORECASE
)
DIRECTION_RE = re.compile(
    r"\bfavou?r\w*|\bimprov\w*|\breduc\w*|\bincreas\w*|\bdecreas\w*|\bbenefi\w*|\bharm\w*|"
    r"\beffective\w*|\bsuperior\w*|\binferior\w*|\bworse\b|\bbetter\b|\blower\b|\bhigher\b",
    re.IGNORECASE,
)
CONSISTENT_RE = re.compile(
    r"\bconsistent(ly)?\b|\ball (of )?(the )?(included )?studies\b|\bevery study\b|\buniform",
    re.IGNORECASE,
)
MAJORITY_RE = re.compile(
    r"\b(most|majority|minority|half) of (the )?(included )?studies\b|"
    r"\bmost studies\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+", text) if s.strip()]


def numbers_of(text: str) -> set[float]:
    return {f.value for f in find_numbers(mask_identifiers(text))}


def finding_problems(
    group: SynthesisGroup, evidence_text: dict[str, list[str]], comparative: bool
) -> list[str]:
    """``evidence_text`` maps record_id → the texts of its linked items/arms (reported text,
    verbatim quotes)."""
    problems: list[str] = []
    for f in group.findings:
        sources = set().union(*(numbers_of(t) for t in evidence_text.get(f.record_id, [])))
        stray = sorted(numbers_of(f.finding) - sources)
        if stray:
            problems.append(
                f"finding for report {f.record_id} states {', '.join(f'{x:g}' for x in stray)}, "
                "which does not appear in its linked evidence (no quantitative statement "
                "beyond the extracted data)"
            )
        if f.direction != "not_applicable" and not comparative:
            problems.append(
                f"finding for report {f.record_id}: a direction of effect needs a comparison; "
                "single-group findings have no direction of effect"
            )
    return problems


FAVOUR_RE = re.compile(
    r"\b(?:in\s+favou?r\s+of|favou?r(?:s|ed|ing)?)\s+(?:the\s+)?"
    r"([A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,3})",
    re.IGNORECASE,
)
COMPARATOR_WORDS = frozenset(
    {"comparator", "comparators", "control", "controls", "placebo", "sham", "usual",
     "standard", "comparison", "no-treatment", "untreated"}
)  # fmt: skip
INTERVENTION_WORDS = frozenset({"intervention", "interventions", "treatment", "experimental"})
STOP = frozenset({"the", "and", "with", "for", "of", "or", "arm", "group", "groups", "care"})


def _tokens(text: str | None) -> set[str]:
    return {t for t in re.findall(r"[a-z][\w-]+", (text or "").lower()) if t not in STOP}


@dataclass(frozen=True, slots=True)
class Arms:
    """Words that name each side of the comparison (protocol wording plus generic terms)."""

    intervention: frozenset[str]
    comparator: frozenset[str]

    @classmethod
    def of(cls, intervention: str | None, comparator: str | None) -> Arms:
        i = _tokens(intervention) | INTERVENTION_WORDS
        c = _tokens(comparator) | COMPARATOR_WORDS
        return cls(frozenset(i - c), frozenset(c - i))

    def side(self, phrase: str) -> str | None:
        words = _tokens(phrase)
        hit_i, hit_c = bool(words & self.intervention), bool(words & self.comparator)
        if hit_i == hit_c:
            return None  # unresolved: cannot be verified mechanically
        return "favours_intervention" if hit_i else "favours_comparator"


def selected(group: SynthesisGroup) -> list[SynthesisFinding]:
    """One result per study/cohort (the voting units; checked against study identities and
    participant overlap by :func:`src.reporting.state.group_problems`)."""
    return [f for f in group.findings if f.selected_for_unit]


def summary_problems(group: SynthesisGroup, arms: Arms | None = None) -> list[str]:
    """Statements the structured, approved data contradict or cannot support (refused)."""
    arms = arms or Arms.of(None, None)
    problems: list[str] = []
    text = group.summary
    for s in _sentences(text):
        if POOLED_RE.search(s) and not NEGATION.search(s):
            problems.append(
                f"the summary presents a pooled/overall estimate ({s[:80]!r}); no meta-analysis "
                "was performed for this group"
            )
        if SIGNIF_RE.search(s):
            problems.append(
                f"the summary counts or describes statistical significance ({s[:80]!r}); "
                "vote counting based on statistical significance is not an acceptable "
                "synthesis method"
            )
    units = selected(group)
    allowed = set().union(*(numbers_of(f.finding) for f in group.findings))
    if group.synthesis_method == "vote_counting_direction_of_effect":
        allowed |= {float(v) for v in direction_counts(group).values()}
        allowed.add(float(len(units)))
    stray = sorted(numbers_of(text) - allowed)
    if stray:
        problems.append(
            "the summary states "
            + ", ".join(f"{x:g}" for x in stray)
            + ", which does not appear in the approved study-level findings"
        )
    directions = [f.direction for f in units]
    votes = [d for d in directions if d in VOTE_DIRECTIONS]
    for s in _sentences(text):
        negated = bool(NEGATION.search(s))
        if DIRECTION_RE.search(s) and not negated and not votes:
            problems.append(
                f"directional statement ({s[:80]!r}) but no selected study result was judged "
                "to favour either group"
            )
        if CONSISTENT_RE.search(s) and DIRECTION_RE.search(s) and len(set(directions)) > 1:
            problems.append(
                f"the summary claims a consistent direction ({s[:80]!r}) but the selected "
                "study-level directions differ: " + ", ".join(sorted(set(directions)))
            )
        for m in FAVOUR_RE.finditer(s):
            claimed = arms.side(m.group(1))
            if claimed is None or negated:
                continue
            who = "intervention" if claimed == "favours_intervention" else "comparator"
            if claimed not in votes:
                problems.append(
                    f"the summary says results favoured the {who} ({s[:80]!r}), but no selected "
                    f"study result was judged to favour the {who}: this contradicts the "
                    "recorded directions"
                )
            elif CONSISTENT_RE.search(s) and any(d != claimed for d in directions):
                problems.append(
                    f"the summary says all results consistently favoured the {who} "
                    f"({s[:80]!r}), but the recorded directions are "
                    + ", ".join(sorted(set(directions)))
                )
        if MAJORITY_RE.search(s) and group.synthesis_method != "vote_counting_direction_of_effect":
            problems.append(
                f"'{MAJORITY_RE.search(s).group(0)}' is a vote count; use the vote-counting "  # type: ignore[union-attr]
                "method (direction of effect) so the counts are tabulated from approved "
                "directions"
            )
    return problems


FAVOUR_VERB_RE = re.compile(r"favou?r\w*", re.IGNORECASE)


def unverifiable_statements(group: SynthesisGroup, arms: Arms) -> list[str]:
    """Directional wording the structured directions can neither confirm nor contradict
    (e.g. 'X reduced complications', or a favoured side that cannot be identified). These
    are interpretations: the Auditor routes them to an explicit human review.

    A sentence is exempt only if *every* directional cue in it was checked. A favour clause
    whose side was identified (and so compared with the recorded directions by
    :func:`summary_problems`) exempts only its own verb: any other directional word in the
    sentence — including words inside the clause's target phrase — still needs review, so
    adding a checkable clause never hides an otherwise unchanged interpretation. A negated
    sentence or an unidentified side always needs review."""
    out: list[str] = []
    for s in _sentences(group.summary):
        if not DIRECTION_RE.search(s):
            continue
        if NEGATION.search(s):
            out.append(s.strip())
            continue
        residual = list(s)
        checked_all = True
        for m in FAVOUR_RE.finditer(s):
            if arms.side(m.group(1)) is None:
                checked_all = False
                break
            verb = FAVOUR_VERB_RE.search(m.group(0))
            assert verb is not None
            for i in range(m.start() + verb.start(), m.start() + verb.end()):
                residual[i] = " "
        if not checked_all or DIRECTION_RE.search("".join(residual)):
            out.append(s.strip())
    return out


def direction_counts(group: SynthesisGroup) -> dict[str, int]:
    """Directions of the selected result of each study/cohort (one per voting unit)."""
    return dict(sorted(Counter(f.direction for f in selected(group)).items()))


def direction_text(group: SynthesisGroup) -> str:
    """The vote-count statement, generated from the structured directions only."""
    counts = direction_counts(group)
    n = sum(counts.values())
    fav_i = counts.get("favours_intervention", 0)
    fav_c = counts.get("favours_comparator", 0)
    other = [f"{k} {DIRECTION_LABEL[d]}" for d, k in counts.items() if d not in VOTE_DIRECTIONS]
    text = (
        f"Of {n} independent study units (one selected result per study or cohort), {fav_i} "
        f"favoured the intervention and {fav_c} favoured the comparator in the observed "
        "direction of effect"
    )
    if other:
        text += "; " + "; ".join(other)
    return text + (
        " (vote counting based on the direction of effect as judged by the reviewers, "
        "irrespective of statistical significance; no statistical test was performed and no "
        "effect size is implied)"
    )


def methods_text(group: SynthesisGroup, outcome_name: str) -> str:
    basis = ", ".join(b.replace("_", " ") for b in group.grouping_basis)
    parts = [
        f"For {group.title} ({outcome_name}), studies were grouped by {basis}: "
        f"{group.rationale.rstrip('.')}.",
        f"Meta-analysis was not undertaken ({group.reason_not_pooled.replace('_', ' ')}: "
        f"{group.reason_note.rstrip('.')}).",
        f"Standardised metric: {group.standardised_metric.rstrip('.')}.",
        f"Synthesis method: {METHOD_LABEL[group.synthesis_method]}; "
        f"{group.method_justification.rstrip('.')}.",
    ]
    if group.protocol_change:
        parts.append(
            "The grouping changed from the protocol: "
            f"{(group.protocol_change_rationale or '').rstrip('.')}."
        )
    else:
        parts.append("The grouping was the one planned in the protocol.")
    if group.selection_rule:
        parts.append(
            "Where a study or cohort contributed several reports or results, one result "
            f"represented it: {group.selection_rule.rstrip('.')}."
        )
    if group.prioritisation_criteria:
        parts.append(f"Prioritisation: {group.prioritisation_criteria.rstrip('.')}.")
    if group.heterogeneity_investigation:
        parts.append(f"Heterogeneity: {group.heterogeneity_investigation.rstrip('.')}.")
    parts.append(f"Certainty: {group.certainty_method.rstrip('.')}.")
    parts.append(f"Presentation: {group.presentation.rstrip('.')}.")
    return " ".join(parts)
