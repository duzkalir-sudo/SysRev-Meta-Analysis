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

from src.core.schemas.reporting import DIRECTION_LABEL, METHOD_LABEL, SynthesisGroup
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
DIRECTIONAL = frozenset({"favours_intervention", "favours_comparator"})


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


def summary_problems(group: SynthesisGroup) -> list[str]:
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
    allowed = set().union(*(numbers_of(f.finding) for f in group.findings))
    if group.synthesis_method == "vote_counting_direction_of_effect":
        allowed |= {float(v) for v in direction_counts(group).values()}
        allowed.add(float(len(group.findings)))
    stray = sorted(numbers_of(text) - allowed)
    if stray:
        problems.append(
            "the summary states "
            + ", ".join(f"{x:g}" for x in stray)
            + ", which does not appear in the approved study-level findings"
        )
    directions = [f.direction for f in group.findings]
    directional = [d for d in directions if d in DIRECTIONAL]
    for s in _sentences(text):
        if DIRECTION_RE.search(s) and not NEGATION.search(s) and not directional:
            problems.append(
                f"directional statement ({s[:80]!r}) but no study-level finding was judged to "
                "favour either group"
            )
        if CONSISTENT_RE.search(s) and DIRECTION_RE.search(s) and len(set(directions)) > 1:
            problems.append(
                f"the summary claims a consistent direction ({s[:80]!r}) but the study-level "
                "directions differ: " + ", ".join(sorted(set(directions)))
            )
        if MAJORITY_RE.search(s) and group.synthesis_method != "vote_counting_direction_of_effect":
            problems.append(
                f"'{MAJORITY_RE.search(s).group(0)}' is a vote count; use the vote-counting "  # type: ignore[union-attr]
                "method (direction of effect) so the counts are tabulated from approved "
                "directions"
            )
    return problems


def direction_counts(group: SynthesisGroup) -> dict[str, int]:
    return dict(sorted(Counter(f.direction for f in group.findings).items()))


def direction_text(group: SynthesisGroup) -> str:
    counts = direction_counts(group)
    parts = [f"{n} {DIRECTION_LABEL[d]}" for d, n in counts.items()]
    return (
        f"Of {len(group.findings)} studies, " + "; ".join(parts) + " (vote counting based on "
        "the direction of effect as judged by the reviewers; no statistical test was performed "
        "and no effect size is implied)"
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
    if group.prioritisation_criteria:
        parts.append(f"Prioritisation: {group.prioritisation_criteria.rstrip('.')}.")
    if group.heterogeneity_investigation:
        parts.append(f"Heterogeneity: {group.heterogeneity_investigation.rstrip('.')}.")
    parts.append(f"Certainty: {group.certainty_method.rstrip('.')}.")
    parts.append(f"Presentation: {group.presentation.rstrip('.')}.")
    return " ".join(parts)
