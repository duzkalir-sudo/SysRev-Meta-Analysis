"""Reporting-standard catalogues: PRISMA 2020 (main and abstract checklists) and SWiM.

Item wording is reproduced verbatim from the primary publications, retrieved from PubMed
Central on 2026-10-30 (both articles are distributed under CC BY 4.0, which permits
reproduction with attribution):

* PRISMA 2020: Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an
  updated guideline for reporting systematic reviews. BMJ 2021;372:n71 (Table 1, 27 items
  with sub-items; Table 2, PRISMA 2020 for Abstracts, 12 items). doi:10.1136/bmj.n71
* SWiM: Campbell M, McKenzie JE, Sowden A, et al. Synthesis without meta-analysis (SWiM) in
  systematic reviews: reporting guideline. BMJ 2020;368:l6890 (Table 1, nine items; item 1
  has sub-items 1a and 1b). doi:10.1136/bmj.l6890

Each item also lists, for *deterministic location suggestions only*, the manuscript
sections where it is usually reported, evidence-package keys whose presence in a section
marks a candidate location, and the evidence-fingerprint components whose change makes a
human decision on the item stale. A suggestion never marks an item as reported: PRISMA
2020 itself notes that the suggested location is not prescriptive and that the checklist
must not be used to judge the conduct of a review.
"""

from __future__ import annotations

from dataclasses import dataclass

PRISMA_REFERENCE = (
    "Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an updated guideline "
    "for reporting systematic reviews. BMJ 2021;372:n71. doi:10.1136/bmj.n71 (CC BY 4.0)"
)
SWIM_REFERENCE = (
    "Campbell M, McKenzie JE, Sowden A, et al. Synthesis without meta-analysis (SWiM) in "
    "systematic reviews: reporting guideline. BMJ 2020;368:l6890. doi:10.1136/bmj.l6890 "
    "(CC BY 4.0)"
)
STANDARD_VERSION = {
    "prisma_2020_main": "PRISMA 2020 checklist (Page et al., BMJ 2021;372:n71, Table 1)",
    "prisma_2020_abstract": "PRISMA 2020 for Abstracts (Page et al., BMJ 2021;372:n71, Table 2)",
    "swim": "SWiM reporting guideline (Campbell et al., BMJ 2020;368:l6890, Table 1)",
}


@dataclass(frozen=True, slots=True)
class ChecklistItem:
    checklist: str
    item_id: str
    section: str  # the checklist's own section heading
    topic: str
    text: str  # verbatim requirement
    sections: tuple[str, ...]  # manuscript sections where a candidate location is sought
    cue_keys: tuple[str, ...] = ()  # evidence keys / key prefixes marking a candidate
    cue_terms: tuple[str, ...] = ()  # lower-case phrases marking a candidate (weak signal)
    evidence: tuple[str, ...] = ()  # fingerprint components the item depends on


M = "prisma_2020_main"
A = "prisma_2020_abstract"
S = "swim"

PRISMA_MAIN: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        M,
        "1",
        "Title",
        "Title",
        "Identify the report as a systematic review.",
        ("title",),
        cue_terms=("systematic review",),
    ),
    ChecklistItem(
        M,
        "2",
        "Abstract",
        "Abstract",
        "See the PRISMA 2020 for Abstracts checklist.",
        ("abstract",),
    ),
    ChecklistItem(
        M,
        "3",
        "Introduction",
        "Rationale",
        "Describe the rationale for the review in the context of existing knowledge.",
        ("introduction",),
    ),
    ChecklistItem(
        M,
        "4",
        "Introduction",
        "Objectives",
        "Provide an explicit statement of the objective(s) or question(s) the review addresses.",
        ("introduction",),
        ("protocol.question",),
        evidence=("protocol_sha256",),
    ),
    ChecklistItem(
        M,
        "5",
        "Methods",
        "Eligibility criteria",
        "Specify the inclusion and exclusion criteria for the review and how studies were "
        "grouped for the syntheses.",
        ("methods",),
        ("protocol.inclusion", "protocol.exclusion", "synthesis."),
        evidence=("protocol_sha256", "reporting_sha256"),
    ),
    ChecklistItem(
        M,
        "6",
        "Methods",
        "Information sources",
        "Specify all databases, registers, websites, organisations, reference lists and other "
        "sources searched or consulted to identify studies. Specify the date when each source "
        "was last searched or consulted.",
        ("methods",),
        ("search.statement", "search.databases"),
        evidence=("search_log_sha256", "protocol_sha256"),
    ),
    ChecklistItem(
        M,
        "7",
        "Methods",
        "Search strategy",
        "Present the full search strategies for all databases, registers and websites, "
        "including any filters and limits used.",
        ("methods", "supplementary"),
        ("search.statement",),
        ("search strateg", "prisma-s"),
        evidence=("search_log_sha256", "protocol_sha256"),
    ),
    ChecklistItem(
        M,
        "8",
        "Methods",
        "Selection process",
        "Specify the methods used to decide whether a study met the inclusion criteria of the "
        "review, including how many reviewers screened each record and each report retrieved, "
        "whether they worked independently, and if applicable, details of automation tools "
        "used in the process.",
        ("methods",),
        ("screening.title_abstract", "screening.full_text"),
        evidence=("prisma_sha256",),
    ),
    ChecklistItem(
        M,
        "9",
        "Methods",
        "Data collection process",
        "Specify the methods used to collect data from reports, including how many reviewers "
        "collected data from each report, whether they worked independently, any processes "
        "for obtaining or confirming data from study investigators, and if applicable, "
        "details of automation tools used in the process.",
        ("methods",),
        ("extraction.methods",),
        evidence=("extraction_input_sha256",),
    ),
    ChecklistItem(
        M,
        "10a",
        "Methods",
        "Data items",
        "List and define all outcomes for which data were sought. Specify whether all results "
        "that were compatible with each outcome domain in each study were sought (e.g. for "
        "all measures, time points, analyses), and if not, the methods used to decide which "
        "results to collect.",
        ("methods",),
        ("protocol.outcomes",),
        evidence=("protocol_sha256",),
    ),
    ChecklistItem(
        M,
        "10b",
        "Methods",
        "Data items",
        "List and define all other variables for which data were sought (e.g. participant and "
        "intervention characteristics, funding sources). Describe any assumptions made about "
        "any missing or unclear information.",
        ("methods",),
        evidence=("extraction_input_sha256",),
    ),
    ChecklistItem(
        M,
        "11",
        "Methods",
        "Study risk of bias assessment",
        "Specify the methods used to assess risk of bias in the included studies, including "
        "details of the tool(s) used, how many reviewers assessed each study and whether they "
        "worked independently, and if applicable, details of automation tools used in the "
        "process.",
        ("methods",),
        ("rob.methods",),
        evidence=("rob_sha256",),
    ),
    ChecklistItem(
        M,
        "12",
        "Methods",
        "Effect measures",
        "Specify for each outcome the effect measure(s) (e.g. risk ratio, mean difference) "
        "used in the synthesis or presentation of results.",
        ("methods",),
        ("stats.methods",),
        evidence=("analysis_plan_sha256",),
    ),
    ChecklistItem(
        M,
        "13a",
        "Methods",
        "Synthesis methods",
        "Describe the processes used to decide which studies were eligible for each synthesis "
        "(e.g. tabulating the study intervention characteristics and comparing against the "
        "planned groups for each synthesis (item #5)).",
        ("methods",),
        ("stats.methods", "synthesis.methods"),
        evidence=("analysis_plan_sha256", "reporting_sha256"),
    ),
    ChecklistItem(
        M,
        "13b",
        "Methods",
        "Synthesis methods",
        "Describe any methods required to prepare the data for presentation or synthesis, "
        "such as handling of missing summary statistics, or data conversions.",
        ("methods",),
        ("stats.methods",),
        evidence=("analysis_plan_sha256",),
    ),
    ChecklistItem(
        M,
        "13c",
        "Methods",
        "Synthesis methods",
        "Describe any methods used to tabulate or visually display results of individual "
        "studies and syntheses.",
        ("methods", "tables", "figures"),
        ("table:", "figure:"),
        evidence=("analysis_plan_sha256", "reporting_sha256"),
    ),
    ChecklistItem(
        M,
        "13d",
        "Methods",
        "Synthesis methods",
        "Describe any methods used to synthesise results and provide a rationale for the "
        "choice(s). If meta-analysis was performed, describe the model(s), method(s) to "
        "identify the presence and extent of statistical heterogeneity, and software "
        "package(s) used.",
        ("methods",),
        ("stats.methods", "stats.software", "synthesis.methods"),
        evidence=("analysis_plan_sha256", "stats_output_sha256", "reporting_sha256"),
    ),
    ChecklistItem(
        M,
        "13e",
        "Methods",
        "Synthesis methods",
        "Describe any methods used to explore possible causes of heterogeneity among study "
        "results (e.g. subgroup analysis, meta-regression).",
        ("methods",),
        evidence=("analysis_plan_sha256",),
    ),
    ChecklistItem(
        M,
        "13f",
        "Methods",
        "Synthesis methods",
        "Describe any sensitivity analyses conducted to assess robustness of the synthesised "
        "results.",
        ("methods",),
        ("stats.methods",),
        ("sensitivity",),
        evidence=("analysis_plan_sha256",),
    ),
    ChecklistItem(
        M,
        "14",
        "Methods",
        "Reporting bias assessment",
        "Describe any methods used to assess risk of bias due to missing results in a "
        "synthesis (arising from reporting biases).",
        ("methods",),
        ("grade.methods",),
        ("reporting bias", "publication bias"),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        M,
        "15",
        "Methods",
        "Certainty assessment",
        "Describe any methods used to assess certainty (or confidence) in the body of "
        "evidence for an outcome.",
        ("methods",),
        ("grade.methods",),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        M,
        "16a",
        "Results",
        "Study selection",
        "Describe the results of the search and selection process, from the number of records "
        "identified in the search to the number of studies included in the review, ideally "
        "using a flow diagram.",
        ("results", "figures"),
        ("prisma.", "figure:prisma"),
        evidence=("prisma_sha256",),
    ),
    ChecklistItem(
        M,
        "16b",
        "Results",
        "Study selection",
        "Cite studies that might appear to meet the inclusion criteria, but which were "
        "excluded, and explain why they were excluded.",
        ("results", "supplementary"),
        ("prisma.exclusion_reasons",),
        evidence=("prisma_sha256",),
    ),
    ChecklistItem(
        M,
        "17",
        "Results",
        "Study characteristics",
        "Cite each included study and present its characteristics.",
        ("results", "tables"),
        ("table:study_characteristics",),
        evidence=("extraction_input_sha256", "references_sha256"),
    ),
    ChecklistItem(
        M,
        "18",
        "Results",
        "Risk of bias in studies",
        "Present assessments of risk of bias for each included study.",
        ("results", "tables"),
        ("rob.summary", "study."),
        evidence=("rob_sha256",),
    ),
    ChecklistItem(
        M,
        "19",
        "Results",
        "Results of individual studies",
        "For all outcomes, present, for each study: (a) summary statistics for each group "
        "(where appropriate) and (b) an effect estimate and its precision (e.g. "
        "confidence/credible interval), ideally using structured tables or plots.",
        ("results", "tables", "figures"),
        ("figure:forest:", "synthesis."),
        evidence=("stats_output_sha256", "reporting_sha256"),
    ),
    ChecklistItem(
        M,
        "20a",
        "Results",
        "Results of syntheses",
        "For each synthesis, briefly summarise the characteristics and risk of bias among "
        "contributing studies.",
        ("results",),
        ("rob.summary", "synthesis."),
        evidence=("rob_sha256", "reporting_sha256"),
    ),
    ChecklistItem(
        M,
        "20b",
        "Results",
        "Results of syntheses",
        "Present results of all statistical syntheses conducted. If meta-analysis was done, "
        "present for each the summary estimate and its precision (e.g. confidence/credible "
        "interval) and measures of statistical heterogeneity. If comparing groups, describe "
        "the direction of the effect.",
        ("results", "tables"),
        ("stats.",),
        evidence=("stats_output_sha256",),
    ),
    ChecklistItem(
        M,
        "20c",
        "Results",
        "Results of syntheses",
        "Present results of all investigations of possible causes of heterogeneity among "
        "study results.",
        ("results",),
        evidence=("stats_output_sha256",),
    ),
    ChecklistItem(
        M,
        "20d",
        "Results",
        "Results of syntheses",
        "Present results of all sensitivity analyses conducted to assess the robustness of the "
        "synthesised results.",
        ("results",),
        cue_terms=("sensitivity",),
        evidence=("stats_output_sha256",),
    ),
    ChecklistItem(
        M,
        "21",
        "Results",
        "Reporting biases",
        "Present assessments of risk of bias due to missing results (arising from reporting "
        "biases) for each synthesis assessed.",
        ("results", "tables"),
        ("grade.",),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        M,
        "22",
        "Results",
        "Certainty of evidence",
        "Present assessments of certainty (or confidence) in the body of evidence for each "
        "outcome assessed.",
        ("results", "tables"),
        ("grade.", "table:sof", "table:evidence_summary"),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        M,
        "23a",
        "Discussion",
        "Discussion",
        "Provide a general interpretation of the results in the context of other evidence.",
        ("discussion",),
    ),
    ChecklistItem(
        M,
        "23b",
        "Discussion",
        "Discussion",
        "Discuss any limitations of the evidence included in the review.",
        ("limitations", "discussion"),
        ("grade.limitations", "stats."),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        M,
        "23c",
        "Discussion",
        "Discussion",
        "Discuss any limitations of the review processes used.",
        ("limitations", "discussion"),
    ),
    ChecklistItem(
        M,
        "23d",
        "Discussion",
        "Discussion",
        "Discuss implications of the results for practice, policy, and future research.",
        ("discussion", "conclusions"),
        cue_terms=("implication",),
    ),
    ChecklistItem(
        M,
        "24a",
        "Other information",
        "Registration and protocol",
        "Provide registration information for the review, including register name and "
        "registration number, or state that the review was not registered.",
        ("methods", "abstract"),
        ("protocol.prospero_id",),
        ("regist",),
        evidence=("protocol_sha256",),
    ),
    ChecklistItem(
        M,
        "24b",
        "Other information",
        "Registration and protocol",
        "Indicate where the review protocol can be accessed, or state that a protocol was not "
        "prepared.",
        ("methods", "supplementary"),
        cue_terms=("protocol",),
        evidence=("protocol_sha256",),
    ),
    ChecklistItem(
        M,
        "24c",
        "Other information",
        "Registration and protocol",
        "Describe and explain any amendments to information provided at registration or in "
        "the protocol.",
        ("methods",),
        cue_terms=("amend",),
        evidence=("protocol_sha256",),
    ),
    ChecklistItem(
        M,
        "25",
        "Other information",
        "Support",
        "Describe sources of financial or non-financial support for the review, and the role "
        "of the funders or sponsors in the review.",
        ("supplementary", "methods"),
        cue_terms=("funding", "funded", "support"),
    ),
    ChecklistItem(
        M,
        "26",
        "Other information",
        "Competing interests",
        "Declare any competing interests of review authors.",
        ("supplementary",),
        cue_terms=("competing interest", "conflict of interest"),
    ),
    ChecklistItem(
        M,
        "27",
        "Other information",
        "Availability of data, code, and other materials",
        "Report which of the following are publicly available and where they can be found: "
        "template data collection forms; data extracted from included studies; data used for "
        "all analyses; analytic code; any other materials used in the review.",
        ("supplementary",),
        cue_terms=("available", "repository", "data availability"),
    ),
)

PRISMA_ABSTRACT: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        A,
        "A1",
        "Title",
        "Title",
        "Identify the report as a systematic review.",
        ("title",),
        cue_terms=("systematic review",),
    ),
    ChecklistItem(
        A,
        "A2",
        "Background",
        "Objectives",
        "Provide an explicit statement of the main objective(s) or question(s) the review "
        "addresses.",
        ("abstract",),
        ("protocol.question",),
        evidence=("protocol_sha256",),
    ),
    ChecklistItem(
        A,
        "A3",
        "Methods",
        "Eligibility criteria",
        "Specify the inclusion and exclusion criteria for the review.",
        ("abstract",),
        ("protocol.designs", "protocol.population"),
        evidence=("protocol_sha256",),
    ),
    ChecklistItem(
        A,
        "A4",
        "Methods",
        "Information sources",
        "Specify the information sources (e.g. databases, registers) used to identify studies "
        "and the date when each was last searched.",
        ("abstract",),
        ("search.databases", "search.last_date"),
        evidence=("search_log_sha256",),
    ),
    ChecklistItem(
        A,
        "A5",
        "Methods",
        "Risk of bias",
        "Specify the methods used to assess risk of bias in the included studies.",
        ("abstract",),
        ("rob.",),
        evidence=("rob_sha256",),
    ),
    ChecklistItem(
        A,
        "A6",
        "Methods",
        "Synthesis of results",
        "Specify the methods used to present and synthesise results.",
        ("abstract",),
        cue_terms=("statistic", "synthes"),
    ),
    ChecklistItem(
        A,
        "A7",
        "Results",
        "Included studies",
        "Give the total number of included studies and participants and summarise relevant "
        "characteristics of studies.",
        ("abstract",),
        ("prisma.studies_included",),
        evidence=("prisma_sha256",),
    ),
    ChecklistItem(
        A,
        "A8",
        "Results",
        "Synthesis of results",
        "Present results for main outcomes, preferably indicating the number of included "
        "studies and participants for each. If meta-analysis was done, report the summary "
        "estimate and confidence/credible interval. If comparing groups, indicate the "
        "direction of the effect (i.e. which group is favoured).",
        ("abstract",),
        ("stats.",),
        evidence=("stats_output_sha256",),
    ),
    ChecklistItem(
        A,
        "A9",
        "Discussion",
        "Limitations of evidence",
        "Provide a brief summary of the limitations of the evidence included in the review "
        "(e.g. study risk of bias, inconsistency and imprecision).",
        ("abstract",),
        ("grade.",),
        ("limitation", "certainty"),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        A,
        "A10",
        "Discussion",
        "Interpretation",
        "Provide a general interpretation of the results and important implications.",
        ("abstract",),
        cue_terms=("conclusion",),
    ),
    ChecklistItem(
        A,
        "A11",
        "Other",
        "Funding",
        "Specify the primary source of funding for the review.",
        ("abstract",),
        cue_terms=("funding", "funded"),
    ),
    ChecklistItem(
        A,
        "A12",
        "Other",
        "Registration",
        "Provide the register name and registration number.",
        ("abstract",),
        ("protocol.prospero_id",),
        ("regist",),
        evidence=("protocol_sha256",),
    ),
)

SWIM_ITEMS: tuple[ChecklistItem, ...] = (
    ChecklistItem(
        S,
        "1a",
        "Methods",
        "Grouping studies for synthesis",
        "Provide a description of, and rationale for, the groups used in the synthesis (eg, "
        "groupings of populations, interventions, outcomes, study design)",
        ("methods",),
        ("synthesis.methods",),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        S,
        "1b",
        "Methods",
        "Grouping studies for synthesis",
        "Detail and provide rationale for any changes made subsequent to the protocol in the "
        "groups used in the synthesis",
        ("methods",),
        ("synthesis.methods",),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        S,
        "2",
        "Methods",
        "Describe the standardised metric and transformation methods used",
        "Describe the standardised metric for each outcome. Explain why the metric(s) was "
        "chosen and describe any methods used to transform the intervention effects, as "
        "reported in the study, to the standardised metric, citing any methodological "
        "guidance consulted",
        ("methods",),
        ("synthesis.methods",),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        S,
        "3",
        "Methods",
        "Describe the synthesis methods",
        "Describe and justify the methods used to synthesise the effects for each outcome when "
        "it was not possible to undertake a meta-analysis of effect estimates",
        ("methods",),
        ("synthesis.methods",),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        S,
        "4",
        "Methods",
        "Criteria used to prioritise results for summary and synthesis",
        "Where applicable, provide the criteria used, with supporting justification, to select "
        "the particular studies, or a particular study, for the main synthesis or to draw "
        "conclusions from the synthesis (eg, based on study design, risk of bias assessments, "
        "directness in relation to the review question)",
        ("methods",),
        ("synthesis.methods",),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        S,
        "5",
        "Methods",
        "Investigation of heterogeneity in reported effects",
        "State the method(s) used to examine heterogeneity in reported effects when it was not "
        "possible to undertake a meta-analysis of effect estimates and its extensions to "
        "investigate heterogeneity",
        ("methods",),
        ("synthesis.methods",),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        S,
        "6",
        "Methods",
        "Certainty of evidence",
        "Describe the methods used to assess the certainty of the synthesis findings",
        ("methods",),
        ("synthesis.methods", "grade.methods"),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        S,
        "7",
        "Methods",
        "Data presentation methods",
        "Describe the graphical and tabular methods used to present the effects (eg, tables, "
        "forest plots, harvest plots). Specify key study characteristics (eg, study design, "
        "risk of bias) used to order the studies, in the text and any tables or graphs, "
        "clearly referencing the studies included",
        ("methods", "tables"),
        ("synthesis.methods", "table:synthesis"),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        S,
        "8",
        "Results",
        "Reporting results",
        "For each comparison and outcome, provide a description of the synthesised findings "
        "and the certainty of the findings. Describe the result in language that is "
        "consistent with the question the synthesis addresses, and indicate which studies "
        "contribute to the synthesis",
        ("results",),
        ("synthesis.",),
        evidence=("reporting_sha256",),
    ),
    ChecklistItem(
        S,
        "9",
        "Discussion",
        "Limitations of the synthesis",
        "Report the limitations of the synthesis methods used and/or the groupings used in "
        "the synthesis and how these affect the conclusions that can be drawn in relation to "
        "the original review question",
        ("limitations", "discussion"),
        ("synthesis.",),
        evidence=("reporting_sha256",),
    ),
)

CATALOGUES: dict[str, tuple[ChecklistItem, ...]] = {
    M: PRISMA_MAIN,
    A: PRISMA_ABSTRACT,
    S: SWIM_ITEMS,
}


def item(checklist: str, item_id: str) -> ChecklistItem:
    for it in CATALOGUES.get(checklist, ()):
        if it.item_id == item_id:
            return it
    raise KeyError(f"{checklist} has no item {item_id}")


assert len(PRISMA_MAIN) == 42, "PRISMA 2020: 27 items, 42 entries with sub-items"
assert len(PRISMA_ABSTRACT) == 12
assert len(SWIM_ITEMS) == 10, "SWiM: nine items, ten entries with 1a/1b"
