"""Advisory claim-support review by a separately configured model.

For a citation the Auditor could not settle (``claim_support`` findings that require human
review), the model is shown only the manuscript sentence and the human-recorded excerpt(s) of
the cited source, and asked whether the excerpt supports the sentence. It must not use
outside knowledge. The answer is stored as a :class:`ClaimReview` labelled as automated
output: it never changes the finding, never counts as an acknowledgement, and can never
approve a manuscript.
"""

from __future__ import annotations

import json

from src.core.schemas.audit import AuditFinding
from src.core.schemas.manuscript import ClaimReview
from src.llm.provider import LLMConfig, LLMProvider, LLMRequest, LLMResponseError, parse_json_object

PROMPT_VERSION = "claim-review-v1"
SYSTEM = (
    "You check citations in a medical manuscript. You are given one sentence and excerpts "
    "from the source it cites. Judge ONLY from the excerpts whether they support every "
    "factual element of the sentence (numbers, direction, population, outcome, certainty). "
    "Do not use outside knowledge. If the excerpts are insufficient, answer 'unclear'. Reply "
    'with one JSON object: {"verdict": "supports"|"does_not_support"|"unclear", '
    '"rationale": "<one short paragraph>"}.'
)


class LLMClaimReviewer:
    def __init__(self, provider: LLMProvider, config: LLMConfig) -> None:
        self.provider = provider
        self.config = config

    def request(self, sentence: str, excerpts: list[str]) -> LLMRequest:
        prompt = json.dumps({"sentence": sentence, "excerpts": excerpts}, ensure_ascii=False)
        return LLMRequest(
            system=SYSTEM,
            prompt=prompt,
            response_schema={"type": "object", "required": ["verdict", "rationale"]},
            temperature=None,
            max_output_tokens=self.config.max_output_tokens,
        )

    def review(
        self, project_id: str, report_id: str, finding: AuditFinding, excerpts: list[str]
    ) -> ClaimReview:
        req = self.request(finding.excerpt, excerpts)
        resp = self.provider.complete(req)
        data = parse_json_object(resp.text)
        verdict = data.get("verdict")
        rationale = data.get("rationale")
        if verdict not in ("supports", "does_not_support", "unclear"):
            raise LLMResponseError(f"invalid verdict {verdict!r}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise LLMResponseError("missing rationale")
        return ClaimReview(
            project_id=project_id,
            audit_report_id=report_id,
            finding_id=finding.finding_id,
            verdict=verdict,
            rationale=rationale.strip()[:4000],
            provider=self.config.provider,
            model_id=self.config.model_id,
            effort=self.config.effort,
            prompt_sha256=req.sha256,
            response_sha256=resp.sha256,
        )
