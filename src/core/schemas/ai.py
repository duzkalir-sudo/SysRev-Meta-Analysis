"""Advisory AI verification of extracted values (never a human review)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.core.schemas.base import AwareDatetime, StrictModel, new_id, utc_now


class AIVerification(StrictModel):
    """A second model's opinion on whether a pending item's quote supports its value.

    Purely advisory: it never changes an item's review status, never counts as an
    acceptance, and is shown to the human reviewer labelled as automated output.
    """

    verification_id: str = Field(default_factory=new_id)
    item_id: str
    record_id: str
    verdict: Literal["supports", "does_not_support", "unclear"]
    rationale: str = Field(min_length=1, max_length=4000)
    verifier_identity: str
    provider: str
    model_id: str
    effort: str | None = None
    call_id: str | None = None
    prompt_sha256: str
    response_sha256: str
    is_human: Literal[False] = False
    created_at: AwareDatetime = Field(default_factory=utc_now)
