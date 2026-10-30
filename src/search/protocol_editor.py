"""Stage 1 protocol editing: validated drafts, verification, authenticated approval.

Rules:

* Protocols can be edited only in Stage 1. After Gate 1 the approved protocol is locked;
  amendments after Gate 1 are not supported yet (see ARCHITECTURE.md §9.7).
* A draft is validated by :class:`PICOProtocol` on every save. Approval fields, version
  numbers and verification flags in the submitted data are ignored: MeSH terms become
  verified only through :func:`verify_mesh` (NLM MeSH lookup via ``PubMedClient``), and a
  strategy becomes syntax-validated only through an API preview of that exact strategy or
  an authenticated human attestation (subscription databases).
* Verification survives a save only for an unchanged MeSH descriptor or an unchanged
  strategy (same content hash). Any change to an approved protocol starts a new draft
  version, which voids the approval until an enrolled reviewer approves again.
* Every save, verification and approval is written to the append-only protocol event log
  with the full protocol payload, so earlier drafts remain reconstructable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from src.core.enums import ActorType, Phase
from src.core.exceptions import SysRevError
from src.core.schemas.base import new_id, utc_now
from src.core.schemas.protocol import STRATEGY_EXECUTABLE_FIELDS, PICOProtocol
from src.core.schemas.search import ProtocolEvent
from src.state.repository import Repository

PROTECTED_KEYS = frozenset(
    {"protocol_id", "version", "status", "approved_by", "approved_at", "approved_reviewer_id"}
)
MESH_VERIFICATION_KEYS = frozenset({"verified", "verified_at"})
STRATEGY_VALIDATION_KEYS = frozenset(
    {"syntax_validated", "validation_method", "validated_by", "validated_at", "validation_note"}
)
# Which downstream results a change can affect.
MATERIAL_GROUPS: dict[str, tuple[str, ...]] = {
    "eligibility": ("research_question", "population", "intervention", "comparator",
                    "review_design", "study_designs", "inclusion_criteria",
                    "exclusion_criteria"),
    "outcomes": ("outcomes",),
    "search": ("selected_databases", "search_strategies", "mesh_terms"),
    "administrative": ("review_title", "prospero_id"),
}  # fmt: skip


class ProtocolEditError(SysRevError):
    """The protocol cannot be changed in the project's current state."""


class Reviewer(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def display_name(self) -> str: ...


class MeSHChecker(Protocol):
    async def mesh_descriptor_exists(self, descriptor: str) -> bool: ...


@dataclass(slots=True)
class SaveResult:
    protocol: PICOProtocol
    new_version: bool
    material_changes: dict[str, list[str]] = field(default_factory=dict)
    invalidated: list[str] = field(default_factory=list)


def protocol_sha256(protocol: PICOProtocol) -> str:
    data = json.dumps(protocol.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _comparable(protocol: PICOProtocol) -> dict[str, Any]:
    """Content of a protocol without approval, version and verification state."""
    data = protocol.model_dump(mode="json", exclude=set(PROTECTED_KEYS))
    data["mesh_terms"] = [{k: v for k, v in m.items() if k not in MESH_VERIFICATION_KEYS}
                          for m in data["mesh_terms"]]  # fmt: skip
    data["search_strategies"] = [{k: s[k] for k in STRATEGY_EXECUTABLE_FIELDS}
                                 for s in data["search_strategies"]]  # fmt: skip
    return data


def material_changes(old: PICOProtocol | None, new: PICOProtocol) -> dict[str, list[str]]:
    """Changed fields grouped by the downstream results they can affect."""
    if old is None:
        return {}
    a, b = _comparable(old), _comparable(new)
    out: dict[str, list[str]] = {}
    for group, fields in MATERIAL_GROUPS.items():
        changed = [f for f in fields if a.get(f) != b.get(f)]
        if changed:
            out[group] = changed
    return out


def _require_stage1(repo: Repository, project_id: str) -> None:
    phase = Phase(repo.get_project(project_id).phase)
    if phase != Phase.STAGE1_PROTOCOL:
        raise ProtocolEditError(
            f"the protocol can only be changed in {Phase.STAGE1_PROTOCOL} (project is in "
            f"{phase}); after Gate 1 the approved protocol is locked"
        )


def _require_draft(repo: Repository, project_id: str) -> PICOProtocol:
    _require_stage1(repo, project_id)
    protocol = repo.latest_protocol(project_id)
    if protocol is None:
        raise ProtocolEditError("no protocol draft saved yet")
    if protocol.status != "draft":
        raise ProtocolEditError(
            "the current version is approved; save an amended draft first (this voids the "
            "approval until it is approved again)"
        )
    return protocol


def _event(
    protocol: PICOProtocol,
    action: Literal["draft_saved", "mesh_verified", "syntax_validated", "syntax_attested",
                    "approved"],
    reviewer: Reviewer,
    **kw: Any,
) -> ProtocolEvent:  # fmt: skip
    return ProtocolEvent(
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.version,
        action=action,
        protocol_sha256=protocol_sha256(protocol),
        actor=reviewer.display_name,
        actor_type=ActorType.HUMAN,
        reviewer_id=reviewer.id,
        **kw,
    )


def _strip(items: Sequence[Any], keys: frozenset[str]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        data = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        out.append({k: v for k, v in data.items() if k not in keys})
    return out


def save_draft(
    repo: Repository, project_id: str, data: Mapping[str, Any], *, reviewer: Reviewer
) -> SaveResult:
    """Validate and store a draft; returns what changed and which results it invalidated.

    Raises ``pydantic.ValidationError`` (a ``ValueError``) for an invalid protocol.
    """
    _require_stage1(repo, project_id)
    repo.require_active_reviewer(reviewer.id)
    previous = repo.latest_protocol(project_id)
    clean = {k: v for k, v in dict(data).items() if k not in PROTECTED_KEYS}
    clean["mesh_terms"] = _strip(clean.get("mesh_terms") or [], MESH_VERIFICATION_KEYS)
    clean["search_strategies"] = _strip(
        clean.get("search_strategies") or [], STRATEGY_VALIDATION_KEYS
    )
    if previous is None:
        version, protocol_id = 1, new_id()
    elif previous.status == "draft":
        version, protocol_id = previous.version, previous.protocol_id
    else:
        version, protocol_id = previous.version + 1, new_id()
    draft = PICOProtocol.model_validate(
        clean | {"protocol_id": protocol_id, "version": version, "status": "draft"}
    )
    changes = material_changes(previous, draft)
    if previous is not None and previous.status != "draft" and not changes:
        raise ProtocolEditError("nothing changed: the approved protocol stays current")

    invalidated: list[str] = []
    if previous is not None:
        verified = {m.descriptor.casefold(): m for m in previous.mesh_terms if m.verified}
        mesh = []
        for m in draft.mesh_terms:
            old = verified.pop(m.descriptor.casefold(), None)
            mesh.append(
                m.model_copy(update={"verified": True, "verified_at": old.verified_at})
                if old is not None and old.descriptor_ui == m.descriptor_ui
                else m
            )
        invalidated += [f"MeSH verification of '{m.descriptor}' (term changed or removed)"
                        for m in verified.values()]  # fmt: skip
        validated = {s.sha256: s for s in previous.search_strategies if s.syntax_validated}
        strategies = []
        for s in draft.search_strategies:
            old_s = validated.pop(s.sha256, None)
            keys = STRATEGY_VALIDATION_KEYS if old_s is not None else frozenset()
            carried = {k: getattr(old_s, k) for k in keys}
            strategies.append(s.model_copy(update=carried))
        invalidated += [
            f"syntax validation of the {s.database} strategy {s.sha256[:12]} (strategy "
            "changed or removed; its earlier previews no longer describe the protocol)"
            for s in validated.values()
        ]
        draft = PICOProtocol.model_validate(
            draft.model_dump() | {"mesh_terms": mesh, "search_strategies": strategies}
        )
        if previous.status != "draft":
            invalidated.append(
                f"approval of version {previous.version}: version {version} is a new draft "
                "and must be approved again before Gate 1"
            )
    repo.save_protocol(project_id, draft)
    repo.add_protocol_event(
        project_id,
        _event(draft, "draft_saved", reviewer, material_changes=changes, invalidated=invalidated),
        draft,
    )
    return SaveResult(draft, previous is None or previous.status != "draft", changes, invalidated)


async def verify_mesh(
    repo: Repository, project_id: str, checker: MeSHChecker, *, reviewer: Reviewer
) -> dict[str, bool]:
    """Look up every unverified MeSH descriptor in NLM MeSH; mark only those found."""
    draft = _require_draft(repo, project_id)
    repo.require_active_reviewer(reviewer.id)
    results: dict[str, bool] = {}
    terms = []
    for m in draft.mesh_terms:
        if m.verified:
            terms.append(m)
            continue
        found = await checker.mesh_descriptor_exists(m.descriptor)
        results[m.descriptor] = found
        terms.append(m.model_copy(update={"verified": True, "verified_at": utc_now()}) if found
                     else m)  # fmt: skip
    if results:
        updated = PICOProtocol.model_validate(draft.model_dump() | {"mesh_terms": terms})
        repo.save_protocol(project_id, updated)
        missing = sorted(d for d, ok in results.items() if not ok)
        repo.add_protocol_event(
            project_id,
            _event(updated, "mesh_verified", reviewer,
                   note=("not found in MeSH: " + "; ".join(missing)) if missing else None),
            updated,
        )  # fmt: skip
    return results


def record_strategy_validation(
    repo: Repository,
    project_id: str,
    strategy_sha256: str,
    *,
    method: Literal["api_preview", "api_accepted", "human_attestation"],
    reviewer: Reviewer,
    note: str | None = None,
) -> PICOProtocol:
    draft = _require_draft(repo, project_id)
    repo.require_active_reviewer(reviewer.id)
    if draft.strategy(strategy_sha256) is None:
        raise ProtocolEditError(f"no strategy {strategy_sha256[:12]} in the current draft")
    strategies = [
        s.model_copy(update={"syntax_validated": True, "validation_method": method,
                             "validated_by": reviewer.display_name, "validated_at": utc_now(),
                             "validation_note": note})
        if s.sha256 == strategy_sha256 else s
        for s in draft.search_strategies
    ]  # fmt: skip
    updated = PICOProtocol.model_validate(draft.model_dump() | {"search_strategies": strategies})
    repo.save_protocol(project_id, updated)
    action: Literal["syntax_validated", "syntax_attested"] = (
        "syntax_attested" if method == "human_attestation" else "syntax_validated"
    )
    repo.add_protocol_event(
        project_id, _event(updated, action, reviewer, note=note or strategy_sha256), updated
    )
    return updated


def approve(
    repo: Repository, project_id: str, *, reviewer: Reviewer, note: str | None = None
) -> PICOProtocol:
    """Authenticated approval of the current draft (required before Gate 1 can pass)."""
    draft = _require_draft(repo, project_id)
    active = repo.require_active_reviewer(reviewer.id)
    blockers = draft.approval_blockers()
    if blockers:
        raise ProtocolEditError("protocol cannot be approved: " + "; ".join(blockers))
    approved = PICOProtocol.model_validate(
        draft.model_dump()
        | {"status": "approved", "approved_by": active.display_name, "approved_at": utc_now(),
           "approved_reviewer_id": active.id}
    )  # fmt: skip
    repo.save_protocol(project_id, approved)
    repo.supersede_older_protocols(project_id, approved.version)
    repo.add_protocol_event(project_id, _event(approved, "approved", reviewer, note=note), approved)
    return approved
