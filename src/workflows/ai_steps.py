"""Start the existing AI-assisted workflow steps with the project's per-task models.

This module only wires configuration, credentials, the audit trail and the budget guard
into the existing runners — :class:`DualScreeningRunner`, :class:`LLMExtractor` +
:func:`submit_proposals`, :class:`LLMRoBDrafter` and :func:`run_stage5`. It adds no
workflow logic of its own, never records a human decision, and never moves the state
machine: AI output stays a proposal until an authenticated human acts on it through the
API. Statistics are computed by the deterministic engine only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.enums import ActorType, Phase, ScreeningStage, StudyDesign
from src.core.exceptions import SysRevError
from src.core.schemas.protocol import PICOProtocol
from src.extraction.ai_verification import LLMVerifier, verifiable
from src.extraction.master_table import submit_proposals
from src.extraction.proposals import LLMExtractor
from src.extraction.rob import LLMRoBDrafter
from src.fulltext.storage import DocumentStore
from src.llm.catalogue import ModelCapability, ModelCatalogue, rough_prompt_tokens
from src.llm.credentials import CredentialResolver, redact
from src.llm.provider import LLMConfig, LLMProvider, LLMRunHaltError
from src.llm.tasks import AITask, ProjectAISettings, build_api_provider
from src.llm.usage import AuditedProvider, BudgetGuard, RedundantCallError
from src.pdf_extractor import parse_pdf
from src.pdf_extractor.models import ParsedDocument
from src.screening.llm_screening import (
    DualScreeningRunner,
    LLMScreener,
    ScreeningRunSummary,
    full_text_source,
    title_abstract_source,
)
from src.state.repository import Repository
from src.stats_engine.stage5 import Stage5Artifacts, run_stage5

ProviderBuilder = Callable[[LLMConfig], tuple[LLMProvider, ModelCapability | None]]
SCREENING_ROLES = {
    AITask.SCREENING_REVIEWER_1: "reviewer_1",
    AITask.SCREENING_REVIEWER_2: "reviewer_2",
}


class WorkflowStepError(SysRevError):
    """The step cannot run in the project's current state."""


@dataclass
class StepReport:
    """Outcome of one batch step, per item: done / skipped / failed, and a halt reason."""

    step: str
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    halted: str | None = None
    warnings: list[str] = field(default_factory=list)


class AIWorkflow:
    def __init__(
        self,
        repo: Repository,
        project_id: str,
        *,
        catalogue: ModelCatalogue,
        credentials: CredentialResolver,
        store: DocumentStore,
        provider_builder: ProviderBuilder | None = None,
        commit: Callable[[], None] | None = None,
    ) -> None:
        self.repo = repo
        self.project_id = project_id
        self.catalogue = catalogue
        self.credentials = credentials
        self.store = store
        self._build: ProviderBuilder = provider_builder or (
            lambda cfg: build_api_provider(cfg, catalogue, credentials)
        )
        self._commit = commit or repo.session.commit

    # ------------------------------------------------------------------ configuration
    def settings(self) -> ProjectAISettings:
        settings = self.repo.latest_ai_settings(self.project_id)
        if settings is None:
            raise WorkflowStepError("no AI model settings saved for this project")
        return settings

    def preflight(self, task: AITask) -> list[str]:
        """Problems that would stop ``task`` — checked locally, without any API call."""
        try:
            config = self.settings().config_for(task)
            self.catalogue.validate(config)
            self.credentials.get(config.provider)
        except SysRevError as exc:
            return [str(exc)]
        return []

    def budget(self) -> BudgetGuard:
        s = self.settings()
        totals = self.repo.llm_usage_totals(self.project_id)
        return BudgetGuard(
            spent_usd=float(totals["cost_usd_estimate"]), tokens=int(totals["tokens"]),
            unpriced_calls=int(totals["unpriced_successful_calls"]), warn_usd=s.warn_usd,
            stop_usd=s.stop_usd, warn_tokens=s.warn_tokens, stop_tokens=s.stop_tokens,
        )  # fmt: skip

    def _provider(
        self, task: AITask, budget: BudgetGuard, *, dedupe: bool
    ) -> tuple[AuditedProvider, LLMConfig]:
        config = self.settings().config_for(task)
        inner, capability = self._build(config)
        if capability is not None:
            budget.require_priced(capability)
        audited = AuditedProvider(
            inner, project_id=self.project_id, task=task.value, label=config.label,
            effort=config.effort, capability=capability, sink=self.repo.record_llm_call,
            budget=budget,
            already_answered=(
                (lambda key: self.repo.has_successful_call(self.project_id, key))
                if dedupe else None
            ),
        )  # fmt: skip
        return audited, config

    def estimate(self, task: AITask, prompts_chars: Sequence[int]) -> dict[str, Any]:
        """Rough pre-run estimate: input tokens ≈ chars/4; output bounded by the limit."""
        config = self.settings().config_for(task)
        cap = self.catalogue.validate(config)
        n = len(prompts_chars)
        input_tokens = sum(rough_prompt_tokens("x" * c) for c in prompts_chars)
        out: dict[str, Any] = {
            "task": task.value, "model": f"{config.provider}/{config.model_id}",
            "effort": config.effort or "provider default", "calls": n,
            "input_tokens_estimate": input_tokens,
            "max_output_tokens_total": n * config.max_output_tokens,
            "cost_usd_upper_bound": None,
            "note": "rough estimate (≈4 characters/token); output assumed at the "
            "configured maximum, so the bound is pessimistic; invoices are authoritative",
        }  # fmt: skip
        if cap.pricing is not None:
            p = cap.pricing
            out["cost_usd_upper_bound"] = round(
                (input_tokens * max(p.input, p.long_prompt_input or 0)
                 + n * config.max_output_tokens * max(p.output, p.long_prompt_output or 0))
                / 1_000_000, 4,
            )  # fmt: skip
        return out

    # ------------------------------------------------------------------ helpers
    def _require_phase(self, *phases: Phase) -> None:
        phase = Phase(self.repo.get_project(self.project_id).phase)
        if phase not in phases:
            allowed = ", ".join(p.value for p in phases)
            raise WorkflowStepError(f"this step runs in {allowed}; the project is in {phase}")

    def _protocol(self) -> PICOProtocol:
        protocol = self.repo.latest_protocol(self.project_id)
        if protocol is None or protocol.status != "approved":
            raise WorkflowStepError("an approved protocol is required")
        return protocol

    def document(self, record_id: str) -> ParsedDocument:
        item = self.repo.get_fulltext_item(self.project_id, record_id)
        if item is None or item.pdf_filename is None or item.pdf_sha256 is None:
            raise WorkflowStepError(f"record {record_id} has no linked PDF")
        if item.match_unconfirmed:
            raise WorkflowStepError(f"the PDF of {record_id} has an unconfirmed identity")
        data = self.store.read_verified(item.pdf_filename, item.pdf_sha256)
        return parse_pdf(data, filename=item.pdf_filename, expected_sha256=item.pdf_sha256)

    def _screeners(self, budget: BudgetGuard) -> list[LLMScreener]:
        from src.core.enums import ReviewerRole

        out = []
        for task, role in SCREENING_ROLES.items():
            provider, config = self._provider(task, budget, dedupe=False)
            out.append(LLMScreener(provider, config, ReviewerRole(role)))
        return out

    # ------------------------------------------------------------------ steps
    def screening_independence_note(self) -> str | None:
        s = self.settings()
        a = s.tasks.get(AITask.SCREENING_REVIEWER_1)
        b = s.tasks.get(AITask.SCREENING_REVIEWER_2)
        if a and b and (a.provider, a.model_id, a.effort) == (b.provider, b.model_id, b.effort):
            return (
                "both automated passes use the same model and effort: they are repeated runs "
                "of one model, not independent assessors (and never human reviewers)"
            )
        return None

    def run_title_abstract_screening(self, *, limit: int | None = None) -> ScreeningRunSummary:
        self._require_phase(Phase.STAGE2_SEARCH_SCREENING)
        protocol = self._protocol()
        budget = self.budget()
        decided = {
            (d.record_id, d.reviewer)
            for d in self.repo.load_screening_decisions(self.project_id)
            if d.stage == ScreeningStage.TITLE_ABSTRACT
        }
        records = [
            r for r in self.repo.load_records(self.project_id, unique_only=True)
            if not {(r.record_id, "reviewer_1"), (r.record_id, "reviewer_2")} <= decided
        ]  # fmt: skip
        sources = [title_abstract_source(r) for r in records[:limit]]
        runner = DualScreeningRunner(
            self.repo, self.project_id, protocol, self._screeners(budget), commit=self._commit
        )
        summary = runner.run(sources)
        self._commit()
        return summary

    def run_full_text_screening(self, *, limit: int | None = None) -> ScreeningRunSummary:
        self._require_phase(Phase.STAGE3_FULLTEXT_RETRIEVAL, Phase.STAGE4_EXTRACTION_ROB)
        protocol = self._protocol()
        budget = self.budget()
        sources = []
        for record_id in sorted(
            self.repo.included_record_ids(self.project_id, ScreeningStage.TITLE_ABSTRACT)
        ):
            try:
                sources.append(full_text_source(record_id, self.document(record_id)))
            except (WorkflowStepError, SysRevError):
                continue  # no usable PDF yet: listed in the manifest, not screened
            if limit is not None and len(sources) >= limit:
                break
        runner = DualScreeningRunner(
            self.repo, self.project_id, protocol, self._screeners(budget), commit=self._commit
        )
        summary = runner.run(sources)
        self._commit()
        return summary

    def run_extraction(self, record_ids: Sequence[str], *, force: bool = False) -> StepReport:
        self._require_phase(Phase.STAGE4_EXTRACTION_ROB)
        protocol = self._protocol()
        budget = self.budget()
        provider, config = self._provider(AITask.EXTRACTION, budget, dedupe=not force)
        extractor = LLMExtractor(provider, config)
        report = StepReport("extraction")
        existing = {
            i.record_id
            for i in self.repo.load_extraction_items(self.project_id, include_superseded=True)
            if i.proposer_identity == extractor.identity
        }
        for record_id in record_ids:
            if record_id in existing and not force:
                report.skipped.append(f"{record_id}: already has proposals from this model")
                continue
            try:
                doc = self.document(record_id)
                proposals = extractor.propose(protocol, doc)
                items = submit_proposals(
                    self.repo, self.project_id, record_id, doc, proposals,
                    proposed_by=ActorType.AGENT, proposer_identity=extractor.identity,
                )  # fmt: skip
                self._commit()
                report.done.append(f"{record_id}: {len(items)} proposal(s), pending human review")
            except LLMRunHaltError as exc:
                report.halted = str(exc)
                break
            except RedundantCallError as exc:
                report.skipped.append(f"{record_id}: {exc}")
            except Exception as exc:  # per-record failure: nothing stored for that record
                report.failed.append(f"{record_id}: {redact(str(exc))[:300]}")
            finally:
                self._commit()  # keep the audit trail of every call
        report.warnings = budget.warnings()
        return report

    def run_verification(self, *, force: bool = False) -> StepReport:
        self._require_phase(Phase.STAGE4_EXTRACTION_ROB)
        settings = self.settings()
        budget = self.budget()
        provider, config = self._provider(AITask.VERIFICATION, budget, dedupe=not force)
        verifier = LLMVerifier(provider, config)
        report = StepReport("verification")
        ext = settings.tasks.get(AITask.EXTRACTION)
        if ext and (ext.provider, ext.model_id) == (config.provider, config.model_id):
            report.warnings.append(
                "the verifier uses the same model as the extractor: not an independent check"
            )
        done = {
            v.item_id
            for v in self.repo.load_ai_verifications(self.project_id)
            if v.verifier_identity == verifier.identity
        }
        items = [i for i in self.repo.load_extraction_items(self.project_id) if verifiable(i)]
        docs: dict[str, ParsedDocument] = {}
        for item in items:
            if item.item_id in done and not force:
                report.skipped.append(f"{item.item_id}: already verified by this model")
                continue
            try:
                if item.record_id not in docs:
                    docs[item.record_id] = self.document(item.record_id)
                before = len(provider.records)
                verification = verifier.verify(item, docs[item.record_id])
                call_id = provider.records[-1].call_id if len(provider.records) > before else None
                verification = verification.model_copy(update={"call_id": call_id})
                self.repo.save_ai_verification(self.project_id, verification)
                report.done.append(f"{item.item_id}: {verification.verdict} (advisory)")
            except LLMRunHaltError as exc:
                report.halted = str(exc)
                break
            except RedundantCallError as exc:
                report.skipped.append(f"{item.item_id}: {exc}")
            except Exception as exc:
                report.failed.append(f"{item.item_id}: {redact(str(exc))[:300]}")
            finally:
                self._commit()
        report.warnings += budget.warnings()
        return report

    def run_rob_draft(
        self, record_id: str, design: StudyDesign, *, outcome_id: str | None = None,
        force: bool = False,
    ) -> StepReport:  # fmt: skip
        self._require_phase(Phase.STAGE4_EXTRACTION_ROB)
        protocol = self._protocol()
        budget = self.budget()
        provider, config = self._provider(AITask.RISK_OF_BIAS, budget, dedupe=not force)
        drafter = LLMRoBDrafter(provider, config)
        report = StepReport("risk_of_bias")
        try:
            assessment = drafter.draft(
                record_id, self.document(record_id), design=design,
                review_design=protocol.review_design, outcome_id=outcome_id,
            )  # fmt: skip
            self.repo.save_rob(self.project_id, assessment)
            report.done.append(f"{record_id}: draft {assessment.assessment_id} awaits approval")
        except LLMRunHaltError as exc:
            report.halted = str(exc)
        except RedundantCallError as exc:
            report.skipped.append(f"{record_id}: {exc}")
        except Exception as exc:
            report.failed.append(f"{record_id}: {redact(str(exc))[:300]}")
        finally:
            self._commit()
        report.warnings = budget.warnings()
        return report

    def run_statistics(self, output_dir: Path) -> Stage5Artifacts:
        """Deterministic Stage 5 engine (no LLM involved)."""
        artifacts = run_stage5(self.repo, self.project_id, output_dir=output_dir)
        self._commit()
        return artifacts
