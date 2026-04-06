"""Fix execution engine for applying approved fixes and handling rejections.

The FixExecutor is the component that takes approved FixProposals from the
approval gate and applies them. It also handles rejection feedback by
requesting alternative suggestions from the troubleshooter and retrying
the approval loop.

Architecture:
- FixExecutor receives approved proposals from the ApprovalCoordinator
- It applies changes via a pluggable ChangeApplier protocol (file edits,
  config changes, etc.)
- On rejection, it collects user feedback and asks the troubleshooter
  (via the orchestrator) for alternative proposals
- A retry budget limits the number of rejection-retry cycles
- The configurable autonomy policy controls whether fixes can be
  auto-applied or always require approval

Safety:
- All changes are validated before application
- A dry-run mode previews changes without modifying files
- Rollback tracking records what was changed for undo capability
- The executor never bypasses the approval gate (unless policy allows)

Integration:
- Called by the orchestrator hub after the approval gate returns
- Never communicates directly with sub-agents (orchestrator routes)
- Reports results back to the orchestrator for the reporter agent
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from test_runner.agents.troubleshooter.models import (
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.autonomy.approval import (
    ApprovalCoordinator,
    ApprovalResponse,
    ApprovalScope,
    ApprovalStatus,
    ProposalDecision,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FixOutcome(str, enum.Enum):
    """Outcome of applying a single fix."""

    APPLIED = "applied"          # Successfully applied
    FAILED = "failed"            # Application failed (e.g., file not found)
    SKIPPED = "skipped"          # User skipped this fix
    REJECTED = "rejected"        # User rejected this fix
    DRY_RUN = "dry_run"          # Dry-run only, not actually applied
    ROLLED_BACK = "rolled_back"  # Applied then rolled back


class RetryOutcome(str, enum.Enum):
    """Outcome of a rejection-retry cycle."""

    APPROVED = "approved"         # Alternative was approved
    REJECTED_AGAIN = "rejected_again"  # Alternative also rejected
    NO_ALTERNATIVES = "no_alternatives"  # No alternatives available
    BUDGET_EXHAUSTED = "budget_exhausted"  # Max retries reached
    ERROR = "error"               # Error during retry


# ---------------------------------------------------------------------------
# Change applier protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ChangeApplier(Protocol):
    """Protocol for applying proposed changes to files.

    Implementations handle the mechanics of modifying files,
    configurations, or other artifacts. The FixExecutor delegates
    to a ChangeApplier for the actual filesystem operations.
    """

    async def apply_change(
        self, change: ProposedChange, *, dry_run: bool = False
    ) -> ChangeResult:
        """Apply a single proposed change.

        Args:
            change: The change to apply.
            dry_run: If True, validate but don't actually modify.

        Returns:
            Result of the application attempt.
        """
        ...

    async def rollback_change(
        self, change: ProposedChange, backup: str
    ) -> bool:
        """Rollback a previously applied change.

        Args:
            change: The change that was applied.
            backup: The original content to restore.

        Returns:
            True if rollback succeeded.
        """
        ...


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ChangeResult(BaseModel, frozen=True):
    """Result of applying a single ProposedChange.

    Attributes:
        change: The change that was applied.
        success: Whether the change was applied successfully.
        message: Human-readable result message.
        backup_content: Original file content before the change
            (for rollback support).
        dry_run: Whether this was a dry-run (no actual modification).
    """

    change: ProposedChange
    success: bool
    message: str = ""
    backup_content: str = ""
    dry_run: bool = False


class FixApplicationResult(BaseModel, frozen=True):
    """Result of applying all changes in a single FixProposal.

    Attributes:
        proposal: The fix proposal that was applied.
        outcome: Overall outcome of the application.
        change_results: Results for each individual change.
        error_message: Error message if application failed.
        duration_seconds: Time taken to apply all changes.
    """

    proposal: FixProposal
    outcome: FixOutcome
    change_results: list[ChangeResult] = Field(default_factory=list)
    error_message: str = ""
    duration_seconds: float = 0.0

    @property
    def all_changes_succeeded(self) -> bool:
        """True if all individual changes succeeded."""
        return all(cr.success for cr in self.change_results)

    @property
    def applied_count(self) -> int:
        """Number of changes that were successfully applied."""
        return sum(1 for cr in self.change_results if cr.success)

    @property
    def failed_count(self) -> int:
        """Number of changes that failed."""
        return sum(1 for cr in self.change_results if not cr.success)


class RejectionFeedback(BaseModel, frozen=True):
    """Structured feedback from a rejection, used to guide alternatives.

    Attributes:
        proposal: The rejected proposal.
        user_feedback: Free-text feedback from the user.
        rejection_reason: Categorized reason for rejection.
        attempt_number: Which retry attempt this rejection is from.
    """

    proposal: FixProposal
    user_feedback: str = ""
    rejection_reason: str = ""
    attempt_number: int = 1


class RetryResult(BaseModel, frozen=True):
    """Result of a rejection-retry cycle.

    Attributes:
        original_proposal: The proposal that was rejected.
        outcome: Outcome of the retry attempt.
        alternative_proposal: The alternative that was proposed (if any).
        application_result: Result of applying the alternative (if approved).
        feedback_history: All rejection feedbacks in this retry chain.
        attempts_made: Total attempts (including original).
    """

    original_proposal: FixProposal
    outcome: RetryOutcome
    alternative_proposal: FixProposal | None = None
    application_result: FixApplicationResult | None = None
    feedback_history: list[RejectionFeedback] = Field(default_factory=list)
    attempts_made: int = 1


class FixExecutionReport(BaseModel, frozen=True):
    """Complete report of fix execution across all proposals.

    Attributes:
        applied: Results for proposals that were applied.
        rejected: Results for proposals that were rejected.
        retried: Results for proposals that went through retry cycles.
        skipped: Proposals that were skipped.
        total_proposals: Total proposals considered.
        total_applied: Number successfully applied.
        total_rejected: Number ultimately rejected.
        total_retried: Number that went through retry.
        duration_seconds: Total time for the execution phase.
    """

    applied: list[FixApplicationResult] = Field(default_factory=list)
    rejected: list[RejectionFeedback] = Field(default_factory=list)
    retried: list[RetryResult] = Field(default_factory=list)
    skipped: list[FixProposal] = Field(default_factory=list)
    total_proposals: int = 0
    total_applied: int = 0
    total_rejected: int = 0
    total_retried: int = 0
    duration_seconds: float = 0.0

    @property
    def has_failures(self) -> bool:
        """True if any applied fix had a failure."""
        return any(
            r.outcome == FixOutcome.FAILED for r in self.applied
        )

    def summary_line(self) -> str:
        """One-line summary for CLI output."""
        return (
            f"Fix execution: {self.total_applied} applied, "
            f"{self.total_rejected} rejected, "
            f"{self.total_retried} retried, "
            f"{len(self.skipped)} skipped"
        )


# ---------------------------------------------------------------------------
# Alternative suggestion generator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AlternativeGenerator(Protocol):
    """Protocol for generating alternative fix suggestions.

    The orchestrator provides an implementation that routes through
    the troubleshooter agent. The FixExecutor calls this when a
    proposal is rejected and the user's feedback suggests alternatives
    might help.
    """

    async def generate_alternative(
        self,
        rejected_proposal: FixProposal,
        feedback: RejectionFeedback,
        previous_alternatives: Sequence[FixProposal],
    ) -> FixProposal | None:
        """Generate an alternative fix based on rejection feedback.

        Args:
            rejected_proposal: The proposal that was rejected.
            feedback: User's feedback on why it was rejected.
            previous_alternatives: Previously rejected alternatives
                (to avoid repeating suggestions).

        Returns:
            A new FixProposal, or None if no alternative can be generated.
        """
        ...


# ---------------------------------------------------------------------------
# File-based change applier (default implementation)
# ---------------------------------------------------------------------------


class FileChangeApplier:
    """Applies ProposedChanges by modifying files on disk.

    This is the default ChangeApplier for local execution targets.
    It reads files, applies string replacements, and tracks backups
    for rollback support.
    """

    def __init__(self, working_directory: str = "") -> None:
        self._working_directory = working_directory

    def _resolve_path(self, file_path: str) -> str:
        """Resolve a file path relative to the working directory."""
        import os

        if os.path.isabs(file_path):
            return file_path
        if self._working_directory:
            return os.path.join(self._working_directory, file_path)
        return file_path

    async def apply_change(
        self, change: ProposedChange, *, dry_run: bool = False
    ) -> ChangeResult:
        """Apply a file modification change.

        For 'modify' changes: replaces original_snippet with proposed_snippet.
        For 'add' changes: appends proposed_snippet to the file.
        For 'delete' changes: removes original_snippet from the file.
        For 'config' changes: treated as 'modify'.
        """
        import os

        resolved = self._resolve_path(change.file_path)

        # Validate file exists for modify/delete
        if change.change_type in ("modify", "delete", "config"):
            if not os.path.isfile(resolved):
                return ChangeResult(
                    change=change,
                    success=False,
                    message=f"File not found: {resolved}",
                    dry_run=dry_run,
                )

        try:
            if change.change_type == "add" and not os.path.isfile(resolved):
                original_content = ""
            else:
                with open(resolved, "r", encoding="utf-8") as f:
                    original_content = f.read()
        except (OSError, UnicodeDecodeError) as exc:
            return ChangeResult(
                change=change,
                success=False,
                message=f"Failed to read file: {exc}",
                dry_run=dry_run,
            )

        # Compute new content
        if change.change_type in ("modify", "config"):
            if not change.original_snippet:
                return ChangeResult(
                    change=change,
                    success=False,
                    message="No original_snippet provided for modify change",
                    dry_run=dry_run,
                )
            if change.original_snippet not in original_content:
                return ChangeResult(
                    change=change,
                    success=False,
                    message=(
                        f"Original snippet not found in {change.file_path}. "
                        "File may have been modified since analysis."
                    ),
                    dry_run=dry_run,
                )
            new_content = original_content.replace(
                change.original_snippet, change.proposed_snippet, 1
            )
        elif change.change_type == "add":
            new_content = original_content + change.proposed_snippet
        elif change.change_type == "delete":
            if not change.original_snippet:
                return ChangeResult(
                    change=change,
                    success=False,
                    message="No original_snippet provided for delete change",
                    dry_run=dry_run,
                )
            new_content = original_content.replace(
                change.original_snippet, "", 1
            )
        else:
            return ChangeResult(
                change=change,
                success=False,
                message=f"Unknown change_type: {change.change_type}",
                dry_run=dry_run,
            )

        if dry_run:
            return ChangeResult(
                change=change,
                success=True,
                message=f"Dry-run: would modify {change.file_path}",
                backup_content=original_content,
                dry_run=True,
            )

        # Write the change
        try:
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(new_content)
        except OSError as exc:
            return ChangeResult(
                change=change,
                success=False,
                message=f"Failed to write file: {exc}",
                dry_run=dry_run,
            )

        return ChangeResult(
            change=change,
            success=True,
            message=f"Applied {change.change_type} to {change.file_path}",
            backup_content=original_content,
            dry_run=False,
        )

    async def rollback_change(
        self, change: ProposedChange, backup: str
    ) -> bool:
        """Restore a file to its backup content."""
        resolved = self._resolve_path(change.file_path)
        try:
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(backup)
            return True
        except OSError as exc:
            logger.error("Rollback failed for %s: %s", change.file_path, exc)
            return False


# ---------------------------------------------------------------------------
# Pattern-based alternative generator (default implementation)
# ---------------------------------------------------------------------------


class PatternAlternativeGenerator:
    """Generates alternative fix proposals based on rejection feedback.

    Uses the original proposal's alternative_fixes list and the
    rejection feedback to construct a new proposal. This is the
    default implementation that doesn't require LLM calls.

    For LLM-powered alternatives, the orchestrator can provide a
    custom AlternativeGenerator that routes through the FixGenerator.
    """

    def __init__(
        self,
        llm_generator: AlternativeGenerator | None = None,
    ) -> None:
        self._llm_generator = llm_generator

    async def generate_alternative(
        self,
        rejected_proposal: FixProposal,
        feedback: RejectionFeedback,
        previous_alternatives: Sequence[FixProposal],
    ) -> FixProposal | None:
        """Generate an alternative from the proposal's alternative_fixes.

        If an LLM generator is available and no pattern-based alternatives
        remain, delegates to the LLM generator.

        Returns None if no alternatives are available.
        """
        # Collect previously used titles to avoid repeats
        used_titles = {rejected_proposal.title} | {
            p.title for p in previous_alternatives
        }

        # Try pattern-based alternatives first
        remaining = [
            alt for alt in rejected_proposal.alternative_fixes
            if alt not in used_titles
        ]

        if remaining:
            alt_description = remaining[0]
            # Construct an alternative proposal from the description
            return FixProposal(
                failure_id=rejected_proposal.failure_id,
                title=alt_description[:200],
                description=(
                    f"Alternative fix (after rejection of: "
                    f"{rejected_proposal.title})\n\n"
                    f"User feedback: {feedback.user_feedback or 'none'}\n\n"
                    f"{alt_description}"
                ),
                category=rejected_proposal.category,
                confidence=FixConfidence.LOW,
                confidence_score=max(
                    rejected_proposal.confidence_score - 0.15, 0.1
                ),
                affected_files=rejected_proposal.affected_files,
                proposed_changes=[],  # Alternatives start without concrete changes
                rationale=(
                    f"Alternative suggested after rejection. "
                    f"Original: {rejected_proposal.title}. "
                    f"Feedback: {feedback.user_feedback or 'none'}"
                ),
                alternative_fixes=[
                    a for a in remaining[1:]  # Remove the one we're using
                ],
                requires_user_action=True,
                user_action_description=(
                    f"Review this alternative approach: {alt_description}"
                ),
                metadata={
                    **rejected_proposal.metadata,
                    "is_alternative": True,
                    "attempt_number": feedback.attempt_number + 1,
                    "original_proposal_title": rejected_proposal.title,
                    "rejection_feedback": feedback.user_feedback,
                },
            )

        # Try LLM generator if available
        if self._llm_generator is not None:
            return await self._llm_generator.generate_alternative(
                rejected_proposal, feedback, previous_alternatives,
            )

        return None


# ---------------------------------------------------------------------------
# FixExecutor — main execution engine
# ---------------------------------------------------------------------------


@dataclass
class FixExecutorConfig:
    """Configuration for the fix executor.

    Attributes:
        max_retries: Maximum number of rejection-retry cycles per proposal.
        dry_run: If True, validate changes without applying them.
        rollback_on_failure: If True, rollback all changes in a proposal
            if any single change fails.
        require_all_changes: If True, all changes in a proposal must
            succeed for the proposal to be considered applied.
    """

    max_retries: int = 3
    dry_run: bool = False
    rollback_on_failure: bool = True
    require_all_changes: bool = True


class FixExecutor:
    """Executes approved fixes and manages the rejection-retry loop.

    The FixExecutor is the bridge between approval and application:
    1. Receives approved proposals from the ApprovalCoordinator
    2. Applies changes via the ChangeApplier
    3. On rejection, uses the AlternativeGenerator to propose new fixes
    4. Re-submits alternatives through the ApprovalCoordinator
    5. Tracks all results for the reporter

    The executor respects the autonomy policy and never bypasses
    the approval gate.

    Usage::

        executor = FixExecutor(
            applier=FileChangeApplier(),
            coordinator=approval_coordinator,
            alternative_gen=PatternAlternativeGenerator(),
        )

        # Execute all approved fixes
        report = await executor.execute_approved(
            approved_proposals,
            rejected_proposals,
            response,
        )

        # Or run the full approval-execute-retry loop
        report = await executor.run_fix_cycle(proposal_set)
    """

    def __init__(
        self,
        applier: ChangeApplier,
        coordinator: ApprovalCoordinator | None = None,
        alternative_gen: AlternativeGenerator | None = None,
        config: FixExecutorConfig | None = None,
    ) -> None:
        self._applier = applier
        self._coordinator = coordinator
        self._alternative_gen = alternative_gen or PatternAlternativeGenerator()
        self._config = config or FixExecutorConfig()
        self._execution_history: list[FixApplicationResult] = []
        self._retry_history: list[RetryResult] = []

    @property
    def config(self) -> FixExecutorConfig:
        """Current executor configuration."""
        return self._config

    @property
    def execution_history(self) -> list[FixApplicationResult]:
        """History of all fix applications."""
        return list(self._execution_history)

    @property
    def retry_history(self) -> list[RetryResult]:
        """History of all retry cycles."""
        return list(self._retry_history)

    def reset(self) -> None:
        """Reset execution and retry history."""
        self._execution_history.clear()
        self._retry_history.clear()

    # ----- Core execution -----

    async def apply_proposal(
        self, proposal: FixProposal
    ) -> FixApplicationResult:
        """Apply all changes in a single FixProposal.

        Applies changes in order. If rollback_on_failure is True and
        any change fails, all previously applied changes are rolled back.

        Args:
            proposal: The approved fix proposal to apply.

        Returns:
            FixApplicationResult with outcomes for all changes.
        """
        if not proposal.proposed_changes:
            result = FixApplicationResult(
                proposal=proposal,
                outcome=FixOutcome.SKIPPED,
                error_message="No concrete changes to apply",
            )
            self._execution_history.append(result)
            return result

        start_time = time.monotonic()
        change_results: list[ChangeResult] = []
        applied_backups: list[tuple[ProposedChange, str]] = []

        for change in proposal.proposed_changes:
            cr = await self._applier.apply_change(
                change, dry_run=self._config.dry_run,
            )
            change_results.append(cr)

            if cr.success and not self._config.dry_run:
                applied_backups.append((change, cr.backup_content))
            elif not cr.success and self._config.require_all_changes:
                # Rollback previously applied changes
                if self._config.rollback_on_failure and applied_backups:
                    await self._rollback_changes(applied_backups)

                elapsed = time.monotonic() - start_time
                result = FixApplicationResult(
                    proposal=proposal,
                    outcome=FixOutcome.FAILED,
                    change_results=change_results,
                    error_message=f"Change failed: {cr.message}",
                    duration_seconds=elapsed,
                )
                self._execution_history.append(result)
                return result

        elapsed = time.monotonic() - start_time
        outcome = (
            FixOutcome.DRY_RUN
            if self._config.dry_run
            else FixOutcome.APPLIED
        )
        result = FixApplicationResult(
            proposal=proposal,
            outcome=outcome,
            change_results=change_results,
            duration_seconds=elapsed,
        )
        self._execution_history.append(result)
        return result

    async def _rollback_changes(
        self, applied_backups: list[tuple[ProposedChange, str]]
    ) -> None:
        """Rollback a list of applied changes in reverse order."""
        for change, backup in reversed(applied_backups):
            success = await self._applier.rollback_change(change, backup)
            if success:
                logger.info("Rolled back change to %s", change.file_path)
            else:
                logger.error(
                    "Failed to rollback change to %s", change.file_path,
                )

    # ----- Rejection handling -----

    def build_rejection_feedback(
        self,
        proposal: FixProposal,
        response: ApprovalResponse,
        attempt_number: int = 1,
    ) -> RejectionFeedback:
        """Build structured rejection feedback from an approval response.

        Args:
            proposal: The rejected proposal.
            response: The approval response containing feedback.
            attempt_number: Which attempt this rejection is from.

        Returns:
            A RejectionFeedback object for the alternative generator.
        """
        # Find per-proposal feedback if available
        per_proposal_feedback = ""
        for decision in response.decisions:
            if (
                decision.failure_id == proposal.failure_id
                and decision.status == ApprovalStatus.REJECTED
            ):
                per_proposal_feedback = decision.feedback
                break

        return RejectionFeedback(
            proposal=proposal,
            user_feedback=per_proposal_feedback or response.feedback,
            rejection_reason=response.feedback or "User rejected the fix",
            attempt_number=attempt_number,
        )

    async def handle_rejection(
        self,
        proposal: FixProposal,
        feedback: RejectionFeedback,
        previous_alternatives: list[FixProposal] | None = None,
    ) -> RetryResult:
        """Handle a rejection by generating and submitting an alternative.

        This method:
        1. Checks the retry budget
        2. Generates an alternative proposal
        3. Submits it for approval (if coordinator is available)
        4. Applies it if approved
        5. Returns the retry result

        Without a coordinator, the alternative is generated but not
        submitted — it's returned for the caller to handle.

        Args:
            proposal: The rejected proposal.
            feedback: Structured rejection feedback.
            previous_alternatives: Previously rejected alternatives.

        Returns:
            RetryResult with the outcome of the retry attempt.
        """
        prev = previous_alternatives or []
        attempt = feedback.attempt_number

        # Check retry budget
        if attempt > self._config.max_retries:
            result = RetryResult(
                original_proposal=proposal,
                outcome=RetryOutcome.BUDGET_EXHAUSTED,
                feedback_history=[feedback],
                attempts_made=attempt,
            )
            self._retry_history.append(result)
            return result

        # Generate alternative
        try:
            alternative = await self._alternative_gen.generate_alternative(
                proposal, feedback, prev,
            )
        except Exception as exc:
            logger.error("Alternative generation failed: %s", exc)
            result = RetryResult(
                original_proposal=proposal,
                outcome=RetryOutcome.ERROR,
                feedback_history=[feedback],
                attempts_made=attempt,
            )
            self._retry_history.append(result)
            return result

        if alternative is None:
            result = RetryResult(
                original_proposal=proposal,
                outcome=RetryOutcome.NO_ALTERNATIVES,
                feedback_history=[feedback],
                attempts_made=attempt,
            )
            self._retry_history.append(result)
            return result

        # If no coordinator, return the alternative without submitting
        if self._coordinator is None:
            result = RetryResult(
                original_proposal=proposal,
                outcome=RetryOutcome.APPROVED,  # Tentatively — caller decides
                alternative_proposal=alternative,
                feedback_history=[feedback],
                attempts_made=attempt,
            )
            self._retry_history.append(result)
            return result

        # Submit alternative for approval
        alt_set = FixProposalSet(
            proposals=[alternative],
            analysis_summary=f"Alternative fix (attempt {attempt + 1})",
            total_failures_analyzed=1,
            total_proposals_generated=1,
        )

        try:
            alt_response = await self._coordinator.submit(
                alt_set,
                scope=ApprovalScope.SINGLE,
                context_summary=(
                    f"Alternative fix for {proposal.failure_id} "
                    f"(attempt {attempt + 1}/{self._config.max_retries}). "
                    f"Previous feedback: {feedback.user_feedback or 'none'}"
                ),
            )
        except Exception as exc:
            logger.error("Alternative approval submission failed: %s", exc)
            result = RetryResult(
                original_proposal=proposal,
                outcome=RetryOutcome.ERROR,
                alternative_proposal=alternative,
                feedback_history=[feedback],
                attempts_made=attempt,
            )
            self._retry_history.append(result)
            return result

        if alt_response.is_approved:
            # Apply the approved alternative
            app_result = await self.apply_proposal(alternative)
            result = RetryResult(
                original_proposal=proposal,
                outcome=RetryOutcome.APPROVED,
                alternative_proposal=alternative,
                application_result=app_result,
                feedback_history=[feedback],
                attempts_made=attempt + 1,
            )
            self._retry_history.append(result)
            return result

        # Alternative was also rejected — recurse if budget allows
        new_feedback = self.build_rejection_feedback(
            alternative, alt_response, attempt_number=attempt + 1,
        )
        return await self.handle_rejection(
            proposal,
            new_feedback,
            previous_alternatives=prev + [alternative],
        )

    # ----- Full execution cycle -----

    async def execute_approved(
        self,
        approved: Sequence[FixProposal],
        rejected: Sequence[FixProposal],
        response: ApprovalResponse,
    ) -> FixExecutionReport:
        """Execute approved fixes and handle rejections with retries.

        This is the main entry point after the approval gate returns.
        It applies approved fixes and optionally retries rejected ones
        with alternative suggestions.

        Args:
            approved: Proposals that were approved by the user.
            rejected: Proposals that were rejected by the user.
            response: The full approval response (for feedback extraction).

        Returns:
            A FixExecutionReport with all outcomes.
        """
        start_time = time.monotonic()

        applied_results: list[FixApplicationResult] = []
        rejection_feedbacks: list[RejectionFeedback] = []
        retry_results: list[RetryResult] = []
        skipped: list[FixProposal] = []

        # Apply approved proposals
        for proposal in approved:
            result = await self.apply_proposal(proposal)
            if result.outcome in (FixOutcome.APPLIED, FixOutcome.DRY_RUN):
                applied_results.append(result)
            elif result.outcome == FixOutcome.SKIPPED:
                skipped.append(proposal)
            else:
                applied_results.append(result)

        # Handle rejected proposals with retry
        for proposal in rejected:
            feedback = self.build_rejection_feedback(proposal, response)
            rejection_feedbacks.append(feedback)

            # Attempt retry with alternatives
            retry_result = await self.handle_rejection(proposal, feedback)
            retry_results.append(retry_result)

            # If the retry produced an applied result, add it
            if (
                retry_result.outcome == RetryOutcome.APPROVED
                and retry_result.application_result is not None
            ):
                applied_results.append(retry_result.application_result)

        elapsed = time.monotonic() - start_time

        return FixExecutionReport(
            applied=applied_results,
            rejected=rejection_feedbacks,
            retried=retry_results,
            skipped=skipped,
            total_proposals=len(approved) + len(rejected),
            total_applied=sum(
                1 for r in applied_results
                if r.outcome in (FixOutcome.APPLIED, FixOutcome.DRY_RUN)
            ),
            total_rejected=sum(
                1 for r in retry_results
                if r.outcome
                in (
                    RetryOutcome.REJECTED_AGAIN,
                    RetryOutcome.NO_ALTERNATIVES,
                    RetryOutcome.BUDGET_EXHAUSTED,
                )
            ),
            total_retried=len(retry_results),
            duration_seconds=elapsed,
        )

    async def run_fix_cycle(
        self,
        proposal_set: FixProposalSet,
        *,
        context_summary: str = "",
    ) -> FixExecutionReport:
        """Run the full approval-execute-retry cycle.

        This orchestrates the entire flow:
        1. Submit proposals for approval
        2. Apply approved fixes
        3. Retry rejected fixes with alternatives
        4. Return a comprehensive report

        Requires a coordinator to be configured.

        Args:
            proposal_set: Fix proposals from the troubleshooter.
            context_summary: Context for the approval request.

        Returns:
            Complete FixExecutionReport.

        Raises:
            RuntimeError: If no coordinator is configured.
        """
        if self._coordinator is None:
            raise RuntimeError(
                "FixExecutor requires an ApprovalCoordinator for run_fix_cycle"
            )

        if not proposal_set.proposals:
            return FixExecutionReport(
                total_proposals=0,
                duration_seconds=0.0,
            )

        # Submit for approval
        response = await self._coordinator.submit(
            proposal_set,
            context_summary=context_summary,
        )

        # Build the last request from coordinator history
        last_request = self._coordinator.history[-1][0]

        # Split into approved and rejected
        approved = self._coordinator.get_approved_proposals(
            last_request, response,
        )
        rejected_proposals = self._coordinator.get_rejected_proposals(
            last_request, response,
        )

        return await self.execute_approved(
            approved, rejected_proposals, response,
        )

    # ----- Reporting -----

    def summary(self) -> dict[str, Any]:
        """Serializable summary of executor state."""
        return {
            "config": {
                "max_retries": self._config.max_retries,
                "dry_run": self._config.dry_run,
                "rollback_on_failure": self._config.rollback_on_failure,
            },
            "execution_history_count": len(self._execution_history),
            "retry_history_count": len(self._retry_history),
            "total_applied": sum(
                1
                for r in self._execution_history
                if r.outcome == FixOutcome.APPLIED
            ),
            "total_failed": sum(
                1
                for r in self._execution_history
                if r.outcome == FixOutcome.FAILED
            ),
        }
