"""Tests for fix execution logic — applying approved fixes after user approval.

Tests verify that:
- Fixes are only applied after explicit user approval via ApprovalManager
- Approved proposals are applied through the ChangeApplier
- Rejected proposals trigger alternative generation and re-approval
- Dry-run mode previews without modifying files
- Rollback-on-failure restores files when a change fails
- Retry budget limits the number of rejection-retry cycles
- run_fix_cycle orchestrates the full approval→execute→retry flow
- The FixExecutor never bypasses the approval gate
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Sequence

import pytest

from test_runner.agents.troubleshooter.models import (
    FailureCategory,
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.autonomy.approval import (
    ApprovalCoordinator,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalScope,
    ApprovalStatus,
    CallbackApprovalGate,
    ProposalDecision,
)
from test_runner.autonomy.fix_executor import (
    AlternativeGenerator,
    ChangeApplier,
    ChangeResult,
    FileChangeApplier,
    FixApplicationResult,
    FixExecutionReport,
    FixExecutor,
    FixExecutorConfig,
    FixOutcome,
    PatternAlternativeGenerator,
    RejectionFeedback,
    RetryOutcome,
    RetryResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_change(
    *,
    file_path: str = "src/app.py",
    description: str = "Fix import",
    original_snippet: str = "import foo",
    proposed_snippet: str = "import bar",
    change_type: str = "modify",
) -> ProposedChange:
    return ProposedChange(
        file_path=file_path,
        description=description,
        original_snippet=original_snippet,
        proposed_snippet=proposed_snippet,
        change_type=change_type,
    )


def _make_proposal(
    *,
    failure_id: str = "test_foo",
    title: str = "Fix something",
    confidence: FixConfidence = FixConfidence.MEDIUM,
    confidence_score: float = 0.6,
    changes: list[ProposedChange] | None = None,
    alternative_fixes: list[str] | None = None,
) -> FixProposal:
    return FixProposal(
        failure_id=failure_id,
        title=title,
        description=f"Description for {title}",
        confidence=confidence,
        confidence_score=confidence_score,
        category=FailureCategory.ASSERTION,
        proposed_changes=changes or [],
        alternative_fixes=alternative_fixes or [],
    )


def _make_proposal_set(
    proposals: list[FixProposal] | None = None,
) -> FixProposalSet:
    props = proposals or [_make_proposal()]
    return FixProposalSet(
        proposals=props,
        analysis_summary="Test analysis",
        total_failures_analyzed=len(props),
        total_proposals_generated=len(props),
    )


def _approval_response(
    request_id: str = "test",
    status: ApprovalStatus = ApprovalStatus.APPROVED,
    decisions: list[ProposalDecision] | None = None,
    feedback: str = "",
) -> ApprovalResponse:
    return ApprovalResponse(
        request_id=request_id,
        status=status,
        decisions=decisions or [],
        feedback=feedback,
    )


class FakeChangeApplier:
    """ChangeApplier that tracks calls without touching the filesystem."""

    def __init__(
        self,
        *,
        fail_indices: set[int] | None = None,
    ) -> None:
        self.applied: list[ProposedChange] = []
        self.rolled_back: list[tuple[ProposedChange, str]] = []
        self._fail_indices = fail_indices or set()
        self._call_count = 0

    async def apply_change(
        self, change: ProposedChange, *, dry_run: bool = False
    ) -> ChangeResult:
        idx = self._call_count
        self._call_count += 1

        if idx in self._fail_indices:
            return ChangeResult(
                change=change,
                success=False,
                message=f"Simulated failure at index {idx}",
                dry_run=dry_run,
            )

        self.applied.append(change)
        return ChangeResult(
            change=change,
            success=True,
            message=f"Applied {change.change_type} to {change.file_path}",
            backup_content=f"backup-{change.file_path}",
            dry_run=dry_run,
        )

    async def rollback_change(
        self, change: ProposedChange, backup: str
    ) -> bool:
        self.rolled_back.append((change, backup))
        return True


class FakeAlternativeGenerator:
    """Returns predetermined alternatives or None."""

    def __init__(self, alternatives: list[FixProposal | None]) -> None:
        self._alternatives = list(alternatives)
        self._call_count = 0

    async def generate_alternative(
        self,
        rejected_proposal: FixProposal,
        feedback: RejectionFeedback,
        previous_alternatives: Sequence[FixProposal],
    ) -> FixProposal | None:
        if self._call_count < len(self._alternatives):
            alt = self._alternatives[self._call_count]
            self._call_count += 1
            return alt
        return None


# ---------------------------------------------------------------------------
# FixExecutor — apply_proposal tests
# ---------------------------------------------------------------------------


class TestApplyProposal:
    """Test that apply_proposal correctly applies changes only for approved proposals."""

    @pytest.mark.asyncio
    async def test_apply_proposal_with_changes(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)

        changes = [_make_change(), _make_change(file_path="src/util.py")]
        proposal = _make_proposal(changes=changes)
        result = await executor.apply_proposal(proposal)

        assert result.outcome == FixOutcome.APPLIED
        assert result.all_changes_succeeded is True
        assert result.applied_count == 2
        assert result.failed_count == 0
        assert len(applier.applied) == 2
        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_apply_proposal_no_changes_skipped(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)

        proposal = _make_proposal(changes=[])
        result = await executor.apply_proposal(proposal)

        assert result.outcome == FixOutcome.SKIPPED
        assert result.error_message == "No concrete changes to apply"
        assert len(applier.applied) == 0

    @pytest.mark.asyncio
    async def test_apply_proposal_dry_run(self):
        applier = FakeChangeApplier()
        config = FixExecutorConfig(dry_run=True)
        executor = FixExecutor(applier=applier, config=config)

        proposal = _make_proposal(changes=[_make_change()])
        result = await executor.apply_proposal(proposal)

        assert result.outcome == FixOutcome.DRY_RUN
        assert result.all_changes_succeeded is True
        # Dry-run still calls the applier (with dry_run=True)
        assert len(applier.applied) == 1

    @pytest.mark.asyncio
    async def test_apply_proposal_failure_with_rollback(self):
        # Second change fails → first should be rolled back
        applier = FakeChangeApplier(fail_indices={1})
        config = FixExecutorConfig(
            rollback_on_failure=True,
            require_all_changes=True,
        )
        executor = FixExecutor(applier=applier, config=config)

        changes = [
            _make_change(file_path="a.py"),
            _make_change(file_path="b.py"),
        ]
        proposal = _make_proposal(changes=changes)
        result = await executor.apply_proposal(proposal)

        assert result.outcome == FixOutcome.FAILED
        assert "Simulated failure" in result.error_message
        # First change was applied, then rolled back
        assert len(applier.applied) == 1
        assert len(applier.rolled_back) == 1
        assert applier.rolled_back[0][0].file_path == "a.py"

    @pytest.mark.asyncio
    async def test_apply_proposal_failure_without_rollback(self):
        applier = FakeChangeApplier(fail_indices={1})
        config = FixExecutorConfig(
            rollback_on_failure=False,
            require_all_changes=True,
        )
        executor = FixExecutor(applier=applier, config=config)

        changes = [
            _make_change(file_path="a.py"),
            _make_change(file_path="b.py"),
        ]
        proposal = _make_proposal(changes=changes)
        result = await executor.apply_proposal(proposal)

        assert result.outcome == FixOutcome.FAILED
        # No rollback should have occurred
        assert len(applier.rolled_back) == 0

    @pytest.mark.asyncio
    async def test_apply_proposal_tracks_history(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)

        p1 = _make_proposal(failure_id="f1", changes=[_make_change()])
        p2 = _make_proposal(failure_id="f2", changes=[_make_change()])

        await executor.apply_proposal(p1)
        await executor.apply_proposal(p2)

        assert len(executor.execution_history) == 2
        assert executor.execution_history[0].proposal.failure_id == "f1"
        assert executor.execution_history[1].proposal.failure_id == "f2"

    @pytest.mark.asyncio
    async def test_reset_clears_history(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)

        await executor.apply_proposal(
            _make_proposal(changes=[_make_change()])
        )
        assert len(executor.execution_history) == 1

        executor.reset()
        assert len(executor.execution_history) == 0
        assert len(executor.retry_history) == 0


# ---------------------------------------------------------------------------
# execute_approved — approval-gated execution
# ---------------------------------------------------------------------------


class TestExecuteApproved:
    """Test that execute_approved only applies fixes that were explicitly approved."""

    @pytest.mark.asyncio
    async def test_approved_proposals_are_applied(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)

        approved = [
            _make_proposal(failure_id="a", changes=[_make_change()]),
            _make_proposal(failure_id="b", changes=[_make_change()]),
        ]
        response = _approval_response(status=ApprovalStatus.APPROVED)
        report = await executor.execute_approved(approved, [], response)

        assert report.total_applied == 2
        assert report.total_proposals == 2
        assert len(report.applied) == 2
        assert all(
            r.outcome == FixOutcome.APPLIED for r in report.applied
        )

    @pytest.mark.asyncio
    async def test_rejected_proposals_are_not_applied(self):
        """Rejected proposals must NOT be applied — they go to retry instead."""
        applier = FakeChangeApplier()
        alt_gen = FakeAlternativeGenerator([None])  # No alternatives
        executor = FixExecutor(
            applier=applier, alternative_gen=alt_gen,
        )

        rejected = [
            _make_proposal(failure_id="r1", changes=[_make_change()]),
        ]
        response = _approval_response(
            status=ApprovalStatus.REJECTED,
            feedback="Not right",
        )

        report = await executor.execute_approved([], rejected, response)

        # The rejected proposal was not applied directly
        assert report.total_applied == 0
        assert len(report.rejected) == 1
        assert report.rejected[0].user_feedback == "Not right"
        # But retry was attempted
        assert report.total_retried == 1
        assert report.retried[0].outcome == RetryOutcome.NO_ALTERNATIVES

    @pytest.mark.asyncio
    async def test_mixed_approved_and_rejected(self):
        """Mix of approved and rejected proposals."""
        applier = FakeChangeApplier()
        alt_gen = FakeAlternativeGenerator([None])
        executor = FixExecutor(
            applier=applier, alternative_gen=alt_gen,
        )

        approved = [
            _make_proposal(failure_id="a", changes=[_make_change()]),
        ]
        rejected = [
            _make_proposal(failure_id="r", changes=[_make_change()]),
        ]
        response = _approval_response(
            status=ApprovalStatus.APPROVED,
            decisions=[
                ProposalDecision(
                    proposal_index=0, failure_id="a",
                    status=ApprovalStatus.APPROVED,
                ),
                ProposalDecision(
                    proposal_index=1, failure_id="r",
                    status=ApprovalStatus.REJECTED,
                    feedback="Looks wrong",
                ),
            ],
        )

        report = await executor.execute_approved(
            approved, rejected, response,
        )

        assert report.total_applied == 1
        assert report.total_rejected == 1
        assert report.total_proposals == 2

    @pytest.mark.asyncio
    async def test_skipped_proposals_tracked(self):
        """Proposals with no concrete changes are tracked as skipped."""
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)

        approved = [
            _make_proposal(failure_id="s", changes=[]),  # No changes
        ]
        response = _approval_response(status=ApprovalStatus.APPROVED)

        report = await executor.execute_approved(approved, [], response)

        assert len(report.skipped) == 1
        assert report.skipped[0].failure_id == "s"

    @pytest.mark.asyncio
    async def test_report_summary_line(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)

        report = await executor.execute_approved(
            [_make_proposal(changes=[_make_change()])],
            [],
            _approval_response(),
        )
        line = report.summary_line()
        assert "1 applied" in line

    @pytest.mark.asyncio
    async def test_report_has_failures_flag(self):
        applier = FakeChangeApplier(fail_indices={0})
        executor = FixExecutor(applier=applier)

        report = await executor.execute_approved(
            [_make_proposal(changes=[_make_change()])],
            [],
            _approval_response(),
        )
        assert report.has_failures is True


# ---------------------------------------------------------------------------
# Rejection handling and retry loop
# ---------------------------------------------------------------------------


class TestRejectionHandling:
    @pytest.mark.asyncio
    async def test_handle_rejection_budget_exhausted(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(
            applier=applier,
            config=FixExecutorConfig(max_retries=2),
        )

        proposal = _make_proposal()
        feedback = RejectionFeedback(
            proposal=proposal,
            user_feedback="No",
            attempt_number=3,  # Exceeds max_retries=2
        )

        result = await executor.handle_rejection(proposal, feedback)
        assert result.outcome == RetryOutcome.BUDGET_EXHAUSTED
        assert result.attempts_made == 3

    @pytest.mark.asyncio
    async def test_handle_rejection_no_alternatives(self):
        alt_gen = FakeAlternativeGenerator([None])
        applier = FakeChangeApplier()
        executor = FixExecutor(
            applier=applier,
            alternative_gen=alt_gen,
        )

        proposal = _make_proposal()
        feedback = RejectionFeedback(
            proposal=proposal,
            user_feedback="Wrong approach",
            attempt_number=1,
        )

        result = await executor.handle_rejection(proposal, feedback)
        assert result.outcome == RetryOutcome.NO_ALTERNATIVES

    @pytest.mark.asyncio
    async def test_handle_rejection_alternative_approved_via_coordinator(self):
        """Alternative is generated, approved, and applied."""
        alt_proposal = _make_proposal(
            failure_id="test_foo",
            title="Alternative fix",
            changes=[_make_change(file_path="alt.py")],
        )
        alt_gen = FakeAlternativeGenerator([alt_proposal])
        applier = FakeChangeApplier()

        # Coordinator that auto-approves
        async def approve_all(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
                decisions=[
                    ProposalDecision(
                        proposal_index=i,
                        failure_id=p.failure_id,
                        status=ApprovalStatus.APPROVED,
                    )
                    for i, p in enumerate(req.proposals)
                ],
            )

        gate = CallbackApprovalGate(callback=approve_all)
        coordinator = ApprovalCoordinator(gate=gate)

        executor = FixExecutor(
            applier=applier,
            coordinator=coordinator,
            alternative_gen=alt_gen,
        )

        proposal = _make_proposal()
        feedback = RejectionFeedback(
            proposal=proposal, user_feedback="Try something else",
            attempt_number=1,
        )

        result = await executor.handle_rejection(proposal, feedback)
        assert result.outcome == RetryOutcome.APPROVED
        assert result.alternative_proposal is not None
        assert result.alternative_proposal.title == "Alternative fix"
        assert result.application_result is not None
        assert result.application_result.outcome == FixOutcome.APPLIED

    @pytest.mark.asyncio
    async def test_handle_rejection_without_coordinator(self):
        """Without coordinator, alternative is returned but not submitted."""
        alt_proposal = _make_proposal(title="Alt fix")
        alt_gen = FakeAlternativeGenerator([alt_proposal])
        applier = FakeChangeApplier()

        executor = FixExecutor(
            applier=applier,
            coordinator=None,
            alternative_gen=alt_gen,
        )

        proposal = _make_proposal()
        feedback = RejectionFeedback(
            proposal=proposal, attempt_number=1,
        )

        result = await executor.handle_rejection(proposal, feedback)
        # Returns APPROVED tentatively — caller decides
        assert result.outcome == RetryOutcome.APPROVED
        assert result.alternative_proposal is not None
        assert result.application_result is None  # Not applied

    @pytest.mark.asyncio
    async def test_handle_rejection_alternative_also_rejected_recurses(self):
        """If alternative is also rejected, recursion continues until budget."""
        alt1 = _make_proposal(title="Alt 1")
        alt2 = _make_proposal(title="Alt 2")
        alt_gen = FakeAlternativeGenerator([alt1, alt2, None])

        rejection_count = 0

        async def always_reject(req: ApprovalRequest) -> ApprovalResponse:
            nonlocal rejection_count
            rejection_count += 1
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.REJECTED,
                feedback=f"Rejected attempt {rejection_count}",
                decisions=[
                    ProposalDecision(
                        proposal_index=i,
                        failure_id=p.failure_id,
                        status=ApprovalStatus.REJECTED,
                    )
                    for i, p in enumerate(req.proposals)
                ],
            )

        gate = CallbackApprovalGate(callback=always_reject)
        coordinator = ApprovalCoordinator(gate=gate)
        applier = FakeChangeApplier()

        executor = FixExecutor(
            applier=applier,
            coordinator=coordinator,
            alternative_gen=alt_gen,
            config=FixExecutorConfig(max_retries=3),
        )

        proposal = _make_proposal()
        feedback = RejectionFeedback(
            proposal=proposal, attempt_number=1,
        )

        result = await executor.handle_rejection(proposal, feedback)
        # Should exhaust alternatives then report NO_ALTERNATIVES
        assert result.outcome == RetryOutcome.NO_ALTERNATIVES

    @pytest.mark.asyncio
    async def test_handle_rejection_error_in_generator(self):
        """Error during alternative generation is caught gracefully."""

        class FailingGenerator:
            async def generate_alternative(self, *args, **kwargs):
                raise RuntimeError("LLM down")

        applier = FakeChangeApplier()
        executor = FixExecutor(
            applier=applier,
            alternative_gen=FailingGenerator(),  # type: ignore
        )

        proposal = _make_proposal()
        feedback = RejectionFeedback(proposal=proposal, attempt_number=1)

        result = await executor.handle_rejection(proposal, feedback)
        assert result.outcome == RetryOutcome.ERROR

    @pytest.mark.asyncio
    async def test_handle_rejection_error_in_approval_submission(self):
        """Error during coordinator.submit is caught gracefully.

        The CallbackApprovalGate catches callback errors and returns
        a REJECTED response, so the retry loop treats this as another
        rejection. With no more alternatives, it becomes NO_ALTERNATIVES.
        """
        alt_proposal = _make_proposal(title="Alt")
        alt_gen = FakeAlternativeGenerator([alt_proposal])

        async def error_gate(req: ApprovalRequest) -> ApprovalResponse:
            raise RuntimeError("Gate broken")

        gate = CallbackApprovalGate(callback=error_gate)
        coordinator = ApprovalCoordinator(gate=gate)
        applier = FakeChangeApplier()

        executor = FixExecutor(
            applier=applier,
            coordinator=coordinator,
            alternative_gen=alt_gen,
        )

        proposal = _make_proposal()
        feedback = RejectionFeedback(proposal=proposal, attempt_number=1)

        result = await executor.handle_rejection(proposal, feedback)
        # Gate error → REJECTED response → recurse → no more alts
        assert result.outcome == RetryOutcome.NO_ALTERNATIVES

    def test_build_rejection_feedback(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)

        proposal = _make_proposal(failure_id="f1")
        response = _approval_response(
            status=ApprovalStatus.REJECTED,
            feedback="Overall feedback",
            decisions=[
                ProposalDecision(
                    proposal_index=0,
                    failure_id="f1",
                    status=ApprovalStatus.REJECTED,
                    feedback="Specific feedback for f1",
                ),
            ],
        )

        fb = executor.build_rejection_feedback(proposal, response)
        assert fb.user_feedback == "Specific feedback for f1"
        assert fb.attempt_number == 1

    def test_build_rejection_feedback_fallback_to_overall(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)

        proposal = _make_proposal(failure_id="f2")
        response = _approval_response(
            status=ApprovalStatus.REJECTED,
            feedback="Overall feedback",
        )

        fb = executor.build_rejection_feedback(proposal, response)
        assert fb.user_feedback == "Overall feedback"


# ---------------------------------------------------------------------------
# run_fix_cycle — full approval → execute → retry loop
# ---------------------------------------------------------------------------


class TestRunFixCycle:
    @pytest.mark.asyncio
    async def test_run_fix_cycle_requires_coordinator(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier, coordinator=None)

        with pytest.raises(RuntimeError, match="requires an ApprovalCoordinator"):
            await executor.run_fix_cycle(_make_proposal_set())

    @pytest.mark.asyncio
    async def test_run_fix_cycle_empty_proposals(self):
        async def approve_all(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )

        gate = CallbackApprovalGate(callback=approve_all)
        coordinator = ApprovalCoordinator(gate=gate)
        applier = FakeChangeApplier()

        executor = FixExecutor(applier=applier, coordinator=coordinator)
        report = await executor.run_fix_cycle(FixProposalSet())

        assert report.total_proposals == 0
        assert report.total_applied == 0

    @pytest.mark.asyncio
    async def test_run_fix_cycle_all_approved(self):
        """Full cycle: all proposals approved and applied."""

        async def approve_all(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
                decisions=[
                    ProposalDecision(
                        proposal_index=i,
                        failure_id=p.failure_id,
                        status=ApprovalStatus.APPROVED,
                    )
                    for i, p in enumerate(req.proposals)
                ],
            )

        gate = CallbackApprovalGate(callback=approve_all)
        coordinator = ApprovalCoordinator(gate=gate)
        applier = FakeChangeApplier()

        executor = FixExecutor(applier=applier, coordinator=coordinator)

        proposals = [
            _make_proposal(failure_id="a", changes=[_make_change()]),
            _make_proposal(failure_id="b", changes=[_make_change()]),
        ]
        ps = _make_proposal_set(proposals)

        report = await executor.run_fix_cycle(ps, context_summary="test")

        assert report.total_applied == 2
        assert report.total_rejected == 0

    @pytest.mark.asyncio
    async def test_run_fix_cycle_all_rejected_no_alternatives(self):
        """Full cycle: all proposals rejected, no alternatives available."""

        async def reject_all(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.REJECTED,
                feedback="All wrong",
                decisions=[
                    ProposalDecision(
                        proposal_index=i,
                        failure_id=p.failure_id,
                        status=ApprovalStatus.REJECTED,
                    )
                    for i, p in enumerate(req.proposals)
                ],
            )

        gate = CallbackApprovalGate(callback=reject_all)
        coordinator = ApprovalCoordinator(gate=gate)
        applier = FakeChangeApplier()
        alt_gen = FakeAlternativeGenerator([None])

        executor = FixExecutor(
            applier=applier,
            coordinator=coordinator,
            alternative_gen=alt_gen,
        )

        proposals = [
            _make_proposal(failure_id="r1", changes=[_make_change()]),
        ]
        ps = _make_proposal_set(proposals)
        report = await executor.run_fix_cycle(ps)

        assert report.total_applied == 0
        assert report.total_rejected == 1
        assert report.total_retried == 1

    @pytest.mark.asyncio
    async def test_run_fix_cycle_mixed_with_retry_success(self):
        """One approved, one rejected with a successful retry alternative."""
        call_count = 0

        async def mixed_gate(req: ApprovalRequest) -> ApprovalResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: approve first, reject second
                return ApprovalResponse(
                    request_id=req.request_id,
                    status=ApprovalStatus.APPROVED,
                    decisions=[
                        ProposalDecision(
                            proposal_index=0,
                            failure_id=req.proposals[0].failure_id,
                            status=ApprovalStatus.APPROVED,
                        ),
                        ProposalDecision(
                            proposal_index=1,
                            failure_id=req.proposals[1].failure_id,
                            status=ApprovalStatus.REJECTED,
                            feedback="Try differently",
                        ),
                    ],
                )
            else:
                # Second call (retry): approve the alternative
                return ApprovalResponse(
                    request_id=req.request_id,
                    status=ApprovalStatus.APPROVED,
                    decisions=[
                        ProposalDecision(
                            proposal_index=0,
                            failure_id=req.proposals[0].failure_id,
                            status=ApprovalStatus.APPROVED,
                        ),
                    ],
                )

        gate = CallbackApprovalGate(callback=mixed_gate)
        coordinator = ApprovalCoordinator(gate=gate)
        applier = FakeChangeApplier()

        alt_proposal = _make_proposal(
            failure_id="r1",
            title="Better fix",
            changes=[_make_change(file_path="alt.py")],
        )
        alt_gen = FakeAlternativeGenerator([alt_proposal])

        executor = FixExecutor(
            applier=applier,
            coordinator=coordinator,
            alternative_gen=alt_gen,
        )

        proposals = [
            _make_proposal(failure_id="a1", changes=[_make_change()]),
            _make_proposal(failure_id="r1", changes=[_make_change()]),
        ]
        ps = _make_proposal_set(proposals)
        report = await executor.run_fix_cycle(ps)

        # First proposal approved directly, second approved via retry
        assert report.total_applied == 2
        assert report.total_retried == 1


# ---------------------------------------------------------------------------
# PatternAlternativeGenerator
# ---------------------------------------------------------------------------


class TestPatternAlternativeGenerator:
    @pytest.mark.asyncio
    async def test_generates_from_alternative_fixes_list(self):
        gen = PatternAlternativeGenerator()
        proposal = _make_proposal(
            alternative_fixes=["Use mock instead", "Disable caching"],
        )
        feedback = RejectionFeedback(
            proposal=proposal,
            user_feedback="Too risky",
            attempt_number=1,
        )

        alt = await gen.generate_alternative(proposal, feedback, [])
        assert alt is not None
        assert "Use mock instead" in alt.title
        assert alt.confidence == FixConfidence.LOW
        assert alt.metadata.get("is_alternative") is True

    @pytest.mark.asyncio
    async def test_skips_previously_used_alternatives(self):
        gen = PatternAlternativeGenerator()
        proposal = _make_proposal(
            alternative_fixes=["Alt A", "Alt B"],
        )
        # "Alt A" was already used
        prev = [_make_proposal(title="Alt A")]
        feedback = RejectionFeedback(
            proposal=proposal, attempt_number=1,
        )

        alt = await gen.generate_alternative(proposal, feedback, prev)
        assert alt is not None
        assert "Alt B" in alt.title

    @pytest.mark.asyncio
    async def test_returns_none_when_no_alternatives_remain(self):
        gen = PatternAlternativeGenerator()
        proposal = _make_proposal(alternative_fixes=[])
        feedback = RejectionFeedback(proposal=proposal, attempt_number=1)

        alt = await gen.generate_alternative(proposal, feedback, [])
        assert alt is None

    @pytest.mark.asyncio
    async def test_delegates_to_llm_generator_when_no_patterns(self):
        llm_alt = _make_proposal(title="LLM suggestion")

        class FakeLLMGen:
            async def generate_alternative(self, *args, **kwargs):
                return llm_alt

        gen = PatternAlternativeGenerator(llm_generator=FakeLLMGen())  # type: ignore
        proposal = _make_proposal(alternative_fixes=[])
        feedback = RejectionFeedback(proposal=proposal, attempt_number=1)

        alt = await gen.generate_alternative(proposal, feedback, [])
        assert alt is not None
        assert alt.title == "LLM suggestion"


# ---------------------------------------------------------------------------
# FileChangeApplier — real filesystem tests
# ---------------------------------------------------------------------------


class TestFileChangeApplier:
    @pytest.mark.asyncio
    async def test_apply_modify_change(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write("import foo\nprint('hello')\n")
            path = f.name

        try:
            applier = FileChangeApplier()
            change = _make_change(
                file_path=path,
                original_snippet="import foo",
                proposed_snippet="import bar",
                change_type="modify",
            )
            result = await applier.apply_change(change)

            assert result.success is True
            with open(path) as f:
                content = f.read()
            assert "import bar" in content
            assert "import foo" not in content
            assert result.backup_content == "import foo\nprint('hello')\n"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_apply_modify_file_not_found(self):
        applier = FileChangeApplier()
        change = _make_change(
            file_path="/nonexistent/path/file.py",
            change_type="modify",
        )
        result = await applier.apply_change(change)
        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_apply_modify_snippet_not_found(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write("different content\n")
            path = f.name

        try:
            applier = FileChangeApplier()
            change = _make_change(
                file_path=path,
                original_snippet="import foo",
                change_type="modify",
            )
            result = await applier.apply_change(change)
            assert result.success is False
            assert "not found" in result.message.lower()
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_apply_add_change(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write("# existing\n")
            path = f.name

        try:
            applier = FileChangeApplier()
            change = _make_change(
                file_path=path,
                proposed_snippet="\n# new line\n",
                change_type="add",
            )
            result = await applier.apply_change(change)
            assert result.success is True
            with open(path) as f:
                content = f.read()
            assert "# existing" in content
            assert "# new line" in content
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_apply_delete_change(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write("keep\nremove_me\nkeep\n")
            path = f.name

        try:
            applier = FileChangeApplier()
            change = _make_change(
                file_path=path,
                original_snippet="remove_me\n",
                change_type="delete",
            )
            result = await applier.apply_change(change)
            assert result.success is True
            with open(path) as f:
                content = f.read()
            assert "remove_me" not in content
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_dry_run_does_not_modify(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write("import foo\n")
            path = f.name

        try:
            applier = FileChangeApplier()
            change = _make_change(
                file_path=path,
                original_snippet="import foo",
                proposed_snippet="import bar",
                change_type="modify",
            )
            result = await applier.apply_change(change, dry_run=True)
            assert result.success is True
            assert result.dry_run is True
            # File should be unchanged
            with open(path) as f:
                content = f.read()
            assert "import foo" in content
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_rollback_restores_content(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write("modified content\n")
            path = f.name

        try:
            applier = FileChangeApplier()
            change = _make_change(file_path=path)
            original = "original content\n"

            success = await applier.rollback_change(change, original)
            assert success is True
            with open(path) as f:
                content = f.read()
            assert content == original
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_working_directory_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.py")
            with open(filepath, "w") as f:
                f.write("import foo\n")

            applier = FileChangeApplier(working_directory=tmpdir)
            change = _make_change(
                file_path="test.py",
                original_snippet="import foo",
                proposed_snippet="import bar",
            )
            result = await applier.apply_change(change)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_unknown_change_type_fails(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write("content\n")
            path = f.name

        try:
            applier = FileChangeApplier()
            change = _make_change(
                file_path=path,
                change_type="unknown_type",
            )
            result = await applier.apply_change(change)
            assert result.success is False
            assert "unknown" in result.message.lower()
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_modify_no_original_snippet_fails(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write("content\n")
            path = f.name

        try:
            applier = FileChangeApplier()
            change = _make_change(
                file_path=path,
                original_snippet="",
                change_type="modify",
            )
            result = await applier.apply_change(change)
            assert result.success is False
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Model / report tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_fix_outcome_values(self):
        assert FixOutcome.APPLIED.value == "applied"
        assert FixOutcome.FAILED.value == "failed"
        assert FixOutcome.DRY_RUN.value == "dry_run"
        assert FixOutcome.ROLLED_BACK.value == "rolled_back"

    def test_retry_outcome_values(self):
        assert RetryOutcome.APPROVED.value == "approved"
        assert RetryOutcome.BUDGET_EXHAUSTED.value == "budget_exhausted"
        assert RetryOutcome.NO_ALTERNATIVES.value == "no_alternatives"

    def test_fix_application_result_properties(self):
        proposal = _make_proposal(changes=[_make_change()])
        cr_ok = ChangeResult(
            change=_make_change(), success=True, message="ok",
        )
        cr_fail = ChangeResult(
            change=_make_change(), success=False, message="fail",
        )

        result = FixApplicationResult(
            proposal=proposal,
            outcome=FixOutcome.APPLIED,
            change_results=[cr_ok, cr_fail],
        )
        assert result.all_changes_succeeded is False
        assert result.applied_count == 1
        assert result.failed_count == 1

    def test_fix_execution_report_summary(self):
        report = FixExecutionReport(
            total_proposals=5,
            total_applied=3,
            total_rejected=1,
            total_retried=1,
        )
        line = report.summary_line()
        assert "3 applied" in line
        assert "1 rejected" in line
        assert "1 retried" in line

    def test_executor_summary(self):
        applier = FakeChangeApplier()
        executor = FixExecutor(applier=applier)
        s = executor.summary()
        assert s["config"]["max_retries"] == 3
        assert s["execution_history_count"] == 0

    def test_rejection_feedback_frozen(self):
        fb = RejectionFeedback(
            proposal=_make_proposal(),
            user_feedback="test",
        )
        with pytest.raises(Exception):
            fb.user_feedback = "changed"  # type: ignore

    def test_change_result_frozen(self):
        cr = ChangeResult(
            change=_make_change(), success=True,
        )
        with pytest.raises(Exception):
            cr.success = False  # type: ignore


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_file_change_applier_is_change_applier(self):
        assert isinstance(FileChangeApplier(), ChangeApplier)

    def test_pattern_alternative_generator_is_alternative_generator(self):
        assert isinstance(
            PatternAlternativeGenerator(), AlternativeGenerator
        )

    def test_fake_change_applier_is_change_applier(self):
        assert isinstance(FakeChangeApplier(), ChangeApplier)
