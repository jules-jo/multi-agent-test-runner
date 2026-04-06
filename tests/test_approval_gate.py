"""Tests for the human-in-the-loop approval gate workflow.

Tests cover:
- ApprovalRequest/ApprovalResponse model behavior
- CliApprovalGate with injected input (single, batch, individual)
- CallbackApprovalGate (sync, async, timeout, error)
- AutoApprovalGate (threshold routing, fallback delegation)
- ApprovalCoordinator (build, submit, history, approved/rejected extraction)
- Edge cases: empty proposals, timeout, keyboard interrupt
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

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
    AutoApprovalGate,
    CallbackApprovalGate,
    CliApprovalGate,
    ProposalDecision,
)
from test_runner.autonomy.policy import AutonomyPolicyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proposal(
    *,
    failure_id: str = "test_foo",
    title: str = "Fix something",
    confidence: FixConfidence = FixConfidence.MEDIUM,
    confidence_score: float = 0.6,
    category: FailureCategory = FailureCategory.ASSERTION,
    affected_files: list[str] | None = None,
    proposed_changes: list[ProposedChange] | None = None,
    requires_user_action: bool = False,
) -> FixProposal:
    return FixProposal(
        failure_id=failure_id,
        title=title,
        description=f"Description for {title}",
        confidence=confidence,
        confidence_score=confidence_score,
        category=category,
        affected_files=affected_files or [],
        proposed_changes=proposed_changes or [],
        requires_user_action=requires_user_action,
    )


def _make_proposal_set(proposals: list[FixProposal] | None = None) -> FixProposalSet:
    props = proposals or [_make_proposal()]
    return FixProposalSet(
        proposals=props,
        analysis_summary="Test analysis",
        total_failures_analyzed=len(props),
        total_proposals_generated=len(props),
    )


def _make_input_fn(responses: list[str]):
    """Create an input function that returns responses in order."""
    queue = deque(responses)

    def _input(prompt: str) -> str:
        if not queue:
            raise EOFError("No more responses")
        return queue.popleft()

    return _input


# ---------------------------------------------------------------------------
# ApprovalRequest tests
# ---------------------------------------------------------------------------


class TestApprovalRequest:
    def test_proposal_count(self):
        req = ApprovalRequest(
            request_id="test-001",
            proposals=[_make_proposal(), _make_proposal(failure_id="test_bar")],
        )
        assert req.proposal_count == 2

    def test_high_confidence_count(self):
        req = ApprovalRequest(
            request_id="test-001",
            proposals=[
                _make_proposal(confidence=FixConfidence.HIGH),
                _make_proposal(confidence=FixConfidence.LOW),
                _make_proposal(confidence=FixConfidence.HIGH),
            ],
        )
        assert req.high_confidence_count == 2

    def test_format_summary(self):
        req = ApprovalRequest(
            request_id="test-001",
            proposals=[_make_proposal()],
            context_summary="Running pytest suite",
            failure_count=3,
        )
        summary = req.format_summary()
        assert "test-001" in summary
        assert "Proposals: 1" in summary
        assert "Running pytest suite" in summary

    def test_format_summary_no_context(self):
        req = ApprovalRequest(request_id="test-002", proposals=[])
        summary = req.format_summary()
        assert "Proposals: 0" in summary
        assert "Context" not in summary


# ---------------------------------------------------------------------------
# ApprovalResponse tests
# ---------------------------------------------------------------------------


class TestApprovalResponse:
    def test_is_approved(self):
        resp = ApprovalResponse(
            request_id="test-001",
            status=ApprovalStatus.APPROVED,
        )
        assert resp.is_approved is True
        assert resp.is_rejected is False

    def test_is_auto_approved(self):
        resp = ApprovalResponse(
            request_id="test-001",
            status=ApprovalStatus.AUTO_APPROVED,
        )
        assert resp.is_approved is True

    def test_is_rejected(self):
        resp = ApprovalResponse(
            request_id="test-001",
            status=ApprovalStatus.REJECTED,
        )
        assert resp.is_rejected is True
        assert resp.is_approved is False

    def test_approved_indices(self):
        resp = ApprovalResponse(
            request_id="test-001",
            status=ApprovalStatus.APPROVED,
            decisions=[
                ProposalDecision(proposal_index=0, failure_id="a", status=ApprovalStatus.APPROVED),
                ProposalDecision(proposal_index=1, failure_id="b", status=ApprovalStatus.REJECTED),
                ProposalDecision(proposal_index=2, failure_id="c", status=ApprovalStatus.APPROVED),
            ],
        )
        assert resp.approved_indices == [0, 2]
        assert resp.rejected_indices == [1]

    def test_no_decisions(self):
        resp = ApprovalResponse(
            request_id="test-001",
            status=ApprovalStatus.APPROVED,
        )
        assert resp.approved_indices == []
        assert resp.rejected_indices == []


# ---------------------------------------------------------------------------
# ProposalDecision tests
# ---------------------------------------------------------------------------


class TestProposalDecision:
    def test_frozen(self):
        d = ProposalDecision(
            proposal_index=0,
            failure_id="test_foo",
            status=ApprovalStatus.APPROVED,
        )
        with pytest.raises(Exception):
            d.status = ApprovalStatus.REJECTED  # type: ignore[misc]

    def test_with_feedback(self):
        d = ProposalDecision(
            proposal_index=0,
            failure_id="test_foo",
            status=ApprovalStatus.REJECTED,
            feedback="Not sure this is right",
        )
        assert d.feedback == "Not sure this is right"


# ---------------------------------------------------------------------------
# CliApprovalGate tests
# ---------------------------------------------------------------------------


class TestCliApprovalGate:
    @pytest.mark.asyncio
    async def test_single_approve(self):
        output: list[str] = []
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["y"]),
            output_fn=output.append,
        )
        request = ApprovalRequest(
            request_id="cli-001",
            proposals=[_make_proposal()],
            scope=ApprovalScope.SINGLE,
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.APPROVED
        assert response.is_approved is True
        assert len(response.decisions) == 1
        assert response.decisions[0].status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_single_reject_with_reason(self):
        output: list[str] = []
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["n", "I disagree with the fix"]),
            output_fn=output.append,
        )
        request = ApprovalRequest(
            request_id="cli-002",
            proposals=[_make_proposal()],
            scope=ApprovalScope.SINGLE,
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.REJECTED
        assert response.feedback == "I disagree with the fix"

    @pytest.mark.asyncio
    async def test_single_skip(self):
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["s"]),
            output_fn=lambda _: None,
        )
        request = ApprovalRequest(
            request_id="cli-003",
            proposals=[_make_proposal()],
            scope=ApprovalScope.SINGLE,
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_batch_approve_all(self):
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["y"]),
            output_fn=lambda _: None,
        )
        proposals = [_make_proposal(failure_id=f"t{i}") for i in range(3)]
        request = ApprovalRequest(
            request_id="cli-004",
            proposals=proposals,
            scope=ApprovalScope.BATCH,
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.APPROVED
        assert len(response.decisions) == 3
        assert all(d.status == ApprovalStatus.APPROVED for d in response.decisions)

    @pytest.mark.asyncio
    async def test_batch_reject_all(self):
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["n", ""]),
            output_fn=lambda _: None,
        )
        proposals = [_make_proposal(failure_id=f"t{i}") for i in range(2)]
        request = ApprovalRequest(
            request_id="cli-005",
            proposals=proposals,
            scope=ApprovalScope.BATCH,
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_batch_individual_review(self):
        # "i" for individual, then "y", "n", "s" for each
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["i", "y", "n", "s"]),
            output_fn=lambda _: None,
        )
        proposals = [
            _make_proposal(failure_id="t0", title="Fix A"),
            _make_proposal(failure_id="t1", title="Fix B"),
            _make_proposal(failure_id="t2", title="Fix C"),
        ]
        request = ApprovalRequest(
            request_id="cli-006",
            proposals=proposals,
            scope=ApprovalScope.BATCH,
        )
        response = await gate.request_approval(request)
        # Overall approved because at least one approved
        assert response.status == ApprovalStatus.APPROVED
        assert response.decisions[0].status == ApprovalStatus.APPROVED
        assert response.decisions[1].status == ApprovalStatus.REJECTED
        assert response.decisions[2].status == ApprovalStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_eof_defaults_to_reject(self):
        gate = CliApprovalGate(
            input_fn=_make_input_fn([]),  # Will raise EOFError
            output_fn=lambda _: None,
        )
        request = ApprovalRequest(
            request_id="cli-007",
            proposals=[_make_proposal()],
            scope=ApprovalScope.SINGLE,
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_response_time_recorded(self):
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["y"]),
            output_fn=lambda _: None,
        )
        request = ApprovalRequest(
            request_id="cli-008",
            proposals=[_make_proposal()],
            scope=ApprovalScope.SINGLE,
        )
        response = await gate.request_approval(request)
        assert response.response_time_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_individual_all_skipped(self):
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["i", "s", "s"]),
            output_fn=lambda _: None,
        )
        proposals = [
            _make_proposal(failure_id="t0"),
            _make_proposal(failure_id="t1"),
        ]
        request = ApprovalRequest(
            request_id="cli-009",
            proposals=proposals,
            scope=ApprovalScope.BATCH,
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_display_includes_proposal_details(self):
        output: list[str] = []
        proposal = _make_proposal(
            title="Fix import error",
            confidence=FixConfidence.HIGH,
            confidence_score=0.95,
            affected_files=["src/app.py"],
            proposed_changes=[
                ProposedChange(
                    file_path="src/app.py",
                    description="Fix import",
                    original_snippet="import foo",
                    proposed_snippet="import bar",
                    change_type="modify",
                )
            ],
            requires_user_action=True,
        )
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["y"]),
            output_fn=output.append,
        )
        request = ApprovalRequest(
            request_id="cli-010",
            proposals=[proposal],
            scope=ApprovalScope.SINGLE,
        )
        await gate.request_approval(request)
        full_output = "\n".join(output)
        assert "Fix import error" in full_output
        assert "HIGH" in full_output
        assert "src/app.py" in full_output


# ---------------------------------------------------------------------------
# CallbackApprovalGate tests
# ---------------------------------------------------------------------------


class TestCallbackApprovalGate:
    @pytest.mark.asyncio
    async def test_sync_callback(self):
        def callback(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )

        gate = CallbackApprovalGate(callback=callback)
        request = ApprovalRequest(
            request_id="cb-001",
            proposals=[_make_proposal()],
        )
        response = await gate.request_approval(request)
        assert response.is_approved is True

    @pytest.mark.asyncio
    async def test_async_callback(self):
        async def callback(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.REJECTED,
                feedback="Not now",
            )

        gate = CallbackApprovalGate(callback=callback)
        request = ApprovalRequest(
            request_id="cb-002",
            proposals=[_make_proposal()],
        )
        response = await gate.request_approval(request)
        assert response.is_rejected is True
        assert response.feedback == "Not now"

    @pytest.mark.asyncio
    async def test_callback_timeout(self):
        async def slow_callback(req: ApprovalRequest) -> ApprovalResponse:
            await asyncio.sleep(10)
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )

        gate = CallbackApprovalGate(callback=slow_callback, timeout_seconds=0.1)
        request = ApprovalRequest(
            request_id="cb-003",
            proposals=[_make_proposal()],
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.TIMED_OUT

    @pytest.mark.asyncio
    async def test_callback_error(self):
        def failing_callback(req: ApprovalRequest) -> ApprovalResponse:
            raise RuntimeError("Network error")

        gate = CallbackApprovalGate(callback=failing_callback)
        request = ApprovalRequest(
            request_id="cb-004",
            proposals=[_make_proposal()],
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.REJECTED
        assert "Network error" in response.feedback

    @pytest.mark.asyncio
    async def test_request_timeout_overrides_default(self):
        async def slow_callback(req: ApprovalRequest) -> ApprovalResponse:
            await asyncio.sleep(10)
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )

        gate = CallbackApprovalGate(callback=slow_callback, timeout_seconds=60)
        request = ApprovalRequest(
            request_id="cb-005",
            proposals=[_make_proposal()],
            timeout_seconds=0.1,  # Override default
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.TIMED_OUT


# ---------------------------------------------------------------------------
# AutoApprovalGate tests
# ---------------------------------------------------------------------------


class TestAutoApprovalGate:
    @pytest.mark.asyncio
    async def test_auto_approve_high_confidence(self):
        gate = AutoApprovalGate(min_confidence=0.85)
        request = ApprovalRequest(
            request_id="auto-001",
            proposals=[
                _make_proposal(
                    confidence=FixConfidence.HIGH,
                    confidence_score=0.95,
                ),
            ],
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.AUTO_APPROVED
        assert response.decisions[0].status == ApprovalStatus.AUTO_APPROVED

    @pytest.mark.asyncio
    async def test_reject_low_confidence_no_fallback(self):
        gate = AutoApprovalGate(min_confidence=0.85)
        request = ApprovalRequest(
            request_id="auto-002",
            proposals=[
                _make_proposal(
                    confidence=FixConfidence.LOW,
                    confidence_score=0.3,
                ),
            ],
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.REJECTED
        assert response.decisions[0].status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_mixed_with_fallback(self):
        # Fallback gate that approves everything
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

        fallback = CallbackApprovalGate(callback=approve_all)
        gate = AutoApprovalGate(min_confidence=0.85, fallback_gate=fallback)

        request = ApprovalRequest(
            request_id="auto-003",
            proposals=[
                _make_proposal(
                    failure_id="high",
                    confidence=FixConfidence.HIGH,
                    confidence_score=0.95,
                ),
                _make_proposal(
                    failure_id="low",
                    confidence=FixConfidence.LOW,
                    confidence_score=0.3,
                ),
            ],
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.APPROVED

        # First should be auto-approved, second manually approved via fallback
        decisions_by_idx = {d.proposal_index: d for d in response.decisions}
        assert decisions_by_idx[0].status == ApprovalStatus.AUTO_APPROVED
        assert decisions_by_idx[1].status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_custom_allowed_confidences(self):
        gate = AutoApprovalGate(
            min_confidence=0.5,
            allowed_confidences=frozenset({FixConfidence.HIGH, FixConfidence.MEDIUM}),
        )
        request = ApprovalRequest(
            request_id="auto-004",
            proposals=[
                _make_proposal(
                    confidence=FixConfidence.MEDIUM,
                    confidence_score=0.6,
                ),
            ],
        )
        response = await gate.request_approval(request)
        assert response.status == ApprovalStatus.AUTO_APPROVED


# ---------------------------------------------------------------------------
# ApprovalCoordinator tests
# ---------------------------------------------------------------------------


class TestApprovalCoordinator:
    @pytest.mark.asyncio
    async def test_submit_and_history(self):
        async def approve_cb(req: ApprovalRequest) -> ApprovalResponse:
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

        gate = CallbackApprovalGate(callback=approve_cb)
        coord = ApprovalCoordinator(gate=gate)

        proposal_set = _make_proposal_set()
        response = await coord.submit(
            proposal_set,
            context_summary="Test run",
        )
        assert response.is_approved is True
        assert coord.total_requests == 1
        assert coord.total_approved == 1
        assert coord.total_rejected == 0

    @pytest.mark.asyncio
    async def test_build_request_increments_id(self):
        gate = CallbackApprovalGate(
            callback=lambda req: ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )
        )
        coord = ApprovalCoordinator(gate=gate)

        ps = _make_proposal_set()
        req1 = coord.build_request(ps)
        req2 = coord.build_request(ps)
        assert req1.request_id == "approval-0001"
        assert req2.request_id == "approval-0002"

    @pytest.mark.asyncio
    async def test_get_approved_proposals_blanket(self):
        gate = CallbackApprovalGate(
            callback=lambda req: ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )
        )
        coord = ApprovalCoordinator(gate=gate)

        proposals = [_make_proposal(failure_id="a"), _make_proposal(failure_id="b")]
        ps = _make_proposal_set(proposals)
        request = coord.build_request(ps)
        response = ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.APPROVED,
        )
        approved = coord.get_approved_proposals(request, response)
        assert len(approved) == 2

    @pytest.mark.asyncio
    async def test_get_approved_proposals_individual(self):
        gate = CallbackApprovalGate(
            callback=lambda req: ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )
        )
        coord = ApprovalCoordinator(gate=gate)

        proposals = [
            _make_proposal(failure_id="a"),
            _make_proposal(failure_id="b"),
            _make_proposal(failure_id="c"),
        ]
        ps = _make_proposal_set(proposals)
        request = coord.build_request(ps)
        response = ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.APPROVED,
            decisions=[
                ProposalDecision(proposal_index=0, failure_id="a", status=ApprovalStatus.APPROVED),
                ProposalDecision(proposal_index=1, failure_id="b", status=ApprovalStatus.REJECTED),
                ProposalDecision(proposal_index=2, failure_id="c", status=ApprovalStatus.APPROVED),
            ],
        )
        approved = coord.get_approved_proposals(request, response)
        assert len(approved) == 2
        assert all(p.failure_id in ("a", "c") for p in approved)

        rejected = coord.get_rejected_proposals(request, response)
        assert len(rejected) == 1
        assert rejected[0].failure_id == "b"

    @pytest.mark.asyncio
    async def test_get_rejected_proposals_blanket(self):
        gate = CallbackApprovalGate(
            callback=lambda req: ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.REJECTED,
            )
        )
        coord = ApprovalCoordinator(gate=gate)

        proposals = [_make_proposal(failure_id="a")]
        ps = _make_proposal_set(proposals)
        request = coord.build_request(ps)
        response = ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.REJECTED,
        )
        rejected = coord.get_rejected_proposals(request, response)
        assert len(rejected) == 1

    @pytest.mark.asyncio
    async def test_auto_approvable_flag(self):
        gate = CallbackApprovalGate(
            callback=lambda req: ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )
        )
        # auto_fix_enabled with conservative policy (execute_threshold=0.95)
        policy = AutonomyPolicyConfig.conservative()
        coord = ApprovalCoordinator(
            gate=gate, policy=policy, auto_fix_enabled=True
        )

        # All HIGH confidence above threshold
        proposals = [
            _make_proposal(
                confidence=FixConfidence.HIGH,
                confidence_score=0.96,
            ),
        ]
        ps = _make_proposal_set(proposals)
        request = coord.build_request(ps)
        assert request.auto_approvable is True

    @pytest.mark.asyncio
    async def test_not_auto_approvable_when_disabled(self):
        gate = CallbackApprovalGate(
            callback=lambda req: ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )
        )
        coord = ApprovalCoordinator(gate=gate, auto_fix_enabled=False)

        proposals = [
            _make_proposal(confidence=FixConfidence.HIGH, confidence_score=0.99),
        ]
        ps = _make_proposal_set(proposals)
        request = coord.build_request(ps)
        assert request.auto_approvable is False

    @pytest.mark.asyncio
    async def test_summary(self):
        async def approve_cb(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )

        gate = CallbackApprovalGate(callback=approve_cb)
        coord = ApprovalCoordinator(gate=gate)

        await coord.submit(_make_proposal_set())
        summary = coord.summary()
        assert summary["total_requests"] == 1
        assert summary["total_approved"] == 1
        assert len(summary["history"]) == 1
        assert summary["history"][0]["status"] == "approved"

    @pytest.mark.asyncio
    async def test_reset(self):
        async def approve_cb(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )

        gate = CallbackApprovalGate(callback=approve_cb)
        coord = ApprovalCoordinator(gate=gate)

        await coord.submit(_make_proposal_set())
        assert coord.total_requests == 1
        coord.reset()
        assert coord.total_requests == 0
        assert coord.history == []

    @pytest.mark.asyncio
    async def test_multiple_submissions_tracked(self):
        call_count = 0

        async def alternate_cb(req: ApprovalRequest) -> ApprovalResponse:
            nonlocal call_count
            call_count += 1
            status = ApprovalStatus.APPROVED if call_count % 2 == 1 else ApprovalStatus.REJECTED
            return ApprovalResponse(
                request_id=req.request_id,
                status=status,
            )

        gate = CallbackApprovalGate(callback=alternate_cb)
        coord = ApprovalCoordinator(gate=gate)

        await coord.submit(_make_proposal_set())
        await coord.submit(_make_proposal_set())
        await coord.submit(_make_proposal_set())

        assert coord.total_requests == 3
        assert coord.total_approved == 2
        assert coord.total_rejected == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestApprovalEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_proposal_set(self):
        gate = CliApprovalGate(
            input_fn=_make_input_fn(["y"]),
            output_fn=lambda _: None,
        )
        request = ApprovalRequest(
            request_id="edge-001",
            proposals=[],
            scope=ApprovalScope.BATCH,
        )
        response = await gate.request_approval(request)
        # With no proposals, batch approve all = approve nothing
        assert response.status == ApprovalStatus.APPROVED
        assert len(response.decisions) == 0

    @pytest.mark.asyncio
    async def test_coordinator_with_empty_set(self):
        async def approve_cb(req: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=req.request_id,
                status=ApprovalStatus.APPROVED,
            )

        gate = CallbackApprovalGate(callback=approve_cb)
        coord = ApprovalCoordinator(gate=gate)

        empty_set = FixProposalSet()
        request = coord.build_request(empty_set)
        assert request.proposal_count == 0

    def test_approval_status_values(self):
        """Ensure all expected statuses exist."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.SKIPPED.value == "skipped"
        assert ApprovalStatus.TIMED_OUT.value == "timed_out"
        assert ApprovalStatus.AUTO_APPROVED.value == "auto_approved"

    def test_approval_scope_values(self):
        assert ApprovalScope.SINGLE.value == "single"
        assert ApprovalScope.BATCH.value == "batch"
        assert ApprovalScope.SESSION.value == "session"

    @pytest.mark.asyncio
    async def test_gate_protocol_compliance(self):
        """Verify all gate implementations satisfy the protocol."""
        from test_runner.autonomy.approval import ApprovalGate

        assert isinstance(CliApprovalGate(), ApprovalGate)
        assert isinstance(
            CallbackApprovalGate(callback=lambda r: None),
            ApprovalGate,
        )
        assert isinstance(AutoApprovalGate(), ApprovalGate)
