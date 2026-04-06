"""Human-in-the-loop approval gate for fix proposals.

The approval gate presents proposed fixes to the user and blocks execution
until explicit approval or rejection is received. This is the core safety
mechanism ensuring that the troubleshooter never auto-executes fixes
without user consent.

Architecture:
- ApprovalRequest wraps a FixProposal (or batch) with presentation metadata
- ApprovalResponse captures the user's decision with optional feedback
- ApprovalGate is an abstract protocol that concrete implementations
  (CLI prompt, messaging platform callback, etc.) must satisfy
- CliApprovalGate is the built-in CLI implementation using stdin/stdout
- The gate is invoked by the orchestrator hub after receiving fix proposals
  from the troubleshooter, ensuring sub-agents never interact directly

The configurable autonomy policy controls whether the gate is engaged:
- In diagnose-only mode (default), the gate always blocks
- In future auto-fix modes, the policy can bypass the gate for
  high-confidence fixes (not implemented yet, architecture ready)
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from test_runner.agents.troubleshooter.models import (
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.autonomy.policy import AutonomyPolicyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ApprovalStatus(str, enum.Enum):
    """Outcome of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"        # User chose to skip (neither approve nor reject)
    TIMED_OUT = "timed_out"    # No response within the timeout window
    AUTO_APPROVED = "auto_approved"  # Policy allowed auto-approval (future)


class ApprovalScope(str, enum.Enum):
    """Scope of an approval decision."""

    SINGLE = "single"          # Applies to one proposal
    BATCH = "batch"            # Applies to all proposals in the set
    SESSION = "session"        # Applies to all proposals in the session


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ApprovalRequest(BaseModel, frozen=True):
    """A request for user approval of one or more fix proposals.

    Created by the orchestrator and passed to the approval gate.
    Contains all context the user needs to make an informed decision.

    Attributes:
        request_id: Unique identifier for this approval request.
        proposals: The fix proposals awaiting approval.
        scope: Whether this is a single, batch, or session-level request.
        context_summary: Human-readable summary of the test run context.
        failure_count: Total number of failures in the run.
        auto_approvable: Whether the policy would allow auto-approval
            (informational; the gate still blocks in diagnose-only mode).
        timeout_seconds: Maximum time to wait for a response (0 = no limit).
        metadata: Additional context for the approval UI.
    """

    request_id: str
    proposals: list[FixProposal] = Field(default_factory=list)
    scope: ApprovalScope = ApprovalScope.BATCH
    context_summary: str = ""
    failure_count: int = 0
    auto_approvable: bool = False
    timeout_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def proposal_count(self) -> int:
        """Number of proposals in this request."""
        return len(self.proposals)

    @property
    def high_confidence_count(self) -> int:
        """Number of proposals with HIGH confidence."""
        return sum(
            1 for p in self.proposals if p.confidence == FixConfidence.HIGH
        )

    def format_summary(self) -> str:
        """Human-readable summary of the approval request."""
        lines = [
            f"Approval Request: {self.request_id}",
            f"  Scope: {self.scope.value}",
            f"  Proposals: {self.proposal_count}",
            f"  High confidence: {self.high_confidence_count}",
            f"  Failures addressed: {self.failure_count}",
        ]
        if self.context_summary:
            lines.append(f"  Context: {self.context_summary}")
        return "\n".join(lines)


class ProposalDecision(BaseModel, frozen=True):
    """User's decision on a single proposal within a batch.

    Attributes:
        proposal_index: Index of the proposal in the request's list.
        failure_id: The failure ID this proposal addresses.
        status: Approved, rejected, or skipped.
        feedback: Optional user feedback or reason for rejection.
    """

    proposal_index: int
    failure_id: str
    status: ApprovalStatus
    feedback: str = ""


class ApprovalResponse(BaseModel, frozen=True):
    """The user's response to an approval request.

    Captures the overall decision plus per-proposal decisions for batch
    requests. The orchestrator uses this to determine which (if any)
    proposals to apply.

    Attributes:
        request_id: Matches the ApprovalRequest.request_id.
        status: Overall approval status.
        decisions: Per-proposal decisions (for batch scope).
        feedback: User's overall feedback or reason.
        responded_at: Timestamp when the response was received.
        response_time_seconds: How long the user took to respond.
        metadata: Additional context from the approval UI.
    """

    request_id: str
    status: ApprovalStatus
    decisions: list[ProposalDecision] = Field(default_factory=list)
    feedback: str = ""
    responded_at: float = Field(default_factory=time.time)
    response_time_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_approved(self) -> bool:
        """True if the overall status is approved or auto-approved."""
        return self.status in (
            ApprovalStatus.APPROVED,
            ApprovalStatus.AUTO_APPROVED,
        )

    @property
    def is_rejected(self) -> bool:
        """True if the overall status is rejected."""
        return self.status == ApprovalStatus.REJECTED

    @property
    def approved_indices(self) -> list[int]:
        """Indices of individually approved proposals."""
        return [
            d.proposal_index
            for d in self.decisions
            if d.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
        ]

    @property
    def rejected_indices(self) -> list[int]:
        """Indices of individually rejected proposals."""
        return [
            d.proposal_index
            for d in self.decisions
            if d.status == ApprovalStatus.REJECTED
        ]


# ---------------------------------------------------------------------------
# Approval gate protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ApprovalGate(Protocol):
    """Protocol for approval gate implementations.

    Any approval gate must implement request_approval as an async method
    that presents proposals to the user and blocks until a response is
    received (or timeout is hit).

    Implementations:
    - CliApprovalGate: interactive CLI prompt (built-in)
    - Future: Teams/Slack bot callback, web UI, etc.
    """

    async def request_approval(
        self, request: ApprovalRequest
    ) -> ApprovalResponse:
        """Present an approval request and wait for user response.

        This method MUST block until the user responds or the timeout
        specified in the request expires. It must never auto-approve
        unless the request explicitly marks auto_approvable=True AND
        the implementation supports it.

        Args:
            request: The approval request with proposals and context.

        Returns:
            An ApprovalResponse with the user's decision.
        """
        ...


# ---------------------------------------------------------------------------
# CLI approval gate implementation
# ---------------------------------------------------------------------------


def _format_proposal_for_cli(index: int, proposal: FixProposal) -> str:
    """Format a single proposal for CLI display."""
    lines: list[str] = []
    lines.append(f"  ── Proposal #{index + 1} ──")
    lines.append(f"  Failure: {proposal.failure_id}")
    lines.append(f"  Title:   {proposal.title}")
    lines.append(f"  Confidence: {proposal.confidence.value.upper()} ({proposal.confidence_score:.0%})")
    lines.append(f"  Category: {proposal.category.value}")

    if proposal.description:
        lines.append(f"  Description: {proposal.description}")

    if proposal.rationale:
        lines.append(f"  Rationale: {proposal.rationale}")

    if proposal.affected_files:
        lines.append(f"  Affected files: {', '.join(proposal.affected_files)}")

    for j, change in enumerate(proposal.proposed_changes):
        lines.append(f"    Change {j + 1}: [{change.change_type}] {change.file_path}")
        lines.append(f"      {change.description}")
        if change.has_diff:
            lines.append(f"      - {change.original_snippet}")
            lines.append(f"      + {change.proposed_snippet}")

    if proposal.requires_user_action:
        lines.append(f"  ⚠ Manual action required: {proposal.user_action_description}")

    if proposal.alternative_fixes:
        lines.append(f"  Alternatives: {'; '.join(proposal.alternative_fixes)}")

    return "\n".join(lines)


class CliApprovalGate:
    """Interactive CLI approval gate using stdin/stdout.

    Presents fix proposals in a readable format and prompts the user
    to approve, reject, or skip each proposal (or all at once for
    batch scope).

    The input_fn parameter allows injection of a custom input function
    for testing (replacing the blocking ``input()`` builtin).

    Args:
        input_fn: Callable that prompts the user and returns their input.
            Defaults to the builtin ``input``. For async contexts, this
            is called via ``asyncio.to_thread``.
        output_fn: Callable for printing output. Defaults to ``print``.
    """

    def __init__(
        self,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._input_fn = input_fn or input
        self._output_fn = output_fn or print

    def _display(self, text: str) -> None:
        self._output_fn(text)

    def _prompt(self, prompt: str) -> str:
        return self._input_fn(prompt)

    async def _async_prompt(self, prompt: str) -> str:
        """Run the blocking input function in a thread."""
        return await asyncio.to_thread(self._prompt, prompt)

    async def request_approval(
        self, request: ApprovalRequest
    ) -> ApprovalResponse:
        """Present proposals via CLI and collect user decisions.

        For batch scope, prompts once for all proposals.
        For single scope, prompts for each proposal individually.

        Returns an ApprovalResponse with the user's decisions.
        """
        start_time = time.monotonic()

        self._display("\n" + "=" * 60)
        self._display("  FIX PROPOSAL APPROVAL")
        self._display("=" * 60)
        self._display(request.format_summary())
        self._display("")

        # Display all proposals
        for i, proposal in enumerate(request.proposals):
            self._display(_format_proposal_for_cli(i, proposal))
            self._display("")

        self._display("-" * 60)

        if request.scope == ApprovalScope.SINGLE and len(request.proposals) == 1:
            # Single proposal approval
            response = await self._approve_single(request, start_time)
        else:
            # Batch approval
            response = await self._approve_batch(request, start_time)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Approval response: status=%s time=%.1fs",
            response.status.value,
            elapsed,
        )
        return response

    async def _approve_single(
        self, request: ApprovalRequest, start_time: float
    ) -> ApprovalResponse:
        """Handle single-proposal approval."""
        prompt_text = "Apply this fix? [y]es / [n]o / [s]kip: "

        try:
            answer = (await self._async_prompt(prompt_text)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        elapsed = time.monotonic() - start_time
        proposal = request.proposals[0]
        status, decision_status = self._parse_answer(answer)

        decision = ProposalDecision(
            proposal_index=0,
            failure_id=proposal.failure_id,
            status=decision_status,
        )

        feedback = ""
        if status == ApprovalStatus.REJECTED:
            try:
                feedback = (await self._async_prompt("Reason (optional): ")).strip()
            except (EOFError, KeyboardInterrupt):
                feedback = ""

        return ApprovalResponse(
            request_id=request.request_id,
            status=status,
            decisions=[decision],
            feedback=feedback,
            response_time_seconds=elapsed,
        )

    async def _approve_batch(
        self, request: ApprovalRequest, start_time: float
    ) -> ApprovalResponse:
        """Handle batch approval — approve/reject all or individually."""
        prompt_text = (
            "Apply all fixes? [y]es all / [n]o all / [i]ndividual review: "
        )

        try:
            answer = (await self._async_prompt(prompt_text)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer.startswith("i"):
            return await self._approve_individually(request, start_time)

        elapsed = time.monotonic() - start_time
        status, decision_status = self._parse_answer(answer)

        decisions = [
            ProposalDecision(
                proposal_index=i,
                failure_id=p.failure_id,
                status=decision_status,
            )
            for i, p in enumerate(request.proposals)
        ]

        feedback = ""
        if status == ApprovalStatus.REJECTED:
            try:
                feedback = (await self._async_prompt("Reason (optional): ")).strip()
            except (EOFError, KeyboardInterrupt):
                feedback = ""

        return ApprovalResponse(
            request_id=request.request_id,
            status=status,
            decisions=decisions,
            feedback=feedback,
            response_time_seconds=elapsed,
        )

    async def _approve_individually(
        self, request: ApprovalRequest, start_time: float
    ) -> ApprovalResponse:
        """Handle per-proposal approval within a batch."""
        decisions: list[ProposalDecision] = []
        approved_count = 0

        for i, proposal in enumerate(request.proposals):
            prompt_text = (
                f"Proposal #{i + 1}/{len(request.proposals)} "
                f"({proposal.confidence.value.upper()}): {proposal.title}\n"
                f"  Apply? [y]es / [n]o / [s]kip: "
            )
            try:
                answer = (await self._async_prompt(prompt_text)).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"

            _, decision_status = self._parse_answer(answer)
            if decision_status == ApprovalStatus.APPROVED:
                approved_count += 1

            decisions.append(
                ProposalDecision(
                    proposal_index=i,
                    failure_id=proposal.failure_id,
                    status=decision_status,
                )
            )

        elapsed = time.monotonic() - start_time

        # Overall status: approved if at least one approved, rejected if none
        if approved_count > 0:
            overall = ApprovalStatus.APPROVED
        elif all(d.status == ApprovalStatus.SKIPPED for d in decisions):
            overall = ApprovalStatus.SKIPPED
        else:
            overall = ApprovalStatus.REJECTED

        return ApprovalResponse(
            request_id=request.request_id,
            status=overall,
            decisions=decisions,
            response_time_seconds=elapsed,
        )

    @staticmethod
    def _parse_answer(answer: str) -> tuple[ApprovalStatus, ApprovalStatus]:
        """Parse a y/n/s answer into (overall_status, decision_status)."""
        if answer.startswith("y"):
            return ApprovalStatus.APPROVED, ApprovalStatus.APPROVED
        elif answer.startswith("s"):
            return ApprovalStatus.SKIPPED, ApprovalStatus.SKIPPED
        else:
            return ApprovalStatus.REJECTED, ApprovalStatus.REJECTED


# ---------------------------------------------------------------------------
# Programmatic / callback approval gate
# ---------------------------------------------------------------------------


class CallbackApprovalGate:
    """Approval gate that delegates to an async callback.

    Useful for messaging platform integrations (Teams, Slack) or
    web UIs where the approval happens out-of-band and is delivered
    via a callback/webhook.

    Args:
        callback: Async function that receives an ApprovalRequest and
            returns an ApprovalResponse. The callback is responsible
            for presenting the request and collecting the response.
        timeout_seconds: Default timeout if the request doesn't specify one.
    """

    def __init__(
        self,
        callback: Callable[[ApprovalRequest], Any],
        timeout_seconds: float = 300.0,
    ) -> None:
        self._callback = callback
        self._default_timeout = timeout_seconds

    async def request_approval(
        self, request: ApprovalRequest
    ) -> ApprovalResponse:
        """Delegate to the callback, with timeout handling."""
        timeout = request.timeout_seconds or self._default_timeout

        try:
            result = self._callback(request)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                response = await asyncio.wait_for(result, timeout=timeout)
            else:
                response = result
        except asyncio.TimeoutError:
            logger.warning(
                "Approval timed out after %.1fs for request %s",
                timeout,
                request.request_id,
            )
            return ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.TIMED_OUT,
                feedback=f"No response within {timeout}s",
            )
        except Exception as exc:
            logger.error(
                "Approval callback failed for request %s: %s",
                request.request_id,
                exc,
            )
            return ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.REJECTED,
                feedback=f"Callback error: {exc}",
            )

        return response


# ---------------------------------------------------------------------------
# Auto-approval gate (for future auto-fix modes)
# ---------------------------------------------------------------------------


class AutoApprovalGate:
    """Approval gate that auto-approves based on confidence thresholds.

    Only used when the autonomy policy explicitly enables auto-fix mode.
    Proposals below the confidence threshold are still routed to the
    user via the fallback gate.

    Args:
        min_confidence: Minimum confidence score for auto-approval.
        allowed_confidences: Set of FixConfidence levels that can be
            auto-approved. Defaults to HIGH only.
        fallback_gate: Gate to use for proposals that don't meet the
            auto-approval criteria. If None, those proposals are rejected.
    """

    def __init__(
        self,
        min_confidence: float = 0.85,
        allowed_confidences: frozenset[FixConfidence] | None = None,
        fallback_gate: ApprovalGate | None = None,
    ) -> None:
        self._min_confidence = min_confidence
        self._allowed_confidences = allowed_confidences or frozenset(
            {FixConfidence.HIGH}
        )
        self._fallback_gate = fallback_gate

    async def request_approval(
        self, request: ApprovalRequest
    ) -> ApprovalResponse:
        """Auto-approve qualifying proposals, delegate the rest."""
        auto_decisions: list[ProposalDecision] = []
        needs_manual: list[tuple[int, FixProposal]] = []

        for i, proposal in enumerate(request.proposals):
            if (
                proposal.confidence in self._allowed_confidences
                and proposal.confidence_score >= self._min_confidence
            ):
                auto_decisions.append(
                    ProposalDecision(
                        proposal_index=i,
                        failure_id=proposal.failure_id,
                        status=ApprovalStatus.AUTO_APPROVED,
                    )
                )
            else:
                needs_manual.append((i, proposal))

        if not needs_manual:
            # All auto-approved
            return ApprovalResponse(
                request_id=request.request_id,
                status=ApprovalStatus.AUTO_APPROVED,
                decisions=auto_decisions,
            )

        if self._fallback_gate is not None:
            # Route remaining to fallback
            manual_request = ApprovalRequest(
                request_id=f"{request.request_id}-manual",
                proposals=[p for _, p in needs_manual],
                scope=request.scope,
                context_summary=request.context_summary,
                failure_count=request.failure_count,
                timeout_seconds=request.timeout_seconds,
                metadata=request.metadata,
            )
            manual_response = await self._fallback_gate.request_approval(
                manual_request
            )
            # Merge decisions
            for j, (orig_idx, _) in enumerate(needs_manual):
                if j < len(manual_response.decisions):
                    d = manual_response.decisions[j]
                    auto_decisions.append(
                        ProposalDecision(
                            proposal_index=orig_idx,
                            failure_id=d.failure_id,
                            status=d.status,
                            feedback=d.feedback,
                        )
                    )
                else:
                    auto_decisions.append(
                        ProposalDecision(
                            proposal_index=orig_idx,
                            failure_id=needs_manual[j][1].failure_id,
                            status=ApprovalStatus.REJECTED,
                        )
                    )
        else:
            # No fallback — reject non-qualifying proposals
            for orig_idx, proposal in needs_manual:
                auto_decisions.append(
                    ProposalDecision(
                        proposal_index=orig_idx,
                        failure_id=proposal.failure_id,
                        status=ApprovalStatus.REJECTED,
                        feedback="Below auto-approval threshold",
                    )
                )

        # Sort decisions by index for consistency
        auto_decisions.sort(key=lambda d: d.proposal_index)

        # Overall status
        has_approved = any(
            d.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
            for d in auto_decisions
        )
        overall = ApprovalStatus.APPROVED if has_approved else ApprovalStatus.REJECTED

        return ApprovalResponse(
            request_id=request.request_id,
            status=overall,
            decisions=auto_decisions,
        )


# ---------------------------------------------------------------------------
# Approval coordinator — used by the orchestrator
# ---------------------------------------------------------------------------


class ApprovalCoordinator:
    """Coordinates the approval workflow between orchestrator and gate.

    The orchestrator creates an ApprovalCoordinator with the configured
    gate and uses it to manage the approval lifecycle:
    1. Build an ApprovalRequest from fix proposals
    2. Submit to the gate (blocks until user responds)
    3. Process the response to determine which fixes to apply
    4. Record the approval history

    This class owns no LLM state and is purely a workflow coordinator.

    Args:
        gate: The approval gate implementation to use.
        policy: Autonomy policy for determining auto-approvability.
        auto_fix_enabled: Whether auto-fix mode is active.
    """

    def __init__(
        self,
        gate: ApprovalGate,
        policy: AutonomyPolicyConfig | None = None,
        auto_fix_enabled: bool = False,
    ) -> None:
        self._gate = gate
        self._policy = policy or AutonomyPolicyConfig()
        self._auto_fix_enabled = auto_fix_enabled
        self._history: list[tuple[ApprovalRequest, ApprovalResponse]] = []
        self._request_counter = 0

    @property
    def gate(self) -> ApprovalGate:
        """The approval gate implementation."""
        return self._gate

    @property
    def history(self) -> list[tuple[ApprovalRequest, ApprovalResponse]]:
        """History of all approval request/response pairs."""
        return list(self._history)

    @property
    def total_requests(self) -> int:
        """Total number of approval requests made."""
        return len(self._history)

    @property
    def total_approved(self) -> int:
        """Total number of approved requests."""
        return sum(1 for _, r in self._history if r.is_approved)

    @property
    def total_rejected(self) -> int:
        """Total number of rejected requests."""
        return sum(1 for _, r in self._history if r.is_rejected)

    def build_request(
        self,
        proposal_set: FixProposalSet,
        *,
        scope: ApprovalScope = ApprovalScope.BATCH,
        context_summary: str = "",
        timeout_seconds: float = 0.0,
    ) -> ApprovalRequest:
        """Build an ApprovalRequest from a FixProposalSet.

        Args:
            proposal_set: The fix proposals from the troubleshooter.
            scope: Approval scope (single, batch, or session).
            context_summary: Summary of the test run context.
            timeout_seconds: Timeout for the approval (0 = no limit).

        Returns:
            An ApprovalRequest ready to submit to the gate.
        """
        self._request_counter += 1
        request_id = f"approval-{self._request_counter:04d}"

        # Determine auto-approvability based on policy
        auto_approvable = (
            self._auto_fix_enabled
            and proposal_set.high_confidence_count == len(proposal_set.proposals)
            and all(
                p.confidence_score >= self._policy.execute_threshold
                for p in proposal_set.proposals
            )
        )

        return ApprovalRequest(
            request_id=request_id,
            proposals=list(proposal_set.by_confidence()),
            scope=scope,
            context_summary=context_summary,
            failure_count=proposal_set.total_failures_analyzed,
            auto_approvable=auto_approvable,
            timeout_seconds=timeout_seconds,
        )

    async def submit(
        self,
        proposal_set: FixProposalSet,
        *,
        scope: ApprovalScope = ApprovalScope.BATCH,
        context_summary: str = "",
        timeout_seconds: float = 0.0,
    ) -> ApprovalResponse:
        """Build and submit an approval request, blocking until response.

        This is the main entry point for the orchestrator. It:
        1. Builds the request
        2. Submits to the gate (blocks)
        3. Records the result in history
        4. Returns the response

        Args:
            proposal_set: Fix proposals from the troubleshooter.
            scope: Approval scope.
            context_summary: Test run context summary.
            timeout_seconds: Timeout for the approval.

        Returns:
            The user's ApprovalResponse.
        """
        request = self.build_request(
            proposal_set,
            scope=scope,
            context_summary=context_summary,
            timeout_seconds=timeout_seconds,
        )

        logger.info(
            "Submitting approval request %s: %d proposals, scope=%s",
            request.request_id,
            request.proposal_count,
            request.scope.value,
        )

        response = await self._gate.request_approval(request)
        self._history.append((request, response))

        logger.info(
            "Approval response for %s: status=%s (approved=%s)",
            request.request_id,
            response.status.value,
            response.is_approved,
        )

        return response

    def get_approved_proposals(
        self, request: ApprovalRequest, response: ApprovalResponse
    ) -> list[FixProposal]:
        """Extract the proposals that were approved.

        Args:
            request: The original approval request.
            response: The user's response.

        Returns:
            List of approved FixProposal objects.
        """
        if response.is_approved and not response.decisions:
            # Blanket approval — all proposals approved
            return list(request.proposals)

        approved_indices = set(response.approved_indices)
        return [
            request.proposals[i]
            for i in approved_indices
            if i < len(request.proposals)
        ]

    def get_rejected_proposals(
        self, request: ApprovalRequest, response: ApprovalResponse
    ) -> list[FixProposal]:
        """Extract the proposals that were rejected.

        Args:
            request: The original approval request.
            response: The user's response.

        Returns:
            List of rejected FixProposal objects.
        """
        if response.is_rejected and not response.decisions:
            # Blanket rejection — all proposals rejected
            return list(request.proposals)

        rejected_indices = set(response.rejected_indices)
        return [
            request.proposals[i]
            for i in rejected_indices
            if i < len(request.proposals)
        ]

    def summary(self) -> dict[str, Any]:
        """Serializable summary of the approval history."""
        return {
            "total_requests": self.total_requests,
            "total_approved": self.total_approved,
            "total_rejected": self.total_rejected,
            "auto_fix_enabled": self._auto_fix_enabled,
            "history": [
                {
                    "request_id": req.request_id,
                    "proposal_count": req.proposal_count,
                    "scope": req.scope.value,
                    "status": resp.status.value,
                    "response_time_seconds": round(
                        resp.response_time_seconds, 2
                    ),
                }
                for req, resp in self._history
            ],
        }

    def reset(self) -> None:
        """Clear approval history."""
        self._history.clear()
        self._request_counter = 0
