"""Diagnostic step-counting guard for the troubleshooter agent.

Tracks diagnostic iterations (higher-level than individual tool calls) and
enforces a configurable max limit (~10 steps by default). When the limit is
reached or the diagnosis completes, the guard produces a structured
``DiagnosisSummary`` that the orchestrator can consume.

Each "diagnostic step" represents one logical investigation action — e.g.
reading a failure log, inspecting a source file, checking the environment.
A single step may involve one or more tool calls under the hood, but the
guard tracks the *diagnostic intent*, not the raw tool invocations.

Design decisions:
- Pydantic models for DiagnosisSummary for validation + serialization
- Guard is a standalone component (not coupled to StepCounter) because
  it tracks a different granularity — diagnostic iterations vs tool calls
- Configurable limit with a sensible default of 10
- Immutable summary output for safe cross-agent sharing
- Completion reason is explicit (limit_reached, completed, error)
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, Field, computed_field

logger = logging.getLogger(__name__)

# Default maximum diagnostic iterations before forced summary.
DEFAULT_MAX_DIAGNOSTIC_STEPS: int = 10

# Warning threshold — fraction of budget consumed before warning.
_DIAGNOSTIC_WARNING_FRACTION: float = 0.70  # warn at 70%


class CompletionReason(str, Enum):
    """Why the diagnostic session ended."""

    COMPLETED = "completed"           # Agent finished diagnosis naturally
    LIMIT_REACHED = "limit_reached"   # Hit the max step limit
    ERROR = "error"                   # An error forced early termination
    MANUAL_STOP = "manual_stop"       # External stop (e.g. user cancellation)


class DiagnosticStep(BaseModel, frozen=True):
    """Record of a single diagnostic iteration.

    Attributes:
        step_number: 1-based index of this step.
        action: What the agent did (e.g. "read_failure_log", "inspect_source").
        target: What was investigated (e.g. file path, test ID).
        finding: What was discovered in this step.
        confidence_delta: Change in diagnostic confidence after this step.
            Positive means we're more certain, negative means less.
        timestamp: Unix timestamp when the step was recorded.
        metadata: Additional context for this step.
    """

    step_number: int
    action: str
    target: str = ""
    finding: str = ""
    confidence_delta: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagnosisSummary(BaseModel, frozen=True):
    """Structured summary produced when diagnosis completes or limit is reached.

    This is the canonical output of the diagnostic step guard, consumed
    by the orchestrator hub and passed to the reporter for output.

    Attributes:
        completion_reason: Why the diagnosis ended.
        total_steps: Number of diagnostic steps taken.
        max_steps: The configured step limit.
        steps: Ordered list of diagnostic step records.
        root_cause: Identified root cause (may be empty if inconclusive).
        confidence: Overall diagnostic confidence in [0.0, 1.0].
        evidence: Supporting evidence collected during diagnosis.
        proposed_fixes: Ordered list of fix suggestions for the user.
        alternative_causes: Other possible explanations.
        unresolved_questions: What could not be determined within budget.
        start_time: Unix timestamp when diagnosis started.
        end_time: Unix timestamp when diagnosis ended.
        metadata: Extensible metadata.
    """

    completion_reason: CompletionReason
    total_steps: int
    max_steps: int
    steps: list[DiagnosticStep] = Field(default_factory=list)
    root_cause: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    proposed_fixes: list[str] = Field(default_factory=list)
    alternative_causes: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the diagnostic session."""
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 3)
        return 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def budget_used_fraction(self) -> float:
        """Fraction of the step budget consumed."""
        if self.max_steps <= 0:
            return 1.0
        return round(min(self.total_steps / self.max_steps, 1.0), 4)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_conclusive(self) -> bool:
        """True if a root cause was identified with reasonable confidence."""
        return bool(self.root_cause) and self.confidence >= 0.5

    @computed_field  # type: ignore[prop-decorator]
    @property
    def was_truncated(self) -> bool:
        """True if the diagnosis was cut short by the step limit."""
        return self.completion_reason == CompletionReason.LIMIT_REACHED

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        status = "CONCLUSIVE" if self.is_conclusive else "INCONCLUSIVE"
        reason = self.completion_reason.value.replace("_", " ")
        return (
            f"[{status}] {self.total_steps}/{self.max_steps} steps, "
            f"confidence={self.confidence:.0%}, reason={reason}"
        )

    def to_report_dict(self) -> dict[str, Any]:
        """Serializable dict optimized for reporting channels."""
        return {
            "completion_reason": self.completion_reason.value,
            "total_steps": self.total_steps,
            "max_steps": self.max_steps,
            "budget_used_fraction": self.budget_used_fraction,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "is_conclusive": self.is_conclusive,
            "was_truncated": self.was_truncated,
            "evidence": self.evidence,
            "proposed_fixes": self.proposed_fixes,
            "alternative_causes": self.alternative_causes,
            "unresolved_questions": self.unresolved_questions,
            "duration_seconds": self.duration_seconds,
            "steps": [
                {
                    "step_number": s.step_number,
                    "action": s.action,
                    "target": s.target,
                    "finding": s.finding,
                    "confidence_delta": s.confidence_delta,
                }
                for s in self.steps
            ],
        }


class DiagnosticStepGuard:
    """Guards diagnostic iterations and enforces a step limit.

    The guard tracks each diagnostic iteration the troubleshooter performs,
    enforces a configurable maximum (default ~10), and produces a structured
    ``DiagnosisSummary`` when the session ends — either naturally or because
    the limit was reached.

    Usage::

        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()

        # Each diagnostic iteration:
        if guard.can_proceed():
            guard.record_step("read_failure_log", target="test_foo.py",
                              finding="ImportError on line 42")
        else:
            summary = guard.finalize(reason=CompletionReason.LIMIT_REACHED)

        # When done:
        summary = guard.finalize(
            root_cause="Missing dependency 'requests'",
            confidence=0.85,
            proposed_fixes=["pip install requests"],
        )

    Thread-safety: NOT thread-safe. Designed for single-agent use within
    one diagnostic session.
    """

    def __init__(self, max_steps: int = DEFAULT_MAX_DIAGNOSTIC_STEPS) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self._max_steps = max_steps
        self._steps: list[DiagnosticStep] = []
        self._start_time: float = 0.0
        self._running_confidence: float = 0.0
        self._evidence: list[str] = []
        self._started = False
        self._finalized = False

    # -- Properties -----------------------------------------------------------

    @property
    def max_steps(self) -> int:
        """Maximum diagnostic iterations allowed."""
        return self._max_steps

    @property
    def steps_taken(self) -> int:
        """Number of diagnostic steps completed so far."""
        return len(self._steps)

    @property
    def remaining(self) -> int:
        """Diagnostic steps remaining before the limit."""
        return max(0, self._max_steps - self.steps_taken)

    @property
    def is_at_limit(self) -> bool:
        """True when the step limit has been reached."""
        return self.steps_taken >= self._max_steps

    @property
    def is_warning(self) -> bool:
        """True when the warning threshold has been crossed."""
        if self._max_steps <= 0:
            return True
        return self.steps_taken >= int(self._max_steps * _DIAGNOSTIC_WARNING_FRACTION)

    @property
    def usage_fraction(self) -> float:
        """Fraction of the budget consumed, in [0.0, 1.0]."""
        if self._max_steps <= 0:
            return 1.0
        return min(self.steps_taken / self._max_steps, 1.0)

    @property
    def current_confidence(self) -> float:
        """Running diagnostic confidence."""
        return self._running_confidence

    @property
    def steps(self) -> list[DiagnosticStep]:
        """Read-only view of recorded steps."""
        return list(self._steps)

    @property
    def is_started(self) -> bool:
        """True if the guard has been started."""
        return self._started

    @property
    def is_finalized(self) -> bool:
        """True if the guard has produced a final summary."""
        return self._finalized

    # -- Lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Begin a diagnostic session. Resets any prior state."""
        self._steps.clear()
        self._start_time = time.time()
        self._running_confidence = 0.0
        self._evidence.clear()
        self._started = True
        self._finalized = False
        logger.debug(
            "Diagnostic step guard started (max_steps=%d)", self._max_steps,
        )

    def can_proceed(self) -> bool:
        """Check if another diagnostic step is allowed.

        Returns False if:
        - The guard hasn't been started
        - The guard has been finalized
        - The step limit has been reached
        """
        if not self._started or self._finalized:
            return False
        return not self.is_at_limit

    def record_step(
        self,
        action: str,
        *,
        target: str = "",
        finding: str = "",
        confidence_delta: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Record a diagnostic iteration.

        Args:
            action: What the agent did (e.g. "read_failure_log").
            target: What was investigated (e.g. file path).
            finding: What was discovered.
            confidence_delta: Change in confidence from this step.
            metadata: Additional context.

        Returns:
            True if the step was recorded. False if the limit has been
            reached and no more steps are allowed.

        Raises:
            RuntimeError: If the guard hasn't been started or is finalized.
        """
        if not self._started:
            raise RuntimeError("DiagnosticStepGuard.start() must be called first")
        if self._finalized:
            raise RuntimeError("DiagnosticStepGuard has been finalized")

        if self.is_at_limit:
            logger.warning(
                "Diagnostic step limit reached (%d/%d). Rejecting step: %s",
                self.steps_taken, self._max_steps, action,
            )
            return False

        step = DiagnosticStep(
            step_number=self.steps_taken + 1,
            action=action,
            target=target,
            finding=finding,
            confidence_delta=confidence_delta,
            metadata=metadata or {},
        )
        self._steps.append(step)

        # Update running confidence (clamped to [0, 1])
        self._running_confidence = max(
            0.0, min(1.0, self._running_confidence + confidence_delta),
        )

        # Accumulate evidence from findings
        if finding:
            self._evidence.append(finding)

        if self.is_at_limit:
            logger.info(
                "Diagnostic step limit reached (%d/%d) after step: %s",
                self.steps_taken, self._max_steps, action,
            )
        elif self.is_warning:
            logger.info(
                "Diagnostic budget warning: %d/%d steps used (%d remaining)",
                self.steps_taken, self._max_steps, self.remaining,
            )

        return True

    def finalize(
        self,
        *,
        reason: CompletionReason | None = None,
        root_cause: str = "",
        confidence: float | None = None,
        proposed_fixes: list[str] | None = None,
        alternative_causes: list[str] | None = None,
        unresolved_questions: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DiagnosisSummary:
        """Produce the final diagnosis summary.

        Call this when the diagnosis completes or the step limit is reached.
        After finalization, no more steps can be recorded.

        Args:
            reason: Why the session ended. If None, auto-detected based on
                whether the limit was reached.
            root_cause: The identified root cause (empty if inconclusive).
            confidence: Final confidence. Defaults to running confidence.
            proposed_fixes: Ordered list of fix suggestions.
            alternative_causes: Other possible explanations.
            unresolved_questions: What could not be determined.
            metadata: Additional context.

        Returns:
            A frozen DiagnosisSummary with all collected data.

        Raises:
            RuntimeError: If the guard hasn't been started.
            RuntimeError: If already finalized.
        """
        if not self._started:
            raise RuntimeError("DiagnosticStepGuard.start() must be called first")
        if self._finalized:
            raise RuntimeError("DiagnosticStepGuard has already been finalized")

        self._finalized = True
        end_time = time.time()

        # Auto-detect completion reason
        if reason is None:
            if self.is_at_limit:
                reason = CompletionReason.LIMIT_REACHED
            else:
                reason = CompletionReason.COMPLETED

        final_confidence = confidence if confidence is not None else self._running_confidence

        summary = DiagnosisSummary(
            completion_reason=reason,
            total_steps=self.steps_taken,
            max_steps=self._max_steps,
            steps=list(self._steps),
            root_cause=root_cause,
            confidence=final_confidence,
            evidence=list(self._evidence),
            proposed_fixes=proposed_fixes or [],
            alternative_causes=alternative_causes or [],
            unresolved_questions=unresolved_questions or [],
            start_time=self._start_time,
            end_time=end_time,
            metadata=metadata or {},
        )

        logger.info(
            "Diagnostic session finalized: %s", summary.summary_line(),
        )

        return summary

    def budget_status_message(self) -> str:
        """Human-readable budget status for agent context injection."""
        if self.is_at_limit:
            return (
                f"[DIAGNOSTIC LIMIT REACHED] You have used all {self._max_steps} "
                f"diagnostic steps. Summarize your findings now."
            )
        if self.is_warning:
            return (
                f"[DIAGNOSTIC WARNING] {self.remaining} of {self._max_steps} "
                f"diagnostic steps remaining. Begin wrapping up."
            )
        return (
            f"[DIAGNOSTIC OK] {self.steps_taken}/{self._max_steps} steps used, "
            f"{self.remaining} remaining."
        )

    def reset(self) -> None:
        """Reset for a new diagnostic session."""
        self._steps.clear()
        self._start_time = 0.0
        self._running_confidence = 0.0
        self._evidence.clear()
        self._started = False
        self._finalized = False
