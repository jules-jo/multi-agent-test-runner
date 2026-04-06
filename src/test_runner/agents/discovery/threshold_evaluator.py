"""Confidence threshold evaluator for step-cap escalation.

Integrates the step counter with confidence scoring to implement the
core escalation logic: when the step cap is reached and confidence
remains below 60%, the evaluator triggers escalation to the
orchestrator or troubleshooter agent.

The evaluator is invoked by the discovery agent after each step to
determine if exploration should continue, and by the orchestrator
when the discovery agent completes (or is force-stopped).

Escalation targets:
- **orchestrator**: default target when confidence is low and the
  agent has no specific failure hypothesis to investigate.
- **troubleshooter**: target when the agent suspects a structural
  problem (e.g. missing deps, broken configs) that a troubleshooter
  could diagnose.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from test_runner.agents.discovery.step_counter import StepCounter
from test_runner.models.confidence import (
    ConfidenceModel,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceTier,
    WARN_THRESHOLD,
)

logger = logging.getLogger(__name__)

# The threshold below which escalation is triggered at step cap.
# This is the canonical 60% boundary from the spec.
ESCALATION_CONFIDENCE_THRESHOLD: float = 0.60


class EscalationTarget(str, enum.Enum):
    """Where to route an escalation request."""

    ORCHESTRATOR = "orchestrator"
    TROUBLESHOOTER = "troubleshooter"


class EscalationReason(str, enum.Enum):
    """Why the evaluator decided to escalate."""

    LOW_CONFIDENCE_AT_CAP = "low_confidence_at_step_cap"
    BUDGET_EXHAUSTED_NO_FINDINGS = "budget_exhausted_no_findings"
    STRUCTURAL_ISSUE_DETECTED = "structural_issue_detected"
    NO_SIGNALS_COLLECTED = "no_signals_collected"


@dataclass(frozen=True)
class EscalationResult:
    """Describes an escalation triggered by the threshold evaluator.

    Attributes:
        should_escalate: True if escalation is needed.
        target: Which agent should handle the escalation.
        reason: Why escalation was triggered.
        confidence_score: The confidence score at escalation time.
        confidence_tier: The tier classification at escalation time.
        steps_taken: Number of steps consumed when escalation triggered.
        step_cap: The hard cap that was configured.
        message: Human-readable summary for logging/UI.
        metadata: Additional context for the receiving agent.
    """

    should_escalate: bool
    target: EscalationTarget
    reason: EscalationReason
    confidence_score: float
    confidence_tier: ConfidenceTier
    steps_taken: int
    step_cap: int
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Serializable representation for logging / handoff."""
        return {
            "should_escalate": self.should_escalate,
            "target": self.target.value,
            "reason": self.reason.value,
            "confidence_score": round(self.confidence_score, 4),
            "confidence_tier": self.confidence_tier.value,
            "steps_taken": self.steps_taken,
            "step_cap": self.step_cap,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ThresholdCheckResult:
    """Result of a threshold check (may or may not trigger escalation).

    Attributes:
        needs_escalation: True if escalation is warranted.
        escalation: Populated when needs_escalation is True.
        confidence_result: The confidence evaluation used for the check.
        budget_remaining: Steps remaining in the budget.
        can_continue: True if the agent can continue exploring.
    """

    needs_escalation: bool
    escalation: EscalationResult | None
    confidence_result: ConfidenceResult
    budget_remaining: int
    can_continue: bool

    def summary(self) -> dict[str, Any]:
        """Serializable representation."""
        result: dict[str, Any] = {
            "needs_escalation": self.needs_escalation,
            "can_continue": self.can_continue,
            "budget_remaining": self.budget_remaining,
            "confidence": self.confidence_result.summary(),
        }
        if self.escalation is not None:
            result["escalation"] = self.escalation.summary()
        return result


class ConfidenceThresholdEvaluator:
    """Evaluates confidence against thresholds with step-cap awareness.

    The evaluator checks two conditions for escalation:
    1. The step cap has been reached (budget exhausted).
    2. Confidence remains below the escalation threshold (default 60%).

    When both conditions are met, it produces an EscalationResult
    routing the issue to the orchestrator or troubleshooter.

    The evaluator also supports mid-exploration checks: if the budget
    is nearing exhaustion and confidence is trending low, it can
    recommend early wrap-up.

    Args:
        step_counter: The step counter tracking exploration budget.
        confidence_model: The model used to aggregate signals.
        escalation_threshold: Confidence below which escalation triggers
            at step cap (default 0.60).
        structural_issue_indicators: Signal names that, when present with
            low scores, suggest a structural problem warranting the
            troubleshooter rather than the orchestrator.
    """

    def __init__(
        self,
        step_counter: StepCounter,
        confidence_model: ConfidenceModel | None = None,
        escalation_threshold: float = ESCALATION_CONFIDENCE_THRESHOLD,
        structural_issue_indicators: Sequence[str] | None = None,
    ) -> None:
        if not (0.0 <= escalation_threshold <= 1.0):
            raise ValueError(
                f"escalation_threshold must be in [0, 1], "
                f"got {escalation_threshold}"
            )
        self._step_counter = step_counter
        self._confidence_model = confidence_model or ConfidenceModel()
        self._escalation_threshold = escalation_threshold
        self._structural_indicators = set(
            structural_issue_indicators or [
                "pytest_in_pyproject",
                "jest_in_package_json",
                "go_module_detected",
                "cargo_dev_deps",
                "npm_test_script",
            ]
        )

    @property
    def escalation_threshold(self) -> float:
        """The confidence threshold for escalation at step cap."""
        return self._escalation_threshold

    @property
    def step_counter(self) -> StepCounter:
        """The step counter being monitored."""
        return self._step_counter

    @property
    def confidence_model(self) -> ConfidenceModel:
        """The confidence model used for evaluation."""
        return self._confidence_model

    def evaluate(
        self,
        signals: Sequence[ConfidenceSignal],
    ) -> ThresholdCheckResult:
        """Evaluate current signals against thresholds and step budget.

        This is the main entry point. Call it after each step or when
        the step budget is exhausted to determine whether to escalate.

        Args:
            signals: Current confidence signals from collectors.

        Returns:
            A ThresholdCheckResult with escalation info if needed.
        """
        confidence_result = self._confidence_model.evaluate(signals)
        budget_remaining = self._step_counter.remaining
        is_exhausted = self._step_counter.is_exhausted

        logger.debug(
            "Threshold check: score=%.4f, budget=%d/%d, exhausted=%s",
            confidence_result.score,
            self._step_counter.steps_taken,
            self._step_counter.hard_cap,
            is_exhausted,
        )

        # Primary escalation condition: step cap reached + low confidence
        if is_exhausted and confidence_result.score < self._escalation_threshold:
            escalation = self._build_escalation(
                confidence_result=confidence_result,
                signals=signals,
            )
            return ThresholdCheckResult(
                needs_escalation=True,
                escalation=escalation,
                confidence_result=confidence_result,
                budget_remaining=0,
                can_continue=False,
            )

        # Budget exhausted but confidence is acceptable — no escalation
        if is_exhausted:
            return ThresholdCheckResult(
                needs_escalation=False,
                escalation=None,
                confidence_result=confidence_result,
                budget_remaining=0,
                can_continue=False,
            )

        # Budget available — can continue
        return ThresholdCheckResult(
            needs_escalation=False,
            escalation=None,
            confidence_result=confidence_result,
            budget_remaining=budget_remaining,
            can_continue=True,
        )

    def check_at_step_cap(
        self,
        signals: Sequence[ConfidenceSignal],
    ) -> EscalationResult | None:
        """Convenience: check specifically for step-cap escalation.

        Returns an EscalationResult if escalation is needed, or None
        if the step cap hasn't been reached or confidence is sufficient.

        This is a simpler API for callers that only care about the
        step-cap boundary condition.
        """
        if not self._step_counter.is_exhausted:
            return None

        confidence_result = self._confidence_model.evaluate(signals)
        if confidence_result.score >= self._escalation_threshold:
            return None

        return self._build_escalation(
            confidence_result=confidence_result,
            signals=signals,
        )

    def _build_escalation(
        self,
        confidence_result: ConfidenceResult,
        signals: Sequence[ConfidenceSignal],
    ) -> EscalationResult:
        """Build an EscalationResult with appropriate target and reason."""
        target = self._determine_target(signals, confidence_result)
        reason = self._determine_reason(signals, confidence_result)
        message = self._build_message(confidence_result, target, reason)

        metadata: dict[str, Any] = {
            "step_summary": self._step_counter.summary(),
            "confidence_summary": confidence_result.summary(),
            "signal_count": len(signals),
            "positive_signal_count": sum(
                1 for s in signals if s.score > 0
            ),
        }

        # Include structural issue details when routing to troubleshooter
        if target == EscalationTarget.TROUBLESHOOTER:
            metadata["structural_issues"] = self._collect_structural_issues(
                signals
            )

        logger.info(
            "Escalation triggered: target=%s reason=%s score=%.4f",
            target.value,
            reason.value,
            confidence_result.score,
        )

        return EscalationResult(
            should_escalate=True,
            target=target,
            reason=reason,
            confidence_score=confidence_result.score,
            confidence_tier=confidence_result.tier,
            steps_taken=self._step_counter.steps_taken,
            step_cap=self._step_counter.hard_cap,
            message=message,
            metadata=metadata,
        )

    def _determine_target(
        self,
        signals: Sequence[ConfidenceSignal],
        confidence_result: ConfidenceResult,
    ) -> EscalationTarget:
        """Decide whether to escalate to orchestrator or troubleshooter.

        Routes to troubleshooter when structural issues are detected
        (e.g. framework config exists but has problems), otherwise
        routes to orchestrator for user clarification.
        """
        structural_issues = self._collect_structural_issues(signals)
        if structural_issues:
            return EscalationTarget.TROUBLESHOOTER
        return EscalationTarget.ORCHESTRATOR

    def _determine_reason(
        self,
        signals: Sequence[ConfidenceSignal],
        confidence_result: ConfidenceResult,
    ) -> EscalationReason:
        """Classify the escalation reason."""
        if not signals:
            return EscalationReason.NO_SIGNALS_COLLECTED

        positive_count = sum(1 for s in signals if s.score > 0)
        if positive_count == 0:
            return EscalationReason.BUDGET_EXHAUSTED_NO_FINDINGS

        structural_issues = self._collect_structural_issues(signals)
        if structural_issues:
            return EscalationReason.STRUCTURAL_ISSUE_DETECTED

        return EscalationReason.LOW_CONFIDENCE_AT_CAP

    def _collect_structural_issues(
        self,
        signals: Sequence[ConfidenceSignal],
    ) -> list[dict[str, Any]]:
        """Find signals that indicate structural problems.

        A structural issue is when a framework indicator file exists
        (score > 0) but its detection signal has a low score, suggesting
        the config is broken or incomplete.
        """
        issues: list[dict[str, Any]] = []
        for signal in signals:
            if signal.name in self._structural_indicators:
                # Framework config found but overall confidence still low
                # suggests a structural problem
                if signal.score > 0 and signal.score < 0.5:
                    issues.append({
                        "signal": signal.name,
                        "score": signal.score,
                        "evidence": signal.evidence,
                    })
        return issues

    def _build_message(
        self,
        confidence_result: ConfidenceResult,
        target: EscalationTarget,
        reason: EscalationReason,
    ) -> str:
        """Build a human-readable escalation message."""
        score_pct = confidence_result.score * 100
        threshold_pct = self._escalation_threshold * 100

        base = (
            f"Discovery confidence ({score_pct:.1f}%) remains below "
            f"the {threshold_pct:.0f}% threshold after exhausting "
            f"the {self._step_counter.hard_cap}-step investigation budget."
        )

        if target == EscalationTarget.TROUBLESHOOTER:
            return (
                f"{base} Structural issues detected — escalating to "
                f"troubleshooter for diagnosis."
            )

        match reason:
            case EscalationReason.NO_SIGNALS_COLLECTED:
                return (
                    f"{base} No signals were collected — the project "
                    f"may be empty or inaccessible."
                )
            case EscalationReason.BUDGET_EXHAUSTED_NO_FINDINGS:
                return (
                    f"{base} No positive findings — the project may not "
                    f"contain recognizable test artifacts."
                )
            case _:
                return (
                    f"{base} Escalating to orchestrator for user "
                    f"clarification or additional guidance."
                )
