"""Confidence tracker for continuous monitoring across discovery steps.

Maintains a rolling history of confidence evaluations taken after each
discovery step. The tracker detects when confidence *remains* below the
escalation threshold (default 60%) after the step cap is reached, and
produces structured escalation requests targeting the orchestrator or
troubleshooter agent.

Key behaviours:
- Records a confidence snapshot after every step
- Computes a rolling trend (improving / stable / declining)
- At step-cap exhaustion, checks whether confidence has *persistently*
  stayed below the threshold (not just a one-off dip)
- Produces escalation results compatible with the orchestrator hub's
  routing logic

The tracker is owned by the discovery agent and consulted by the
orchestrator hub when the discovery session ends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from test_runner.agents.discovery.step_counter import StepCounter
from test_runner.agents.discovery.threshold_evaluator import (
    ConfidenceThresholdEvaluator,
    EscalationResult,
    EscalationReason,
    EscalationTarget,
    ESCALATION_CONFIDENCE_THRESHOLD,
    ThresholdCheckResult,
)
from test_runner.models.confidence import (
    ConfidenceModel,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceTier,
)

logger = logging.getLogger(__name__)

# Minimum number of snapshots needed before trend analysis is meaningful.
_MIN_TREND_SNAPSHOTS: int = 3

# Trend classification thresholds (applied to slope of recent scores).
_IMPROVING_SLOPE_THRESHOLD: float = 0.02  # >= 2% improvement per step
_DECLINING_SLOPE_THRESHOLD: float = -0.02  # <= -2% decline per step


class ConfidenceTrend(str):
    """Trend direction of confidence over recent steps."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ConfidenceSnapshot:
    """A single confidence evaluation captured at a specific step.

    Attributes:
        step_number: The step at which this snapshot was taken.
        score: The aggregated confidence score at this step.
        tier: The confidence tier at this step.
        signal_count: Number of signals used for this evaluation.
        positive_signal_count: Signals with score > 0.
    """

    step_number: int
    score: float
    tier: ConfidenceTier
    signal_count: int = 0
    positive_signal_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation."""
        return {
            "step": self.step_number,
            "score": round(self.score, 4),
            "tier": self.tier.value,
            "signal_count": self.signal_count,
            "positive_signal_count": self.positive_signal_count,
        }


@dataclass
class TrackingResult:
    """Result of a confidence tracking check.

    Combines the current threshold check with historical trend
    information to give the orchestrator a complete picture.

    Attributes:
        threshold_check: The underlying point-in-time threshold check.
        trend: The confidence trend over recent steps.
        history_length: Number of snapshots recorded so far.
        was_ever_above_threshold: True if any snapshot had score >= threshold.
        consecutive_below_count: Number of consecutive recent snapshots
            below the escalation threshold.
        average_score: Mean score across all recorded snapshots.
        latest_score: Most recent confidence score.
    """

    threshold_check: ThresholdCheckResult
    trend: str  # ConfidenceTrend value
    history_length: int
    was_ever_above_threshold: bool
    consecutive_below_count: int
    average_score: float
    latest_score: float

    @property
    def needs_escalation(self) -> bool:
        """True if the threshold check triggers escalation."""
        return self.threshold_check.needs_escalation

    @property
    def can_continue(self) -> bool:
        """True if exploration can continue."""
        return self.threshold_check.can_continue

    @property
    def escalation(self) -> EscalationResult | None:
        """The escalation result, if any."""
        return self.threshold_check.escalation

    def summary(self) -> dict[str, Any]:
        """Serializable representation for logging / handoff."""
        result: dict[str, Any] = {
            "needs_escalation": self.needs_escalation,
            "can_continue": self.can_continue,
            "trend": self.trend,
            "history_length": self.history_length,
            "was_ever_above_threshold": self.was_ever_above_threshold,
            "consecutive_below_count": self.consecutive_below_count,
            "average_score": round(self.average_score, 4),
            "latest_score": round(self.latest_score, 4),
            "threshold_check": self.threshold_check.summary(),
        }
        if self.escalation is not None:
            result["escalation"] = self.escalation.summary()
        return result


class ConfidenceTracker:
    """Tracks confidence across discovery steps and manages escalation.

    The tracker wraps a :class:`ConfidenceThresholdEvaluator` and adds
    historical trend analysis. After each discovery step, the caller
    records a snapshot. When the step budget is exhausted, the tracker
    determines whether confidence has persistently stayed below the
    escalation threshold and produces a structured escalation result.

    Usage::

        tracker = ConfidenceTracker(step_counter, evaluator)

        # After each discovery step:
        signals = collect_signals(...)
        result = tracker.record_and_check(signals)
        if result.needs_escalation:
            # Route to orchestrator/troubleshooter
            escalation = result.escalation
            ...
        elif not result.can_continue:
            # Budget exhausted but confidence is OK
            ...

    Args:
        step_counter: Shared step counter for budget tracking.
        threshold_evaluator: Evaluator for point-in-time checks.
        escalation_threshold: Confidence below which escalation triggers
            at step cap (default 0.60).
    """

    def __init__(
        self,
        step_counter: StepCounter,
        threshold_evaluator: ConfidenceThresholdEvaluator | None = None,
        escalation_threshold: float = ESCALATION_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._step_counter = step_counter
        self._escalation_threshold = escalation_threshold

        if threshold_evaluator is not None:
            self._evaluator = threshold_evaluator
        else:
            self._evaluator = ConfidenceThresholdEvaluator(
                step_counter=step_counter,
                escalation_threshold=escalation_threshold,
            )

        self._history: list[ConfidenceSnapshot] = []
        self._last_escalation: EscalationResult | None = None

    # -- Properties -----------------------------------------------------------

    @property
    def step_counter(self) -> StepCounter:
        """The step counter being tracked."""
        return self._step_counter

    @property
    def threshold_evaluator(self) -> ConfidenceThresholdEvaluator:
        """The underlying threshold evaluator."""
        return self._evaluator

    @property
    def escalation_threshold(self) -> float:
        """The confidence threshold for escalation."""
        return self._escalation_threshold

    @property
    def history(self) -> list[ConfidenceSnapshot]:
        """Chronological list of confidence snapshots."""
        return list(self._history)

    @property
    def last_escalation(self) -> EscalationResult | None:
        """Most recent escalation result, if any."""
        return self._last_escalation

    @property
    def latest_score(self) -> float:
        """Most recent confidence score, or 0.0 if no history."""
        if not self._history:
            return 0.0
        return self._history[-1].score

    @property
    def average_score(self) -> float:
        """Mean confidence score across all snapshots."""
        if not self._history:
            return 0.0
        return sum(s.score for s in self._history) / len(self._history)

    @property
    def was_ever_above_threshold(self) -> bool:
        """True if any snapshot had score >= escalation threshold."""
        return any(
            s.score >= self._escalation_threshold for s in self._history
        )

    @property
    def consecutive_below_count(self) -> int:
        """Number of consecutive recent snapshots below threshold."""
        count = 0
        for snapshot in reversed(self._history):
            if snapshot.score < self._escalation_threshold:
                count += 1
            else:
                break
        return count

    # -- Recording and checking ----------------------------------------------

    def record_snapshot(
        self,
        signals: Sequence[ConfidenceSignal],
    ) -> ConfidenceSnapshot:
        """Record a confidence snapshot from current signals.

        Does NOT perform escalation checking — use :meth:`record_and_check`
        for the combined operation, or call :meth:`check` separately.

        Args:
            signals: Current confidence signals from collectors.

        Returns:
            The recorded snapshot.
        """
        confidence_result = self._evaluator.confidence_model.evaluate(signals)
        snapshot = ConfidenceSnapshot(
            step_number=self._step_counter.steps_taken,
            score=confidence_result.score,
            tier=confidence_result.tier,
            signal_count=len(signals),
            positive_signal_count=sum(1 for s in signals if s.score > 0),
        )
        self._history.append(snapshot)

        logger.debug(
            "Confidence snapshot step=%d score=%.4f tier=%s "
            "(history=%d snapshots)",
            snapshot.step_number,
            snapshot.score,
            snapshot.tier.value,
            len(self._history),
        )

        return snapshot

    def check(
        self,
        signals: Sequence[ConfidenceSignal],
    ) -> TrackingResult:
        """Perform a threshold check with trend analysis.

        Does NOT record a snapshot — use :meth:`record_and_check` for
        the combined operation.

        Args:
            signals: Current confidence signals for threshold evaluation.

        Returns:
            A TrackingResult combining the threshold check with trend data.
        """
        threshold_check = self._evaluator.evaluate(signals)

        if threshold_check.needs_escalation and threshold_check.escalation:
            self._last_escalation = threshold_check.escalation

        trend = self._compute_trend()
        latest = self._history[-1].score if self._history else 0.0

        return TrackingResult(
            threshold_check=threshold_check,
            trend=trend,
            history_length=len(self._history),
            was_ever_above_threshold=self.was_ever_above_threshold,
            consecutive_below_count=self.consecutive_below_count,
            average_score=self.average_score,
            latest_score=latest,
        )

    def record_and_check(
        self,
        signals: Sequence[ConfidenceSignal],
    ) -> TrackingResult:
        """Record a snapshot and perform a threshold check in one call.

        This is the primary API. Call it after each discovery step:
        1. Records the current confidence as a snapshot
        2. Evaluates against the threshold/budget
        3. Computes the confidence trend
        4. Returns a TrackingResult with all information

        Args:
            signals: Current confidence signals from collectors.

        Returns:
            A TrackingResult with escalation info if needed.
        """
        self.record_snapshot(signals)
        return self.check(signals)

    def check_at_cap(
        self,
        signals: Sequence[ConfidenceSignal],
    ) -> EscalationResult | None:
        """Check for escalation specifically at step-cap exhaustion.

        This is a convenience method for the orchestrator to call when
        the discovery session ends. It checks whether:
        1. The step cap has been reached
        2. Confidence remains below 60%

        If both conditions are met and confidence has *persistently*
        been below the threshold (never rose above it), the escalation
        is flagged as more severe in the metadata.

        Args:
            signals: Final confidence signals from the discovery session.

        Returns:
            An EscalationResult if escalation is needed, None otherwise.
        """
        escalation = self._evaluator.check_at_step_cap(signals)

        if escalation is not None:
            self._last_escalation = escalation

            # Enrich with tracking metadata
            enriched_metadata = dict(escalation.metadata)
            enriched_metadata["tracking"] = {
                "history_length": len(self._history),
                "average_score": round(self.average_score, 4),
                "was_ever_above_threshold": self.was_ever_above_threshold,
                "consecutive_below_count": self.consecutive_below_count,
                "trend": self._compute_trend(),
                "persistent_low_confidence": not self.was_ever_above_threshold,
            }

            # Rebuild the escalation with enriched metadata
            escalation = EscalationResult(
                should_escalate=escalation.should_escalate,
                target=escalation.target,
                reason=escalation.reason,
                confidence_score=escalation.confidence_score,
                confidence_tier=escalation.confidence_tier,
                steps_taken=escalation.steps_taken,
                step_cap=escalation.step_cap,
                message=escalation.message,
                metadata=enriched_metadata,
            )
            self._last_escalation = escalation

        return escalation

    # -- Trend analysis -------------------------------------------------------

    def _compute_trend(self) -> str:
        """Compute the confidence trend from recent snapshots.

        Uses linear regression slope over the last N snapshots
        (minimum 3) to classify the trend as improving, stable,
        or declining.
        """
        if len(self._history) < _MIN_TREND_SNAPSHOTS:
            return ConfidenceTrend.INSUFFICIENT_DATA

        recent = self._history[-_MIN_TREND_SNAPSHOTS:]
        scores = [s.score for s in recent]

        # Simple slope: (last - first) / (n - 1)
        n = len(scores)
        slope = (scores[-1] - scores[0]) / (n - 1) if n > 1 else 0.0

        if slope >= _IMPROVING_SLOPE_THRESHOLD:
            return ConfidenceTrend.IMPROVING
        if slope <= _DECLINING_SLOPE_THRESHOLD:
            return ConfidenceTrend.DECLINING
        return ConfidenceTrend.STABLE

    # -- Reset / serialization ------------------------------------------------

    def reset(self) -> None:
        """Reset tracking state for a new discovery session."""
        self._history.clear()
        self._last_escalation = None

    def summary(self) -> dict[str, Any]:
        """Serializable summary of the tracker's state."""
        return {
            "history_length": len(self._history),
            "latest_score": round(self.latest_score, 4),
            "average_score": round(self.average_score, 4),
            "trend": self._compute_trend(),
            "was_ever_above_threshold": self.was_ever_above_threshold,
            "consecutive_below_count": self.consecutive_below_count,
            "escalation_threshold": self._escalation_threshold,
            "has_escalation": self._last_escalation is not None,
            "snapshots": [s.to_dict() for s in self._history],
        }
