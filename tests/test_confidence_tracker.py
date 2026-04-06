"""Tests for confidence tracking and escalation logic.

Verifies that:
- Confidence history is recorded correctly across steps
- Trend analysis (improving/stable/declining) works
- Escalation triggers when confidence remains below 60% at step cap
- Escalation does NOT trigger when confidence >= 60% at step cap
- Escalation metadata includes tracking enrichment
- Persistent low confidence is flagged
- Integration with DiscoveryAgent and OrchestratorHub
"""

from __future__ import annotations

import pytest

from test_runner.agents.discovery.confidence_tracker import (
    ConfidenceSnapshot,
    ConfidenceTracker,
    ConfidenceTrend,
    TrackingResult,
)
from test_runner.agents.discovery.step_counter import StepCounter
from test_runner.agents.discovery.threshold_evaluator import (
    ConfidenceThresholdEvaluator,
    EscalationReason,
    EscalationTarget,
    ESCALATION_CONFIDENCE_THRESHOLD,
)
from test_runner.models.confidence import (
    ConfidenceModel,
    ConfidenceSignal,
    ConfidenceTier,
)
from test_runner.orchestrator.hub import (
    EscalationRecord,
    OrchestratorHub,
    RunPhase,
    RunState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signals(score: float, count: int = 3) -> list[ConfidenceSignal]:
    """Create uniform signals with the given score."""
    return [
        ConfidenceSignal(name=f"signal_{i}", weight=1.0, score=score)
        for i in range(count)
    ]


def _exhaust_counter(counter: StepCounter) -> None:
    """Consume all steps in the counter."""
    while not counter.is_exhausted:
        counter.increment("test_tool")


def _make_tracker(hard_cap: int = 5, threshold: float = 0.60) -> ConfidenceTracker:
    """Create a tracker with default settings."""
    counter = StepCounter(hard_cap=hard_cap)
    return ConfidenceTracker(
        step_counter=counter,
        escalation_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# ConfidenceSnapshot
# ---------------------------------------------------------------------------


class TestConfidenceSnapshot:
    def test_to_dict(self) -> None:
        snap = ConfidenceSnapshot(
            step_number=3,
            score=0.75,
            tier=ConfidenceTier.MEDIUM,
            signal_count=5,
            positive_signal_count=4,
        )
        d = snap.to_dict()
        assert d["step"] == 3
        assert d["score"] == 0.75
        assert d["tier"] == "medium"
        assert d["signal_count"] == 5
        assert d["positive_signal_count"] == 4


# ---------------------------------------------------------------------------
# ConfidenceTracker construction
# ---------------------------------------------------------------------------


class TestTrackerConstruction:
    def test_default_construction(self) -> None:
        counter = StepCounter(hard_cap=10)
        tracker = ConfidenceTracker(step_counter=counter)
        assert tracker.escalation_threshold == 0.60
        assert tracker.step_counter is counter
        assert len(tracker.history) == 0
        assert tracker.last_escalation is None

    def test_custom_threshold(self) -> None:
        counter = StepCounter(hard_cap=10)
        tracker = ConfidenceTracker(
            step_counter=counter,
            escalation_threshold=0.75,
        )
        assert tracker.escalation_threshold == 0.75

    def test_custom_evaluator(self) -> None:
        counter = StepCounter(hard_cap=10)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            escalation_threshold=0.80,
        )
        tracker = ConfidenceTracker(
            step_counter=counter,
            threshold_evaluator=evaluator,
        )
        assert tracker.threshold_evaluator is evaluator


# ---------------------------------------------------------------------------
# Recording snapshots
# ---------------------------------------------------------------------------


class TestRecordSnapshot:
    def test_records_snapshot_correctly(self) -> None:
        tracker = _make_tracker()
        tracker.step_counter.increment("tool_1")

        signals = _make_signals(0.45)
        snap = tracker.record_snapshot(signals)

        assert snap.step_number == 1
        assert abs(snap.score - 0.45) < 0.001
        assert snap.tier == ConfidenceTier.LOW
        assert snap.signal_count == 3
        assert snap.positive_signal_count == 3
        assert len(tracker.history) == 1

    def test_multiple_snapshots(self) -> None:
        tracker = _make_tracker()
        for i in range(3):
            tracker.step_counter.increment(f"tool_{i}")
            tracker.record_snapshot(_make_signals(0.30 + i * 0.15))

        assert len(tracker.history) == 3
        assert tracker.history[0].score < tracker.history[2].score

    def test_latest_score_empty(self) -> None:
        tracker = _make_tracker()
        assert tracker.latest_score == 0.0

    def test_latest_score_after_recording(self) -> None:
        tracker = _make_tracker()
        tracker.step_counter.increment("tool")
        tracker.record_snapshot(_make_signals(0.72))
        assert abs(tracker.latest_score - 0.72) < 0.001

    def test_average_score(self) -> None:
        tracker = _make_tracker()
        for score in [0.20, 0.40, 0.60]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))
        assert abs(tracker.average_score - 0.40) < 0.001


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------


class TestTrendAnalysis:
    def test_insufficient_data(self) -> None:
        tracker = _make_tracker()
        tracker.step_counter.increment("t1")
        tracker.record_snapshot(_make_signals(0.50))
        result = tracker.check(_make_signals(0.50))
        assert result.trend == ConfidenceTrend.INSUFFICIENT_DATA

    def test_improving_trend(self) -> None:
        tracker = _make_tracker(hard_cap=10)
        for score in [0.30, 0.45, 0.60]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))
        result = tracker.check(_make_signals(0.60))
        assert result.trend == ConfidenceTrend.IMPROVING

    def test_declining_trend(self) -> None:
        tracker = _make_tracker(hard_cap=10)
        for score in [0.80, 0.60, 0.40]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))
        result = tracker.check(_make_signals(0.40))
        assert result.trend == ConfidenceTrend.DECLINING

    def test_stable_trend(self) -> None:
        tracker = _make_tracker(hard_cap=10)
        for score in [0.50, 0.51, 0.50]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))
        result = tracker.check(_make_signals(0.50))
        assert result.trend == ConfidenceTrend.STABLE


# ---------------------------------------------------------------------------
# Core escalation: record_and_check
# ---------------------------------------------------------------------------


class TestRecordAndCheck:
    def test_escalation_at_cap_with_low_confidence(self) -> None:
        """Core test: confidence < 60% at step cap -> escalation."""
        tracker = _make_tracker(hard_cap=5)
        _exhaust_counter(tracker.step_counter)

        result = tracker.record_and_check(_make_signals(0.30))

        assert result.needs_escalation is True
        assert result.can_continue is False
        assert result.escalation is not None
        assert result.escalation.should_escalate is True
        assert result.escalation.confidence_score < 0.60
        assert result.latest_score < 0.60

    def test_no_escalation_at_cap_with_sufficient_confidence(self) -> None:
        """Confidence >= 60% at step cap -> no escalation."""
        tracker = _make_tracker(hard_cap=5)
        _exhaust_counter(tracker.step_counter)

        result = tracker.record_and_check(_make_signals(0.70))

        assert result.needs_escalation is False
        assert result.escalation is None
        assert result.can_continue is False  # Budget exhausted

    def test_no_escalation_before_cap(self) -> None:
        """Before step cap, low confidence doesn't trigger escalation."""
        tracker = _make_tracker(hard_cap=10)
        tracker.step_counter.increment("tool")

        result = tracker.record_and_check(_make_signals(0.20))

        assert result.needs_escalation is False
        assert result.can_continue is True

    def test_escalation_at_exactly_59_percent(self) -> None:
        tracker = _make_tracker(hard_cap=3)
        _exhaust_counter(tracker.step_counter)

        result = tracker.record_and_check(_make_signals(0.59))
        assert result.needs_escalation is True

    def test_no_escalation_at_exactly_60_percent(self) -> None:
        tracker = _make_tracker(hard_cap=3)
        _exhaust_counter(tracker.step_counter)

        result = tracker.record_and_check(_make_signals(0.60))
        assert result.needs_escalation is False

    def test_history_is_recorded_during_check(self) -> None:
        tracker = _make_tracker(hard_cap=10)
        tracker.step_counter.increment("tool")
        tracker.record_and_check(_make_signals(0.50))
        assert len(tracker.history) == 1


# ---------------------------------------------------------------------------
# Persistent low confidence tracking
# ---------------------------------------------------------------------------


class TestPersistentLowConfidence:
    def test_was_ever_above_threshold_false(self) -> None:
        tracker = _make_tracker(hard_cap=5)
        for score in [0.10, 0.20, 0.30, 0.40, 0.50]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))
        assert tracker.was_ever_above_threshold is False

    def test_was_ever_above_threshold_true(self) -> None:
        tracker = _make_tracker(hard_cap=5)
        for score in [0.10, 0.70, 0.30, 0.40, 0.50]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))
        assert tracker.was_ever_above_threshold is True

    def test_consecutive_below_count(self) -> None:
        tracker = _make_tracker(hard_cap=6)
        for score in [0.70, 0.50, 0.40, 0.30]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))
        assert tracker.consecutive_below_count == 3

    def test_consecutive_below_count_all_below(self) -> None:
        tracker = _make_tracker(hard_cap=5)
        for score in [0.10, 0.20, 0.30]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))
        assert tracker.consecutive_below_count == 3

    def test_consecutive_below_count_all_above(self) -> None:
        tracker = _make_tracker(hard_cap=5)
        for score in [0.70, 0.80, 0.90]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))
        assert tracker.consecutive_below_count == 0


# ---------------------------------------------------------------------------
# check_at_cap with enriched metadata
# ---------------------------------------------------------------------------


class TestCheckAtCap:
    def test_returns_none_before_cap(self) -> None:
        tracker = _make_tracker(hard_cap=10)
        tracker.step_counter.increment("tool")
        result = tracker.check_at_cap(_make_signals(0.30))
        assert result is None

    def test_returns_none_at_cap_with_high_confidence(self) -> None:
        tracker = _make_tracker(hard_cap=5)
        _exhaust_counter(tracker.step_counter)
        result = tracker.check_at_cap(_make_signals(0.80))
        assert result is None

    def test_returns_escalation_at_cap_with_low_confidence(self) -> None:
        tracker = _make_tracker(hard_cap=5)
        # Record some history before checking
        for score in [0.20, 0.25, 0.30, 0.35, 0.40]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))

        result = tracker.check_at_cap(_make_signals(0.40))

        assert result is not None
        assert result.should_escalate is True
        assert result.confidence_score < 0.60
        assert tracker.last_escalation is result

    def test_enriched_metadata_includes_tracking_info(self) -> None:
        tracker = _make_tracker(hard_cap=3)
        for score in [0.20, 0.25, 0.30]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))

        result = tracker.check_at_cap(_make_signals(0.30))

        assert result is not None
        assert "tracking" in result.metadata
        tracking = result.metadata["tracking"]
        assert tracking["history_length"] == 3
        assert tracking["was_ever_above_threshold"] is False
        assert tracking["persistent_low_confidence"] is True
        assert "average_score" in tracking
        assert "trend" in tracking
        assert "consecutive_below_count" in tracking

    def test_persistent_low_confidence_flag_when_never_above(self) -> None:
        tracker = _make_tracker(hard_cap=3)
        for score in [0.10, 0.20, 0.30]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))

        result = tracker.check_at_cap(_make_signals(0.30))
        assert result.metadata["tracking"]["persistent_low_confidence"] is True

    def test_not_persistent_when_was_above(self) -> None:
        tracker = _make_tracker(hard_cap=3)
        for score in [0.80, 0.30, 0.20]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))

        result = tracker.check_at_cap(_make_signals(0.20))
        assert result.metadata["tracking"]["persistent_low_confidence"] is False


# ---------------------------------------------------------------------------
# Tracker reset
# ---------------------------------------------------------------------------


class TestTrackerReset:
    def test_reset_clears_history(self) -> None:
        tracker = _make_tracker(hard_cap=10)
        tracker.step_counter.increment("tool")
        tracker.record_snapshot(_make_signals(0.50))
        assert len(tracker.history) == 1

        tracker.reset()
        assert len(tracker.history) == 0
        assert tracker.last_escalation is None
        assert tracker.latest_score == 0.0


# ---------------------------------------------------------------------------
# Tracker summary
# ---------------------------------------------------------------------------


class TestTrackerSummary:
    def test_summary_structure(self) -> None:
        tracker = _make_tracker(hard_cap=5)
        for score in [0.30, 0.40, 0.50]:
            tracker.step_counter.increment("tool")
            tracker.record_snapshot(_make_signals(score))

        summary = tracker.summary()
        assert summary["history_length"] == 3
        assert "latest_score" in summary
        assert "average_score" in summary
        assert "trend" in summary
        assert "was_ever_above_threshold" in summary
        assert "consecutive_below_count" in summary
        assert "escalation_threshold" in summary
        assert "has_escalation" in summary
        assert len(summary["snapshots"]) == 3


# ---------------------------------------------------------------------------
# TrackingResult structure
# ---------------------------------------------------------------------------


class TestTrackingResult:
    def test_summary_serializable(self) -> None:
        tracker = _make_tracker(hard_cap=5)
        _exhaust_counter(tracker.step_counter)

        # Record some history first
        for _ in range(3):
            tracker.record_snapshot(_make_signals(0.30))

        result = tracker.record_and_check(_make_signals(0.30))
        summary = result.summary()

        assert "needs_escalation" in summary
        assert "can_continue" in summary
        assert "trend" in summary
        assert "history_length" in summary
        assert "was_ever_above_threshold" in summary
        assert "consecutive_below_count" in summary
        assert "average_score" in summary
        assert "latest_score" in summary
        assert "threshold_check" in summary


# ---------------------------------------------------------------------------
# Integration: OrchestratorHub escalation handling
# ---------------------------------------------------------------------------


class TestOrchestratorEscalationHandling:
    def test_handle_escalation_to_troubleshooter(self) -> None:
        """Escalation targeting troubleshooter transitions to TROUBLESHOOTING."""
        from test_runner.agents.discovery.threshold_evaluator import (
            EscalationResult,
            EscalationReason,
            EscalationTarget,
        )

        state = RunState(request="run tests", phase=RunPhase.DISCOVERY)
        escalation = EscalationResult(
            should_escalate=True,
            target=EscalationTarget.TROUBLESHOOTER,
            reason=EscalationReason.STRUCTURAL_ISSUE_DETECTED,
            confidence_score=0.35,
            confidence_tier=ConfidenceTier.LOW,
            steps_taken=20,
            step_cap=20,
            message="Low confidence with structural issues",
            metadata={"structural_issues": [{"signal": "pytest_in_pyproject"}]},
        )

        # Create hub with minimal config
        from test_runner.config import Config

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        hub = OrchestratorHub(config)

        new_phase = hub.handle_escalation(state, escalation, "discovery")

        assert new_phase == RunPhase.TROUBLESHOOTING
        assert state.phase == RunPhase.TROUBLESHOOTING
        assert len(state.escalations) == 1
        assert state.escalations[0].target == "troubleshooter"
        assert state.escalations[0].confidence_score == 0.35

    def test_handle_escalation_to_orchestrator(self) -> None:
        """Escalation targeting orchestrator stays in current phase."""
        from test_runner.agents.discovery.threshold_evaluator import (
            EscalationResult,
            EscalationReason,
            EscalationTarget,
        )

        state = RunState(request="run tests", phase=RunPhase.DISCOVERY)
        escalation = EscalationResult(
            should_escalate=True,
            target=EscalationTarget.ORCHESTRATOR,
            reason=EscalationReason.LOW_CONFIDENCE_AT_CAP,
            confidence_score=0.45,
            confidence_tier=ConfidenceTier.LOW,
            steps_taken=20,
            step_cap=20,
            message="Low confidence, needs clarification",
        )

        from test_runner.config import Config

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        hub = OrchestratorHub(config)

        new_phase = hub.handle_escalation(state, escalation, "discovery")

        assert new_phase == RunPhase.DISCOVERY  # Stays in current phase
        assert len(state.escalations) == 1
        assert state.escalations[0].target == "orchestrator"

    def test_escalation_summary(self) -> None:
        from test_runner.agents.discovery.threshold_evaluator import (
            EscalationResult,
            EscalationReason,
            EscalationTarget,
        )

        state = RunState(request="run tests", phase=RunPhase.DISCOVERY)
        state.escalations.append(
            EscalationRecord(
                source_agent="discovery",
                target="troubleshooter",
                reason="structural_issue_detected",
                confidence_score=0.35,
                steps_taken=20,
            )
        )
        state.escalations.append(
            EscalationRecord(
                source_agent="discovery",
                target="orchestrator",
                reason="low_confidence_at_step_cap",
                confidence_score=0.45,
                steps_taken=15,
            )
        )

        from test_runner.config import Config

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        hub = OrchestratorHub(config)
        summary = hub.get_escalation_summary(state)

        assert summary["total_escalations"] == 2
        assert summary["routed_to_troubleshooter"] is True
        assert summary["needs_user_clarification"] is True

    def test_escalation_record_to_dict(self) -> None:
        record = EscalationRecord(
            source_agent="discovery",
            target="troubleshooter",
            reason="structural_issue_detected",
            confidence_score=0.35,
            steps_taken=20,
            metadata={"key": "value"},
        )
        d = record.to_dict()
        assert d["source_agent"] == "discovery"
        assert d["target"] == "troubleshooter"
        assert d["confidence_score"] == 0.35
        assert d["metadata"] == {"key": "value"}


# ---------------------------------------------------------------------------
# End-to-end: tracker -> orchestrator escalation flow
# ---------------------------------------------------------------------------


class TestEndToEndEscalationFlow:
    def test_full_flow_low_confidence_escalation(self) -> None:
        """Simulate a full discovery session where confidence stays low."""
        # Set up tracker with small budget
        counter = StepCounter(hard_cap=5)
        tracker = ConfidenceTracker(step_counter=counter)

        # Simulate 5 steps with persistently low confidence
        for i in range(5):
            counter.increment(f"tool_{i}")
            tracker.record_snapshot(_make_signals(0.20 + i * 0.05))

        # Check at cap — should trigger escalation
        escalation = tracker.check_at_cap(_make_signals(0.40))
        assert escalation is not None
        assert escalation.should_escalate is True
        assert escalation.confidence_score < 0.60

        # Route through orchestrator
        from test_runner.config import Config

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        hub = OrchestratorHub(config)
        state = RunState(request="run tests", phase=RunPhase.DISCOVERY)

        new_phase = hub.handle_escalation(state, escalation, "discovery")

        # Should stay with orchestrator (no structural issues)
        assert len(state.escalations) == 1
        assert state.escalations[0].source_agent == "discovery"

    def test_full_flow_structural_issue_escalation(self) -> None:
        """Simulate discovery where structural issues route to troubleshooter."""
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            structural_issue_indicators=["pytest_in_pyproject"],
        )
        tracker = ConfidenceTracker(
            step_counter=counter,
            threshold_evaluator=evaluator,
        )

        # Simulate steps
        for _ in range(3):
            counter.increment("tool")

        # Record history with structural issues
        structural_signals = [
            ConfidenceSignal(
                name="pytest_in_pyproject",
                weight=0.9,
                score=0.3,
                evidence={"file": "pyproject.toml"},
            ),
        ]
        for _ in range(3):
            tracker.record_snapshot(structural_signals)

        escalation = tracker.check_at_cap(structural_signals)
        assert escalation is not None
        assert escalation.target == EscalationTarget.TROUBLESHOOTER

        # Route through orchestrator
        from test_runner.config import Config

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        hub = OrchestratorHub(config)
        state = RunState(request="run tests", phase=RunPhase.DISCOVERY)

        new_phase = hub.handle_escalation(state, escalation, "discovery")
        assert new_phase == RunPhase.TROUBLESHOOTING
        assert state.phase == RunPhase.TROUBLESHOOTING

    def test_full_flow_confidence_recovers(self) -> None:
        """Confidence starts low but rises above 60% — no escalation."""
        counter = StepCounter(hard_cap=5)
        tracker = ConfidenceTracker(step_counter=counter)

        # Simulate steps with improving confidence
        scores = [0.20, 0.35, 0.50, 0.65, 0.80]
        for i, score in enumerate(scores):
            counter.increment(f"tool_{i}")
            tracker.record_snapshot(_make_signals(score))

        # At cap, confidence is 0.80 — no escalation
        escalation = tracker.check_at_cap(_make_signals(0.80))
        assert escalation is None
        assert tracker.was_ever_above_threshold is True
