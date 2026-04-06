"""Tests for the confidence threshold evaluator and step-cap escalation.

Verifies that:
- Escalation triggers when confidence < 60% at step cap
- No escalation when confidence >= 60% at step cap
- No escalation when step cap not yet reached
- Correct escalation target (orchestrator vs troubleshooter)
- Correct escalation reason classification
- Integration with DiscoveryAgent
"""

from __future__ import annotations

import pytest

from test_runner.agents.discovery.step_counter import StepCounter
from test_runner.agents.discovery.threshold_evaluator import (
    ConfidenceThresholdEvaluator,
    ESCALATION_CONFIDENCE_THRESHOLD,
    EscalationReason,
    EscalationResult,
    EscalationTarget,
    ThresholdCheckResult,
)
from test_runner.models.confidence import (
    ConfidenceModel,
    ConfidenceSignal,
    ConfidenceTier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signals(score: float, count: int = 3) -> list[ConfidenceSignal]:
    """Create a list of uniform signals with the given score."""
    return [
        ConfidenceSignal(name=f"signal_{i}", weight=1.0, score=score)
        for i in range(count)
    ]


def _make_mixed_signals(
    scores: list[tuple[str, float, float]],
) -> list[ConfidenceSignal]:
    """Create signals from (name, weight, score) triples."""
    return [
        ConfidenceSignal(name=name, weight=weight, score=score)
        for name, weight, score in scores
    ]


def _exhaust_counter(counter: StepCounter) -> None:
    """Consume all steps in the counter."""
    while not counter.is_exhausted:
        counter.increment("test_tool")


# ---------------------------------------------------------------------------
# ESCALATION_CONFIDENCE_THRESHOLD constant
# ---------------------------------------------------------------------------


class TestEscalationThresholdConstant:
    def test_threshold_is_60_percent(self) -> None:
        assert ESCALATION_CONFIDENCE_THRESHOLD == 0.60


# ---------------------------------------------------------------------------
# ConfidenceThresholdEvaluator construction
# ---------------------------------------------------------------------------


class TestEvaluatorConstruction:
    def test_default_construction(self) -> None:
        counter = StepCounter(hard_cap=10)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        assert evaluator.escalation_threshold == 0.60
        assert evaluator.step_counter is counter

    def test_custom_threshold(self) -> None:
        counter = StepCounter(hard_cap=10)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            escalation_threshold=0.75,
        )
        assert evaluator.escalation_threshold == 0.75

    def test_custom_confidence_model(self) -> None:
        counter = StepCounter(hard_cap=10)
        model = ConfidenceModel(execute_threshold=0.95, warn_threshold=0.70)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            confidence_model=model,
        )
        assert evaluator.confidence_model is model

    def test_invalid_threshold_raises(self) -> None:
        counter = StepCounter(hard_cap=10)
        with pytest.raises(ValueError, match="escalation_threshold"):
            ConfidenceThresholdEvaluator(
                step_counter=counter,
                escalation_threshold=1.5,
            )

    def test_negative_threshold_raises(self) -> None:
        counter = StepCounter(hard_cap=10)
        with pytest.raises(ValueError, match="escalation_threshold"):
            ConfidenceThresholdEvaluator(
                step_counter=counter,
                escalation_threshold=-0.1,
            )


# ---------------------------------------------------------------------------
# Core escalation logic: evaluate()
# ---------------------------------------------------------------------------


class TestEvaluateEscalation:
    """Test the primary escalation condition: cap reached + low confidence."""

    def test_escalation_when_cap_reached_and_confidence_below_60(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.30)
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is True
        assert result.can_continue is False
        assert result.budget_remaining == 0
        assert result.escalation is not None
        assert result.escalation.should_escalate is True
        assert result.escalation.confidence_score < 0.60

    def test_no_escalation_when_cap_reached_and_confidence_at_60(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.60)
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is False
        assert result.escalation is None
        assert result.can_continue is False  # Budget exhausted

    def test_no_escalation_when_cap_reached_and_confidence_above_60(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.85)
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is False
        assert result.escalation is None

    def test_no_escalation_when_cap_not_reached(self) -> None:
        counter = StepCounter(hard_cap=10)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        counter.increment("tool_1")  # 1 of 10

        signals = _make_signals(score=0.20)
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is False
        assert result.can_continue is True
        assert result.budget_remaining == 9

    def test_escalation_at_exactly_zero_confidence(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.0)
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is True
        assert result.escalation is not None
        assert result.escalation.confidence_score == 0.0

    def test_escalation_with_empty_signals_at_cap(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        result = evaluator.evaluate([])

        assert result.needs_escalation is True
        assert result.escalation is not None
        assert result.escalation.confidence_score == 0.0

    def test_escalation_at_59_percent(self) -> None:
        """Just below the 60% threshold should escalate."""
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.59)
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is True

    def test_no_escalation_at_60_percent_exactly(self) -> None:
        """Exactly at the 60% threshold should NOT escalate."""
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.60)
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is False

    def test_custom_threshold_respected(self) -> None:
        """Custom escalation threshold of 75% should trigger at 70%."""
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            escalation_threshold=0.75,
        )
        _exhaust_counter(counter)

        signals = _make_signals(score=0.70)
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is True


# ---------------------------------------------------------------------------
# check_at_step_cap() convenience method
# ---------------------------------------------------------------------------


class TestCheckAtStepCap:
    def test_returns_none_when_cap_not_reached(self) -> None:
        counter = StepCounter(hard_cap=10)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        counter.increment("tool")

        result = evaluator.check_at_step_cap(_make_signals(0.20))
        assert result is None

    def test_returns_none_when_cap_reached_but_high_confidence(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        result = evaluator.check_at_step_cap(_make_signals(0.90))
        assert result is None

    def test_returns_escalation_when_cap_reached_and_low_confidence(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        result = evaluator.check_at_step_cap(_make_signals(0.40))
        assert result is not None
        assert isinstance(result, EscalationResult)
        assert result.should_escalate is True
        assert result.confidence_score < 0.60


# ---------------------------------------------------------------------------
# Escalation target routing
# ---------------------------------------------------------------------------


class TestEscalationTarget:
    def test_routes_to_orchestrator_by_default(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.30)
        result = evaluator.evaluate(signals)

        assert result.escalation is not None
        assert result.escalation.target == EscalationTarget.ORCHESTRATOR

    def test_routes_to_troubleshooter_on_structural_issues(self) -> None:
        """When a structural indicator has a low but non-zero score,
        route to troubleshooter instead of orchestrator."""
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            structural_issue_indicators=["pytest_in_pyproject"],
        )
        _exhaust_counter(counter)

        # pytest_in_pyproject has a non-zero but low score = structural issue
        signals = [
            ConfidenceSignal(
                name="pytest_in_pyproject",
                weight=0.9,
                score=0.3,
                evidence={"file": "pyproject.toml", "matched": True},
            ),
            ConfidenceSignal(name="other_signal", weight=0.5, score=0.1),
        ]
        result = evaluator.evaluate(signals)

        assert result.escalation is not None
        assert result.escalation.target == EscalationTarget.TROUBLESHOOTER

    def test_routes_to_orchestrator_when_no_structural_issues(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            structural_issue_indicators=["pytest_in_pyproject"],
        )
        _exhaust_counter(counter)

        # All signals have either 0.0 or high scores — no structural issues
        signals = [
            ConfidenceSignal(name="some_signal", weight=0.8, score=0.0),
            ConfidenceSignal(name="another", weight=0.5, score=0.1),
        ]
        result = evaluator.evaluate(signals)

        assert result.escalation is not None
        assert result.escalation.target == EscalationTarget.ORCHESTRATOR


# ---------------------------------------------------------------------------
# Escalation reason classification
# ---------------------------------------------------------------------------


class TestEscalationReason:
    def test_no_signals_collected_reason(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        result = evaluator.evaluate([])

        assert result.escalation is not None
        assert result.escalation.reason == EscalationReason.NO_SIGNALS_COLLECTED

    def test_no_findings_reason(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        # All signals have score=0
        signals = _make_signals(score=0.0)
        result = evaluator.evaluate(signals)

        assert result.escalation is not None
        assert (
            result.escalation.reason
            == EscalationReason.BUDGET_EXHAUSTED_NO_FINDINGS
        )

    def test_low_confidence_at_cap_reason(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        # Some positive signals but overall low confidence
        signals = _make_signals(score=0.40)
        result = evaluator.evaluate(signals)

        assert result.escalation is not None
        assert (
            result.escalation.reason
            == EscalationReason.LOW_CONFIDENCE_AT_CAP
        )

    def test_structural_issue_reason(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            structural_issue_indicators=["jest_in_package_json"],
        )
        _exhaust_counter(counter)

        signals = [
            ConfidenceSignal(
                name="jest_in_package_json",
                weight=0.9,
                score=0.3,
            ),
        ]
        result = evaluator.evaluate(signals)

        assert result.escalation is not None
        assert (
            result.escalation.reason
            == EscalationReason.STRUCTURAL_ISSUE_DETECTED
        )


# ---------------------------------------------------------------------------
# EscalationResult structure
# ---------------------------------------------------------------------------


class TestEscalationResultStructure:
    def test_escalation_result_fields(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.35)
        result = evaluator.evaluate(signals)

        esc = result.escalation
        assert esc is not None
        assert esc.should_escalate is True
        assert esc.steps_taken == 5
        assert esc.step_cap == 5
        assert 0.0 <= esc.confidence_score <= 1.0
        assert isinstance(esc.confidence_tier, ConfidenceTier)
        assert isinstance(esc.message, str)
        assert len(esc.message) > 0

    def test_escalation_result_summary(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.35)
        result = evaluator.evaluate(signals)

        summary = result.escalation.summary()
        assert "should_escalate" in summary
        assert "target" in summary
        assert "reason" in summary
        assert "confidence_score" in summary
        assert "confidence_tier" in summary
        assert "steps_taken" in summary
        assert "step_cap" in summary
        assert "message" in summary
        assert "metadata" in summary

    def test_escalation_metadata_includes_step_summary(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.35)
        result = evaluator.evaluate(signals)

        meta = result.escalation.metadata
        assert "step_summary" in meta
        assert "confidence_summary" in meta
        assert "signal_count" in meta
        assert meta["signal_count"] == 3

    def test_troubleshooter_metadata_includes_structural_issues(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            structural_issue_indicators=["pytest_in_pyproject"],
        )
        _exhaust_counter(counter)

        signals = [
            ConfidenceSignal(
                name="pytest_in_pyproject", weight=0.9, score=0.3,
            ),
        ]
        result = evaluator.evaluate(signals)

        meta = result.escalation.metadata
        assert "structural_issues" in meta
        assert len(meta["structural_issues"]) > 0


# ---------------------------------------------------------------------------
# ThresholdCheckResult structure
# ---------------------------------------------------------------------------


class TestThresholdCheckResultStructure:
    def test_result_when_can_continue(self) -> None:
        counter = StepCounter(hard_cap=10)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)

        signals = _make_signals(score=0.50)
        result = evaluator.evaluate(signals)

        assert result.can_continue is True
        assert result.needs_escalation is False
        assert result.budget_remaining == 10
        assert result.confidence_result is not None

    def test_summary_serializable(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.30)
        result = evaluator.evaluate(signals)

        summary = result.summary()
        assert "needs_escalation" in summary
        assert "can_continue" in summary
        assert "budget_remaining" in summary
        assert "confidence" in summary
        assert "escalation" in summary  # Present because escalation triggered


# ---------------------------------------------------------------------------
# Escalation message content
# ---------------------------------------------------------------------------


class TestEscalationMessages:
    def test_message_mentions_confidence_percentage(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.35)
        result = evaluator.evaluate(signals)

        msg = result.escalation.message
        assert "35.0%" in msg
        assert "60%" in msg

    def test_message_mentions_step_budget(self) -> None:
        counter = StepCounter(hard_cap=5)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.35)
        result = evaluator.evaluate(signals)

        assert "5-step" in result.escalation.message

    def test_troubleshooter_message(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            structural_issue_indicators=["pytest_in_pyproject"],
        )
        _exhaust_counter(counter)

        signals = [
            ConfidenceSignal(
                name="pytest_in_pyproject", weight=0.9, score=0.3,
            ),
        ]
        result = evaluator.evaluate(signals)

        assert "troubleshooter" in result.escalation.message.lower()

    def test_no_signals_message(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        result = evaluator.evaluate([])
        assert "no signals" in result.escalation.message.lower()

    def test_no_findings_message(self) -> None:
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_signals(score=0.0)
        result = evaluator.evaluate(signals)

        assert "no positive findings" in result.escalation.message.lower()


# ---------------------------------------------------------------------------
# Integration with DiscoveryAgent
# ---------------------------------------------------------------------------


class TestDiscoveryAgentIntegration:
    def test_agent_has_threshold_evaluator(self) -> None:
        from test_runner.agents.discovery.agent import DiscoveryAgent

        agent = DiscoveryAgent(hard_cap_steps=10)
        assert agent.threshold_evaluator is not None
        assert isinstance(
            agent.threshold_evaluator, ConfidenceThresholdEvaluator
        )

    def test_agent_evaluate_confidence_at_cap_triggers_escalation(
        self,
    ) -> None:
        from test_runner.agents.discovery.agent import DiscoveryAgent

        agent = DiscoveryAgent(hard_cap_steps=5)
        _exhaust_counter(agent.step_counter)

        signals = _make_signals(score=0.30)
        result = agent.evaluate_confidence_at_cap(signals)

        assert result is not None
        assert result.should_escalate is True
        assert agent.last_escalation is result
        assert agent.state.escalation_reason is not None

    def test_agent_evaluate_confidence_at_cap_no_escalation(self) -> None:
        from test_runner.agents.discovery.agent import DiscoveryAgent

        agent = DiscoveryAgent(hard_cap_steps=5)
        _exhaust_counter(agent.step_counter)

        signals = _make_signals(score=0.80)
        result = agent.evaluate_confidence_at_cap(signals)

        assert result is None
        assert agent.last_escalation is None

    def test_agent_check_threshold_mid_exploration(self) -> None:
        from test_runner.agents.discovery.agent import DiscoveryAgent

        agent = DiscoveryAgent(hard_cap_steps=10)
        agent.step_counter.increment("tool_1")

        signals = _make_signals(score=0.20)
        result = agent.check_threshold(signals)

        assert result.can_continue is True
        assert result.needs_escalation is False

    def test_agent_handoff_summary_includes_escalation(self) -> None:
        from test_runner.agents.discovery.agent import DiscoveryAgent

        agent = DiscoveryAgent(hard_cap_steps=3)
        _exhaust_counter(agent.step_counter)

        signals = _make_signals(score=0.25)
        agent.evaluate_confidence_at_cap(signals)

        summary = agent.get_handoff_summary()
        assert "escalation" in summary
        assert summary["escalation"]["should_escalate"] is True

    def test_agent_handoff_summary_no_escalation(self) -> None:
        from test_runner.agents.discovery.agent import DiscoveryAgent

        agent = DiscoveryAgent(hard_cap_steps=10)
        summary = agent.get_handoff_summary()
        assert "escalation" not in summary


# ---------------------------------------------------------------------------
# Edge cases with mixed signals
# ---------------------------------------------------------------------------


class TestMixedSignals:
    def test_weighted_average_below_threshold(self) -> None:
        """Mixed signals that average below 60% should trigger escalation."""
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_mixed_signals([
            ("high_signal", 0.3, 0.90),  # High but low weight
            ("low_signal_1", 0.8, 0.20),  # Low and high weight
            ("low_signal_2", 0.9, 0.30),  # Low and high weight
        ])
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is True
        assert result.escalation.confidence_score < 0.60

    def test_weighted_average_above_threshold(self) -> None:
        """Mixed signals that average above 60% should NOT trigger escalation."""
        counter = StepCounter(hard_cap=3)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = _make_mixed_signals([
            ("high_signal", 0.9, 0.80),  # High weight and score
            ("low_signal", 0.1, 0.10),  # Low weight
        ])
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is False

    def test_single_signal_below_threshold(self) -> None:
        counter = StepCounter(hard_cap=2)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust_counter(counter)

        signals = [ConfidenceSignal(name="only", weight=1.0, score=0.50)]
        result = evaluator.evaluate(signals)

        assert result.needs_escalation is True
        assert result.escalation.confidence_score == 0.50
