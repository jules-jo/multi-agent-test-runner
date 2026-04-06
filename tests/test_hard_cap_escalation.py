"""Tests for AC 4: Hard cap of ~20 investigation steps during discovery,
escalate if confidence remains <60%.

This module validates the end-to-end behaviour of the hard cap enforcement:
1. The default investigation budget is 20 steps.
2. Each tool invocation counts as exactly one step.
3. Tools return a budget-exceeded error once the cap is reached.
4. When the cap is hit with confidence < 60%, escalation is triggered.
5. When the cap is hit with confidence >= 60%, no escalation occurs.
6. The orchestrator correctly routes escalations from discovery.
7. The step counter, threshold evaluator, confidence tracker, discovery
   agent, and orchestrator hub all integrate correctly.
"""

from __future__ import annotations

import pytest

from test_runner.agents.base import AgentRole
from test_runner.agents.discovery.agent import (
    DiscoveryAgent,
    create_discovery_agent,
)
from test_runner.agents.discovery.confidence_tracker import (
    ConfidenceTracker,
    ConfidenceTrend,
)
from test_runner.agents.discovery.step_counter import (
    BUDGET_EXCEEDED_RESPONSE,
    DEFAULT_HARD_CAP,
    StepCounter,
)
from test_runner.agents.discovery.threshold_evaluator import (
    ConfidenceThresholdEvaluator,
    ESCALATION_CONFIDENCE_THRESHOLD,
    EscalationReason,
    EscalationTarget,
)
from test_runner.config import Config
from test_runner.models.confidence import ConfidenceSignal, ConfidenceTier
from test_runner.orchestrator.hub import (
    OrchestratorHub,
    RunPhase,
    RunState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signals(score: float, count: int = 3) -> list[ConfidenceSignal]:
    return [
        ConfidenceSignal(name=f"sig_{i}", weight=1.0, score=score)
        for i in range(count)
    ]


def _exhaust(counter: StepCounter) -> None:
    while not counter.is_exhausted:
        counter.increment("tool")


def _make_hub() -> OrchestratorHub:
    config = Config(llm_base_url="http://test", api_key="test", model_id="test")
    return OrchestratorHub(config)


# ---------------------------------------------------------------------------
# 1. Default hard cap is 20
# ---------------------------------------------------------------------------


class TestDefaultHardCapIs20:
    def test_constant_equals_20(self) -> None:
        assert DEFAULT_HARD_CAP == 20

    def test_step_counter_default(self) -> None:
        assert StepCounter().hard_cap == 20

    def test_discovery_agent_default(self) -> None:
        agent = DiscoveryAgent()
        assert agent.hard_cap_steps == 20
        assert agent.step_counter.hard_cap == 20

    def test_sdk_agent_instructions_mention_20(self) -> None:
        sdk_agent = create_discovery_agent()
        assert "20" in sdk_agent.instructions


# ---------------------------------------------------------------------------
# 2. Every tool invocation counts as one step
# ---------------------------------------------------------------------------


class TestEachToolCallCountsOneStep:
    def test_increment_adds_exactly_one(self) -> None:
        c = StepCounter(hard_cap=20)
        for i in range(1, 21):
            assert c.increment(f"tool_{i}") is True
            assert c.steps_taken == i

    def test_twenty_increments_exhaust_budget(self) -> None:
        c = StepCounter(hard_cap=20)
        for _ in range(20):
            c.increment("t")
        assert c.is_exhausted is True
        assert c.remaining == 0

    def test_21st_call_is_rejected(self) -> None:
        c = StepCounter(hard_cap=20)
        for _ in range(20):
            c.increment("t")
        assert c.increment("extra") is False
        assert c.steps_taken == 20  # unchanged


# ---------------------------------------------------------------------------
# 3. Tools return budget-exceeded once cap is hit
# ---------------------------------------------------------------------------


class TestToolBudgetExceededResponse:
    def test_response_shape(self) -> None:
        assert BUDGET_EXCEEDED_RESPONSE["error"] == "step_budget_exhausted"
        assert "message" in BUDGET_EXCEEDED_RESPONSE

    def test_budget_status_message_at_cap(self) -> None:
        c = StepCounter(hard_cap=20)
        for _ in range(20):
            c.increment("t")
        msg = c.budget_status_message()
        assert "BUDGET EXHAUSTED" in msg
        assert "20" in msg


# ---------------------------------------------------------------------------
# 4. Escalation threshold is 60%
# ---------------------------------------------------------------------------


class TestEscalationThresholdIs60:
    def test_constant(self) -> None:
        assert ESCALATION_CONFIDENCE_THRESHOLD == 0.60

    def test_evaluator_default(self) -> None:
        ev = ConfidenceThresholdEvaluator(step_counter=StepCounter())
        assert ev.escalation_threshold == 0.60


# ---------------------------------------------------------------------------
# 5. Escalation triggers at cap with confidence < 60%
# ---------------------------------------------------------------------------


class TestEscalationAtCapBelowThreshold:
    @pytest.mark.parametrize("score", [0.0, 0.10, 0.30, 0.50, 0.59])
    def test_escalation_triggers(self, score: float) -> None:
        counter = StepCounter(hard_cap=20)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust(counter)

        result = evaluator.evaluate(_signals(score))
        assert result.needs_escalation is True
        assert result.escalation is not None
        assert result.escalation.should_escalate is True
        assert result.escalation.steps_taken == 20
        assert result.escalation.step_cap == 20
        assert result.can_continue is False

    def test_escalation_message_references_60_percent(self) -> None:
        counter = StepCounter(hard_cap=20)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust(counter)

        result = evaluator.evaluate(_signals(0.45))
        assert "60%" in result.escalation.message
        assert "20-step" in result.escalation.message


# ---------------------------------------------------------------------------
# 6. No escalation at cap with confidence >= 60%
# ---------------------------------------------------------------------------


class TestNoEscalationAtCapAboveThreshold:
    @pytest.mark.parametrize("score", [0.60, 0.70, 0.85, 0.95, 1.0])
    def test_no_escalation(self, score: float) -> None:
        counter = StepCounter(hard_cap=20)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        _exhaust(counter)

        result = evaluator.evaluate(_signals(score))
        assert result.needs_escalation is False
        assert result.escalation is None
        assert result.can_continue is False  # budget exhausted, but OK


# ---------------------------------------------------------------------------
# 7. No escalation before cap even with low confidence
# ---------------------------------------------------------------------------


class TestNoEscalationBeforeCap:
    def test_low_confidence_but_budget_remaining(self) -> None:
        counter = StepCounter(hard_cap=20)
        evaluator = ConfidenceThresholdEvaluator(step_counter=counter)
        counter.increment("tool_1")  # 1 of 20

        result = evaluator.evaluate(_signals(0.10))
        assert result.needs_escalation is False
        assert result.can_continue is True
        assert result.budget_remaining == 19


# ---------------------------------------------------------------------------
# 8. DiscoveryAgent integrates step cap and confidence escalation
# ---------------------------------------------------------------------------


class TestDiscoveryAgentHardCapIntegration:
    def test_agent_should_escalate_at_step_cap(self) -> None:
        agent = DiscoveryAgent(hard_cap_steps=20)
        agent.state.current_confidence = 0.95  # high confidence
        for _ in range(20):
            agent.step_counter.increment("t")
        # Even with high confidence, step exhaustion triggers should_escalate
        assert agent.should_escalate() is True
        assert "exhausted" in agent.state.escalation_reason.lower()

    def test_evaluate_confidence_at_cap_low(self) -> None:
        agent = DiscoveryAgent(hard_cap_steps=20)
        _exhaust(agent.step_counter)

        esc = agent.evaluate_confidence_at_cap(_signals(0.40))
        assert esc is not None
        assert esc.should_escalate is True
        assert agent.last_escalation is esc
        assert agent.state.escalation_reason is not None

    def test_evaluate_confidence_at_cap_high(self) -> None:
        agent = DiscoveryAgent(hard_cap_steps=20)
        _exhaust(agent.step_counter)

        esc = agent.evaluate_confidence_at_cap(_signals(0.80))
        assert esc is None
        assert agent.last_escalation is None

    def test_handoff_includes_step_budget_and_escalation(self) -> None:
        agent = DiscoveryAgent(hard_cap_steps=20)
        _exhaust(agent.step_counter)
        agent.evaluate_confidence_at_cap(_signals(0.35))

        summary = agent.get_handoff_summary()
        assert summary["step_budget"]["steps_taken"] == 20
        assert summary["step_budget"]["hard_cap"] == 20
        assert summary["step_budget"]["is_exhausted"] is True
        assert "escalation" in summary
        assert summary["escalation"]["should_escalate"] is True

    def test_check_threshold_mid_exploration(self) -> None:
        agent = DiscoveryAgent(hard_cap_steps=20)
        agent.step_counter.increment("tool")
        result = agent.check_threshold(_signals(0.30))
        assert result.can_continue is True
        assert result.needs_escalation is False


# ---------------------------------------------------------------------------
# 9. ConfidenceTracker tracks history and escalates at cap
# ---------------------------------------------------------------------------


class TestTrackerFullSessionAtCap20:
    def test_20_step_session_low_confidence_escalates(self) -> None:
        counter = StepCounter(hard_cap=20)
        tracker = ConfidenceTracker(step_counter=counter)

        # Simulate 20 steps with persistently low confidence
        for i in range(20):
            counter.increment(f"tool_{i}")
            tracker.record_snapshot(_signals(0.25 + i * 0.01))

        escalation = tracker.check_at_cap(_signals(0.44))
        assert escalation is not None
        assert escalation.should_escalate is True
        assert escalation.confidence_score < 0.60
        assert escalation.steps_taken == 20
        assert escalation.step_cap == 20

        # Metadata enriched with tracking info
        assert "tracking" in escalation.metadata
        assert escalation.metadata["tracking"]["history_length"] == 20
        assert escalation.metadata["tracking"]["persistent_low_confidence"] is True

    def test_20_step_session_high_confidence_no_escalation(self) -> None:
        counter = StepCounter(hard_cap=20)
        tracker = ConfidenceTracker(step_counter=counter)

        for i in range(20):
            counter.increment(f"tool_{i}")
            tracker.record_snapshot(_signals(0.70 + i * 0.01))

        escalation = tracker.check_at_cap(_signals(0.89))
        assert escalation is None

    def test_20_step_session_confidence_recovers(self) -> None:
        counter = StepCounter(hard_cap=20)
        tracker = ConfidenceTracker(step_counter=counter)

        # Start low, recover above 60%
        for i in range(20):
            counter.increment(f"tool_{i}")
            score = 0.20 + i * 0.04  # 0.20 -> 0.96
            tracker.record_snapshot(_signals(min(score, 1.0)))

        escalation = tracker.check_at_cap(_signals(0.80))
        assert escalation is None
        assert tracker.was_ever_above_threshold is True


# ---------------------------------------------------------------------------
# 10. Orchestrator routes discovery escalation correctly
# ---------------------------------------------------------------------------


class TestOrchestratorHandlesDiscoveryEscalation:
    def test_low_confidence_escalation_to_orchestrator(self) -> None:
        """Default route: orchestrator for user clarification."""
        agent = DiscoveryAgent(hard_cap_steps=20)
        _exhaust(agent.step_counter)
        esc = agent.evaluate_confidence_at_cap(_signals(0.40))
        assert esc is not None

        hub = _make_hub()
        state = RunState(request="run tests", phase=RunPhase.DISCOVERY)
        new_phase = hub.handle_escalation(state, esc, "discovery")

        assert len(state.escalations) == 1
        record = state.escalations[0]
        assert record.source_agent == "discovery"
        assert record.confidence_score < 0.60
        assert record.steps_taken == 20

    def test_structural_issue_routes_to_troubleshooter(self) -> None:
        counter = StepCounter(hard_cap=20)
        evaluator = ConfidenceThresholdEvaluator(
            step_counter=counter,
            structural_issue_indicators=["pytest_in_pyproject"],
        )
        _exhaust(counter)

        signals = [
            ConfidenceSignal(
                name="pytest_in_pyproject", weight=0.9, score=0.3,
                evidence={"file": "pyproject.toml"},
            ),
        ]
        result = evaluator.evaluate(signals)
        assert result.escalation.target == EscalationTarget.TROUBLESHOOTER

        hub = _make_hub()
        state = RunState(request="run tests", phase=RunPhase.DISCOVERY)
        new_phase = hub.handle_escalation(state, result.escalation, "discovery")

        assert new_phase == RunPhase.TROUBLESHOOTING
        assert state.phase == RunPhase.TROUBLESHOOTING

    def test_escalation_summary_captures_all(self) -> None:
        hub = _make_hub()
        state = RunState(request="run tests", phase=RunPhase.DISCOVERY)

        # Simulate two escalations
        agent = DiscoveryAgent(hard_cap_steps=20)
        _exhaust(agent.step_counter)
        esc = agent.evaluate_confidence_at_cap(_signals(0.35))
        hub.handle_escalation(state, esc, "discovery")

        summary = hub.get_escalation_summary(state)
        assert summary["total_escalations"] == 1
        assert summary["needs_user_clarification"] is True


# ---------------------------------------------------------------------------
# 11. Full end-to-end: 20-step discovery -> low confidence -> escalation
# ---------------------------------------------------------------------------


class TestEndToEndHardCapEscalation:
    def test_complete_flow(self) -> None:
        """Simulate the full AC 4 scenario end-to-end:
        1. Discovery agent explores for 20 steps
        2. Confidence stays below 60%
        3. Step cap triggers budget exhaustion
        4. Threshold evaluator detects low confidence
        5. Orchestrator receives and records escalation
        """
        # --- Setup ---
        agent = DiscoveryAgent(hard_cap_steps=20)
        tracker = ConfidenceTracker(
            step_counter=agent.step_counter,
            threshold_evaluator=agent.threshold_evaluator,
        )

        # --- Simulate 20 discovery steps with low confidence ---
        for i in range(20):
            allowed = agent.step_counter.increment(f"discovery_tool_{i}")
            assert allowed is True
            score = 0.30 + (i * 0.005)  # slowly improving: 0.30 -> 0.395
            tracker.record_snapshot(_signals(score))

        # --- Verify budget exhaustion ---
        assert agent.step_counter.is_exhausted is True
        assert agent.step_counter.remaining == 0
        assert agent.step_counter.steps_taken == 20
        assert agent.should_escalate() is True

        # --- 21st call is rejected ---
        assert agent.step_counter.increment("extra_tool") is False

        # --- Check confidence at cap ---
        final_signals = _signals(0.395)
        escalation = tracker.check_at_cap(final_signals)
        assert escalation is not None
        assert escalation.should_escalate is True
        assert escalation.confidence_score < 0.60
        assert escalation.steps_taken == 20
        assert escalation.step_cap == 20

        # --- Orchestrator handles escalation ---
        hub = _make_hub()
        state = RunState(request="run all python tests", phase=RunPhase.DISCOVERY)
        new_phase = hub.handle_escalation(state, escalation, "discovery")

        assert len(state.escalations) == 1
        record = state.escalations[0]
        assert record.source_agent == "discovery"
        assert record.confidence_score < 0.60
        assert record.steps_taken == 20

        # --- Verify tracking metadata ---
        assert "tracking" in escalation.metadata
        tracking = escalation.metadata["tracking"]
        assert tracking["history_length"] == 20
        assert tracking["persistent_low_confidence"] is True
        assert tracking["was_ever_above_threshold"] is False

    def test_complete_flow_confidence_sufficient(self) -> None:
        """When confidence reaches >= 60% at cap, no escalation needed."""
        agent = DiscoveryAgent(hard_cap_steps=20)
        tracker = ConfidenceTracker(
            step_counter=agent.step_counter,
            threshold_evaluator=agent.threshold_evaluator,
        )

        for i in range(20):
            agent.step_counter.increment(f"tool_{i}")
            score = 0.40 + (i * 0.02)  # 0.40 -> 0.78
            tracker.record_snapshot(_signals(score))

        assert agent.step_counter.is_exhausted is True

        escalation = tracker.check_at_cap(_signals(0.78))
        assert escalation is None  # No escalation needed
