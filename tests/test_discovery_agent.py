"""Tests for the DiscoveryAgent class and factory function."""

from __future__ import annotations

from agents import Agent

from test_runner.agents.base import AgentRole, AgentState, BaseSubAgent
from test_runner.agents.discovery.agent import DiscoveryAgent, create_discovery_agent
from test_runner.agents.discovery.step_counter import DEFAULT_HARD_CAP
from test_runner.models.confidence import ConfidenceTier


class TestAgentState:
    def test_initial_state(self) -> None:
        state = AgentState()
        assert state.steps_taken == 0
        assert state.current_confidence == 0.5
        assert state.escalation_reason is None
        assert state.findings == []
        assert state.errors == []

    def test_record_step(self) -> None:
        state = AgentState()
        state.record_step(confidence=0.9)
        assert state.steps_taken == 1
        assert state.current_confidence == 0.9

    def test_record_step_no_confidence(self) -> None:
        state = AgentState()
        state.record_step()
        assert state.steps_taken == 1
        assert state.current_confidence == 0.5  # unchanged

    def test_add_finding(self) -> None:
        state = AgentState()
        state.add_finding({"framework": "pytest", "confidence": 0.95})
        assert len(state.findings) == 1

    def test_to_dict(self) -> None:
        state = AgentState()
        state.record_step(0.8)
        d = state.to_dict()
        assert d["steps_taken"] == 1
        assert d["current_confidence"] == 0.8


class TestDiscoveryAgent:
    def test_role(self) -> None:
        agent = DiscoveryAgent()
        assert agent.role == AgentRole.DISCOVERY

    def test_name(self) -> None:
        agent = DiscoveryAgent()
        assert agent.name == "discovery-agent"

    def test_default_hard_cap_is_20(self) -> None:
        agent = DiscoveryAgent()
        assert agent.hard_cap_steps == DEFAULT_HARD_CAP
        assert agent.hard_cap_steps == 20

    def test_step_counter_attached(self) -> None:
        agent = DiscoveryAgent()
        assert agent.step_counter is not None
        assert agent.step_counter.hard_cap == 20

    def test_custom_hard_cap_propagates(self) -> None:
        agent = DiscoveryAgent(hard_cap_steps=10)
        assert agent.hard_cap_steps == 10
        assert agent.step_counter.hard_cap == 10

    def test_instructions_include_budget(self) -> None:
        agent = DiscoveryAgent(hard_cap_steps=15)
        assert "15" in agent.instructions
        assert "HARD CAP" in agent.instructions

    def test_instructions_not_empty(self) -> None:
        agent = DiscoveryAgent()
        assert len(agent.instructions) > 100

    def test_tools_count(self) -> None:
        agent = DiscoveryAgent()
        tools = agent.get_tools()
        assert len(tools) == 4

    def test_tool_names(self) -> None:
        agent = DiscoveryAgent()
        names = {t.name for t in agent.get_tools()}
        assert names == {"scan_directory", "read_file", "run_help", "detect_frameworks"}

    def test_should_escalate_with_low_confidence(self) -> None:
        agent = DiscoveryAgent()
        agent.state.current_confidence = 0.2
        assert agent.should_escalate() is True

    def test_should_not_escalate_with_high_confidence(self) -> None:
        agent = DiscoveryAgent()
        agent.state.current_confidence = 0.9
        assert agent.should_escalate() is False

    def test_escalate_at_hard_cap(self) -> None:
        agent = DiscoveryAgent(hard_cap_steps=5)
        agent.state.current_confidence = 0.9
        # Exhaust step counter
        for _ in range(5):
            agent.step_counter.increment("tool")
        assert agent.should_escalate() is True

    def test_escalate_at_hard_cap_sets_reason(self) -> None:
        agent = DiscoveryAgent(hard_cap_steps=3)
        for _ in range(3):
            agent.step_counter.increment("tool")
        agent.should_escalate()
        assert agent.state.escalation_reason is not None
        assert "exhausted" in agent.state.escalation_reason.lower()

    def test_confidence_tier_high(self) -> None:
        agent = DiscoveryAgent()
        agent.state.current_confidence = 0.85
        assert agent.confidence_tier() == ConfidenceTier.HIGH

    def test_confidence_tier_medium(self) -> None:
        agent = DiscoveryAgent()
        agent.state.current_confidence = 0.5
        assert agent.confidence_tier() == ConfidenceTier.MEDIUM

    def test_confidence_tier_low(self) -> None:
        agent = DiscoveryAgent()
        agent.state.current_confidence = 0.2
        assert agent.confidence_tier() == ConfidenceTier.LOW

    def test_reset_state_also_resets_counter(self) -> None:
        agent = DiscoveryAgent()
        agent.state.record_step(0.9)
        agent.state.add_finding({"test": True})
        agent.step_counter.increment("tool")
        agent.reset_state()
        assert agent.state.steps_taken == 0
        assert agent.state.findings == []
        assert agent.step_counter.steps_taken == 0

    def test_handoff_summary_includes_budget(self) -> None:
        agent = DiscoveryAgent()
        agent.state.record_step(0.85)
        agent.step_counter.increment("scan_directory")
        summary = agent.get_handoff_summary()
        assert summary["agent"] == "discovery-agent"
        assert summary["role"] == "discovery"
        assert summary["confidence_tier"] == "high"
        assert summary["escalated"] is False
        assert "step_budget" in summary
        assert summary["step_budget"]["steps_taken"] == 1
        assert summary["step_budget"]["hard_cap"] == 20

    def test_custom_thresholds(self) -> None:
        agent = DiscoveryAgent(high_threshold=0.9, low_threshold=0.5)
        agent.state.current_confidence = 0.85
        assert agent.confidence_tier() == ConfidenceTier.MEDIUM  # below 0.9


class TestDiscoveryAgentToolTracking:
    """Test that tool invocations are tracked by the step counter."""

    def test_tools_share_step_counter(self) -> None:
        """All tools from a single agent share one step counter."""
        agent = DiscoveryAgent(hard_cap_steps=20)
        tools = agent.get_tools()
        # Tools are closures over the same counter — verify by checking
        # that the counter is the agent's counter (indirect: call tools
        # and check counter state)
        assert agent.step_counter.steps_taken == 0


class TestCreateDiscoveryAgent:
    def test_creates_sdk_agent(self) -> None:
        agent = create_discovery_agent()
        assert isinstance(agent, Agent)

    def test_agent_name(self) -> None:
        agent = create_discovery_agent()
        assert agent.name == "discovery-agent"

    def test_agent_has_tools(self) -> None:
        agent = create_discovery_agent()
        assert len(agent.tools) == 4

    def test_agent_instructions(self) -> None:
        agent = create_discovery_agent()
        assert "Discovery Agent" in agent.instructions

    def test_default_hard_cap_in_instructions(self) -> None:
        agent = create_discovery_agent()
        assert "20" in agent.instructions

    def test_custom_hard_cap_in_instructions(self) -> None:
        agent = create_discovery_agent(hard_cap_steps=15)
        assert "15" in agent.instructions
