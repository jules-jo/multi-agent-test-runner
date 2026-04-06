"""Tests for the TroubleshooterAgent class and its tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from test_runner.agents.base import AgentRole
from test_runner.agents.troubleshooter.agent import (
    TROUBLESHOOTER_HARD_CAP,
    TROUBLESHOOTER_INSTRUCTIONS,
    TroubleshooterAgent,
    create_troubleshooter_agent,
)
from test_runner.models.confidence import ConfidenceTier
from test_runner.tools.troubleshooter_tools import (
    TROUBLESHOOTER_TOOLS,
    _check_logs_impl,
    _inspect_env_impl,
    _list_processes_impl,
    _read_file_impl,
)


# ---------------------------------------------------------------------------
# TroubleshooterAgent class tests
# ---------------------------------------------------------------------------


class TestTroubleshooterAgentInit:
    """Test TroubleshooterAgent initialization and properties."""

    def test_default_init(self):
        agent = TroubleshooterAgent()
        assert agent.role == AgentRole.TROUBLESHOOTER
        assert agent.name == "troubleshooter-agent"
        assert agent.hard_cap_steps == TROUBLESHOOTER_HARD_CAP
        assert agent.auto_fix_enabled is False
        assert agent.diagnosis is None

    def test_custom_init(self):
        agent = TroubleshooterAgent(
            hard_cap_steps=10,
            high_threshold=0.90,
            low_threshold=0.50,
            auto_fix_enabled=False,
        )
        assert agent.hard_cap_steps == 10
        assert agent.high_threshold == 0.90
        assert agent.low_threshold == 0.50

    def test_instructions_contain_hard_cap(self):
        agent = TroubleshooterAgent(hard_cap_steps=42)
        assert "42" in agent.instructions

    def test_instructions_mention_read_only(self):
        agent = TroubleshooterAgent()
        assert "READ-ONLY" in agent.instructions

    def test_instructions_mention_diagnose(self):
        agent = TroubleshooterAgent()
        assert "diagnose" in agent.instructions.lower()

    def test_get_tools_returns_four(self):
        agent = TroubleshooterAgent()
        tools = agent.get_tools()
        assert len(tools) == 4


class TestTroubleshooterAgentState:
    """Test state management and escalation."""

    def test_reset_state(self):
        agent = TroubleshooterAgent()
        agent.state.record_step(0.9)
        agent.record_diagnosis(
            root_cause="test",
            evidence=["e1"],
            confidence=0.9,
            proposed_fix=["fix1"],
        )
        agent.reset_state()
        assert agent.state.steps_taken == 0
        assert agent.diagnosis is None
        assert agent.step_counter.steps_taken == 0

    def test_should_escalate_on_budget_exhaustion(self):
        agent = TroubleshooterAgent(hard_cap_steps=2)
        agent._step_counter.increment("t1", "test")
        agent._step_counter.increment("t2", "test")
        assert agent.should_escalate() is True

    def test_should_escalate_on_low_confidence(self):
        agent = TroubleshooterAgent()
        agent.state.current_confidence = 0.2  # Below low_threshold
        assert agent.should_escalate() is True

    def test_should_not_escalate_on_high_confidence(self):
        agent = TroubleshooterAgent()
        agent.state.current_confidence = 0.9
        assert agent.should_escalate() is False

    def test_confidence_tier_high(self):
        agent = TroubleshooterAgent()
        agent.state.current_confidence = 0.85
        assert agent.confidence_tier() == ConfidenceTier.HIGH

    def test_confidence_tier_medium(self):
        agent = TroubleshooterAgent()
        agent.state.current_confidence = 0.55
        assert agent.confidence_tier() == ConfidenceTier.MEDIUM

    def test_confidence_tier_low(self):
        agent = TroubleshooterAgent()
        agent.state.current_confidence = 0.2
        assert agent.confidence_tier() == ConfidenceTier.LOW


class TestTroubleshooterDiagnosis:
    """Test diagnosis recording and handoff."""

    def test_record_diagnosis(self):
        agent = TroubleshooterAgent()
        diag = agent.record_diagnosis(
            root_cause="Missing dependency",
            evidence=["ImportError in stderr", "No 'requests' in requirements.txt"],
            confidence=0.85,
            proposed_fix=["Add requests to requirements.txt", "Run pip install"],
            alternative_causes=["Version mismatch"],
        )
        assert diag["root_cause"] == "Missing dependency"
        assert len(diag["evidence"]) == 2
        assert diag["confidence"] == 0.85
        assert len(diag["proposed_fix"]) == 2
        assert diag["alternative_causes"] == ["Version mismatch"]
        assert diag["auto_fix_enabled"] is False
        assert agent.diagnosis is diag
        assert agent.state.current_confidence == 0.85

    def test_handoff_summary_includes_diagnosis(self):
        agent = TroubleshooterAgent()
        agent.record_diagnosis(
            root_cause="Config error",
            evidence=["missing key"],
            confidence=0.7,
            proposed_fix=["add key"],
        )
        summary = agent.get_handoff_summary()
        assert summary["agent"] == "troubleshooter-agent"
        assert summary["role"] == "troubleshooter"
        assert "diagnosis" in summary
        assert summary["diagnosis"]["root_cause"] == "Config error"
        assert "step_budget" in summary
        assert summary["auto_fix_enabled"] is False

    def test_handoff_summary_without_diagnosis(self):
        agent = TroubleshooterAgent()
        summary = agent.get_handoff_summary()
        assert "diagnosis" not in summary


# ---------------------------------------------------------------------------
# Tool implementation tests
# ---------------------------------------------------------------------------


class TestReadFileImpl:
    """Test the read_file tool implementation."""

    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text("line1\nline2\nline3\n")
        result = _read_file_impl(str(f))
        assert result["returned_lines"] == 3
        assert "line1" in result["content"]
        assert result["truncated"] is False

    def test_read_nonexistent_file(self):
        result = _read_file_impl("/nonexistent/path.txt")
        assert "error" in result

    def test_read_file_truncation(self, tmp_path):
        f = tmp_path / "big.log"
        f.write_text("\n".join(f"line{i}" for i in range(1000)))
        result = _read_file_impl(str(f), max_lines=10)
        assert result["returned_lines"] == 10
        assert result["truncated"] is True

    def test_read_directory_returns_error(self, tmp_path):
        result = _read_file_impl(str(tmp_path))
        assert "error" in result


class TestCheckLogsImpl:
    """Test the check_logs tool implementation."""

    def test_tail_behavior(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("\n".join(f"log line {i}" for i in range(200)))
        result = _check_logs_impl(str(f), tail_lines=10)
        assert result["returned_lines"] == 10
        assert "log line 199" in result["content"]

    def test_pattern_filter(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("INFO normal\nERROR failure\nINFO ok\nERROR crash\n")
        result = _check_logs_impl(str(f), pattern="ERROR")
        assert result["returned_lines"] == 2
        assert "failure" in result["content"]
        assert "crash" in result["content"]

    def test_nonexistent_log(self):
        result = _check_logs_impl("/nonexistent/app.log")
        assert "error" in result


class TestInspectEnvImpl:
    """Test the inspect_env tool implementation."""

    def test_returns_variables(self):
        result = _inspect_env_impl()
        assert "variables" in result
        assert result["total_vars"] > 0

    def test_filter_prefix(self):
        os.environ["TEST_TROUBLESHOOTER_VAR"] = "hello"
        try:
            result = _inspect_env_impl(filter_prefix="TEST_TROUBLESHOOTER")
            assert "TEST_TROUBLESHOOTER_VAR" in result["variables"]
        finally:
            del os.environ["TEST_TROUBLESHOOTER_VAR"]

    def test_masks_sensitive_values(self):
        os.environ["MY_SECRET_KEY"] = "supersecret"
        try:
            result = _inspect_env_impl()
            assert result["variables"].get("MY_SECRET_KEY") == "****MASKED****"
        finally:
            del os.environ["MY_SECRET_KEY"]

    def test_includes_python_info(self):
        result = _inspect_env_impl(include_python=True)
        assert "python" in result
        assert "version" in result["python"]

    def test_excludes_python_info(self):
        result = _inspect_env_impl(include_python=False)
        assert "python" not in result


class TestListProcessesImpl:
    """Test the list_processes tool implementation."""

    def test_returns_processes(self):
        result = _list_processes_impl()
        assert "processes" in result
        # Should find at least one process on any running system
        assert result["total_shown"] > 0

    def test_filter_pattern(self):
        result = _list_processes_impl(filter_pattern="python")
        # All returned processes should contain "python"
        for proc in result["processes"]:
            assert "python" in proc["command"].lower() or "python" in proc["user"].lower()

    def test_limit(self):
        result = _list_processes_impl(limit=3)
        assert result["total_shown"] <= 3


# ---------------------------------------------------------------------------
# Budget-tracked tool tests
# ---------------------------------------------------------------------------


class TestBudgetTrackedTools:
    """Test that tracked tools respect the step budget."""

    def test_tools_count_steps(self, tmp_path):
        agent = TroubleshooterAgent(hard_cap_steps=5)
        tools = agent.get_tools()
        # Step counter starts at 0
        assert agent.step_counter.steps_taken == 0

    def test_budget_exhaustion_returns_error(self):
        agent = TroubleshooterAgent(hard_cap_steps=1)
        # Use up the budget
        agent.step_counter.increment("manual", "test")
        # Now all tools should report budget exceeded
        assert agent.step_counter.is_exhausted


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestCreateTroubleshooterAgent:
    """Test the create_troubleshooter_agent factory."""

    def test_creates_agent_with_defaults(self):
        agent = create_troubleshooter_agent()
        assert agent.name == "troubleshooter-agent"
        assert len(agent.tools) == 4

    def test_creates_agent_with_config(self):
        from test_runner.config import Config

        config = Config(
            llm_base_url="http://test",
            api_key="key",
            model_id="test-model",
        )
        agent = create_troubleshooter_agent(config=config)
        assert agent.model == "test-model"

    def test_creates_agent_with_custom_cap(self):
        agent = create_troubleshooter_agent(hard_cap_steps=10)
        assert agent.name == "troubleshooter-agent"


# ---------------------------------------------------------------------------
# Registration in agents __init__ tests
# ---------------------------------------------------------------------------


class TestTroubleshooterRegistration:
    """Test that TroubleshooterAgent is properly exported from agents package."""

    def test_importable_from_agents(self):
        from test_runner.agents import TroubleshooterAgent as TA
        assert TA is not None

    def test_factory_importable_from_agents(self):
        from test_runner.agents import create_troubleshooter_agent as factory
        assert factory is not None

    def test_in_all_exports(self):
        import test_runner.agents as agents_mod
        assert "TroubleshooterAgent" in agents_mod.__all__
        assert "create_troubleshooter_agent" in agents_mod.__all__

    def test_role_enum_has_troubleshooter(self):
        assert AgentRole.TROUBLESHOOTER.value == "troubleshooter"
