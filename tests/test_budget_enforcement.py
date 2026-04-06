"""Tests for autonomy budget configuration and enforcement.

Covers:
- AgentBudget validation and properties
- AgentBudgetConfig per-agent resolution
- BudgetTracker lifecycle (begin/end cycle, token recording, check)
- BudgetGuard pre/post checks and escalation
- BudgetExceededError semantics
- Default budget presets
- Orchestrator hub integration with budget enforcement
"""

from __future__ import annotations

import time

import pytest

from test_runner.agents.base import AgentRole
from test_runner.autonomy.budget import (
    AgentBudget,
    AgentBudgetConfig,
    BudgetExceededError,
    BudgetExceededReason,
    BudgetGuard,
    BudgetStatus,
    BudgetTracker,
    default_budget_config,
)


# ---------------------------------------------------------------------------
# AgentBudget
# ---------------------------------------------------------------------------


class TestAgentBudget:
    def test_defaults_are_unlimited(self) -> None:
        b = AgentBudget()
        assert b.is_unlimited
        assert not b.has_iteration_limit
        assert not b.has_token_limit
        assert not b.has_time_limit

    def test_limits_enabled(self) -> None:
        b = AgentBudget(max_iterations=10, max_token_spend=1000, max_wall_clock_seconds=60.0)
        assert not b.is_unlimited
        assert b.has_iteration_limit
        assert b.has_token_limit
        assert b.has_time_limit

    def test_negative_iterations_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            AgentBudget(max_iterations=-1)

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_token_spend"):
            AgentBudget(max_token_spend=-1)

    def test_negative_wall_clock_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_wall_clock_seconds"):
            AgentBudget(max_wall_clock_seconds=-1.0)

    def test_to_dict(self) -> None:
        b = AgentBudget(max_iterations=5, max_token_spend=100, max_wall_clock_seconds=30.0)
        d = b.to_dict()
        assert d == {
            "max_iterations": 5,
            "max_token_spend": 100,
            "max_wall_clock_seconds": 30.0,
        }


# ---------------------------------------------------------------------------
# AgentBudgetConfig
# ---------------------------------------------------------------------------


class TestAgentBudgetConfig:
    def test_global_default_used_when_no_override(self) -> None:
        default = AgentBudget(max_iterations=10)
        config = AgentBudgetConfig(global_default=default)
        assert config.budget_for(AgentRole.DISCOVERY) == default
        assert config.budget_for(AgentRole.EXECUTOR) == default

    def test_per_agent_override(self) -> None:
        default = AgentBudget(max_iterations=10)
        discovery = AgentBudget(max_iterations=5)
        config = AgentBudgetConfig(
            global_default=default,
            per_agent={AgentRole.DISCOVERY: discovery},
        )
        assert config.budget_for(AgentRole.DISCOVERY) == discovery
        assert config.budget_for(AgentRole.EXECUTOR) == default

    def test_to_dict(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=10),
            per_agent={AgentRole.REPORTER: AgentBudget(max_iterations=3)},
        )
        d = config.to_dict()
        assert "global_default" in d
        assert "reporter" in d["per_agent"]


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------


class TestBudgetTracker:
    def test_initial_state_no_usage(self) -> None:
        tracker = BudgetTracker()
        status = tracker.check(AgentRole.DISCOVERY)
        assert status.iterations_used == 0
        assert status.tokens_used == 0
        assert not status.exceeded

    def test_iteration_tracking(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=3),
        )
        tracker = BudgetTracker(config)

        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY)
        assert tracker.check(AgentRole.DISCOVERY).iterations_used == 1
        assert not tracker.is_exceeded(AgentRole.DISCOVERY)

        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY)
        assert tracker.check(AgentRole.DISCOVERY).iterations_used == 2

        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY)
        # 3 iterations used == max_iterations => exceeded
        assert tracker.is_exceeded(AgentRole.DISCOVERY)
        status = tracker.check(AgentRole.DISCOVERY)
        assert status.exceeded
        assert BudgetExceededReason.ITERATIONS in status.exceeded_reasons

    def test_token_tracking(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_token_spend=1000),
        )
        tracker = BudgetTracker(config)

        tracker.record_tokens(AgentRole.EXECUTOR, 500)
        assert not tracker.is_exceeded(AgentRole.EXECUTOR)

        tracker.record_tokens(AgentRole.EXECUTOR, 500)
        assert tracker.is_exceeded(AgentRole.EXECUTOR)
        status = tracker.check(AgentRole.EXECUTOR)
        assert BudgetExceededReason.TOKEN_SPEND in status.exceeded_reasons

    def test_token_tracking_via_end_cycle(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_token_spend=100),
        )
        tracker = BudgetTracker(config)
        tracker.begin_cycle(AgentRole.REPORTER)
        tracker.end_cycle(AgentRole.REPORTER, tokens=100)
        assert tracker.is_exceeded(AgentRole.REPORTER)

    def test_wall_clock_tracking(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_wall_clock_seconds=0.05),
        )
        tracker = BudgetTracker(config)
        tracker.begin_cycle(AgentRole.DISCOVERY)
        time.sleep(0.06)
        tracker.end_cycle(AgentRole.DISCOVERY)
        assert tracker.is_exceeded(AgentRole.DISCOVERY)
        status = tracker.check(AgentRole.DISCOVERY)
        assert BudgetExceededReason.WALL_CLOCK in status.exceeded_reasons

    def test_in_progress_wall_clock_counted(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_wall_clock_seconds=0.01),
        )
        tracker = BudgetTracker(config)
        tracker.begin_cycle(AgentRole.DISCOVERY)
        time.sleep(0.02)
        # Check while cycle is still in progress
        status = tracker.check(AgentRole.DISCOVERY)
        assert status.exceeded
        assert status.wall_clock_used >= 0.01

    def test_negative_tokens_rejected(self) -> None:
        tracker = BudgetTracker()
        with pytest.raises(ValueError, match="tokens must be >= 0"):
            tracker.record_tokens(AgentRole.DISCOVERY, -1)

    def test_reset_single_agent(self) -> None:
        tracker = BudgetTracker()
        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY, tokens=100)
        tracker.begin_cycle(AgentRole.EXECUTOR)
        tracker.end_cycle(AgentRole.EXECUTOR, tokens=200)

        tracker.reset(AgentRole.DISCOVERY)
        assert tracker.check(AgentRole.DISCOVERY).iterations_used == 0
        assert tracker.check(AgentRole.EXECUTOR).iterations_used == 1

    def test_reset_all(self) -> None:
        tracker = BudgetTracker()
        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY)
        tracker.begin_cycle(AgentRole.EXECUTOR)
        tracker.end_cycle(AgentRole.EXECUTOR)

        tracker.reset()
        assert tracker.check(AgentRole.DISCOVERY).iterations_used == 0
        assert tracker.check(AgentRole.EXECUTOR).iterations_used == 0

    def test_all_statuses(self) -> None:
        tracker = BudgetTracker()
        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY)
        tracker.begin_cycle(AgentRole.EXECUTOR)
        tracker.end_cycle(AgentRole.EXECUTOR)

        statuses = tracker.all_statuses()
        assert AgentRole.DISCOVERY in statuses
        assert AgentRole.EXECUTOR in statuses

    def test_snapshot(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=10),
        )
        tracker = BudgetTracker(config)
        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY, tokens=50)
        snap = tracker.snapshot()
        assert "config" in snap
        assert "agents" in snap
        assert "discovery" in snap["agents"]

    def test_multiple_reasons_can_be_exceeded(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(
                max_iterations=1,
                max_token_spend=50,
            ),
        )
        tracker = BudgetTracker(config)
        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY, tokens=100)

        status = tracker.check(AgentRole.DISCOVERY)
        assert status.exceeded
        assert BudgetExceededReason.ITERATIONS in status.exceeded_reasons
        assert BudgetExceededReason.TOKEN_SPEND in status.exceeded_reasons


# ---------------------------------------------------------------------------
# BudgetStatus
# ---------------------------------------------------------------------------


class TestBudgetStatus:
    def test_remaining_with_limits(self) -> None:
        status = BudgetStatus(
            role=AgentRole.DISCOVERY,
            budget=AgentBudget(max_iterations=10, max_token_spend=1000, max_wall_clock_seconds=60.0),
            iterations_used=3,
            tokens_used=200,
            wall_clock_used=15.0,
        )
        assert status.iterations_remaining == 7
        assert status.tokens_remaining == 800
        assert status.wall_clock_remaining == 45.0

    def test_remaining_unlimited(self) -> None:
        status = BudgetStatus(
            role=AgentRole.DISCOVERY,
            budget=AgentBudget(),
            iterations_used=100,
        )
        assert status.iterations_remaining is None
        assert status.tokens_remaining is None
        assert status.wall_clock_remaining is None

    def test_remaining_does_not_go_negative(self) -> None:
        status = BudgetStatus(
            role=AgentRole.DISCOVERY,
            budget=AgentBudget(max_iterations=5),
            iterations_used=10,
        )
        assert status.iterations_remaining == 0

    def test_summary(self) -> None:
        status = BudgetStatus(
            role=AgentRole.DISCOVERY,
            budget=AgentBudget(max_iterations=5),
            iterations_used=5,
            exceeded=True,
            exceeded_reasons=(BudgetExceededReason.ITERATIONS,),
        )
        s = status.summary()
        assert s["exceeded"] is True
        assert "exceeded_reasons" in s
        assert s["max_iterations"] == 5


# ---------------------------------------------------------------------------
# BudgetGuard
# ---------------------------------------------------------------------------


class TestBudgetGuard:
    def test_pre_check_passes_within_budget(self) -> None:
        tracker = BudgetTracker(AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=10),
        ))
        guard = BudgetGuard(tracker)
        status = guard.pre_check(AgentRole.DISCOVERY)
        assert not status.exceeded

    def test_pre_check_raises_when_exceeded(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=1),
        )
        tracker = BudgetTracker(config)
        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY)

        guard = BudgetGuard(tracker)
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.pre_check(AgentRole.DISCOVERY)
        assert exc_info.value.status.exceeded
        assert "iterations_exceeded" in str(exc_info.value)

    def test_post_check_returns_status(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=1),
        )
        tracker = BudgetTracker(config)
        tracker.begin_cycle(AgentRole.DISCOVERY)
        tracker.end_cycle(AgentRole.DISCOVERY)

        guard = BudgetGuard(tracker)
        status = guard.post_check(AgentRole.DISCOVERY)
        assert status.exceeded

    def test_is_within_budget(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=5),
        )
        tracker = BudgetTracker(config)
        guard = BudgetGuard(tracker)

        assert guard.is_within_budget(AgentRole.DISCOVERY)
        for _ in range(5):
            tracker.begin_cycle(AgentRole.DISCOVERY)
            tracker.end_cycle(AgentRole.DISCOVERY)
        assert not guard.is_within_budget(AgentRole.DISCOVERY)

    def test_remaining_summary(self) -> None:
        config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=10, max_token_spend=5000),
        )
        tracker = BudgetTracker(config)
        tracker.begin_cycle(AgentRole.EXECUTOR)
        tracker.end_cycle(AgentRole.EXECUTOR, tokens=1000)

        guard = BudgetGuard(tracker)
        summary = guard.remaining_summary(AgentRole.EXECUTOR)
        assert summary["role"] == "executor"
        assert summary["iterations_remaining"] == 9
        assert summary["tokens_remaining"] == 4000


# ---------------------------------------------------------------------------
# BudgetExceededError
# ---------------------------------------------------------------------------


class TestBudgetExceededError:
    def test_error_message_contains_details(self) -> None:
        status = BudgetStatus(
            role=AgentRole.TROUBLESHOOTER,
            budget=AgentBudget(max_iterations=3),
            iterations_used=3,
            exceeded=True,
            exceeded_reasons=(BudgetExceededReason.ITERATIONS,),
        )
        err = BudgetExceededError(status)
        assert "troubleshooter" in str(err)
        assert "iterations_exceeded" in str(err)
        assert err.status is status

    def test_multiple_reasons_in_message(self) -> None:
        status = BudgetStatus(
            role=AgentRole.DISCOVERY,
            budget=AgentBudget(max_iterations=1, max_token_spend=100),
            iterations_used=1,
            tokens_used=200,
            exceeded=True,
            exceeded_reasons=(
                BudgetExceededReason.ITERATIONS,
                BudgetExceededReason.TOKEN_SPEND,
            ),
        )
        err = BudgetExceededError(status)
        msg = str(err)
        assert "iterations_exceeded" in msg
        assert "token_spend_exceeded" in msg


# ---------------------------------------------------------------------------
# Default budget presets
# ---------------------------------------------------------------------------


class TestDefaultBudgetConfig:
    def test_all_roles_have_overrides(self) -> None:
        config = default_budget_config()
        for role in AgentRole:
            budget = config.budget_for(role)
            assert not budget.is_unlimited, f"{role.value} should have limits"

    def test_discovery_has_iteration_limit(self) -> None:
        config = default_budget_config()
        budget = config.budget_for(AgentRole.DISCOVERY)
        assert budget.has_iteration_limit
        assert budget.max_iterations > 0

    def test_executor_has_longer_wall_clock(self) -> None:
        config = default_budget_config()
        exec_budget = config.budget_for(AgentRole.EXECUTOR)
        disc_budget = config.budget_for(AgentRole.DISCOVERY)
        assert exec_budget.max_wall_clock_seconds > disc_budget.max_wall_clock_seconds

    def test_reporter_has_lower_limits(self) -> None:
        config = default_budget_config()
        reporter = config.budget_for(AgentRole.REPORTER)
        discovery = config.budget_for(AgentRole.DISCOVERY)
        assert reporter.max_iterations < discovery.max_iterations
        assert reporter.max_token_spend < discovery.max_token_spend

    def test_serializable(self) -> None:
        config = default_budget_config()
        d = config.to_dict()
        assert "global_default" in d
        assert "per_agent" in d
        assert len(d["per_agent"]) == len(AgentRole)


# ---------------------------------------------------------------------------
# Orchestrator hub budget integration
# ---------------------------------------------------------------------------


class TestOrchestratorBudgetIntegration:
    """Test that the orchestrator hub properly integrates budget enforcement."""

    def test_hub_has_budget_tracker(self) -> None:
        from test_runner.config import Config
        from test_runner.orchestrator.hub import OrchestratorHub

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        hub = OrchestratorHub(config)
        assert hub.budget_tracker is not None
        assert hub.budget_guard is not None

    def test_hub_accepts_custom_budget_config(self) -> None:
        from test_runner.config import Config
        from test_runner.orchestrator.hub import OrchestratorHub

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        custom = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=2),
        )
        hub = OrchestratorHub(config, budget_config=custom)
        budget = hub.budget_tracker.config.budget_for(AgentRole.DISCOVERY)
        assert budget.max_iterations == 2

    def test_hub_budget_snapshot(self) -> None:
        from test_runner.config import Config
        from test_runner.orchestrator.hub import OrchestratorHub

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        hub = OrchestratorHub(config)
        snap = hub.get_budget_snapshot()
        assert "config" in snap
        assert "agents" in snap

    def test_enforce_budget_raises_on_exceeded(self) -> None:
        from test_runner.config import Config
        from test_runner.orchestrator.hub import OrchestratorHub, RunState

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        budget_config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=0),
            per_agent={
                AgentRole.DISCOVERY: AgentBudget(max_iterations=1),
            },
        )
        hub = OrchestratorHub(config, budget_config=budget_config)

        # Exhaust the discovery budget
        hub.budget_tracker.begin_cycle(AgentRole.DISCOVERY)
        hub.budget_tracker.end_cycle(AgentRole.DISCOVERY)

        state = RunState(request="test")
        with pytest.raises(BudgetExceededError):
            hub._enforce_budget_or_escalate(state, AgentRole.DISCOVERY)

    def test_enforce_budget_passes_within_limits(self) -> None:
        from test_runner.config import Config
        from test_runner.orchestrator.hub import OrchestratorHub, RunState

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        hub = OrchestratorHub(config)
        state = RunState(request="test")
        # Should not raise — default budgets are generous
        status = hub._enforce_budget_or_escalate(state, AgentRole.DISCOVERY)
        assert not status.exceeded

    def test_begin_end_budget_cycle(self) -> None:
        from test_runner.config import Config
        from test_runner.orchestrator.hub import OrchestratorHub

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        hub = OrchestratorHub(config)

        hub._begin_budget_cycle(AgentRole.EXECUTOR)
        status = hub._end_budget_cycle(AgentRole.EXECUTOR, tokens=500)
        assert status.iterations_used == 1
        assert status.tokens_used == 500


@pytest.mark.asyncio
class TestOrchestratorRunBudgetEnforcement:
    """Test that the orchestrator's run() method handles budget errors."""

    async def test_run_handles_budget_exceeded_gracefully(self) -> None:
        from test_runner.config import Config
        from test_runner.orchestrator.hub import OrchestratorHub, RunPhase

        config = Config(llm_base_url="http://test", api_key="test", model_id="test")
        # Set discovery to max_iterations=0 so it's immediately exceeded
        budget_config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=100),
            per_agent={
                AgentRole.DISCOVERY: AgentBudget(
                    max_iterations=0,  # unlimited (won't trigger)
                    max_token_spend=1,  # will be exceeded by any use
                ),
            },
        )
        hub = OrchestratorHub(config, budget_config=budget_config)
        # Pre-exhaust the discovery token budget
        hub.budget_tracker.record_tokens(AgentRole.DISCOVERY, 1)

        state = await hub.run("run tests")
        assert state.phase == RunPhase.FAILED
        assert any("budget" in e.lower() or "Budget" in e for e in state.errors)
        assert any(e.reason == "budget_exceeded" for e in state.escalations)
