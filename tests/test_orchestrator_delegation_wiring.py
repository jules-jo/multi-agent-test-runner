"""Tests for state management and budget enforcement wiring in orchestrator delegation.

Verifies Sub-AC 3:
- Each sub-agent call updates shared state (AgentStateStore) via delegation cycles
- Each sub-agent call is checked against its autonomy budget before and after execution
- Discovery delegation tracks budget + state correctly
- Executor delegation tracks budget + state correctly
- Budget exceeded during delegation produces proper escalation
- State store reflects correct agent status after each delegation phase
- Error paths correctly close delegation cycles and budget cycles
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from test_runner.agents.base import AgentRole
from test_runner.agents.discovery.agent import DiscoveryAgent
from test_runner.agents.parser import TestFramework, TestIntent
from test_runner.autonomy.budget import (
    AgentBudget,
    AgentBudgetConfig,
    BudgetExceededError,
    BudgetTracker,
)
from test_runner.config import Config
from test_runner.execution.command_translator import TestCommand
from test_runner.execution.executor import TaskAttemptRecord, TaskExecutor
from test_runner.execution.targets import ExecutionResult, ExecutionStatus
from test_runner.orchestrator.hub import (
    OrchestratorHub,
    RunPhase,
    RunState,
)
from test_runner.orchestrator.state_store import AgentStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> Config:
    return Config(
        llm_base_url="http://localhost:8080/v1",
        api_key="test-key",
        model_id="test-model",
    )


def _make_test_command(display: str = "pytest tests/") -> TestCommand:
    return TestCommand(
        command=["pytest", "tests/"],
        display=display,
        framework=TestFramework.PYTEST,
    )


def _make_intent_resolution(commands: list[TestCommand] | None = None):
    """Create a mock IntentResolution."""
    from test_runner.agents.intent_service import IntentResolution, ParseMode
    from test_runner.agents.parser import ParsedTestRequest

    cmds = commands or [_make_test_command()]
    parsed = ParsedTestRequest(
        intent=TestIntent.RUN,
        framework=TestFramework.PYTEST,
        confidence=0.85,
    )
    from test_runner.execution.command_translator import TranslationResult

    translation = TranslationResult(
        commands=cmds,
        warnings=[],
    )
    return IntentResolution(
        parsed_request=parsed,
        translation=translation,
        parse_mode_used=ParseMode.OFFLINE,
    )


def _make_execution_result(
    status: ExecutionStatus = ExecutionStatus.PASSED,
    stdout: str = "1 passed",
    stderr: str = "",
    exit_code: int = 0,
) -> ExecutionResult:
    return ExecutionResult(
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.5,
        command_display="pytest tests/",
    )


# ---------------------------------------------------------------------------
# Discovery delegation wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDiscoveryDelegationWiring:
    """Test that _delegate_to_discovery properly wires state + budget."""

    async def test_discovery_creates_delegation_cycle_in_state_store(self) -> None:
        """Discovery delegation should create and complete a cycle in state store."""
        config = _make_config()
        hub = OrchestratorHub(config)
        state = RunState(request="run all tests")
        resolution = _make_intent_resolution()

        await hub._delegate_to_discovery(state, resolution)

        # State store should have a completed discovery cycle
        record = state.agent_store.get_agent(AgentRole.DISCOVERY)
        assert record is not None
        assert record.status == AgentStatus.COMPLETED
        assert record.total_cycles == 1

        cycles = state.agent_store.cycles_for(AgentRole.DISCOVERY)
        assert len(cycles) == 1
        assert cycles[0].status == AgentStatus.COMPLETED
        assert cycles[0].finished_at > cycles[0].started_at

    async def test_discovery_updates_budget_tracker(self) -> None:
        """Discovery delegation should begin+end a budget cycle."""
        config = _make_config()
        hub = OrchestratorHub(config)
        state = RunState(request="run tests")
        resolution = _make_intent_resolution()

        await hub._delegate_to_discovery(state, resolution)

        # Budget tracker should show 1 iteration used for discovery
        status = hub.budget_tracker.check(AgentRole.DISCOVERY)
        assert status.iterations_used == 1
        assert status.wall_clock_used > 0

    async def test_discovery_populates_discovered_tests(self) -> None:
        """Discovery should populate state.discovered_tests."""
        config = _make_config()
        hub = OrchestratorHub(config)
        state = RunState(request="run tests")
        resolution = _make_intent_resolution([
            _make_test_command("pytest tests/unit"),
            _make_test_command("pytest tests/integration"),
        ])

        await hub._delegate_to_discovery(state, resolution)

        assert len(state.discovered_tests) == 2
        assert state.discovered_tests[0]["command"] == "pytest tests/unit"
        assert state.discovered_tests[1]["command"] == "pytest tests/integration"

    async def test_discovery_budget_pre_check_raises_on_exhausted(self) -> None:
        """If discovery budget is already exhausted, should raise."""
        config = _make_config()
        budget_config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=100),
            per_agent={
                AgentRole.DISCOVERY: AgentBudget(max_iterations=1),
            },
        )
        hub = OrchestratorHub(config, budget_config=budget_config)

        # Exhaust discovery budget
        hub.budget_tracker.begin_cycle(AgentRole.DISCOVERY)
        hub.budget_tracker.end_cycle(AgentRole.DISCOVERY)

        state = RunState(request="run tests")
        resolution = _make_intent_resolution()

        with pytest.raises(BudgetExceededError):
            await hub._delegate_to_discovery(state, resolution)

    async def test_discovery_handoff_summary_in_state_store(self) -> None:
        """Handoff summary from discovery should be stored in state store cycle."""
        config = _make_config()
        hub = OrchestratorHub(config)
        state = RunState(request="run tests")
        resolution = _make_intent_resolution()

        await hub._delegate_to_discovery(state, resolution)

        cycle = state.agent_store.latest_cycle_for(AgentRole.DISCOVERY)
        assert cycle is not None
        assert "agent" in cycle.output_summary
        assert cycle.output_summary["agent"] == "discovery-agent"
        assert "state" in cycle.output_summary

    async def test_discovery_error_path_closes_cycle(self) -> None:
        """If discovery raises, cycle should be marked FAILED and budget ended."""
        config = _make_config()

        # Create a discovery agent that raises during reset_state
        # (which is called at the start of delegation)
        class BrokenDiscovery(DiscoveryAgent):
            _call_count = 0

            def reset_state(self) -> None:
                super().reset_state()
                # Raise on the first method call after reset
                original_add = self.state.add_finding

                def exploding_add(finding):
                    raise RuntimeError("boom")

                self.state.add_finding = exploding_add  # type: ignore[method-assign]

        broken = BrokenDiscovery()
        hub = OrchestratorHub(config, discovery=broken)
        state = RunState(request="run tests")
        resolution = _make_intent_resolution()

        with pytest.raises(RuntimeError, match="boom"):
            await hub._delegate_to_discovery(state, resolution)

        # State store should reflect failure
        record = state.agent_store.get_agent(AgentRole.DISCOVERY)
        assert record is not None
        assert record.status == AgentStatus.FAILED
        assert "boom" in record.errors[0]

        # Budget should still have recorded the cycle
        status = hub.budget_tracker.check(AgentRole.DISCOVERY)
        assert status.iterations_used == 1


# ---------------------------------------------------------------------------
# Executor delegation wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecutorDelegationWiring:
    """Test that _delegate_to_executor properly wires state + budget."""

    async def test_executor_creates_delegation_cycle(self) -> None:
        """Executor delegation should create and complete a cycle."""
        config = _make_config()
        cmd = _make_test_command()
        result = _make_execution_result()

        # Mock the executor's execute_batch
        mock_executor = TaskExecutor()
        record = TaskAttemptRecord(
            task_id="task-0001",
            command=cmd,
            max_attempts=3,
        )
        record.record_attempt(result)

        mock_executor.execute_batch = AsyncMock(return_value=[record])
        mock_executor.batch_summary = MagicMock(return_value={
            "total_tasks": 1,
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "timed_out": 0,
            "total_attempts": 1,
            "total_duration_seconds": 0.5,
            "tasks": [record.to_summary()],
        })

        hub = OrchestratorHub(config, executor=mock_executor)
        state = RunState(request="run tests")
        resolution = _make_intent_resolution([cmd])

        await hub._delegate_to_executor(state, resolution)

        # State store should have a completed executor cycle
        agent_record = state.agent_store.get_agent(AgentRole.EXECUTOR)
        assert agent_record is not None
        assert agent_record.status == AgentStatus.COMPLETED
        assert agent_record.total_cycles == 1

    async def test_executor_updates_budget_tracker(self) -> None:
        """Executor should begin+end a budget cycle."""
        config = _make_config()
        cmd = _make_test_command()
        result = _make_execution_result()

        mock_executor = TaskExecutor()
        record = TaskAttemptRecord(
            task_id="task-0001", command=cmd, max_attempts=3,
        )
        record.record_attempt(result)
        mock_executor.execute_batch = AsyncMock(return_value=[record])
        mock_executor.batch_summary = MagicMock(return_value={
            "total_tasks": 1, "passed": 1, "failed": 0,
            "errored": 0, "timed_out": 0,
            "total_attempts": 1, "total_duration_seconds": 0.5,
            "tasks": [],
        })

        hub = OrchestratorHub(config, executor=mock_executor)
        state = RunState(request="run tests")
        resolution = _make_intent_resolution([cmd])

        await hub._delegate_to_executor(state, resolution)

        status = hub.budget_tracker.check(AgentRole.EXECUTOR)
        assert status.iterations_used == 1
        assert status.wall_clock_used > 0

    async def test_executor_stores_results_in_state(self) -> None:
        """Execution results should be stored in state.execution_results."""
        config = _make_config()
        cmd = _make_test_command()
        result = _make_execution_result()

        mock_executor = TaskExecutor()
        record = TaskAttemptRecord(
            task_id="task-0001", command=cmd, max_attempts=3,
        )
        record.record_attempt(result)
        mock_executor.execute_batch = AsyncMock(return_value=[record])
        mock_executor.batch_summary = MagicMock(return_value={
            "total_tasks": 1, "passed": 1, "failed": 0,
            "errored": 0, "timed_out": 0,
            "total_attempts": 1, "total_duration_seconds": 0.5,
            "tasks": [],
        })

        hub = OrchestratorHub(config, executor=mock_executor)
        state = RunState(request="run tests")
        resolution = _make_intent_resolution([cmd])

        await hub._delegate_to_executor(state, resolution)

        assert len(state.execution_results) == 1
        assert state.execution_results[0]["task_id"] == "task-0001"
        assert state.execution_results[0]["final_status"] == "passed"

    async def test_executor_budget_pre_check_raises_on_exhausted(self) -> None:
        """If executor budget is exhausted, should raise."""
        config = _make_config()
        budget_config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=100),
            per_agent={
                AgentRole.EXECUTOR: AgentBudget(max_iterations=1),
            },
        )
        hub = OrchestratorHub(config, budget_config=budget_config)

        # Exhaust executor budget
        hub.budget_tracker.begin_cycle(AgentRole.EXECUTOR)
        hub.budget_tracker.end_cycle(AgentRole.EXECUTOR)

        state = RunState(request="run tests")
        resolution = _make_intent_resolution()

        with pytest.raises(BudgetExceededError):
            await hub._delegate_to_executor(state, resolution)

    async def test_executor_error_path_closes_cycle(self) -> None:
        """If executor raises, cycle should be marked FAILED."""
        config = _make_config()
        mock_executor = TaskExecutor()
        mock_executor.execute_batch = AsyncMock(
            side_effect=RuntimeError("execution failed"),
        )

        hub = OrchestratorHub(config, executor=mock_executor)
        state = RunState(request="run tests")
        resolution = _make_intent_resolution()

        with pytest.raises(RuntimeError, match="execution failed"):
            await hub._delegate_to_executor(state, resolution)

        record = state.agent_store.get_agent(AgentRole.EXECUTOR)
        assert record is not None
        assert record.status == AgentStatus.FAILED
        assert "execution failed" in record.errors[0]

        # Budget cycle should still be closed
        status = hub.budget_tracker.check(AgentRole.EXECUTOR)
        assert status.iterations_used == 1

    async def test_executor_wires_attempts_to_reporter(self) -> None:
        """Executor attempts should be routed through the reporter callback."""
        config = _make_config()
        cmd = _make_test_command()
        result = _make_execution_result(stdout="PASSED: test_foo")

        mock_executor = TaskExecutor()
        record = TaskAttemptRecord(
            task_id="task-0001", command=cmd, max_attempts=3,
        )
        record.record_attempt(result)
        mock_executor.execute_batch = AsyncMock(return_value=[record])
        mock_executor.batch_summary = MagicMock(return_value={
            "total_tasks": 1, "passed": 1, "failed": 0,
            "errored": 0, "timed_out": 0,
            "total_attempts": 1, "total_duration_seconds": 0.5,
            "tasks": [],
        })

        hub = OrchestratorHub(config, executor=mock_executor)
        state = RunState(request="run tests")
        resolution = _make_intent_resolution([cmd])

        await hub._start_reporter(state)
        await hub._delegate_to_executor(state, resolution)
        mock_executor._on_attempt("task-0001", record, result)  # type: ignore[misc]
        await asyncio.sleep(0)

        assert hub.reporter.stats.total == 1
        assert hub.reporter.stats.passed == 1


# ---------------------------------------------------------------------------
# Full run() integration — state + budget across all phases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFullRunDelegationWiring:
    """Test that the full run() method wires state + budget for all sub-agents."""

    async def test_run_tracks_discovery_in_state_store(self) -> None:
        """A full run should show discovery delegation in state store."""
        config = _make_config()
        resolution = _make_intent_resolution()

        # Mock executor to return passing results
        mock_executor = TaskExecutor()
        cmd = _make_test_command()
        record = TaskAttemptRecord(
            task_id="task-0001", command=cmd, max_attempts=3,
        )
        record.record_attempt(_make_execution_result())
        mock_executor.execute_batch = AsyncMock(return_value=[record])
        mock_executor.batch_summary = MagicMock(return_value={
            "total_tasks": 1, "passed": 1, "failed": 0,
            "errored": 0, "timed_out": 0,
            "total_attempts": 1, "total_duration_seconds": 0.5,
            "tasks": [],
        })

        hub = OrchestratorHub(config, executor=mock_executor)

        # Mock intent resolution to avoid LLM calls
        hub._resolve_intent = AsyncMock(return_value=resolution)

        state = await hub.run("run all tests")

        # Discovery should have a completed cycle
        disc_record = state.agent_store.get_agent(AgentRole.DISCOVERY)
        assert disc_record is not None
        assert disc_record.total_cycles == 1

    async def test_run_tracks_executor_in_state_store(self) -> None:
        """A full run should show executor delegation in state store."""
        config = _make_config()
        resolution = _make_intent_resolution()

        mock_executor = TaskExecutor()
        cmd = _make_test_command()
        record = TaskAttemptRecord(
            task_id="task-0001", command=cmd, max_attempts=3,
        )
        record.record_attempt(_make_execution_result())
        mock_executor.execute_batch = AsyncMock(return_value=[record])
        mock_executor.batch_summary = MagicMock(return_value={
            "total_tasks": 1, "passed": 1, "failed": 0,
            "errored": 0, "timed_out": 0,
            "total_attempts": 1, "total_duration_seconds": 0.5,
            "tasks": [],
        })

        hub = OrchestratorHub(config, executor=mock_executor)
        hub._resolve_intent = AsyncMock(return_value=resolution)

        state = await hub.run("run all tests")

        exec_record = state.agent_store.get_agent(AgentRole.EXECUTOR)
        assert exec_record is not None
        assert exec_record.total_cycles == 1

    async def test_run_budget_tracking_across_phases(self) -> None:
        """Budget tracker should reflect usage from all agent delegations."""
        config = _make_config()
        resolution = _make_intent_resolution()

        mock_executor = TaskExecutor()
        cmd = _make_test_command()
        record = TaskAttemptRecord(
            task_id="task-0001", command=cmd, max_attempts=3,
        )
        record.record_attempt(_make_execution_result())
        mock_executor.execute_batch = AsyncMock(return_value=[record])
        mock_executor.batch_summary = MagicMock(return_value={
            "total_tasks": 1, "passed": 1, "failed": 0,
            "errored": 0, "timed_out": 0,
            "total_attempts": 1, "total_duration_seconds": 0.5,
            "tasks": [],
        })

        hub = OrchestratorHub(config, executor=mock_executor)
        hub._resolve_intent = AsyncMock(return_value=resolution)

        state = await hub.run("run all tests")

        # Each agent should have consumed at least 1 budget iteration
        disc_status = hub.budget_tracker.check(AgentRole.DISCOVERY)
        assert disc_status.iterations_used >= 1

        exec_status = hub.budget_tracker.check(AgentRole.EXECUTOR)
        assert exec_status.iterations_used >= 1

        reporter_status = hub.budget_tracker.check(AgentRole.REPORTER)
        assert reporter_status.iterations_used >= 1

    async def test_run_budget_exceeded_mid_flow_produces_escalation(self) -> None:
        """Budget exceeded during execution should produce escalation record."""
        config = _make_config()
        resolution = _make_intent_resolution()

        # Set executor budget to 0 iterations (unlimited = no limit)
        # Instead, set token budget very low and pre-exhaust
        budget_config = AgentBudgetConfig(
            global_default=AgentBudget(max_iterations=100),
            per_agent={
                AgentRole.EXECUTOR: AgentBudget(
                    max_iterations=0,  # unlimited
                    max_token_spend=1,  # will be exceeded
                ),
            },
        )

        hub = OrchestratorHub(config, budget_config=budget_config)
        hub._resolve_intent = AsyncMock(return_value=resolution)

        # Pre-exhaust executor token budget
        hub.budget_tracker.record_tokens(AgentRole.EXECUTOR, 1)

        state = await hub.run("run all tests")

        # Should have failed with budget escalation
        assert state.phase == RunPhase.FAILED
        assert any(e.reason == "budget_exceeded" for e in state.escalations)

    async def test_state_store_snapshot_includes_all_delegated_agents(self) -> None:
        """After a full run, snapshot should include all agents that were delegated."""
        config = _make_config()
        resolution = _make_intent_resolution()

        mock_executor = TaskExecutor()
        cmd = _make_test_command()
        record = TaskAttemptRecord(
            task_id="task-0001", command=cmd, max_attempts=3,
        )
        record.record_attempt(_make_execution_result())
        mock_executor.execute_batch = AsyncMock(return_value=[record])
        mock_executor.batch_summary = MagicMock(return_value={
            "total_tasks": 1, "passed": 1, "failed": 0,
            "errored": 0, "timed_out": 0,
            "total_attempts": 1, "total_duration_seconds": 0.5,
            "tasks": [],
        })

        hub = OrchestratorHub(config, executor=mock_executor)
        hub._resolve_intent = AsyncMock(return_value=resolution)

        state = await hub.run("run all tests")

        snap = state.agent_store.snapshot()
        assert "discovery" in snap["agents"]
        assert "executor" in snap["agents"]
        assert "reporter" in snap["agents"]
        # At least 3 cycles: discovery + executor + reporter
        assert snap["total_cycles"] >= 3


# ---------------------------------------------------------------------------
# Hub constructor wiring
# ---------------------------------------------------------------------------


class TestHubConstructorWiring:
    """Test that the hub properly accepts and exposes discovery/executor."""

    def test_hub_has_discovery_agent(self) -> None:
        config = _make_config()
        hub = OrchestratorHub(config)
        assert hub.discovery is not None
        assert isinstance(hub.discovery, DiscoveryAgent)

    def test_hub_has_executor(self) -> None:
        config = _make_config()
        hub = OrchestratorHub(config)
        assert hub.executor is not None
        assert isinstance(hub.executor, TaskExecutor)

    def test_hub_accepts_custom_discovery(self) -> None:
        config = _make_config()
        custom = DiscoveryAgent(hard_cap_steps=5)
        hub = OrchestratorHub(config, discovery=custom)
        assert hub.discovery is custom
        assert hub.discovery.step_counter.hard_cap == 5

    def test_hub_accepts_custom_executor(self) -> None:
        from test_runner.execution.executor import ExecutionPolicy

        config = _make_config()
        custom = TaskExecutor(policy=ExecutionPolicy.strict())
        hub = OrchestratorHub(config, executor=custom)
        assert hub.executor is custom
        assert hub.executor.policy.max_attempts == 1
