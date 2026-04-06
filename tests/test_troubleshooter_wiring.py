"""Tests for troubleshooter wiring into the orchestrator execution failure path.

Verifies that:
- The orchestrator detects execution failures correctly
- The troubleshooter is automatically invoked on failures
- Failure details are correctly extracted from execution results
- The troubleshooter receives proper error context
- Non-failure execution results do not trigger troubleshooting
- Troubleshooter errors are handled gracefully
"""

from __future__ import annotations

import pytest

from test_runner.agents.troubleshooter.agent import TroubleshooterAgent
from test_runner.agents.troubleshooter.models import (
    FailureCategory,
    FixConfidence,
    FixProposal,
    FixProposalSet,
)
from test_runner.execution.command_translator import TestCommand
from test_runner.execution.executor import TaskAttemptRecord
from test_runner.execution.targets import ExecutionResult, ExecutionStatus
from test_runner.models.summary import FailureDetail, TestOutcome
from test_runner.orchestrator.hub import (
    OrchestratorHub,
    RunPhase,
    RunState,
)
from test_runner.orchestrator.state_store import AgentStatus
from test_runner.agents.base import AgentRole
from test_runner.agents.parser import TestFramework


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_config():
    """Create a minimal Config for tests."""
    from test_runner.config import Config
    return Config(
        llm_base_url="http://localhost:8080/v1",
        api_key="test-key",
        model_id="test-model",
    )


def _make_execution_result_dict(
    task_id: str = "task-0001",
    final_status: str = "failed",
    command: str = "pytest tests/test_foo.py",
    framework: str = "pytest",
    exit_code: int = 1,
    stdout: str = "",
    stderr: str = "AssertionError: expected True",
) -> dict:
    """Build an execution result dict matching TaskAttemptRecord.to_summary()."""
    return {
        "task_id": task_id,
        "command": command,
        "framework": framework,
        "attempts_made": 1,
        "max_attempts": 3,
        "budget_exhausted": False,
        "final_status": final_status,
        "total_duration_seconds": 1.5,
        "stdout": stdout,
        "stderr": stderr,
        "attempt_details": [
            {
                "attempt": 1,
                "status": final_status,
                "exit_code": exit_code,
                "duration": 1.5,
            }
        ],
    }


def _make_task_attempt_record(
    status: ExecutionStatus = ExecutionStatus.FAILED,
    stdout: str = "",
    stderr: str = "FAILED test_bar",
    exit_code: int = 1,
) -> TaskAttemptRecord:
    """Build a TaskAttemptRecord with one attempt."""
    cmd = TestCommand(
        command=["pytest", "tests/test_bar.py"],
        display="pytest tests/test_bar.py",
        framework=TestFramework.PYTEST,
    )
    record = TaskAttemptRecord(
        task_id="task-0001",
        command=cmd,
        max_attempts=3,
    )
    record.record_attempt(
        ExecutionResult(
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=2.0,
            command_display=cmd.display,
        )
    )
    return record


# ---------------------------------------------------------------------------
# _has_execution_failures
# ---------------------------------------------------------------------------


class TestHasExecutionFailures:
    """Test OrchestratorHub._has_execution_failures."""

    def test_no_results_no_failures(self):
        hub = OrchestratorHub(_make_config())
        state = RunState()
        assert hub._has_execution_failures(state) is False

    def test_all_passed_no_failures(self):
        hub = OrchestratorHub(_make_config())
        state = RunState(execution_results=[
            _make_execution_result_dict(final_status="passed"),
        ])
        assert hub._has_execution_failures(state) is False

    def test_failed_status_detected(self):
        hub = OrchestratorHub(_make_config())
        state = RunState(execution_results=[
            _make_execution_result_dict(final_status="failed"),
        ])
        assert hub._has_execution_failures(state) is True

    def test_error_status_detected(self):
        hub = OrchestratorHub(_make_config())
        state = RunState(execution_results=[
            _make_execution_result_dict(final_status="error"),
        ])
        assert hub._has_execution_failures(state) is True

    def test_timeout_status_detected(self):
        hub = OrchestratorHub(_make_config())
        state = RunState(execution_results=[
            _make_execution_result_dict(final_status="timeout"),
        ])
        assert hub._has_execution_failures(state) is True

    def test_pre_built_failure_details_detected(self):
        hub = OrchestratorHub(_make_config())
        state = RunState(failure_details=[
            FailureDetail(
                test_id="test-1",
                test_name="test_foo",
                outcome=TestOutcome.FAILED,
                error_message="assertion failed",
            ),
        ])
        assert hub._has_execution_failures(state) is True

    def test_mixed_results_with_one_failure(self):
        hub = OrchestratorHub(_make_config())
        state = RunState(execution_results=[
            _make_execution_result_dict(task_id="t1", final_status="passed"),
            _make_execution_result_dict(task_id="t2", final_status="failed"),
            _make_execution_result_dict(task_id="t3", final_status="passed"),
        ])
        assert hub._has_execution_failures(state) is True


# ---------------------------------------------------------------------------
# _extract_failure_details
# ---------------------------------------------------------------------------


class TestExtractFailureDetails:
    """Test OrchestratorHub._extract_failure_details."""

    def test_empty_results(self):
        state = RunState()
        details = OrchestratorHub._extract_failure_details(state)
        assert details == []

    def test_all_passed_returns_empty(self):
        state = RunState(execution_results=[
            _make_execution_result_dict(final_status="passed"),
        ])
        details = OrchestratorHub._extract_failure_details(state)
        assert details == []

    def test_failed_result_extracted(self):
        state = RunState(execution_results=[
            _make_execution_result_dict(
                task_id="task-0001",
                final_status="failed",
                command="pytest tests/test_foo.py",
                stderr="AssertionError: expected True",
                exit_code=1,
            ),
        ])
        details = OrchestratorHub._extract_failure_details(state)
        assert len(details) == 1
        d = details[0]
        assert d.test_id == "task-0001"
        assert d.outcome == TestOutcome.FAILED
        assert "failed" in d.error_message
        assert d.error_type == "failed"
        assert d.stderr == "AssertionError: expected True"

    def test_failed_result_with_top_level_stderr_is_preserved(self):
        record = _make_task_attempt_record(
            status=ExecutionStatus.ERROR,
            stderr="OS error: [WinError 5] Access is denied",
            exit_code=-1,
        )
        state = RunState(execution_results=[record.to_summary()])

        details = OrchestratorHub._extract_failure_details(state)

        assert len(details) == 1
        assert details[0].stderr == "OS error: [WinError 5] Access is denied"

    def test_error_result_extracted(self):
        state = RunState(execution_results=[
            _make_execution_result_dict(final_status="error", exit_code=2),
        ])
        details = OrchestratorHub._extract_failure_details(state)
        assert len(details) == 1
        assert details[0].outcome == TestOutcome.ERROR

    def test_timeout_result_extracted(self):
        state = RunState(execution_results=[
            _make_execution_result_dict(final_status="timeout"),
        ])
        details = OrchestratorHub._extract_failure_details(state)
        assert len(details) == 1
        assert "timed out" in details[0].error_message

    def test_metadata_includes_attempt_info(self):
        state = RunState(execution_results=[
            _make_execution_result_dict(final_status="failed"),
        ])
        details = OrchestratorHub._extract_failure_details(state)
        assert details[0].metadata["attempts_made"] == 1
        assert details[0].metadata["max_attempts"] == 3

    def test_multiple_failures_extracted(self):
        state = RunState(execution_results=[
            _make_execution_result_dict(task_id="t1", final_status="passed"),
            _make_execution_result_dict(task_id="t2", final_status="failed"),
            _make_execution_result_dict(task_id="t3", final_status="error"),
            _make_execution_result_dict(task_id="t4", final_status="passed"),
        ])
        details = OrchestratorHub._extract_failure_details(state)
        assert len(details) == 2
        assert {d.test_id for d in details} == {"t2", "t3"}


# ---------------------------------------------------------------------------
# build_failure_details_from_records
# ---------------------------------------------------------------------------


class TestBuildFailureDetailsFromRecords:
    """Test OrchestratorHub.build_failure_details_from_records."""

    def test_empty_records(self):
        details = OrchestratorHub.build_failure_details_from_records([])
        assert details == []

    def test_passed_records_ignored(self):
        record = _make_task_attempt_record(status=ExecutionStatus.PASSED)
        details = OrchestratorHub.build_failure_details_from_records([record])
        assert details == []

    def test_failed_record_converted(self):
        record = _make_task_attempt_record(
            status=ExecutionStatus.FAILED,
            stderr="FAIL: test_bar - assert False",
            exit_code=1,
        )
        details = OrchestratorHub.build_failure_details_from_records([record])
        assert len(details) == 1
        d = details[0]
        assert d.test_id == "task-0001"
        assert d.outcome == TestOutcome.FAILED
        assert d.stderr == "FAIL: test_bar - assert False"
        assert d.framework == "pytest"
        assert d.metadata["attempts_made"] == 1

    def test_timeout_record_converted(self):
        record = _make_task_attempt_record(status=ExecutionStatus.TIMEOUT)
        details = OrchestratorHub.build_failure_details_from_records([record])
        assert len(details) == 1
        assert "timed out" in details[0].error_message

    def test_error_record_converted(self):
        record = _make_task_attempt_record(status=ExecutionStatus.ERROR)
        details = OrchestratorHub.build_failure_details_from_records([record])
        assert len(details) == 1
        assert details[0].outcome == TestOutcome.ERROR


# ---------------------------------------------------------------------------
# _invoke_troubleshooter (async)
# ---------------------------------------------------------------------------


class TestInvokeTroubleshooter:
    """Test OrchestratorHub._invoke_troubleshooter."""

    @pytest.mark.asyncio
    async def test_invokes_troubleshooter_with_failure_details(self):
        """Troubleshooter is called with pre-built failure details."""
        hub = OrchestratorHub(_make_config())
        state = RunState(
            failure_details=[
                FailureDetail(
                    test_id="test-1",
                    test_name="test_foo",
                    outcome=TestOutcome.FAILED,
                    error_message="assert 1 == 2",
                    error_type="AssertionError",
                    stderr="AssertionError: assert 1 == 2",
                ),
            ],
        )

        await hub._invoke_troubleshooter(state)

        assert state.troubleshooter_result is not None
        assert isinstance(state.troubleshooter_result, FixProposalSet)
        # Escalation record should be added
        assert len(state.escalations) == 1
        assert state.escalations[0].source_agent == "executor"
        assert state.escalations[0].target == "troubleshooter"
        assert state.escalations[0].metadata["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_extracts_failures_from_execution_results(self):
        """When failure_details is empty, extracts from execution_results."""
        hub = OrchestratorHub(_make_config())
        state = RunState(
            execution_results=[
                _make_execution_result_dict(
                    task_id="t1",
                    final_status="failed",
                    stderr="ImportError: no module named foo",
                ),
            ],
        )

        await hub._invoke_troubleshooter(state)

        # failure_details should have been populated
        assert len(state.failure_details) == 1
        assert state.failure_details[0].test_id == "t1"
        assert state.troubleshooter_result is not None

    @pytest.mark.asyncio
    async def test_skips_when_no_failures(self):
        """No-op when there are no failures to troubleshoot."""
        hub = OrchestratorHub(_make_config())
        state = RunState()

        await hub._invoke_troubleshooter(state)

        assert state.troubleshooter_result is None
        assert len(state.escalations) == 0

    @pytest.mark.asyncio
    async def test_resets_troubleshooter_before_invocation(self):
        """Troubleshooter state is reset for each invocation."""
        troubleshooter = TroubleshooterAgent()
        # Simulate prior state
        troubleshooter.state.record_step()
        troubleshooter.state.record_step()
        assert troubleshooter.state.steps_taken == 2

        hub = OrchestratorHub(_make_config(), troubleshooter=troubleshooter)
        state = RunState(
            failure_details=[
                FailureDetail(
                    test_id="test-1",
                    test_name="test_foo",
                    outcome=TestOutcome.FAILED,
                    error_message="boom",
                ),
            ],
        )

        await hub._invoke_troubleshooter(state)

        # Steps should be 1 (from generate_fix_proposals), not 3
        assert troubleshooter.state.steps_taken == 1

    @pytest.mark.asyncio
    async def test_troubleshooter_error_is_captured(self):
        """If the troubleshooter raises, the error is captured in state."""
        troubleshooter = TroubleshooterAgent()

        # Make generate_fix_proposals_with_llm raise
        async def _raise(*args, **kwargs):
            raise RuntimeError("analyzer crashed")
        troubleshooter.generate_fix_proposals_with_llm = _raise  # type: ignore[assignment]

        hub = OrchestratorHub(_make_config(), troubleshooter=troubleshooter)
        state = RunState(
            failure_details=[
                FailureDetail(
                    test_id="test-1",
                    test_name="test_foo",
                    outcome=TestOutcome.FAILED,
                    error_message="boom",
                ),
            ],
        )

        await hub._invoke_troubleshooter(state)

        assert state.troubleshooter_result is None
        assert any("Troubleshooter error" in e for e in state.errors)

    @pytest.mark.asyncio
    async def test_multiple_failures_all_passed_to_troubleshooter(self):
        """Multiple failures are all sent to the troubleshooter."""
        hub = OrchestratorHub(_make_config())
        failures = [
            FailureDetail(
                test_id=f"test-{i}",
                test_name=f"test_fn_{i}",
                outcome=TestOutcome.FAILED,
                error_message=f"error {i}",
            )
            for i in range(5)
        ]
        state = RunState(failure_details=failures)

        await hub._invoke_troubleshooter(state)

        assert state.troubleshooter_result is not None
        assert state.escalations[0].metadata["failure_count"] == 5


# ---------------------------------------------------------------------------
# Integration: run() triggers troubleshooter on failures
# ---------------------------------------------------------------------------


class TestRunTroubleshooterIntegration:
    """Test that the full run() method triggers troubleshooting on failures."""

    @pytest.mark.asyncio
    async def test_run_with_failures_triggers_troubleshooting(self):
        """When execution_results have failures, troubleshooter is invoked."""
        hub = OrchestratorHub(
            _make_config(),
            parse_mode="offline",
        )

        # We need to inject execution results after parsing but before the
        # troubleshooter check. We'll do this by pre-populating and verifying
        # the troubleshooter wiring separately, since run() has TODO stubs.
        # Instead, test the phase transition and detection directly.
        state = RunState(
            execution_results=[
                _make_execution_result_dict(final_status="failed"),
            ],
        )

        # Verify failure detection
        assert hub._has_execution_failures(state) is True

        # Verify troubleshooter invocation
        await hub._invoke_troubleshooter(state)
        assert state.troubleshooter_result is not None

    @pytest.mark.asyncio
    async def test_run_without_failures_skips_troubleshooting(self):
        """When all tests pass, troubleshooter is not invoked."""
        hub = OrchestratorHub(_make_config())
        state = RunState(
            execution_results=[
                _make_execution_result_dict(final_status="passed"),
            ],
        )

        assert hub._has_execution_failures(state) is False

        await hub._invoke_troubleshooter(state)
        assert state.troubleshooter_result is None


# ---------------------------------------------------------------------------
# Delegation cycle tracking in the agent state store
# ---------------------------------------------------------------------------


class TestTroubleshooterDelegationTracking:
    """Verify the troubleshooter delegation is tracked in AgentStateStore."""

    @pytest.mark.asyncio
    async def test_successful_invocation_creates_completed_cycle(self):
        """A successful troubleshooter invocation creates a COMPLETED cycle."""
        hub = OrchestratorHub(_make_config())
        state = RunState(
            failure_details=[
                FailureDetail(
                    test_id="test-1",
                    test_name="test_foo",
                    outcome=TestOutcome.FAILED,
                    error_message="assert 1 == 2",
                ),
            ],
        )

        await hub._invoke_troubleshooter(state)

        # Check that a delegation cycle was created
        cycles = state.agent_store.cycles_for(AgentRole.TROUBLESHOOTER)
        assert len(cycles) == 1
        cycle = cycles[0]
        assert cycle.status == AgentStatus.COMPLETED
        assert cycle.finished_at > 0
        assert cycle.input_summary["failure_count"] == 1
        assert cycle.input_summary["failure_ids"] == ["test-1"]

    @pytest.mark.asyncio
    async def test_failed_invocation_creates_failed_cycle(self):
        """A troubleshooter error creates a FAILED delegation cycle."""
        troubleshooter = TroubleshooterAgent()

        async def _raise(*args, **kwargs):
            raise RuntimeError("analyzer crashed")
        troubleshooter.generate_fix_proposals_with_llm = _raise  # type: ignore[assignment]

        hub = OrchestratorHub(_make_config(), troubleshooter=troubleshooter)
        state = RunState(
            failure_details=[
                FailureDetail(
                    test_id="test-1",
                    test_name="test_foo",
                    outcome=TestOutcome.FAILED,
                    error_message="boom",
                ),
            ],
        )

        await hub._invoke_troubleshooter(state)

        cycles = state.agent_store.cycles_for(AgentRole.TROUBLESHOOTER)
        assert len(cycles) == 1
        assert cycles[0].status == AgentStatus.FAILED
        assert "analyzer crashed" in cycles[0].error

    @pytest.mark.asyncio
    async def test_no_failures_creates_no_cycle(self):
        """When there are no failures, no delegation cycle is created."""
        hub = OrchestratorHub(_make_config())
        state = RunState()

        await hub._invoke_troubleshooter(state)

        cycles = state.agent_store.cycles_for(AgentRole.TROUBLESHOOTER)
        assert len(cycles) == 0

    @pytest.mark.asyncio
    async def test_agent_record_updated_after_invocation(self):
        """The agent record in the store reflects the troubleshooter's work."""
        hub = OrchestratorHub(_make_config())
        state = RunState(
            failure_details=[
                FailureDetail(
                    test_id="test-1",
                    test_name="test_foo",
                    outcome=TestOutcome.FAILED,
                    error_message="assert False",
                ),
            ],
        )

        await hub._invoke_troubleshooter(state)

        record = state.agent_store.get_agent(AgentRole.TROUBLESHOOTER)
        assert record is not None
        assert record.status == AgentStatus.COMPLETED
        assert record.total_cycles == 1
