"""Tests for execution attempt capping (~3 per task).

Verifies that the TaskExecutor respects the max_attempts budget,
retries only retriable failures, and stops on non-retriable failures.
"""

from __future__ import annotations

import pytest

from test_runner.agents.parser import TestFramework, TestIntent
from test_runner.execution.command_translator import TestCommand
from test_runner.execution.executor import (
    ExecutionPolicy,
    TaskAttemptRecord,
    TaskExecutor,
)
from test_runner.execution.targets import (
    ExecutionResult,
    ExecutionStatus,
    ExecutionTarget,
    SSHTarget,
)


# ---------------------------------------------------------------------------
# Helpers: mock execution target
# ---------------------------------------------------------------------------


class MockTarget(ExecutionTarget):
    """Controllable mock target that returns pre-configured results."""

    def __init__(self, results: list[ExecutionResult]) -> None:
        self._results = list(results)
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def call_count(self) -> int:
        return self._call_count

    async def execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        idx = min(self._call_count, len(self._results) - 1)
        self._call_count += 1
        return self._results[idx]


def _make_command(display: str = "pytest tests/") -> TestCommand:
    return TestCommand(
        command=["pytest", "tests/"],
        display=display,
        framework=TestFramework.PYTEST,
    )


def _make_result(
    status: ExecutionStatus = ExecutionStatus.PASSED,
    exit_code: int = 0,
    duration: float = 1.0,
) -> ExecutionResult:
    return ExecutionResult(
        status=status,
        exit_code=exit_code,
        stdout="output",
        stderr="",
        duration_seconds=duration,
        command_display="pytest tests/",
    )


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------


class TestExecutionPolicy:
    """Tests for ExecutionPolicy configuration and retry logic."""

    def test_default_max_attempts_is_three(self):
        policy = ExecutionPolicy.default()
        assert policy.max_attempts == 3

    def test_strict_policy_is_one_attempt(self):
        policy = ExecutionPolicy.strict()
        assert policy.max_attempts == 1
        assert not policy.retry_on_infra_error
        assert not policy.retry_on_timeout

    def test_lenient_policy_retries_test_failures(self):
        policy = ExecutionPolicy.lenient()
        assert policy.max_attempts == 3
        assert policy.retry_on_test_failure

    def test_invalid_max_attempts_raises(self):
        with pytest.raises(ValueError, match="max_attempts"):
            ExecutionPolicy(max_attempts=0)

    def test_invalid_backoff_factor_raises(self):
        with pytest.raises(ValueError, match="timeout_backoff_factor"):
            ExecutionPolicy(timeout_backoff_factor=0.5)

    def test_should_retry_on_success_is_false(self):
        policy = ExecutionPolicy.default()
        result = _make_result(ExecutionStatus.PASSED)
        assert not policy.should_retry(result)

    def test_should_retry_on_infra_error(self):
        policy = ExecutionPolicy.default()
        result = _make_result(ExecutionStatus.ERROR, exit_code=-1)
        assert policy.should_retry(result)

    def test_should_retry_on_timeout(self):
        policy = ExecutionPolicy.default()
        result = _make_result(ExecutionStatus.TIMEOUT, exit_code=-1)
        assert policy.should_retry(result)

    def test_should_not_retry_test_failure_by_default(self):
        policy = ExecutionPolicy.default()
        result = _make_result(ExecutionStatus.FAILED, exit_code=1)
        assert not policy.should_retry(result)

    def test_should_retry_test_failure_when_enabled(self):
        policy = ExecutionPolicy(retry_on_test_failure=True)
        result = _make_result(ExecutionStatus.FAILED, exit_code=1)
        assert policy.should_retry(result)


# ---------------------------------------------------------------------------
# TaskAttemptRecord tests
# ---------------------------------------------------------------------------


class TestTaskAttemptRecord:
    """Tests for the attempt tracking record."""

    def test_fresh_record_is_pending(self):
        record = TaskAttemptRecord(task_id="t1", command=_make_command())
        assert record.attempt_count == 0
        assert record.attempts_remaining == 3
        assert not record.budget_exhausted
        assert record.final_status == ExecutionStatus.PENDING

    def test_record_attempt_increments_count(self):
        record = TaskAttemptRecord(task_id="t1", command=_make_command())
        record.record_attempt(_make_result(ExecutionStatus.FAILED, exit_code=1))
        assert record.attempt_count == 1
        assert record.attempts_remaining == 2

    def test_budget_exhausted_at_max(self):
        record = TaskAttemptRecord(
            task_id="t1", command=_make_command(), max_attempts=2
        )
        record.record_attempt(_make_result(ExecutionStatus.ERROR, exit_code=-1))
        assert not record.budget_exhausted
        record.record_attempt(_make_result(ExecutionStatus.ERROR, exit_code=-1))
        assert record.budget_exhausted
        assert record.attempts_remaining == 0

    def test_total_duration_sums_all_attempts(self):
        record = TaskAttemptRecord(task_id="t1", command=_make_command())
        record.record_attempt(_make_result(duration=1.5))
        record.record_attempt(_make_result(duration=2.0))
        assert record.total_duration == pytest.approx(3.5)

    def test_attempt_numbers_are_sequential(self):
        record = TaskAttemptRecord(task_id="t1", command=_make_command())
        record.record_attempt(_make_result())
        record.record_attempt(_make_result())
        assert record.attempts[0].attempt == 1
        assert record.attempts[1].attempt == 2

    def test_to_summary_format(self):
        record = TaskAttemptRecord(task_id="t1", command=_make_command(), max_attempts=3)
        record.record_attempt(_make_result(ExecutionStatus.PASSED))
        summary = record.to_summary()
        assert summary["task_id"] == "t1"
        assert summary["attempts_made"] == 1
        assert summary["max_attempts"] == 3
        assert summary["final_status"] == "passed"
        assert not summary["budget_exhausted"]
        assert summary["stdout"] == "output"
        assert summary["stderr"] == ""


# ---------------------------------------------------------------------------
# TaskExecutor tests
# ---------------------------------------------------------------------------


class TestTaskExecutor:
    """Tests for the executor with attempt capping."""

    @pytest.mark.asyncio
    async def test_passes_on_first_attempt(self):
        """Success on first try — only 1 attempt used."""
        target = MockTarget([_make_result(ExecutionStatus.PASSED)])
        executor = TaskExecutor(policy=ExecutionPolicy.default())

        record = await executor.execute_task(_make_command(), target=target)

        assert record.attempt_count == 1
        assert record.final_status == ExecutionStatus.PASSED
        assert target.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_infra_error_up_to_cap(self):
        """Infrastructure error retried up to max_attempts (~3)."""
        error_result = _make_result(ExecutionStatus.ERROR, exit_code=-1)
        target = MockTarget([error_result, error_result, error_result])
        executor = TaskExecutor(policy=ExecutionPolicy.default())

        record = await executor.execute_task(_make_command(), target=target)

        assert record.attempt_count == 3
        assert record.budget_exhausted
        assert record.final_status == ExecutionStatus.ERROR
        assert target.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_timeout_up_to_cap(self):
        """Timeout retried up to max_attempts."""
        timeout_result = _make_result(ExecutionStatus.TIMEOUT, exit_code=-1)
        target = MockTarget([timeout_result, timeout_result, timeout_result])
        executor = TaskExecutor(policy=ExecutionPolicy.default())

        record = await executor.execute_task(_make_command(), target=target)

        assert record.attempt_count == 3
        assert record.budget_exhausted

    @pytest.mark.asyncio
    async def test_succeeds_on_retry(self):
        """Infra error then success — only 2 attempts used."""
        results = [
            _make_result(ExecutionStatus.ERROR, exit_code=-1),
            _make_result(ExecutionStatus.PASSED),
        ]
        target = MockTarget(results)
        executor = TaskExecutor(policy=ExecutionPolicy.default())

        record = await executor.execute_task(_make_command(), target=target)

        assert record.attempt_count == 2
        assert record.final_status == ExecutionStatus.PASSED
        assert not record.budget_exhausted

    @pytest.mark.asyncio
    async def test_test_failure_not_retried_by_default(self):
        """Test failure (non-retriable) stops after 1 attempt."""
        fail_result = _make_result(ExecutionStatus.FAILED, exit_code=1)
        target = MockTarget([fail_result, fail_result, fail_result])
        executor = TaskExecutor(policy=ExecutionPolicy.default())

        record = await executor.execute_task(_make_command(), target=target)

        assert record.attempt_count == 1  # No retry for test failures
        assert record.final_status == ExecutionStatus.FAILED
        assert target.call_count == 1

    @pytest.mark.asyncio
    async def test_custom_max_attempts(self):
        """Custom policy with max_attempts=5."""
        error_result = _make_result(ExecutionStatus.ERROR, exit_code=-1)
        target = MockTarget([error_result] * 5)
        policy = ExecutionPolicy(max_attempts=5)
        executor = TaskExecutor(policy=policy)

        record = await executor.execute_task(_make_command(), target=target)

        assert record.attempt_count == 5
        assert record.budget_exhausted
        assert target.call_count == 5

    @pytest.mark.asyncio
    async def test_strict_policy_no_retries(self):
        """Strict policy: exactly 1 attempt, no retries."""
        error_result = _make_result(ExecutionStatus.ERROR, exit_code=-1)
        target = MockTarget([error_result])
        executor = TaskExecutor(policy=ExecutionPolicy.strict())

        record = await executor.execute_task(_make_command(), target=target)

        assert record.attempt_count == 1
        assert record.budget_exhausted
        assert target.call_count == 1

    @pytest.mark.asyncio
    async def test_batch_execution_independent_budgets(self):
        """Each command in a batch gets its own attempt budget."""
        # cmd1: error->pass (2 attempts), cmd2: pass (1 attempt)
        target1_results = [
            _make_result(ExecutionStatus.ERROR, exit_code=-1),
            _make_result(ExecutionStatus.PASSED),
        ]
        target = MockTarget(
            target1_results + [_make_result(ExecutionStatus.PASSED)]
        )
        executor = TaskExecutor(policy=ExecutionPolicy.default())

        cmd1 = _make_command("cmd1")
        cmd2 = _make_command("cmd2")
        records = await executor.execute_batch([cmd1, cmd2], target=target)

        assert len(records) == 2
        assert records[0].attempt_count == 2
        assert records[0].final_status == ExecutionStatus.PASSED
        assert records[1].attempt_count == 1
        assert records[1].final_status == ExecutionStatus.PASSED

    @pytest.mark.asyncio
    async def test_batch_summary(self):
        """Batch summary aggregates all task records."""
        target = MockTarget([
            _make_result(ExecutionStatus.PASSED),
            _make_result(ExecutionStatus.FAILED, exit_code=1),
        ])
        executor = TaskExecutor(policy=ExecutionPolicy.default())

        await executor.execute_batch(
            [_make_command("pass"), _make_command("fail")],
            target=target,
        )

        summary = executor.batch_summary()
        assert summary["total_tasks"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["total_attempts"] == 2

    @pytest.mark.asyncio
    async def test_callback_invoked_per_attempt(self):
        """on_attempt callback fires for each execution attempt."""
        results = [
            _make_result(ExecutionStatus.ERROR, exit_code=-1),
            _make_result(ExecutionStatus.PASSED),
        ]
        target = MockTarget(results)
        callback_calls: list[tuple[str, int, str]] = []

        def on_attempt(task_id, record, result):
            callback_calls.append((task_id, record.attempt_count, result.status.value))

        executor = TaskExecutor(
            policy=ExecutionPolicy.default(),
            on_attempt=on_attempt,
        )

        await executor.execute_task(_make_command(), target=target)

        assert len(callback_calls) == 2
        assert callback_calls[0][1] == 1  # first attempt
        assert callback_calls[0][2] == "error"
        assert callback_calls[1][1] == 2  # second attempt
        assert callback_calls[1][2] == "passed"

    @pytest.mark.asyncio
    async def test_catalog_ssh_metadata_creates_ssh_target(self, monkeypatch):
        captured: list[str] = []

        async def fake_execute(self, command, *, working_directory="", env=None, timeout=None):
            captured.append(self.name)
            return _make_result(ExecutionStatus.PASSED)

        monkeypatch.setattr(SSHTarget, "execute", fake_execute)

        executor = TaskExecutor(policy=ExecutionPolicy.default())
        command = TestCommand(
            command=["./bin/device-check"],
            display="./bin/device-check",
            framework=TestFramework.SCRIPT,
            metadata={
                "catalog_system_transport": "ssh",
                "catalog_system_config": {
                    "alias": "lab-a",
                    "hostname": "lab-a.internal.example",
                    "username": "runner",
                },
            },
        )

        records = await executor.execute_batch([command])

        assert len(records) == 1
        assert records[0].final_status == ExecutionStatus.PASSED
        assert captured == ["ssh:lab-a"]


# ---------------------------------------------------------------------------
# ExecutionResult tests
# ---------------------------------------------------------------------------


class TestExecutionResult:
    """Tests for ExecutionResult properties."""

    def test_success_property(self):
        assert _make_result(ExecutionStatus.PASSED).success
        assert not _make_result(ExecutionStatus.FAILED).success

    def test_is_retriable(self):
        assert _make_result(ExecutionStatus.ERROR).is_retriable
        assert _make_result(ExecutionStatus.TIMEOUT).is_retriable
        assert not _make_result(ExecutionStatus.FAILED).is_retriable
        assert not _make_result(ExecutionStatus.PASSED).is_retriable
