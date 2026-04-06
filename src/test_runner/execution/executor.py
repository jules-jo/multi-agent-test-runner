"""Executor agent: runs test commands with attempt capping and retry logic.

Each task (test command) is capped at approximately 3 execution attempts.
The executor distinguishes between retriable failures (infrastructure errors,
timeouts) and non-retriable failures (test assertion failures). Only
retriable failures consume additional attempts.

The attempt budget is configurable via ExecutionPolicy and integrates with
the autonomy policy architecture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from test_runner.execution.command_translator import TestCommand
from test_runner.execution.targets import (
    ExecutionResult,
    ExecutionStatus,
    ExecutionTarget,
    LocalTarget,
    SSHTarget,
    TargetRegistry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution policy (configurable attempt caps and retry behaviour)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionPolicy:
    """Configurable policy for execution attempt limits and retry behaviour.

    Attributes:
        max_attempts: Hard cap on execution attempts per task (~3 default).
        retry_on_infra_error: Whether to retry on infrastructure errors.
        retry_on_timeout: Whether to retry on command timeouts.
        retry_on_test_failure: Whether to retry on actual test failures.
            Defaults to False — deterministic test failures don't benefit
            from blind retries.
        retry_delay_seconds: Delay between retry attempts (0 = immediate).
        timeout_backoff_factor: Multiply timeout by this on each retry
            (e.g. 1.5 means 50% more time on each retry).
    """

    max_attempts: int = 3
    retry_on_infra_error: bool = True
    retry_on_timeout: bool = True
    retry_on_test_failure: bool = False
    retry_delay_seconds: float = 0.0
    timeout_backoff_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.timeout_backoff_factor < 1.0:
            raise ValueError("timeout_backoff_factor must be >= 1.0")

    def should_retry(self, result: ExecutionResult) -> bool:
        """Determine if a result is eligible for retry based on policy."""
        if result.success:
            return False
        if result.status == ExecutionStatus.ERROR and self.retry_on_infra_error:
            return True
        if result.status == ExecutionStatus.TIMEOUT and self.retry_on_timeout:
            return True
        if result.status == ExecutionStatus.FAILED and self.retry_on_test_failure:
            return True
        return False

    @classmethod
    def strict(cls) -> ExecutionPolicy:
        """No retries at all — one shot per task."""
        return cls(max_attempts=1, retry_on_infra_error=False, retry_on_timeout=False)

    @classmethod
    def default(cls) -> ExecutionPolicy:
        """Default policy: ~3 attempts, retry only infra errors."""
        return cls()

    @classmethod
    def lenient(cls) -> ExecutionPolicy:
        """Lenient: retry everything including test failures, with backoff."""
        return cls(
            max_attempts=3,
            retry_on_infra_error=True,
            retry_on_timeout=True,
            retry_on_test_failure=True,
            retry_delay_seconds=1.0,
            timeout_backoff_factor=1.5,
        )


# ---------------------------------------------------------------------------
# Task execution tracker
# ---------------------------------------------------------------------------


@dataclass
class TaskAttemptRecord:
    """Tracks all attempts for a single task (test command).

    Attributes:
        task_id: Unique identifier for this task.
        command: The TestCommand being executed.
        attempts: Ordered list of execution results (one per attempt).
        max_attempts: The cap applied to this task.
    """

    task_id: str
    command: TestCommand
    attempts: list[ExecutionResult] = field(default_factory=list)
    max_attempts: int = 3

    @property
    def attempt_count(self) -> int:
        """Number of attempts made so far."""
        return len(self.attempts)

    @property
    def attempts_remaining(self) -> int:
        """Number of attempts still available."""
        return max(0, self.max_attempts - self.attempt_count)

    @property
    def budget_exhausted(self) -> bool:
        """Whether the attempt budget has been fully consumed."""
        return self.attempt_count >= self.max_attempts

    @property
    def latest_result(self) -> ExecutionResult | None:
        """Most recent execution result, or None if no attempts yet."""
        return self.attempts[-1] if self.attempts else None

    @property
    def final_status(self) -> ExecutionStatus:
        """The effective final status after all attempts."""
        if not self.attempts:
            return ExecutionStatus.PENDING
        return self.latest_result.status  # type: ignore[union-attr]

    @property
    def total_duration(self) -> float:
        """Total wall-clock time across all attempts."""
        return sum(a.duration_seconds for a in self.attempts)

    def record_attempt(self, result: ExecutionResult) -> None:
        """Record a new attempt result."""
        result_with_attempt = ExecutionResult(
            status=result.status,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration_seconds,
            command_display=result.command_display,
            attempt=self.attempt_count + 1,
            metadata=result.metadata,
        )
        self.attempts.append(result_with_attempt)

    def to_summary(self) -> dict[str, Any]:
        """Generate a summary dict for reporting."""
        last = self.latest_result
        return {
            "task_id": self.task_id,
            "command": self.command.display,
            "framework": self.command.framework.value,
            "attempts_made": self.attempt_count,
            "max_attempts": self.max_attempts,
            "budget_exhausted": self.budget_exhausted,
            "final_status": self.final_status.value,
            "total_duration_seconds": self.total_duration,
            "stdout": last.stdout if last else "",
            "stderr": last.stderr if last else "",
            "attempt_details": [
                {
                    "attempt": a.attempt,
                    "status": a.status.value,
                    "exit_code": a.exit_code,
                    "duration": a.duration_seconds,
                }
                for a in self.attempts
            ],
        }


# ---------------------------------------------------------------------------
# Executor: runs tasks with attempt capping
# ---------------------------------------------------------------------------


# Type alias for execution event callbacks (for reporter integration)
ExecutionCallback = Callable[[str, TaskAttemptRecord, ExecutionResult], None]


class TaskExecutor:
    """Executes test commands with configurable attempt capping.

    Each task is allowed up to ``policy.max_attempts`` (~3 by default).
    Only retriable failures (infrastructure errors, timeouts) trigger
    retries — deterministic test failures do not.

    Usage::

        executor = TaskExecutor(policy=ExecutionPolicy.default())
        record = await executor.execute_task(command, target=LocalTarget())

        # Check results
        print(record.final_status)
        print(f"Attempts: {record.attempt_count}/{record.max_attempts}")
    """

    def __init__(
        self,
        *,
        policy: ExecutionPolicy | None = None,
        target_registry: TargetRegistry | None = None,
        on_attempt: ExecutionCallback | None = None,
    ) -> None:
        self._policy = policy or ExecutionPolicy.default()
        self._registry = target_registry or TargetRegistry()
        self._on_attempt = on_attempt
        self._task_counter = 0
        self._all_records: list[TaskAttemptRecord] = []

    @property
    def policy(self) -> ExecutionPolicy:
        return self._policy

    @property
    def records(self) -> list[TaskAttemptRecord]:
        """All task records from this executor's lifetime."""
        return list(self._all_records)

    def _next_task_id(self) -> str:
        self._task_counter += 1
        return f"task-{self._task_counter:04d}"

    def _resolve_command_target(
        self,
        command: TestCommand,
        *,
        target: ExecutionTarget | None,
        target_name: str,
    ) -> tuple[ExecutionTarget | None, str]:
        """Resolve the effective execution target for one command."""
        if target is not None or target_name != "local":
            return target, target_name

        metadata = command.metadata or {}
        if str(metadata.get("catalog_system_transport", "")).lower() != "ssh":
            return None, target_name

        system_config = metadata.get("catalog_system_config")
        if not isinstance(system_config, dict):
            logger.warning(
                "SSH catalog metadata missing system config for command %s; "
                "falling back to local target",
                command.display,
            )
            return None, target_name

        return SSHTarget.from_metadata(system_config), "ssh"

    async def execute_task(
        self,
        command: TestCommand,
        *,
        target: ExecutionTarget | None = None,
        target_name: str = "local",
    ) -> TaskAttemptRecord:
        """Execute a single test command with retry/attempt capping.

        Args:
            command: The test command to execute.
            target: Explicit execution target (overrides target_name).
            target_name: Name of a registered target to use.

        Returns:
            TaskAttemptRecord with all attempt results and final status.
        """
        if target is None:
            target = self._registry.get(target_name) or self._registry.get_default()

        task_id = self._next_task_id()
        record = TaskAttemptRecord(
            task_id=task_id,
            command=command,
            max_attempts=self._policy.max_attempts,
        )

        logger.info(
            "Executing task %s: %s (max %d attempts)",
            task_id,
            command.display,
            self._policy.max_attempts,
        )

        current_timeout = command.timeout

        for attempt_num in range(1, self._policy.max_attempts + 1):
            logger.info(
                "Task %s attempt %d/%d",
                task_id,
                attempt_num,
                self._policy.max_attempts,
            )

            result = await target.execute(
                command.command,
                working_directory=command.working_directory,
                env=command.env or None,
                timeout=current_timeout,
            )

            record.record_attempt(result)

            will_retry = (
                not result.success
                and self._policy.should_retry(result)
                and not record.budget_exhausted
            )
            result.metadata["will_retry"] = will_retry
            result.metadata["is_final_attempt"] = not will_retry

            # Notify callback (for reporter streaming)
            if self._on_attempt:
                try:
                    self._on_attempt(task_id, record, result)
                except Exception as cb_err:
                    logger.warning("Attempt callback error: %s", cb_err)

            # Success — no more attempts needed
            if result.success:
                logger.info("Task %s passed on attempt %d", task_id, attempt_num)
                break

            # Check if we should retry
            if not self._policy.should_retry(result):
                logger.info(
                    "Task %s failed (non-retriable: %s) on attempt %d",
                    task_id,
                    result.status.value,
                    attempt_num,
                )
                break

            # Check if budget allows another attempt
            if record.budget_exhausted:
                logger.warning(
                    "Task %s: attempt budget exhausted (%d/%d)",
                    task_id,
                    record.attempt_count,
                    record.max_attempts,
                )
                break

            # Apply retry delay and timeout backoff
            if self._policy.retry_delay_seconds > 0:
                import asyncio
                await asyncio.sleep(self._policy.retry_delay_seconds)

            if current_timeout and self._policy.timeout_backoff_factor > 1.0:
                current_timeout = int(current_timeout * self._policy.timeout_backoff_factor)

        self._all_records.append(record)

        logger.info(
            "Task %s complete: %s after %d attempt(s) (%.2fs total)",
            task_id,
            record.final_status.value,
            record.attempt_count,
            record.total_duration,
        )

        return record

    async def execute_batch(
        self,
        commands: list[TestCommand],
        *,
        target: ExecutionTarget | None = None,
        target_name: str = "local",
    ) -> list[TaskAttemptRecord]:
        """Execute multiple test commands sequentially with attempt capping.

        Each command independently gets its own attempt budget.

        Args:
            commands: List of test commands to execute.
            target: Explicit execution target.
            target_name: Name of a registered target.

        Returns:
            List of TaskAttemptRecords, one per command.
        """
        records: list[TaskAttemptRecord] = []
        for cmd in commands:
            resolved_target, resolved_target_name = self._resolve_command_target(
                cmd,
                target=target,
                target_name=target_name,
            )
            record = await self.execute_task(
                cmd,
                target=resolved_target,
                target_name=resolved_target_name,
            )
            records.append(record)
        return records

    def batch_summary(self) -> dict[str, Any]:
        """Generate a summary of all tasks executed by this executor."""
        total = len(self._all_records)
        passed = sum(1 for r in self._all_records if r.final_status == ExecutionStatus.PASSED)
        failed = sum(1 for r in self._all_records if r.final_status == ExecutionStatus.FAILED)
        errored = sum(1 for r in self._all_records if r.final_status == ExecutionStatus.ERROR)
        timed_out = sum(1 for r in self._all_records if r.final_status == ExecutionStatus.TIMEOUT)
        total_attempts = sum(r.attempt_count for r in self._all_records)
        total_duration = sum(r.total_duration for r in self._all_records)

        return {
            "total_tasks": total,
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "timed_out": timed_out,
            "total_attempts": total_attempts,
            "total_duration_seconds": total_duration,
            "tasks": [r.to_summary() for r in self._all_records],
        }
