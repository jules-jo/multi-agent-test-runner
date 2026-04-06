"""Reporter agent — subscribes to executor output and emits TestResultEvents.

The Reporter agent acts as a real-time bridge between the executor and
reporting channels.  It:

1. Subscribes to executor output via callback registration
2. Parses raw output lines through framework-specific OutputParsers
3. Emits TestResultEvent objects to registered ReporterBase channels
   individually as they arrive (NOT batched)
4. Tracks aggregate statistics for the final run summary

The Reporter never communicates directly with other sub-agents — the
orchestrator hub manages all routing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agents import Agent

from test_runner.agents.base import AgentRole, BaseSubAgent
from test_runner.agents.reporter.output_parser import (
    OutputParser,
    OutputParserRegistry,
)
from test_runner.agents.reporter.rollup import (
    RollupConfig,
    RollupSummaryGenerator,
    RunStatisticsAdapter,
)
from test_runner.config import Config
from test_runner.execution.executor import TaskAttemptRecord
from test_runner.execution.targets import ExecutionResult, ExecutionStatus
from test_runner.reporting.base import ReporterBase, StreamEvent
from test_runner.reporting.events import (
    EventType,
    RunEvent,
    TestResultEvent,
    TestStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregate statistics tracker
# ---------------------------------------------------------------------------


@dataclass
class RunStatistics:
    """Tracks aggregate statistics across all test results in a run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    events: list[TestResultEvent] = field(default_factory=list)

    def record(self, event: TestResultEvent) -> None:
        """Record a single test result event."""
        self.total += 1
        self.events.append(event)
        if event.status == TestStatus.PASS:
            self.passed += 1
        elif event.status == TestStatus.FAIL:
            self.failed += 1
        elif event.status == TestStatus.ERROR:
            self.errors += 1
        elif event.status == TestStatus.SKIP:
            self.skipped += 1

    def finalize(self) -> None:
        """Mark the run as complete."""
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Wall-clock duration of the run in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.failed == 0 and self.errors == 0

    def to_summary(self) -> dict[str, Any]:
        """Generate summary dict for on_run_end callbacks."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration": round(self.duration, 3),
            "all_passed": self.all_passed,
        }

    def collect_failure_details(self) -> list[dict[str, Any]]:
        """Collect structured failure details with associated logs from all failed/errored events.

        Returns a list of dicts, each containing the test identity, error info,
        and all captured log output (stdout, stderr, error_details).
        """
        details: list[dict[str, Any]] = []
        for event in self.events:
            if event.status in (TestStatus.FAIL, TestStatus.ERROR):
                detail: dict[str, Any] = {
                    "test_name": event.test_name,
                    "status": event.status.value,
                    "duration": event.duration,
                    "message": event.message,
                    "error_details": event.error_details,
                    "stdout": event.stdout,
                    "stderr": event.stderr,
                    "file_path": event.file_path,
                    "line_number": event.line_number,
                    "suite": event.suite,
                    "has_logs": bool(event.stdout or event.stderr or event.error_details),
                }
                details.append(detail)
        return details

    def to_full_summary(self) -> dict[str, Any]:
        """Generate a complete summary dict including failure details with logs.

        Extends ``to_summary()`` with a ``failure_details`` key containing
        structured records for every failed/errored test, including all
        captured log output.
        """
        base = self.to_summary()
        base["failure_details"] = self.collect_failure_details()
        return base


# ---------------------------------------------------------------------------
# Status mapping from ExecutionStatus to TestStatus
# ---------------------------------------------------------------------------

_EXEC_TO_TEST_STATUS: dict[ExecutionStatus, TestStatus] = {
    ExecutionStatus.PASSED: TestStatus.PASS,
    ExecutionStatus.FAILED: TestStatus.FAIL,
    ExecutionStatus.ERROR: TestStatus.ERROR,
    ExecutionStatus.TIMEOUT: TestStatus.ERROR,
    ExecutionStatus.SKIPPED: TestStatus.SKIP,
}


# ---------------------------------------------------------------------------
# Reporter Agent instructions (for LLM-based analysis in future)
# ---------------------------------------------------------------------------

REPORTER_INSTRUCTIONS = """\
You are the Reporter Agent for a multi-agent test runner system.

Your job is to process test execution results and produce clear,
actionable reports.  You receive real-time test result events and
must:

1. Track pass/fail/error/skip counts
2. Identify patterns in failures (e.g. same module, same error type)
3. Produce a concise summary when the run completes
4. Highlight the most important failures for developer attention

## Output Guidelines

- Lead with the overall status (all passed, or N failures)
- Group failures by category when possible
- Include timing information for slow tests
- Never propose fixes — that is the troubleshooter agent's job
- Keep summaries concise but complete
"""


# ---------------------------------------------------------------------------
# Reporter Agent
# ---------------------------------------------------------------------------


class ReporterAgent(BaseSubAgent):
    """Reporter sub-agent that streams test results in real-time.

    The reporter subscribes to executor output via a callback and
    parses each line through the appropriate OutputParser to produce
    TestResultEvent objects.  Events are emitted immediately to all
    registered reporting channels (ReporterBase instances), enabling
    real-time streaming.

    Usage::

        reporter = ReporterAgent()
        reporter.add_channel(my_cli_reporter)

        # Wire up as executor callback
        executor = TaskExecutor(on_attempt=reporter.on_execution_attempt)

        # Or feed output lines directly
        async for event in reporter.process_output(output_lines, "pytest"):
            print(event)

        # Get final summary
        summary = reporter.get_run_summary()
    """

    def __init__(
        self,
        *,
        parser_registry: OutputParserRegistry | None = None,
        hard_cap_steps: int = 50,
        high_threshold: float = 0.80,
        low_threshold: float = 0.40,
        rollup_config: RollupConfig | None = None,
    ) -> None:
        super().__init__(
            role=AgentRole.REPORTER,
            hard_cap_steps=hard_cap_steps,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
        self._parser_registry = parser_registry or OutputParserRegistry()
        self._channels: list[ReporterBase] = []
        self._stats = RunStatistics()
        self._event_callbacks: list[Callable[[TestResultEvent], None]] = []
        self._async_event_callbacks: list[
            Callable[[TestResultEvent], Any]
        ] = []
        self._rollup_config = rollup_config or RollupConfig()
        self._rollup_generator: RollupSummaryGenerator | None = None

    @property
    def name(self) -> str:
        return "reporter-agent"

    @property
    def instructions(self) -> str:
        return REPORTER_INSTRUCTIONS

    @property
    def stats(self) -> RunStatistics:
        """Current run statistics."""
        return self._stats

    @property
    def channels(self) -> list[ReporterBase]:
        """Registered reporting channels."""
        return list(self._channels)

    @property
    def rollup_config(self) -> RollupConfig:
        """Current rollup configuration."""
        return self._rollup_config

    @rollup_config.setter
    def rollup_config(self, value: RollupConfig) -> None:
        """Update the rollup configuration."""
        self._rollup_config = value
        if self._rollup_generator is not None:
            self._rollup_generator.config = value

    @property
    def rollup_generator(self) -> RollupSummaryGenerator | None:
        """The active rollup generator, if any."""
        return self._rollup_generator

    def get_tools(self) -> list[Any]:
        """Reporter has no LLM tools — it processes output programmatically."""
        return []

    # ----- Channel management -----

    def add_channel(self, channel: ReporterBase) -> None:
        """Register a reporting channel to receive events."""
        self._channels.append(channel)
        logger.info("Reporter: added channel %s", type(channel).__name__)

    def remove_channel(self, channel: ReporterBase) -> None:
        """Remove a reporting channel."""
        self._channels = [c for c in self._channels if c is not channel]

    # ----- Event callback registration (for non-channel consumers) -----

    def on_event(self, callback: Callable[[TestResultEvent], None]) -> None:
        """Register a synchronous callback for each TestResultEvent."""
        self._event_callbacks.append(callback)

    def on_event_async(
        self, callback: Callable[[TestResultEvent], Any]
    ) -> None:
        """Register an async callback for each TestResultEvent."""
        self._async_event_callbacks.append(callback)

    # ----- Core: emit a single event -----

    async def _emit_event(self, event: StreamEvent) -> None:
        """Emit an event to all channels and callbacks immediately."""
        # Send to reporting channels
        for channel in self._channels:
            try:
                await channel.on_event(event)
            except Exception as exc:
                logger.warning(
                    "Reporter: channel %s error: %s",
                    type(channel).__name__,
                    exc,
                )

        # Fire sync callbacks for TestResultEvent
        if isinstance(event, TestResultEvent):
            for cb in self._event_callbacks:
                try:
                    cb(event)
                except Exception as exc:
                    logger.warning("Reporter: sync callback error: %s", exc)

            for cb in self._async_event_callbacks:
                try:
                    result = cb(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.warning("Reporter: async callback error: %s", exc)

    async def _emit_test_result(self, event: TestResultEvent) -> None:
        """Record stats and emit a test result event."""
        self._stats.record(event)
        self.state.record_step(confidence=self.state.current_confidence)
        self.state.add_finding(
            {
                "test": event.test_name,
                "status": event.status.value,
                "duration": event.duration,
            }
        )
        await self._emit_event(event)

    # ----- Process output lines (real-time streaming) -----

    async def process_output(
        self,
        lines: list[str] | str,
        framework: str = "generic",
    ) -> list[TestResultEvent]:
        """Parse executor output and emit TestResultEvents in real-time.

        Each line is fed to the appropriate OutputParser. As soon as a
        test result is detected, it is emitted immediately — NOT batched.

        Args:
            lines: Output lines (list or newline-separated string).
            framework: Framework identifier for parser selection.

        Returns:
            List of all TestResultEvent objects emitted.
        """
        if isinstance(lines, str):
            lines = lines.splitlines()

        parser = self._parser_registry.get(framework)
        emitted: list[TestResultEvent] = []

        for line in lines:
            for event in parser.feed_line(line):
                await self._emit_test_result(event)
                emitted.append(event)

        # Flush any remaining buffered results
        for event in parser.flush():
            await self._emit_test_result(event)
            emitted.append(event)

        return emitted

    # ----- Executor callback integration -----

    def on_execution_attempt(
        self,
        task_id: str,
        record: TaskAttemptRecord,
        result: ExecutionResult,
    ) -> None:
        """Callback for TaskExecutor.on_attempt — synchronous entry point.

        This is designed to be passed as the ``on_attempt`` callback to
        ``TaskExecutor``.  It schedules async processing on the running
        event loop.

        Args:
            task_id: The executor's task identifier.
            record: The full attempt record for this task.
            result: The result of this specific attempt.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._handle_execution_result(task_id, record, result)
            )
        except RuntimeError:
            # No running event loop — fall back to sync emission
            logger.debug(
                "Reporter: no event loop, creating fallback event for %s",
                task_id,
            )
            self._handle_execution_result_sync(task_id, record, result)

    async def _handle_execution_result(
        self,
        task_id: str,
        record: TaskAttemptRecord,
        result: ExecutionResult,
    ) -> None:
        """Async handler: parse executor output and emit events."""
        # Determine framework from the command
        framework = record.command.framework.value if hasattr(
            record.command, "framework"
        ) else "generic"
        will_retry = bool(result.metadata.get("will_retry"))

        # First, try to parse individual test results from stdout
        events = await self.process_output(result.stdout, framework)

        if will_retry and not events:
            await self._emit_event(
                RunEvent(
                    event_type=EventType.LOG,
                    message=(
                        f"Task {task_id} attempt {result.attempt}: "
                        f"{result.status.value}; retrying"
                    ),
                )
            )
            return

        # If no individual results were parsed, emit a single summary event
        # based on the execution result status
        if not events:
            status = _EXEC_TO_TEST_STATUS.get(
                result.status, TestStatus.ERROR
            )
            fallback_event = TestResultEvent(
                test_name=record.command.display or task_id,
                status=status,
                duration=result.duration_seconds,
                stdout=result.stdout[:2000] if result.stdout else "",
                stderr=result.stderr[:2000] if result.stderr else "",
                message=f"Task {task_id} attempt {result.attempt}: {result.status.value}",
            )
            await self._emit_test_result(fallback_event)

    def _handle_execution_result_sync(
        self,
        task_id: str,
        record: TaskAttemptRecord,
        result: ExecutionResult,
    ) -> None:
        """Sync fallback when no event loop is available."""
        if result.metadata.get("will_retry"):
            return
        status = _EXEC_TO_TEST_STATUS.get(result.status, TestStatus.ERROR)
        event = TestResultEvent(
            test_name=record.command.display or task_id,
            status=status,
            duration=result.duration_seconds,
            stdout=result.stdout[:2000] if result.stdout else "",
            stderr=result.stderr[:2000] if result.stderr else "",
            message=f"Task {task_id} attempt {result.attempt}: {result.status.value}",
        )
        self._stats.record(event)
        self.state.record_step()
        # Fire sync callbacks
        for cb in self._event_callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.warning("Reporter: sync callback error: %s", exc)

    # ----- Run lifecycle -----

    async def start_run(self) -> None:
        """Signal the start of a test run to all channels.

        Also starts the periodic rollup summary generator if configured.
        """
        self._stats = RunStatistics()
        run_event = RunEvent(
            event_type=EventType.RUN_STARTED,
            message="Test run started",
        )
        await self._emit_event(run_event)
        for channel in self._channels:
            try:
                await channel.on_run_start()
            except Exception as exc:
                logger.warning("Reporter: on_run_start error: %s", exc)

        # Start periodic rollup generator
        self._rollup_generator = RollupSummaryGenerator(
            source=RunStatisticsAdapter(self._stats),
            on_rollup=self._emit_event,
            config=self._rollup_config,
        )
        await self._rollup_generator.start()

    async def end_run(self) -> dict[str, Any]:
        """Signal the end of a test run and return the summary.

        Stops the periodic rollup generator and emits a final summary that
        includes full failure details with associated logs (stdout, stderr,
        error_details) for every failed/errored test.

        Returns:
            Summary dict with pass/fail/error/skip counts, duration, and
            ``failure_details`` — a list of structured records for each
            failure including captured log output.
        """
        # Stop periodic rollup generator
        if self._rollup_generator is not None:
            await self._rollup_generator.stop()
            self._rollup_generator = None

        self._stats.finalize()
        summary = self._stats.to_full_summary()

        run_event = RunEvent(
            event_type=EventType.RUN_COMPLETED,
            message="Test run completed",
            data=summary,
        )
        await self._emit_event(run_event)

        for channel in self._channels:
            try:
                await channel.on_run_end(summary)
            except Exception as exc:
                logger.warning("Reporter: on_run_end error: %s", exc)

        return summary

    def get_run_summary(self) -> dict[str, Any]:
        """Get current run statistics as a summary dict."""
        return self._stats.to_summary()

    async def generate_rollup_now(self) -> RunEvent:
        """Generate and emit a rollup summary immediately.

        Useful for on-demand status checks outside the periodic timer.

        Returns:
            The generated ``RunEvent`` with rollup data.
        """
        source = RunStatisticsAdapter(self._stats)
        if self._rollup_generator is not None:
            event = self._rollup_generator.generate_now()
        else:
            # Create a temporary generator for one-shot use
            from test_runner.agents.reporter.rollup import (
                RollupSummaryGenerator,
            )

            temp = RollupSummaryGenerator(
                source=source,
                on_rollup=self._emit_event,
                config=self._rollup_config,
            )
            event = temp.generate_now()
        await self._emit_event(event)
        return event

    # ----- BaseSubAgent overrides -----

    def reset_state(self) -> None:
        """Reset state and statistics for a new run."""
        super().reset_state()
        self._stats = RunStatistics()

    def get_handoff_summary(self) -> dict[str, Any]:
        """Include run statistics in the handoff summary."""
        summary = super().get_handoff_summary()
        summary["run_stats"] = self._stats.to_summary()
        return summary


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_reporter_agent(
    config: Config | None = None,
    channels: list[ReporterBase] | None = None,
) -> Agent:
    """Create an OpenAI Agents SDK Agent instance for reporting.

    The reporter agent is primarily programmatic (no LLM tools), but
    wrapping it in an Agent enables the orchestrator to delegate
    summary generation and failure analysis via LLM.

    Args:
        config: Application configuration. Uses defaults if None.
        channels: Initial reporting channels to register.

    Returns:
        An agents.Agent configured as the reporter sub-agent.
    """
    reporter = ReporterAgent()

    for ch in (channels or []):
        reporter.add_channel(ch)

    model = config.model_id if config else "gpt-4o"

    agent = Agent(
        name=reporter.name,
        instructions=reporter.instructions,
        tools=reporter.get_tools(),
        model=model,
    )
    return agent
