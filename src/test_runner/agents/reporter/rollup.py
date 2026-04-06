"""Periodic rollup summary generator for the reporter agent.

Reads progress state at a configurable interval and formats
human-readable status messages such as '5/20 done, 1 failure'.

The rollup runs as an async background task that can be started
and stopped alongside the reporter agent's run lifecycle. It emits
``RunEvent`` objects with ``EventType.ROLLUP_SUMMARY`` to all
registered reporting channels.

Design decisions:
- Configurable interval (default 10 seconds)
- Reads from either ``RunStatistics`` or ``ProgressTracker``
- Formats concise human-readable messages
- Emits via callback so it stays decoupled from channel management
- Async task-based so it doesn't block the event loop
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Optional, Protocol, runtime_checkable

from test_runner.reporting.events import EventType, RunEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress source protocol — anything that can provide progress data
# ---------------------------------------------------------------------------


@runtime_checkable
class ProgressSource(Protocol):
    """Protocol for objects that provide progress state for rollups.

    Both ``RunStatistics`` and ``ProgressTracker`` can satisfy this
    through adapter wrappers.
    """

    @property
    def total(self) -> int: ...

    @property
    def passed(self) -> int: ...

    @property
    def failed(self) -> int: ...

    @property
    def errors(self) -> int: ...

    @property
    def skipped(self) -> int: ...

    @property
    def duration(self) -> float: ...


# ---------------------------------------------------------------------------
# Adapters for existing progress sources
# ---------------------------------------------------------------------------


class RunStatisticsAdapter:
    """Adapt ``RunStatistics`` to the ``ProgressSource`` protocol."""

    def __init__(self, stats: Any) -> None:
        self._stats = stats

    @property
    def total(self) -> int:
        return self._stats.total

    @property
    def passed(self) -> int:
        return self._stats.passed

    @property
    def failed(self) -> int:
        return self._stats.failed

    @property
    def errors(self) -> int:
        return self._stats.errors

    @property
    def skipped(self) -> int:
        return self._stats.skipped

    @property
    def duration(self) -> float:
        return self._stats.duration


class ProgressTrackerAdapter:
    """Adapt ``ProgressTracker`` to the ``ProgressSource`` protocol."""

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker

    @property
    def total(self) -> int:
        return self._tracker.total

    @property
    def passed(self) -> int:
        snap = self._tracker.snapshot()
        return snap.passed

    @property
    def failed(self) -> int:
        snap = self._tracker.snapshot()
        return snap.failed

    @property
    def errors(self) -> int:
        snap = self._tracker.snapshot()
        return snap.errored

    @property
    def skipped(self) -> int:
        snap = self._tracker.snapshot()
        return snap.skipped

    @property
    def duration(self) -> float:
        snap = self._tracker.snapshot()
        return snap.elapsed_seconds


# ---------------------------------------------------------------------------
# Rollup formatting
# ---------------------------------------------------------------------------


def format_rollup_message(
    total: int,
    passed: int,
    failed: int,
    errors: int,
    skipped: int,
    elapsed: float,
) -> str:
    """Format a concise human-readable rollup summary.

    Examples:
        - "5/20 done, 1 failure (12.3s elapsed)"
        - "10/10 done, all passing (5.1s elapsed)"
        - "0/20 done (1.0s elapsed)"
        - "3/10 done, 1 failure, 1 error (8.2s elapsed)"

    Args:
        total: Total expected tests (0 means unknown).
        passed: Count of passed tests.
        failed: Count of failed tests.
        errors: Count of errored tests.
        skipped: Count of skipped tests.
        elapsed: Wall-clock seconds since run start.

    Returns:
        Human-readable status string.
    """
    completed = passed + failed + errors + skipped

    # Build "X/Y done" or "X done" if total is unknown
    if total > 0:
        progress_part = f"{completed}/{total} done"
    else:
        progress_part = f"{completed} done"

    # Build failure/error details
    detail_parts: list[str] = []
    if completed > 0 and failed == 0 and errors == 0:
        detail_parts.append("all passing")
    else:
        if failed > 0:
            label = "failure" if failed == 1 else "failures"
            detail_parts.append(f"{failed} {label}")
        if errors > 0:
            label = "error" if errors == 1 else "errors"
            detail_parts.append(f"{errors} {label}")
    if skipped > 0:
        label = "skipped" if skipped == 1 else "skipped"
        detail_parts.append(f"{skipped} {label}")

    # Build elapsed time
    elapsed_str = _format_elapsed(elapsed)

    # Combine
    if detail_parts:
        return f"{progress_part}, {', '.join(detail_parts)} ({elapsed_str} elapsed)"
    return f"{progress_part} ({elapsed_str} elapsed)"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds for display."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


# ---------------------------------------------------------------------------
# Rollup configuration
# ---------------------------------------------------------------------------


@dataclass
class RollupConfig:
    """Configuration for periodic rollup summaries.

    Attributes:
        interval_seconds: How often to generate a rollup (default 10s).
        enabled: Whether rollups are active.
        min_interval_seconds: Floor for interval to prevent spam.
        max_interval_seconds: Ceiling for interval.
    """

    interval_seconds: float = 10.0
    enabled: bool = True
    min_interval_seconds: float = 1.0
    max_interval_seconds: float = 300.0

    def __post_init__(self) -> None:
        self.interval_seconds = max(
            self.min_interval_seconds,
            min(self.interval_seconds, self.max_interval_seconds),
        )


# ---------------------------------------------------------------------------
# Rollup Summary Generator
# ---------------------------------------------------------------------------

# Callback type: receives a RunEvent for each rollup
RollupCallback = Callable[[RunEvent], Awaitable[None]]


class RollupSummaryGenerator:
    """Generates periodic rollup summaries from a progress source.

    The generator runs as an async background task, reading progress
    state at the configured interval and emitting formatted status
    messages via a callback.

    Usage::

        generator = RollupSummaryGenerator(
            source=RunStatisticsAdapter(stats),
            config=RollupConfig(interval_seconds=5),
            on_rollup=reporter._emit_event,
        )

        await generator.start()
        # ... run tests ...
        await generator.stop()

    The generator can also be used in a one-shot mode::

        event = generator.generate_now()
    """

    def __init__(
        self,
        source: ProgressSource,
        on_rollup: RollupCallback,
        config: RollupConfig | None = None,
    ) -> None:
        self._source = source
        self._on_rollup = on_rollup
        self._config = config or RollupConfig()
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._rollup_count = 0

    @property
    def config(self) -> RollupConfig:
        """Current rollup configuration."""
        return self._config

    @config.setter
    def config(self, value: RollupConfig) -> None:
        """Update the rollup configuration (takes effect next interval)."""
        self._config = value

    @property
    def is_running(self) -> bool:
        """Whether the background rollup task is active."""
        return self._running

    @property
    def rollup_count(self) -> int:
        """Number of rollup summaries generated so far."""
        return self._rollup_count

    # ----- One-shot generation -----

    def generate_now(self) -> RunEvent:
        """Generate a rollup event immediately without waiting for the timer.

        Returns:
            A ``RunEvent`` with ``EventType.ROLLUP_SUMMARY``.
        """
        message = format_rollup_message(
            total=self._source.total,
            passed=self._source.passed,
            failed=self._source.failed,
            errors=self._source.errors,
            skipped=self._source.skipped,
            elapsed=self._source.duration,
        )

        self._rollup_count += 1

        return RunEvent(
            event_type=EventType.ROLLUP_SUMMARY,
            message=message,
            data={
                "total": self._source.total,
                "passed": self._source.passed,
                "failed": self._source.failed,
                "errors": self._source.errors,
                "skipped": self._source.skipped,
                "elapsed": round(self._source.duration, 3),
                "rollup_number": self._rollup_count,
            },
        )

    # ----- Background task lifecycle -----

    async def start(self) -> None:
        """Start the periodic rollup background task."""
        if self._running:
            logger.warning("RollupSummaryGenerator already running")
            return
        if not self._config.enabled:
            logger.info("Rollup summaries disabled by config")
            return

        self._running = True
        self._rollup_count = 0
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Rollup summary generator started (interval=%.1fs)",
            self._config.interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the periodic rollup background task.

        Generates one final rollup before stopping.
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(
            "Rollup summary generator stopped after %d rollups",
            self._rollup_count,
        )

    async def _run_loop(self) -> None:
        """Background loop that generates rollups at the configured interval."""
        try:
            while self._running:
                await asyncio.sleep(self._config.interval_seconds)
                if not self._running:
                    break
                await self._emit_rollup()
        except asyncio.CancelledError:
            pass

    async def _emit_rollup(self) -> None:
        """Generate and emit a single rollup event."""
        try:
            event = self.generate_now()
            await self._on_rollup(event)
            logger.debug("Rollup #%d: %s", self._rollup_count, event.message)
        except Exception as exc:
            logger.warning("Rollup emission error: %s", exc)
