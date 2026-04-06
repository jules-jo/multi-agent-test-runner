"""Progress tracking data model for test execution.

Provides a thread-safe, observable progress tracker that aggregates
test counts (total, completed, passed, failed, skipped) during test runs.
The executor agent updates this model, and the orchestrator hub shares it
across sub-agents via RunState.

Design decisions:
- Thread-safe via threading.Lock for concurrent executor updates
- Callback-based observation for real-time streaming/reporting
- Immutable snapshots for safe cross-agent reads
- Timing data (start/end/elapsed) for reporting completeness
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class TestStatus(str, Enum):
    """Status of an individual test result."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class TestResult:
    """Immutable record of a single test execution result.

    Attributes:
        test_id: Unique identifier for the test (e.g. file::test_name).
        name: Human-readable name of the test.
        status: Pass/fail/skip/error outcome.
        duration_seconds: Wall-clock time for this test.
        output: Captured stdout/stderr (truncated if needed).
        error_message: Error or failure message, if any.
        framework: Test framework that ran this test (e.g. "pytest", "jest").
        metadata: Additional framework-specific data.
    """

    test_id: str
    name: str
    status: TestStatus
    duration_seconds: float = 0.0
    output: str = ""
    error_message: str = ""
    framework: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status in (TestStatus.FAILED, TestStatus.ERROR)


@dataclass(frozen=True)
class ProgressSnapshot:
    """Immutable point-in-time snapshot of progress counters.

    Safe to pass across agent boundaries without synchronization.
    """

    total: int
    completed: int
    passed: int
    failed: int
    skipped: int
    errored: int
    start_time: Optional[float]
    elapsed_seconds: float
    is_running: bool
    is_complete: bool
    results: tuple[TestResult, ...]

    @property
    def pending(self) -> int:
        """Tests remaining to be executed."""
        return self.total - self.completed

    @property
    def pass_rate(self) -> float:
        """Pass rate as a fraction in [0.0, 1.0]. Returns 0.0 if no tests completed."""
        if self.completed == 0:
            return 0.0
        return self.passed / self.completed

    @property
    def success(self) -> bool:
        """True if all completed tests passed (no failures or errors)."""
        return self.is_complete and self.failed == 0 and self.errored == 0

    def summary(self) -> dict[str, Any]:
        """Serializable summary for reporting and logging."""
        return {
            "total": self.total,
            "completed": self.completed,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errored": self.errored,
            "pending": self.pending,
            "pass_rate": round(self.pass_rate, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "is_running": self.is_running,
            "is_complete": self.is_complete,
            "success": self.success if self.is_complete else None,
        }


# Type alias for progress callbacks
ProgressCallback = Callable[[ProgressSnapshot], None]


class ProgressTracker:
    """Thread-safe mutable progress tracker for test execution.

    The executor agent calls :meth:`record_result` as each test completes.
    Observers (reporter, CLI streamer) register via :meth:`on_progress`
    and receive :class:`ProgressSnapshot` callbacks on every update.

    Usage::

        tracker = ProgressTracker(total=10)
        tracker.on_progress(lambda snap: print(snap.summary()))
        tracker.start()

        # Executor records results as tests complete
        tracker.record_result(TestResult(
            test_id="tests/test_foo.py::test_bar",
            name="test_bar",
            status=TestStatus.PASSED,
            duration_seconds=0.42,
        ))

        tracker.finish()
        final = tracker.snapshot()
        assert final.is_complete
    """

    def __init__(self, total: int = 0) -> None:
        self._lock = threading.Lock()
        self._total = total
        self._passed = 0
        self._failed = 0
        self._skipped = 0
        self._errored = 0
        self._results: list[TestResult] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._callbacks: list[ProgressCallback] = []

    # -- Configuration -------------------------------------------------------

    def set_total(self, total: int) -> None:
        """Update the total expected test count (e.g. after discovery)."""
        with self._lock:
            self._total = total
            self._notify()

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Mark the start of test execution."""
        with self._lock:
            self._start_time = time.monotonic()
            self._notify()

    def finish(self) -> None:
        """Mark the end of test execution."""
        with self._lock:
            self._end_time = time.monotonic()
            self._notify()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._start_time is not None and self._end_time is None

    @property
    def is_complete(self) -> bool:
        with self._lock:
            return self._end_time is not None

    # -- Recording results ---------------------------------------------------

    def record_result(self, result: TestResult) -> None:
        """Record a single test result and update counters.

        This is the primary method called by the executor agent.
        """
        with self._lock:
            self._results.append(result)
            if result.status == TestStatus.PASSED:
                self._passed += 1
            elif result.status == TestStatus.FAILED:
                self._failed += 1
            elif result.status == TestStatus.SKIPPED:
                self._skipped += 1
            elif result.status == TestStatus.ERROR:
                self._errored += 1
            self._notify()

    def record_results(self, results: list[TestResult]) -> None:
        """Record multiple test results in a single batch."""
        with self._lock:
            for result in results:
                self._results.append(result)
                if result.status == TestStatus.PASSED:
                    self._passed += 1
                elif result.status == TestStatus.FAILED:
                    self._failed += 1
                elif result.status == TestStatus.SKIPPED:
                    self._skipped += 1
                elif result.status == TestStatus.ERROR:
                    self._errored += 1
            self._notify()

    # -- Observation ---------------------------------------------------------

    def on_progress(self, callback: ProgressCallback) -> None:
        """Register a callback invoked on every progress update.

        Callbacks receive an immutable :class:`ProgressSnapshot`.
        """
        with self._lock:
            self._callbacks.append(callback)

    def _notify(self) -> None:
        """Notify all registered callbacks with a current snapshot.

        Must be called while holding ``self._lock``.
        """
        if not self._callbacks:
            return
        snap = self._snapshot_unlocked()
        for cb in self._callbacks:
            try:
                cb(snap)
            except Exception:
                pass  # Don't let a bad callback break the tracker

    # -- Snapshots -----------------------------------------------------------

    def snapshot(self) -> ProgressSnapshot:
        """Return an immutable snapshot of current progress."""
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> ProgressSnapshot:
        """Build a snapshot without acquiring the lock (caller must hold it)."""
        now = time.monotonic()
        if self._start_time is not None:
            end = self._end_time if self._end_time is not None else now
            elapsed = end - self._start_time
        else:
            elapsed = 0.0

        completed = self._passed + self._failed + self._skipped + self._errored

        return ProgressSnapshot(
            total=self._total,
            completed=completed,
            passed=self._passed,
            failed=self._failed,
            skipped=self._skipped,
            errored=self._errored,
            start_time=self._start_time,
            elapsed_seconds=elapsed,
            is_running=self._start_time is not None and self._end_time is None,
            is_complete=self._end_time is not None,
            results=tuple(self._results),
        )

    # -- Reset ---------------------------------------------------------------

    def reset(self) -> None:
        """Reset all counters and state for a fresh run."""
        with self._lock:
            self._total = 0
            self._passed = 0
            self._failed = 0
            self._skipped = 0
            self._errored = 0
            self._results.clear()
            self._start_time = None
            self._end_time = None
