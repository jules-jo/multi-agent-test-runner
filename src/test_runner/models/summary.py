"""Summary data model for test run results.

Provides structured, serializable models for the final test run summary
including pass/fail counts, execution time, and detailed failure records
with captured logs. This is the canonical output model used by the reporter
agent and consumed by all reporting channels.

Design decisions:
- Pydantic BaseModel for validation, serialization, and schema generation
- Immutable (frozen) models for safe cross-agent sharing
- FailureDetail captures full context: logs, stack traces, error type
- TestRunSummary is the top-level envelope returned by end_run()
- Factory method from_progress_snapshot() bridges the ProgressTracker world
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field, computed_field


class TestOutcome(str, Enum):
    """Outcome of an individual test case."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class FailureDetail(BaseModel, frozen=True):
    """Structured record of a single test failure or error.

    Captures everything needed for troubleshooting: the test identity,
    error classification, full log output, and optional source location.

    Attributes:
        test_id: Unique identifier (e.g. ``tests/test_foo.py::test_bar``).
        test_name: Human-readable test name.
        outcome: Whether this was a failure (assertion) or error (exception).
        error_message: Short description of the failure.
        error_type: Exception class name or error category (e.g. ``AssertionError``).
        stack_trace: Full stack trace if available.
        stdout: Captured standard output from the test.
        stderr: Captured standard error from the test.
        log_output: Combined/additional log output captured during the test.
        duration_seconds: How long the test ran before failing.
        file_path: Source file containing the test.
        line_number: Line number of the failure, if known.
        framework: Test framework that ran this test (e.g. ``pytest``).
        metadata: Additional framework-specific data.
    """

    test_id: str
    test_name: str
    outcome: TestOutcome
    error_message: str = ""
    error_type: str = ""
    stack_trace: str = ""
    stdout: str = ""
    stderr: str = ""
    log_output: str = ""
    duration_seconds: float = 0.0
    file_path: str = ""
    line_number: Optional[int] = None
    framework: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_logs(self) -> bool:
        """True if any log output (stdout, stderr, or log_output) was captured."""
        return bool(self.stdout or self.stderr or self.log_output)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def combined_logs(self) -> str:
        """All captured output concatenated with section headers."""
        parts: list[str] = []
        if self.stdout:
            parts.append(f"--- stdout ---\n{self.stdout}")
        if self.stderr:
            parts.append(f"--- stderr ---\n{self.stderr}")
        if self.log_output:
            parts.append(f"--- logs ---\n{self.log_output}")
        if self.stack_trace:
            parts.append(f"--- traceback ---\n{self.stack_trace}")
        return "\n\n".join(parts)


class TestCaseSummary(BaseModel, frozen=True):
    """Summary record for a single test case (pass or fail).

    Lighter than FailureDetail — used for the full results list.
    """

    test_id: str
    test_name: str
    outcome: TestOutcome
    duration_seconds: float = 0.0
    framework: str = ""
    error_message: str = ""


class TestRunSummary(BaseModel, frozen=True):
    """Top-level summary of a complete test run.

    This is the canonical output model returned by the reporter agent's
    ``end_run()`` method and consumed by all reporting channels.

    Attributes:
        run_id: Unique identifier for this run.
        total: Total number of tests executed.
        passed: Number of tests that passed.
        failed: Number of tests that failed (assertion failures).
        errors: Number of tests that errored (unexpected exceptions).
        skipped: Number of tests that were skipped.
        start_time: Unix timestamp when the run started.
        end_time: Unix timestamp when the run ended.
        duration_seconds: Wall-clock duration of the entire run.
        success: True if all tests passed (no failures or errors).
        framework: Primary test framework used (if single-framework run).
        failures: Structured details for every failed/errored test.
        results: Summary records for all test cases.
        ai_analysis: Optional LLM-generated analysis of failures.
        metadata: Extensible metadata (execution target, config, etc.).
    """

    run_id: str = ""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    success: bool = True
    framework: str = ""
    failures: list[FailureDetail] = Field(default_factory=list)
    results: list[TestCaseSummary] = Field(default_factory=list)
    ai_analysis: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failure_count(self) -> int:
        """Total failures + errors."""
        return self.failed + self.errors

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        """Pass rate as a fraction in [0.0, 1.0]. Returns 0.0 if no tests ran."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_failures(self) -> bool:
        """True if any tests failed or errored."""
        return self.failure_count > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pending(self) -> int:
        """Tests unaccounted for (total - passed - failed - errors - skipped)."""
        return self.total - self.passed - self.failed - self.errors - self.skipped

    def failure_summary_lines(self) -> list[str]:
        """One-line summary for each failure, suitable for CLI or chat output."""
        lines: list[str] = []
        for f in self.failures:
            prefix = "ERROR" if f.outcome == TestOutcome.ERROR else "FAIL"
            loc = f""
            if f.file_path:
                loc = f" ({f.file_path}"
                if f.line_number is not None:
                    loc += f":{f.line_number}"
                loc += ")"
            msg = f.error_message[:120] if f.error_message else "no message"
            lines.append(f"  [{prefix}] {f.test_name}{loc}: {msg}")
        return lines

    def to_report_dict(self) -> dict[str, Any]:
        """Serializable dict suitable for JSON reporting and channel payloads.

        This is a flattened view optimized for reporting — use
        ``.model_dump()`` for the full Pydantic serialization.
        """
        return {
            "run_id": self.run_id,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "failure_count": self.failure_count,
            "pass_rate": round(self.pass_rate, 4),
            "duration_seconds": round(self.duration_seconds, 3),
            "success": self.success,
            "has_failures": self.has_failures,
            "framework": self.framework,
            "ai_analysis": self.ai_analysis,
            "failures": [
                {
                    "test_id": f.test_id,
                    "test_name": f.test_name,
                    "outcome": f.outcome.value,
                    "error_message": f.error_message,
                    "error_type": f.error_type,
                    "has_logs": f.has_logs,
                    "duration_seconds": f.duration_seconds,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "stack_trace": f.stack_trace,
                    "stdout": f.stdout,
                    "stderr": f.stderr,
                    "log_output": f.log_output,
                    "combined_logs": f.combined_logs,
                }
                for f in self.failures
            ],
        }

    # -- Factory methods ---------------------------------------------------

    @classmethod
    def from_counts(
        cls,
        *,
        total: int,
        passed: int,
        failed: int,
        errors: int = 0,
        skipped: int = 0,
        duration_seconds: float = 0.0,
        failures: Sequence[FailureDetail] | None = None,
        **kwargs: Any,
    ) -> TestRunSummary:
        """Create a summary from aggregate counts.

        Convenience factory for building a summary when you have counts
        but not individual TestResult objects.
        """
        return cls(
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_seconds=duration_seconds,
            success=(failed == 0 and errors == 0),
            failures=list(failures) if failures else [],
            **kwargs,
        )

    @classmethod
    def from_test_result_events(
        cls,
        events: Sequence[Any],
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        run_id: str = "",
        framework: str = "",
        ai_analysis: str = "",
    ) -> TestRunSummary:
        """Build a summary from a sequence of TestResultEvent objects.

        Bridges the reporting.events world to the summary model. Imports
        TestResultEvent/TestStatus locally to avoid circular imports.
        """
        from test_runner.reporting.events import TestResultEvent, TestStatus

        passed = failed = errors = skipped = 0
        failure_details: list[FailureDetail] = []
        case_summaries: list[TestCaseSummary] = []

        for ev in events:
            if not isinstance(ev, TestResultEvent):
                continue

            # Map event status to outcome
            outcome_map = {
                TestStatus.PASS: TestOutcome.PASSED,
                TestStatus.FAIL: TestOutcome.FAILED,
                TestStatus.ERROR: TestOutcome.ERROR,
                TestStatus.SKIP: TestOutcome.SKIPPED,
            }
            outcome = outcome_map.get(ev.status, TestOutcome.ERROR)

            # Count
            if outcome == TestOutcome.PASSED:
                passed += 1
            elif outcome == TestOutcome.FAILED:
                failed += 1
            elif outcome == TestOutcome.ERROR:
                errors += 1
            elif outcome == TestOutcome.SKIPPED:
                skipped += 1

            # Case summary for every test
            case_summaries.append(
                TestCaseSummary(
                    test_id=ev.test_name,
                    test_name=ev.test_name,
                    outcome=outcome,
                    duration_seconds=ev.duration,
                    framework=framework,
                    error_message=ev.message if outcome in (TestOutcome.FAILED, TestOutcome.ERROR) else "",
                )
            )

            # Failure detail for failures/errors only
            if outcome in (TestOutcome.FAILED, TestOutcome.ERROR):
                failure_details.append(
                    FailureDetail(
                        test_id=ev.test_name,
                        test_name=ev.test_name,
                        outcome=outcome,
                        error_message=ev.message,
                        error_type=ev.error_details if hasattr(ev, "error_details") else "",
                        stdout=ev.stdout if hasattr(ev, "stdout") else "",
                        stderr=ev.stderr if hasattr(ev, "stderr") else "",
                        duration_seconds=ev.duration,
                        file_path=ev.file_path if hasattr(ev, "file_path") else "",
                        line_number=ev.line_number if hasattr(ev, "line_number") else None,
                        framework=framework,
                    )
                )

        total = passed + failed + errors + skipped
        now = time.time()
        actual_start = start_time or now
        actual_end = end_time or now
        duration = actual_end - actual_start if start_time else 0.0

        return cls(
            run_id=run_id,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            start_time=actual_start,
            end_time=actual_end,
            duration_seconds=duration,
            success=(failed == 0 and errors == 0 and total > 0),
            framework=framework,
            failures=failure_details,
            results=case_summaries,
            ai_analysis=ai_analysis,
        )

    @classmethod
    def from_progress_snapshot(
        cls,
        snapshot: Any,
        *,
        run_id: str = "",
        framework: str = "",
        ai_analysis: str = "",
    ) -> TestRunSummary:
        """Build a summary from a ProgressSnapshot.

        Bridges the models.progress world to the summary model. Imports
        ProgressSnapshot locally to avoid circular imports.
        """
        from test_runner.models.progress import ProgressSnapshot, TestResult, TestStatus

        if not isinstance(snapshot, ProgressSnapshot):
            raise TypeError(f"Expected ProgressSnapshot, got {type(snapshot)}")

        failure_details: list[FailureDetail] = []
        case_summaries: list[TestCaseSummary] = []

        status_to_outcome = {
            TestStatus.PASSED: TestOutcome.PASSED,
            TestStatus.FAILED: TestOutcome.FAILED,
            TestStatus.ERROR: TestOutcome.ERROR,
            TestStatus.SKIPPED: TestOutcome.SKIPPED,
        }

        for result in snapshot.results:
            outcome = status_to_outcome.get(result.status, TestOutcome.ERROR)

            case_summaries.append(
                TestCaseSummary(
                    test_id=result.test_id,
                    test_name=result.name,
                    outcome=outcome,
                    duration_seconds=result.duration_seconds,
                    framework=result.framework or framework,
                    error_message=result.error_message if outcome in (TestOutcome.FAILED, TestOutcome.ERROR) else "",
                )
            )

            if outcome in (TestOutcome.FAILED, TestOutcome.ERROR):
                failure_details.append(
                    FailureDetail(
                        test_id=result.test_id,
                        test_name=result.name,
                        outcome=outcome,
                        error_message=result.error_message,
                        stdout=result.output,
                        duration_seconds=result.duration_seconds,
                        framework=result.framework or framework,
                        metadata=result.metadata,
                    )
                )

        return cls(
            run_id=run_id,
            total=snapshot.total,
            passed=snapshot.passed,
            failed=snapshot.failed,
            errors=snapshot.errored,
            skipped=snapshot.skipped,
            start_time=snapshot.start_time,
            duration_seconds=snapshot.elapsed_seconds,
            success=snapshot.success if snapshot.is_complete else False,
            framework=framework,
            failures=failure_details,
            results=case_summaries,
            ai_analysis=ai_analysis,
        )
