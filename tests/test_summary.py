"""Tests for the summary data model (TestRunSummary, FailureDetail, etc.)."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from test_runner.models.summary import (
    FailureDetail,
    TestCaseSummary,
    TestOutcome,
    TestRunSummary,
)
from test_runner.models.progress import (
    ProgressSnapshot,
    ProgressTracker,
    TestResult,
    TestStatus,
)
from test_runner.reporting.events import (
    TestResultEvent,
    TestStatus as EventTestStatus,
)


# ---------------------------------------------------------------------------
# FailureDetail
# ---------------------------------------------------------------------------


class TestFailureDetail:
    """Tests for the FailureDetail model."""

    def test_minimal_construction(self):
        fd = FailureDetail(
            test_id="tests/test_foo.py::test_bar",
            test_name="test_bar",
            outcome=TestOutcome.FAILED,
        )
        assert fd.test_id == "tests/test_foo.py::test_bar"
        assert fd.test_name == "test_bar"
        assert fd.outcome == TestOutcome.FAILED
        assert fd.error_message == ""
        assert fd.stdout == ""
        assert fd.stderr == ""
        assert fd.has_logs is False
        assert fd.combined_logs == ""

    def test_full_construction_with_logs(self):
        fd = FailureDetail(
            test_id="tests/test_foo.py::test_bar",
            test_name="test_bar",
            outcome=TestOutcome.ERROR,
            error_message="ZeroDivisionError: division by zero",
            error_type="ZeroDivisionError",
            stack_trace="Traceback ...\n  File ...\nZeroDivisionError",
            stdout="some stdout output",
            stderr="some stderr output",
            log_output="DEBUG: entering test_bar",
            duration_seconds=1.5,
            file_path="tests/test_foo.py",
            line_number=42,
            framework="pytest",
            metadata={"xfail": False},
        )
        assert fd.has_logs is True
        assert "--- stdout ---" in fd.combined_logs
        assert "--- stderr ---" in fd.combined_logs
        assert "--- logs ---" in fd.combined_logs
        assert "--- traceback ---" in fd.combined_logs
        assert fd.duration_seconds == 1.5
        assert fd.line_number == 42

    def test_has_logs_with_only_stdout(self):
        fd = FailureDetail(
            test_id="t", test_name="t", outcome=TestOutcome.FAILED, stdout="output"
        )
        assert fd.has_logs is True

    def test_has_logs_with_only_stderr(self):
        fd = FailureDetail(
            test_id="t", test_name="t", outcome=TestOutcome.FAILED, stderr="err"
        )
        assert fd.has_logs is True

    def test_frozen_immutability(self):
        fd = FailureDetail(
            test_id="t", test_name="t", outcome=TestOutcome.FAILED
        )
        with pytest.raises(ValidationError):
            fd.test_name = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self):
        fd = FailureDetail(
            test_id="tests/test_x.py::test_y",
            test_name="test_y",
            outcome=TestOutcome.ERROR,
            error_message="boom",
            stdout="out",
            stderr="err",
        )
        d = fd.model_dump()
        assert d["test_id"] == "tests/test_x.py::test_y"
        assert d["outcome"] == "error"
        assert d["has_logs"] is True
        # Reconstruct
        fd2 = FailureDetail.model_validate(d)
        assert fd2 == fd


# ---------------------------------------------------------------------------
# TestCaseSummary
# ---------------------------------------------------------------------------


class TestTestCaseSummary:
    def test_construction(self):
        cs = TestCaseSummary(
            test_id="t1",
            test_name="test_one",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.05,
        )
        assert cs.outcome == TestOutcome.PASSED
        assert cs.duration_seconds == 0.05

    def test_frozen(self):
        cs = TestCaseSummary(
            test_id="t1", test_name="test_one", outcome=TestOutcome.PASSED
        )
        with pytest.raises(ValidationError):
            cs.outcome = TestOutcome.FAILED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestRunSummary — construction and computed fields
# ---------------------------------------------------------------------------


class TestTestRunSummary:
    def test_empty_summary(self):
        s = TestRunSummary()
        assert s.total == 0
        assert s.passed == 0
        assert s.failed == 0
        assert s.errors == 0
        assert s.skipped == 0
        assert s.pass_rate == 0.0
        assert s.failure_count == 0
        assert s.has_failures is False
        assert s.success is True  # vacuously true, no failures

    def test_all_passed(self):
        s = TestRunSummary(total=5, passed=5, duration_seconds=1.2)
        assert s.pass_rate == 1.0
        assert s.success is True
        assert s.has_failures is False
        assert s.failure_count == 0
        assert s.pending == 0

    def test_mixed_results(self):
        s = TestRunSummary(
            total=10,
            passed=7,
            failed=2,
            errors=1,
            skipped=0,
            duration_seconds=5.5,
            success=False,
        )
        assert s.pass_rate == 0.7
        assert s.failure_count == 3
        assert s.has_failures is True
        assert s.success is False

    def test_pending_calculation(self):
        s = TestRunSummary(total=10, passed=3, failed=1, errors=0, skipped=1)
        assert s.pending == 5  # 10 - 3 - 1 - 0 - 1

    def test_failure_summary_lines(self):
        failures = [
            FailureDetail(
                test_id="t1",
                test_name="test_alpha",
                outcome=TestOutcome.FAILED,
                error_message="assert 1 == 2",
                file_path="tests/test_a.py",
                line_number=10,
            ),
            FailureDetail(
                test_id="t2",
                test_name="test_beta",
                outcome=TestOutcome.ERROR,
                error_message="RuntimeError: oops",
            ),
        ]
        s = TestRunSummary(
            total=5, passed=3, failed=1, errors=1, failures=failures, success=False
        )
        lines = s.failure_summary_lines()
        assert len(lines) == 2
        assert "[FAIL]" in lines[0]
        assert "test_alpha" in lines[0]
        assert "tests/test_a.py:10" in lines[0]
        assert "[ERROR]" in lines[1]
        assert "test_beta" in lines[1]

    def test_to_report_dict(self):
        failure = FailureDetail(
            test_id="t1",
            test_name="test_x",
            outcome=TestOutcome.FAILED,
            error_message="bad",
            stdout="some output",
        )
        s = TestRunSummary(
            run_id="run-123",
            total=3,
            passed=2,
            failed=1,
            duration_seconds=2.5,
            success=False,
            failures=[failure],
        )
        d = s.to_report_dict()
        assert d["run_id"] == "run-123"
        assert d["total"] == 3
        assert d["pass_rate"] == pytest.approx(0.6667, abs=0.001)
        assert d["has_failures"] is True
        assert len(d["failures"]) == 1
        assert d["failures"][0]["has_logs"] is True

    def test_to_report_dict_includes_logs_in_failures(self):
        """Failure entries in to_report_dict include associated log data."""
        failure = FailureDetail(
            test_id="t1",
            test_name="test_x",
            outcome=TestOutcome.FAILED,
            error_message="assert 1 == 2",
            error_type="AssertionError",
            stack_trace="Traceback ...\nAssertionError",
            stdout="captured stdout line",
            stderr="captured stderr line",
            log_output="DEBUG: test_x entered",
            file_path="tests/test_x.py",
            line_number=42,
        )
        s = TestRunSummary(
            total=1, failed=1, success=False, failures=[failure]
        )
        d = s.to_report_dict()
        f = d["failures"][0]
        assert f["stdout"] == "captured stdout line"
        assert f["stderr"] == "captured stderr line"
        assert f["log_output"] == "DEBUG: test_x entered"
        assert f["stack_trace"] == "Traceback ...\nAssertionError"
        assert f["file_path"] == "tests/test_x.py"
        assert f["line_number"] == 42
        assert "--- stdout ---" in f["combined_logs"]
        assert "--- stderr ---" in f["combined_logs"]
        assert "--- logs ---" in f["combined_logs"]
        assert "--- traceback ---" in f["combined_logs"]

    def test_to_report_dict_failure_without_logs(self):
        """Failure without captured output still serializes correctly."""
        failure = FailureDetail(
            test_id="t1",
            test_name="test_y",
            outcome=TestOutcome.ERROR,
            error_message="RuntimeError: boom",
        )
        s = TestRunSummary(
            total=1, errors=1, success=False, failures=[failure]
        )
        d = s.to_report_dict()
        f = d["failures"][0]
        assert f["has_logs"] is False
        assert f["stdout"] == ""
        assert f["stderr"] == ""
        assert f["combined_logs"] == ""

    def test_frozen(self):
        s = TestRunSummary(total=1, passed=1)
        with pytest.raises(ValidationError):
            s.total = 99  # type: ignore[misc]

    def test_serialization_roundtrip(self):
        s = TestRunSummary(
            run_id="abc",
            total=2,
            passed=1,
            failed=1,
            duration_seconds=1.0,
            success=False,
            failures=[
                FailureDetail(
                    test_id="t1",
                    test_name="test_fail",
                    outcome=TestOutcome.FAILED,
                    error_message="nope",
                )
            ],
        )
        d = s.model_dump()
        s2 = TestRunSummary.model_validate(d)
        assert s2.run_id == "abc"
        assert s2.total == 2
        assert len(s2.failures) == 1
        assert s2.failures[0].error_message == "nope"


# ---------------------------------------------------------------------------
# Factory: from_counts
# ---------------------------------------------------------------------------


class TestFromCounts:
    def test_basic(self):
        s = TestRunSummary.from_counts(
            total=10, passed=8, failed=1, errors=1, duration_seconds=3.0
        )
        assert s.total == 10
        assert s.passed == 8
        assert s.failed == 1
        assert s.errors == 1
        assert s.success is False
        assert s.duration_seconds == 3.0

    def test_all_passed_success(self):
        s = TestRunSummary.from_counts(total=5, passed=5, failed=0)
        assert s.success is True

    def test_with_failures(self):
        fd = FailureDetail(
            test_id="t1", test_name="t", outcome=TestOutcome.FAILED, error_message="x"
        )
        s = TestRunSummary.from_counts(
            total=2, passed=1, failed=1, failures=[fd]
        )
        assert len(s.failures) == 1
        assert s.failures[0].error_message == "x"


# ---------------------------------------------------------------------------
# Factory: from_test_result_events
# ---------------------------------------------------------------------------


class TestFromTestResultEvents:
    def test_from_events(self):
        now = time.time()
        events = [
            TestResultEvent(
                test_name="test_a",
                status=EventTestStatus.PASS,
                duration=0.1,
            ),
            TestResultEvent(
                test_name="test_b",
                status=EventTestStatus.FAIL,
                duration=0.5,
                message="assertion failed",
                stdout="output here",
                stderr="err here",
                file_path="tests/test_b.py",
                line_number=20,
            ),
            TestResultEvent(
                test_name="test_c",
                status=EventTestStatus.ERROR,
                duration=0.3,
                message="RuntimeError",
                error_details="RuntimeError",
            ),
            TestResultEvent(
                test_name="test_d",
                status=EventTestStatus.SKIP,
                duration=0.0,
            ),
        ]
        s = TestRunSummary.from_test_result_events(
            events,
            start_time=now - 2.0,
            end_time=now,
            run_id="run-1",
            framework="pytest",
        )
        assert s.total == 4
        assert s.passed == 1
        assert s.failed == 1
        assert s.errors == 1
        assert s.skipped == 1
        assert s.success is False
        assert s.duration_seconds == pytest.approx(2.0, abs=0.1)
        assert len(s.failures) == 2
        assert len(s.results) == 4

        # Check failure details have logs
        fail_detail = next(f for f in s.failures if f.test_name == "test_b")
        assert fail_detail.stdout == "output here"
        assert fail_detail.stderr == "err here"
        assert fail_detail.has_logs is True

    def test_empty_events(self):
        s = TestRunSummary.from_test_result_events([])
        assert s.total == 0
        assert s.success is False  # no tests = not success


# ---------------------------------------------------------------------------
# Factory: from_progress_snapshot
# ---------------------------------------------------------------------------


class TestFromProgressSnapshot:
    def test_from_snapshot(self):
        tracker = ProgressTracker(total=3)
        tracker.start()
        tracker.record_result(
            TestResult(
                test_id="t1",
                name="test_pass",
                status=TestStatus.PASSED,
                duration_seconds=0.1,
            )
        )
        tracker.record_result(
            TestResult(
                test_id="t2",
                name="test_fail",
                status=TestStatus.FAILED,
                duration_seconds=0.5,
                error_message="assert False",
                output="captured output",
            )
        )
        tracker.record_result(
            TestResult(
                test_id="t3",
                name="test_err",
                status=TestStatus.ERROR,
                duration_seconds=0.2,
                error_message="RuntimeError",
            )
        )
        tracker.finish()
        snapshot = tracker.snapshot()

        s = TestRunSummary.from_progress_snapshot(
            snapshot, run_id="run-2", framework="pytest"
        )
        assert s.total == 3
        assert s.passed == 1
        assert s.failed == 1
        assert s.errors == 1
        assert s.success is False
        assert s.duration_seconds > 0
        assert len(s.failures) == 2
        assert len(s.results) == 3

        # Check failure detail logs
        fail = next(f for f in s.failures if f.test_name == "test_fail")
        assert fail.error_message == "assert False"
        assert fail.stdout == "captured output"
        assert fail.has_logs is True

    def test_rejects_non_snapshot(self):
        with pytest.raises(TypeError, match="Expected ProgressSnapshot"):
            TestRunSummary.from_progress_snapshot({"not": "a snapshot"})
