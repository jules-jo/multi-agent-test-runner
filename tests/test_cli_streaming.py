"""Tests for CLIStreamingReporter — real-time coloured terminal output."""

from __future__ import annotations

import asyncio
from io import StringIO

import pytest
from rich.console import Console

from test_runner.reporting.cli_streaming import (
    CLIStreamingReporter,
    _format_duration,
    create_cli_reporter,
)
from test_runner.reporting.events import (
    EventType,
    RunEvent,
    TestResultEvent,
    TestStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reporter(*, verbose: bool = False, show_stdout: bool = False, show_stderr: bool = True) -> tuple[CLIStreamingReporter, StringIO]:
    """Create a reporter writing to a StringIO buffer (no ANSI codes)."""
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=120)
    reporter = CLIStreamingReporter(
        console=console,
        verbose=verbose,
        show_stdout=show_stdout,
        show_stderr=show_stderr,
    )
    return reporter, buf


def _pass_event(name: str = "test_add", duration: float = 0.012, **kw) -> TestResultEvent:
    return TestResultEvent(test_name=name, status=TestStatus.PASS, duration=duration, **kw)


def _fail_event(name: str = "test_div", duration: float = 0.045, **kw) -> TestResultEvent:
    return TestResultEvent(
        test_name=name,
        status=TestStatus.FAIL,
        duration=duration,
        error_details="ZeroDivisionError: division by zero",
        **kw,
    )


def _error_event(name: str = "test_crash", duration: float = 0.1, **kw) -> TestResultEvent:
    return TestResultEvent(
        test_name=name,
        status=TestStatus.ERROR,
        duration=duration,
        message="Unexpected error in fixture",
        **kw,
    )


def _skip_event(name: str = "test_windows_only", duration: float = 0.0, **kw) -> TestResultEvent:
    return TestResultEvent(test_name=name, status=TestStatus.SKIP, duration=duration, message="requires windows", **kw)


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_milliseconds(self):
        assert _format_duration(0.012) == "12ms"

    def test_zero(self):
        assert _format_duration(0.0) == "0ms"

    def test_seconds(self):
        assert _format_duration(1.234) == "1.23s"

    def test_minutes(self):
        result = _format_duration(125.7)
        assert result == "2m 6s"

    def test_boundary_under_one_second(self):
        assert _format_duration(0.999) == "999ms"

    def test_boundary_one_second(self):
        assert _format_duration(1.0) == "1.00s"

    def test_boundary_sixty_seconds(self):
        result = _format_duration(60.0)
        assert result == "1m 0s"


# ---------------------------------------------------------------------------
# Test result rendering
# ---------------------------------------------------------------------------


class TestRenderTestResult:
    @pytest.mark.asyncio
    async def test_pass_event_shows_green_pass(self):
        reporter, buf = _make_reporter()
        await reporter.on_event(_pass_event())
        output = buf.getvalue()
        assert "PASS" in output
        assert "test_add" in output

    @pytest.mark.asyncio
    async def test_fail_event_shows_red_fail(self):
        reporter, buf = _make_reporter()
        await reporter.on_event(_fail_event())
        output = buf.getvalue()
        assert "FAIL" in output
        assert "test_div" in output

    @pytest.mark.asyncio
    async def test_fail_event_shows_error_details(self):
        reporter, buf = _make_reporter()
        await reporter.on_event(_fail_event())
        output = buf.getvalue()
        assert "ZeroDivisionError" in output

    @pytest.mark.asyncio
    async def test_error_event_shows_err(self):
        reporter, buf = _make_reporter()
        await reporter.on_event(_error_event())
        output = buf.getvalue()
        assert "ERR" in output
        assert "test_crash" in output

    @pytest.mark.asyncio
    async def test_skip_event_shows_skip(self):
        reporter, buf = _make_reporter()
        await reporter.on_event(_skip_event())
        output = buf.getvalue()
        assert "SKIP" in output
        assert "test_windows_only" in output

    @pytest.mark.asyncio
    async def test_timing_displayed(self):
        reporter, buf = _make_reporter()
        await reporter.on_event(_pass_event(duration=0.012))
        output = buf.getvalue()
        assert "12ms" in output

    @pytest.mark.asyncio
    async def test_suite_prefix_in_name(self):
        reporter, buf = _make_reporter()
        await reporter.on_event(_pass_event(suite="test_math"))
        output = buf.getvalue()
        assert "test_math::test_add" in output

    @pytest.mark.asyncio
    async def test_slow_test_gets_highlighted(self):
        """Tests over the slow threshold should have highlighted duration."""
        reporter, buf = _make_reporter()
        await reporter.on_event(_pass_event(duration=6.5))
        output = buf.getvalue()
        # At minimum the duration string should be present
        assert "6.50s" in output

    @pytest.mark.asyncio
    async def test_counts_tracked(self):
        reporter, buf = _make_reporter()
        await reporter.on_event(_pass_event())
        await reporter.on_event(_fail_event())
        await reporter.on_event(_skip_event())
        assert reporter._total == 3
        assert reporter._counts[TestStatus.PASS] == 1
        assert reporter._counts[TestStatus.FAIL] == 1
        assert reporter._counts[TestStatus.SKIP] == 1


# ---------------------------------------------------------------------------
# Failure detail rendering
# ---------------------------------------------------------------------------


class TestFailureDetail:
    @pytest.mark.asyncio
    async def test_stderr_shown_by_default(self):
        reporter, buf = _make_reporter(show_stderr=True)
        event = TestResultEvent(
            test_name="t", status=TestStatus.FAIL, duration=0.01,
            stderr="traceback line 1\ntraceback line 2",
        )
        await reporter.on_event(event)
        output = buf.getvalue()
        assert "traceback line 1" in output

    @pytest.mark.asyncio
    async def test_stdout_hidden_by_default(self):
        reporter, buf = _make_reporter(show_stdout=False)
        event = TestResultEvent(
            test_name="t", status=TestStatus.FAIL, duration=0.01,
            stdout="some captured output",
            error_details="AssertionError",
        )
        await reporter.on_event(event)
        output = buf.getvalue()
        assert "some captured output" not in output

    @pytest.mark.asyncio
    async def test_stdout_shown_when_enabled(self):
        reporter, buf = _make_reporter(show_stdout=True)
        event = TestResultEvent(
            test_name="t", status=TestStatus.FAIL, duration=0.01,
            stdout="captured output here",
            error_details="AssertionError",
        )
        await reporter.on_event(event)
        output = buf.getvalue()
        assert "captured output here" in output

    @pytest.mark.asyncio
    async def test_message_fallback_when_no_error_details(self):
        reporter, buf = _make_reporter()
        event = TestResultEvent(
            test_name="t", status=TestStatus.FAIL, duration=0.01,
            message="assertion failed: expected 3 got 4",
        )
        await reporter.on_event(event)
        output = buf.getvalue()
        assert "assertion failed" in output


# ---------------------------------------------------------------------------
# Run lifecycle events
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    @pytest.mark.asyncio
    async def test_on_run_start_prints_header(self):
        reporter, buf = _make_reporter()
        await reporter.on_run_start()
        output = buf.getvalue()
        assert "Test Run Started" in output
        assert "─" in output

    @pytest.mark.asyncio
    async def test_on_run_end_prints_summary(self):
        reporter, buf = _make_reporter()
        await reporter.on_run_start()
        await reporter.on_event(_pass_event())
        await reporter.on_event(_pass_event(name="test_sub"))
        summary = {"total": 2, "passed": 2, "failed": 0, "errors": 0, "skipped": 0, "duration": 0.05, "all_passed": True}
        await reporter.on_run_end(summary)
        output = buf.getvalue()
        assert "All tests passed" in output
        assert "2 passed" in output

    @pytest.mark.asyncio
    async def test_on_run_end_with_failures(self):
        reporter, buf = _make_reporter()
        summary = {"total": 3, "passed": 1, "failed": 2, "errors": 0, "skipped": 0, "duration": 1.5, "all_passed": False}
        await reporter.on_run_end(summary)
        output = buf.getvalue()
        assert "2 test(s) failed" in output
        assert "1 passed" in output
        assert "2 failed" in output

    @pytest.mark.asyncio
    async def test_on_run_end_with_errors_only(self):
        reporter, buf = _make_reporter()
        summary = {"total": 1, "passed": 0, "failed": 0, "errors": 1, "skipped": 0, "duration": 0.2, "all_passed": False}
        await reporter.on_run_end(summary)
        output = buf.getvalue()
        assert "1 test(s) errored" in output
        assert "1 errors" in output

    @pytest.mark.asyncio
    async def test_on_run_end_no_tests(self):
        reporter, buf = _make_reporter()
        summary = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "duration": 0.0, "all_passed": False}
        await reporter.on_run_end(summary)
        output = buf.getvalue()
        assert "No tests were executed" in output

    @pytest.mark.asyncio
    async def test_duration_in_summary(self):
        reporter, buf = _make_reporter()
        summary = {"total": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0, "duration": 2.345, "all_passed": True}
        await reporter.on_run_end(summary)
        output = buf.getvalue()
        assert "2.35s" in output

    @pytest.mark.asyncio
    async def test_ai_analysis_in_summary(self):
        reporter, buf = _make_reporter()
        summary = {
            "total": 1, "passed": 0, "failed": 1, "errors": 0,
            "skipped": 0, "duration": 1.0, "all_passed": False,
            "ai_analysis": "The failure is a flaky network test.",
        }
        await reporter.on_run_end(summary)
        output = buf.getvalue()
        assert "AI Analysis" in output
        assert "flaky network test" in output


# ---------------------------------------------------------------------------
# RunEvent rendering
# ---------------------------------------------------------------------------


class TestRunEventRendering:
    @pytest.mark.asyncio
    async def test_discovery_event_displayed(self):
        reporter, buf = _make_reporter()
        event = RunEvent(event_type=EventType.DISCOVERY, message="Found 42 tests")
        await reporter.on_event(event)
        output = buf.getvalue()
        assert "Found 42 tests" in output

    @pytest.mark.asyncio
    async def test_log_event_hidden_when_not_verbose(self):
        reporter, buf = _make_reporter(verbose=False)
        event = RunEvent(event_type=EventType.LOG, message="debug info")
        await reporter.on_event(event)
        output = buf.getvalue()
        assert "debug info" not in output

    @pytest.mark.asyncio
    async def test_log_event_shown_when_verbose(self):
        reporter, buf = _make_reporter(verbose=True)
        event = RunEvent(event_type=EventType.LOG, message="debug info")
        await reporter.on_event(event)
        output = buf.getvalue()
        assert "debug info" in output

    @pytest.mark.asyncio
    async def test_troubleshoot_event_displayed(self):
        reporter, buf = _make_reporter()
        event = RunEvent(event_type=EventType.TROUBLESHOOT, message="Possible fix: install deps")
        await reporter.on_event(event)
        output = buf.getvalue()
        assert "Possible fix" in output


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFailureDetailSummaryRendering:
    """Tests for failure details with logs in the summary footer."""

    @pytest.mark.asyncio
    async def test_failure_details_rendered_in_summary(self):
        reporter, buf = _make_reporter()
        summary = {
            "total": 2, "passed": 1, "failed": 1, "errors": 0,
            "skipped": 0, "duration": 1.0, "all_passed": False,
            "failure_details": [
                {
                    "test_name": "test_divide",
                    "status": "fail",
                    "duration": 0.5,
                    "message": "assert 1 / 0",
                    "error_details": "ZeroDivisionError",
                    "stdout": "captured stdout line",
                    "stderr": "captured stderr line",
                    "file_path": "tests/test_math.py",
                    "line_number": 42,
                    "suite": "test_math",
                    "has_logs": True,
                },
            ],
        }
        await reporter.on_run_end(summary)
        output = buf.getvalue()

        assert "Failure Details:" in output
        assert "test_divide" in output
        assert "FAIL" in output
        assert "tests/test_math.py:42" in output
        assert "assert 1 / 0" in output
        assert "ZeroDivisionError" in output
        assert "[stdout]" in output
        assert "captured stdout line" in output
        assert "[stderr]" in output
        assert "captured stderr line" in output

    @pytest.mark.asyncio
    async def test_failure_details_no_logs(self):
        reporter, buf = _make_reporter()
        summary = {
            "total": 1, "passed": 0, "failed": 1, "errors": 0,
            "skipped": 0, "duration": 0.5, "all_passed": False,
            "failure_details": [
                {
                    "test_name": "test_simple_fail",
                    "status": "fail",
                    "duration": 0.1,
                    "message": "bad assertion",
                    "error_details": "",
                    "stdout": "",
                    "stderr": "",
                    "file_path": "",
                    "line_number": None,
                    "suite": "",
                    "has_logs": False,
                },
            ],
        }
        await reporter.on_run_end(summary)
        output = buf.getvalue()

        assert "Failure Details:" in output
        assert "test_simple_fail" in output
        assert "bad assertion" in output
        # No log sections should appear
        assert "[stdout]" not in output
        assert "[stderr]" not in output

    @pytest.mark.asyncio
    async def test_no_failure_details_section_when_all_pass(self):
        reporter, buf = _make_reporter()
        summary = {
            "total": 2, "passed": 2, "failed": 0, "errors": 0,
            "skipped": 0, "duration": 0.5, "all_passed": True,
            "failure_details": [],
        }
        await reporter.on_run_end(summary)
        output = buf.getvalue()
        assert "Failure Details:" not in output

    @pytest.mark.asyncio
    async def test_multiple_failure_details_rendered(self):
        reporter, buf = _make_reporter()
        summary = {
            "total": 3, "passed": 1, "failed": 1, "errors": 1,
            "skipped": 0, "duration": 2.0, "all_passed": False,
            "failure_details": [
                {
                    "test_name": "test_alpha",
                    "status": "fail",
                    "duration": 0.3,
                    "message": "expected true",
                    "error_details": "",
                    "stdout": "alpha stdout",
                    "stderr": "",
                    "file_path": "tests/test_a.py",
                    "line_number": 10,
                    "suite": "",
                    "has_logs": True,
                },
                {
                    "test_name": "test_beta",
                    "status": "error",
                    "duration": 0.7,
                    "message": "RuntimeError: crash",
                    "error_details": "",
                    "stdout": "",
                    "stderr": "beta stderr",
                    "file_path": "tests/test_b.py",
                    "line_number": 55,
                    "suite": "",
                    "has_logs": True,
                },
            ],
        }
        await reporter.on_run_end(summary)
        output = buf.getvalue()

        assert "1)" in output
        assert "test_alpha" in output
        assert "alpha stdout" in output
        assert "2)" in output
        assert "test_beta" in output
        assert "beta stderr" in output

    @pytest.mark.asyncio
    async def test_summary_without_failure_details_key(self):
        """Backward-compatible: summary dicts without failure_details should work."""
        reporter, buf = _make_reporter()
        summary = {
            "total": 1, "passed": 0, "failed": 1, "errors": 0,
            "skipped": 0, "duration": 1.0, "all_passed": False,
        }
        await reporter.on_run_end(summary)
        output = buf.getvalue()
        assert "1 test(s) failed" in output
        # No crash, no Failure Details section
        assert "Failure Details:" not in output


class TestFactory:
    def test_create_cli_reporter_returns_instance(self):
        r = create_cli_reporter()
        assert isinstance(r, CLIStreamingReporter)

    def test_create_cli_reporter_with_file(self):
        buf = StringIO()
        r = create_cli_reporter(file=buf)
        assert isinstance(r, CLIStreamingReporter)

    def test_create_cli_reporter_verbose(self):
        r = create_cli_reporter(verbose=True)
        assert r._verbose is True

    def test_reporter_is_cli_reporter_base(self):
        from test_runner.reporting.base import CLIReporterBase
        r = create_cli_reporter()
        assert isinstance(r, CLIReporterBase)


# ---------------------------------------------------------------------------
# Integration: full run sequence
# ---------------------------------------------------------------------------


class TestFullRunSequence:
    @pytest.mark.asyncio
    async def test_complete_run_flow(self):
        """Simulate a complete test run with mixed results."""
        reporter, buf = _make_reporter(verbose=True)

        # Start
        await reporter.on_run_start()

        # Discovery event
        await reporter.on_event(RunEvent(event_type=EventType.DISCOVERY, message="Found 4 tests"))

        # Test results
        await reporter.on_event(_pass_event("test_add", 0.005))
        await reporter.on_event(_pass_event("test_sub", 0.003))
        await reporter.on_event(_fail_event("test_div_zero", 0.045))
        await reporter.on_event(_skip_event("test_platform", 0.0))

        # End
        summary = {
            "total": 4, "passed": 2, "failed": 1, "errors": 0,
            "skipped": 1, "duration": 0.053, "all_passed": False,
        }
        await reporter.on_run_end(summary)

        output = buf.getvalue()

        # Verify key elements present
        assert "Test Run Started" in output
        assert "Found 4 tests" in output
        assert "test_add" in output
        assert "PASS" in output
        assert "FAIL" in output
        assert "SKIP" in output
        assert "test_div_zero" in output
        assert "ZeroDivisionError" in output
        assert "1 test(s) failed" in output
        assert "2 passed" in output
        assert "1 failed" in output
        assert "1 skipped" in output
        assert reporter._total == 4
