"""Tests for the Reporter agent and output parsers."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_runner.agents.base import AgentRole
from test_runner.agents.reporter.agent import (
    ReporterAgent,
    RunStatistics,
    _EXEC_TO_TEST_STATUS,
)
from test_runner.agents.reporter.output_parser import (
    GenericOutputParser,
    JestOutputParser,
    OutputParserRegistry,
    PytestOutputParser,
)
from test_runner.execution.command_translator import TestCommand
from test_runner.execution.targets import ExecutionResult, ExecutionStatus
from test_runner.reporting.base import ReporterBase, StreamEvent
from test_runner.reporting.events import (
    EventType,
    RunEvent,
    TestResultEvent,
    TestStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeChannel(ReporterBase):
    """In-memory channel that records all events for assertions."""

    def __init__(self) -> None:
        self.events: list[StreamEvent] = []
        self.run_started = False
        self.run_ended = False
        self.end_summary: dict = {}

    async def on_event(self, event: StreamEvent) -> None:
        self.events.append(event)

    async def on_run_start(self) -> None:
        self.run_started = True

    async def on_run_end(self, summary: dict) -> None:
        self.run_ended = True
        self.end_summary = summary


@pytest.fixture
def channel() -> FakeChannel:
    return FakeChannel()


@pytest.fixture
def reporter(channel: FakeChannel) -> ReporterAgent:
    agent = ReporterAgent()
    agent.add_channel(channel)
    return agent


# ---------------------------------------------------------------------------
# PytestOutputParser tests
# ---------------------------------------------------------------------------


class TestPytestOutputParser:
    def test_parses_passed(self):
        parser = PytestOutputParser()
        events = list(parser.feed_line(
            "tests/test_foo.py::test_bar PASSED"
        ))
        assert len(events) == 1
        assert events[0].test_name == "test_bar"
        assert events[0].status == TestStatus.PASS
        assert events[0].file_path == "tests/test_foo.py"

    def test_parses_failed(self):
        parser = PytestOutputParser()
        events = list(parser.feed_line(
            "tests/test_foo.py::test_baz FAILED"
        ))
        assert len(events) == 1
        assert events[0].status == TestStatus.FAIL

    def test_parses_error(self):
        parser = PytestOutputParser()
        events = list(parser.feed_line(
            "tests/test_foo.py::test_err ERROR"
        ))
        assert len(events) == 1
        assert events[0].status == TestStatus.ERROR

    def test_parses_skipped(self):
        parser = PytestOutputParser()
        events = list(parser.feed_line(
            "tests/test_foo.py::test_skip SKIPPED"
        ))
        assert len(events) == 1
        assert events[0].status == TestStatus.SKIP

    def test_parses_with_duration(self):
        parser = PytestOutputParser()
        events = list(parser.feed_line(
            "tests/test_foo.py::test_slow PASSED (1.23s)"
        ))
        assert len(events) == 1
        assert events[0].duration == 1.23

    def test_parses_with_progress_indicator(self):
        parser = PytestOutputParser()
        events = list(parser.feed_line(
            "tests/test_foo.py::test_bar PASSED [50%]"
        ))
        assert len(events) == 1
        assert events[0].status == TestStatus.PASS

    def test_ignores_non_matching_lines(self):
        parser = PytestOutputParser()
        events = list(parser.feed_line("collecting ... collected 5 items"))
        assert len(events) == 0

    def test_suite_set_from_file_path(self):
        parser = PytestOutputParser()
        events = list(parser.feed_line(
            "tests/test_foo.py::test_bar PASSED"
        ))
        assert events[0].suite == "tests/test_foo.py"


# ---------------------------------------------------------------------------
# GenericOutputParser tests
# ---------------------------------------------------------------------------


class TestGenericOutputParser:
    def test_parses_tap_ok(self):
        parser = GenericOutputParser()
        events = list(parser.feed_line("ok 1 - first test"))
        assert len(events) == 1
        assert events[0].test_name == "first test"
        assert events[0].status == TestStatus.PASS

    def test_parses_tap_not_ok(self):
        parser = GenericOutputParser()
        events = list(parser.feed_line("not ok 2 - second test"))
        assert len(events) == 1
        assert events[0].status == TestStatus.FAIL

    def test_parses_tap_no_description(self):
        parser = GenericOutputParser()
        events = list(parser.feed_line("ok 3"))
        assert len(events) == 1
        assert events[0].test_name == "test-3"

    def test_ignores_empty_lines(self):
        parser = GenericOutputParser()
        events = list(parser.feed_line(""))
        assert len(events) == 0

    def test_flush_returns_nothing(self):
        parser = GenericOutputParser()
        events = list(parser.flush())
        assert len(events) == 0


# ---------------------------------------------------------------------------
# JestOutputParser tests
# ---------------------------------------------------------------------------


class TestJestOutputParser:
    def test_parses_pass_with_duration(self):
        parser = JestOutputParser()
        events = list(parser.feed_line("  ✓ should work (5 ms)"))
        assert len(events) == 1
        assert events[0].test_name == "should work"
        assert events[0].status == TestStatus.PASS
        assert events[0].duration == pytest.approx(0.005)

    def test_parses_fail(self):
        parser = JestOutputParser()
        events = list(parser.feed_line("  ✕ should fail (12 ms)"))
        assert len(events) == 1
        assert events[0].status == TestStatus.FAIL

    def test_parses_skip(self):
        parser = JestOutputParser()
        events = list(parser.feed_line("  ○ skipped test name"))
        assert len(events) == 1
        assert events[0].status == TestStatus.SKIP


# ---------------------------------------------------------------------------
# OutputParserRegistry tests
# ---------------------------------------------------------------------------


class TestOutputParserRegistry:
    def test_default_parsers_registered(self):
        reg = OutputParserRegistry()
        assert "pytest" in reg.supported_frameworks
        assert "jest" in reg.supported_frameworks
        assert "generic" in reg.supported_frameworks

    def test_get_known_framework(self):
        reg = OutputParserRegistry()
        parser = reg.get("pytest")
        assert isinstance(parser, PytestOutputParser)

    def test_get_unknown_falls_back_to_generic(self):
        reg = OutputParserRegistry()
        parser = reg.get("unknown_framework")
        assert isinstance(parser, GenericOutputParser)

    def test_get_returns_fresh_instance(self):
        reg = OutputParserRegistry()
        p1 = reg.get("pytest")
        p2 = reg.get("pytest")
        assert p1 is not p2


# ---------------------------------------------------------------------------
# RunStatistics tests
# ---------------------------------------------------------------------------


class TestRunStatistics:
    def test_record_pass(self):
        stats = RunStatistics()
        event = TestResultEvent(
            test_name="t1", status=TestStatus.PASS, duration=0.1
        )
        stats.record(event)
        assert stats.total == 1
        assert stats.passed == 1

    def test_record_fail(self):
        stats = RunStatistics()
        event = TestResultEvent(
            test_name="t1", status=TestStatus.FAIL, duration=0.1
        )
        stats.record(event)
        assert stats.failed == 1

    def test_record_error(self):
        stats = RunStatistics()
        event = TestResultEvent(
            test_name="t1", status=TestStatus.ERROR, duration=0.1
        )
        stats.record(event)
        assert stats.errors == 1

    def test_record_skip(self):
        stats = RunStatistics()
        event = TestResultEvent(
            test_name="t1", status=TestStatus.SKIP, duration=0.0
        )
        stats.record(event)
        assert stats.skipped == 1

    def test_all_passed(self):
        stats = RunStatistics()
        stats.record(TestResultEvent(
            test_name="t1", status=TestStatus.PASS, duration=0.1
        ))
        assert stats.all_passed is True

    def test_all_passed_false_when_failures(self):
        stats = RunStatistics()
        stats.record(TestResultEvent(
            test_name="t1", status=TestStatus.PASS, duration=0.1
        ))
        stats.record(TestResultEvent(
            test_name="t2", status=TestStatus.FAIL, duration=0.1
        ))
        assert stats.all_passed is False

    def test_all_passed_false_when_empty(self):
        stats = RunStatistics()
        assert stats.all_passed is False

    def test_to_summary(self):
        stats = RunStatistics()
        stats.record(TestResultEvent(
            test_name="t1", status=TestStatus.PASS, duration=0.1
        ))
        stats.finalize()
        summary = stats.to_summary()
        assert summary["total"] == 1
        assert summary["passed"] == 1
        assert "duration" in summary


# ---------------------------------------------------------------------------
# ReporterAgent tests
# ---------------------------------------------------------------------------


class TestReporterAgent:
    def test_role_is_reporter(self):
        agent = ReporterAgent()
        assert agent.role == AgentRole.REPORTER

    def test_name(self):
        agent = ReporterAgent()
        assert agent.name == "reporter-agent"

    def test_no_tools(self):
        agent = ReporterAgent()
        assert agent.get_tools() == []

    def test_add_remove_channel(self, channel: FakeChannel):
        agent = ReporterAgent()
        agent.add_channel(channel)
        assert len(agent.channels) == 1
        agent.remove_channel(channel)
        assert len(agent.channels) == 0


class TestReporterAgentProcessOutput:
    """Test real-time parsing and emission of test results."""

    @pytest.mark.asyncio
    async def test_process_pytest_output_emits_events(
        self, reporter: ReporterAgent, channel: FakeChannel
    ):
        """Events are emitted one-by-one as lines are processed."""
        lines = [
            "tests/test_a.py::test_one PASSED",
            "tests/test_a.py::test_two FAILED",
            "tests/test_b.py::test_three SKIPPED",
        ]
        events = await reporter.process_output(lines, "pytest")

        assert len(events) == 3
        assert events[0].status == TestStatus.PASS
        assert events[1].status == TestStatus.FAIL
        assert events[2].status == TestStatus.SKIP

        # All events were sent to channel
        test_events = [
            e for e in channel.events if isinstance(e, TestResultEvent)
        ]
        assert len(test_events) == 3

    @pytest.mark.asyncio
    async def test_process_output_string_input(
        self, reporter: ReporterAgent
    ):
        """Accepts a single string with newlines."""
        output = (
            "tests/test_a.py::test_one PASSED\n"
            "tests/test_a.py::test_two FAILED\n"
        )
        events = await reporter.process_output(output, "pytest")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_process_output_updates_stats(
        self, reporter: ReporterAgent
    ):
        lines = [
            "tests/test_a.py::test_one PASSED",
            "tests/test_a.py::test_two FAILED",
        ]
        await reporter.process_output(lines, "pytest")
        assert reporter.stats.total == 2
        assert reporter.stats.passed == 1
        assert reporter.stats.failed == 1

    @pytest.mark.asyncio
    async def test_process_output_records_steps(
        self, reporter: ReporterAgent
    ):
        lines = ["tests/test_a.py::test_one PASSED"]
        await reporter.process_output(lines, "pytest")
        assert reporter.state.steps_taken == 1

    @pytest.mark.asyncio
    async def test_process_output_generic_tap(
        self, reporter: ReporterAgent, channel: FakeChannel
    ):
        lines = ["ok 1 - first", "not ok 2 - second"]
        events = await reporter.process_output(lines, "generic")
        assert len(events) == 2
        assert events[0].status == TestStatus.PASS
        assert events[1].status == TestStatus.FAIL

    @pytest.mark.asyncio
    async def test_events_emitted_individually_not_batched(
        self, reporter: ReporterAgent
    ):
        """Verify events are emitted one at a time, not batched."""
        emission_order: list[str] = []

        async def track_event(event: TestResultEvent) -> None:
            emission_order.append(event.test_name)

        reporter.on_event_async(track_event)

        lines = [
            "tests/test_a.py::test_one PASSED",
            "tests/test_a.py::test_two FAILED",
            "tests/test_a.py::test_three PASSED",
        ]
        await reporter.process_output(lines, "pytest")

        assert emission_order == ["test_one", "test_two", "test_three"]

    @pytest.mark.asyncio
    async def test_sync_callback_fired(self, reporter: ReporterAgent):
        """Synchronous on_event callbacks are invoked."""
        received: list[TestResultEvent] = []
        reporter.on_event(lambda e: received.append(e))

        await reporter.process_output(
            ["tests/test_a.py::test_one PASSED"], "pytest"
        )
        assert len(received) == 1
        assert received[0].test_name == "test_one"


class TestReporterAgentRunLifecycle:
    @pytest.mark.asyncio
    async def test_start_run(
        self, reporter: ReporterAgent, channel: FakeChannel
    ):
        await reporter.start_run()
        assert channel.run_started is True
        run_events = [
            e for e in channel.events
            if isinstance(e, RunEvent)
            and e.event_type == EventType.RUN_STARTED
        ]
        assert len(run_events) == 1

    @pytest.mark.asyncio
    async def test_end_run(
        self, reporter: ReporterAgent, channel: FakeChannel
    ):
        await reporter.start_run()
        await reporter.process_output(
            ["tests/test_a.py::test_one PASSED"], "pytest"
        )
        summary = await reporter.end_run()

        assert channel.run_ended is True
        assert summary["total"] == 1
        assert summary["passed"] == 1
        assert channel.end_summary == summary

    @pytest.mark.asyncio
    async def test_get_run_summary(self, reporter: ReporterAgent):
        await reporter.process_output(
            ["tests/test_a.py::test_one PASSED"], "pytest"
        )
        summary = reporter.get_run_summary()
        assert summary["total"] == 1

    @pytest.mark.asyncio
    async def test_reset_state_clears_stats(self, reporter: ReporterAgent):
        await reporter.process_output(
            ["tests/test_a.py::test_one PASSED"], "pytest"
        )
        assert reporter.stats.total == 1
        reporter.reset_state()
        assert reporter.stats.total == 0

    @pytest.mark.asyncio
    async def test_handoff_summary_includes_stats(
        self, reporter: ReporterAgent
    ):
        await reporter.process_output(
            ["tests/test_a.py::test_one PASSED"], "pytest"
        )
        handoff = reporter.get_handoff_summary()
        assert handoff["role"] == "reporter"
        assert handoff["run_stats"]["total"] == 1


class TestRunStatisticsFailureDetails:
    """Tests for RunStatistics.collect_failure_details and to_full_summary."""

    def test_collect_failure_details_empty(self):
        stats = RunStatistics()
        assert stats.collect_failure_details() == []

    def test_collect_failure_details_no_failures(self):
        stats = RunStatistics()
        stats.record(TestResultEvent(
            test_name="t1", status=TestStatus.PASS, duration=0.1
        ))
        stats.record(TestResultEvent(
            test_name="t2", status=TestStatus.SKIP, duration=0.0
        ))
        assert stats.collect_failure_details() == []

    def test_collect_failure_details_with_logs(self):
        stats = RunStatistics()
        stats.record(TestResultEvent(
            test_name="test_pass", status=TestStatus.PASS, duration=0.1
        ))
        stats.record(TestResultEvent(
            test_name="test_fail",
            status=TestStatus.FAIL,
            duration=0.5,
            message="assert 1 == 2",
            stdout="captured stdout",
            stderr="captured stderr",
            error_details="AssertionError",
            file_path="tests/test_a.py",
            line_number=42,
            suite="tests/test_a.py",
        ))
        stats.record(TestResultEvent(
            test_name="test_error",
            status=TestStatus.ERROR,
            duration=0.3,
            message="RuntimeError: boom",
            stderr="traceback here",
        ))

        details = stats.collect_failure_details()
        assert len(details) == 2

        # First failure
        d0 = details[0]
        assert d0["test_name"] == "test_fail"
        assert d0["status"] == "fail"
        assert d0["message"] == "assert 1 == 2"
        assert d0["stdout"] == "captured stdout"
        assert d0["stderr"] == "captured stderr"
        assert d0["error_details"] == "AssertionError"
        assert d0["file_path"] == "tests/test_a.py"
        assert d0["line_number"] == 42
        assert d0["has_logs"] is True

        # Second failure (error)
        d1 = details[1]
        assert d1["test_name"] == "test_error"
        assert d1["status"] == "error"
        assert d1["stderr"] == "traceback here"
        assert d1["has_logs"] is True

    def test_collect_failure_details_no_logs(self):
        stats = RunStatistics()
        stats.record(TestResultEvent(
            test_name="test_fail",
            status=TestStatus.FAIL,
            duration=0.1,
            message="failed",
        ))
        details = stats.collect_failure_details()
        assert len(details) == 1
        assert details[0]["has_logs"] is False
        assert details[0]["stdout"] == ""
        assert details[0]["stderr"] == ""

    def test_to_full_summary_includes_failure_details(self):
        stats = RunStatistics()
        stats.record(TestResultEvent(
            test_name="test_pass", status=TestStatus.PASS, duration=0.1
        ))
        stats.record(TestResultEvent(
            test_name="test_fail",
            status=TestStatus.FAIL,
            duration=0.5,
            message="assert False",
            stdout="output here",
        ))
        stats.finalize()
        summary = stats.to_full_summary()

        # Has all the base keys
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert "duration" in summary

        # Has failure_details
        assert "failure_details" in summary
        assert len(summary["failure_details"]) == 1
        assert summary["failure_details"][0]["test_name"] == "test_fail"
        assert summary["failure_details"][0]["stdout"] == "output here"
        assert summary["failure_details"][0]["has_logs"] is True


class TestReporterAgentEndRunWithFailureDetails:
    """Tests that end_run() returns failure details with logs."""

    @pytest.mark.asyncio
    async def test_end_run_includes_failure_details(
        self, reporter: ReporterAgent, channel: FakeChannel
    ):
        await reporter.start_run()
        # Process a mix of passing and failing tests
        lines = [
            "tests/test_a.py::test_pass PASSED",
            "tests/test_a.py::test_fail FAILED",
        ]
        await reporter.process_output(lines, "pytest")

        summary = await reporter.end_run()

        assert "failure_details" in summary
        assert isinstance(summary["failure_details"], list)
        # The pytest parser doesn't capture stdout/stderr from output lines,
        # but the structure should be present
        assert len(summary["failure_details"]) == 1
        assert summary["failure_details"][0]["test_name"] == "test_fail"
        assert summary["failure_details"][0]["status"] == "fail"

    @pytest.mark.asyncio
    async def test_end_run_no_failures_empty_details(
        self, reporter: ReporterAgent, channel: FakeChannel
    ):
        await reporter.start_run()
        await reporter.process_output(
            ["tests/test_a.py::test_pass PASSED"], "pytest"
        )
        summary = await reporter.end_run()
        assert summary["failure_details"] == []

    @pytest.mark.asyncio
    async def test_end_run_summary_passed_to_channel(
        self, reporter: ReporterAgent, channel: FakeChannel
    ):
        """Channel receives the summary dict with failure_details."""
        await reporter.start_run()
        await reporter.process_output(
            ["tests/test_a.py::test_fail FAILED"], "pytest"
        )
        await reporter.end_run()

        assert "failure_details" in channel.end_summary
        assert len(channel.end_summary["failure_details"]) == 1


class TestReporterAgentChannelErrors:
    @pytest.mark.asyncio
    async def test_channel_error_does_not_crash(
        self, reporter: ReporterAgent
    ):
        """A failing channel should not prevent other events."""

        class FailingChannel(ReporterBase):
            async def on_event(self, event: StreamEvent) -> None:
                raise RuntimeError("channel broke")

        reporter.add_channel(FailingChannel())
        # Should not raise
        events = await reporter.process_output(
            ["tests/test_a.py::test_one PASSED"], "pytest"
        )
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash(
        self, reporter: ReporterAgent
    ):
        def bad_callback(event: TestResultEvent) -> None:
            raise ValueError("callback broke")

        reporter.on_event(bad_callback)
        events = await reporter.process_output(
            ["tests/test_a.py::test_one PASSED"], "pytest"
        )
        assert len(events) == 1
