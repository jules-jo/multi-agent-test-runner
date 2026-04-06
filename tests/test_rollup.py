"""Tests for periodic rollup summary generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from test_runner.agents.reporter.rollup import (
    ProgressSource,
    ProgressTrackerAdapter,
    RollupConfig,
    RollupSummaryGenerator,
    RunStatisticsAdapter,
    format_rollup_message,
)
from test_runner.agents.reporter.agent import ReporterAgent, RunStatistics
from test_runner.reporting.base import ReporterBase, StreamEvent
from test_runner.reporting.events import (
    EventType,
    RunEvent,
    TestResultEvent,
    TestStatus,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeProgressSource:
    """Minimal ProgressSource for testing."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0


class FakeChannel(ReporterBase):
    """In-memory channel that records all events."""

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


# ---------------------------------------------------------------------------
# format_rollup_message tests
# ---------------------------------------------------------------------------


class TestFormatRollupMessage:
    def test_no_tests_done(self):
        msg = format_rollup_message(
            total=20, passed=0, failed=0, errors=0, skipped=0, elapsed=1.0
        )
        assert "0/20 done" in msg
        assert "1.0s elapsed" in msg

    def test_some_done_all_passing(self):
        msg = format_rollup_message(
            total=20, passed=5, failed=0, errors=0, skipped=0, elapsed=12.3
        )
        assert "5/20 done" in msg
        assert "all passing" in msg
        assert "12.3s elapsed" in msg

    def test_all_done_all_passing(self):
        msg = format_rollup_message(
            total=10, passed=10, failed=0, errors=0, skipped=0, elapsed=5.1
        )
        assert "10/10 done" in msg
        assert "all passing" in msg

    def test_single_failure(self):
        msg = format_rollup_message(
            total=20, passed=4, failed=1, errors=0, skipped=0, elapsed=8.0
        )
        assert "5/20 done" in msg
        assert "1 failure" in msg
        # singular form
        assert "failures" not in msg

    def test_multiple_failures(self):
        msg = format_rollup_message(
            total=20, passed=3, failed=2, errors=0, skipped=0, elapsed=8.0
        )
        assert "5/20 done" in msg
        assert "2 failures" in msg

    def test_failure_and_error(self):
        msg = format_rollup_message(
            total=10, passed=1, failed=1, errors=1, skipped=0, elapsed=8.2
        )
        assert "3/10 done" in msg
        assert "1 failure" in msg
        assert "1 error" in msg

    def test_multiple_errors(self):
        msg = format_rollup_message(
            total=10, passed=0, failed=0, errors=3, skipped=0, elapsed=2.0
        )
        assert "3 errors" in msg

    def test_with_skipped(self):
        msg = format_rollup_message(
            total=10, passed=5, failed=0, errors=0, skipped=2, elapsed=3.0
        )
        assert "7/10 done" in msg
        assert "2 skipped" in msg

    def test_unknown_total(self):
        msg = format_rollup_message(
            total=0, passed=5, failed=0, errors=0, skipped=0, elapsed=3.0
        )
        assert "5 done" in msg
        # Should not have "5/0"
        assert "/0" not in msg

    def test_millisecond_elapsed(self):
        msg = format_rollup_message(
            total=10, passed=1, failed=0, errors=0, skipped=0, elapsed=0.5
        )
        assert "500ms elapsed" in msg

    def test_minute_elapsed(self):
        msg = format_rollup_message(
            total=100, passed=50, failed=0, errors=0, skipped=0, elapsed=90.0
        )
        assert "1m 30s elapsed" in msg


# ---------------------------------------------------------------------------
# RollupConfig tests
# ---------------------------------------------------------------------------


class TestRollupConfig:
    def test_default_interval(self):
        config = RollupConfig()
        assert config.interval_seconds == 10.0
        assert config.enabled is True

    def test_custom_interval(self):
        config = RollupConfig(interval_seconds=5.0)
        assert config.interval_seconds == 5.0

    def test_interval_clamped_to_min(self):
        config = RollupConfig(interval_seconds=0.1)
        assert config.interval_seconds == 1.0

    def test_interval_clamped_to_max(self):
        config = RollupConfig(interval_seconds=500.0)
        assert config.interval_seconds == 300.0

    def test_disabled(self):
        config = RollupConfig(enabled=False)
        assert config.enabled is False


# ---------------------------------------------------------------------------
# RunStatisticsAdapter tests
# ---------------------------------------------------------------------------


class TestRunStatisticsAdapter:
    def test_adapts_run_statistics(self):
        stats = RunStatistics()
        stats.record(TestResultEvent(
            test_name="t1", status=TestStatus.PASS, duration=0.1
        ))
        stats.record(TestResultEvent(
            test_name="t2", status=TestStatus.FAIL, duration=0.2
        ))

        adapter = RunStatisticsAdapter(stats)
        assert adapter.total == 2
        assert adapter.passed == 1
        assert adapter.failed == 1
        assert adapter.errors == 0
        assert adapter.skipped == 0
        assert adapter.duration > 0


# ---------------------------------------------------------------------------
# RollupSummaryGenerator tests
# ---------------------------------------------------------------------------


class TestRollupSummaryGenerator:
    def test_generate_now(self):
        source = FakeProgressSource(
            total=20, passed=5, failed=1, errors=0, skipped=0, duration=12.3
        )
        callback = AsyncMock()

        gen = RollupSummaryGenerator(
            source=source,
            on_rollup=callback,
        )

        event = gen.generate_now()
        assert isinstance(event, RunEvent)
        assert event.event_type == EventType.ROLLUP_SUMMARY
        assert "6/20 done" in event.message  # 5 passed + 1 failed = 6 completed
        assert "1 failure" in event.message
        assert event.data["total"] == 20
        assert event.data["passed"] == 5
        assert event.data["failed"] == 1
        assert event.data["rollup_number"] == 1
        assert gen.rollup_count == 1

    def test_generate_now_increments_count(self):
        source = FakeProgressSource(total=10, passed=5, duration=1.0)
        callback = AsyncMock()
        gen = RollupSummaryGenerator(source=source, on_rollup=callback)

        gen.generate_now()
        gen.generate_now()
        gen.generate_now()
        assert gen.rollup_count == 3

    @pytest.mark.asyncio
    async def test_start_stop(self):
        source = FakeProgressSource(total=10, passed=3, duration=1.0)
        callback = AsyncMock()
        config = RollupConfig(interval_seconds=1.0)

        gen = RollupSummaryGenerator(
            source=source, on_rollup=callback, config=config
        )

        assert not gen.is_running
        await gen.start()
        assert gen.is_running
        await gen.stop()
        assert not gen.is_running

    @pytest.mark.asyncio
    async def test_start_when_disabled(self):
        source = FakeProgressSource(total=10)
        callback = AsyncMock()
        config = RollupConfig(enabled=False)

        gen = RollupSummaryGenerator(
            source=source, on_rollup=callback, config=config
        )

        await gen.start()
        assert not gen.is_running

    @pytest.mark.asyncio
    async def test_periodic_emission(self):
        """Generator emits rollups at the configured interval."""
        source = FakeProgressSource(
            total=10, passed=3, failed=1, duration=5.0
        )
        emitted_events: list[RunEvent] = []

        async def capture(event: RunEvent) -> None:
            emitted_events.append(event)

        config = RollupConfig(interval_seconds=1.0, min_interval_seconds=0.1)
        # Override to allow fast testing
        config.interval_seconds = 0.1

        gen = RollupSummaryGenerator(
            source=source, on_rollup=capture, config=config
        )

        await gen.start()
        await asyncio.sleep(0.35)  # Should fire ~3 rollups
        await gen.stop()

        assert len(emitted_events) >= 2  # At least 2 rollups
        for event in emitted_events:
            assert event.event_type == EventType.ROLLUP_SUMMARY
            assert "4/10 done" in event.message  # 3 passed + 1 failed = 4

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self):
        source = FakeProgressSource(total=5)
        callback = AsyncMock()
        gen = RollupSummaryGenerator(
            source=source,
            on_rollup=callback,
            config=RollupConfig(interval_seconds=1.0),
        )
        await gen.start()
        await gen.start()  # Should warn, not crash
        await gen.stop()

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash_loop(self):
        source = FakeProgressSource(total=10, passed=5, duration=1.0)
        call_count = 0

        async def failing_callback(event: RunEvent) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")

        config = RollupConfig(interval_seconds=1.0, min_interval_seconds=0.05)
        config.interval_seconds = 0.05

        gen = RollupSummaryGenerator(
            source=source, on_rollup=failing_callback, config=config
        )
        await gen.start()
        await asyncio.sleep(0.2)
        await gen.stop()

        # Should have continued after the error
        assert call_count >= 2

    def test_config_update(self):
        source = FakeProgressSource(total=10)
        callback = AsyncMock()
        gen = RollupSummaryGenerator(source=source, on_rollup=callback)

        new_config = RollupConfig(interval_seconds=30.0)
        gen.config = new_config
        assert gen.config.interval_seconds == 30.0


# ---------------------------------------------------------------------------
# ReporterAgent rollup integration tests
# ---------------------------------------------------------------------------


class TestReporterAgentRollupIntegration:
    @pytest.mark.asyncio
    async def test_start_run_creates_rollup_generator(self):
        reporter = ReporterAgent(
            rollup_config=RollupConfig(interval_seconds=5.0)
        )
        channel = FakeChannel()
        reporter.add_channel(channel)

        await reporter.start_run()
        assert reporter.rollup_generator is not None
        assert reporter.rollup_generator.is_running
        await reporter.end_run()
        assert reporter.rollup_generator is None

    @pytest.mark.asyncio
    async def test_end_run_stops_rollup_generator(self):
        reporter = ReporterAgent(
            rollup_config=RollupConfig(interval_seconds=5.0)
        )
        channel = FakeChannel()
        reporter.add_channel(channel)

        await reporter.start_run()
        gen = reporter.rollup_generator
        assert gen is not None
        assert gen.is_running

        await reporter.end_run()
        assert not gen.is_running

    @pytest.mark.asyncio
    async def test_generate_rollup_now_emits_event(self):
        reporter = ReporterAgent()
        channel = FakeChannel()
        reporter.add_channel(channel)

        # Process some results first
        await reporter.process_output(
            [
                "tests/test_a.py::test_one PASSED",
                "tests/test_a.py::test_two FAILED",
            ],
            "pytest",
        )

        event = await reporter.generate_rollup_now()
        assert event.event_type == EventType.ROLLUP_SUMMARY
        assert "2" in event.message  # 2 done
        assert "1 failure" in event.message

        # Should have been sent to channel
        rollup_events = [
            e for e in channel.events
            if isinstance(e, RunEvent)
            and e.event_type == EventType.ROLLUP_SUMMARY
        ]
        assert len(rollup_events) == 1

    @pytest.mark.asyncio
    async def test_generate_rollup_now_without_start_run(self):
        """Rollup works even without calling start_run (one-shot mode)."""
        reporter = ReporterAgent()
        channel = FakeChannel()
        reporter.add_channel(channel)

        await reporter.process_output(
            ["tests/test_a.py::test_one PASSED"], "pytest"
        )

        event = await reporter.generate_rollup_now()
        assert event.event_type == EventType.ROLLUP_SUMMARY
        assert "1" in event.message

    @pytest.mark.asyncio
    async def test_rollup_config_setter(self):
        reporter = ReporterAgent()
        new_config = RollupConfig(interval_seconds=30.0)
        reporter.rollup_config = new_config
        assert reporter.rollup_config.interval_seconds == 30.0

    @pytest.mark.asyncio
    async def test_periodic_rollup_during_run(self):
        """Rollup events appear during a run with short interval."""
        config = RollupConfig(
            interval_seconds=0.1, min_interval_seconds=0.05
        )
        reporter = ReporterAgent(rollup_config=config)
        channel = FakeChannel()
        reporter.add_channel(channel)

        await reporter.start_run()
        await reporter.process_output(
            ["tests/test_a.py::test_one PASSED"], "pytest"
        )
        await asyncio.sleep(0.3)  # Wait for rollups
        await reporter.end_run()

        rollup_events = [
            e for e in channel.events
            if isinstance(e, RunEvent)
            and e.event_type == EventType.ROLLUP_SUMMARY
        ]
        assert len(rollup_events) >= 1

    @pytest.mark.asyncio
    async def test_disabled_rollup(self):
        """No rollup generator when disabled."""
        config = RollupConfig(enabled=False)
        reporter = ReporterAgent(rollup_config=config)
        channel = FakeChannel()
        reporter.add_channel(channel)

        await reporter.start_run()
        # Generator exists but should not be running
        assert reporter.rollup_generator is not None
        assert not reporter.rollup_generator.is_running

        await asyncio.sleep(0.1)
        await reporter.end_run()

        rollup_events = [
            e for e in channel.events
            if isinstance(e, RunEvent)
            and e.event_type == EventType.ROLLUP_SUMMARY
        ]
        assert len(rollup_events) == 0


# ---------------------------------------------------------------------------
# ProgressTrackerAdapter tests
# ---------------------------------------------------------------------------


class TestProgressTrackerAdapter:
    def test_adapts_progress_tracker(self):
        from test_runner.models.progress import (
            ProgressTracker,
            TestResult,
            TestStatus as PTestStatus,
        )

        tracker = ProgressTracker(total=5)
        tracker.start()
        tracker.record_result(TestResult(
            test_id="t1", name="test1",
            status=PTestStatus.PASSED, duration_seconds=0.1,
        ))
        tracker.record_result(TestResult(
            test_id="t2", name="test2",
            status=PTestStatus.FAILED, duration_seconds=0.2,
        ))

        adapter = ProgressTrackerAdapter(tracker)
        assert adapter.total == 5
        assert adapter.passed == 1
        assert adapter.failed == 1
        assert adapter.errors == 0
        assert adapter.skipped == 0
        assert adapter.duration > 0

    def test_rollup_from_progress_tracker(self):
        from test_runner.models.progress import (
            ProgressTracker,
            TestResult,
            TestStatus as PTestStatus,
        )

        tracker = ProgressTracker(total=10)
        tracker.start()
        tracker.record_result(TestResult(
            test_id="t1", name="test1",
            status=PTestStatus.PASSED, duration_seconds=0.1,
        ))

        adapter = ProgressTrackerAdapter(tracker)
        msg = format_rollup_message(
            total=adapter.total,
            passed=adapter.passed,
            failed=adapter.failed,
            errors=adapter.errors,
            skipped=adapter.skipped,
            elapsed=adapter.duration,
        )
        assert "1/10 done" in msg
        assert "all passing" in msg
