"""Tests for periodic rollup wiring in the orchestrator execution loop.

Complements ``test_reporter_wiring.py`` (which tests reporter lifecycle
and output routing) by focusing on the periodic rollup aspect:

1. Rollup events are emitted at intervals during execution and reach channels
2. Rollup is not emitted when disabled
3. Rollup data includes expected progress fields
4. Rollup reaches all registered channels simultaneously
5. Channel errors don't block rollup delivery to other channels
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from test_runner.agents.reporter.agent import ReporterAgent
from test_runner.agents.reporter.rollup import RollupConfig
from test_runner.config import Config
from test_runner.orchestrator.hub import OrchestratorHub, RunPhase, RunState
from test_runner.reporting.base import ReporterBase, StreamEvent
from test_runner.reporting.events import EventType, RunEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class InMemoryChannel(ReporterBase):
    """In-memory channel that records all events."""

    def __init__(self) -> None:
        self.events: list[StreamEvent] = []
        self.run_started = False
        self.run_ended = False
        self.end_summary: dict[str, Any] = {}

    async def on_event(self, event: StreamEvent) -> None:
        self.events.append(event)

    async def on_run_start(self) -> None:
        self.run_started = True

    async def on_run_end(self, summary: dict) -> None:
        self.run_ended = True
        self.end_summary = summary

    @property
    def rollup_events(self) -> list[RunEvent]:
        return [
            e
            for e in self.events
            if isinstance(e, RunEvent) and e.event_type == EventType.ROLLUP_SUMMARY
        ]


def _make_config() -> Config:
    return Config(
        llm_base_url="http://localhost:8080/v1",
        api_key="test-key",
        model_id="test-model",
    )


# ---------------------------------------------------------------------------
# Periodic rollup emission during execution
# ---------------------------------------------------------------------------


class TestPeriodicRollupDuringExecution:
    """Verify that rollup summaries are emitted at intervals during execution."""

    @pytest.mark.asyncio
    async def test_rollup_events_emitted_during_execution(self):
        """Rollup events appear when execution takes longer than the interval."""
        rc = RollupConfig(interval_seconds=0.05, min_interval_seconds=0.02)
        ch = InMemoryChannel()
        hub = OrchestratorHub(
            _make_config(), parse_mode="offline", rollup_config=rc, channels=[ch],
        )
        state = RunState(request="run tests")

        # Simulate execution phase with reporter lifecycle
        await hub._start_reporter(state)
        try:
            await asyncio.sleep(0.15)  # enough for ~3 rollups
        finally:
            await hub._finalize_reporter(state)

        assert len(ch.rollup_events) >= 1
        for evt in ch.rollup_events:
            assert evt.event_type == EventType.ROLLUP_SUMMARY
            assert "done" in evt.message

    @pytest.mark.asyncio
    async def test_no_rollup_when_disabled(self):
        """No rollup events when rollup is disabled."""
        rc = RollupConfig(enabled=False)
        ch = InMemoryChannel()
        hub = OrchestratorHub(
            _make_config(), parse_mode="offline", rollup_config=rc, channels=[ch],
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)
        await asyncio.sleep(0.1)
        await hub._finalize_reporter(state)

        assert len(ch.rollup_events) == 0

    @pytest.mark.asyncio
    async def test_rollup_includes_structured_data(self):
        """Each rollup event carries structured progress data."""
        rc = RollupConfig(interval_seconds=0.05, min_interval_seconds=0.02)
        ch = InMemoryChannel()
        hub = OrchestratorHub(
            _make_config(), parse_mode="offline", rollup_config=rc, channels=[ch],
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)
        try:
            # Feed results so rollup has non-zero data
            await hub.feed_execution_output(
                ["tests/test_a.py::test_one PASSED"], "pytest",
            )
            await asyncio.sleep(0.1)
        finally:
            await hub._finalize_reporter(state)

        assert len(ch.rollup_events) >= 1
        data = ch.rollup_events[0].data
        assert "total" in data
        assert "passed" in data
        assert "failed" in data
        assert "elapsed" in data
        assert "rollup_number" in data
        assert data["passed"] >= 1

    @pytest.mark.asyncio
    async def test_rollup_message_reflects_progress(self):
        """Rollup message text reflects tests completed so far."""
        rc = RollupConfig(interval_seconds=0.05, min_interval_seconds=0.02)
        ch = InMemoryChannel()
        hub = OrchestratorHub(
            _make_config(), parse_mode="offline", rollup_config=rc, channels=[ch],
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)
        try:
            await hub.feed_execution_output(
                [
                    "tests/test_a.py::test_one PASSED",
                    "tests/test_a.py::test_two FAILED",
                ],
                "pytest",
            )
            await asyncio.sleep(0.1)
        finally:
            await hub._finalize_reporter(state)

        assert len(ch.rollup_events) >= 1
        msg = ch.rollup_events[0].message
        assert "2 done" in msg  # 1 pass + 1 fail = 2
        assert "1 failure" in msg

    @pytest.mark.asyncio
    async def test_rollup_generator_cleaned_up_after_finalize(self):
        """Generator is removed after finalization."""
        rc = RollupConfig(interval_seconds=1.0)
        ch = InMemoryChannel()
        hub = OrchestratorHub(
            _make_config(), parse_mode="offline", rollup_config=rc, channels=[ch],
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)
        assert hub.reporter.rollup_generator is not None
        assert hub.reporter.rollup_generator.is_running

        await hub._finalize_reporter(state)
        assert hub.reporter.rollup_generator is None


# ---------------------------------------------------------------------------
# Multi-channel rollup delivery
# ---------------------------------------------------------------------------


class TestRollupMultipleChannels:
    """Verify rollup events are delivered to all registered channels."""

    @pytest.mark.asyncio
    async def test_rollup_reaches_all_channels(self):
        rc = RollupConfig(interval_seconds=0.05, min_interval_seconds=0.02)
        ch1 = InMemoryChannel()
        ch2 = InMemoryChannel()
        hub = OrchestratorHub(
            _make_config(), parse_mode="offline", rollup_config=rc, channels=[ch1, ch2],
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)
        try:
            await asyncio.sleep(0.12)
        finally:
            await hub._finalize_reporter(state)

        assert len(ch1.rollup_events) >= 1
        assert len(ch2.rollup_events) >= 1

    @pytest.mark.asyncio
    async def test_channel_error_does_not_block_rollup(self):
        """A failing channel doesn't prevent other channels from receiving rollups."""
        rc = RollupConfig(interval_seconds=0.05, min_interval_seconds=0.02)

        class FailingChannel(ReporterBase):
            async def on_event(self, event: StreamEvent) -> None:
                raise RuntimeError("channel error")

            async def on_run_start(self) -> None:
                pass

            async def on_run_end(self, summary: dict) -> None:
                pass

        good_ch = InMemoryChannel()
        hub = OrchestratorHub(
            _make_config(),
            parse_mode="offline",
            rollup_config=rc,
            channels=[FailingChannel(), good_ch],
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)
        try:
            await asyncio.sleep(0.12)
        finally:
            await hub._finalize_reporter(state)

        assert len(good_ch.rollup_events) >= 1


# ---------------------------------------------------------------------------
# Full run() integration with rollup
# ---------------------------------------------------------------------------


class TestRunIntegrationWithRollup:
    """Verify rollup works through a full run() call."""

    @pytest.mark.asyncio
    async def test_run_completes_with_rollup_enabled(self):
        """A normal run() with rollup enabled completes cleanly."""
        rc = RollupConfig(interval_seconds=1.0)
        ch = InMemoryChannel()
        hub = OrchestratorHub(
            _make_config(), parse_mode="offline", rollup_config=rc, channels=[ch],
        )
        state = await hub.run("run all tests")

        assert state.phase == RunPhase.COMPLETE
        assert ch.run_started is True
        assert ch.run_ended is True
        assert hub.reporter.rollup_generator is None  # cleaned up

    @pytest.mark.asyncio
    async def test_run_completes_with_rollup_disabled(self):
        """A normal run() with rollup disabled still completes."""
        rc = RollupConfig(enabled=False)
        ch = InMemoryChannel()
        hub = OrchestratorHub(
            _make_config(), parse_mode="offline", rollup_config=rc, channels=[ch],
        )
        state = await hub.run("run tests")

        assert state.phase == RunPhase.COMPLETE
        assert len(ch.rollup_events) == 0
