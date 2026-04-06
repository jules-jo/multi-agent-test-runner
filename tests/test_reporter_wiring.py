"""Tests for reporter wiring into the orchestrator hub.

Verifies end-to-end integration:
- The orchestrator wires the reporter agent at construction time
- The reporter's run lifecycle (start_run / end_run) is managed by the hub
- Execution output is routed through the orchestrator to the reporter
- Test results are streamed to reporting channels (CLI) in real-time
- Delegation cycles are tracked in the agent state store
- The reporter is safely stopped when errors occur
- Multiple channels receive events simultaneously
"""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any

import pytest

from test_runner.agents.base import AgentRole
from test_runner.agents.reporter.agent import ReporterAgent
from test_runner.agents.reporter.rollup import RollupConfig
from test_runner.config import Config
from test_runner.orchestrator.hub import OrchestratorHub, RunPhase, RunState
from test_runner.orchestrator.state_store import AgentStatus
from test_runner.reporting.base import ReporterBase, StreamEvent
from test_runner.reporting.cli_streaming import CLIStreamingReporter
from test_runner.reporting.events import (
    EventType,
    RunEvent,
    TestResultEvent,
    TestStatus,
)


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _make_config() -> Config:
    """Minimal config for tests."""
    return Config(
        llm_base_url="http://localhost:8080/v1",
        api_key="test-key",
        model_id="test-model",
    )


class FakeChannel(ReporterBase):
    """In-memory recording channel for assertions."""

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
    def test_result_events(self) -> list[TestResultEvent]:
        return [e for e in self.events if isinstance(e, TestResultEvent)]

    @property
    def run_events(self) -> list[RunEvent]:
        return [e for e in self.events if isinstance(e, RunEvent)]


@pytest.fixture
def channel() -> FakeChannel:
    return FakeChannel()


@pytest.fixture
def config() -> Config:
    return _make_config()


# ---------------------------------------------------------------------------
# Constructor wiring
# ---------------------------------------------------------------------------


class TestOrchestratorReporterConstruction:
    """Verify the reporter is wired at construction time."""

    def test_default_reporter_created(self, config: Config):
        """Hub creates a default ReporterAgent if none provided."""
        hub = OrchestratorHub(config, parse_mode="offline")
        assert isinstance(hub.reporter, ReporterAgent)
        assert hub.reporter.name == "reporter-agent"

    def test_custom_reporter_injected(self, config: Config):
        """Hub accepts an injected ReporterAgent."""
        custom = ReporterAgent(hard_cap_steps=10)
        hub = OrchestratorHub(config, parse_mode="offline", reporter=custom)
        assert hub.reporter is custom
        assert hub.reporter.hard_cap_steps == 10

    def test_channels_registered_at_construction(
        self, config: Config, channel: FakeChannel
    ):
        """Channels passed to the hub are registered on the reporter."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        assert len(hub.reporter.channels) == 1

    def test_multiple_channels_registered(self, config: Config):
        """Multiple channels are all registered."""
        ch1 = FakeChannel()
        ch2 = FakeChannel()
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[ch1, ch2]
        )
        assert len(hub.reporter.channels) == 2

    def test_rollup_config_applied(self, config: Config):
        """Custom RollupConfig is passed through to the reporter."""
        rc = RollupConfig(interval_seconds=5.0)
        hub = OrchestratorHub(
            config, parse_mode="offline", rollup_config=rc
        )
        assert hub.reporter.rollup_config.interval_seconds == 5.0


# ---------------------------------------------------------------------------
# Reporter start/finalize lifecycle
# ---------------------------------------------------------------------------


class TestReporterLifecycle:
    """Verify the orchestrator manages reporter start/stop correctly."""

    @pytest.mark.asyncio
    async def test_start_reporter_signals_channels(
        self, config: Config, channel: FakeChannel
    ):
        """_start_reporter calls start_run on the reporter, notifying channels."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)

        assert channel.run_started is True
        # Run-started event emitted
        started_events = [
            e for e in channel.run_events
            if e.event_type == EventType.RUN_STARTED
        ]
        assert len(started_events) == 1

    @pytest.mark.asyncio
    async def test_start_reporter_creates_delegation_cycle(
        self, config: Config, channel: FakeChannel
    ):
        """A delegation cycle is started in the agent state store."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)

        cycle = state.agent_store.latest_cycle_for(AgentRole.REPORTER)
        assert cycle is not None
        assert cycle.status == AgentStatus.RUNNING

    @pytest.mark.asyncio
    async def test_finalize_reporter_produces_summary(
        self, config: Config, channel: FakeChannel
    ):
        """_finalize_reporter populates state.report with summary."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)
        # Process some output
        await hub.feed_execution_output(
            ["tests/test_a.py::test_one PASSED"],
            framework="pytest",
        )
        await hub._finalize_reporter(state)

        assert state.report["total"] == 1
        assert state.report["passed"] == 1
        assert channel.run_ended is True

    @pytest.mark.asyncio
    async def test_finalize_reporter_closes_delegation_cycle(
        self, config: Config, channel: FakeChannel
    ):
        """Delegation cycle is marked COMPLETED after finalization."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = RunState(request="run tests")

        await hub._start_reporter(state)
        await hub._finalize_reporter(state)

        cycle = state.agent_store.latest_cycle_for(AgentRole.REPORTER)
        assert cycle is not None
        assert cycle.status == AgentStatus.COMPLETED
        assert cycle.finished_at > 0

    @pytest.mark.asyncio
    async def test_reporter_reset_between_runs(self, config: Config):
        """Reporter state is reset before each run."""
        reporter = ReporterAgent()
        hub = OrchestratorHub(
            config, parse_mode="offline", reporter=reporter
        )
        state = RunState(request="run tests")

        # Simulate prior usage
        reporter.state.record_step()
        reporter.state.record_step()
        assert reporter.state.steps_taken == 2

        await hub._start_reporter(state)
        # Should be reset
        assert reporter.state.steps_taken == 0


# ---------------------------------------------------------------------------
# Execution output routing
# ---------------------------------------------------------------------------


class TestExecutionOutputRouting:
    """Verify the orchestrator routes executor output to the reporter."""

    @pytest.mark.asyncio
    async def test_feed_execution_output_streams_to_channel(
        self, config: Config, channel: FakeChannel
    ):
        """Output fed via feed_execution_output reaches the channel."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = RunState(request="run tests")
        await hub._start_reporter(state)

        await hub.feed_execution_output(
            "tests/test_a.py::test_one PASSED\n"
            "tests/test_a.py::test_two FAILED\n",
            framework="pytest",
        )

        assert len(channel.test_result_events) == 2
        assert channel.test_result_events[0].status == TestStatus.PASS
        assert channel.test_result_events[1].status == TestStatus.FAIL

    @pytest.mark.asyncio
    async def test_feed_execution_output_list_input(
        self, config: Config, channel: FakeChannel
    ):
        """List of lines is accepted."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = RunState(request="run tests")
        await hub._start_reporter(state)

        await hub.feed_execution_output(
            ["tests/test_a.py::test_one PASSED"],
            framework="pytest",
        )

        assert len(channel.test_result_events) == 1

    @pytest.mark.asyncio
    async def test_events_emitted_individually(
        self, config: Config
    ):
        """Events arrive one-by-one in the order they are parsed."""
        emission_order: list[str] = []
        reporter = ReporterAgent()

        async def track(event: TestResultEvent) -> None:
            emission_order.append(event.test_name)

        reporter.on_event_async(track)

        hub = OrchestratorHub(
            config, parse_mode="offline", reporter=reporter
        )
        state = RunState(request="run tests")
        await hub._start_reporter(state)

        await hub.feed_execution_output(
            [
                "tests/test_a.py::test_alpha PASSED",
                "tests/test_a.py::test_beta FAILED",
                "tests/test_a.py::test_gamma PASSED",
            ],
            framework="pytest",
        )

        assert emission_order == ["test_alpha", "test_beta", "test_gamma"]

    @pytest.mark.asyncio
    async def test_stats_updated_during_streaming(
        self, config: Config, channel: FakeChannel
    ):
        """Reporter stats reflect results fed through the orchestrator."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = RunState(request="run tests")
        await hub._start_reporter(state)

        await hub.feed_execution_output(
            [
                "tests/test_a.py::test_one PASSED",
                "tests/test_a.py::test_two PASSED",
                "tests/test_a.py::test_three FAILED",
            ],
            framework="pytest",
        )

        assert hub.reporter.stats.total == 3
        assert hub.reporter.stats.passed == 2
        assert hub.reporter.stats.failed == 1


# ---------------------------------------------------------------------------
# CLI streaming integration
# ---------------------------------------------------------------------------


class TestCLIStreamingIntegration:
    """Verify the full path: orchestrator → reporter → CLIStreamingReporter."""

    @pytest.mark.asyncio
    async def test_cli_receives_events(self, config: Config):
        """CLIStreamingReporter connected via hub receives test results."""
        output = StringIO()
        cli = CLIStreamingReporter(file=output)

        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[cli]
        )
        state = RunState(request="run tests")
        await hub._start_reporter(state)

        await hub.feed_execution_output(
            [
                "tests/test_a.py::test_one PASSED",
                "tests/test_a.py::test_two FAILED",
            ],
            framework="pytest",
        )
        await hub._finalize_reporter(state)

        rendered = output.getvalue()
        assert "PASS" in rendered
        assert "FAIL" in rendered
        assert "test_one" in rendered
        assert "test_two" in rendered

    @pytest.mark.asyncio
    async def test_cli_summary_rendered(self, config: Config):
        """Final summary is rendered by CLIStreamingReporter."""
        output = StringIO()
        cli = CLIStreamingReporter(file=output)

        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[cli]
        )
        state = RunState(request="run tests")
        await hub._start_reporter(state)

        await hub.feed_execution_output(
            ["tests/test_a.py::test_one PASSED"],
            framework="pytest",
        )
        await hub._finalize_reporter(state)

        rendered = output.getvalue()
        # Summary should show all passed
        assert "passed" in rendered.lower() or "1 passed" in rendered


# ---------------------------------------------------------------------------
# Multi-channel
# ---------------------------------------------------------------------------


class TestMultiChannelRouting:
    """Verify events reach multiple channels simultaneously."""

    @pytest.mark.asyncio
    async def test_two_channels_receive_same_events(self, config: Config):
        ch1 = FakeChannel()
        ch2 = FakeChannel()
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[ch1, ch2]
        )
        state = RunState(request="run tests")
        await hub._start_reporter(state)

        await hub.feed_execution_output(
            ["tests/test_a.py::test_one PASSED"],
            framework="pytest",
        )
        await hub._finalize_reporter(state)

        assert len(ch1.test_result_events) == 1
        assert len(ch2.test_result_events) == 1
        assert ch1.run_ended is True
        assert ch2.run_ended is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestReporterErrorHandling:
    """Verify reporter is safely stopped when errors occur."""

    @pytest.mark.asyncio
    async def test_stop_reporter_safely_no_crash(self, config: Config):
        """_stop_reporter_safely does not raise even if reporter isn't started."""
        hub = OrchestratorHub(config, parse_mode="offline")
        state = RunState(request="run tests")
        # Should not raise
        await hub._stop_reporter_safely(state)

    @pytest.mark.asyncio
    async def test_channel_error_does_not_stop_reporting(
        self, config: Config
    ):
        """A failing channel should not prevent the reporter from finishing."""

        class FailingChannel(ReporterBase):
            async def on_event(self, event: StreamEvent) -> None:
                raise RuntimeError("channel exploded")

        good = FakeChannel()
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[good, FailingChannel()]
        )
        state = RunState(request="run tests")
        await hub._start_reporter(state)

        # Should not raise despite FailingChannel
        await hub.feed_execution_output(
            ["tests/test_a.py::test_one PASSED"],
            framework="pytest",
        )
        await hub._finalize_reporter(state)

        # Good channel still got the event
        assert len(good.test_result_events) == 1
        assert state.report["total"] == 1


# ---------------------------------------------------------------------------
# End-to-end integration via run()
# ---------------------------------------------------------------------------


class TestRunIntegration:
    """Full run() integration — reporter lifecycle is managed."""

    @pytest.mark.asyncio
    async def test_run_starts_and_stops_reporter(
        self, config: Config, channel: FakeChannel
    ):
        """The run() method manages the reporter lifecycle."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = await hub.run("run all tests")

        # Reporter should have been started and stopped
        assert channel.run_started is True
        assert channel.run_ended is True

        # Report should be in state
        assert isinstance(state.report, dict)
        assert "total" in state.report

        # Delegation cycle tracked
        cycle = state.agent_store.latest_cycle_for(AgentRole.REPORTER)
        assert cycle is not None
        assert cycle.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_report_in_complete_state(
        self, config: Config, channel: FakeChannel
    ):
        """State reaches COMPLETE with report populated."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = await hub.run("run tests")

        assert state.phase == RunPhase.COMPLETE
        assert state.report is not None

    @pytest.mark.asyncio
    async def test_run_records_task_level_result_when_no_test_lines_parse(
        self, config: Config, channel: FakeChannel
    ):
        """A completed run still reports task-level results when parsing yields no individual tests."""
        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[channel]
        )
        state = await hub.run("run tests")

        assert state.report["total"] >= 1

    @pytest.mark.asyncio
    async def test_run_with_cli_streaming(self, config: Config):
        """Full run with CLIStreamingReporter connected."""
        output = StringIO()
        cli = CLIStreamingReporter(file=output)

        hub = OrchestratorHub(
            config, parse_mode="offline", channels=[cli]
        )
        state = await hub.run("run pytest tests")

        assert state.phase == RunPhase.COMPLETE
        rendered = output.getvalue()
        # At minimum, the run header and summary should be rendered
        assert "Test Run Started" in rendered
