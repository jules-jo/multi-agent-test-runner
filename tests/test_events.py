"""Tests for the streaming event model and callback protocol."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from test_runner.events.models import (
    ErrorInfo,
    SuiteFinishedEvent,
    SuiteStartedEvent,
    TestEvent,
    TestFinishedEvent,
    TestSkippedEvent,
    TestStartedEvent,
    TestStatus,
)
from test_runner.events.callbacks import EventCallback, EventCallbackRegistry


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestEventModels:
    """Verify event model construction, defaults, and serialization."""

    def test_test_status_values(self):
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.ERROR.value == "error"
        assert TestStatus.SKIPPED.value == "skipped"

    def test_error_info_defaults(self):
        err = ErrorInfo(message="boom")
        assert err.message == "boom"
        assert err.type == ""
        assert err.traceback is None
        assert err.details == {}

    def test_error_info_full(self):
        err = ErrorInfo(
            message="assertion failed",
            type="AssertionError",
            traceback="File ...",
            details={"line": 42},
        )
        assert err.type == "AssertionError"
        assert err.details["line"] == 42

    def test_base_event_auto_fields(self):
        e = TestEvent()
        assert e.event_id  # non-empty
        assert e.timestamp is not None
        assert e.suite_run_id is None

    def test_suite_started_event(self):
        ev = SuiteStartedEvent(
            suite_name="unit-tests",
            framework="pytest",
            total_tests=10,
            suite_run_id="run-1",
        )
        assert ev.kind == "suite_started"
        assert ev.suite_name == "unit-tests"
        assert ev.framework == "pytest"
        assert ev.total_tests == 10

    def test_suite_finished_event(self):
        ev = SuiteFinishedEvent(
            suite_name="unit-tests",
            total=10,
            passed=8,
            failed=1,
            errors=1,
            skipped=0,
            duration_ms=1234.5,
        )
        assert ev.kind == "suite_finished"
        assert ev.total == 10
        assert ev.passed == 8

    def test_test_started_event(self):
        ev = TestStartedEvent(test_name="test_foo", test_file="tests/test_foo.py")
        assert ev.kind == "test_started"
        assert ev.test_name == "test_foo"

    def test_test_finished_event_pass(self):
        ev = TestFinishedEvent(
            test_name="test_bar",
            status=TestStatus.PASSED,
            duration_ms=42.0,
        )
        assert ev.status == TestStatus.PASSED
        assert ev.error is None

    def test_test_finished_event_fail_with_error(self):
        ev = TestFinishedEvent(
            test_name="test_baz",
            status=TestStatus.FAILED,
            duration_ms=100.0,
            error=ErrorInfo(message="expected 1 got 2", type="AssertionError"),
        )
        assert ev.status == TestStatus.FAILED
        assert ev.error is not None
        assert ev.error.type == "AssertionError"

    def test_test_skipped_event(self):
        ev = TestSkippedEvent(test_name="test_skip", reason="not on linux")
        assert ev.kind == "test_skipped"
        assert ev.reason == "not on linux"

    def test_event_serialization_roundtrip(self):
        ev = TestFinishedEvent(
            test_name="test_ser",
            status=TestStatus.PASSED,
            duration_ms=5.0,
        )
        data = ev.model_dump()
        restored = TestFinishedEvent.model_validate(data)
        assert restored.test_name == ev.test_name
        assert restored.status == ev.status

    def test_unique_event_ids(self):
        ids = {TestEvent().event_id for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Callback protocol tests
# ---------------------------------------------------------------------------


@dataclass
class SpyCallback:
    """Test double that records every callback invocation."""

    suite_started: list[SuiteStartedEvent] = field(default_factory=list)
    test_started: list[TestStartedEvent] = field(default_factory=list)
    test_finished: list[TestFinishedEvent] = field(default_factory=list)
    test_skipped: list[TestSkippedEvent] = field(default_factory=list)
    suite_finished: list[SuiteFinishedEvent] = field(default_factory=list)
    generic: list[TestEvent] = field(default_factory=list)

    async def on_suite_started(self, event: SuiteStartedEvent) -> None:
        self.suite_started.append(event)

    async def on_test_started(self, event: TestStartedEvent) -> None:
        self.test_started.append(event)

    async def on_test_finished(self, event: TestFinishedEvent) -> None:
        self.test_finished.append(event)

    async def on_test_skipped(self, event: TestSkippedEvent) -> None:
        self.test_skipped.append(event)

    async def on_suite_finished(self, event: SuiteFinishedEvent) -> None:
        self.suite_finished.append(event)

    async def on_event(self, event: TestEvent) -> None:
        self.generic.append(event)


class TestEventCallbackProtocol:
    """Verify that SpyCallback satisfies the EventCallback protocol."""

    def test_spy_is_event_callback(self):
        assert isinstance(SpyCallback(), EventCallback)


class TestEventCallbackRegistry:
    """Verify registry dispatch behavior."""

    async def test_emit_suite_started(self):
        reg = EventCallbackRegistry()
        spy = SpyCallback()
        reg.register(spy)

        ev = SuiteStartedEvent(suite_name="s1", framework="pytest")
        await reg.emit(ev)

        assert len(spy.suite_started) == 1
        assert spy.suite_started[0] is ev
        # Generic handler also called
        assert len(spy.generic) == 1

    async def test_emit_test_finished(self):
        reg = EventCallbackRegistry()
        spy = SpyCallback()
        reg.register(spy)

        ev = TestFinishedEvent(
            test_name="t1", status=TestStatus.PASSED, duration_ms=1.0
        )
        await reg.emit(ev)

        assert len(spy.test_finished) == 1
        assert len(spy.generic) == 1

    async def test_emit_test_skipped(self):
        reg = EventCallbackRegistry()
        spy = SpyCallback()
        reg.register(spy)

        ev = TestSkippedEvent(test_name="t1", reason="skip")
        await reg.emit(ev)

        assert len(spy.test_skipped) == 1

    async def test_multiple_callbacks(self):
        reg = EventCallbackRegistry()
        spy1 = SpyCallback()
        spy2 = SpyCallback()
        reg.register(spy1)
        reg.register(spy2)

        ev = TestStartedEvent(test_name="t1")
        await reg.emit(ev)

        assert len(spy1.test_started) == 1
        assert len(spy2.test_started) == 1

    async def test_unregister(self):
        reg = EventCallbackRegistry()
        spy = SpyCallback()
        reg.register(spy)
        reg.unregister(spy)

        ev = TestStartedEvent(test_name="t1")
        await reg.emit(ev)

        assert len(spy.test_started) == 0

    async def test_error_in_callback_does_not_propagate(self):
        """A broken callback must not halt other callbacks or the run."""

        @dataclass
        class BrokenCallback:
            async def on_suite_started(self, event: SuiteStartedEvent) -> None:
                raise RuntimeError("kaboom")

            async def on_test_started(self, event: TestStartedEvent) -> None:
                pass

            async def on_test_finished(self, event: TestFinishedEvent) -> None:
                pass

            async def on_test_skipped(self, event: TestSkippedEvent) -> None:
                pass

            async def on_suite_finished(self, event: SuiteFinishedEvent) -> None:
                pass

            async def on_event(self, event: TestEvent) -> None:
                pass

        reg = EventCallbackRegistry()
        spy = SpyCallback()
        broken = BrokenCallback()
        reg.register(broken)
        reg.register(spy)

        ev = SuiteStartedEvent(suite_name="s1", framework="pytest")
        # Should NOT raise
        await reg.emit(ev)

        # Healthy spy still received the event
        assert len(spy.suite_started) == 1

    async def test_callbacks_property_returns_copy(self):
        reg = EventCallbackRegistry()
        spy = SpyCallback()
        reg.register(spy)
        cbs = reg.callbacks
        cbs.clear()
        assert len(reg.callbacks) == 1  # original unaffected
