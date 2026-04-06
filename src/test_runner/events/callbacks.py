"""Callback protocol for consuming streaming test events.

Provides a pluggable abstraction so reporting channels (CLI, Teams, file, etc.)
can independently subscribe to the event stream without coupling to each other
or to the execution layer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

from test_runner.events.models import (
    SuiteFinishedEvent,
    SuiteStartedEvent,
    TestEvent,
    TestFinishedEvent,
    TestSkippedEvent,
    TestStartedEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback protocol — any reporting channel implements this
# ---------------------------------------------------------------------------

@runtime_checkable
class EventCallback(Protocol):
    """Protocol that reporting channels must satisfy.

    All methods are async to support non-blocking I/O (e.g. posting to a
    messaging platform). Implementations may choose to be sync internally
    by simply awaiting nothing, but the interface is consistently async.
    """

    async def on_suite_started(self, event: SuiteStartedEvent) -> None:
        """Called when a test suite begins."""
        ...

    async def on_test_started(self, event: TestStartedEvent) -> None:
        """Called when an individual test begins."""
        ...

    async def on_test_finished(self, event: TestFinishedEvent) -> None:
        """Called when an individual test completes."""
        ...

    async def on_test_skipped(self, event: TestSkippedEvent) -> None:
        """Called when a test is skipped."""
        ...

    async def on_suite_finished(self, event: SuiteFinishedEvent) -> None:
        """Called when a test suite completes."""
        ...

    async def on_event(self, event: TestEvent) -> None:
        """Catch-all for any event type not covered by specific handlers.

        The registry dispatches to specific handlers first, then always
        calls on_event for generic processing (logging, forwarding, etc.).
        """
        ...


# ---------------------------------------------------------------------------
# Registry — fan-out to multiple callbacks
# ---------------------------------------------------------------------------

class EventCallbackRegistry:
    """Manages multiple EventCallback subscribers and dispatches events.

    The orchestrator or executor holds a single registry instance and calls
    ``emit()`` for each event.  The registry fans out to all registered
    callbacks concurrently.
    """

    def __init__(self) -> None:
        self._callbacks: list[EventCallback] = []

    def register(self, callback: EventCallback) -> None:
        """Add a callback subscriber."""
        self._callbacks.append(callback)

    def unregister(self, callback: EventCallback) -> None:
        """Remove a callback subscriber."""
        self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    @property
    def callbacks(self) -> list[EventCallback]:
        """Read-only view of registered callbacks."""
        return list(self._callbacks)

    async def emit(self, event: TestEvent) -> None:
        """Dispatch *event* to every registered callback.

        Specific handler methods (on_suite_started, on_test_finished, etc.)
        are called based on event type, followed by the generic ``on_event``
        for every event.

        Errors in individual callbacks are logged but never propagate —
        one broken reporter must not halt the test run.
        """
        tasks: list[asyncio.Task[None]] = []

        for cb in self._callbacks:
            tasks.extend(self._handler_tasks(cb, event))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    logger.error(
                        "Event callback error (task %d): %s", i, result, exc_info=result
                    )

    # ------------------------------------------------------------------

    @staticmethod
    def _handler_tasks(
        cb: EventCallback, event: TestEvent
    ) -> list[asyncio.Task[None]]:
        """Build the list of async tasks for a single callback + event."""
        loop = asyncio.get_event_loop()
        tasks: list[asyncio.Task[None]] = []

        # Dispatch to type-specific handler
        if isinstance(event, SuiteStartedEvent):
            tasks.append(loop.create_task(cb.on_suite_started(event)))
        elif isinstance(event, TestFinishedEvent):
            tasks.append(loop.create_task(cb.on_test_finished(event)))
        elif isinstance(event, TestStartedEvent):
            tasks.append(loop.create_task(cb.on_test_started(event)))
        elif isinstance(event, TestSkippedEvent):
            tasks.append(loop.create_task(cb.on_test_skipped(event)))
        elif isinstance(event, SuiteFinishedEvent):
            tasks.append(loop.create_task(cb.on_suite_finished(event)))

        # Always call the generic handler
        tasks.append(loop.create_task(cb.on_event(event)))

        return tasks
