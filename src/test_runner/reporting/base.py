"""Base interface for pluggable reporting channels."""

from __future__ import annotations

import abc
from typing import Union

from .events import RunEvent, TestResultEvent

# Union of all streamable event types
StreamEvent = Union[TestResultEvent, RunEvent]


class ReporterBase(abc.ABC):
    """Abstract base for all reporting channel implementations.

    Every reporter receives a stream of events (test results, lifecycle
    updates, logs) and decides how to render them.  Implementations must
    be async-safe so they can be driven from the orchestrator's event loop.
    """

    @abc.abstractmethod
    async def on_event(self, event: StreamEvent) -> None:
        """Handle a single streaming event.

        Implementations should be non-blocking.  Heavy I/O (e.g. HTTP
        posts to a messaging platform) should be dispatched without
        blocking the caller.
        """

    async def on_run_start(self) -> None:
        """Called once when a test run begins. Override for setup."""

    async def on_run_end(self, summary: dict) -> None:
        """Called once when a test run ends with an aggregated summary.

        Args:
            summary: Dict with keys like ``total``, ``passed``, ``failed``,
                ``errors``, ``skipped``, ``duration``, ``ai_analysis``.
        """


class CLIReporterBase(ReporterBase):
    """Marker base for CLI-specific reporters.

    CLI reporters stream to stdout/stderr and may use rich formatting.
    This subclass exists so the orchestrator can distinguish CLI reporters
    from messaging-platform reporters when routing events.
    """
