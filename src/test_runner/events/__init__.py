"""Streaming event model and callback protocol for real-time test results."""

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

__all__ = [
    "ErrorInfo",
    "EventCallback",
    "EventCallbackRegistry",
    "SuiteFinishedEvent",
    "SuiteStartedEvent",
    "TestEvent",
    "TestFinishedEvent",
    "TestSkippedEvent",
    "TestStartedEvent",
    "TestStatus",
]
