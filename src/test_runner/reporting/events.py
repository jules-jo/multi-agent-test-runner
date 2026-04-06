"""Streaming event models for test result reporting."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class TestStatus(str, enum.Enum):
    """Status of an individual test execution."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class EventType(str, enum.Enum):
    """Types of streaming events emitted during a test run."""

    # Lifecycle events
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"

    # Suite-level events
    SUITE_STARTED = "suite_started"
    SUITE_COMPLETED = "suite_completed"

    # Individual test events
    TEST_STARTED = "test_started"
    TEST_RESULT = "test_result"

    # Informational
    LOG = "log"
    DISCOVERY = "discovery"
    TROUBLESHOOT = "troubleshoot"

    # Periodic rollup
    ROLLUP_SUMMARY = "rollup_summary"


@dataclass(frozen=True)
class TestResultEvent:
    """Immutable event representing the result of a single test execution.

    This is the primary event streamed from the executor to the reporter
    during a test run.
    """

    test_name: str
    status: TestStatus
    duration: float  # seconds
    message: str = ""

    # Optional metadata for richer reporting
    suite: str = ""
    file_path: str = ""
    line_number: int | None = None
    stdout: str = ""
    stderr: str = ""
    error_details: str = ""

    # Envelope fields
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    event_type: EventType = EventType.TEST_RESULT

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status in (TestStatus.FAIL, TestStatus.ERROR)


@dataclass(frozen=True)
class RunEvent:
    """Generic envelope for non-test-result streaming events.

    Covers lifecycle, log, discovery, and troubleshooting events.
    """

    event_type: EventType
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
