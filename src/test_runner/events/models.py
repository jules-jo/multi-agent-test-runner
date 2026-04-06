"""Event data models for real-time test result streaming.

These models represent the structured events emitted during test execution,
designed to be framework-agnostic and serializable for multi-channel reporting.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    """Outcome status for a completed test."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAIL = "xfail"      # Expected failure
    XPASS = "xpass"       # Unexpected pass


class ErrorInfo(BaseModel):
    """Structured error information attached to a failed/errored test."""

    message: str = Field(description="Human-readable error summary")
    type: str = Field(default="", description="Exception class name or error category")
    traceback: Optional[str] = Field(
        default=None,
        description="Full traceback string, if available",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for framework-specific error context",
    )


class TestEvent(BaseModel):
    """Base class for all streaming test events.

    Every event carries a unique id, a monotonic timestamp, and an optional
    correlation id linking it to a parent suite run.
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    suite_run_id: Optional[str] = Field(
        default=None,
        description="Correlation id grouping events from the same suite execution",
    )


# ---------------------------------------------------------------------------
# Suite-level events
# ---------------------------------------------------------------------------

class SuiteStartedEvent(TestEvent):
    """Emitted when a test suite execution begins."""

    kind: str = Field(default="suite_started", frozen=True)
    suite_name: str = Field(description="Logical name of the test suite or target")
    framework: str = Field(default="unknown", description="Test framework identifier (pytest, jest, script, etc.)")
    total_tests: Optional[int] = Field(
        default=None,
        description="Expected test count when known upfront",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary suite-level metadata (paths, tags, etc.)",
    )


class SuiteFinishedEvent(TestEvent):
    """Emitted when a test suite execution completes."""

    kind: str = Field(default="suite_finished", frozen=True)
    suite_name: str
    total: int = Field(description="Total tests executed")
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_ms: float = Field(description="Wall-clock duration in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual test events
# ---------------------------------------------------------------------------

class TestStartedEvent(TestEvent):
    """Emitted when an individual test begins execution."""

    kind: str = Field(default="test_started", frozen=True)
    test_name: str = Field(description="Fully-qualified test name (e.g. tests/test_foo.py::test_bar)")
    test_file: Optional[str] = Field(default=None, description="Source file path, if known")


class TestFinishedEvent(TestEvent):
    """Emitted when an individual test completes (pass, fail, or error)."""

    kind: str = Field(default="test_finished", frozen=True)
    test_name: str
    status: TestStatus
    duration_ms: float = Field(description="Test duration in milliseconds")
    error: Optional[ErrorInfo] = Field(
        default=None,
        description="Error details when status is FAILED or ERROR",
    )
    output: Optional[str] = Field(
        default=None,
        description="Captured stdout/stderr, truncated if necessary",
    )
    test_file: Optional[str] = Field(default=None)


class TestSkippedEvent(TestEvent):
    """Emitted when a test is skipped before execution."""

    kind: str = Field(default="test_skipped", frozen=True)
    test_name: str
    reason: str = Field(default="", description="Skip reason if provided by the framework")
    test_file: Optional[str] = Field(default=None)
