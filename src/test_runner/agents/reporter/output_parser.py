"""Parsers that convert raw executor output into TestResultEvent objects.

Each parser handles a specific test framework's output format. The
OutputParserRegistry allows looking up the appropriate parser for a
given framework.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterator

from test_runner.reporting.events import TestResultEvent, TestStatus


class OutputParser(ABC):
    """Base class for framework-specific output parsers.

    Parsers are incremental — they accept output lines one at a time
    and yield TestResultEvent objects as soon as an individual test
    result can be determined.  This enables real-time streaming rather
    than waiting for the full output.
    """

    @property
    @abstractmethod
    def framework(self) -> str:
        """Framework identifier this parser handles (e.g. 'pytest')."""
        ...

    @abstractmethod
    def feed_line(self, line: str) -> Iterator[TestResultEvent]:
        """Feed a single output line and yield any completed test results.

        Implementations should maintain internal state as needed to
        correlate multi-line output into single test results.

        Args:
            line: A single line of executor output.

        Yields:
            TestResultEvent for each completed test result detected.
        """
        ...

    def flush(self) -> Iterator[TestResultEvent]:
        """Flush any buffered state at end-of-output.

        Called when the executor output stream ends.  Override if the
        parser buffers partial results that should be emitted.

        Yields:
            Any remaining TestResultEvent objects.
        """
        return iter(())


class PytestOutputParser(OutputParser):
    """Parse pytest verbose output into TestResultEvent objects.

    Recognises lines like:
        tests/test_foo.py::test_bar PASSED
        tests/test_foo.py::test_baz FAILED
        tests/test_foo.py::test_qux SKIPPED
        tests/test_foo.py::test_err ERROR
    """

    # Pattern: file::test_name STATUS (optional duration)
    _RESULT_RE = re.compile(
        r"^(?P<path>[^\s]+?)::(?P<name>[^\s]+)\s+"
        r"(?P<status>PASSED|FAILED|ERROR|SKIPPED)"
        r"(?:\s+\[.*\])?"  # optional progress indicator like [50%]
        r"(?:\s+\((?P<dur>[\d.]+)s\))?"  # optional duration
    )

    _STATUS_MAP = {
        "PASSED": TestStatus.PASS,
        "FAILED": TestStatus.FAIL,
        "ERROR": TestStatus.ERROR,
        "SKIPPED": TestStatus.SKIP,
    }

    @property
    def framework(self) -> str:
        return "pytest"

    def feed_line(self, line: str) -> Iterator[TestResultEvent]:
        stripped = line.strip()
        m = self._RESULT_RE.match(stripped)
        if m:
            duration = float(m.group("dur")) if m.group("dur") else 0.0
            yield TestResultEvent(
                test_name=m.group("name"),
                status=self._STATUS_MAP[m.group("status")],
                duration=duration,
                file_path=m.group("path"),
                suite=m.group("path"),
            )


class GenericOutputParser(OutputParser):
    """Fallback parser for arbitrary scripts and unknown frameworks.

    Looks for common patterns:
        - Lines containing PASS, FAIL, ERROR, SKIP (case-insensitive)
        - TAP-style output: "ok 1 - test name", "not ok 2 - test name"
        - Exit-code based: produces a single result at flush time
    """

    # TAP-style: "ok 1 - description" or "not ok 2 - description"
    _TAP_RE = re.compile(
        r"^(?P<not>not\s+)?ok\s+(?P<num>\d+)(?:\s+-\s+(?P<desc>.+))?$"
    )

    # Generic pass/fail markers
    _GENERIC_PASS_RE = re.compile(r"\bPASS(?:ED)?\b", re.IGNORECASE)
    _GENERIC_FAIL_RE = re.compile(r"\bFAIL(?:ED|URE)?\b", re.IGNORECASE)
    _GENERIC_ERROR_RE = re.compile(r"\bERROR\b", re.IGNORECASE)
    _GENERIC_SKIP_RE = re.compile(r"\bSKIP(?:PED)?\b", re.IGNORECASE)

    def __init__(self) -> None:
        self._found_tap = False

    @property
    def framework(self) -> str:
        return "generic"

    def feed_line(self, line: str) -> Iterator[TestResultEvent]:
        stripped = line.strip()
        if not stripped:
            return

        # Try TAP format first
        tap = self._TAP_RE.match(stripped)
        if tap:
            self._found_tap = True
            is_fail = tap.group("not") is not None
            name = tap.group("desc") or f"test-{tap.group('num')}"
            yield TestResultEvent(
                test_name=name,
                status=TestStatus.FAIL if is_fail else TestStatus.PASS,
                duration=0.0,
                message=stripped,
            )
            return

        # Don't double-parse if we detected TAP format
        if self._found_tap:
            return


class JestOutputParser(OutputParser):
    """Parse Jest verbose output into TestResultEvent objects.

    Recognises lines like:
        ✓ should do something (5 ms)
        ✕ should fail (12 ms)
        ○ skipped test name
    """

    _PASS_RE = re.compile(
        r"^\s*[✓✔]\s+(?P<name>.+?)(?:\s+\((?P<dur>\d+)\s*ms\))?\s*$"
    )
    _FAIL_RE = re.compile(
        r"^\s*[✕✗×]\s+(?P<name>.+?)(?:\s+\((?P<dur>\d+)\s*ms\))?\s*$"
    )
    _SKIP_RE = re.compile(
        r"^\s*[○⊘]\s+(?:skipped\s+)?(?P<name>.+?)\s*$"
    )

    _current_suite: str = ""

    @property
    def framework(self) -> str:
        return "jest"

    def feed_line(self, line: str) -> Iterator[TestResultEvent]:
        # Track suite context
        stripped = line.strip()

        # Suite header: lines that don't match test patterns and end without timing
        m = self._PASS_RE.match(stripped)
        if m:
            dur_ms = int(m.group("dur")) if m.group("dur") else 0
            yield TestResultEvent(
                test_name=m.group("name"),
                status=TestStatus.PASS,
                duration=dur_ms / 1000.0,
                suite=self._current_suite,
            )
            return

        m = self._FAIL_RE.match(stripped)
        if m:
            dur_ms = int(m.group("dur")) if m.group("dur") else 0
            yield TestResultEvent(
                test_name=m.group("name"),
                status=TestStatus.FAIL,
                duration=dur_ms / 1000.0,
                suite=self._current_suite,
            )
            return

        m = self._SKIP_RE.match(stripped)
        if m:
            yield TestResultEvent(
                test_name=m.group("name"),
                status=TestStatus.SKIP,
                duration=0.0,
                suite=self._current_suite,
            )
            return


class OutputParserRegistry:
    """Registry of output parsers keyed by framework name.

    Provides lookup and fallback to GenericOutputParser for unknown
    frameworks.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, type[OutputParser]] = {}
        # Register built-in parsers
        self.register(PytestOutputParser)
        self.register(JestOutputParser)
        self.register(GenericOutputParser)

    def register(self, parser_cls: type[OutputParser]) -> None:
        """Register a parser class. Instantiates to read its framework name."""
        instance = parser_cls()
        self._parsers[instance.framework] = parser_cls

    def get(self, framework: str) -> OutputParser:
        """Get a fresh parser instance for the given framework.

        Falls back to GenericOutputParser if framework is unknown.
        """
        cls = self._parsers.get(framework, GenericOutputParser)
        return cls()

    @property
    def supported_frameworks(self) -> list[str]:
        return list(self._parsers.keys())
