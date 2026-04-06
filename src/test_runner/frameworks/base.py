"""Base protocol and data models for framework adapters.

Every framework adapter (pytest, jest, go test, etc.) implements
``FrameworkAdapter`` to provide detection, command building, and output parsing
for its specific test framework.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from test_runner.agents.parser import ParsedTestRequest, TestFramework, TestIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection models
# ---------------------------------------------------------------------------


class DetectionSignal(str, Enum):
    """Types of evidence used to detect a framework."""

    CONFIG_FILE = "config_file"          # e.g. pytest.ini, pyproject.toml [tool.pytest]
    DEPENDENCY_FILE = "dependency_file"  # e.g. requirements.txt with pytest
    TEST_FILE_PATTERN = "test_file_pattern"  # e.g. test_*.py files found
    IMPORT_STATEMENT = "import_statement"    # e.g. "import pytest" in source
    CLI_AVAILABLE = "cli_available"          # e.g. `pytest --version` succeeds
    LOCKFILE = "lockfile"                    # e.g. uv.lock, poetry.lock mentions


@dataclass
class DetectionResult:
    """Result of checking whether a framework is present in a project.

    Attributes:
        detected: Whether the framework was detected.
        framework: Which framework this result pertains to.
        confidence: Confidence level (0.0-1.0) of the detection.
        signals: Evidence that was found (or not found).
        config_path: Path to framework config if discovered.
        version: Detected version string, if available.
        details: Human-readable explanation of the detection.
    """

    detected: bool
    framework: TestFramework
    confidence: float = 0.0
    signals: list[DetectionSignal] = field(default_factory=list)
    config_path: str = ""
    version: str = ""
    details: str = ""


# ---------------------------------------------------------------------------
# Output parsing models
# ---------------------------------------------------------------------------


class TestOutcome(str, Enum):
    """Outcome of a single test case."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAIL = "xfail"       # Expected failure
    XPASS = "xpass"        # Unexpected pass
    DESELECTED = "deselected"


@dataclass
class TestCaseResult:
    """Parsed result of a single test case.

    Attributes:
        name: Fully qualified test name (e.g. tests/test_foo.py::TestBar::test_baz).
        outcome: The outcome of this test case.
        duration_seconds: How long this test took, if available.
        message: Short failure/error message, if applicable.
        traceback: Full traceback text for failures/errors.
        file_path: Source file containing the test.
        line_number: Line number in the source file, if available.
        metadata: Extra framework-specific metadata.
    """

    name: str
    outcome: TestOutcome
    duration_seconds: float = 0.0
    message: str = ""
    traceback: str = ""
    file_path: str = ""
    line_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedTestOutput:
    """Structured result of parsing framework output.

    Attributes:
        test_cases: Individual test case results.
        total: Total number of tests.
        passed: Number of passed tests.
        failed: Number of failed tests.
        errors: Number of errored tests.
        skipped: Number of skipped tests.
        duration_seconds: Total test suite duration.
        raw_stdout: Original stdout text.
        raw_stderr: Original stderr text.
        exit_code: Process exit code.
        summary_line: The framework's own summary line (e.g. "5 passed, 1 failed").
        warnings: Any warnings emitted during parsing.
        metadata: Extra framework-specific metadata.
    """

    test_cases: list[TestCaseResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    raw_stdout: str = ""
    raw_stderr: str = ""
    exit_code: int = 0
    summary_line: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether all tests passed (no failures or errors)."""
        return self.failed == 0 and self.errors == 0

    @property
    def xfailed(self) -> int:
        """Count of expected failures."""
        return sum(1 for tc in self.test_cases if tc.outcome == TestOutcome.XFAIL)

    @property
    def xpassed(self) -> int:
        """Count of unexpected passes."""
        return sum(1 for tc in self.test_cases if tc.outcome == TestOutcome.XPASS)

    @property
    def failure_details(self) -> list[TestCaseResult]:
        """Return only failed and errored test cases."""
        return [
            tc for tc in self.test_cases
            if tc.outcome in (TestOutcome.FAILED, TestOutcome.ERROR)
        ]


# ---------------------------------------------------------------------------
# Framework adapter protocol
# ---------------------------------------------------------------------------


class FrameworkAdapter(ABC):
    """Abstract base for framework adapters.

    Each adapter handles three concerns for a single test framework:
    1. **Detection**: Determine if the framework is present in a project.
    2. **Command building**: Build CLI command tokens from a ParsedTestRequest.
    3. **Output parsing**: Parse raw output into structured ParsedTestOutput.
    """

    @property
    @abstractmethod
    def framework(self) -> TestFramework:
        """The framework this adapter handles."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for display (e.g. 'pytest', 'Jest')."""
        ...

    # -- Detection -----------------------------------------------------------

    @abstractmethod
    async def detect(self, project_root: str) -> DetectionResult:
        """Detect whether this framework is present in the given project.

        Args:
            project_root: Absolute path to the project root directory.

        Returns:
            DetectionResult indicating presence, confidence, and evidence.
        """
        ...

    # -- Command building ----------------------------------------------------

    @abstractmethod
    def build_command(
        self,
        request: ParsedTestRequest,
    ) -> list[str]:
        """Build CLI command tokens for executing the test request.

        Args:
            request: The structured test request.

        Returns:
            List of command tokens (e.g. ["pytest", "-v", "tests/"]).
        """
        ...

    # -- Output parsing ------------------------------------------------------

    @abstractmethod
    def parse_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> ParsedTestOutput:
        """Parse raw command output into structured results.

        Args:
            stdout: Standard output from the test command.
            stderr: Standard error from the test command.
            exit_code: Process exit code.

        Returns:
            ParsedTestOutput with structured test results.
        """
        ...
