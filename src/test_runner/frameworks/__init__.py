"""Framework adapters for test detection, command building, and output parsing.

Each adapter implements the FrameworkAdapter protocol, providing:
- Detection: Does this project use this framework?
- Command building: Translate intent into CLI commands.
- Output parsing: Parse raw stdout/stderr into structured results.
"""

from test_runner.frameworks.base import (
    DetectionResult,
    DetectionSignal,
    FrameworkAdapter,
    ParsedTestOutput,
    TestCaseResult,
    TestOutcome,
)
from test_runner.frameworks.jest_adapter import JestAdapter
from test_runner.frameworks.pytest_adapter import PytestAdapter

__all__ = [
    "DetectionResult",
    "DetectionSignal",
    "FrameworkAdapter",
    "JestAdapter",
    "ParsedTestOutput",
    "PytestAdapter",
    "TestCaseResult",
    "TestOutcome",
]
