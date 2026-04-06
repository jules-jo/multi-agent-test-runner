"""Pytest framework adapter: detection, command building, and output parsing.

Provides full lifecycle support for pytest-based projects:
- Detects pytest via config files, dependency files, test file patterns, and CLI.
- Builds correct ``pytest`` CLI commands for various intents (run, list, rerun, etc.).
- Parses pytest output (both standard and verbose) into structured results.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from test_runner.agents.parser import ParsedTestRequest, TestFramework, TestIntent
from test_runner.frameworks.base import (
    DetectionResult,
    DetectionSignal,
    FrameworkAdapter,
    ParsedTestOutput,
    TestCaseResult,
    TestOutcome,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns for pytest output parsing
# ---------------------------------------------------------------------------

# Summary line: "====== 5 passed, 2 failed, 1 skipped in 1.23s ======"
_SUMMARY_RE = re.compile(
    r"=+\s+"
    r"(?P<summary>.*?)"
    r"\s+in\s+(?P<duration>[\d.]+)s?"
    r"\s*=+",
)

# Individual count tokens within the summary: "5 passed", "2 failed", etc.
_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<label>passed|failed|error|errors|skipped|"
    r"xfailed|xpassed|warnings?|deselected)",
)

# Verbose mode test result line: "tests/test_foo.py::test_bar PASSED"
# Also matches: "tests/test_foo.py::TestClass::test_bar FAILED  [ 50%]"
_VERBOSE_RESULT_RE = re.compile(
    r"^(?P<nodeid>\S+::\S+)\s+(?P<outcome>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)"
    r"(?:\s+\[[\s\d]+%\])?\s*$",
    re.MULTILINE,
)

# Short test summary info header
_SHORT_SUMMARY_HEADER_RE = re.compile(
    r"^=+\s+short test summary info\s+=+$",
    re.MULTILINE,
)

# FAILED line in short summary: "FAILED tests/test_foo.py::test_bar - AssertionError: ..."
_SHORT_SUMMARY_FAILED_RE = re.compile(
    r"^FAILED\s+(?P<nodeid>\S+?)(?:\s+-\s+(?P<message>.+))?$",
    re.MULTILINE,
)

# ERROR line in short summary
_SHORT_SUMMARY_ERROR_RE = re.compile(
    r"^ERROR\s+(?P<nodeid>\S+?)(?:\s+-\s+(?P<message>.+))?$",
    re.MULTILINE,
)

# Duration per test in verbose mode: "tests/test_foo.py::test_bar PASSED (0.01s)"
_DURATION_RE = re.compile(
    r"(?P<nodeid>\S+::\S+)\s+(?:PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)"
    r"\s*\((?P<duration>[\d.]+)s\)",
)

# Collect-only output line: "<Module test_foo.py>" or "<Function test_bar>"
_COLLECT_RE = re.compile(
    r"^<(?P<kind>Module|Class|Function|Method)\s+(?P<name>.+)>$",
    re.MULTILINE,
)

# Collect-only -q output: "tests/test_foo.py::test_bar"
_COLLECT_Q_RE = re.compile(
    r"^(?P<nodeid>\S+::\S+)\s*$",
    re.MULTILINE,
)

# FAILURES section: "_______ test_name _______" separator
_FAILURE_HEADER_RE = re.compile(
    r"^_{3,}\s+(?P<name>.+?)\s+_{3,}$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Pytest adapter
# ---------------------------------------------------------------------------


class PytestAdapter(FrameworkAdapter):
    """Full-lifecycle adapter for the pytest framework.

    Detection:
        Checks for pytest config files (pytest.ini, pyproject.toml [tool.pytest],
        setup.cfg, conftest.py), dependency references (requirements*.txt,
        pyproject.toml, Pipfile), test file patterns (test_*.py), and CLI
        availability.

    Command building:
        Constructs ``pytest`` CLI commands for run, list, rerun-failed, and
        run-specific intents with proper flags.

    Output parsing:
        Parses both standard and verbose pytest output to extract per-test
        results, summary counts, duration, and failure details.
    """

    # -- FrameworkAdapter interface ------------------------------------------

    @property
    def framework(self) -> TestFramework:
        return TestFramework.PYTEST

    @property
    def display_name(self) -> str:
        return "pytest"

    # -- Detection -----------------------------------------------------------

    async def detect(self, project_root: str) -> DetectionResult:
        """Detect pytest presence by checking multiple signals."""
        root = Path(project_root)
        signals: list[DetectionSignal] = []
        details_parts: list[str] = []
        config_path = ""
        version = ""

        # 1. Check for pytest config files
        config_path_found = self._check_config_files(root)
        if config_path_found:
            signals.append(DetectionSignal.CONFIG_FILE)
            config_path = config_path_found
            details_parts.append(f"Config found: {config_path_found}")

        # 2. Check dependency files
        if self._check_dependency_files(root):
            signals.append(DetectionSignal.DEPENDENCY_FILE)
            details_parts.append("pytest listed in dependencies")

        # 3. Check for test file patterns
        test_files = self._find_test_files(root)
        if test_files:
            signals.append(DetectionSignal.TEST_FILE_PATTERN)
            details_parts.append(f"Found {len(test_files)} test file(s)")

        # 4. Check for conftest.py (strong pytest signal)
        conftest_files = list(root.rglob("conftest.py"))
        # Exclude .venv and node_modules
        conftest_files = [
            f for f in conftest_files
            if not any(part in f.parts for part in (".venv", "node_modules", "__pycache__"))
        ]
        if conftest_files:
            signals.append(DetectionSignal.IMPORT_STATEMENT)  # conftest implies pytest
            details_parts.append(f"Found {len(conftest_files)} conftest.py file(s)")

        # 5. Check CLI availability
        cli_version = await self._check_cli(project_root)
        if cli_version:
            signals.append(DetectionSignal.CLI_AVAILABLE)
            version = cli_version
            details_parts.append(f"CLI available: pytest {version}")

        # 6. Check lockfiles
        if self._check_lockfiles(root):
            signals.append(DetectionSignal.LOCKFILE)
            details_parts.append("pytest found in lockfile")

        # Calculate confidence based on number and type of signals
        confidence = self._calculate_confidence(signals)
        detected = confidence >= 0.3

        return DetectionResult(
            detected=detected,
            framework=TestFramework.PYTEST,
            confidence=confidence,
            signals=signals,
            config_path=config_path,
            version=version,
            details="; ".join(details_parts) if details_parts else "No pytest signals found",
        )

    def _check_config_files(self, root: Path) -> str:
        """Check for pytest configuration files. Returns path if found."""
        # Direct pytest config files
        for name in ("pytest.ini", "tox.ini"):
            path = root / name
            if path.is_file():
                return str(path)

        # setup.cfg with [tool:pytest] section
        setup_cfg = root / "setup.cfg"
        if setup_cfg.is_file():
            try:
                content = setup_cfg.read_text(encoding="utf-8", errors="replace")
                if "[tool:pytest]" in content:
                    return str(setup_cfg)
            except OSError:
                pass

        # pyproject.toml with [tool.pytest] section
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")
                if "[tool.pytest" in content:
                    return str(pyproject)
            except OSError:
                pass

        return ""

    def _check_dependency_files(self, root: Path) -> bool:
        """Check if pytest appears in dependency/requirements files."""
        # requirements*.txt
        for req_file in root.glob("requirements*.txt"):
            try:
                content = req_file.read_text(encoding="utf-8", errors="replace").lower()
                if "pytest" in content:
                    return True
            except OSError:
                continue

        # pyproject.toml dependencies
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace").lower()
                if "pytest" in content:
                    return True
            except OSError:
                pass

        # Pipfile
        pipfile = root / "Pipfile"
        if pipfile.is_file():
            try:
                content = pipfile.read_text(encoding="utf-8", errors="replace").lower()
                if "pytest" in content:
                    return True
            except OSError:
                pass

        return False

    def _find_test_files(self, root: Path, max_depth: int = 4) -> list[str]:
        """Find test_*.py or *_test.py files, excluding common virtual env dirs."""
        exclude_dirs = {".venv", "venv", "node_modules", "__pycache__", ".tox", ".eggs", ".git"}
        test_files: list[str] = []

        def _scan(directory: Path, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                for entry in directory.iterdir():
                    if entry.is_dir():
                        if entry.name not in exclude_dirs:
                            _scan(entry, depth + 1)
                    elif entry.is_file() and entry.suffix == ".py":
                        if entry.name.startswith("test_") or entry.name.endswith("_test.py"):
                            test_files.append(str(entry.relative_to(root)))
            except PermissionError:
                pass

        _scan(root, 0)
        return test_files[:50]  # Cap to avoid huge lists

    def _check_lockfiles(self, root: Path) -> bool:
        """Check lockfiles for pytest references."""
        for lockfile_name in ("uv.lock", "poetry.lock", "Pipfile.lock"):
            lockfile = root / lockfile_name
            if lockfile.is_file():
                try:
                    content = lockfile.read_text(encoding="utf-8", errors="replace").lower()
                    if "pytest" in content:
                        return True
                except OSError:
                    pass
        return False

    @staticmethod
    async def _check_cli(project_root: str) -> str:
        """Check if pytest CLI is available and return its version."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pytest", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = (stdout or stderr).decode("utf-8", errors="replace").strip()
            # Parse version from output like "pytest 8.1.1"
            match = re.search(r"pytest\s+([\d.]+)", output)
            if match:
                return match.group(1)
            return output[:30] if output else ""
        except (FileNotFoundError, asyncio.TimeoutError, OSError):
            return ""

    @staticmethod
    def _calculate_confidence(signals: list[DetectionSignal]) -> float:
        """Calculate detection confidence based on collected signals.

        Signal weights:
        - CONFIG_FILE: 0.35 (strong — explicit pytest config)
        - CLI_AVAILABLE: 0.20 (moderate — pytest installed)
        - DEPENDENCY_FILE: 0.15
        - TEST_FILE_PATTERN: 0.10
        - IMPORT_STATEMENT (conftest): 0.30 (strong pytest-specific signal)
        - LOCKFILE: 0.10
        """
        weights: dict[DetectionSignal, float] = {
            DetectionSignal.CONFIG_FILE: 0.35,
            DetectionSignal.CLI_AVAILABLE: 0.20,
            DetectionSignal.DEPENDENCY_FILE: 0.15,
            DetectionSignal.TEST_FILE_PATTERN: 0.10,
            DetectionSignal.IMPORT_STATEMENT: 0.30,
            DetectionSignal.LOCKFILE: 0.10,
        }
        total = sum(weights.get(s, 0.05) for s in signals)
        return min(total, 1.0)

    # -- Command building ----------------------------------------------------

    def build_command(self, request: ParsedTestRequest) -> list[str]:
        """Build pytest CLI command from a parsed test request.

        Dispatches to intent-specific builders for proper flag handling.
        """
        match request.intent:
            case TestIntent.LIST:
                return self._build_list(request)
            case TestIntent.RERUN_FAILED:
                return self._build_rerun_failed(request)
            case TestIntent.RUN_SPECIFIC:
                return self._build_run_specific(request)
            case TestIntent.RUN | _:
                return self._build_run(request)

    def _build_run(self, req: ParsedTestRequest) -> list[str]:
        """Build a standard pytest run command."""
        cmd = ["pytest", "-v"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def _build_list(self, req: ParsedTestRequest) -> list[str]:
        """Build a pytest collect-only (list/discover) command."""
        cmd = ["pytest", "--collect-only", "-q"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def _build_rerun_failed(self, req: ParsedTestRequest) -> list[str]:
        """Build a pytest rerun-failed command."""
        cmd = ["pytest", "--lf", "-v"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def _build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        """Build a pytest run-specific command (node ID or -k pattern)."""
        cmd = ["pytest", "-v"]
        if req.scope:
            # If scope contains "::" it's a node ID, pass directly
            # Otherwise treat as a -k pattern
            if "::" in req.scope:
                cmd.append(req.scope)
            else:
                cmd.extend(["-k", req.scope])
        cmd.extend(req.extra_args)
        return cmd

    # -- Output parsing ------------------------------------------------------

    def parse_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> ParsedTestOutput:
        """Parse pytest stdout/stderr into structured test results.

        Handles both standard and verbose pytest output formats, extracting:
        - Per-test results (from verbose mode lines)
        - Summary counts (from the summary line)
        - Failure details (from FAILURES section and short test summary)
        - Total duration
        """
        result = ParsedTestOutput(
            raw_stdout=stdout,
            raw_stderr=stderr,
            exit_code=exit_code,
        )

        combined = stdout + "\n" + stderr

        # 1. Parse summary line for aggregate counts and duration
        self._parse_summary_line(combined, result)

        # 2. Parse verbose result lines for per-test outcomes
        self._parse_verbose_results(stdout, result)

        # 3. Parse failure details from FAILURES section
        self._parse_failure_sections(stdout, result)

        # 4. Enrich failures with short summary info
        self._parse_short_summary(stdout, result)

        # 5. Parse durations if available
        self._parse_durations(stdout, result)

        # 6. Reconcile: if we got summary counts but no per-test results,
        #    use the summary counts; if we have per-test results, recount
        self._reconcile_counts(result)

        return result

    def _parse_summary_line(self, text: str, result: ParsedTestOutput) -> None:
        """Extract counts and duration from the pytest summary line."""
        match = _SUMMARY_RE.search(text)
        if not match:
            return

        result.summary_line = match.group(0).strip()

        # Parse duration
        try:
            result.duration_seconds = float(match.group("duration"))
        except (ValueError, IndexError):
            pass

        # Parse individual counts from the summary
        summary_text = match.group("summary")
        for count_match in _COUNT_RE.finditer(summary_text):
            count = int(count_match.group("count"))
            label = count_match.group("label").rstrip("s")  # normalize plurals

            if label == "passed":
                result.passed = count
            elif label == "failed":
                result.failed = count
            elif label == "error":
                result.errors = count
            elif label == "skipped":
                result.skipped = count
            elif label == "warning":
                pass  # warnings count, not test count
            elif label == "deselected":
                pass  # track in metadata
            elif label == "xfailed":
                result.metadata["xfailed"] = count
            elif label == "xpassed":
                result.metadata["xpassed"] = count

        result.total = result.passed + result.failed + result.errors + result.skipped

    def _parse_verbose_results(self, stdout: str, result: ParsedTestOutput) -> None:
        """Parse per-test result lines from verbose mode output."""
        outcome_map = {
            "PASSED": TestOutcome.PASSED,
            "FAILED": TestOutcome.FAILED,
            "ERROR": TestOutcome.ERROR,
            "SKIPPED": TestOutcome.SKIPPED,
            "XFAIL": TestOutcome.XFAIL,
            "XPASS": TestOutcome.XPASS,
        }

        for match in _VERBOSE_RESULT_RE.finditer(stdout):
            nodeid = match.group("nodeid")
            outcome_str = match.group("outcome")
            outcome = outcome_map.get(outcome_str, TestOutcome.PASSED)

            # Extract file path from nodeid (everything before first "::")
            file_path = nodeid.split("::")[0] if "::" in nodeid else ""

            tc = TestCaseResult(
                name=nodeid,
                outcome=outcome,
                file_path=file_path,
            )
            result.test_cases.append(tc)

    def _parse_failure_sections(self, stdout: str, result: ParsedTestOutput) -> None:
        """Parse the FAILURES section for detailed tracebacks."""
        # Find all failure headers and their content
        headers = list(_FAILURE_HEADER_RE.finditer(stdout))
        if not headers:
            return

        for i, header_match in enumerate(headers):
            test_name = header_match.group("name").strip()
            # Content extends from after this header to the next header or FAILURES end
            start = header_match.end()
            if i + 1 < len(headers):
                end = headers[i + 1].start()
            else:
                # Find the end of FAILURES section (next "====" section)
                section_end = stdout.find("\n=", start)
                end = section_end if section_end != -1 else len(stdout)

            traceback_text = stdout[start:end].strip()

            # Try to match this to an existing test case result
            matched = False
            for tc in result.test_cases:
                if test_name in tc.name or tc.name.endswith(test_name):
                    tc.traceback = traceback_text
                    matched = True
                    break

            # If no matching test case, create one
            if not matched:
                tc = TestCaseResult(
                    name=test_name,
                    outcome=TestOutcome.FAILED,
                    traceback=traceback_text,
                )
                result.test_cases.append(tc)

    def _parse_short_summary(self, stdout: str, result: ParsedTestOutput) -> None:
        """Parse the short test summary info section for failure messages."""
        # Extract FAILED lines
        for match in _SHORT_SUMMARY_FAILED_RE.finditer(stdout):
            nodeid = match.group("nodeid")
            message = match.group("message") or ""

            # Try to enrich existing test case
            for tc in result.test_cases:
                if tc.name == nodeid or nodeid in tc.name:
                    if not tc.message:
                        tc.message = message
                    break
            else:
                # No existing test case found — create one
                tc = TestCaseResult(
                    name=nodeid,
                    outcome=TestOutcome.FAILED,
                    message=message,
                )
                result.test_cases.append(tc)

        # Extract ERROR lines
        for match in _SHORT_SUMMARY_ERROR_RE.finditer(stdout):
            nodeid = match.group("nodeid")
            message = match.group("message") or ""

            for tc in result.test_cases:
                if tc.name == nodeid or nodeid in tc.name:
                    if not tc.message:
                        tc.message = message
                    break
            else:
                tc = TestCaseResult(
                    name=nodeid,
                    outcome=TestOutcome.ERROR,
                    message=message,
                )
                result.test_cases.append(tc)

    def _parse_durations(self, stdout: str, result: ParsedTestOutput) -> None:
        """Parse per-test durations from verbose output."""
        for match in _DURATION_RE.finditer(stdout):
            nodeid = match.group("nodeid")
            try:
                duration = float(match.group("duration"))
            except ValueError:
                continue

            for tc in result.test_cases:
                if tc.name == nodeid:
                    tc.duration_seconds = duration
                    break

    def _reconcile_counts(self, result: ParsedTestOutput) -> None:
        """Reconcile summary counts with per-test results.

        If we have per-test results, recount from them (they're more reliable).
        If we only have summary counts, use those.
        """
        if result.test_cases:
            # Recount from actual test cases
            result.passed = sum(
                1 for tc in result.test_cases if tc.outcome == TestOutcome.PASSED
            )
            result.failed = sum(
                1 for tc in result.test_cases if tc.outcome == TestOutcome.FAILED
            )
            result.errors = sum(
                1 for tc in result.test_cases if tc.outcome == TestOutcome.ERROR
            )
            result.skipped = sum(
                1 for tc in result.test_cases if tc.outcome == TestOutcome.SKIPPED
            )
            result.total = len(result.test_cases)
        # If no test cases but we have summary counts, total was set in _parse_summary_line

    # -- Collect-only parsing ------------------------------------------------

    def parse_collect_output(self, stdout: str) -> list[str]:
        """Parse pytest --collect-only -q output to extract test node IDs.

        Args:
            stdout: Output from ``pytest --collect-only -q``.

        Returns:
            List of test node IDs (e.g. ["tests/test_foo.py::test_bar"]).
        """
        nodeids: list[str] = []
        for match in _COLLECT_Q_RE.finditer(stdout):
            nodeid = match.group("nodeid").strip()
            if nodeid:
                nodeids.append(nodeid)
        return nodeids
