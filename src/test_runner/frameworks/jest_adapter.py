"""Jest framework adapter: detection, command building, and output parsing.

Provides full lifecycle support for Jest-based JavaScript/TypeScript projects:
- Detects Jest via config files, dependency files, test file patterns, and CLI.
- Builds correct ``jest`` / ``npx jest`` CLI commands for various intents.
- Parses Jest output (both standard and verbose/JSON) into structured results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

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
# Regex patterns for Jest output parsing
# ---------------------------------------------------------------------------

# Summary line: "Tests:  3 passed, 1 failed, 4 total"
_TESTS_SUMMARY_RE = re.compile(
    r"Tests:\s+(?P<summary>.+?total)",
)

# Individual count tokens: "3 passed", "1 failed", "2 skipped", etc.
_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<label>passed|failed|skipped|pending|todo|total)",
)

# Suite summary: "Test Suites:  2 passed, 1 failed, 3 total"
_SUITES_SUMMARY_RE = re.compile(
    r"Test Suites:\s+(?P<summary>.+?total)",
)

# Time line: "Time:        1.234 s" or "Time:        1234 ms"
_TIME_RE = re.compile(
    r"Time:\s+(?P<duration>[\d.]+)\s*(?P<unit>m?s)\b",
)

# Verbose test result lines:
#   ✓ should do something (5 ms)
#   ✕ should fail (10 ms)
#   ○ skipped should skip
#   ✓ test name (5ms)
_VERBOSE_PASS_RE = re.compile(
    r"^\s*[✓✔√⦁]\s+(?P<name>.+?)(?:\s+\((?P<duration>\d+)\s*m?s\))?\s*$",
    re.MULTILINE,
)
_VERBOSE_FAIL_RE = re.compile(
    r"^\s*[✕✗×⦿]\s+(?P<name>.+?)(?:\s+\((?P<duration>\d+)\s*m?s\))?\s*$",
    re.MULTILINE,
)
_VERBOSE_SKIP_RE = re.compile(
    r"^\s*[○◌]\s+(?:skipped\s+)?(?P<name>.+?)\s*$",
    re.MULTILINE,
)
_VERBOSE_PENDING_RE = re.compile(
    r"^\s*[○◌]\s+(?:pending\s+)?(?P<name>.+?)\s*$",
    re.MULTILINE,
)

# PASS/FAIL suite header: "PASS src/__tests__/foo.test.js" or "FAIL src/..."
_SUITE_HEADER_RE = re.compile(
    r"^\s*(?P<outcome>PASS|FAIL)\s+(?P<file>\S+)\s*$",
    re.MULTILINE,
)

# Describe block header (verbose): "  DescribeName"
_DESCRIBE_RE = re.compile(
    r"^\s{2,4}(?P<name>[A-Z]\S.*?)\s*$",
    re.MULTILINE,
)

# Failure block: "● DescribeName › test name" or "● test name"
_FAILURE_HEADER_RE = re.compile(
    r"^\s*●\s+(?P<name>.+?)\s*$",
    re.MULTILINE,
)

# Jest --json "testResults" entries have structured data
# (handled separately via JSON parsing)

# Jest --listTests output: one file path per line
_LIST_TEST_RE = re.compile(
    r"^(?P<path>\S+\.(?:test|spec)\.(?:js|jsx|ts|tsx|mjs|cjs))\s*$",
    re.MULTILINE,
)

# ANSI escape code stripper
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI color/style codes from text."""
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Jest adapter
# ---------------------------------------------------------------------------


class JestAdapter(FrameworkAdapter):
    """Full-lifecycle adapter for the Jest test framework.

    Detection:
        Checks for Jest config files (jest.config.js/ts/mjs, package.json jest
        section), dependency references (package.json, yarn.lock, package-lock.json),
        test file patterns (*.test.js, *.spec.ts, etc.), and CLI availability.

    Command building:
        Constructs ``npx jest`` CLI commands for run, list, rerun-failed, and
        run-specific intents with proper flags.

    Output parsing:
        Parses both standard and verbose Jest output to extract per-test
        results, summary counts, duration, and failure details. Also supports
        parsing ``--json`` structured output.
    """

    # -- FrameworkAdapter interface ------------------------------------------

    @property
    def framework(self) -> TestFramework:
        return TestFramework.JEST

    @property
    def display_name(self) -> str:
        return "Jest"

    # -- Detection -----------------------------------------------------------

    async def detect(self, project_root: str) -> DetectionResult:
        """Detect Jest presence by checking multiple signals."""
        root = Path(project_root)
        signals: list[DetectionSignal] = []
        details_parts: list[str] = []
        config_path = ""
        version = ""

        # 1. Check for Jest config files
        config_path_found = self._check_config_files(root)
        if config_path_found:
            signals.append(DetectionSignal.CONFIG_FILE)
            config_path = config_path_found
            details_parts.append(f"Config found: {config_path_found}")

        # 2. Check dependency files (package.json)
        if self._check_dependency_files(root):
            signals.append(DetectionSignal.DEPENDENCY_FILE)
            details_parts.append("jest listed in dependencies")

        # 3. Check for test file patterns
        test_files = self._find_test_files(root)
        if test_files:
            signals.append(DetectionSignal.TEST_FILE_PATTERN)
            details_parts.append(f"Found {len(test_files)} test file(s)")

        # 4. Check for Jest-specific imports in test files
        if self._check_jest_imports(root, test_files):
            signals.append(DetectionSignal.IMPORT_STATEMENT)
            details_parts.append("Jest imports found in test files")

        # 5. Check CLI availability
        cli_version = await self._check_cli(project_root)
        if cli_version:
            signals.append(DetectionSignal.CLI_AVAILABLE)
            version = cli_version
            details_parts.append(f"CLI available: jest {version}")

        # 6. Check lockfiles
        if self._check_lockfiles(root):
            signals.append(DetectionSignal.LOCKFILE)
            details_parts.append("jest found in lockfile")

        # Calculate confidence based on number and type of signals
        confidence = self._calculate_confidence(signals)
        detected = confidence >= 0.3

        return DetectionResult(
            detected=detected,
            framework=TestFramework.JEST,
            confidence=confidence,
            signals=signals,
            config_path=config_path,
            version=version,
            details="; ".join(details_parts) if details_parts else "No Jest signals found",
        )

    def _check_config_files(self, root: Path) -> str:
        """Check for Jest configuration files. Returns path if found."""
        # Standalone Jest config files
        for name in (
            "jest.config.js",
            "jest.config.ts",
            "jest.config.mjs",
            "jest.config.cjs",
            "jest.config.json",
        ):
            path = root / name
            if path.is_file():
                return str(path)

        # package.json with "jest" key
        pkg_json = root / "package.json"
        if pkg_json.is_file():
            try:
                content = pkg_json.read_text(encoding="utf-8", errors="replace")
                data = json.loads(content)
                if "jest" in data:
                    return str(pkg_json)
            except (json.JSONDecodeError, OSError):
                pass

        return ""

    def _check_dependency_files(self, root: Path) -> bool:
        """Check if jest appears in package.json dependencies."""
        pkg_json = root / "package.json"
        if pkg_json.is_file():
            try:
                content = pkg_json.read_text(encoding="utf-8", errors="replace")
                data = json.loads(content)
                # Check devDependencies, dependencies, and scripts
                for section in ("devDependencies", "dependencies", "peerDependencies"):
                    deps = data.get(section, {})
                    if isinstance(deps, dict) and "jest" in deps:
                        return True
                # Check if jest is referenced in scripts
                scripts = data.get("scripts", {})
                if isinstance(scripts, dict):
                    for script_val in scripts.values():
                        if isinstance(script_val, str) and "jest" in script_val:
                            return True
            except (json.JSONDecodeError, OSError):
                pass
        return False

    def _find_test_files(
        self, root: Path, max_depth: int = 4,
    ) -> list[str]:
        """Find Jest test files (*.test.js, *.spec.ts, etc.)."""
        exclude_dirs = {
            "node_modules", ".git", "dist", "build", "coverage",
            ".next", ".nuxt", ".venv", "venv", "__pycache__",
        }
        test_extensions = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
        test_patterns = (".test.", ".spec.")
        test_dirs = {"__tests__"}
        test_files: list[str] = []

        def _scan(directory: Path, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                for entry in directory.iterdir():
                    if entry.is_dir():
                        if entry.name not in exclude_dirs:
                            # Files inside __tests__ directories are test files
                            if entry.name in test_dirs:
                                _scan_all_in_test_dir(entry, depth + 1)
                            else:
                                _scan(entry, depth + 1)
                    elif entry.is_file() and entry.suffix in test_extensions:
                        stem = entry.name
                        if any(pat in stem for pat in test_patterns):
                            test_files.append(str(entry.relative_to(root)))
            except PermissionError:
                pass

        def _scan_all_in_test_dir(directory: Path, depth: int) -> None:
            """All JS/TS files in __tests__ dirs count as test files."""
            if depth > max_depth:
                return
            try:
                for entry in directory.iterdir():
                    if entry.is_dir() and entry.name not in exclude_dirs:
                        _scan_all_in_test_dir(entry, depth + 1)
                    elif entry.is_file() and entry.suffix in test_extensions:
                        test_files.append(str(entry.relative_to(root)))
            except PermissionError:
                pass

        _scan(root, 0)
        return test_files[:50]  # Cap to avoid huge lists

    def _check_jest_imports(self, root: Path, test_files: list[str]) -> bool:
        """Check if test files contain Jest-specific imports or globals."""
        jest_markers = (
            "from '@jest",
            'from "@jest',
            "require('jest",
            'require("jest',
            "jest.mock(",
            "jest.fn(",
            "jest.spyOn(",
            "describe(",
            "it(",
            "expect(",
        )
        # Check up to 5 test files to avoid scanning too many
        for rel_path in test_files[:5]:
            full_path = root / rel_path
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                if any(marker in content for marker in jest_markers):
                    return True
            except OSError:
                continue
        return False

    def _check_lockfiles(self, root: Path) -> bool:
        """Check lockfiles for jest references."""
        for lockfile_name in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml"):
            lockfile = root / lockfile_name
            if lockfile.is_file():
                try:
                    content = lockfile.read_text(encoding="utf-8", errors="replace")
                    if "jest" in content.lower():
                        return True
                except OSError:
                    pass
        return False

    @staticmethod
    async def _check_cli(project_root: str) -> str:
        """Check if jest CLI is available and return its version."""
        # Try npx jest first, then global jest
        for cmd in (["npx", "jest", "--version"], ["jest", "--version"]):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=project_root,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=15,
                )
                output = (stdout or b"").decode("utf-8", errors="replace").strip()
                # Jest --version outputs just the version number, e.g. "29.7.0"
                match = re.search(r"(\d+\.\d+\.\d+)", output)
                if match:
                    return match.group(1)
            except (FileNotFoundError, asyncio.TimeoutError, OSError):
                continue
        return ""

    @staticmethod
    def _calculate_confidence(signals: list[DetectionSignal]) -> float:
        """Calculate detection confidence based on collected signals.

        Signal weights:
        - CONFIG_FILE: 0.35 (strong — explicit jest config)
        - CLI_AVAILABLE: 0.20 (moderate — jest installed)
        - DEPENDENCY_FILE: 0.20 (moderate — jest in package.json)
        - TEST_FILE_PATTERN: 0.10
        - IMPORT_STATEMENT: 0.15 (jest imports/globals)
        - LOCKFILE: 0.10
        """
        weights: dict[DetectionSignal, float] = {
            DetectionSignal.CONFIG_FILE: 0.35,
            DetectionSignal.CLI_AVAILABLE: 0.20,
            DetectionSignal.DEPENDENCY_FILE: 0.20,
            DetectionSignal.TEST_FILE_PATTERN: 0.10,
            DetectionSignal.IMPORT_STATEMENT: 0.15,
            DetectionSignal.LOCKFILE: 0.10,
        }
        total = sum(weights.get(s, 0.05) for s in signals)
        return min(total, 1.0)

    # -- Command building ----------------------------------------------------

    def build_command(self, request: ParsedTestRequest) -> list[str]:
        """Build Jest CLI command from a parsed test request.

        Uses ``npx jest`` as the base command to ensure the project-local
        version is used. Dispatches to intent-specific builders.
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
        """Build a standard Jest run command."""
        cmd = ["npx", "jest", "--verbose"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def _build_list(self, req: ParsedTestRequest) -> list[str]:
        """Build a Jest list/discover command."""
        cmd = ["npx", "jest", "--listTests"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def _build_rerun_failed(self, req: ParsedTestRequest) -> list[str]:
        """Build a Jest rerun-failed command."""
        cmd = ["npx", "jest", "--onlyFailures", "--verbose"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def _build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        """Build a Jest run-specific command (file or test name pattern)."""
        cmd = ["npx", "jest", "--verbose"]
        if req.scope:
            # If scope looks like a file path, pass directly
            # Otherwise use -t for test name pattern matching
            if any(
                req.scope.endswith(ext)
                for ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
            ) or "/" in req.scope:
                cmd.append(req.scope)
            else:
                cmd.extend(["-t", req.scope])
        cmd.extend(req.extra_args)
        return cmd

    # -- Output parsing ------------------------------------------------------

    def parse_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> ParsedTestOutput:
        """Parse Jest stdout/stderr into structured test results.

        Handles both standard and verbose Jest output formats. Also supports
        Jest's ``--json`` output format for precise structured results.
        """
        result = ParsedTestOutput(
            raw_stdout=stdout,
            raw_stderr=stderr,
            exit_code=exit_code,
        )

        # Jest often writes results to stderr
        combined = _strip_ansi(stdout + "\n" + stderr)

        # 1. Try JSON parsing first (if --json flag was used)
        if self._try_parse_json(combined, result):
            return result

        # 2. Parse summary line for aggregate counts and duration
        self._parse_summary_line(combined, result)

        # 3. Parse time
        self._parse_time(combined, result)

        # 4. Parse verbose result lines for per-test outcomes
        self._parse_verbose_results(combined, result)

        # 5. Parse failure details
        self._parse_failure_sections(combined, result)

        # 6. Parse suite-level PASS/FAIL headers for file info
        self._parse_suite_headers(combined, result)

        # 7. Reconcile counts
        self._reconcile_counts(result)

        return result

    def _try_parse_json(self, text: str, result: ParsedTestOutput) -> bool:
        """Attempt to parse Jest --json output. Returns True if successful."""
        # Look for JSON object in output
        json_start = text.find('{"numFailedTestSuites"')
        if json_start == -1:
            json_start = text.find('{"numFailedTests"')
        if json_start == -1:
            return False

        # Find matching closing brace
        brace_count = 0
        json_end = json_start
        for i in range(json_start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        try:
            data = json.loads(text[json_start:json_end])
        except (json.JSONDecodeError, ValueError):
            return False

        # Extract aggregate stats
        result.passed = data.get("numPassedTests", 0)
        result.failed = data.get("numFailedTests", 0)
        result.skipped = data.get("numPendingTests", 0)
        result.total = data.get("numTotalTests", 0)

        # Extract per-test results from testResults
        for suite in data.get("testResults", []):
            suite_file = suite.get("name", "")
            for assertion in suite.get("assertionResults", []):
                outcome = self._map_json_status(assertion.get("status", ""))
                ancestors = assertion.get("ancestorTitles", [])
                title = assertion.get("title", "")
                full_name = " › ".join(ancestors + [title]) if ancestors else title

                duration_ms = assertion.get("duration", 0) or 0

                tc = TestCaseResult(
                    name=full_name,
                    outcome=outcome,
                    duration_seconds=duration_ms / 1000.0,
                    file_path=suite_file,
                    message="\n".join(assertion.get("failureMessages", [])),
                )
                result.test_cases.append(tc)

        # Duration
        if "startTime" in data and data.get("testResults"):
            # Approximate from start/end of last suite
            pass

        result.metadata["json_parsed"] = True
        result.metadata["num_suites_passed"] = data.get("numPassedTestSuites", 0)
        result.metadata["num_suites_failed"] = data.get("numFailedTestSuites", 0)
        result.metadata["num_suites_total"] = data.get("numTotalTestSuites", 0)

        return True

    @staticmethod
    def _map_json_status(status: str) -> TestOutcome:
        """Map Jest JSON assertion status to TestOutcome."""
        mapping = {
            "passed": TestOutcome.PASSED,
            "failed": TestOutcome.FAILED,
            "pending": TestOutcome.SKIPPED,
            "skipped": TestOutcome.SKIPPED,
            "todo": TestOutcome.SKIPPED,
            "disabled": TestOutcome.SKIPPED,
        }
        return mapping.get(status, TestOutcome.ERROR)

    def _parse_summary_line(self, text: str, result: ParsedTestOutput) -> None:
        """Extract counts from the Jest summary line."""
        match = _TESTS_SUMMARY_RE.search(text)
        if not match:
            return

        result.summary_line = match.group(0).strip()
        summary_text = match.group("summary")

        for count_match in _COUNT_RE.finditer(summary_text):
            count = int(count_match.group("count"))
            label = count_match.group("label")

            if label == "passed":
                result.passed = count
            elif label == "failed":
                result.failed = count
            elif label in ("skipped", "pending", "todo"):
                result.skipped += count
            elif label == "total":
                result.total = count

    def _parse_time(self, text: str, result: ParsedTestOutput) -> None:
        """Extract total duration from the Time line."""
        match = _TIME_RE.search(text)
        if not match:
            return
        try:
            duration = float(match.group("duration"))
            unit = match.group("unit")
            if unit == "ms":
                duration /= 1000.0
            result.duration_seconds = duration
        except (ValueError, IndexError):
            pass

    def _parse_verbose_results(self, text: str, result: ParsedTestOutput) -> None:
        """Parse per-test result lines from verbose mode output."""
        # Track current file context from PASS/FAIL headers
        current_file = ""
        current_describe = ""

        # Process line by line for context tracking
        lines = text.split("\n")
        for line in lines:
            stripped = line.rstrip()

            # Check for suite header
            suite_match = _SUITE_HEADER_RE.match(stripped)
            if suite_match:
                current_file = suite_match.group("file")
                current_describe = ""
                continue

            # Check for describe block
            # Simple heuristic: indented non-result line that looks like a name
            if re.match(r"^\s{2,4}[A-Z]", stripped) and not any(
                c in stripped for c in ("✓", "✕", "✗", "×", "✔", "√", "○", "●")
            ):
                current_describe = stripped.strip()
                continue

        # Now parse individual test results with regex
        for match in _VERBOSE_PASS_RE.finditer(text):
            name = match.group("name").strip()
            duration_ms = int(match.group("duration")) if match.group("duration") else 0
            tc = TestCaseResult(
                name=name,
                outcome=TestOutcome.PASSED,
                duration_seconds=duration_ms / 1000.0,
                file_path=current_file,
            )
            result.test_cases.append(tc)

        for match in _VERBOSE_FAIL_RE.finditer(text):
            name = match.group("name").strip()
            duration_ms = int(match.group("duration")) if match.group("duration") else 0
            tc = TestCaseResult(
                name=name,
                outcome=TestOutcome.FAILED,
                duration_seconds=duration_ms / 1000.0,
                file_path=current_file,
            )
            result.test_cases.append(tc)

        for match in _VERBOSE_SKIP_RE.finditer(text):
            name = match.group("name").strip()
            tc = TestCaseResult(
                name=name,
                outcome=TestOutcome.SKIPPED,
                file_path=current_file,
            )
            result.test_cases.append(tc)

    def _parse_failure_sections(self, text: str, result: ParsedTestOutput) -> None:
        """Parse failure detail blocks marked with ● in Jest output."""
        headers = list(_FAILURE_HEADER_RE.finditer(text))
        if not headers:
            return

        for i, header_match in enumerate(headers):
            test_name = header_match.group("name").strip()
            start = header_match.end()

            # Content extends to the next ● header or end of relevant section
            if i + 1 < len(headers):
                end = headers[i + 1].start()
            else:
                # Look for end markers
                section_end = text.find("\nTest Suites:", start)
                if section_end == -1:
                    section_end = text.find("\nTests:", start)
                end = section_end if section_end != -1 else len(text)

            traceback_text = text[start:end].strip()

            # Try to match to existing test case
            matched = False
            for tc in result.test_cases:
                if test_name in tc.name or tc.name in test_name:
                    tc.traceback = traceback_text
                    if not tc.message:
                        # Extract first meaningful line as message
                        for line in traceback_text.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("at "):
                                tc.message = line
                                break
                    matched = True
                    break

            if not matched:
                # Create a new test case for this failure
                message = ""
                for line in traceback_text.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("at "):
                        message = line
                        break

                tc = TestCaseResult(
                    name=test_name,
                    outcome=TestOutcome.FAILED,
                    traceback=traceback_text,
                    message=message,
                )
                result.test_cases.append(tc)

    def _parse_suite_headers(self, text: str, result: ParsedTestOutput) -> None:
        """Parse PASS/FAIL suite headers to enrich file path metadata."""
        suite_outcomes: dict[str, str] = {}
        for match in _SUITE_HEADER_RE.finditer(text):
            suite_outcomes[match.group("file")] = match.group("outcome")

        result.metadata["suite_results"] = suite_outcomes

    def _reconcile_counts(self, result: ParsedTestOutput) -> None:
        """Reconcile summary counts with per-test results.

        If we have per-test results, recount from them.
        If we only have summary counts, use those.
        """
        if result.test_cases:
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

    # -- List output parsing -------------------------------------------------

    def parse_list_output(self, stdout: str) -> list[str]:
        """Parse jest --listTests output to extract test file paths.

        Args:
            stdout: Output from ``npx jest --listTests``.

        Returns:
            List of test file paths.
        """
        clean = _strip_ansi(stdout)
        paths: list[str] = []
        for line in clean.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("{"):
                # Accept lines that look like file paths
                if any(
                    ext in line
                    for ext in (".test.", ".spec.", "__tests__")
                ):
                    paths.append(line)
                elif line.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
                    paths.append(line)
        return paths
