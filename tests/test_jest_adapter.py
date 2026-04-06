"""Tests for the Jest framework adapter.

Covers detection, command building, and output parsing.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from test_runner.agents.parser import ParsedTestRequest, TestFramework, TestIntent
from test_runner.frameworks.base import (
    DetectionSignal,
    TestOutcome,
)
from test_runner.frameworks.jest_adapter import JestAdapter, _strip_ansi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter():
    return JestAdapter()


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal Jest project directory."""
    # package.json with jest dependency
    (tmp_path / "package.json").write_text(
        json.dumps({
            "name": "test-project",
            "devDependencies": {"jest": "^29.0.0"},
            "scripts": {"test": "jest"},
        })
    )
    # jest.config.js
    (tmp_path / "jest.config.js").write_text(
        "module.exports = { testEnvironment: 'node' };\n"
    )
    # A test file
    src_dir = tmp_path / "src" / "__tests__"
    src_dir.mkdir(parents=True)
    (src_dir / "example.test.js").write_text(
        textwrap.dedent("""\
        describe('Example', () => {
          it('should work', () => {
            expect(true).toBe(true);
          });
        });
        """)
    )
    return tmp_path


def _make_request(
    intent: TestIntent = TestIntent.RUN,
    framework: TestFramework = TestFramework.JEST,
    scope: str = "",
    extra_args: list[str] | None = None,
    working_directory: str = "",
) -> ParsedTestRequest:
    return ParsedTestRequest(
        intent=intent,
        framework=framework,
        scope=scope,
        extra_args=extra_args or [],
        working_directory=working_directory,
        raw_request="test request",
    )


# ===========================================================================
# Detection tests
# ===========================================================================


class TestJestDetection:
    """Tests for Jest detection logic."""

    @pytest.mark.asyncio
    async def test_detect_with_jest_config_js(self, adapter, tmp_path):
        (tmp_path / "jest.config.js").write_text("module.exports = {};\n")
        result = await adapter.detect(str(tmp_path))
        assert result.detected is True
        assert result.framework == TestFramework.JEST
        assert DetectionSignal.CONFIG_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_jest_config_ts(self, adapter, tmp_path):
        (tmp_path / "jest.config.ts").write_text("export default {};\n")
        result = await adapter.detect(str(tmp_path))
        assert result.detected is True
        assert DetectionSignal.CONFIG_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_jest_config_json(self, adapter, tmp_path):
        (tmp_path / "jest.config.json").write_text("{}\n")
        result = await adapter.detect(str(tmp_path))
        assert result.detected is True
        assert DetectionSignal.CONFIG_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_jest_config_mjs(self, adapter, tmp_path):
        (tmp_path / "jest.config.mjs").write_text("export default {};\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.CONFIG_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_jest_config_cjs(self, adapter, tmp_path):
        (tmp_path / "jest.config.cjs").write_text("module.exports = {};\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.CONFIG_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_package_json_jest_key(self, adapter, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"name": "test", "jest": {"testEnvironment": "node"}})
        )
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.CONFIG_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_dev_dependency(self, adapter, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"devDependencies": {"jest": "^29.0.0"}})
        )
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.DEPENDENCY_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_jest_in_scripts(self, adapter, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest --coverage"}})
        )
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.DEPENDENCY_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_test_files(self, adapter, tmp_path):
        (tmp_path / "foo.test.js").write_text("test('x', () => {});\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.TEST_FILE_PATTERN in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_spec_files(self, adapter, tmp_path):
        (tmp_path / "bar.spec.ts").write_text("describe('x', () => {});\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.TEST_FILE_PATTERN in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_tests_directory(self, adapter, tmp_path):
        tests_dir = tmp_path / "__tests__"
        tests_dir.mkdir()
        (tests_dir / "foo.js").write_text("test('x', () => {});\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.TEST_FILE_PATTERN in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_jest_imports(self, adapter, tmp_path):
        (tmp_path / "foo.test.js").write_text(
            "jest.mock('./module');\ntest('x', () => {});\n"
        )
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.IMPORT_STATEMENT in result.signals

    @pytest.mark.asyncio
    async def test_detect_empty_directory(self, adapter, tmp_path):
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.CONFIG_FILE not in result.signals
        assert DetectionSignal.DEPENDENCY_FILE not in result.signals
        assert DetectionSignal.TEST_FILE_PATTERN not in result.signals

    @pytest.mark.asyncio
    async def test_detect_excludes_node_modules(self, adapter, tmp_path):
        nm = tmp_path / "node_modules" / "some-pkg"
        nm.mkdir(parents=True)
        (nm / "foo.test.js").write_text("test('x', () => {});\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.TEST_FILE_PATTERN not in result.signals

    @pytest.mark.asyncio
    async def test_detect_full_project(self, adapter, project_dir):
        """Full project with config, deps, test files — high confidence."""
        result = await adapter.detect(str(project_dir))
        assert result.detected is True
        assert result.confidence >= 0.5
        assert len(result.signals) >= 2

    @pytest.mark.asyncio
    async def test_detect_lockfile_yarn(self, adapter, tmp_path):
        (tmp_path / "yarn.lock").write_text('jest@^29.0.0:\n  version "29.7.0"\n')
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.LOCKFILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_lockfile_package_lock(self, adapter, tmp_path):
        (tmp_path / "package-lock.json").write_text(
            json.dumps({"dependencies": {"jest": {"version": "29.7.0"}}})
        )
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.LOCKFILE in result.signals

    @pytest.mark.asyncio
    async def test_confidence_increases_with_signals(self, adapter, tmp_path):
        """More signals should produce higher confidence."""
        r1 = await adapter.detect(str(tmp_path))

        (tmp_path / "jest.config.js").write_text("module.exports = {};\n")
        r2 = await adapter.detect(str(tmp_path))

        (tmp_path / "foo.test.js").write_text("test('x', () => {});\n")
        r3 = await adapter.detect(str(tmp_path))

        assert r2.confidence >= r1.confidence
        assert r3.confidence >= r2.confidence


# ===========================================================================
# Command building tests
# ===========================================================================


class TestJestCommandBuilding:
    """Tests for Jest command building."""

    def test_build_run_basic(self, adapter):
        req = _make_request(intent=TestIntent.RUN)
        cmd = adapter.build_command(req)
        assert cmd[:2] == ["npx", "jest"]
        assert "--verbose" in cmd

    def test_build_run_with_scope(self, adapter):
        req = _make_request(intent=TestIntent.RUN, scope="src/__tests__/")
        cmd = adapter.build_command(req)
        assert "src/__tests__/" in cmd

    def test_build_run_with_extra_args(self, adapter):
        req = _make_request(
            intent=TestIntent.RUN,
            extra_args=["--coverage", "--bail"],
        )
        cmd = adapter.build_command(req)
        assert "--coverage" in cmd
        assert "--bail" in cmd

    def test_build_list(self, adapter):
        req = _make_request(intent=TestIntent.LIST)
        cmd = adapter.build_command(req)
        assert "--listTests" in cmd

    def test_build_list_with_scope(self, adapter):
        req = _make_request(intent=TestIntent.LIST, scope="src/")
        cmd = adapter.build_command(req)
        assert "--listTests" in cmd
        assert "src/" in cmd

    def test_build_rerun_failed(self, adapter):
        req = _make_request(intent=TestIntent.RERUN_FAILED)
        cmd = adapter.build_command(req)
        assert "--onlyFailures" in cmd
        assert "--verbose" in cmd

    def test_build_run_specific_file_path(self, adapter):
        req = _make_request(
            intent=TestIntent.RUN_SPECIFIC,
            scope="src/__tests__/auth.test.ts",
        )
        cmd = adapter.build_command(req)
        assert "src/__tests__/auth.test.ts" in cmd
        assert "-t" not in cmd

    def test_build_run_specific_path_with_slash(self, adapter):
        req = _make_request(
            intent=TestIntent.RUN_SPECIFIC,
            scope="src/components/Button",
        )
        cmd = adapter.build_command(req)
        assert "src/components/Button" in cmd
        assert "-t" not in cmd

    def test_build_run_specific_test_name(self, adapter):
        req = _make_request(
            intent=TestIntent.RUN_SPECIFIC,
            scope="should handle login",
        )
        cmd = adapter.build_command(req)
        assert "-t" in cmd
        assert "should handle login" in cmd

    def test_build_unknown_intent_defaults_to_run(self, adapter):
        req = _make_request(intent=TestIntent.UNKNOWN)
        cmd = adapter.build_command(req)
        assert cmd[:2] == ["npx", "jest"]


# ===========================================================================
# Output parsing tests
# ===========================================================================


class TestJestOutputParsing:
    """Tests for Jest output parsing."""

    def test_parse_simple_pass(self, adapter):
        stdout = textwrap.dedent("""\
            PASS src/__tests__/math.test.js
              Math
                ✓ should add numbers (3 ms)
                ✓ should subtract numbers (1 ms)
                ✓ should multiply numbers (1 ms)

            Test Suites:  1 passed, 0 failed, 1 total
            Tests:        3 passed, 0 failed, 3 total
            Time:         1.234 s
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert result.success is True
        assert result.passed == 3
        assert result.failed == 0
        assert result.total == 3
        assert result.duration_seconds == pytest.approx(1.234)

    def test_parse_mixed_results(self, adapter):
        stdout = textwrap.dedent("""\
            FAIL src/__tests__/auth.test.js
              Auth
                ✓ should login (5 ms)
                ✕ should validate token (10 ms)
                ○ skipped should handle refresh

            Test Suites:  1 failed, 1 total
            Tests:        1 passed, 1 failed, 1 skipped, 3 total
            Time:         2.5 s
        """)
        result = adapter.parse_output(stdout, "", 1)
        assert result.success is False
        assert result.passed == 1
        assert result.failed == 1
        assert result.skipped == 1
        assert result.total == 3

    def test_parse_summary_only(self, adapter):
        """Non-verbose output with just summary lines."""
        stdout = textwrap.dedent("""\
            Test Suites:  3 passed, 0 failed, 3 total
            Tests:        15 passed, 0 failed, 15 total
            Time:         5.678 s
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert result.passed == 15
        assert result.total == 15
        assert result.duration_seconds == pytest.approx(5.678)

    def test_parse_failure_with_traceback(self, adapter):
        stdout = textwrap.dedent("""\
            FAIL src/__tests__/math.test.js
              Math
                ✓ should add (2 ms)
                ✕ should divide (3 ms)

              ● Math › should divide

                expect(received).toBe(expected)

                Expected: 0
                Received: Infinity

                  4 |   it('should divide', () => {
                  5 |     expect(1 / 0).toBe(0);
                     |                   ^
                  6 |   });

                  at Object.<anonymous> (src/__tests__/math.test.js:5:19)

            Test Suites:  1 failed, 1 total
            Tests:        1 passed, 1 failed, 2 total
            Time:         0.5 s
        """)
        result = adapter.parse_output(stdout, "", 1)
        assert result.failed == 1
        assert result.passed == 1

        failures = result.failure_details
        assert len(failures) >= 1
        fail = failures[0]
        assert "divide" in fail.name or "divide" in fail.traceback

    def test_parse_json_output(self, adapter):
        json_data = {
            "numFailedTestSuites": 0,
            "numPassedTestSuites": 2,
            "numTotalTestSuites": 2,
            "numFailedTests": 0,
            "numPassedTests": 5,
            "numPendingTests": 1,
            "numTotalTests": 6,
            "testResults": [
                {
                    "name": "src/__tests__/math.test.js",
                    "assertionResults": [
                        {
                            "ancestorTitles": ["Math"],
                            "title": "should add",
                            "status": "passed",
                            "duration": 3,
                            "failureMessages": [],
                        },
                        {
                            "ancestorTitles": ["Math"],
                            "title": "should subtract",
                            "status": "passed",
                            "duration": 1,
                            "failureMessages": [],
                        },
                    ],
                },
                {
                    "name": "src/__tests__/string.test.js",
                    "assertionResults": [
                        {
                            "ancestorTitles": ["String"],
                            "title": "should concat",
                            "status": "passed",
                            "duration": 2,
                            "failureMessages": [],
                        },
                        {
                            "ancestorTitles": ["String"],
                            "title": "should split",
                            "status": "passed",
                            "duration": 1,
                            "failureMessages": [],
                        },
                        {
                            "ancestorTitles": ["String"],
                            "title": "should trim",
                            "status": "passed",
                            "duration": 1,
                            "failureMessages": [],
                        },
                        {
                            "ancestorTitles": ["String"],
                            "title": "should uppercase",
                            "status": "pending",
                            "duration": None,
                            "failureMessages": [],
                        },
                    ],
                },
            ],
        }
        stdout = json.dumps(json_data)
        result = adapter.parse_output(stdout, "", 0)
        assert result.passed == 5
        assert result.skipped == 1
        assert result.total == 6
        assert len(result.test_cases) == 6
        assert result.metadata.get("json_parsed") is True

    def test_parse_json_with_failures(self, adapter):
        json_data = {
            "numFailedTestSuites": 1,
            "numPassedTestSuites": 0,
            "numTotalTestSuites": 1,
            "numFailedTests": 1,
            "numPassedTests": 1,
            "numPendingTests": 0,
            "numTotalTests": 2,
            "testResults": [
                {
                    "name": "src/__tests__/api.test.js",
                    "assertionResults": [
                        {
                            "ancestorTitles": ["API"],
                            "title": "should fetch",
                            "status": "passed",
                            "duration": 10,
                            "failureMessages": [],
                        },
                        {
                            "ancestorTitles": ["API"],
                            "title": "should handle error",
                            "status": "failed",
                            "duration": 5,
                            "failureMessages": [
                                "Error: expected 404 but got 500",
                            ],
                        },
                    ],
                },
            ],
        }
        stdout = json.dumps(json_data)
        result = adapter.parse_output(stdout, "", 1)
        assert result.failed == 1
        assert result.passed == 1
        assert result.success is False

        failures = result.failure_details
        assert len(failures) == 1
        assert "error" in failures[0].message.lower() or "404" in failures[0].message

    def test_parse_empty_output(self, adapter):
        result = adapter.parse_output("", "", 0)
        assert result.total == 0
        assert result.success is True
        assert result.test_cases == []

    def test_parse_time_in_milliseconds(self, adapter):
        stdout = textwrap.dedent("""\
            Tests:        2 passed, 0 failed, 2 total
            Time:         345 ms
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert result.duration_seconds == pytest.approx(0.345)

    def test_parse_time_in_seconds(self, adapter):
        stdout = textwrap.dedent("""\
            Tests:        10 passed, 0 failed, 10 total
            Time:         3.456 s
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert result.duration_seconds == pytest.approx(3.456)

    def test_parse_stderr_summary(self, adapter):
        """Jest may write summary to stderr."""
        stderr = textwrap.dedent("""\
            Tests:        4 passed, 0 failed, 4 total
            Time:         1.0 s
        """)
        result = adapter.parse_output("", stderr, 0)
        assert result.passed == 4
        assert result.total == 4

    def test_parse_preserves_raw_output(self, adapter):
        stdout = "some output"
        stderr = "some error"
        result = adapter.parse_output(stdout, stderr, 1)
        assert result.raw_stdout == stdout
        assert result.raw_stderr == stderr
        assert result.exit_code == 1

    def test_parse_pending_tests_counted_as_skipped(self, adapter):
        stdout = textwrap.dedent("""\
            Tests:        3 passed, 0 failed, 2 pending, 5 total
            Time:         1.0 s
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert result.skipped == 2
        assert result.passed == 3
        assert result.total == 5

    def test_parse_suite_headers(self, adapter):
        stdout = textwrap.dedent("""\
            PASS src/__tests__/math.test.js
            FAIL src/__tests__/auth.test.js

            Tests:        3 passed, 1 failed, 4 total
            Time:         2.0 s
        """)
        result = adapter.parse_output(stdout, "", 1)
        suite_results = result.metadata.get("suite_results", {})
        assert suite_results.get("src/__tests__/math.test.js") == "PASS"
        assert suite_results.get("src/__tests__/auth.test.js") == "FAIL"

    def test_parse_multiple_failures(self, adapter):
        stdout = textwrap.dedent("""\
            FAIL src/__tests__/api.test.js
              API
                ✕ should fetch data (5 ms)
                ✕ should handle errors (3 ms)

              ● API › should fetch data

                TypeError: fetch is not defined

                  at Object.<anonymous> (src/__tests__/api.test.js:3:5)

              ● API › should handle errors

                ReferenceError: response is not defined

                  at Object.<anonymous> (src/__tests__/api.test.js:8:5)

            Tests:        0 passed, 2 failed, 2 total
            Time:         0.5 s
        """)
        result = adapter.parse_output(stdout, "", 1)
        assert result.failed == 2
        failures = result.failure_details
        assert len(failures) == 2

    def test_success_property(self, adapter):
        stdout = "Tests:        5 passed, 0 failed, 5 total\nTime:         1.0 s"
        result = adapter.parse_output(stdout, "", 0)
        assert result.success is True

        stdout_fail = "Tests:        3 passed, 2 failed, 5 total\nTime:         1.0 s"
        result_fail = adapter.parse_output(stdout_fail, "", 1)
        assert result_fail.success is False

    def test_summary_line_captured(self, adapter):
        stdout = "Tests:        5 passed, 0 failed, 5 total\nTime:         1.0 s"
        result = adapter.parse_output(stdout, "", 0)
        assert "5 passed" in result.summary_line

    def test_parse_list_output(self, adapter):
        stdout = textwrap.dedent("""\
            /home/user/project/src/__tests__/math.test.js
            /home/user/project/src/__tests__/auth.test.ts
            /home/user/project/src/components/__tests__/Button.test.tsx
        """)
        paths = adapter.parse_list_output(stdout)
        assert len(paths) == 3
        assert "/home/user/project/src/__tests__/math.test.js" in paths
        assert "/home/user/project/src/__tests__/auth.test.ts" in paths

    def test_parse_list_output_with_ansi(self, adapter):
        stdout = "\x1b[33m/home/user/project/src/__tests__/math.test.js\x1b[0m\n"
        paths = adapter.parse_list_output(stdout)
        assert len(paths) == 1

    def test_json_ancestor_titles(self, adapter):
        """Verify ancestor titles are joined with › separator."""
        json_data = {
            "numFailedTestSuites": 0,
            "numPassedTestSuites": 1,
            "numTotalTestSuites": 1,
            "numFailedTests": 0,
            "numPassedTests": 1,
            "numPendingTests": 0,
            "numTotalTests": 1,
            "testResults": [
                {
                    "name": "test.js",
                    "assertionResults": [
                        {
                            "ancestorTitles": ["Outer", "Inner"],
                            "title": "should work",
                            "status": "passed",
                            "duration": 1,
                            "failureMessages": [],
                        },
                    ],
                },
            ],
        }
        stdout = json.dumps(json_data)
        result = adapter.parse_output(stdout, "", 0)
        assert result.test_cases[0].name == "Outer › Inner › should work"


# ===========================================================================
# ANSI stripping tests
# ===========================================================================


class TestAnsiStripping:
    """Tests for ANSI escape code removal."""

    def test_strip_ansi_codes(self):
        text = "\x1b[32m✓\x1b[0m should work \x1b[90m(5 ms)\x1b[0m"
        clean = _strip_ansi(text)
        assert "\x1b" not in clean
        assert "should work" in clean

    def test_strip_ansi_no_codes(self):
        text = "plain text"
        assert _strip_ansi(text) == text


# ===========================================================================
# Adapter properties
# ===========================================================================


class TestJestAdapterProperties:
    """Test adapter identity properties."""

    def test_framework(self, adapter):
        assert adapter.framework == TestFramework.JEST

    def test_display_name(self, adapter):
        assert adapter.display_name == "Jest"
