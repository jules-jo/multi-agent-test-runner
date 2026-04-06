"""Tests for the pytest framework adapter.

Covers detection, command building, and output parsing.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from test_runner.agents.parser import ParsedTestRequest, TestFramework, TestIntent
from test_runner.frameworks.base import (
    DetectionSignal,
    TestOutcome,
)
from test_runner.frameworks.pytest_adapter import PytestAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter():
    return PytestAdapter()


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with pytest signals."""
    # pyproject.toml with pytest config
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
        [tool.pytest.ini_options]
        testpaths = ["tests"]
        """)
    )
    # A test file
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_example.py").write_text(
        textwrap.dedent("""\
        def test_hello():
            assert True
        """)
    )
    (tests_dir / "conftest.py").write_text("")
    return tmp_path


def _make_request(
    intent: TestIntent = TestIntent.RUN,
    framework: TestFramework = TestFramework.PYTEST,
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


class TestPytestDetection:
    """Tests for pytest detection logic."""

    @pytest.mark.asyncio
    async def test_detect_with_pyproject_config(self, adapter, project_dir):
        result = await adapter.detect(str(project_dir))
        assert result.detected is True
        assert result.framework == TestFramework.PYTEST
        assert DetectionSignal.CONFIG_FILE in result.signals
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_detect_with_pytest_ini(self, adapter, tmp_path):
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        result = await adapter.detect(str(tmp_path))
        assert result.detected is True
        assert DetectionSignal.CONFIG_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_setup_cfg(self, adapter, tmp_path):
        (tmp_path / "setup.cfg").write_text("[tool:pytest]\naddopts = -v\n")
        result = await adapter.detect(str(tmp_path))
        assert result.detected is True
        assert DetectionSignal.CONFIG_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_requirements_txt(self, adapter, tmp_path):
        (tmp_path / "requirements.txt").write_text("pytest>=8.0\nrequests\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.DEPENDENCY_FILE in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_test_files(self, adapter, tmp_path):
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_something.py").write_text("def test_a(): pass\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.TEST_FILE_PATTERN in result.signals

    @pytest.mark.asyncio
    async def test_detect_with_conftest(self, adapter, tmp_path):
        (tmp_path / "conftest.py").write_text("import pytest\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.IMPORT_STATEMENT in result.signals

    @pytest.mark.asyncio
    async def test_detect_empty_directory(self, adapter, tmp_path):
        result = await adapter.detect(str(tmp_path))
        # May or may not detect depending on CLI availability
        # But should not have config/dep/test file signals
        assert DetectionSignal.CONFIG_FILE not in result.signals
        assert DetectionSignal.DEPENDENCY_FILE not in result.signals
        assert DetectionSignal.TEST_FILE_PATTERN not in result.signals

    @pytest.mark.asyncio
    async def test_detect_excludes_venv(self, adapter, tmp_path):
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "test_internal.py").write_text("def test_x(): pass\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.TEST_FILE_PATTERN not in result.signals

    @pytest.mark.asyncio
    async def test_detect_full_project(self, adapter, project_dir):
        """Full project with config, conftest, test files — high confidence."""
        result = await adapter.detect(str(project_dir))
        assert result.detected is True
        assert result.confidence >= 0.5
        assert len(result.signals) >= 2

    @pytest.mark.asyncio
    async def test_detect_lockfile_signal(self, adapter, tmp_path):
        (tmp_path / "uv.lock").write_text("pytest = '>=8.0'\n")
        result = await adapter.detect(str(tmp_path))
        assert DetectionSignal.LOCKFILE in result.signals

    @pytest.mark.asyncio
    async def test_confidence_increases_with_signals(self, adapter, tmp_path):
        """More signals should produce higher confidence."""
        # No signals
        r1 = await adapter.detect(str(tmp_path))

        # Add config file
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        r2 = await adapter.detect(str(tmp_path))

        # Add test file
        (tmp_path / "test_foo.py").write_text("def test_x(): pass\n")
        r3 = await adapter.detect(str(tmp_path))

        # More signals → higher confidence
        # r2 should have >= r1 confidence (config file adds signal)
        assert r2.confidence >= r1.confidence
        assert r3.confidence >= r2.confidence


# ===========================================================================
# Command building tests
# ===========================================================================


class TestPytestCommandBuilding:
    """Tests for pytest command building."""

    def test_build_run_basic(self, adapter):
        req = _make_request(intent=TestIntent.RUN)
        cmd = adapter.build_command(req)
        assert cmd[0] == "pytest"
        assert "-v" in cmd

    def test_build_run_with_scope(self, adapter):
        req = _make_request(intent=TestIntent.RUN, scope="tests/unit/")
        cmd = adapter.build_command(req)
        assert "tests/unit/" in cmd

    def test_build_run_with_extra_args(self, adapter):
        req = _make_request(intent=TestIntent.RUN, extra_args=["-x", "--tb=short"])
        cmd = adapter.build_command(req)
        assert "-x" in cmd
        assert "--tb=short" in cmd

    def test_build_list(self, adapter):
        req = _make_request(intent=TestIntent.LIST)
        cmd = adapter.build_command(req)
        assert "--collect-only" in cmd
        assert "-q" in cmd

    def test_build_list_with_scope(self, adapter):
        req = _make_request(intent=TestIntent.LIST, scope="tests/")
        cmd = adapter.build_command(req)
        assert "--collect-only" in cmd
        assert "tests/" in cmd

    def test_build_rerun_failed(self, adapter):
        req = _make_request(intent=TestIntent.RERUN_FAILED)
        cmd = adapter.build_command(req)
        assert "--lf" in cmd

    def test_build_run_specific_node_id(self, adapter):
        req = _make_request(
            intent=TestIntent.RUN_SPECIFIC,
            scope="tests/test_foo.py::test_bar",
        )
        cmd = adapter.build_command(req)
        assert "tests/test_foo.py::test_bar" in cmd
        # Should not use -k for node IDs
        assert "-k" not in cmd

    def test_build_run_specific_keyword(self, adapter):
        req = _make_request(
            intent=TestIntent.RUN_SPECIFIC,
            scope="test_login",
        )
        cmd = adapter.build_command(req)
        assert "-k" in cmd
        assert "test_login" in cmd

    def test_build_unknown_intent_defaults_to_run(self, adapter):
        req = _make_request(intent=TestIntent.UNKNOWN)
        cmd = adapter.build_command(req)
        assert cmd[0] == "pytest"


# ===========================================================================
# Output parsing tests
# ===========================================================================


class TestPytestOutputParsing:
    """Tests for pytest output parsing."""

    def test_parse_simple_pass(self, adapter):
        stdout = textwrap.dedent("""\
            ============================= test session starts ==============================
            collected 3 items

            tests/test_foo.py::test_one PASSED                                      [ 33%]
            tests/test_foo.py::test_two PASSED                                      [ 66%]
            tests/test_foo.py::test_three PASSED                                    [100%]

            ============================== 3 passed in 0.42s ===============================
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert result.success is True
        assert result.passed == 3
        assert result.failed == 0
        assert result.total == 3
        assert result.duration_seconds == pytest.approx(0.42)
        assert len(result.test_cases) == 3

    def test_parse_mixed_results(self, adapter):
        stdout = textwrap.dedent("""\
            ============================= test session starts ==============================
            collected 5 items

            tests/test_foo.py::test_pass PASSED                                     [ 20%]
            tests/test_foo.py::test_fail FAILED                                     [ 40%]
            tests/test_foo.py::test_skip SKIPPED                                    [ 60%]
            tests/test_foo.py::test_pass2 PASSED                                    [ 80%]
            tests/test_foo.py::test_error ERROR                                     [100%]

            ============================== 2 passed, 1 failed, 1 skipped, 1 error in 1.23s ===============================
        """)
        result = adapter.parse_output(stdout, "", 1)
        assert result.success is False
        assert result.passed == 2
        assert result.failed == 1
        assert result.skipped == 1
        assert result.errors == 1
        assert result.total == 5

    def test_parse_summary_only(self, adapter):
        """Non-verbose output: only summary line, no per-test results."""
        stdout = textwrap.dedent("""\
            ============================= test session starts ==============================
            collected 10 items
            ..........
            ============================== 10 passed in 2.50s ==============================
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert result.passed == 10
        assert result.total == 10
        assert result.duration_seconds == pytest.approx(2.5)

    def test_parse_failure_with_traceback(self, adapter):
        stdout = textwrap.dedent("""\
            ============================= test session starts ==============================
            collected 2 items

            tests/test_math.py::test_add PASSED                                    [ 50%]
            tests/test_math.py::test_divide FAILED                                 [100%]

            =================================== FAILURES ===================================
            _________________________________ test_divide __________________________________

                def test_divide():
            >       assert 1 / 0 == 0
            E       ZeroDivisionError: division by zero

            tests/test_math.py:5: ZeroDivisionError
            =========================== short test summary info ============================
            FAILED tests/test_math.py::test_divide - ZeroDivisionError: division by zero
            ========================= 1 passed, 1 failed in 0.10s =========================
        """)
        result = adapter.parse_output(stdout, "", 1)
        assert result.failed == 1
        assert result.passed == 1

        # Check failure details
        failures = result.failure_details
        assert len(failures) == 1
        fail = failures[0]
        assert "test_divide" in fail.name
        assert "ZeroDivisionError" in fail.traceback or "ZeroDivisionError" in fail.message

    def test_parse_short_summary_messages(self, adapter):
        stdout = textwrap.dedent("""\
            =========================== short test summary info ============================
            FAILED tests/test_a.py::test_x - AssertionError: expected 1, got 2
            FAILED tests/test_b.py::test_y - TypeError: unsupported
            ========================= 2 failed in 0.50s ===================================
        """)
        result = adapter.parse_output(stdout, "", 1)
        assert result.failed == 2
        messages = [tc.message for tc in result.failure_details]
        assert any("AssertionError" in m for m in messages)
        assert any("TypeError" in m for m in messages)

    def test_parse_xfail_xpass(self, adapter):
        stdout = textwrap.dedent("""\
            tests/test_foo.py::test_expected_fail XFAIL                             [ 50%]
            tests/test_foo.py::test_unexpected_pass XPASS                           [100%]

            ==================== 1 xfailed, 1 xpassed in 0.05s ====================
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert result.xfailed == 1
        assert result.xpassed == 1

    def test_parse_empty_output(self, adapter):
        result = adapter.parse_output("", "", 0)
        assert result.total == 0
        assert result.success is True
        assert result.test_cases == []

    def test_parse_no_tests_collected(self, adapter):
        stdout = textwrap.dedent("""\
            ============================= test session starts ==============================
            collected 0 items

            ============================ no tests ran in 0.01s =============================
        """)
        result = adapter.parse_output(stdout, "", 5)
        assert result.total == 0

    def test_parse_collect_only(self, adapter):
        stdout = textwrap.dedent("""\
            tests/test_foo.py::test_one
            tests/test_foo.py::test_two
            tests/test_bar.py::TestClass::test_method

            3 tests collected in 0.10s
        """)
        nodeids = adapter.parse_collect_output(stdout)
        assert len(nodeids) == 3
        assert "tests/test_foo.py::test_one" in nodeids
        assert "tests/test_bar.py::TestClass::test_method" in nodeids

    def test_parse_preserves_raw_output(self, adapter):
        stdout = "some output"
        stderr = "some error"
        result = adapter.parse_output(stdout, stderr, 1)
        assert result.raw_stdout == stdout
        assert result.raw_stderr == stderr
        assert result.exit_code == 1

    def test_parse_duration_extraction(self, adapter):
        stdout = textwrap.dedent("""\
            ============================== 5 passed in 12.34s ==============================
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert result.duration_seconds == pytest.approx(12.34)

    def test_parse_file_path_extraction(self, adapter):
        stdout = textwrap.dedent("""\
            tests/unit/test_auth.py::TestLogin::test_valid PASSED                   [100%]

            ============================== 1 passed in 0.01s ===============================
        """)
        result = adapter.parse_output(stdout, "", 0)
        assert len(result.test_cases) == 1
        assert result.test_cases[0].file_path == "tests/unit/test_auth.py"
        assert result.test_cases[0].name == "tests/unit/test_auth.py::TestLogin::test_valid"

    def test_success_property(self, adapter):
        stdout = "============================== 3 passed in 0.10s =============================="
        result = adapter.parse_output(stdout, "", 0)
        assert result.success is True

        stdout_fail = "========================= 1 failed in 0.10s ========================="
        result_fail = adapter.parse_output(stdout_fail, "", 1)
        assert result_fail.success is False

    def test_summary_line_captured(self, adapter):
        stdout = "============================== 5 passed in 1.00s =============================="
        result = adapter.parse_output(stdout, "", 0)
        assert "5 passed" in result.summary_line

    def test_parse_multiple_failures_with_tracebacks(self, adapter):
        stdout = textwrap.dedent("""\
            tests/test_a.py::test_x FAILED                                          [ 50%]
            tests/test_a.py::test_y FAILED                                          [100%]

            =================================== FAILURES ===================================
            _____________________________________ test_x _____________________________________

                def test_x():
            >       assert False
            E       AssertionError

            tests/test_a.py:2: AssertionError
            _____________________________________ test_y _____________________________________

                def test_y():
            >       raise ValueError("bad")
            E       ValueError: bad

            tests/test_a.py:5: ValueError
            ========================= 2 failed in 0.10s ===================================
        """)
        result = adapter.parse_output(stdout, "", 1)
        assert result.failed == 2
        failures = result.failure_details
        assert len(failures) == 2

    def test_parse_stderr_in_summary_search(self, adapter):
        """Summary line might appear in stderr for some pytest configs."""
        stderr = "============================== 2 passed in 0.50s =============================="
        result = adapter.parse_output("", stderr, 0)
        assert result.passed == 2


# ===========================================================================
# Adapter properties
# ===========================================================================


class TestPytestAdapterProperties:
    """Test adapter identity properties."""

    def test_framework(self, adapter):
        assert adapter.framework == TestFramework.PYTEST

    def test_display_name(self, adapter):
        assert adapter.display_name == "pytest"
