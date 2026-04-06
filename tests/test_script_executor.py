"""Tests for the generic/arbitrary script executor.

Covers ScriptCommand, ScriptResult, ScriptExecutor, ScriptType classification,
and the full execution pipeline including output capture, exit code handling,
shell mode, env injection, timeouts, batch execution, and output callbacks.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_runner.execution.targets import (
    ExecutionResult,
    ExecutionStatus,
    ExecutionTarget,
    LocalTarget,
    TargetRegistry,
)
from test_runner.execution.script_executor import (
    OutputCallback,
    ScriptCommand,
    ScriptExecutor,
    ScriptResult,
    ScriptType,
    classify_command,
)


# ---------------------------------------------------------------------------
# classify_command tests
# ---------------------------------------------------------------------------


class TestClassifyCommand:
    """Tests for the classify_command heuristic classifier."""

    def test_make_target(self):
        assert classify_command("make test") == ScriptType.MAKE_TARGET
        assert classify_command("make") == ScriptType.MAKE_TARGET
        assert classify_command("make clean build test") == ScriptType.MAKE_TARGET

    def test_npm_scripts(self):
        assert classify_command("npm test") == ScriptType.NPM_SCRIPT
        assert classify_command("npm run test") == ScriptType.NPM_SCRIPT
        assert classify_command("yarn test") == ScriptType.NPM_SCRIPT
        assert classify_command("pnpm run test") == ScriptType.NPM_SCRIPT

    def test_python_script(self):
        assert classify_command("python test.py") == ScriptType.PYTHON_SCRIPT
        assert classify_command("python3 -m pytest") == ScriptType.PYTHON_SCRIPT

    def test_script_file_relative(self):
        assert classify_command("./run_tests.sh") == ScriptType.SCRIPT_FILE
        assert classify_command("/usr/local/bin/test.sh") == ScriptType.SCRIPT_FILE

    def test_script_file_by_extension(self):
        assert classify_command("run_tests.sh") == ScriptType.SCRIPT_FILE
        assert classify_command("test.bash") == ScriptType.SCRIPT_FILE
        assert classify_command("test.pl arg1") == ScriptType.SCRIPT_FILE

    def test_inline_shell_pipe(self):
        assert classify_command("cat foo | grep bar") == ScriptType.INLINE_SHELL

    def test_inline_shell_chain(self):
        assert classify_command("cmd1 && cmd2") == ScriptType.INLINE_SHELL
        assert classify_command("cmd1 || cmd2") == ScriptType.INLINE_SHELL
        assert classify_command("cmd1 ; cmd2") == ScriptType.INLINE_SHELL

    def test_inline_shell_redirect(self):
        assert classify_command("echo hello > out.txt") == ScriptType.INLINE_SHELL
        assert classify_command("echo hello >> out.txt") == ScriptType.INLINE_SHELL

    def test_inline_shell_bash_c(self):
        assert classify_command("bash -c 'echo hello'") == ScriptType.INLINE_SHELL
        assert classify_command("sh -c 'echo hello'") == ScriptType.INLINE_SHELL

    def test_default_shell_command(self):
        assert classify_command("pytest -v") == ScriptType.SHELL_COMMAND
        assert classify_command("cargo test") == ScriptType.SHELL_COMMAND
        assert classify_command("some_unknown_tool --flag") == ScriptType.SHELL_COMMAND


# ---------------------------------------------------------------------------
# ScriptCommand tests
# ---------------------------------------------------------------------------


class TestScriptCommand:
    """Tests for ScriptCommand dataclass and token parsing."""

    def test_auto_tokenize_from_raw(self):
        cmd = ScriptCommand(raw_command="pytest -v tests/")
        assert cmd.tokens == ("pytest", "-v", "tests/")

    def test_auto_tokenize_with_quotes(self):
        cmd = ScriptCommand(raw_command='pytest -k "test_foo or test_bar"')
        assert cmd.tokens == ("pytest", "-k", "test_foo or test_bar")

    def test_explicit_tokens_preserved(self):
        cmd = ScriptCommand(raw_command="ignored", tokens=("a", "b", "c"))
        assert cmd.tokens == ("a", "b", "c")

    def test_command_list(self):
        cmd = ScriptCommand(raw_command="echo hello world")
        assert cmd.command_list == ["echo", "hello", "world"]
        assert isinstance(cmd.command_list, list)

    def test_display_uses_raw_command(self):
        cmd = ScriptCommand(raw_command="pytest -v tests/")
        assert cmd.display == "pytest -v tests/"

    def test_display_falls_back_to_tokens(self):
        cmd = ScriptCommand(raw_command="", tokens=("pytest", "-v"))
        assert cmd.display == "pytest -v"

    def test_default_success_codes(self):
        cmd = ScriptCommand(raw_command="echo ok")
        assert cmd.success_exit_codes == frozenset({0})

    def test_custom_success_codes(self):
        cmd = ScriptCommand(
            raw_command="echo ok",
            success_exit_codes=frozenset({0, 1}),
        )
        assert 1 in cmd.success_exit_codes

    def test_to_test_command(self):
        cmd = ScriptCommand(
            raw_command="python test.py",
            working_directory="/tmp",
            env={"CI": "true"},
            timeout=30,
            script_type=ScriptType.PYTHON_SCRIPT,
            metadata={"source": "user"},
        )
        tc = cmd.to_test_command()
        assert tc.command == ["python", "test.py"]
        assert tc.display == "python test.py"
        assert tc.working_directory == "/tmp"
        assert tc.env == {"CI": "true"}
        assert tc.timeout == 30
        assert tc.metadata["script_type"] == "python_script"
        assert tc.metadata["source"] == "user"

    def test_frozen(self):
        cmd = ScriptCommand(raw_command="echo hi")
        with pytest.raises(AttributeError):
            cmd.raw_command = "nope"  # type: ignore[misc]

    def test_malformed_quotes_fallback(self):
        """shlex.split fails on unmatched quotes; should fallback to split()."""
        cmd = ScriptCommand(raw_command="echo 'hello world")
        # Fallback naive split
        assert len(cmd.tokens) > 0
        assert cmd.tokens[0] == "echo"


# ---------------------------------------------------------------------------
# ScriptResult tests
# ---------------------------------------------------------------------------


class TestScriptResult:
    """Tests for ScriptResult properties and conversion."""

    def _make_result(self, **kwargs: Any) -> ScriptResult:
        defaults = dict(
            exit_code=0,
            stdout="line1\nline2\nline3",
            stderr="",
            combined_output="line1\nline2\nline3",
            duration_seconds=1.5,
            success=True,
            status=ExecutionStatus.PASSED,
            command_display="echo hello",
        )
        defaults.update(kwargs)
        return ScriptResult(**defaults)

    def test_output_lines(self):
        r = self._make_result(stdout="a\nb\nc")
        assert r.output_lines == ["a", "b", "c"]

    def test_output_lines_empty(self):
        r = self._make_result(stdout="")
        assert r.output_lines == []

    def test_error_lines(self):
        r = self._make_result(stderr="err1\nerr2")
        assert r.error_lines == ["err1", "err2"]

    def test_output_summary_short(self):
        r = self._make_result(stdout="a\nb\nc")
        assert r.output_summary == "a\nb\nc"

    def test_output_summary_long(self):
        lines = [f"line{i}" for i in range(20)]
        r = self._make_result(stdout="\n".join(lines))
        summary = r.output_summary
        assert "line0" in summary
        assert "line19" in summary
        assert "omitted" in summary

    def test_has_output_true(self):
        r = self._make_result(stdout="something")
        assert r.has_output is True

    def test_has_output_false(self):
        r = self._make_result(stdout="", stderr="")
        assert r.has_output is False

    def test_has_output_whitespace_only(self):
        r = self._make_result(stdout="   ", stderr="  ")
        assert r.has_output is False

    def test_to_execution_result(self):
        r = self._make_result(
            exit_code=1,
            stdout="out",
            stderr="err",
            duration_seconds=2.0,
            status=ExecutionStatus.FAILED,
        )
        er = r.to_execution_result()
        assert er.exit_code == 1
        assert er.stdout == "out"
        assert er.stderr == "err"
        assert er.duration_seconds == 2.0
        assert er.status == ExecutionStatus.FAILED
        assert er.metadata["timed_out"] is False
        assert er.metadata["success_by_exit_code"] is True  # based on success field


# ---------------------------------------------------------------------------
# Fake execution target for testing
# ---------------------------------------------------------------------------


class FakeTarget(ExecutionTarget):
    """A fake execution target that returns preconfigured results."""

    def __init__(
        self,
        results: list[ExecutionResult] | None = None,
        default_result: ExecutionResult | None = None,
    ) -> None:
        self._results = list(results or [])
        self._default = default_result or ExecutionResult(
            status=ExecutionStatus.PASSED,
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
            command_display="fake",
        )
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "fake"

    async def execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        self.calls.append({
            "command": command,
            "working_directory": working_directory,
            "env": env,
            "timeout": timeout,
        })
        if self._results:
            return self._results.pop(0)
        return self._default


# ---------------------------------------------------------------------------
# ScriptExecutor tests
# ---------------------------------------------------------------------------


class TestScriptExecutor:
    """Tests for the main ScriptExecutor class."""

    @pytest.fixture
    def fake_target(self) -> FakeTarget:
        return FakeTarget()

    @pytest.fixture
    def executor(self, fake_target: FakeTarget) -> ScriptExecutor:
        return ScriptExecutor(target=fake_target)

    # -- Basic run ----------------------------------------------------------

    async def test_run_simple_command(self, executor: ScriptExecutor, fake_target: FakeTarget):
        result = await executor.run("echo hello")
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "ok"
        assert result.command_display == "echo hello"
        # Verify tokens passed to target
        assert fake_target.calls[0]["command"] == ["echo", "hello"]

    async def test_run_captures_exit_code(self, fake_target: FakeTarget):
        fake_target._default = ExecutionResult(
            status=ExecutionStatus.FAILED,
            exit_code=1,
            stdout="",
            stderr="assertion error",
            duration_seconds=0.5,
            command_display="pytest",
        )
        executor = ScriptExecutor(target=fake_target)
        result = await executor.run("pytest -v")
        assert result.success is False
        assert result.exit_code == 1
        assert result.status == ExecutionStatus.FAILED
        assert "assertion error" in result.stderr

    async def test_run_custom_success_codes(self, fake_target: FakeTarget):
        """Exit code 1 should be treated as success when configured."""
        fake_target._default = ExecutionResult(
            status=ExecutionStatus.FAILED,  # target says failed
            exit_code=1,
            stdout="warnings found",
            stderr="",
            duration_seconds=0.2,
            command_display="lint",
        )
        executor = ScriptExecutor(target=fake_target)
        result = await executor.run(
            "lint check",
            success_exit_codes=frozenset({0, 1}),
        )
        # ScriptExecutor interprets exit code 1 as success since it's in the set
        assert result.success is True
        assert result.status == ExecutionStatus.PASSED

    # -- Working directory and env ------------------------------------------

    async def test_run_with_working_directory(self, executor: ScriptExecutor, fake_target: FakeTarget):
        await executor.run("pytest", working_directory="/app/tests")
        assert fake_target.calls[0]["working_directory"] == "/app/tests"

    async def test_run_with_env_vars(self, executor: ScriptExecutor, fake_target: FakeTarget):
        await executor.run("pytest", env={"CI": "true", "VERBOSE": "1"})
        assert fake_target.calls[0]["env"] == {"CI": "true", "VERBOSE": "1"}

    # -- Timeout handling ---------------------------------------------------

    async def test_run_with_timeout(self, executor: ScriptExecutor, fake_target: FakeTarget):
        await executor.run("long_test", timeout=60)
        assert fake_target.calls[0]["timeout"] == 60

    async def test_run_timeout_result(self, fake_target: FakeTarget):
        fake_target._default = ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            exit_code=-1,
            stdout="",
            stderr="timed out",
            duration_seconds=30.0,
            command_display="slow_test",
        )
        executor = ScriptExecutor(target=fake_target)
        result = await executor.run("slow_test", timeout=30)
        assert result.timed_out is True
        assert result.success is False
        assert result.status == ExecutionStatus.TIMEOUT
        assert "timed out" in result.error_message.lower()

    async def test_default_timeout_applied(self, fake_target: FakeTarget):
        executor = ScriptExecutor(target=fake_target, default_timeout=120)
        await executor.run("test")
        assert fake_target.calls[0]["timeout"] == 120

    async def test_explicit_timeout_overrides_default(self, fake_target: FakeTarget):
        executor = ScriptExecutor(target=fake_target, default_timeout=120)
        await executor.run("test", timeout=30)
        assert fake_target.calls[0]["timeout"] == 30

    # -- Shell mode ---------------------------------------------------------

    async def test_shell_mode_wraps_in_bash(self, executor: ScriptExecutor, fake_target: FakeTarget):
        await executor.run("cat foo | grep bar", shell_mode=True)
        cmd = fake_target.calls[0]["command"]
        assert cmd == ["bash", "-c", "cat foo | grep bar"]

    async def test_non_shell_mode_uses_tokens(self, executor: ScriptExecutor, fake_target: FakeTarget):
        await executor.run("pytest -v tests/")
        cmd = fake_target.calls[0]["command"]
        assert cmd == ["pytest", "-v", "tests/"]

    # -- ScriptCommand-based execution --------------------------------------

    async def test_run_command_with_script_command(self, executor: ScriptExecutor, fake_target: FakeTarget):
        script_cmd = ScriptCommand(
            raw_command="python -m pytest tests/unit",
            working_directory="/project",
            env={"DEBUG": "1"},
            timeout=45,
            script_type=ScriptType.PYTHON_SCRIPT,
        )
        result = await executor.run_command(script_cmd)
        assert result.success is True
        assert fake_target.calls[0]["command"] == ["python", "-m", "pytest", "tests/unit"]
        assert fake_target.calls[0]["working_directory"] == "/project"
        assert fake_target.calls[0]["env"] == {"DEBUG": "1"}
        assert fake_target.calls[0]["timeout"] == 45

    async def test_run_command_shell_mode(self, executor: ScriptExecutor, fake_target: FakeTarget):
        script_cmd = ScriptCommand(
            raw_command="echo hello && echo world",
            shell_mode=True,
        )
        result = await executor.run_command(script_cmd)
        assert fake_target.calls[0]["command"] == ["bash", "-c", "echo hello && echo world"]

    async def test_run_command_empty_tokens(self, executor: ScriptExecutor, fake_target: FakeTarget):
        """Empty command should be wrapped in a fallback."""
        script_cmd = ScriptCommand(raw_command="", tokens=())
        result = await executor.run_command(script_cmd)
        cmd = fake_target.calls[0]["command"]
        assert cmd[0] == "bash"  # fallback wrapping

    # -- Combined output ----------------------------------------------------

    async def test_combined_output(self, fake_target: FakeTarget):
        fake_target._default = ExecutionResult(
            status=ExecutionStatus.PASSED,
            exit_code=0,
            stdout="stdout line",
            stderr="stderr line",
            duration_seconds=0.1,
            command_display="test",
        )
        executor = ScriptExecutor(target=fake_target)
        result = await executor.run("test")
        assert "stdout line" in result.combined_output
        assert "stderr line" in result.combined_output

    async def test_combined_output_empty_stderr(self, fake_target: FakeTarget):
        fake_target._default = ExecutionResult(
            status=ExecutionStatus.PASSED,
            exit_code=0,
            stdout="output only",
            stderr="",
            duration_seconds=0.1,
            command_display="test",
        )
        executor = ScriptExecutor(target=fake_target)
        result = await executor.run("test")
        assert result.combined_output == "output only"

    # -- Error handling -----------------------------------------------------

    async def test_infrastructure_error(self, fake_target: FakeTarget):
        fake_target._default = ExecutionResult(
            status=ExecutionStatus.ERROR,
            exit_code=-1,
            stdout="",
            stderr="Command not found: foobar",
            duration_seconds=0.01,
            command_display="foobar",
        )
        executor = ScriptExecutor(target=fake_target)
        result = await executor.run("foobar")
        assert result.success is False
        assert result.status == ExecutionStatus.ERROR
        assert "Command not found" in result.stderr
        assert result.error_message  # non-empty

    async def test_error_message_on_failure(self, fake_target: FakeTarget):
        fake_target._default = ExecutionResult(
            status=ExecutionStatus.FAILED,
            exit_code=2,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            command_display="test",
        )
        executor = ScriptExecutor(target=fake_target)
        result = await executor.run("test")
        assert "exited with code 2" in result.error_message
        assert "expected one of" in result.error_message

    # -- Output callbacks ---------------------------------------------------

    async def test_output_callback_invoked(self, fake_target: FakeTarget):
        fake_target._default = ExecutionResult(
            status=ExecutionStatus.PASSED,
            exit_code=0,
            stdout="line1\nline2",
            stderr="err1",
            duration_seconds=0.1,
            command_display="test",
        )
        captured: list[tuple[str, str]] = []

        def callback(stream: str, line: str) -> None:
            captured.append((stream, line))

        executor = ScriptExecutor(target=fake_target, on_output=callback)
        await executor.run("test")
        assert ("stdout", "line1") in captured
        assert ("stdout", "line2") in captured
        assert ("stderr", "err1") in captured

    async def test_output_callback_exception_handled(self, fake_target: FakeTarget):
        """Callback errors should not crash the executor."""

        def bad_callback(stream: str, line: str) -> None:
            raise RuntimeError("callback boom")

        executor = ScriptExecutor(target=fake_target, on_output=bad_callback)
        result = await executor.run("test")
        assert result.success is True  # should not crash

    # -- History tracking ---------------------------------------------------

    async def test_history_tracking(self, executor: ScriptExecutor):
        await executor.run("cmd1")
        await executor.run("cmd2")
        assert len(executor.history) == 2
        assert executor.history[0].command_display == "cmd1"
        assert executor.history[1].command_display == "cmd2"

    async def test_history_is_copy(self, executor: ScriptExecutor):
        await executor.run("cmd1")
        history = executor.history
        history.clear()
        assert len(executor.history) == 1  # original not affected

    # -- Batch execution ----------------------------------------------------

    async def test_run_multiple_all_pass(self, executor: ScriptExecutor):
        results = await executor.run_multiple(["cmd1", "cmd2", "cmd3"])
        assert len(results) == 3
        assert all(r.success for r in results)

    async def test_run_multiple_stop_on_failure(self, fake_target: FakeTarget):
        fake_target._results = [
            ExecutionResult(
                status=ExecutionStatus.PASSED, exit_code=0,
                stdout="ok", stderr="", duration_seconds=0.1,
                command_display="cmd1",
            ),
            ExecutionResult(
                status=ExecutionStatus.FAILED, exit_code=1,
                stdout="", stderr="fail", duration_seconds=0.1,
                command_display="cmd2",
            ),
            ExecutionResult(
                status=ExecutionStatus.PASSED, exit_code=0,
                stdout="ok", stderr="", duration_seconds=0.1,
                command_display="cmd3",
            ),
        ]
        executor = ScriptExecutor(target=fake_target)
        results = await executor.run_multiple(
            ["cmd1", "cmd2", "cmd3"],
            stop_on_failure=True,
        )
        assert len(results) == 2  # stopped after cmd2
        assert results[0].success is True
        assert results[1].success is False

    async def test_run_multiple_no_stop_on_failure(self, fake_target: FakeTarget):
        fake_target._results = [
            ExecutionResult(
                status=ExecutionStatus.PASSED, exit_code=0,
                stdout="ok", stderr="", duration_seconds=0.1,
                command_display="cmd1",
            ),
            ExecutionResult(
                status=ExecutionStatus.FAILED, exit_code=1,
                stdout="", stderr="fail", duration_seconds=0.1,
                command_display="cmd2",
            ),
            ExecutionResult(
                status=ExecutionStatus.PASSED, exit_code=0,
                stdout="ok", stderr="", duration_seconds=0.1,
                command_display="cmd3",
            ),
        ]
        executor = ScriptExecutor(target=fake_target)
        results = await executor.run_multiple(["cmd1", "cmd2", "cmd3"])
        assert len(results) == 3  # all ran despite failure

    async def test_run_multiple_with_script_commands(self, executor: ScriptExecutor):
        cmds = [
            ScriptCommand(raw_command="pytest -v"),
            "echo hello",
            ScriptCommand(raw_command="make test"),
        ]
        results = await executor.run_multiple(cmds)
        assert len(results) == 3

    # -- Summary ------------------------------------------------------------

    async def test_summary(self, fake_target: FakeTarget):
        fake_target._results = [
            ExecutionResult(
                status=ExecutionStatus.PASSED, exit_code=0,
                stdout="ok", stderr="", duration_seconds=1.0,
                command_display="cmd1",
            ),
            ExecutionResult(
                status=ExecutionStatus.FAILED, exit_code=1,
                stdout="", stderr="err", duration_seconds=2.0,
                command_display="cmd2",
            ),
            ExecutionResult(
                status=ExecutionStatus.TIMEOUT, exit_code=-1,
                stdout="", stderr="timeout", duration_seconds=30.0,
                command_display="cmd3",
            ),
        ]
        executor = ScriptExecutor(target=fake_target)
        await executor.run_multiple(["cmd1", "cmd2", "cmd3"])

        s = executor.summary()
        assert s["total_commands"] == 3
        assert s["passed"] == 1
        assert s["failed"] == 2
        assert s["timed_out"] == 1
        assert s["total_duration_seconds"] == pytest.approx(33.0, abs=0.1)
        assert len(s["commands"]) == 3

    # -- Target resolution --------------------------------------------------

    async def test_target_registry_resolution(self, fake_target: FakeTarget):
        registry = TargetRegistry()
        registry.register(fake_target)
        executor = ScriptExecutor(target_registry=registry)
        result = await executor.run("test", target_name="fake")
        assert result.success is True
        assert len(fake_target.calls) == 1

    async def test_explicit_target_takes_precedence(self, fake_target: FakeTarget):
        other_target = FakeTarget()
        registry = TargetRegistry()
        registry.register(other_target)
        # Explicit target should win over registry
        executor = ScriptExecutor(target=fake_target, target_registry=registry)
        await executor.run("test", target_name="fake")
        assert len(fake_target.calls) == 1
        assert len(other_target.calls) == 0

    # -- Script type classification in run() --------------------------------

    async def test_run_classifies_script_type(self, executor: ScriptExecutor):
        result = await executor.run("make test")
        assert result.script_command is not None
        assert result.script_command.script_type == ScriptType.MAKE_TARGET

    async def test_run_classifies_python_script(self, executor: ScriptExecutor):
        result = await executor.run("python test.py")
        assert result.script_command is not None
        assert result.script_command.script_type == ScriptType.PYTHON_SCRIPT


# ---------------------------------------------------------------------------
# Integration-style test with real LocalTarget
# ---------------------------------------------------------------------------


class TestScriptExecutorIntegration:
    """Integration tests using the real LocalTarget (subprocess)."""

    async def test_run_echo_captures_output(self):
        executor = ScriptExecutor(target=LocalTarget())
        result = await executor.run("echo hello_world")
        assert result.success is True
        assert result.exit_code == 0
        assert "hello_world" in result.stdout

    async def test_run_false_command_fails(self):
        executor = ScriptExecutor(target=LocalTarget())
        result = await executor.run("false")
        assert result.success is False
        assert result.exit_code != 0

    async def test_run_with_env_var(self):
        executor = ScriptExecutor(target=LocalTarget())
        result = await executor.run(
            "printenv MY_TEST_VAR",
            env={"MY_TEST_VAR": "test_value_42"},
        )
        assert result.success is True
        assert "test_value_42" in result.stdout

    async def test_run_shell_mode_pipe(self):
        executor = ScriptExecutor(target=LocalTarget())
        result = await executor.run(
            "echo 'hello world' | tr 'h' 'H'",
            shell_mode=True,
        )
        assert result.success is True
        assert "Hello" in result.stdout

    async def test_run_captures_stderr(self):
        executor = ScriptExecutor(target=LocalTarget())
        result = await executor.run(
            "echo error_output >&2",
            shell_mode=True,
        )
        assert "error_output" in result.stderr

    async def test_run_nonexistent_command(self):
        executor = ScriptExecutor(target=LocalTarget())
        result = await executor.run("nonexistent_command_xyz_12345")
        assert result.success is False
        assert result.status == ExecutionStatus.ERROR

    async def test_run_with_working_directory(self):
        executor = ScriptExecutor(target=LocalTarget())
        result = await executor.run("pwd", working_directory="/tmp")
        assert result.success is True
        # /tmp might resolve to /private/tmp on macOS
        assert "tmp" in result.stdout.lower()

    async def test_timeout_kills_process(self):
        executor = ScriptExecutor(target=LocalTarget())
        result = await executor.run("sleep 60", timeout=1)
        assert result.timed_out is True
        assert result.success is False
        assert result.status == ExecutionStatus.TIMEOUT

    async def test_run_python_script_inline(self):
        executor = ScriptExecutor(target=LocalTarget())
        result = await executor.run(
            "python3 -c \"print('test_output_123')\"",
        )
        assert result.success is True
        assert "test_output_123" in result.stdout

    async def test_run_multiple_integration(self):
        executor = ScriptExecutor(target=LocalTarget())
        results = await executor.run_multiple(
            ["echo first", "echo second", "echo third"],
        )
        assert len(results) == 3
        assert all(r.success for r in results)
        assert "first" in results[0].stdout
        assert "second" in results[1].stdout
        assert "third" in results[2].stdout

    async def test_summary_integration(self):
        executor = ScriptExecutor(target=LocalTarget())
        await executor.run_multiple(["echo ok", "false"])
        s = executor.summary()
        assert s["total_commands"] == 2
        assert s["passed"] == 1
        assert s["failed"] == 1
