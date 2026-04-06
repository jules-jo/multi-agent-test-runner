"""Arbitrary script/command executor with output capture and exit code handling.

Provides a high-level interface for running user-specified test commands
(arbitrary scripts, shell commands, or custom test invocations) that don't
map to a known framework. Handles:

- Shell string parsing into safe command tokens
- Output capture (stdout + stderr, both separate and combined)
- Exit code interpretation with configurable success codes
- Working directory and environment variable injection
- Timeout enforcement
- Streaming output callback support

This module sits between raw ``ExecutionTarget.execute()`` and the full
``TaskExecutor`` pipeline, providing a convenient API for the executor
agent to run arbitrary commands.

Usage::

    script_exec = ScriptExecutor()
    result = await script_exec.run("python tests/my_script.py --verbose")
    print(result.exit_code, result.stdout)

    # Or with full control:
    result = await script_exec.run_command(
        ScriptCommand(
            raw_command="./run_tests.sh integration",
            working_directory="/path/to/project",
            timeout=120,
            env={"CI": "true"},
        )
    )
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from test_runner.agents.parser import TestFramework, TestIntent
from test_runner.execution.command_translator import TestCommand
from test_runner.execution.targets import (
    ExecutionResult,
    ExecutionStatus,
    ExecutionTarget,
    LocalTarget,
    TargetRegistry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Script command model
# ---------------------------------------------------------------------------


class ScriptType(str, Enum):
    """Classification of the script/command being run."""

    SHELL_COMMAND = "shell_command"    # Direct shell command (e.g., "pytest -v")
    SCRIPT_FILE = "script_file"       # Executable script file (e.g., "./run.sh")
    PYTHON_SCRIPT = "python_script"   # Python script (e.g., "python test.py")
    INLINE_SHELL = "inline_shell"     # Inline shell via bash -c
    MAKE_TARGET = "make_target"       # Makefile target (e.g., "make test")
    NPM_SCRIPT = "npm_script"        # npm/yarn script (e.g., "npm test")


@dataclass(frozen=True)
class ScriptCommand:
    """Represents an arbitrary user-specified command to execute.

    Attributes:
        raw_command: The original command string as the user typed it.
        tokens: Parsed command tokens (auto-derived from raw_command if empty).
        working_directory: Directory to run the command in.
        env: Extra environment variables to inject.
        timeout: Timeout in seconds (None = no timeout).
        success_exit_codes: Set of exit codes considered successful (default: {0}).
        script_type: Classification of what kind of script this is.
        capture_combined: If True, merge stderr into stdout for combined output.
        shell_mode: If True, run via shell instead of exec (for pipes, redirects).
        metadata: Extra context for reporting.
    """

    raw_command: str
    tokens: tuple[str, ...] = ()
    working_directory: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int | None = None
    success_exit_codes: frozenset[int] = frozenset({0})
    script_type: ScriptType = ScriptType.SHELL_COMMAND
    capture_combined: bool = False
    shell_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-derive tokens from raw_command if not provided
        if not self.tokens and self.raw_command:
            try:
                parsed = shlex.split(self.raw_command)
            except ValueError:
                # If shlex fails (unmatched quotes etc.), split naively
                parsed = self.raw_command.split()
            object.__setattr__(self, "tokens", tuple(parsed))

    @property
    def command_list(self) -> list[str]:
        """Return tokens as a mutable list (for ExecutionTarget)."""
        return list(self.tokens)

    @property
    def display(self) -> str:
        """Human-readable display of the command."""
        return self.raw_command or " ".join(self.tokens)

    def to_test_command(self) -> TestCommand:
        """Convert to a TestCommand for use with TaskExecutor pipeline."""
        return TestCommand(
            command=self.command_list,
            display=self.display,
            framework=TestFramework.SCRIPT,
            working_directory=self.working_directory,
            env=dict(self.env),
            timeout=self.timeout,
            metadata={
                "script_type": self.script_type.value,
                "success_exit_codes": list(self.success_exit_codes),
                **self.metadata,
            },
        )


# ---------------------------------------------------------------------------
# Script result with enhanced output handling
# ---------------------------------------------------------------------------


@dataclass
class ScriptResult:
    """Enhanced result from running an arbitrary script/command.

    Extends the base ExecutionResult concept with script-specific details
    like combined output, success code interpretation, and structured
    output parsing hints.

    Attributes:
        exit_code: Process exit code.
        stdout: Captured standard output.
        stderr: Captured standard error.
        combined_output: Interleaved stdout+stderr (if capture_combined was True).
        duration_seconds: Wall-clock execution time.
        success: Whether the exit code is in the success set.
        status: Execution status classification.
        command_display: The command that was run.
        timed_out: Whether the command was killed due to timeout.
        error_message: Human-readable error if something went wrong.
        output_lines: Stdout split into lines (convenience).
        error_lines: Stderr split into lines (convenience).
        script_command: The original ScriptCommand that produced this result.
    """

    exit_code: int
    stdout: str
    stderr: str
    combined_output: str
    duration_seconds: float
    success: bool
    status: ExecutionStatus
    command_display: str
    timed_out: bool = False
    error_message: str = ""
    script_command: ScriptCommand | None = None

    @property
    def output_lines(self) -> list[str]:
        """Stdout split into individual lines."""
        return self.stdout.splitlines() if self.stdout else []

    @property
    def error_lines(self) -> list[str]:
        """Stderr split into individual lines."""
        return self.stderr.splitlines() if self.stderr else []

    @property
    def output_summary(self) -> str:
        """Brief summary of the output (first and last few lines)."""
        lines = self.output_lines
        if len(lines) <= 10:
            return self.stdout
        head = "\n".join(lines[:5])
        tail = "\n".join(lines[-5:])
        return f"{head}\n... ({len(lines) - 10} lines omitted) ...\n{tail}"

    @property
    def has_output(self) -> bool:
        """Whether the command produced any output."""
        return bool(self.stdout.strip() or self.stderr.strip())

    def to_execution_result(self) -> ExecutionResult:
        """Convert to a base ExecutionResult for the executor pipeline."""
        return ExecutionResult(
            status=self.status,
            exit_code=self.exit_code,
            stdout=self.stdout,
            stderr=self.stderr,
            duration_seconds=self.duration_seconds,
            command_display=self.command_display,
            metadata={
                "timed_out": self.timed_out,
                "success_by_exit_code": self.success,
                "combined_output_length": len(self.combined_output),
            },
        )


# ---------------------------------------------------------------------------
# Script type classifier
# ---------------------------------------------------------------------------


def classify_command(raw_command: str) -> ScriptType:
    """Classify a raw command string into a ScriptType.

    Uses heuristics to determine what kind of script/command the user
    is trying to run.

    Args:
        raw_command: The raw command string.

    Returns:
        Best-guess ScriptType classification.
    """
    stripped = raw_command.strip()
    lower = stripped.lower()

    # Make targets
    if lower.startswith("make ") or lower == "make":
        return ScriptType.MAKE_TARGET

    # npm/yarn/pnpm scripts
    if re.match(r"^(npm|yarn|pnpm)\s+(run\s+)?", lower):
        return ScriptType.NPM_SCRIPT

    # Python scripts
    if lower.startswith("python ") or lower.startswith("python3 "):
        return ScriptType.PYTHON_SCRIPT

    # Script files
    if stripped.startswith("./") or stripped.startswith("/"):
        return ScriptType.SCRIPT_FILE
    if re.match(r"^[\w/.-]+\.(sh|bash|zsh|fish|pl|rb)(\s|$)", stripped):
        return ScriptType.SCRIPT_FILE

    # Inline shell (pipes, redirects, command chaining)
    if any(op in stripped for op in ("|", "&&", "||", ";", ">>", ">")):
        return ScriptType.INLINE_SHELL

    # Bash -c invocations
    if lower.startswith("bash ") or lower.startswith("sh "):
        return ScriptType.INLINE_SHELL

    return ScriptType.SHELL_COMMAND


# ---------------------------------------------------------------------------
# Main script executor
# ---------------------------------------------------------------------------


# Type alias for output streaming callbacks
OutputCallback = Callable[[str, str], None]  # (stream_name, line)


class ScriptExecutor:
    """Executes arbitrary user-specified commands with output capture.

    Provides a simple, high-level interface for running test scripts and
    commands. Delegates to pluggable ``ExecutionTarget`` instances for
    the actual process management.

    Features:
        - Parses shell command strings into safe tokens
        - Captures stdout and stderr (separately and combined)
        - Interprets exit codes with configurable success criteria
        - Supports working directory and env var injection
        - Enforces timeouts
        - Optionally streams output via callbacks
        - Classifies commands for better reporting

    Usage::

        executor = ScriptExecutor()
        result = await executor.run("pytest -v tests/")
        if result.success:
            print("Tests passed!")
        else:
            print(f"Exit code {result.exit_code}: {result.stderr}")
    """

    def __init__(
        self,
        *,
        target: ExecutionTarget | None = None,
        target_registry: TargetRegistry | None = None,
        default_timeout: int | None = None,
        default_success_codes: frozenset[int] | None = None,
        on_output: OutputCallback | None = None,
    ) -> None:
        """Initialize the script executor.

        Args:
            target: Explicit execution target to use.
            target_registry: Registry to look up targets by name.
            default_timeout: Default timeout for all commands (seconds).
            default_success_codes: Default set of exit codes considered success.
            on_output: Optional callback invoked for each output line.
        """
        self._target = target
        self._registry = target_registry or TargetRegistry()
        self._default_timeout = default_timeout
        self._default_success_codes = default_success_codes or frozenset({0})
        self._on_output = on_output
        self._history: list[ScriptResult] = []

    @property
    def history(self) -> list[ScriptResult]:
        """All results from commands run by this executor."""
        return list(self._history)

    def _resolve_target(self, target_name: str = "local") -> ExecutionTarget:
        """Resolve the execution target to use."""
        if self._target is not None:
            return self._target
        return self._registry.get(target_name) or self._registry.get_default()

    async def run(
        self,
        command: str,
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        success_exit_codes: frozenset[int] | None = None,
        target_name: str = "local",
        shell_mode: bool = False,
    ) -> ScriptResult:
        """Run an arbitrary command string with output capture.

        This is the simplest entry point — pass a command string and get
        back a result with captured output and exit code.

        Args:
            command: The command to run (will be parsed via shlex).
            working_directory: Directory to run in.
            env: Extra environment variables.
            timeout: Timeout in seconds (None uses default_timeout).
            success_exit_codes: Exit codes considered success.
            target_name: Name of execution target to use.
            shell_mode: If True, run via shell for pipes/redirects.

        Returns:
            ScriptResult with captured output and exit code.
        """
        script_type = classify_command(command)

        script_cmd = ScriptCommand(
            raw_command=command,
            working_directory=working_directory,
            env=env or {},
            timeout=timeout or self._default_timeout,
            success_exit_codes=success_exit_codes or self._default_success_codes,
            script_type=script_type,
            shell_mode=shell_mode,
        )

        return await self.run_command(script_cmd, target_name=target_name)

    async def run_command(
        self,
        script_cmd: ScriptCommand,
        *,
        target_name: str = "local",
    ) -> ScriptResult:
        """Run a ScriptCommand with full control over execution parameters.

        Args:
            script_cmd: The command specification.
            target_name: Name of execution target to use.

        Returns:
            ScriptResult with captured output and exit code.
        """
        target = self._resolve_target(target_name)
        command_tokens = self._prepare_tokens(script_cmd)

        logger.info(
            "Running script command: %s (type=%s, timeout=%s, cwd=%s)",
            script_cmd.display,
            script_cmd.script_type.value,
            script_cmd.timeout,
            script_cmd.working_directory or "(default)",
        )

        # Execute via the target
        exec_result = await target.execute(
            command_tokens,
            working_directory=script_cmd.working_directory,
            env=dict(script_cmd.env) if script_cmd.env else None,
            timeout=script_cmd.timeout,
        )

        # Interpret the result
        script_result = self._interpret_result(exec_result, script_cmd)

        # Fire output callbacks
        if self._on_output:
            self._fire_output_callbacks(script_result)

        # Record in history
        self._history.append(script_result)

        logger.info(
            "Script command complete: exit_code=%d success=%s duration=%.2fs",
            script_result.exit_code,
            script_result.success,
            script_result.duration_seconds,
        )

        return script_result

    async def run_multiple(
        self,
        commands: Sequence[str | ScriptCommand],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        target_name: str = "local",
        stop_on_failure: bool = False,
    ) -> list[ScriptResult]:
        """Run multiple commands sequentially.

        Args:
            commands: List of command strings or ScriptCommand objects.
            working_directory: Default working directory for string commands.
            env: Default env vars for string commands.
            timeout: Default timeout for string commands.
            target_name: Execution target name.
            stop_on_failure: If True, stop after the first failure.

        Returns:
            List of ScriptResults, one per command.
        """
        results: list[ScriptResult] = []

        for cmd in commands:
            if isinstance(cmd, str):
                result = await self.run(
                    cmd,
                    working_directory=working_directory,
                    env=env,
                    timeout=timeout,
                    target_name=target_name,
                )
            else:
                result = await self.run_command(cmd, target_name=target_name)

            results.append(result)

            if stop_on_failure and not result.success:
                logger.info(
                    "Stopping batch execution after failure: %s (exit_code=%d)",
                    result.command_display,
                    result.exit_code,
                )
                break

        return results

    def _prepare_tokens(self, script_cmd: ScriptCommand) -> list[str]:
        """Prepare command tokens for execution.

        Handles shell_mode by wrapping in bash -c, and ensures
        tokens are properly formed for the execution target.
        """
        if script_cmd.shell_mode:
            # Wrap in bash -c for shell features (pipes, redirects, etc.)
            return ["bash", "-c", script_cmd.raw_command]

        tokens = script_cmd.command_list
        if not tokens:
            return ["bash", "-c", "echo 'No command specified'; exit 1"]

        return tokens

    def _interpret_result(
        self,
        exec_result: ExecutionResult,
        script_cmd: ScriptCommand,
    ) -> ScriptResult:
        """Interpret an ExecutionResult into a ScriptResult.

        Applies custom success code logic and builds combined output.
        """
        # Determine success based on custom exit codes
        timed_out = exec_result.status == ExecutionStatus.TIMEOUT
        is_error = exec_result.status == ExecutionStatus.ERROR

        if timed_out or is_error:
            success = False
        else:
            success = exec_result.exit_code in script_cmd.success_exit_codes

        # Determine status
        if timed_out:
            status = ExecutionStatus.TIMEOUT
        elif is_error:
            status = ExecutionStatus.ERROR
        elif success:
            status = ExecutionStatus.PASSED
        else:
            status = ExecutionStatus.FAILED

        # Build combined output
        combined = exec_result.stdout
        if exec_result.stderr:
            if combined:
                combined += "\n"
            combined += exec_result.stderr

        # Build error message
        error_message = ""
        if timed_out:
            error_message = f"Command timed out after {script_cmd.timeout}s"
        elif is_error:
            error_message = exec_result.stderr or "Execution infrastructure error"
        elif not success:
            error_message = (
                f"Command exited with code {exec_result.exit_code} "
                f"(expected one of {sorted(script_cmd.success_exit_codes)})"
            )

        return ScriptResult(
            exit_code=exec_result.exit_code,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            combined_output=combined,
            duration_seconds=exec_result.duration_seconds,
            success=success,
            status=status,
            command_display=script_cmd.display,
            timed_out=timed_out,
            error_message=error_message,
            script_command=script_cmd,
        )

    def _fire_output_callbacks(self, result: ScriptResult) -> None:
        """Invoke output callbacks for each line of output."""
        if not self._on_output:
            return
        try:
            for line in result.output_lines:
                self._on_output("stdout", line)
            for line in result.error_lines:
                self._on_output("stderr", line)
        except Exception as exc:
            logger.warning("Output callback error: %s", exc)

    def summary(self) -> dict[str, Any]:
        """Generate a summary of all commands run by this executor."""
        total = len(self._history)
        passed = sum(1 for r in self._history if r.success)
        failed = total - passed
        total_duration = sum(r.duration_seconds for r in self._history)
        timed_out = sum(1 for r in self._history if r.timed_out)

        return {
            "total_commands": total,
            "passed": passed,
            "failed": failed,
            "timed_out": timed_out,
            "total_duration_seconds": total_duration,
            "commands": [
                {
                    "command": r.command_display,
                    "exit_code": r.exit_code,
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "timed_out": r.timed_out,
                    "output_lines": len(r.output_lines),
                    "error_lines": len(r.error_lines),
                }
                for r in self._history
            ],
        }
