"""Command translator: maps ParsedTestRequest into executable test commands.

Translates structured test intent (framework, scope, extra args, etc.) into
concrete CLI commands for known frameworks (pytest, unittest, jest, etc.) and
arbitrary scripts. Uses a pluggable strategy pattern — each framework has a
registered ``FrameworkTranslator`` that knows how to build commands for that
framework's CLI.

Usage:
    translator = CommandTranslator()
    commands = translator.translate(parsed_request)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from test_runner.agents.parser import ParsedTestRequest, TestFramework, TestIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestCommand:
    """A single executable test command ready for an execution target.

    Attributes:
        command: The full CLI command as a list of tokens (e.g. ["pytest", "-v"]).
        display: Human-readable string representation of the command.
        framework: The framework this command belongs to.
        working_directory: Optional explicit working directory override.
        env: Optional extra environment variables to inject.
        timeout: Optional per-command timeout in seconds.
        metadata: Any extra info the execution target might need.
    """

    command: list[str]
    display: str
    framework: TestFramework
    working_directory: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shell_string(self) -> str:
        """Return the command as a single shell-safe string."""
        import shlex
        return " ".join(shlex.quote(tok) for tok in self.command)


@dataclass(frozen=True)
class TranslationResult:
    """Result of translating a ParsedTestRequest into commands.

    Attributes:
        commands: Ordered list of commands to execute.
        warnings: Any non-fatal issues noticed during translation.
        source_request: The original ParsedTestRequest that was translated.
    """

    commands: list[TestCommand]
    warnings: list[str] = field(default_factory=list)
    source_request: ParsedTestRequest | None = None


# ---------------------------------------------------------------------------
# Framework translator interface (strategy pattern)
# ---------------------------------------------------------------------------


class FrameworkTranslator(ABC):
    """Abstract strategy for translating test intent into commands for one framework."""

    @property
    @abstractmethod
    def framework(self) -> TestFramework:
        """The framework this translator handles."""
        ...

    @abstractmethod
    def build_run(self, request: ParsedTestRequest) -> list[str]:
        """Build command tokens for running tests."""
        ...

    @abstractmethod
    def build_list(self, request: ParsedTestRequest) -> list[str]:
        """Build command tokens for listing / discovering tests."""
        ...

    def build_rerun_failed(self, request: ParsedTestRequest) -> list[str]:
        """Build command tokens for re-running only failed tests.

        Default: fall back to a normal run (subclasses can override).
        """
        return self.build_run(request)

    def build_run_specific(self, request: ParsedTestRequest) -> list[str]:
        """Build command tokens for running a specific named test.

        Default: treat scope as a direct selector for a normal run.
        """
        return self.build_run(request)


# ---------------------------------------------------------------------------
# Concrete framework translators
# ---------------------------------------------------------------------------


class PytestTranslator(FrameworkTranslator):
    """Translates test intent into ``pytest`` commands."""

    @property
    def framework(self) -> TestFramework:
        return TestFramework.PYTEST

    def build_run(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["pytest"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_list(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["pytest", "--collect-only", "-q"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_rerun_failed(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["pytest", "--lf"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["pytest"]
        if req.scope:
            # pytest accepts node IDs like file::class::test or -k patterns
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd


class UnittestTranslator(FrameworkTranslator):
    """Translates test intent into ``python -m unittest`` commands."""

    @property
    def framework(self) -> TestFramework:
        return TestFramework.UNITTEST

    def build_run(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["python", "-m", "unittest"]
        if req.scope:
            cmd.append(req.scope)
        else:
            cmd.append("discover")
        cmd.extend(req.extra_args)
        return cmd

    def build_list(self, req: ParsedTestRequest) -> list[str]:
        # unittest doesn't have a native list mode; use discover with verbose dry run
        cmd = ["python", "-m", "unittest", "discover", "-v"]
        if req.scope:
            cmd.extend(["-s", req.scope])
        cmd.extend(req.extra_args)
        return cmd

    def build_rerun_failed(self, req: ParsedTestRequest) -> list[str]:
        # unittest has no built-in rerun-failed; fall back to normal run
        return self.build_run(req)

    def build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["python", "-m", "unittest"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd


class JestTranslator(FrameworkTranslator):
    """Translates test intent into ``npx jest`` commands."""

    @property
    def framework(self) -> TestFramework:
        return TestFramework.JEST

    def build_run(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["npx", "jest"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_list(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["npx", "jest", "--listTests"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_rerun_failed(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["npx", "jest", "--onlyFailures"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["npx", "jest"]
        if req.scope:
            cmd.extend(["-t", req.scope])
        cmd.extend(req.extra_args)
        return cmd


class MochaTranslator(FrameworkTranslator):
    """Translates test intent into ``npx mocha`` commands."""

    @property
    def framework(self) -> TestFramework:
        return TestFramework.MOCHA

    def build_run(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["npx", "mocha"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_list(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["npx", "mocha", "--dry-run"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["npx", "mocha"]
        if req.scope:
            cmd.extend(["--grep", req.scope])
        cmd.extend(req.extra_args)
        return cmd


class GoTestTranslator(FrameworkTranslator):
    """Translates test intent into ``go test`` commands."""

    @property
    def framework(self) -> TestFramework:
        return TestFramework.GO_TEST

    def build_run(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["go", "test"]
        if req.scope:
            cmd.append(req.scope)
        else:
            cmd.append("./...")
        cmd.extend(req.extra_args)
        return cmd

    def build_list(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["go", "test", "-list", ".*"]
        if req.scope:
            cmd.append(req.scope)
        else:
            cmd.append("./...")
        cmd.extend(req.extra_args)
        return cmd

    def build_rerun_failed(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["go", "test", "-run"]
        if req.scope:
            cmd.append(req.scope)
        else:
            cmd.append(".*")
        cmd.extend(req.extra_args)
        return cmd

    def build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["go", "test", "-run"]
        if req.scope:
            cmd.append(req.scope)
        else:
            cmd.append(".*")
        cmd.extend(req.extra_args)
        return cmd


class CargoTestTranslator(FrameworkTranslator):
    """Translates test intent into ``cargo test`` commands."""

    @property
    def framework(self) -> TestFramework:
        return TestFramework.CARGO_TEST

    def build_run(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["cargo", "test"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_list(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["cargo", "test"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(["--", "--list"])
        cmd.extend(req.extra_args)
        return cmd

    def build_rerun_failed(self, req: ParsedTestRequest) -> list[str]:
        # Cargo doesn't have native rerun-failed; use scope as filter
        return self.build_run(req)

    def build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["cargo", "test"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd


class DotnetTestTranslator(FrameworkTranslator):
    """Translates test intent into ``dotnet test`` commands."""

    @property
    def framework(self) -> TestFramework:
        return TestFramework.DOTNET_TEST

    def build_run(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["dotnet", "test"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_list(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["dotnet", "test", "--list-tests"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd

    def build_rerun_failed(self, req: ParsedTestRequest) -> list[str]:
        # dotnet test doesn't have native rerun-failed
        return self.build_run(req)

    def build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        cmd = ["dotnet", "test", "--filter"]
        if req.scope:
            cmd.append(req.scope)
        cmd.extend(req.extra_args)
        return cmd


class ScriptTranslator(FrameworkTranslator):
    """Translates test intent into arbitrary script/command execution.

    The *scope* field is treated as the script path or command string.
    """

    @property
    def framework(self) -> TestFramework:
        return TestFramework.SCRIPT

    def build_run(self, req: ParsedTestRequest) -> list[str]:
        if req.scope:
            cmd = [req.scope]
        else:
            cmd = ["bash", "-c", "echo 'No script specified'"]
        cmd.extend(req.extra_args)
        return cmd

    def build_list(self, req: ParsedTestRequest) -> list[str]:
        # For scripts, listing is the same as a dry-run echo
        return ["echo", "Script target:", req.scope or "(none)"]

    def build_run_specific(self, req: ParsedTestRequest) -> list[str]:
        return self.build_run(req)


# ---------------------------------------------------------------------------
# Main command translator
# ---------------------------------------------------------------------------


class CommandTranslator:
    """Translates ``ParsedTestRequest`` objects into executable ``TestCommand`` lists.

    Maintains a pluggable registry of ``FrameworkTranslator`` strategies.
    Custom translators can be registered at runtime via ``register()``.

    Example::

        translator = CommandTranslator()
        result = translator.translate(parsed_request)
        for cmd in result.commands:
            print(cmd.shell_string)
    """

    # Default translators instantiated once per class
    _DEFAULT_TRANSLATORS: ClassVar[list[type[FrameworkTranslator]]] = [
        PytestTranslator,
        UnittestTranslator,
        JestTranslator,
        MochaTranslator,
        GoTestTranslator,
        CargoTestTranslator,
        DotnetTestTranslator,
        ScriptTranslator,
    ]

    def __init__(self) -> None:
        self._registry: dict[TestFramework, FrameworkTranslator] = {}
        # Register all built-in translators
        for cls in self._DEFAULT_TRANSLATORS:
            instance = cls()
            self._registry[instance.framework] = instance

    # -- Registry management -------------------------------------------------

    def register(self, translator: FrameworkTranslator) -> None:
        """Register (or override) a framework translator.

        Args:
            translator: An instance implementing ``FrameworkTranslator``.
        """
        logger.info("Registering translator for %s", translator.framework.value)
        self._registry[translator.framework] = translator

    def unregister(self, framework: TestFramework) -> None:
        """Remove a framework translator from the registry.

        Args:
            framework: The framework whose translator should be removed.

        Raises:
            KeyError: If the framework is not registered.
        """
        del self._registry[framework]

    @property
    def supported_frameworks(self) -> list[TestFramework]:
        """Return list of currently registered frameworks."""
        return list(self._registry.keys())

    def get_translator(self, framework: TestFramework) -> FrameworkTranslator | None:
        """Look up the translator for a given framework."""
        return self._registry.get(framework)

    # -- Translation ---------------------------------------------------------

    def translate(
        self,
        request: ParsedTestRequest,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> TranslationResult:
        """Translate a parsed test request into executable commands.

        Args:
            request: The structured test request from the parser.
            timeout: Optional per-command timeout override (seconds).
            env: Optional extra environment variables to inject.

        Returns:
            A ``TranslationResult`` containing the commands and any warnings.

        Raises:
            UnsupportedFrameworkError: If the framework has no registered translator
                and is not AUTO_DETECT or UNKNOWN.
        """
        warnings: list[str] = []
        framework = request.framework

        # Handle auto-detect: fall back to pytest as sensible default
        if framework == TestFramework.AUTO_DETECT:
            framework = TestFramework.PYTEST
            warnings.append(
                "Framework auto-detected as pytest (default). "
                "Discovery agent should refine this."
            )

        # Handle unknown framework
        if framework == TestFramework.UNKNOWN:
            raise UnsupportedFrameworkError(
                f"Cannot translate request with unknown framework. "
                f"Raw request: {request.raw_request!r}"
            )

        translator = self._registry.get(framework)
        if translator is None:
            raise UnsupportedFrameworkError(
                f"No translator registered for framework {framework.value!r}"
            )

        # Dispatch to the right builder based on intent
        tokens = self._dispatch_intent(translator, request)

        command = TestCommand(
            command=tokens,
            display=" ".join(tokens),
            framework=framework,
            working_directory=request.working_directory,
            env=env or {},
            timeout=timeout,
            metadata={
                "intent": request.intent.value,
                "scope": request.scope,
                "confidence": request.confidence,
            },
        )

        logger.info(
            "Translated → %s (framework=%s, intent=%s)",
            command.display,
            framework.value,
            request.intent.value,
        )

        return TranslationResult(
            commands=[command],
            warnings=warnings,
            source_request=request,
        )

    @staticmethod
    def _dispatch_intent(
        translator: FrameworkTranslator,
        request: ParsedTestRequest,
    ) -> list[str]:
        """Call the appropriate builder method based on test intent."""
        match request.intent:
            case TestIntent.RUN:
                return translator.build_run(request)
            case TestIntent.LIST:
                return translator.build_list(request)
            case TestIntent.RERUN_FAILED:
                return translator.build_rerun_failed(request)
            case TestIntent.RUN_SPECIFIC:
                return translator.build_run_specific(request)
            case TestIntent.UNKNOWN:
                # Best-effort: treat as a run
                return translator.build_run(request)
            case _:
                return translator.build_run(request)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


    def translate_batch(
        self,
        requests: list[ParsedTestRequest],
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> TranslationResult:
        """Translate multiple parsed test requests into a single result.

        Each request is translated independently and all resulting commands are
        collected into one ``TranslationResult``. Useful when the orchestrator
        needs to run tests across multiple frameworks or scopes from a single
        user request.

        Args:
            requests: List of structured test requests from the parser.
            timeout: Optional per-command timeout override (seconds).
            env: Optional extra environment variables to inject.

        Returns:
            A ``TranslationResult`` containing all commands and any warnings.
        """
        all_commands: list[TestCommand] = []
        all_warnings: list[str] = []

        for request in requests:
            try:
                result = self.translate(request, timeout=timeout, env=env)
                all_commands.extend(result.commands)
                all_warnings.extend(result.warnings)
            except UnsupportedFrameworkError as exc:
                all_warnings.append(
                    f"Skipped unsupported framework: {exc}"
                )

        return TranslationResult(
            commands=all_commands,
            warnings=all_warnings,
            source_request=requests[0] if requests else None,
        )

    @staticmethod
    def validate_command(command: TestCommand) -> list[str]:
        """Validate a translated command for safety before execution.

        Checks for potentially dangerous patterns in commands, particularly
        for arbitrary script execution. Returns a list of warning strings;
        an empty list means the command passes validation.

        Args:
            command: The test command to validate.

        Returns:
            List of validation warnings (empty if clean).
        """
        warnings: list[str] = []
        tokens = command.command

        if not tokens:
            warnings.append("Empty command — nothing to execute.")
            return warnings

        # Shell metacharacter injection check (for individual tokens)
        _DANGEROUS_PATTERNS = re.compile(
            r"[;&|`$]|>\s*/"          # shell operators, redirects to root
            r"|rm\s+-rf"              # destructive commands
            r"|mkfs|dd\s+if="         # disk-level destruction
            r"|chmod\s+777"           # overly permissive
            r"|curl\s+.*\|\s*(?:ba)?sh"  # pipe-to-shell
        )

        full_cmd = " ".join(tokens)
        if _DANGEROUS_PATTERNS.search(full_cmd):
            warnings.append(
                f"Command contains potentially dangerous pattern: {command.display!r}"
            )

        # Script translator specific: warn if scope looks like a URL
        if command.framework == TestFramework.SCRIPT:
            for tok in tokens:
                if tok.startswith(("http://", "https://", "ftp://")):
                    warnings.append(
                        f"Script scope contains URL — verify this is intended: {tok!r}"
                    )

        return warnings

    @staticmethod
    def inject_verbose(command: TestCommand) -> TestCommand:
        """Return a copy of the command with verbose flags injected.

        Adds the appropriate verbosity flag for the framework if not
        already present. Useful for debugging or detailed reporting.

        Args:
            command: The original test command.

        Returns:
            A new TestCommand with verbose flag added, or the original
            if verbose is already present or not supported.
        """
        _VERBOSE_FLAGS: dict[TestFramework, str] = {
            TestFramework.PYTEST: "-v",
            TestFramework.JEST: "--verbose",
            TestFramework.MOCHA: "--reporter=spec",
            TestFramework.GO_TEST: "-v",
            TestFramework.DOTNET_TEST: "--verbosity=detailed",
        }

        # For cargo test, the verbose flag is --nocapture after --
        if command.framework == TestFramework.CARGO_TEST:
            if "--nocapture" in command.command:
                return command
            new_tokens = list(command.command)
            if "--" in new_tokens:
                idx = new_tokens.index("--")
                new_tokens.insert(idx + 1, "--nocapture")
            else:
                new_tokens.extend(["--", "--nocapture"])
            return TestCommand(
                command=new_tokens,
                display=" ".join(new_tokens),
                framework=command.framework,
                working_directory=command.working_directory,
                env=command.env,
                timeout=command.timeout,
                metadata=command.metadata,
            )

        flag = _VERBOSE_FLAGS.get(command.framework)
        if flag is None:
            return command
        if flag in command.command:
            return command

        new_tokens = list(command.command)
        # Insert verbose flag after the base command
        insert_pos = 1
        # For compound commands like "python -m unittest", skip past the base
        if len(new_tokens) >= 3 and new_tokens[:2] == ["python", "-m"]:
            insert_pos = 3
        elif len(new_tokens) >= 2 and new_tokens[0] == "npx":
            insert_pos = 2
        elif len(new_tokens) >= 2 and new_tokens[:2] == ["go", "test"]:
            insert_pos = 2
        elif len(new_tokens) >= 2 and new_tokens[:2] == ["dotnet", "test"]:
            insert_pos = 2

        new_tokens.insert(insert_pos, flag)
        return TestCommand(
            command=new_tokens,
            display=" ".join(new_tokens),
            framework=command.framework,
            working_directory=command.working_directory,
            env=command.env,
            timeout=command.timeout,
            metadata=command.metadata,
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnsupportedFrameworkError(Exception):
    """Raised when no translator is available for the requested framework."""


class CommandValidationError(Exception):
    """Raised when a command fails safety validation."""
