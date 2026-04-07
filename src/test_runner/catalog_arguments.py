"""Runtime argument resolution for catalog-backed test commands.

This layer keeps the catalog closed-world for *what* may run, while allowing
the runner to derive per-request CLI arguments by probing the saved command's
help output on the selected execution system.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from test_runner.agents.parser import ParsedTestRequest
from test_runner.execution.command_translator import TestCommand
from test_runner.execution.targets import (
    ExecutionStatus,
    LocalTarget,
    SSHTarget,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestedValueHint:
    """One runtime value requested by the user in natural language."""

    label: str
    value: str

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(_normalize_tokens(self.label))


@dataclass(frozen=True)
class HelpOption:
    """One CLI option parsed from a help screen."""

    flags: tuple[str, ...]
    description: str
    expects_value: bool

    @property
    def primary_flag(self) -> str:
        return self.flags[0]


@dataclass(frozen=True)
class RuntimeArgumentResolution:
    """Result of mapping request-time parameters onto a saved command."""

    command: TestCommand | None
    warnings: tuple[str, ...] = ()
    needs_clarification: bool = False


_VALUE_HINT_PATTERNS = (
    re.compile(
        r"\bfor\s+(?P<value>\d+)\s+(?P<label>[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bwith\s+(?P<value>\d+)\s+(?P<label>[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<value>\d+)\s+(?P<label>iterations?|loops?|retries?|attempts?|workers?|threads?|passes?|rounds?)\b",
        flags=re.IGNORECASE,
    ),
)


def _normalize_tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    normalized: list[str] = []
    for token in raw_tokens:
        if token.endswith("ies") and len(token) > 3:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        normalized.append(token)
    return normalized


class CatalogArgumentResolver:
    """Derive runtime CLI flags from a request by probing command help."""

    def __init__(self, *, help_timeout_seconds: int = 15) -> None:
        self._help_timeout_seconds = help_timeout_seconds

    async def resolve(
        self,
        command: TestCommand,
        *,
        request: ParsedTestRequest,
    ) -> RuntimeArgumentResolution:
        """Return a command augmented with runtime-derived arguments."""
        hints = self._extract_value_hints(request.raw_request)
        if not hints:
            return RuntimeArgumentResolution(command=command)

        help_text, help_warnings = await self._probe_help_text(command)
        if not help_text:
            alias = str(command.metadata.get("catalog_alias", "saved test"))
            warning = (
                f"Could not inspect supported arguments for {alias!r}; "
                "clarify the requested runtime parameters."
            )
            return RuntimeArgumentResolution(
                command=None,
                warnings=(*help_warnings, warning),
                needs_clarification=True,
            )

        options = self._parse_help_options(help_text)
        if not options:
            alias = str(command.metadata.get("catalog_alias", "saved test"))
            warning = (
                f"Help output for {alias!r} did not reveal any runnable options; "
                "clarify the requested runtime parameters."
            )
            return RuntimeArgumentResolution(
                command=None,
                warnings=(*help_warnings, warning),
                needs_clarification=True,
            )

        runtime_args: list[str] = []
        warnings = list(help_warnings)
        alias = str(command.metadata.get("catalog_alias", "saved test"))

        for hint in hints:
            option = self._select_option(options, hint)
            if option is None:
                warnings.append(
                    f"Could not map requested {hint.label!r} value for {alias!r} "
                    "onto a supported CLI option."
                )
                return RuntimeArgumentResolution(
                    command=None,
                    warnings=tuple(warnings),
                    needs_clarification=True,
                )
            runtime_args.extend([option.primary_flag, hint.value])

        metadata = dict(command.metadata)
        metadata["catalog_runtime_args"] = list(runtime_args)

        updated_command = replace(
            command,
            command=[*command.command, *runtime_args],
            display=" ".join([*command.command, *runtime_args]),
            metadata=metadata,
        )
        return RuntimeArgumentResolution(
            command=updated_command,
            warnings=tuple(warnings),
            needs_clarification=False,
        )

    def _extract_value_hints(self, request: str) -> list[RequestedValueHint]:
        hints: list[RequestedValueHint] = []
        seen: set[tuple[str, str]] = set()
        for pattern in _VALUE_HINT_PATTERNS:
            for match in pattern.finditer(request):
                label = match.group("label").strip()
                value = match.group("value").strip()
                key = (label.lower(), value)
                if key in seen:
                    continue
                seen.add(key)
                hints.append(RequestedValueHint(label=label, value=value))
        return hints

    async def _probe_help_text(self, command: TestCommand) -> tuple[str, tuple[str, ...]]:
        probe_commands = self._build_probe_commands(command)
        if not probe_commands:
            return "", ("Saved command does not expose enough metadata for help probing.",)

        target = self._build_probe_target(command)
        timeout = min(command.timeout or self._help_timeout_seconds, self._help_timeout_seconds)

        warnings: list[str] = []
        for probe in probe_commands:
            result = await target.execute(
                probe,
                working_directory=command.working_directory,
                env=command.env or None,
                timeout=timeout,
            )
            output = "\n".join(
                part for part in [result.stdout.strip(), result.stderr.strip()] if part
            ).strip()
            if output and result.status not in {ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT}:
                return output, tuple(warnings)
            if result.stderr:
                warnings.append(result.stderr.strip())

        return "", tuple(warnings)

    def _build_probe_commands(self, command: TestCommand) -> list[list[str]]:
        metadata = command.metadata or {}
        execution_type = str(metadata.get("catalog_execution_type", "")).strip().lower()
        target = str(metadata.get("catalog_target", "")).strip()
        if not target:
            return []

        system_config = metadata.get("catalog_system_config")
        python_command = "python"
        if isinstance(system_config, dict):
            python_command = str(system_config.get("python_command") or "python")

        if execution_type == "python_script":
            base = [python_command, target]
        else:
            base = [target]
        return [
            [*base, "--help"],
            [*base, "-h"],
        ]

    def _build_probe_target(self, command: TestCommand):
        metadata = command.metadata or {}
        if str(metadata.get("catalog_system_transport", "")).lower() == "ssh":
            system_config = metadata.get("catalog_system_config")
            if isinstance(system_config, dict):
                return SSHTarget.from_metadata(system_config)
        return LocalTarget()

    def _parse_help_options(self, help_text: str) -> list[HelpOption]:
        options: list[HelpOption] = []
        for line in help_text.splitlines():
            stripped = line.rstrip()
            if not stripped.lstrip().startswith("-"):
                continue

            match = re.split(r"\s{2,}|\t", stripped.strip(), maxsplit=1)
            option_segment = match[0]
            description = match[1].strip() if len(match) > 1 else ""

            flags = tuple(
                re.findall(r"--?[A-Za-z0-9][A-Za-z0-9_-]*", option_segment)
            )
            if not flags:
                continue

            expects_value = bool(
                re.search(
                    r"--?[A-Za-z0-9][A-Za-z0-9_-]*(?:[ =](?:<[^>]+>|\[[^\]]+\]|[A-Z][A-Z0-9_-]*|\w+))",
                    option_segment,
                )
            )
            options.append(
                HelpOption(
                    flags=flags,
                    description=description,
                    expects_value=expects_value,
                )
            )
        return options

    def _select_option(
        self,
        options: list[HelpOption],
        hint: RequestedValueHint,
    ) -> HelpOption | None:
        best: tuple[int, HelpOption] | None = None
        label_phrase = " ".join(hint.tokens)

        for option in options:
            if not option.expects_value:
                continue

            haystack_tokens = _normalize_tokens(
                " ".join([*option.flags, option.description])
            )
            score = 0
            for token in hint.tokens:
                if token in haystack_tokens:
                    score += 3
            haystack_phrase = " ".join(haystack_tokens)
            if label_phrase and label_phrase in haystack_phrase:
                score += 5

            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, option)

        if best is None:
            return None
        return best[1]
