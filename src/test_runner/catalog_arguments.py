"""Runtime argument resolution for catalog-backed test commands.

This layer keeps the catalog closed-world for *what* may run, while allowing
the runner to derive per-request CLI arguments by probing the saved command's
help output on the selected execution system.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import PurePath
from typing import Any

from openai import AsyncOpenAI

from test_runner.agents.parser import ParsedTestRequest
from test_runner.config import Config
from test_runner.execution.command_translator import TestCommand
from test_runner.execution.targets import (
    ExecutionStatus,
    LocalTarget,
    SSHTarget,
)

logger = logging.getLogger(__name__)

_SEMANTIC_HINT_SYSTEM_PROMPT = """\
You are a runtime argument planner for a closed-world test runner.

Your job is to read a user's latest request plus the probed CLI help for one
saved test command, then extract candidate argument values the user explicitly
requested or very strongly implied.

Rules:
1. Never invent new values.
2. Never invent command names or flags.
3. Return only label/value pairs that could plausibly map onto the probed CLI.
4. Use short human-readable labels that reflect the user's meaning or the CLI
   help text, such as "name", "display name", "iterations", or "mode".
5. If nothing concrete is present, return an empty hints list.

Return ONLY JSON in this format:
{
  "hints": [
    {"label": "<label>", "value": "<value>"}
  ]
}
"""


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
class RequiredParameter:
    """One required CLI parameter inferred from usage/help text."""

    kind: str
    label: str
    flags: tuple[str, ...] = ()
    metavar: str = ""


@dataclass(frozen=True)
class RuntimeArgumentResolution:
    """Result of mapping request-time parameters onto a saved command."""

    command: TestCommand | None
    warnings: tuple[str, ...] = ()
    needs_clarification: bool = False


@dataclass(frozen=True)
class PlannedHintResult:
    """Candidate value hints from a semantic planner."""

    hints: tuple[RequestedValueHint, ...] = ()
    warnings: tuple[str, ...] = ()


_VALUE_HINT_PATTERNS = (
    re.compile(
        r"\b(?P<label>[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})\s*"
        r"(?:is|are|=|:)\s*"
        r"(?P<value>\"[^\"]+\"|'[^']+'|[^\s,]+)"
        r"(?=(?:\s+and\b|\s*,|$))",
        flags=re.IGNORECASE,
    ),
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


def _normalize_hint_key(label: str, value: str) -> tuple[str, str]:
    return (" ".join(_normalize_tokens(label)), value.strip().lower())


def _strip_wrapping_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped


def _config_has_llm(config: Config | None) -> bool:
    if config is None:
        return False
    return bool(config.llm_base_url and config.api_key and config.model_id)


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_semantic_hint_response(raw: str) -> list[RequestedValueHint]:
    cleaned = _strip_code_fences(raw)
    if not cleaned:
        return []

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Could not parse semantic hint planner response: %s", raw[:200])
        return []

    raw_hints = data.get("hints", [])
    if not isinstance(raw_hints, list):
        return []

    hints: list[RequestedValueHint] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_hints:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = _strip_wrapping_quotes(str(item.get("value", "")).strip())
        if not label or not value:
            continue
        key = _normalize_hint_key(label, value)
        if key in seen:
            continue
        seen.add(key)
        hints.append(RequestedValueHint(label=label, value=value))
    return hints


class OpenAISemanticHintPlanner:
    """Use the configured OpenAI-compatible backend to extract runtime values."""

    def __init__(self, config: Config, *, timeout_seconds: int = 20) -> None:
        self._config = config
        self._timeout_seconds = timeout_seconds
        self._client = AsyncOpenAI(
            base_url=config.llm_base_url,
            api_key=config.api_key,
        )

    async def plan(
        self,
        request_text: str,
        *,
        alias: str,
        help_text: str,
        options: list[HelpOption],
        required_parameters: list[RequiredParameter],
        existing_hints: list[RequestedValueHint],
    ) -> PlannedHintResult:
        option_lines = [
            f"- {' / '.join(option.flags)} :: {option.description or '(no description)'}"
            for option in options
            if option.expects_value
        ]
        required_lines = [
            f"- {parameter.label} ({parameter.kind})"
            for parameter in required_parameters
        ]
        existing_hint_lines = [
            f"- {hint.label} = {hint.value}"
            for hint in existing_hints
        ]
        user_prompt = (
            f"Saved test alias: {alias}\n\n"
            f"Latest user request:\n{request_text}\n\n"
            "Existing extracted hints:\n"
            + ("\n".join(existing_hint_lines) if existing_hint_lines else "- none")
            + "\n\n"
            "Value-taking CLI options:\n"
            + ("\n".join(option_lines) if option_lines else "- none")
            + "\n\n"
            "Currently required parameters:\n"
            + ("\n".join(required_lines) if required_lines else "- none")
            + "\n\n"
            "Help excerpt:\n"
            + help_text[:6000]
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._config.model_id,
                messages=[
                    {"role": "system", "content": _SEMANTIC_HINT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            logger.warning("Semantic runtime-argument planning failed: %s", exc)
            return PlannedHintResult(
                warnings=(f"Semantic runtime-argument planning failed: {exc}",),
            )

        raw_content = response.choices[0].message.content or ""
        return PlannedHintResult(hints=tuple(_parse_semantic_hint_response(raw_content)))


class CatalogArgumentResolver:
    """Derive runtime CLI flags from a request by probing command help."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        help_timeout_seconds: int = 15,
        semantic_hint_planner: OpenAISemanticHintPlanner | Any | None = None,
    ) -> None:
        self._help_timeout_seconds = help_timeout_seconds
        if semantic_hint_planner is not None:
            self._semantic_hint_planner = semantic_hint_planner
        elif _config_has_llm(config):
            self._semantic_hint_planner = OpenAISemanticHintPlanner(config)
        else:
            self._semantic_hint_planner = None

    async def resolve(
        self,
        command: TestCommand,
        *,
        request: ParsedTestRequest,
    ) -> RuntimeArgumentResolution:
        """Return a command augmented with runtime-derived arguments."""
        hints = self._extract_value_hints(request.raw_request)
        help_text, help_warnings = await self._probe_help_text(command)
        if not help_text:
            if not hints:
                return RuntimeArgumentResolution(
                    command=command,
                    warnings=help_warnings,
                )
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
        if not options and hints:
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

        warnings = list(help_warnings)
        alias = str(command.metadata.get("catalog_alias", "saved test"))
        runtime_args: list[str] = []
        planned_positionals: dict[str, str] = {}
        provided_option_flags = {
            token
            for token in command.command[len(self._base_command_tokens(command)):]
            if token.startswith("-")
        }

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
            if option.primary_flag in provided_option_flags:
                continue
            runtime_args.extend([option.primary_flag, hint.value])
            provided_option_flags.add(option.primary_flag)

        required_parameters = self._parse_required_parameters(help_text, command, options)
        missing_parameters = self._find_missing_required_parameters(
            command,
            options=options,
            required=required_parameters,
            runtime_args=runtime_args,
        )
        if self._semantic_hint_planner is not None and (not hints or missing_parameters):
            planned = await self._semantic_hint_planner.plan(
                request.raw_request,
                alias=alias,
                help_text=help_text,
                options=options,
                required_parameters=missing_parameters or required_parameters,
                existing_hints=hints,
            )
            warnings.extend(planned.warnings)
            for hint in planned.hints:
                if _normalize_hint_key(hint.label, hint.value) in {
                    _normalize_hint_key(existing.label, existing.value)
                    for existing in hints
                }:
                    continue
                option = self._select_option(options, hint)
                if option is not None:
                    if option.primary_flag in provided_option_flags:
                        continue
                    runtime_args.extend([option.primary_flag, hint.value])
                    provided_option_flags.add(option.primary_flag)
                    continue

                positional = self._select_positional_parameter(
                    missing_parameters or required_parameters,
                    hint,
                    claimed_labels=set(planned_positionals),
                )
                if positional is not None:
                    planned_positionals[positional.label] = hint.value

        if planned_positionals:
            for parameter in required_parameters:
                if parameter.kind != "positional":
                    continue
                if parameter.label in planned_positionals:
                    runtime_args.append(planned_positionals[parameter.label])

        missing_parameters = self._find_missing_required_parameters(
            command,
            options=options,
            required=required_parameters,
            runtime_args=runtime_args,
        )
        if missing_parameters:
            warnings.append(
                f"Saved test {alias!r} requires additional arguments before it can run: "
                f"{', '.join(param.label for param in missing_parameters)}."
            )
            return RuntimeArgumentResolution(
                command=None,
                warnings=tuple(warnings),
                needs_clarification=True,
            )

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

    def _select_positional_parameter(
        self,
        parameters: list[RequiredParameter],
        hint: RequestedValueHint,
        *,
        claimed_labels: set[str],
    ) -> RequiredParameter | None:
        best: tuple[int, RequiredParameter] | None = None
        hint_tokens = set(hint.tokens)
        for parameter in parameters:
            if parameter.kind != "positional":
                continue
            if parameter.label in claimed_labels:
                continue
            parameter_tokens = set(
                _normalize_tokens(" ".join(filter(None, [parameter.label, parameter.metavar])))
            )
            score = len(hint_tokens & parameter_tokens)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, parameter)
        if best is None:
            return None
        return best[1]

    def _extract_value_hints(self, request: str) -> list[RequestedValueHint]:
        candidates: list[tuple[int, RequestedValueHint]] = []
        seen: set[tuple[str, str]] = set()
        for pattern in _VALUE_HINT_PATTERNS:
            for match in pattern.finditer(request):
                label = match.group("label").strip()
                value = _strip_wrapping_quotes(match.group("value"))
                key = _normalize_hint_key(label, value)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    (
                        match.start(),
                        RequestedValueHint(label=label, value=value),
                    )
                )
        candidates.sort(key=lambda item: item[0])
        return [hint for _, hint in candidates]

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

    def _base_command_tokens(self, command: TestCommand) -> list[str]:
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
            return [python_command, target]
        return [target]

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

    def _parse_required_parameters(
        self,
        help_text: str,
        command: TestCommand,
        options: list[HelpOption],
    ) -> list[RequiredParameter]:
        usage_text = self._extract_usage_text(help_text)
        if not usage_text:
            return []

        required_text = self._strip_optional_usage_segments(usage_text)
        if not required_text.strip():
            return []

        known_base_tokens = self._known_base_usage_tokens(command)
        option_lookup: dict[str, HelpOption] = {}
        for option in options:
            for flag in option.flags:
                option_lookup[flag] = option

        required: list[RequiredParameter] = []
        tokens = re.findall(r"[^\s]+", required_text)
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.lower() == "usage:":
                index += 1
                continue
            if token in known_base_tokens:
                index += 1
                continue
            if token.startswith("-"):
                option = option_lookup.get(token)
                label = token
                metavar = ""
                if option is not None:
                    long_flags = [flag for flag in option.flags if flag.startswith("--")]
                    if long_flags:
                        label = long_flags[0]
                    metavar = self._extract_option_metavar(token, usage_text)
                else:
                    metavar = self._extract_option_metavar(token, usage_text)
                required.append(
                    RequiredParameter(
                        kind="option",
                        label=label,
                        flags=option.flags if option is not None else (token,),
                        metavar=metavar,
                    )
                )
                if metavar and index + 1 < len(tokens) and tokens[index + 1] == metavar:
                    index += 1
                elif (
                    index + 1 < len(tokens)
                    and not tokens[index + 1].startswith("-")
                    and tokens[index + 1] not in known_base_tokens
                ):
                    index += 1
                index += 1
                continue

            positional = token.strip("<>")
            if positional and positional != "...":
                required.append(
                    RequiredParameter(
                        kind="positional",
                        label=positional,
                        metavar=positional,
                    )
                )
            index += 1
        return required

    def _find_missing_required_parameters(
        self,
        command: TestCommand,
        *,
        options: list[HelpOption],
        required: list[RequiredParameter],
        runtime_args: list[str],
    ) -> list[RequiredParameter]:
        if not required:
            return []

        provided_args = [
            *command.command[len(self._base_command_tokens(command)):],
            *runtime_args,
        ]
        value_flags = {
            flag
            for option in options
            if option.expects_value
            for flag in option.flags
        }

        provided_flags = {
            token
            for token in provided_args
            if token.startswith("-")
        }
        positional_values = self._extract_positional_values(
            provided_args,
            value_flags=value_flags,
        )

        missing: list[RequiredParameter] = []
        positional_index = 0
        for parameter in required:
            if parameter.kind == "option":
                if not any(flag in provided_flags for flag in parameter.flags):
                    missing.append(parameter)
                continue

            if positional_index >= len(positional_values):
                missing.append(parameter)
            else:
                positional_index += 1

        return missing

    def _extract_positional_values(
        self,
        args: list[str],
        *,
        value_flags: set[str],
    ) -> list[str]:
        positionals: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token.startswith("-"):
                if token in value_flags and index + 1 < len(args):
                    index += 2
                    continue
                index += 1
                continue
            positionals.append(token)
            index += 1
        return positionals

    def _extract_usage_text(self, help_text: str) -> str:
        lines = help_text.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.lower().startswith("usage:"):
                continue

            usage_parts = [stripped]
            cursor = index + 1
            while cursor < len(lines):
                continuation = lines[cursor]
                if not continuation.startswith(" "):
                    break
                continuation_stripped = continuation.strip()
                if not continuation_stripped:
                    break
                if continuation_stripped.startswith("-"):
                    break
                usage_parts.append(continuation_stripped)
                cursor += 1
            return " ".join(usage_parts)
        return ""

    def _strip_optional_usage_segments(self, usage_text: str) -> str:
        output: list[str] = []
        depth = 0
        for char in usage_text:
            if char == "[":
                depth += 1
                continue
            if char == "]":
                depth = max(0, depth - 1)
                continue
            if depth == 0:
                output.append(char)
        return "".join(output)

    def _known_base_usage_tokens(self, command: TestCommand) -> set[str]:
        base_tokens = self._base_command_tokens(command)
        known: set[str] = set()
        for token in base_tokens:
            known.add(token)
            known.add(PurePath(token).name)
        return {token for token in known if token}

    def _extract_option_metavar(self, flag: str, usage_text: str) -> str:
        match = re.search(
            rf"{re.escape(flag)}(?:[ =](<[^>]+>|\[[^\]]+\]|[A-Z][A-Z0-9_-]*|\w+))",
            usage_text,
        )
        if match is None:
            return ""
        return match.group(1).strip("<>[]")

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
