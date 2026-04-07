"""CLI interface for the multi-agent test runner.

Supports three invocation modes:
  1. One-shot:    test-runner "run all unit tests in src/"
  2. Piped input: echo "run pytest" | test-runner
  3. Interactive:  test-runner -i  (launches REPL prompt)

Architecture supports future adapters (e.g., Teams bot via TS adapter).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import re
import shlex
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from test_runner import __version__
from test_runner.config import Config
from test_runner.agents.intent_service import IntentParserService, ParseMode
from test_runner.catalog import (
    CatalogEntry,
    CatalogExecutionType,
    CatalogRegistry,
    CatalogRepository,
    CatalogSystem,
    CatalogSystemAuthMethod,
    CatalogSystemTransport,
)
from test_runner.execution.factory import get_factory
from test_runner.orchestrator.hub import OrchestratorHub, RunPhase, RunState
from test_runner.reporting import (
    JSONSummaryReporter,
    MarkdownSummaryReporter,
    ReporterBase,
    create_cli_reporter,
)

logger = logging.getLogger(__name__)
console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

MIN_REQUEST_LENGTH = 3
MAX_REQUEST_LENGTH = 2000


class InputValidationError(Exception):
    """Raised when a natural-language request fails validation."""


def validate_request(text: str) -> str:
    """Validate and normalise a natural-language test request.

    Returns the cleaned string or raises ``InputValidationError``.
    """
    cleaned = text.strip()
    if not cleaned:
        raise InputValidationError("Request cannot be empty.")
    if len(cleaned) < MIN_REQUEST_LENGTH:
        raise InputValidationError(
            f"Request too short (minimum {MIN_REQUEST_LENGTH} characters)."
        )
    if len(cleaned) > MAX_REQUEST_LENGTH:
        raise InputValidationError(
            f"Request too long (maximum {MAX_REQUEST_LENGTH} characters)."
        )
    return cleaned


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="test-runner",
        description="Multi-agent test runner — run tests via natural language.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              test-runner "run all pytest tests in tests/"
              test-runner --working-dir /my/project "run unit tests"
              test-runner -i
              echo "run pytest -k test_login" | test-runner
        """),
    )

    parser.add_argument(
        "request",
        nargs="*",
        default=None,
        help="Natural-language test request (omit for interactive mode).",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        default=False,
        help="Launch interactive prompt mode.",
    )
    parser.add_argument(
        "-w", "--working-dir",
        default=None,
        help="Working directory for test execution (default: cwd).",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file (default: .env in cwd).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Execution timeout in seconds (overrides config).",
    )
    parser.add_argument(
        "--target",
        choices=["local", "docker", "ci"],
        default="local",
        help="Execution target (default: local).",
    )
    parser.add_argument(
        "--report-channel",
        action="append",
        default=None,
        help="Reporting channel(s): cli, json, markdown (can be repeated).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse and validate the request without executing tests.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level from config.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Shortcut for --log-level DEBUG.",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    """Print a startup banner."""
    banner = Text("Multi-Agent Test Runner", style="bold cyan")
    banner.append(f" v{__version__}", style="dim")
    console.print(Panel(banner, expand=False))


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

INTERACTIVE_BANNER = textwrap.dedent("""\
    Multi-Agent Test Runner (interactive)
    Type a test request or 'quit' to exit.
""")

EXIT_COMMANDS = frozenset({"quit", "exit", "q", ":q"})
GREETING_INPUTS = frozenset({
    "hello", "hi", "hey", "yo", "sup", "good morning", "good afternoon",
    "good evening",
})
THANKS_INPUTS = frozenset({"thanks", "thank you", "thx"})


class RequestDispatcher(Protocol):
    """Protocol for callables that handle a validated request."""

    def __call__(self, request: str) -> object: ...


@dataclass(frozen=True)
class CatalogCommand:
    """Parsed local catalog-management command."""

    resource: str
    action: str
    alias: str = ""


@dataclass
class InteractiveSessionState:
    """Per-session conversational state for the terminal REPL."""

    last_catalog_alias: str = ""
    last_system_alias: str = ""
    last_request: str = ""
    pending_clarification_aliases: tuple[str, ...] = ()
    pending_system_aliases: tuple[str, ...] = ()
    pending_system_entry_alias: str = ""


@dataclass(frozen=True)
class PreparedInteractiveRequest:
    """Session-aware request prepared for validation and execution."""

    canonical_request: str
    system_override: str = ""
    note: str = ""


def _normalize_frontdoor_text(text: str) -> str:
    """Normalize user input for lightweight front-door handling."""
    normalized = text.strip().lower()
    return normalized.strip("!.? ")


def _print_frontdoor_help() -> None:
    """Print a short usage summary for the terminal CLI."""
    console.print(
        "[cyan]This CLI is for test-running requests, for example:[/cyan]"
    )
    console.print("[dim]- run pytest tests/test_cli.py[/dim]")
    console.print("[dim]- run all tests[/dim]")
    console.print("[dim]- list pytest tests[/dim]")
    console.print("[dim]- rerun failed pytest tests[/dim]")
    console.print("[dim]- unknown saved tests can be registered interactively[/dim]")
    console.print("[dim]- list saved tests[/dim]")
    console.print("[dim]- show test lt[/dim]")
    console.print("[dim]- edit test lt[/dim]")
    console.print("[dim]- delete test lt[/dim]")
    console.print("[dim]- list systems[/dim]")
    console.print("[dim]- show system lab-a[/dim]")
    console.print("[dim]- edit system lab-a[/dim]")
    console.print("[dim]- delete system lab-a[/dim]")
    console.print("[dim]Use `quit` to leave interactive mode.[/dim]")


def _maybe_handle_frontdoor_input(text: str) -> bool:
    """Handle obvious greetings/help locally before validation or orchestration."""
    normalized = _normalize_frontdoor_text(text)
    if not normalized:
        return False

    if normalized in GREETING_INPUTS:
        console.print(
            "[cyan]This terminal is for test-runner requests. Type `help` for examples.[/cyan]"
        )
        return True

    if normalized in THANKS_INPUTS:
        console.print("[cyan]Type another test request or `quit` to exit.[/cyan]")
        return True

    if (
        normalized in {"help", "?", "usage", "what can you do", "commands"}
        or normalized.startswith("help ")
        or "what can you do" in normalized
        or "how do i use" in normalized
        or normalized == "examples"
    ):
        _print_frontdoor_help()
        return True

    return False


def _parse_catalog_command(text: str) -> CatalogCommand | None:
    """Parse deterministic local catalog-management commands."""
    stripped = text.strip()
    normalized = _normalize_frontdoor_text(stripped)

    if normalized in {
        "list saved tests",
        "list saved test aliases",
        "list catalog",
        "list catalog tests",
    }:
        return CatalogCommand(resource="entry", action="list")

    if normalized in {
        "list systems",
        "list saved systems",
        "list catalog systems",
    }:
        return CatalogCommand(resource="system", action="list")

    match = re.match(
        r"^(show|edit|delete|remove)\s+(?:saved\s+)?test\s+(.+?)\s*$",
        stripped,
        flags=re.IGNORECASE,
    )
    if match is not None:
        action = match.group(1).lower()
        alias = match.group(2).strip()
        if action == "remove":
            action = "delete"
        return CatalogCommand(resource="entry", action=action, alias=alias)

    match = re.match(
        r"^(show|edit|delete|remove)\s+(?:saved\s+)?system\s+(.+?)\s*$",
        stripped,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    action = match.group(1).lower()
    alias = match.group(2).strip()
    if action == "remove":
        action = "delete"
    return CatalogCommand(resource="system", action=action, alias=alias)


def _get_catalog_repository(config: Config) -> CatalogRepository | None:
    """Return the configured catalog repository, if any."""
    if not config.test_catalog_path:
        return None
    return CatalogRepository(config.test_catalog_path)


def _match_pending_alias_choice(
    request: str,
    aliases: tuple[str, ...],
) -> str | None:
    """Resolve a clarification reply to one saved alias."""
    if not aliases:
        return None

    candidate_text = re.sub(
        r"\s+on\s+.+$",
        "",
        request.strip(),
        flags=re.IGNORECASE,
    )
    normalized = _normalize_frontdoor_text(candidate_text)
    if not normalized:
        return None

    ordinal_map = {
        "1": 1,
        "first": 1,
        "1st": 1,
        "one": 1,
        "2": 2,
        "second": 2,
        "2nd": 2,
        "two": 2,
        "3": 3,
        "third": 3,
        "3rd": 3,
        "three": 3,
    }
    if normalized in ordinal_map:
        index = ordinal_map[normalized] - 1
        if 0 <= index < len(aliases):
            return aliases[index]

    command_match = re.match(
        r"^(?:run|use|pick|choose|select)\s+(.+?)\s*$",
        candidate_text,
        flags=re.IGNORECASE,
    )
    if command_match is not None:
        normalized = _normalize_frontdoor_text(command_match.group(1))

    for alias in aliases:
        if _normalize_frontdoor_text(alias) == normalized:
            return alias
    return None


def _extract_known_system_override(
    request: str,
    repository: CatalogRepository | None,
) -> str:
    """Extract a saved system alias from `on <system>` or `in <system>` phrasing."""
    if repository is None:
        return ""

    normalized_request = _normalize_frontdoor_text(request)
    systems = sorted(
        repository.list_systems(),
        key=lambda system: len(system.alias),
        reverse=True,
    )
    for system in systems:
        alias = _normalize_frontdoor_text(system.alias)
        if not alias:
            continue
        pattern = rf"(?:^| )(?:on|in) {re.escape(alias)}(?:$| )"
        if re.search(pattern, normalized_request):
            return system.alias
    return ""


def _prepare_interactive_request(
    request: str,
    *,
    config: Config,
    session_state: InteractiveSessionState,
) -> PreparedInteractiveRequest:
    """Rewrite follow-ups and clarification replies into canonical requests."""
    repository = _get_catalog_repository(config)
    stripped = request.strip()
    if not stripped:
        return PreparedInteractiveRequest(canonical_request=request)

    if (
        session_state.pending_system_entry_alias
        and session_state.pending_system_aliases
    ):
        chosen_system = _match_pending_alias_choice(
            stripped,
            session_state.pending_system_aliases,
        )
        if chosen_system:
            return PreparedInteractiveRequest(
                canonical_request=f"run {session_state.pending_system_entry_alias}",
                system_override=chosen_system,
                note=(
                    f"Using saved system {chosen_system!r} for "
                    f"{session_state.pending_system_entry_alias!r}."
                ),
            )

    if session_state.pending_clarification_aliases:
        chosen_alias = _match_pending_alias_choice(
            stripped,
            session_state.pending_clarification_aliases,
        )
        if chosen_alias:
            system_override = _extract_known_system_override(stripped, repository)
            note = (
                f"Using clarification choice {chosen_alias!r}."
                if not system_override
                else f"Using clarification choice {chosen_alias!r} on saved system {system_override!r}."
            )
            return PreparedInteractiveRequest(
                canonical_request=f"run {chosen_alias}",
                system_override=system_override,
                note=note,
            )

    if session_state.last_catalog_alias:
        rerun_match = re.match(
            r"^(?:rerun|run)\s+(?:that|it|the same(?: test)?|same(?: test)?)(?:\s+again)?(?:\s+on\s+(.+?))?\s*$",
            stripped,
            flags=re.IGNORECASE,
        )
        if rerun_match is not None:
            system_override = ""
            if repository is not None and rerun_match.group(1):
                override = repository.get_system(rerun_match.group(1).strip())
                if override is not None:
                    system_override = override.alias
            note = f"Reusing saved alias {session_state.last_catalog_alias!r} from this session."
            if system_override:
                note = (
                    f"Reusing saved alias {session_state.last_catalog_alias!r} "
                    f"on saved system {system_override!r}."
                )
            return PreparedInteractiveRequest(
                canonical_request=f"run {session_state.last_catalog_alias}",
                system_override=system_override,
                note=note,
            )

    system_override = _extract_known_system_override(stripped, repository)
    note = ""
    if system_override:
        note = f"Using saved system override {system_override!r} for this run."
    return PreparedInteractiveRequest(
        canonical_request=stripped,
        system_override=system_override,
        note=note,
    )


def _remember_catalog_reference(
    request: str,
    *,
    config: Config,
    session_state: InteractiveSessionState,
    system_override: str = "",
) -> None:
    """Remember the last matched saved alias/system for follow-up turns."""
    repository = _get_catalog_repository(config)
    if repository is None:
        return

    try:
        registry = CatalogRegistry.from_path(config.test_catalog_path)
    except Exception:  # noqa: BLE001
        return

    resolved = registry.match_request(request)
    if resolved.entry is None:
        return

    session_state.last_catalog_alias = resolved.entry.alias
    if system_override:
        session_state.last_system_alias = system_override
    elif resolved.entry.system:
        session_state.last_system_alias = resolved.entry.system
    session_state.last_request = request
    session_state.pending_clarification_aliases = ()
    session_state.pending_system_aliases = ()
    session_state.pending_system_entry_alias = ""


def _extract_ambiguous_catalog_aliases(state: RunState) -> tuple[str, ...]:
    """Pull catalog alias choices from a failed clarification message."""
    pattern = re.compile(
        r"needs clarification:\s*(.+?)\.\s*$",
        flags=re.IGNORECASE,
    )
    for message in [
        *state.errors,
        *state.intent_resolution.get("warnings", []),
    ]:
        match = pattern.search(message)
        if match is None:
            continue
        aliases = tuple(
            alias.strip()
            for alias in match.group(1).split(",")
            if alias.strip()
        )
        if aliases:
            return aliases
    return ()


def _print_alias_clarification_prompt(aliases: tuple[str, ...]) -> None:
    """Tell the user how to resolve an ambiguous saved-test request."""
    console.print("[yellow]Multiple saved tests matched. Reply with one alias or number:[/yellow]")
    for index, alias in enumerate(aliases, start=1):
        console.print(f"[dim]- {index}. {alias}[/dim]")


def _extract_missing_system_prompt(
    state: RunState,
) -> tuple[str, tuple[str, ...]]:
    """Pull a pending saved-system choice from a failed run state."""
    pattern = re.compile(
        r"Matched catalog entry '(.+?)' but no saved system was specified\. "
        r"Choose one of: (.+?)\.\s*$",
        flags=re.IGNORECASE,
    )
    for message in [
        *state.errors,
        *state.intent_resolution.get("warnings", []),
    ]:
        match = pattern.search(message)
        if match is None:
            continue
        alias = match.group(1).strip()
        systems = tuple(
            system.strip()
            for system in match.group(2).split(",")
            if system.strip()
        )
        if alias and systems:
            return alias, systems
    return "", ()


def _print_system_clarification_prompt(
    entry_alias: str,
    systems: tuple[str, ...],
) -> None:
    """Tell the user to pick a saved system for a matched test."""
    console.print(
        f"[yellow]Saved test {entry_alias!r} needs a system. Reply with one system alias or number:[/yellow]"
    )
    for index, alias in enumerate(systems, start=1):
        console.print(f"[dim]- {index}. {alias}[/dim]")


def _prompt_text(prompt: str, *, default: str | None = None) -> str:
    """Prompt for text input with an optional default."""
    suffix = f" [{default}]" if default not in (None, "") else ""
    response = input(f"{prompt}{suffix}: ").strip()
    if not response and default is not None:
        return default
    return response


def _prompt_required_text(prompt: str, *, default: str | None = None) -> str:
    """Prompt until a non-empty value is entered."""
    while True:
        value = _prompt_text(prompt, default=default)
        if value:
            return value
        console.print("[red]A value is required.[/red]")


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    """Prompt for a yes/no answer."""
    options = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{options}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        console.print("[red]Please answer yes or no.[/red]")


def _prompt_optional_int(prompt: str, *, default: int | None = None) -> int | None:
    """Prompt for an optional positive integer."""
    while True:
        raw = _prompt_text(
            prompt,
            default=str(default) if default is not None else None,
        )
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            console.print("[red]Enter a whole number or leave it blank.[/red]")
            continue
        if value < 1:
            console.print("[red]Value must be greater than zero.[/red]")
            continue
        return value


def _prompt_saved_args(*, default: list[str] | None = None) -> list[str]:
    """Prompt for optional baseline arguments."""
    while True:
        raw = _prompt_text(
            "Baseline arguments (optional)",
            default=" ".join(default or []),
        )
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError as exc:
            console.print(f"[red]Could not parse arguments:[/red] {exc}")


def _prompt_execution_type(
    *,
    default: CatalogExecutionType = CatalogExecutionType.PYTHON_SCRIPT,
) -> CatalogExecutionType:
    """Prompt for the saved execution type."""
    aliases = {
        "python": CatalogExecutionType.PYTHON_SCRIPT,
        "py": CatalogExecutionType.PYTHON_SCRIPT,
        "python_script": CatalogExecutionType.PYTHON_SCRIPT,
        "script": CatalogExecutionType.PYTHON_SCRIPT,
        "binary": CatalogExecutionType.EXECUTABLE,
        "bin": CatalogExecutionType.EXECUTABLE,
        "exe": CatalogExecutionType.EXECUTABLE,
        "executable": CatalogExecutionType.EXECUTABLE,
    }
    while True:
        raw = _prompt_text(
            "Execution type",
            default=default.value,
        ).strip().lower()
        resolved = aliases.get(raw)
        if resolved is not None:
            return resolved
        console.print(
            "[red]Choose one of: python_script, executable.[/red]",
        )


def _prompt_transport(
    *,
    default: CatalogSystemTransport = CatalogSystemTransport.LOCAL,
) -> CatalogSystemTransport:
    """Prompt for the execution transport."""
    aliases = {
        "local": CatalogSystemTransport.LOCAL,
        "ssh": CatalogSystemTransport.SSH,
    }
    while True:
        raw = _prompt_text(
            "System transport",
            default=default.value,
        ).strip().lower()
        resolved = aliases.get(raw)
        if resolved is not None:
            return resolved
        console.print("[red]Choose one of: local, ssh.[/red]")


def _prompt_ssh_auth_method(
    *,
    default: CatalogSystemAuthMethod = CatalogSystemAuthMethod.SSH_KEY,
) -> CatalogSystemAuthMethod:
    """Prompt for the SSH authentication method."""
    aliases = {
        "ssh_key": CatalogSystemAuthMethod.SSH_KEY,
        "key": CatalogSystemAuthMethod.SSH_KEY,
        "ssh-key": CatalogSystemAuthMethod.SSH_KEY,
        "password": CatalogSystemAuthMethod.PASSWORD,
        "pass": CatalogSystemAuthMethod.PASSWORD,
    }
    while True:
        raw = _prompt_text(
            "SSH auth method",
            default=default.value,
        ).strip().lower()
        resolved = aliases.get(raw)
        if resolved is not None:
            return resolved
        console.print("[red]Choose one of: ssh_key, password.[/red]")


def _prompt_keywords(*, default: list[str] | None = None) -> list[str]:
    """Prompt for optional comma-separated keywords."""
    raw = _prompt_text(
        "Keywords (comma-separated)",
        default=", ".join(default or []),
    )
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _prompt_catalog_system(
    alias: str,
    *,
    existing: CatalogSystem | None = None,
) -> CatalogSystem:
    """Collect a new catalog system definition interactively."""
    console.print(
        f"[cyan]Creating new execution system '{alias}'.[/cyan]"
    )
    transport = _prompt_transport(
        default=existing.transport if existing is not None else CatalogSystemTransport.LOCAL,
    )
    description = _prompt_text(
        "System description",
        default=existing.description if existing is not None else "",
    )
    working_directory = _prompt_text(
        "System working directory",
        default=existing.working_directory if existing is not None else "",
    )

    hostname = existing.hostname if existing is not None else ""
    ssh_config_host = existing.ssh_config_host if existing is not None else ""
    username = existing.username if existing is not None else ""
    port: int | None = existing.port if existing is not None else None
    auth_method = (
        existing.auth_method
        if existing is not None
        else CatalogSystemAuthMethod.SSH_KEY
    )
    password_env_var = existing.password_env_var if existing is not None else ""
    python_command = existing.python_command if existing is not None else ""
    credential_ref = existing.credential_ref if existing is not None else ""

    if transport == CatalogSystemTransport.SSH:
        auth_method = _prompt_ssh_auth_method(default=auth_method)
        while True:
            hostname = _prompt_text("SSH hostname", default=hostname)
            ssh_config_host = _prompt_text(
                "SSH config host alias",
                default=ssh_config_host or alias,
            )
            if auth_method == CatalogSystemAuthMethod.PASSWORD:
                if hostname:
                    break
            elif hostname or ssh_config_host:
                break
            console.print(
                "[red]Enter a hostname or ssh config host alias.[/red]"
            )
        username = _prompt_text("SSH username", default=username)
        port = _prompt_optional_int("SSH port", default=port)
        if auth_method == CatalogSystemAuthMethod.PASSWORD:
            password_env_var = _prompt_required_text(
                "Password environment variable",
                default=password_env_var or f"{alias.upper().replace('-', '_')}_SSH_PASSWORD",
            )
            ssh_config_host = ""
        python_command = _prompt_text(
            "Python command on this system",
            default=python_command or "python",
        )
        credential_ref = _prompt_text(
            "Credential reference",
            default=credential_ref or (
                f"env:{password_env_var}"
                if auth_method == CatalogSystemAuthMethod.PASSWORD
                else (f"ssh-config:{alias}" if ssh_config_host else "")
            ),
        )
    else:
        hostname = ""
        ssh_config_host = ""
        username = ""
        port = None
        auth_method = CatalogSystemAuthMethod.SSH_KEY
        password_env_var = ""
        python_command = _prompt_text(
            "Python command on this system",
            default=python_command or "python",
        )
        credential_ref = ""

    return CatalogSystem(
        alias=alias,
        description=description,
        transport=transport,
        hostname=hostname,
        username=username,
        port=port,
        ssh_config_host=ssh_config_host,
        auth_method=auth_method,
        password_env_var=password_env_var,
        python_command=python_command,
        working_directory=working_directory,
        credential_ref=credential_ref,
    )


def _collect_catalog_entry(
    repository: CatalogRepository,
    request: str,
    *,
    existing_entry: CatalogEntry | None = None,
) -> tuple[CatalogEntry, CatalogSystem | None] | None:
    """Collect entry fields, optionally editing an existing definition."""
    existing_alias = existing_entry.alias if existing_entry is not None else ""
    alias = _prompt_required_text("Catalog alias", default=existing_alias or None)
    while alias != existing_alias and repository.has_entry_alias(alias):
        console.print(f"[red]Alias {alias!r} already exists.[/red]")
        alias = _prompt_required_text("Catalog alias", default=existing_alias or None)

    execution_type = _prompt_execution_type(
        default=(
            existing_entry.execution_type
            if existing_entry is not None
            else CatalogExecutionType.PYTHON_SCRIPT
        )
    )
    default_target = (
        existing_entry.target
        if existing_entry is not None
        else (
            "scripts/"
            if execution_type == CatalogExecutionType.PYTHON_SCRIPT
            else "./bin/"
        )
    )
    target = _prompt_required_text(
        "Target path or executable",
        default=default_target,
    )

    current_system_alias = existing_entry.system if existing_entry is not None else ""
    system_alias = _prompt_text(
        "Default system alias (optional; leave blank to choose at run time)",
        default=current_system_alias,
    )

    existing_system = repository.get_system(system_alias) if system_alias else None
    new_system: CatalogSystem | None = None
    if system_alias and existing_system is None:
        new_system = _prompt_catalog_system(system_alias)

    description = _prompt_text(
        "Description",
        default=(
            existing_entry.description
            if existing_entry is not None
            else f"Saved from interactive request: {request}"
        ),
    )

    entry = CatalogEntry(
        alias=alias,
        description=description,
        execution_type=execution_type,
        target=target,
        system=system_alias,
        args=_prompt_saved_args(
            default=existing_entry.args if existing_entry is not None else None,
        ),
        keywords=_prompt_keywords(
            default=existing_entry.keywords if existing_entry is not None else None,
        ),
        working_directory=_prompt_text(
            "Entry working directory",
            default=existing_entry.working_directory if existing_entry is not None else "",
        ),
        timeout=_prompt_optional_int(
            "Timeout seconds",
            default=existing_entry.timeout if existing_entry is not None else None,
        ),
    )
    return entry, new_system


def _render_catalog_registration_summary(
    entry: CatalogEntry,
    *,
    system: CatalogSystem | None = None,
    catalog_path: str,
) -> None:
    """Print the entry/system that is about to be saved."""
    console.print(
        f"[cyan]About to save this definition to {catalog_path}:[/cyan]"
    )
    console.print(f"[dim]- alias: {entry.alias}[/dim]")
    console.print(f"[dim]- type: {entry.execution_type.value}[/dim]")
    console.print(f"[dim]- target: {entry.target}[/dim]")
    console.print(
        f"[dim]- default system: {entry.system or 'choose at run time'}[/dim]"
    )
    if entry.args:
        console.print(f"[dim]- baseline args: {' '.join(entry.args)}[/dim]")
    if entry.keywords:
        console.print(f"[dim]- keywords: {', '.join(entry.keywords)}[/dim]")
    if entry.working_directory:
        console.print(f"[dim]- working directory: {entry.working_directory}[/dim]")
    if entry.timeout is not None:
        console.print(f"[dim]- timeout: {entry.timeout}s[/dim]")
    if system is not None:
        console.print(
            f"[dim]- new system transport: {system.transport.value}[/dim]"
        )


def _teach_catalog_entry(config: Config, request: str) -> CatalogEntry | None:
    """Collect and persist a new catalog entry from interactive prompts."""
    if not config.test_catalog_path:
        console.print(
            "[red]Catalog teaching is unavailable because no catalog path is configured.[/red]"
        )
        return None

    repository = CatalogRepository(config.test_catalog_path)
    collected = _collect_catalog_entry(repository, request)
    if collected is None:
        return None
    entry, new_system = collected

    _render_catalog_registration_summary(
        entry,
        system=new_system,
        catalog_path=config.test_catalog_path,
    )
    if not _prompt_yes_no("Save this catalog entry now?", default=True):
        console.print("[yellow]Registration canceled.[/yellow]")
        return None

    try:
        if new_system is not None:
            repository.add_system(new_system)
        repository.add_entry(entry)
    except ValueError as exc:
        console.print(f"[red]Could not save catalog entry:[/red] {exc}")
        return None

    console.print(
        f"[green]Saved catalog entry '{entry.alias}' in {config.test_catalog_path}.[/green]"
    )
    return entry


def _is_unknown_catalog_request(state: RunState) -> bool:
    """Return True when the run failed because no saved alias matched."""
    return (
        state.phase == RunPhase.FAILED
        and not state.execution_results
        and any("cataloged test definition" in error for error in state.errors)
    )


def _print_catalog_entry_details(
    entry: CatalogEntry,
    *,
    system: CatalogSystem | None = None,
) -> None:
    """Render one catalog entry to the terminal."""
    console.print(f"[cyan]Catalog entry:[/cyan] {entry.alias}")
    console.print(f"[dim]- type: {entry.execution_type.value}[/dim]")
    console.print(f"[dim]- target: {entry.target}[/dim]")
    console.print(
        f"[dim]- default system: {entry.system or 'choose at run time'}[/dim]"
    )
    if entry.description:
        console.print(f"[dim]- description: {entry.description}[/dim]")
    if entry.args:
        console.print(f"[dim]- baseline args: {' '.join(entry.args)}[/dim]")
    if entry.keywords:
        console.print(f"[dim]- keywords: {', '.join(entry.keywords)}[/dim]")
    if entry.working_directory:
        console.print(f"[dim]- working directory: {entry.working_directory}[/dim]")
    if entry.timeout is not None:
        console.print(f"[dim]- timeout: {entry.timeout}s[/dim]")
    if system is not None:
        console.print(
            f"[dim]- transport: {system.transport.value}[/dim]"
        )
        if system.transport == CatalogSystemTransport.SSH:
            console.print(f"[dim]- ssh auth: {system.auth_method.value}[/dim]")
            if system.ssh_config_host:
                console.print(
                    f"[dim]- ssh config host: {system.ssh_config_host}[/dim]"
                )
            if system.hostname:
                console.print(f"[dim]- hostname: {system.hostname}[/dim]")
            if system.username:
                console.print(f"[dim]- username: {system.username}[/dim]")
            if system.password_env_var:
                console.print(
                    f"[dim]- password env var: {system.password_env_var}[/dim]"
                )
        if system.python_command:
            console.print(f"[dim]- python command: {system.python_command}[/dim]")


def _print_catalog_system_details(system: CatalogSystem) -> None:
    """Render one catalog system to the terminal."""
    console.print(f"[cyan]Catalog system:[/cyan] {system.alias}")
    console.print(f"[dim]- transport: {system.transport.value}[/dim]")
    if system.description:
        console.print(f"[dim]- description: {system.description}[/dim]")
    if system.working_directory:
        console.print(f"[dim]- working directory: {system.working_directory}[/dim]")
    if system.transport == CatalogSystemTransport.SSH:
        if system.ssh_config_host:
            console.print(f"[dim]- ssh config host: {system.ssh_config_host}[/dim]")
        if system.hostname:
            console.print(f"[dim]- hostname: {system.hostname}[/dim]")
        if system.username:
            console.print(f"[dim]- username: {system.username}[/dim]")
        if system.port is not None:
            console.print(f"[dim]- port: {system.port}[/dim]")
        if system.credential_ref:
            console.print(f"[dim]- credential ref: {system.credential_ref}[/dim]")


def _handle_catalog_command(
    command: CatalogCommand,
    config: Config,
    *,
    allow_mutation: bool = True,
) -> int:
    """Handle local catalog-management commands."""
    if not config.test_catalog_path:
        console.print(
            "[red]No catalog is configured for this repo.[/red]"
        )
        return 1

    repository = CatalogRepository(config.test_catalog_path)

    if command.resource == "entry":
        if command.action == "list":
            entries = repository.list_entries()
            if not entries:
                console.print(
                    f"[yellow]No saved tests yet. Populate {config.test_catalog_path} to run tests.[/yellow]"
                )
                return 0
            console.print(
                f"[cyan]Saved tests in {config.test_catalog_path}:[/cyan]"
            )
            for entry in entries:
                console.print(
                    f"[dim]- {entry.alias}: {entry.execution_type.value} on "
                    f"{entry.system or 'runtime system'} -> {entry.target}[/dim]"
                )
            return 0

        if not command.alias:
            console.print("[red]A catalog alias is required.[/red]")
            return 1

        entry = repository.get_entry(command.alias)
        if entry is None:
            console.print(
                f"[red]Catalog entry {command.alias!r} does not exist in {config.test_catalog_path}.[/red]"
            )
            return 1

        if command.action == "show":
            _print_catalog_entry_details(
                entry,
                system=repository.get_system(entry.system) if entry.system else None,
            )
            return 0

        if not allow_mutation:
            console.print(
                "[yellow]Dry run does not modify the catalog.[/yellow]"
            )
            return 1

        if command.action == "delete":
            if not _prompt_yes_no(
                f"Delete saved test '{entry.alias}' from {config.test_catalog_path}?",
                default=False,
            ):
                console.print("[yellow]Deletion canceled.[/yellow]")
                return 1
            deleted = repository.delete_entry(entry.alias)
            if deleted is None:
                console.print(f"[red]Catalog entry {entry.alias!r} no longer exists.[/red]")
                return 1
            console.print(
                f"[green]Deleted catalog entry '{entry.alias}'.[/green]"
            )
            return 0

        if command.action == "edit":
            collected = _collect_catalog_entry(
                repository,
                f"edit {entry.alias}",
                existing_entry=entry,
            )
            if collected is None:
                return 1
            updated_entry, new_system = collected
            _render_catalog_registration_summary(
                updated_entry,
                system=new_system,
                catalog_path=config.test_catalog_path,
            )
            if not _prompt_yes_no("Save these catalog changes now?", default=True):
                console.print("[yellow]Edit canceled.[/yellow]")
                return 1
            try:
                if new_system is not None:
                    repository.add_system(new_system)
                repository.update_entry(entry.alias, updated_entry)
            except ValueError as exc:
                console.print(f"[red]Could not update catalog entry:[/red] {exc}")
                return 1
            console.print(
                f"[green]Updated catalog entry '{updated_entry.alias}'.[/green]"
            )
            return 0

        console.print(f"[red]Unsupported catalog action {command.action!r}.[/red]")
        return 1

    if command.resource == "system":
        if command.action == "list":
            systems = repository.list_systems()
            if not systems:
                console.print(
                    f"[yellow]No saved systems yet in {config.test_catalog_path}.[/yellow]"
                )
                return 0
            console.print(
                f"[cyan]Saved systems in {config.test_catalog_path}:[/cyan]"
            )
            for system in systems:
                location = system.ssh_config_host or system.hostname or "current host"
                console.print(
                    f"[dim]- {system.alias}: {system.transport.value} -> {location}[/dim]"
                )
            return 0

        if not command.alias:
            console.print("[red]A system alias is required.[/red]")
            return 1
        system = repository.get_system(command.alias)
        if system is None:
            console.print(
                f"[red]Catalog system {command.alias!r} does not exist in {config.test_catalog_path}.[/red]"
            )
            return 1

        if command.action == "show":
            _print_catalog_system_details(system)
            return 0

        if not allow_mutation:
            console.print(
                "[yellow]Dry run does not modify the catalog.[/yellow]"
            )
            return 1

        if command.action == "delete":
            if not _prompt_yes_no(
                f"Delete saved system '{system.alias}' from {config.test_catalog_path}?",
                default=False,
            ):
                console.print("[yellow]Deletion canceled.[/yellow]")
                return 1
            try:
                deleted = repository.delete_system(system.alias)
            except ValueError as exc:
                console.print(f"[red]Could not delete catalog system:[/red] {exc}")
                return 1
            if deleted is None:
                console.print(f"[red]Catalog system {system.alias!r} no longer exists.[/red]")
                return 1
            console.print(
                f"[green]Deleted catalog system '{system.alias}'.[/green]"
            )
            return 0

        if command.action == "edit":
            updated_system = _prompt_catalog_system(
                system.alias,
                existing=system,
            )
            _print_catalog_system_details(updated_system)
            if not _prompt_yes_no("Save these system changes now?", default=True):
                console.print("[yellow]Edit canceled.[/yellow]")
                return 1
            try:
                repository.update_system(system.alias, updated_system)
            except ValueError as exc:
                console.print(f"[red]Could not update catalog system:[/red] {exc}")
                return 1
            console.print(
                f"[green]Updated catalog system '{updated_system.alias}'.[/green]"
            )
            return 0

        console.print(f"[red]Unsupported catalog action {command.action!r}.[/red]")
        return 1

    console.print(f"[red]Unsupported catalog action {command.action!r}.[/red]")
    return 1


def interactive_loop(*, dispatch_fn: RequestDispatcher | None = None) -> None:
    """Run an interactive REPL that reads requests line by line.

    Parameters
    ----------
    dispatch_fn:
        Callable that receives a validated request string.  When *None*
        (the default during initial build), requests are echoed back.
    """
    console.print(Panel(INTERACTIVE_BANNER.strip(), expand=False, style="cyan"))
    while True:
        try:
            raw = input("test-runner> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            break
        if _maybe_handle_frontdoor_input(stripped):
            continue

        try:
            request = validate_request(stripped)
        except InputValidationError as exc:
            console.print(f"[red]Validation error:[/red] {exc}")
            continue

        if dispatch_fn is not None:
            result = dispatch_fn(request)
            if inspect.isawaitable(result):
                asyncio.run(result)
        else:
            # Placeholder until orchestrator is wired in
            console.print(f"[green][accepted][/green] {request}")


async def interactive_loop_async(
    *,
    dispatch_fn: RequestDispatcher | None = None,
) -> int:
    """Run an interactive REPL backed by the orchestrator-aware request path."""
    console.print(Panel(INTERACTIVE_BANNER.strip(), expand=False, style="cyan"))
    session_exit_code = 0

    while True:
        try:
            raw = input("test-runner> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            break
        if _maybe_handle_frontdoor_input(stripped):
            continue

        if dispatch_fn is not None:
            result = dispatch_fn(stripped)
            if inspect.isawaitable(result):
                request_exit_code = await result
            else:
                request_exit_code = result
            if isinstance(request_exit_code, int):
                session_exit_code = max(session_exit_code, request_exit_code)
        else:
            console.print(f"[green][accepted][/green] {request}")

    return session_exit_code


# ---------------------------------------------------------------------------
# Request resolution (shared by one-shot and interactive modes)
# ---------------------------------------------------------------------------

def _resolve_request(args: argparse.Namespace) -> str | None:
    """Determine the request string from CLI args or stdin.

    Returns the raw (unvalidated) request string, or None when
    interactive mode should be entered instead.
    """
    # Explicit --interactive flag
    if args.interactive:
        return None

    # Positional argument(s)
    if args.request:
        return " ".join(args.request)

    # Piped / redirected stdin
    if not sys.stdin.isatty():
        return sys.stdin.read()

    # Nothing provided → interactive
    return None


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    """Configure logging with the given level."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Orchestrator wiring helpers
# ---------------------------------------------------------------------------

def _config_has_llm(config: Config) -> bool:
    """Return True when the OpenAI-compatible backend is configured."""
    return bool(
        config.llm_base_url
        and config.api_key
        and config.model_id
    )


def _choose_parse_mode(config: Config) -> ParseMode:
    """Prefer AUTO when an LLM backend is configured, otherwise stay offline."""
    if _config_has_llm(config):
        return ParseMode.AUTO
    return ParseMode.OFFLINE


def _build_report_channels(
    requested: Sequence[str] | None,
    *,
    verbose: bool = False,
) -> list[ReporterBase]:
    """Create reporting channels requested on the CLI."""
    channels: list[ReporterBase] = []
    for name in requested or ["cli"]:
        match name:
            case "cli":
                channels.append(create_cli_reporter(verbose=verbose))
            case "json":
                channels.append(JSONSummaryReporter())
            case "markdown":
                channels.append(MarkdownSummaryReporter())
            case _:
                logger.warning("Unsupported report channel %r ignored", name)
    return channels


def _build_execution_target(target_name: str):
    """Create the execution target selected by the user."""
    factory = get_factory()
    lookup_name = {
        "ci": "remote-ci",
    }.get(target_name, target_name)
    return factory.get_or_create(lookup_name, default="local")


def _build_default_command_env(
    target_name: str,
    *,
    working_dir: str | None = None,
) -> dict[str, str]:
    """Inject a PATH that can find console scripts from the active virtualenv."""
    if target_name != "local":
        return {}

    project_venv = Path(working_dir or Path.cwd()) / ".venv"
    candidates: list[str] = []
    for candidate in (
        Path(sys.executable).resolve().parent,
        project_venv / "Scripts",
        project_venv / "bin",
    ):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in candidates:
            candidates.append(candidate_str)

    if not candidates:
        return {}

    current_path = os.environ.get("PATH", "")
    merged_path = os.pathsep.join([*candidates, current_path]) if current_path else os.pathsep.join(candidates)
    return {"PATH": merged_path}


def _create_orchestrator(
    config: Config,
    args: argparse.Namespace,
    *,
    catalog_system_override: str = "",
) -> OrchestratorHub:
    """Create a configured orchestrator for a CLI request."""
    return OrchestratorHub(
        config,
        parse_mode=_choose_parse_mode(config),
        channels=_build_report_channels(
            args.report_channel,
            verbose=args.verbose,
        ),
        execution_target=_build_execution_target(args.target),
        execution_target_name=args.target,
        command_timeout=args.timeout,
        default_command_env=_build_default_command_env(
            args.target,
            working_dir=args.working_dir,
        ),
        default_working_directory=args.working_dir or "",
        catalog_system_override=catalog_system_override,
    )


def _print_run_notes(state: RunState) -> None:
    """Print high-signal notes after a run completes."""
    if state.intent_resolution.get("needs_clarification"):
        console.print(
            "[yellow]Low-confidence request interpretation; review the chosen commands.[/yellow]"
        )
    if state.errors:
        console.print("[red]Run issues:[/red]")
        for error in state.errors:
            console.print(f"  [red]- {error}[/red]")


def _exit_code_from_state(state: RunState) -> int:
    """Convert orchestrator state into a process exit code."""
    if state.phase == RunPhase.FAILED:
        return 1

    failure_statuses = {"failed", "error", "timeout"}
    if any(
        result.get("final_status") in failure_statuses
        for result in state.execution_results
    ):
        return 1

    if state.report.get("failed", 0) or state.report.get("errors", 0):
        return 1

    return 0


def _run_dry_run(
    config: Config,
    request: str,
    *,
    timeout: int | None = None,
    catalog_system_override: str = "",
) -> int:
    """Resolve the request without executing any test commands."""
    parse_mode = _choose_parse_mode(config)
    service = IntentParserService(
        config,
        parse_mode=parse_mode,
        catalog_system_override=catalog_system_override,
    )
    if parse_mode == ParseMode.OFFLINE:
        resolution = service.resolve_offline(request, timeout=timeout)
    else:
        resolution = service.resolve_offline(request, timeout=timeout)
        console.print(
            "[dim]Dry run uses offline intent resolution; full runs can use the configured LLM backend.[/dim]"
        )

    if not resolution.commands:
        console.print("[yellow]Dry run needs clarification before execution.[/yellow]")
        for warning in resolution.warnings:
            console.print(f"[yellow]- {warning}[/yellow]")
        return 1

    console.print(
        f"[green][accepted][/green] Dry run parsed as "
        f"{resolution.intent.value} / {resolution.framework.value}"
    )
    for warning in resolution.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    for command in resolution.commands:
        console.print(f"[dim]Command: {command.display}[/dim]")
    return 0


def _print_request_context(
    args: argparse.Namespace,
    config: Config,
    request: str,
) -> None:
    """Print the request header shared by one-shot and interactive modes."""
    console.print(f"[bold]Request:[/bold] {request}")
    console.print(
        f"[dim]Target: {args.target} | "
        f"Autonomy: {config.autonomy_policy.value} | "
        f"Reports: {args.report_channel or ['cli']}[/dim]"
    )


async def _handle_request(
    args: argparse.Namespace,
    config: Config,
    request: str,
    *,
    allow_catalog_registration: bool = True,
    session_state: InteractiveSessionState | None = None,
) -> int:
    """Handle one validated request through dry-run or full orchestration."""
    prepared_request = PreparedInteractiveRequest(canonical_request=request)
    if session_state is not None and args.interactive:
        prepared_request = _prepare_interactive_request(
            request,
            config=config,
            session_state=session_state,
        )
        request = prepared_request.canonical_request
        if prepared_request.note:
            console.print(f"[dim]{prepared_request.note}[/dim]")

    try:
        request = validate_request(request)
    except InputValidationError as exc:
        console.print(f"[red]Validation error:[/red] {exc}")
        return 1

    if _maybe_handle_frontdoor_input(request):
        return 0

    catalog_command = _parse_catalog_command(request)
    if catalog_command is not None:
        return _handle_catalog_command(
            catalog_command,
            config,
            allow_mutation=not args.dry_run,
        )

    _print_request_context(args, config, request)

    if args.dry_run:
        console.print("\n[yellow]Dry run — parsing request without execution.[/yellow]")
        return _run_dry_run(
            config,
            request,
            timeout=args.timeout,
            catalog_system_override=prepared_request.system_override,
        )

    parse_mode = _choose_parse_mode(config)
    if parse_mode == ParseMode.OFFLINE:
        console.print("[dim]LLM backend not configured; using offline parser.[/dim]")
    else:
        console.print("[dim]Using auto parse mode with offline fallback.[/dim]")

    if prepared_request.system_override:
        orchestrator = _create_orchestrator(
            config,
            args,
            catalog_system_override=prepared_request.system_override,
        )
    else:
        orchestrator = _create_orchestrator(config, args)
    state = await orchestrator.run(request)
    _print_run_notes(state)

    missing_system_alias, missing_system_choices = _extract_missing_system_prompt(state)
    if session_state is not None and missing_system_alias and missing_system_choices:
        session_state.pending_clarification_aliases = ()
        session_state.pending_system_entry_alias = missing_system_alias
        session_state.pending_system_aliases = missing_system_choices
        _print_system_clarification_prompt(
            missing_system_alias,
            missing_system_choices,
        )
        return 0

    ambiguous_aliases = _extract_ambiguous_catalog_aliases(state)
    if session_state is not None and ambiguous_aliases:
        session_state.pending_system_entry_alias = ""
        session_state.pending_system_aliases = ()
        session_state.pending_clarification_aliases = ambiguous_aliases
        _print_alias_clarification_prompt(ambiguous_aliases)
        return 0

    if session_state is not None:
        _remember_catalog_reference(
            request,
            config=config,
            session_state=session_state,
            system_override=prepared_request.system_override,
        )

    if (
        allow_catalog_registration
        and args.interactive
        and not args.dry_run
        and config.test_catalog_path
        and _is_unknown_catalog_request(state)
    ):
        if _prompt_yes_no(
            "No saved catalog entry matched. Register a new test now?",
            default=False,
        ):
            saved_entry = _teach_catalog_entry(config, request)
            if saved_entry is not None and _prompt_yes_no(
                f"Run saved alias '{saved_entry.alias}' now?",
                default=True,
            ):
                return await _handle_request(
                    args,
                    config,
                    f"run {saved_entry.alias}",
                    allow_catalog_registration=False,
                    session_state=session_state,
                )
            return 0 if saved_entry is not None else 1

    return _exit_code_from_state(state)


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------

async def async_main(argv: Sequence[str] | None = None) -> int:
    """Async entry point — parses args, loads config, dispatches request."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load configuration
    config = Config.load(env_file=args.env_file)

    # Determine effective log level
    if args.verbose:
        log_level = "DEBUG"
    elif args.log_level:
        log_level = args.log_level
    else:
        log_level = config.log_level
    _setup_logging(log_level)

    _print_banner()
    logger.debug("Config loaded: autonomy=%s, target=%s", config.autonomy_policy.value, args.target)

    # Resolve request source
    raw_request = _resolve_request(args)

    if raw_request is None:
        session_state = InteractiveSessionState()
        return await interactive_loop_async(
            dispatch_fn=lambda request: _handle_request(
                args,
                config,
                request,
                session_state=session_state,
            ),
        )

    # Validate one-shot request
    try:
        request = validate_request(raw_request)
    except InputValidationError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    return await _handle_request(args, config, request)


# ---------------------------------------------------------------------------
# Synchronous entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    """Synchronous CLI entry point (called by console_scripts)."""
    exit_code = asyncio.run(async_main(argv))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
