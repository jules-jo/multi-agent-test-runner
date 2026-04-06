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
import sys
import textwrap
from pathlib import Path
from typing import Protocol, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from test_runner import __version__
from test_runner.config import Config
from test_runner.agents.intent_service import IntentParserService, ParseMode
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

        try:
            request = validate_request(stripped)
        except InputValidationError as exc:
            console.print(f"[red]Validation error:[/red] {exc}")
            session_exit_code = 1
            continue

        if dispatch_fn is not None:
            result = dispatch_fn(request)
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

    candidates: list[str] = []
    for candidate in (
        Path(sys.executable).resolve().parent,
        Path(working_dir or Path.cwd()) / ".venv" / "bin",
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
) -> int:
    """Resolve the request without executing any test commands."""
    parse_mode = _choose_parse_mode(config)
    service = IntentParserService(config, parse_mode=parse_mode)
    if parse_mode == ParseMode.OFFLINE:
        resolution = service.resolve_offline(request, timeout=timeout)
    else:
        resolution = service.resolve_offline(request, timeout=timeout)
        console.print(
            "[dim]Dry run uses offline intent resolution; full runs can use the configured LLM backend.[/dim]"
        )

    console.print(
        f"[green][accepted][/green] Dry run parsed as "
        f"{resolution.intent.value} / {resolution.framework.value}"
    )
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
) -> int:
    """Handle one validated request through dry-run or full orchestration."""
    if _maybe_handle_frontdoor_input(request):
        return 0

    _print_request_context(args, config, request)

    if args.dry_run:
        console.print("\n[yellow]Dry run — parsing request without execution.[/yellow]")
        return _run_dry_run(config, request, timeout=args.timeout)

    parse_mode = _choose_parse_mode(config)
    if parse_mode == ParseMode.OFFLINE:
        console.print("[dim]LLM backend not configured; using offline parser.[/dim]")
    else:
        console.print("[dim]Using auto parse mode with offline fallback.[/dim]")

    orchestrator = _create_orchestrator(config, args)
    state = await orchestrator.run(request)
    _print_run_notes(state)
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
        return await interactive_loop_async(
            dispatch_fn=lambda request: _handle_request(args, config, request),
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
