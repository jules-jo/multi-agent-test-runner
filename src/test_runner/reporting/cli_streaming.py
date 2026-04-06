"""CLI streaming output handler for real-time test result rendering.

Receives TestResultEvent and RunEvent objects and renders them to the
terminal with colored pass/fail indicators, test names, and timing info
using real-time line-by-line output.

Uses ``rich`` for ANSI-colored output while keeping each event as a
single printed line so output streams cleanly in CI and terminals.
"""

from __future__ import annotations

import sys
import time
from io import StringIO
from typing import Any, TextIO

from rich.console import Console
from rich.text import Text

from .base import CLIReporterBase, StreamEvent
from .events import EventType, RunEvent, TestResultEvent, TestStatus

# ---------------------------------------------------------------------------
# Status rendering helpers
# ---------------------------------------------------------------------------

_STATUS_STYLES: dict[TestStatus, tuple[str, str]] = {
    # (label, rich style)
    TestStatus.PASS: ("PASS", "bold green"),
    TestStatus.FAIL: ("FAIL", "bold red"),
    TestStatus.ERROR: ("ERR ", "bold red"),
    TestStatus.SKIP: ("SKIP", "bold yellow"),
}

_EVENT_STYLES: dict[EventType, tuple[str, str]] = {
    EventType.RUN_STARTED: ("▶ RUN", "bold cyan"),
    EventType.RUN_COMPLETED: ("■ END", "bold cyan"),
    EventType.SUITE_STARTED: ("◆ SUITE", "bold blue"),
    EventType.SUITE_COMPLETED: ("◆ DONE", "bold blue"),
    EventType.LOG: ("  LOG", "dim"),
    EventType.DISCOVERY: ("🔍 DISC", "bold magenta"),
    EventType.TROUBLESHOOT: ("🔧 DIAG", "bold yellow"),
    EventType.TEST_STARTED: ("  ···", "dim"),
    EventType.TEST_RESULT: ("  TST", "dim"),
    EventType.ROLLUP_SUMMARY: ("📊 SUM", "bold cyan"),
}


def _format_duration(seconds: float) -> str:
    """Format a duration for display.

    - Under 1 s  → "123ms"
    - 1–60 s     → "1.23s"
    - Over 60 s  → "1m 23s"
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


# ---------------------------------------------------------------------------
# CLI Streaming Reporter
# ---------------------------------------------------------------------------


class CLIStreamingReporter(CLIReporterBase):
    """Real-time CLI reporter that renders events line-by-line.

    Each ``TestResultEvent`` is printed as a single coloured line::

        ✓ PASS  test_module::test_add ..................... 12ms
        ✗ FAIL  test_module::test_div_zero ............... 45ms
                 ZeroDivisionError: division by zero

    ``RunEvent`` objects (lifecycle, log, discovery) are rendered with
    their own icons and styles.

    Parameters
    ----------
    console:
        A ``rich.Console`` to write to.  Defaults to stderr so test
        output on stdout stays clean.
    show_stdout:
        When True, prints captured stdout for failing tests.
    show_stderr:
        When True, prints captured stderr for failing tests.
    verbose:
        When True, shows passing test detail and all log events.
    file:
        Underlying text stream (overrides console if both given).
    """

    def __init__(
        self,
        *,
        console: Console | None = None,
        show_stdout: bool = False,
        show_stderr: bool = True,
        verbose: bool = False,
        file: TextIO | None = None,
    ) -> None:
        self._console = console or Console(
            stderr=True,
            file=file,
        )
        self._show_stdout = show_stdout
        self._show_stderr = show_stderr
        self._verbose = verbose
        self._run_start_time: float | None = None
        self._counts: dict[TestStatus, int] = {s: 0 for s in TestStatus}
        self._total: int = 0
        self._slow_threshold: float = 5.0  # seconds

    # ----- ReporterBase interface -----

    async def on_event(self, event: StreamEvent) -> None:
        """Route an event to the appropriate renderer."""
        if isinstance(event, TestResultEvent):
            self._render_test_result(event)
        elif isinstance(event, RunEvent):
            self._render_run_event(event)

    async def on_run_start(self) -> None:
        """Print the run header."""
        self._run_start_time = time.time()
        self._counts = {s: 0 for s in TestStatus}
        self._total = 0
        self._console.print()
        self._console.print(
            Text("─" * 60, style="dim"),
        )
        self._console.print(
            Text(" Test Run Started", style="bold cyan"),
        )
        self._console.print(
            Text("─" * 60, style="dim"),
        )

    async def on_run_end(self, summary: dict[str, Any]) -> None:
        """Print the coloured summary footer."""
        self._console.print(
            Text("─" * 60, style="dim"),
        )
        self._render_summary(summary)
        self._console.print(
            Text("─" * 60, style="dim"),
        )
        self._console.print()

    # ----- Private renderers -----

    def _render_test_result(self, event: TestResultEvent) -> None:
        """Render a single test result as one coloured line."""
        label, style = _STATUS_STYLES.get(
            event.status, ("????", "bold white")
        )

        # Icon: ✓ for pass/skip, ✗ for fail/error
        icon = "✓" if event.status in (TestStatus.PASS, TestStatus.SKIP) else "✗"

        # Build the line
        line = Text()
        line.append(f" {icon} ", style=style)
        line.append(f"{label}  ", style=style)

        # Test name (with optional suite prefix)
        display_name = event.test_name
        if event.suite:
            display_name = f"{event.suite}::{event.test_name}"
        line.append(display_name, style="bold" if event.failed else "")

        # Dotted leader + duration
        duration_str = _format_duration(event.duration)
        # Pad with dots between name and duration
        name_len = len(display_name) + len(label) + 5  # icon + spaces
        available = max(60 - name_len - len(duration_str), 2)
        line.append(" " + "·" * available + " ", style="dim")
        # Colour duration if slow
        dur_style = "bold yellow" if event.duration >= self._slow_threshold else "dim"
        line.append(duration_str, style=dur_style)

        self._console.print(line)

        # Track counts
        self._counts[event.status] = self._counts.get(event.status, 0) + 1
        self._total += 1

        # For failures/errors, show extra detail below
        if event.failed:
            self._render_failure_detail(event)
        elif self._verbose and event.message:
            self._console.print(f"         {event.message}", style="dim")

    def _render_failure_detail(self, event: TestResultEvent) -> None:
        """Print error details, stdout, and stderr for a failed test."""
        indent = "         "

        if event.error_details:
            for detail_line in event.error_details.splitlines()[:10]:
                self._console.print(
                    f"{indent}{detail_line}", style="red"
                )

        if event.message and not event.error_details:
            for msg_line in event.message.splitlines()[:5]:
                self._console.print(
                    f"{indent}{msg_line}", style="red"
                )

        if self._show_stdout and event.stdout:
            self._console.print(Text(f"{indent}[stdout]", style="dim"))
            for stdout_line in event.stdout.splitlines()[:20]:
                self._console.print(
                    Text(f"{indent}  {stdout_line}", style="dim")
                )

        if self._show_stderr and event.stderr:
            self._console.print(Text(f"{indent}[stderr]", style="dim yellow"))
            for stderr_line in event.stderr.splitlines()[:20]:
                self._console.print(
                    Text(f"{indent}  {stderr_line}", style="dim yellow")
                )

    def _render_run_event(self, event: RunEvent) -> None:
        """Render a lifecycle / log / discovery event."""
        label, style = _EVENT_STYLES.get(
            event.event_type, ("  ???", "dim")
        )

        # Skip low-priority log events unless verbose
        if event.event_type == EventType.LOG and not self._verbose:
            return

        line = Text()
        line.append(f" {label}  ", style=style)
        line.append(event.message)

        self._console.print(line)

    def _render_summary(self, summary: dict[str, Any]) -> None:
        """Render the final summary block including failure details with logs."""
        total = summary.get("total", self._total)
        passed = summary.get("passed", self._counts.get(TestStatus.PASS, 0))
        failed = summary.get("failed", self._counts.get(TestStatus.FAIL, 0))
        errors = summary.get("errors", self._counts.get(TestStatus.ERROR, 0))
        skipped = summary.get("skipped", self._counts.get(TestStatus.SKIP, 0))
        duration = summary.get("duration")
        all_passed = summary.get("all_passed", (failed == 0 and errors == 0))

        # Overall status line
        if all_passed and total > 0:
            status_line = Text(" ✓ All tests passed!", style="bold green")
        elif total == 0:
            status_line = Text(" ⚠ No tests were executed", style="bold yellow")
        else:
            status_line = Text(
                f" ✗ {failed + errors} test(s) failed", style="bold red"
            )
        self._console.print(status_line)

        # Count breakdown
        counts_line = Text("   ")
        first = True
        for count, label, style in [
            (passed, "passed", "green"),
            (failed, "failed", "red"),
            (errors, "errors", "red"),
            (skipped, "skipped", "yellow"),
        ]:
            if count:
                if not first:
                    counts_line.append(" | ")
                counts_line.append(f"{count} {label}", style=style)
                first = False
        if not first:
            counts_line.append(" | ")
        counts_line.append(f"{total} total")

        self._console.print(counts_line)

        # Timing
        if duration is not None:
            self._console.print(
                Text(f"   Duration: {_format_duration(duration)}", style="dim")
            )

        # Failure details with associated logs
        failure_details = summary.get("failure_details", [])
        if failure_details:
            self._console.print()
            self._console.print(
                Text(" Failure Details:", style="bold red"),
            )
            for i, detail in enumerate(failure_details, 1):
                self._render_failure_detail_summary(i, detail)

        # AI analysis if present
        ai = summary.get("ai_analysis")
        if ai:
            self._console.print()
            self._console.print("   AI Analysis:", style="bold")
            for ai_line in str(ai).splitlines():
                self._console.print(f"   {ai_line}")

    def _render_failure_detail_summary(
        self, index: int, detail: dict[str, Any]
    ) -> None:
        """Render a single failure detail block with associated logs in the summary.

        Args:
            index: 1-based index of the failure.
            detail: Dict with test_name, status, message, error_details,
                stdout, stderr, file_path, line_number etc.
        """
        indent = "   "
        test_name = detail.get("test_name", "unknown")
        status = detail.get("status", "fail").upper()
        duration = detail.get("duration", 0.0)
        message = detail.get("message", "")
        file_path = detail.get("file_path", "")
        line_number = detail.get("line_number")

        # Header line: "  1) [FAIL] test_name (file:line) — 0.45s"
        header = Text(f"{indent}{index}) ")
        status_style = "bold red"
        header.append(f"[{status}] ", style=status_style)
        header.append(test_name, style="bold")
        if file_path:
            loc = f" ({file_path}"
            if line_number is not None:
                loc += f":{line_number}"
            loc += ")"
            header.append(loc, style="dim")
        if duration:
            header.append(f" — {_format_duration(duration)}", style="dim")
        self._console.print(header)

        # Error message
        if message:
            for msg_line in message.splitlines()[:5]:
                self._console.print(
                    f"{indent}   {msg_line}", style="red"
                )

        # Error details (e.g. exception type / traceback snippet)
        error_details = detail.get("error_details", "")
        if error_details:
            for err_line in error_details.splitlines()[:10]:
                self._console.print(
                    f"{indent}   {err_line}", style="red"
                )

        # Associated logs
        has_logs = detail.get("has_logs", False)
        if has_logs:
            stdout = detail.get("stdout", "")
            stderr = detail.get("stderr", "")

            if stdout:
                self._console.print(
                    Text(f"{indent}   [stdout]", style="dim")
                )
                for log_line in stdout.splitlines()[:20]:
                    self._console.print(
                        Text(f"{indent}     {log_line}", style="dim")
                    )

            if stderr:
                self._console.print(
                    Text(f"{indent}   [stderr]", style="dim yellow")
                )
                for log_line in stderr.splitlines()[:20]:
                    self._console.print(
                        Text(f"{indent}     {log_line}", style="dim yellow")
                    )

        self._console.print()  # Blank line between failures


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_cli_reporter(
    *,
    verbose: bool = False,
    show_stdout: bool = False,
    show_stderr: bool = True,
    file: TextIO | None = None,
) -> CLIStreamingReporter:
    """Create a configured CLI streaming reporter.

    This is the preferred way to instantiate the reporter — it keeps
    construction details out of caller code and makes it easy to swap
    defaults later.
    """
    return CLIStreamingReporter(
        verbose=verbose,
        show_stdout=show_stdout,
        show_stderr=show_stderr,
        file=file,
    )
