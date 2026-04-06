"""Final summary rendering and formatting for test run results.

Combines all summary components — pass/fail/error/skip counts, failure
details with captured logs, timing breakdown, AI analysis, troubleshooter
fix proposals, and rollup history — into formatted output suitable for
multiple reporting channels.

The ``FinalSummaryRenderer`` is the single authoritative renderer for
end-of-run summaries.  It accepts a ``FinalSummaryReport`` (which bundles
a ``TestRunSummary`` with an optional ``FixProposalSet``) and produces
output in three formats:

- **Plain text**: For CLI output and log files (no ANSI codes).
- **Rich text**: For terminals that support ANSI via ``rich``.
- **Structured dict**: For messaging platforms (Teams, Slack) and JSON APIs.

Design decisions:

- Stateless renderer — takes a report and returns formatted output.
- Delegates to ``TestRunSummary.failure_summary_lines()`` and
  ``TestRunSummary.to_report_dict()`` for low-level data, then adds
  presentation logic on top.
- Configurable sections: callers can toggle which sections appear.
- Pluggable: implements a simple ``SummaryFormatter`` protocol so
  future formatters (e.g. Markdown for GitHub) can be added.
- Fix proposals section renders troubleshooter output alongside failures
  for a unified view of problems and proposed solutions.
"""

from __future__ import annotations

import abc
import datetime
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from test_runner.agents.troubleshooter.models import (
    FixConfidence,
    FixProposal,
    FixProposalSet,
)
from test_runner.models.summary import (
    FailureDetail,
    TestOutcome,
    TestRunSummary,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalSummaryReport:
    """Unified report that bundles all summary sections for rendering.

    This is the canonical input to ``FinalSummaryRenderer``.  It combines:
    - Test run summary (counts, failures, timing, AI analysis)
    - Optional troubleshooter fix proposals

    By bundling everything into one object we ensure the renderer has a
    single source of truth and callers don't need to pass multiple args.

    Attributes:
        summary: The core test run summary with counts, failures, timing.
        fix_proposals: Optional set of troubleshooter fix proposals.
    """

    summary: TestRunSummary
    fix_proposals: Optional[FixProposalSet] = None

    @property
    def has_fixes(self) -> bool:
        """True if there are any fix proposals."""
        return (
            self.fix_proposals is not None
            and len(self.fix_proposals.proposals) > 0
        )

    @classmethod
    def from_summary(
        cls,
        summary: TestRunSummary,
        fix_proposals: Optional[FixProposalSet] = None,
    ) -> "FinalSummaryReport":
        """Create a report from a summary and optional fix proposals."""
        return cls(summary=summary, fix_proposals=fix_proposals)


@dataclass(frozen=True)
class SummaryRenderConfig:
    """Controls which sections appear in the rendered summary.

    Attributes:
        show_header: Show the overall status banner.
        show_counts: Show pass/fail/error/skip breakdown.
        show_timing: Show duration and timing information.
        show_failures: Show detailed failure records.
        show_failure_logs: Include captured stdout/stderr in failure details.
        show_ai_analysis: Show LLM-generated analysis section.
        show_fixes: Show troubleshooter fix proposals section.
        show_metadata: Show execution metadata (target, framework, etc.).
        max_failure_logs_lines: Cap on log lines per failure (0 = unlimited).
        max_failures_shown: Cap on number of failures to render (0 = unlimited).
        max_stack_trace_lines: Cap on stack trace lines per failure (0 = unlimited).
        max_fixes_shown: Cap on number of fix proposals to render (0 = unlimited).
        show_pass_rate_bar: Render a visual pass-rate bar in text output.
    """

    show_header: bool = True
    show_counts: bool = True
    show_timing: bool = True
    show_failures: bool = True
    show_failure_logs: bool = True
    show_ai_analysis: bool = True
    show_fixes: bool = True
    show_metadata: bool = False
    max_failure_logs_lines: int = 30
    max_failures_shown: int = 0  # 0 = unlimited
    max_stack_trace_lines: int = 15
    max_fixes_shown: int = 0  # 0 = unlimited
    show_pass_rate_bar: bool = True


# ---------------------------------------------------------------------------
# Formatter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SummaryFormatter(Protocol):
    """Protocol for pluggable summary formatters."""

    def format(
        self,
        summary: TestRunSummary,
        config: SummaryRenderConfig,
    ) -> str:
        """Render a TestRunSummary to a string."""
        ...


# ---------------------------------------------------------------------------
# Plain text formatter
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format a duration for human display."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _pass_rate_bar(rate: float, width: int = 30) -> str:
    """Render a visual pass-rate bar like [████████░░░░] 80%."""
    filled = int(rate * width)
    empty = width - filled
    pct = f"{rate * 100:.0f}%"
    return f"[{'█' * filled}{'░' * empty}] {pct}"


def _truncate_lines(text: str, max_lines: int) -> str:
    """Truncate text to max_lines, appending a note if truncated."""
    if max_lines <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    truncated = lines[:max_lines]
    truncated.append(f"  ... ({len(lines) - max_lines} more lines)")
    return "\n".join(truncated)


class PlainTextFormatter:
    """Renders a FinalSummaryReport as plain text (no ANSI codes).

    Suitable for log files, CI output, and terminals without color support.
    """

    def format(
        self,
        summary: TestRunSummary,
        config: SummaryRenderConfig,
        fix_proposals: Optional[FixProposalSet] = None,
    ) -> str:
        """Render the full summary as plain text."""
        sections: list[str] = []

        if config.show_header:
            sections.append(self._render_header(summary))

        if config.show_counts:
            sections.append(self._render_counts(summary, config))

        if config.show_timing:
            sections.append(self._render_timing(summary))

        if config.show_failures and summary.has_failures:
            sections.append(
                self._render_failures(summary, config)
            )

        if config.show_ai_analysis and summary.ai_analysis:
            sections.append(self._render_ai_analysis(summary))

        if config.show_fixes and fix_proposals and fix_proposals.proposals:
            sections.append(self._render_fixes(fix_proposals, config))

        if config.show_metadata and summary.metadata:
            sections.append(self._render_metadata(summary))

        separator = "=" * 60
        body = "\n\n".join(s for s in sections if s)
        return f"{separator}\n{body}\n{separator}"

    def _render_header(self, summary: TestRunSummary) -> str:
        if summary.total == 0:
            return "  WARNING: No tests were executed"
        if summary.success:
            return f"  ALL {summary.total} TESTS PASSED"
        if summary.failed and summary.errors:
            return (
                f"  {summary.failure_count} TEST(S) FAILED OR ERRORED "
                f"out of {summary.total}"
            )
        if summary.errors:
            return f"  {summary.errors} TEST(S) ERRORED out of {summary.total}"
        return f"  {summary.failed} TEST(S) FAILED out of {summary.total}"

    def _render_counts(
        self,
        summary: TestRunSummary,
        config: SummaryRenderConfig,
    ) -> str:
        lines = ["  Results:"]
        parts: list[str] = []
        if summary.passed:
            parts.append(f"{summary.passed} passed")
        if summary.failed:
            parts.append(f"{summary.failed} failed")
        if summary.errors:
            parts.append(f"{summary.errors} errors")
        if summary.skipped:
            parts.append(f"{summary.skipped} skipped")
        parts.append(f"{summary.total} total")
        lines.append(f"    {' | '.join(parts)}")

        if config.show_pass_rate_bar and summary.total > 0:
            lines.append(f"    {_pass_rate_bar(summary.pass_rate)}")

        return "\n".join(lines)

    def _render_timing(self, summary: TestRunSummary) -> str:
        parts = [f"  Duration: {_format_duration(summary.duration_seconds)}"]
        if summary.start_time is not None:
            dt = datetime.datetime.fromtimestamp(
                summary.start_time, tz=datetime.timezone.utc
            )
            parts.append(f"  Started:  {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if summary.framework:
            parts.append(f"  Framework: {summary.framework}")
        if summary.run_id:
            parts.append(f"  Run ID:   {summary.run_id}")
        return "\n".join(parts)

    def _render_failures(
        self,
        summary: TestRunSummary,
        config: SummaryRenderConfig,
    ) -> str:
        lines = [f"  Failures ({summary.failure_count}):"]
        failures = summary.failures
        if config.max_failures_shown > 0:
            failures = failures[: config.max_failures_shown]

        for i, f in enumerate(failures, 1):
            lines.append(self._render_single_failure(i, f, config))

        remaining = summary.failure_count - len(failures)
        if remaining > 0:
            lines.append(f"\n    ... and {remaining} more failure(s)")

        return "\n".join(lines)

    def _render_single_failure(
        self,
        index: int,
        f: FailureDetail,
        config: SummaryRenderConfig,
    ) -> str:
        prefix = "ERROR" if f.outcome == TestOutcome.ERROR else "FAIL"
        loc = ""
        if f.file_path:
            loc = f" ({f.file_path}"
            if f.line_number is not None:
                loc += f":{f.line_number}"
            loc += ")"

        parts = [f"\n    {index}) [{prefix}] {f.test_name}{loc}"]

        if f.duration_seconds > 0:
            parts.append(
                f"       Duration: {_format_duration(f.duration_seconds)}"
            )

        if f.error_message:
            parts.append(f"       {f.error_message}")

        if f.error_type:
            parts.append(f"       Type: {f.error_type}")

        if f.stack_trace and config.show_failure_logs:
            trace = _truncate_lines(
                f.stack_trace, config.max_stack_trace_lines
            )
            parts.append(f"       --- traceback ---")
            for line in trace.splitlines():
                parts.append(f"       {line}")

        if config.show_failure_logs and f.has_logs:
            if f.stdout:
                truncated = _truncate_lines(
                    f.stdout, config.max_failure_logs_lines
                )
                parts.append(f"       --- stdout ---")
                for line in truncated.splitlines():
                    parts.append(f"       {line}")
            if f.stderr:
                truncated = _truncate_lines(
                    f.stderr, config.max_failure_logs_lines
                )
                parts.append(f"       --- stderr ---")
                for line in truncated.splitlines():
                    parts.append(f"       {line}")
            if f.log_output:
                truncated = _truncate_lines(
                    f.log_output, config.max_failure_logs_lines
                )
                parts.append(f"       --- logs ---")
                for line in truncated.splitlines():
                    parts.append(f"       {line}")

        return "\n".join(parts)

    def _render_ai_analysis(self, summary: TestRunSummary) -> str:
        lines = ["  AI Analysis:"]
        for line in summary.ai_analysis.splitlines():
            lines.append(f"    {line}")
        return "\n".join(lines)

    def _render_metadata(self, summary: TestRunSummary) -> str:
        lines = ["  Metadata:"]
        for key, value in summary.metadata.items():
            lines.append(f"    {key}: {value}")
        return "\n".join(lines)

    def _render_fixes(
        self,
        fix_proposals: FixProposalSet,
        config: SummaryRenderConfig,
    ) -> str:
        """Render the troubleshooter fix proposals section."""
        proposals = fix_proposals.by_confidence()
        total = len(proposals)
        if config.max_fixes_shown > 0:
            proposals = proposals[: config.max_fixes_shown]

        lines = [f"  Proposed Fixes ({total}):"]

        if fix_proposals.analysis_summary:
            lines.append(f"    {fix_proposals.analysis_summary}")
            lines.append("")

        for i, proposal in enumerate(proposals, 1):
            lines.append(self._render_single_fix(i, proposal))

        remaining = total - len(proposals)
        if remaining > 0:
            lines.append(f"\n    ... and {remaining} more fix proposal(s)")

        if fix_proposals.budget_exhausted:
            lines.append(
                "    NOTE: Troubleshooter budget exhausted — "
                "some failures were not analyzed."
            )

        return "\n".join(lines)

    def _render_single_fix(self, index: int, proposal: FixProposal) -> str:
        """Render a single fix proposal as plain text."""
        conf = proposal.confidence.value.upper()
        parts = [f"\n    {index}) [{conf}] {proposal.title}"]

        parts.append(f"       Target: {proposal.failure_id}")

        if proposal.category.value != "unknown":
            parts.append(f"       Category: {proposal.category.value}")

        if proposal.confidence_score > 0:
            parts.append(
                f"       Confidence: {proposal.confidence_score:.0%}"
            )

        if proposal.description:
            # Indent description, truncate long ones
            desc_lines = proposal.description.splitlines()[:5]
            for line in desc_lines:
                parts.append(f"       {line}")
            if len(proposal.description.splitlines()) > 5:
                parts.append("       ...")

        if proposal.affected_files:
            files = ", ".join(proposal.affected_files[:5])
            if len(proposal.affected_files) > 5:
                files += f" (+{len(proposal.affected_files) - 5} more)"
            parts.append(f"       Files: {files}")

        if proposal.proposed_changes:
            parts.append(f"       Changes: {len(proposal.proposed_changes)} proposed change(s)")
            for change in proposal.proposed_changes[:3]:
                parts.append(f"         - {change.description}")
            if len(proposal.proposed_changes) > 3:
                parts.append(
                    f"         ... and {len(proposal.proposed_changes) - 3} more"
                )

        if proposal.requires_user_action and proposal.user_action_description:
            parts.append(
                f"       ⚠ User action required: {proposal.user_action_description}"
            )

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Rich text formatter (ANSI terminal)
# ---------------------------------------------------------------------------


class RichTextFormatter:
    """Renders a TestRunSummary using ``rich`` markup for ANSI terminals.

    Returns a string of rich Console output captured to a buffer. Callers
    can also use ``render_to_console()`` to print directly.
    """

    def format(
        self,
        summary: TestRunSummary,
        config: SummaryRenderConfig,
        fix_proposals: Optional[FixProposalSet] = None,
    ) -> str:
        """Render and return as a string with ANSI codes."""
        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)
        self.render_to_console(summary, config, console, fix_proposals)
        return buf.getvalue()

    def render_to_console(
        self,
        summary: TestRunSummary,
        config: SummaryRenderConfig,
        console: Any,
        fix_proposals: Optional[FixProposalSet] = None,
    ) -> None:
        """Render directly to a rich Console instance."""
        from rich.text import Text
        from rich.panel import Panel
        from rich.table import Table

        separator = Text("─" * 60, style="dim")

        console.print(separator)

        # Header
        if config.show_header:
            if summary.total == 0:
                console.print(
                    Text(" ⚠ No tests were executed", style="bold yellow")
                )
            elif summary.success:
                console.print(
                    Text(
                        f" ✓ All {summary.total} tests passed!",
                        style="bold green",
                    )
                )
            else:
                console.print(
                    Text(
                        f" ✗ {summary.failure_count} test(s) failed",
                        style="bold red",
                    )
                )

        # Counts
        if config.show_counts:
            counts = Text("   ")
            first = True
            for count, label, style in [
                (summary.passed, "passed", "green"),
                (summary.failed, "failed", "red"),
                (summary.errors, "errors", "red"),
                (summary.skipped, "skipped", "yellow"),
            ]:
                if count:
                    if not first:
                        counts.append(" | ")
                    counts.append(f"{count} {label}", style=style)
                    first = False
            if not first:
                counts.append(" | ")
            counts.append(f"{summary.total} total")
            console.print(counts)

            if config.show_pass_rate_bar and summary.total > 0:
                bar = _pass_rate_bar(summary.pass_rate)
                bar_style = "green" if summary.pass_rate >= 0.8 else (
                    "yellow" if summary.pass_rate >= 0.5 else "red"
                )
                console.print(Text(f"   {bar}", style=bar_style))

        # Timing
        if config.show_timing:
            timing_parts: list[str] = []
            timing_parts.append(
                f"Duration: {_format_duration(summary.duration_seconds)}"
            )
            if summary.framework:
                timing_parts.append(f"Framework: {summary.framework}")
            if summary.run_id:
                timing_parts.append(f"Run: {summary.run_id}")
            console.print(
                Text(f"   {' | '.join(timing_parts)}", style="dim")
            )

        # Failure details
        if config.show_failures and summary.has_failures:
            console.print()
            console.print(
                Text(
                    f" Failure Details ({summary.failure_count}):",
                    style="bold red",
                )
            )
            failures = summary.failures
            if config.max_failures_shown > 0:
                failures = failures[: config.max_failures_shown]

            for i, f in enumerate(failures, 1):
                self._render_failure(console, i, f, config)

            remaining = summary.failure_count - len(failures)
            if remaining > 0:
                console.print(
                    Text(
                        f"   ... and {remaining} more failure(s)",
                        style="dim",
                    )
                )

        # AI Analysis
        if config.show_ai_analysis and summary.ai_analysis:
            console.print()
            console.print(Text(" AI Analysis:", style="bold cyan"))
            for line in summary.ai_analysis.splitlines():
                console.print(f"   {line}")

        # Fix Proposals
        if (
            config.show_fixes
            and fix_proposals
            and fix_proposals.proposals
        ):
            console.print()
            self._render_fixes_section(console, fix_proposals, config)

        # Metadata
        if config.show_metadata and summary.metadata:
            console.print()
            console.print(Text(" Metadata:", style="bold dim"))
            for key, value in summary.metadata.items():
                console.print(
                    Text(f"   {key}: {value}", style="dim")
                )

        console.print(separator)

    def _render_failure(
        self,
        console: Any,
        index: int,
        f: FailureDetail,
        config: SummaryRenderConfig,
    ) -> None:
        """Render a single failure block with rich formatting."""
        from rich.text import Text

        indent = "   "
        prefix = "ERROR" if f.outcome == TestOutcome.ERROR else "FAIL"

        # Header
        header = Text(f"{indent}{index}) ")
        header.append(f"[{prefix}] ", style="bold red")
        header.append(f.test_name, style="bold")
        if f.file_path:
            loc = f" ({f.file_path}"
            if f.line_number is not None:
                loc += f":{f.line_number}"
            loc += ")"
            header.append(loc, style="dim")
        if f.duration_seconds > 0:
            header.append(
                f" — {_format_duration(f.duration_seconds)}", style="dim"
            )
        console.print(header)

        # Error message
        if f.error_message:
            for line in f.error_message.splitlines()[:5]:
                console.print(f"{indent}   {line}", style="red")

        # Error type
        if f.error_type:
            console.print(
                Text(f"{indent}   Type: {f.error_type}", style="dim red")
            )

        # Stack trace
        if f.stack_trace and config.show_failure_logs:
            trace = _truncate_lines(
                f.stack_trace, config.max_stack_trace_lines
            )
            console.print(
                Text(f"{indent}   --- traceback ---", style="dim")
            )
            for line in trace.splitlines():
                console.print(
                    Text(f"{indent}     {line}", style="dim red")
                )

        # Captured output
        if config.show_failure_logs and f.has_logs:
            if f.stdout:
                truncated = _truncate_lines(
                    f.stdout, config.max_failure_logs_lines
                )
                console.print(
                    Text(f"{indent}   [stdout]", style="dim")
                )
                for line in truncated.splitlines():
                    console.print(
                        Text(f"{indent}     {line}", style="dim")
                    )
            if f.stderr:
                truncated = _truncate_lines(
                    f.stderr, config.max_failure_logs_lines
                )
                console.print(
                    Text(f"{indent}   [stderr]", style="dim yellow")
                )
                for line in truncated.splitlines():
                    console.print(
                        Text(f"{indent}     {line}", style="dim yellow")
                    )
            if f.log_output:
                truncated = _truncate_lines(
                    f.log_output, config.max_failure_logs_lines
                )
                console.print(
                    Text(f"{indent}   [logs]", style="dim")
                )
                for line in truncated.splitlines():
                    console.print(
                        Text(f"{indent}     {line}", style="dim")
                    )

        console.print()  # blank line between failures

    def _render_fixes_section(
        self,
        console: Any,
        fix_proposals: FixProposalSet,
        config: SummaryRenderConfig,
    ) -> None:
        """Render the fix proposals section with rich formatting."""
        from rich.text import Text

        proposals = fix_proposals.by_confidence()
        total = len(proposals)
        if config.max_fixes_shown > 0:
            proposals = proposals[: config.max_fixes_shown]

        console.print(
            Text(
                f" Proposed Fixes ({total}):",
                style="bold magenta",
            )
        )

        if fix_proposals.analysis_summary:
            console.print(
                Text(f"   {fix_proposals.analysis_summary}", style="dim")
            )

        for i, proposal in enumerate(proposals, 1):
            self._render_fix_proposal(console, i, proposal)

        remaining = total - len(proposals)
        if remaining > 0:
            console.print(
                Text(
                    f"   ... and {remaining} more fix proposal(s)",
                    style="dim",
                )
            )

        if fix_proposals.budget_exhausted:
            console.print(
                Text(
                    "   ⚠ Troubleshooter budget exhausted — "
                    "some failures were not analyzed.",
                    style="bold yellow",
                )
            )

    def _render_fix_proposal(
        self,
        console: Any,
        index: int,
        proposal: FixProposal,
    ) -> None:
        """Render a single fix proposal with rich formatting."""
        from rich.text import Text

        indent = "   "
        conf = proposal.confidence.value.upper()

        # Confidence color
        conf_style = {
            "HIGH": "bold green",
            "MEDIUM": "bold yellow",
            "LOW": "bold red",
        }.get(conf, "bold")

        # Header line
        header = Text(f"{indent}{index}) ")
        header.append(f"[{conf}] ", style=conf_style)
        header.append(proposal.title, style="bold")
        console.print(header)

        # Target failure
        console.print(
            Text(f"{indent}   Target: {proposal.failure_id}", style="dim")
        )

        if proposal.category.value != "unknown":
            console.print(
                Text(
                    f"{indent}   Category: {proposal.category.value}",
                    style="dim",
                )
            )

        if proposal.confidence_score > 0:
            console.print(
                Text(
                    f"{indent}   Confidence: {proposal.confidence_score:.0%}",
                    style="dim",
                )
            )

        # Description (truncated)
        if proposal.description:
            desc_lines = proposal.description.splitlines()[:3]
            for line in desc_lines:
                console.print(f"{indent}   {line}")
            if len(proposal.description.splitlines()) > 3:
                console.print(
                    Text(f"{indent}   ...", style="dim")
                )

        # Affected files
        if proposal.affected_files:
            files = ", ".join(proposal.affected_files[:3])
            if len(proposal.affected_files) > 3:
                files += f" (+{len(proposal.affected_files) - 3} more)"
            console.print(
                Text(f"{indent}   Files: {files}", style="cyan")
            )

        # Changes summary
        if proposal.proposed_changes:
            console.print(
                Text(
                    f"{indent}   Changes: {len(proposal.proposed_changes)} proposed",
                    style="dim",
                )
            )

        # User action required
        if proposal.requires_user_action and proposal.user_action_description:
            console.print(
                Text(
                    f"{indent}   ⚠ User action: {proposal.user_action_description}",
                    style="bold yellow",
                )
            )

        console.print()  # blank line between proposals


# ---------------------------------------------------------------------------
# Structured dict formatter (for messaging platforms / JSON APIs)
# ---------------------------------------------------------------------------


class StructuredDictFormatter:
    """Produces a structured dict suitable for messaging platforms and APIs.

    Extends ``TestRunSummary.to_report_dict()`` with presentation-layer
    fields (formatted duration, status label, pass-rate bar, section
    toggles) so consumers don't need to recompute formatting.
    """

    def format(
        self,
        summary: TestRunSummary,
        config: SummaryRenderConfig,
        fix_proposals: Optional[FixProposalSet] = None,
    ) -> dict[str, Any]:
        """Return a structured dict with all summary components."""
        base = summary.to_report_dict()

        # Add presentation fields
        base["formatted_duration"] = _format_duration(
            summary.duration_seconds
        )
        base["pass_rate_display"] = (
            f"{summary.pass_rate * 100:.1f}%"
            if summary.total > 0
            else "N/A"
        )
        base["pass_rate_bar"] = (
            _pass_rate_bar(summary.pass_rate)
            if summary.total > 0
            else ""
        )

        # Status label for messaging embeds
        if summary.total == 0:
            base["status_label"] = "No Tests"
            base["status_emoji"] = "⚠️"
            base["status_color"] = "warning"
        elif summary.success:
            base["status_label"] = "All Passed"
            base["status_emoji"] = "✅"
            base["status_color"] = "success"
        else:
            base["status_label"] = f"{summary.failure_count} Failed"
            base["status_emoji"] = "❌"
            base["status_color"] = "danger"

        has_fixes = (
            fix_proposals is not None
            and len(fix_proposals.proposals) > 0
        )

        # Section visibility (so consumers can conditionally render)
        base["sections"] = {
            "header": config.show_header,
            "counts": config.show_counts,
            "timing": config.show_timing,
            "failures": config.show_failures and summary.has_failures,
            "ai_analysis": config.show_ai_analysis
            and bool(summary.ai_analysis),
            "fixes": config.show_fixes and has_fixes,
            "metadata": config.show_metadata and bool(summary.metadata),
        }

        # Timing section
        if config.show_timing:
            base["timing"] = {
                "duration_seconds": summary.duration_seconds,
                "formatted": _format_duration(summary.duration_seconds),
                "start_time": summary.start_time,
                "end_time": summary.end_time,
            }

        # Metadata section
        if config.show_metadata and summary.metadata:
            base["metadata_entries"] = summary.metadata

        # Fix proposals section
        if config.show_fixes and has_fixes and fix_proposals is not None:
            proposals = fix_proposals.by_confidence()
            if config.max_fixes_shown > 0:
                proposals = proposals[: config.max_fixes_shown]

            base["fix_proposals"] = {
                "total": len(fix_proposals.proposals),
                "shown": len(proposals),
                "analysis_summary": fix_proposals.analysis_summary,
                "budget_exhausted": fix_proposals.budget_exhausted,
                "high_confidence_count": fix_proposals.high_confidence_count,
                "actionable_count": fix_proposals.actionable_count,
                "proposals": [
                    {
                        "failure_id": p.failure_id,
                        "title": p.title,
                        "description": p.description,
                        "category": p.category.value,
                        "confidence": p.confidence.value,
                        "confidence_score": p.confidence_score,
                        "affected_files": p.affected_files,
                        "change_count": p.change_count,
                        "is_actionable": p.is_actionable,
                        "requires_user_action": p.requires_user_action,
                        "user_action_description": p.user_action_description,
                        "summary_line": p.summary_line(),
                    }
                    for p in proposals
                ],
                "summary_lines": fix_proposals.summary_lines(),
            }

        # Failure summary lines (compact one-liners for cards/embeds)
        base["failure_summary_lines"] = summary.failure_summary_lines()

        return base


# ---------------------------------------------------------------------------
# FinalSummaryRenderer — unified entry point
# ---------------------------------------------------------------------------


class FinalSummaryRenderer:
    """Unified renderer that produces final summaries in multiple formats.

    This is the primary entry point for end-of-run summary rendering.
    It holds a configuration and delegates to the appropriate formatter.

    Accepts either a ``FinalSummaryReport`` (preferred) or a plain
    ``TestRunSummary`` for backwards compatibility.

    Usage::

        renderer = FinalSummaryRenderer()

        # Create a unified report
        report = FinalSummaryReport(summary=summary, fix_proposals=fixes)

        # Plain text for logs
        text = renderer.render_text(report)

        # Rich ANSI for terminals
        rich_text = renderer.render_rich(report)

        # Structured dict for APIs/messaging
        data = renderer.render_structured(report)

        # Or render to a specific rich Console
        renderer.render_to_console(report, console)

        # Backwards-compatible: also accepts a plain TestRunSummary
        text = renderer.render_text(summary)
    """

    def __init__(
        self,
        config: SummaryRenderConfig | None = None,
    ) -> None:
        self._config = config or SummaryRenderConfig()
        self._plain = PlainTextFormatter()
        self._rich = RichTextFormatter()
        self._structured = StructuredDictFormatter()

    @property
    def config(self) -> SummaryRenderConfig:
        """Current render configuration."""
        return self._config

    @config.setter
    def config(self, value: SummaryRenderConfig) -> None:
        self._config = value

    @staticmethod
    def _unpack(
        report_or_summary: FinalSummaryReport | TestRunSummary,
    ) -> tuple[TestRunSummary, Optional[FixProposalSet]]:
        """Extract summary and fix_proposals from either input type."""
        if isinstance(report_or_summary, FinalSummaryReport):
            return report_or_summary.summary, report_or_summary.fix_proposals
        return report_or_summary, None

    def render_text(
        self,
        report: FinalSummaryReport | TestRunSummary,
        config: SummaryRenderConfig | None = None,
    ) -> str:
        """Render the summary as plain text (no ANSI codes).

        Args:
            report: A FinalSummaryReport or TestRunSummary to render.
            config: Optional override config. Uses instance config if None.

        Returns:
            Plain text string suitable for log files.
        """
        summary, fixes = self._unpack(report)
        return self._plain.format(summary, config or self._config, fixes)

    def render_rich(
        self,
        report: FinalSummaryReport | TestRunSummary,
        config: SummaryRenderConfig | None = None,
    ) -> str:
        """Render the summary with rich ANSI formatting.

        Args:
            report: A FinalSummaryReport or TestRunSummary to render.
            config: Optional override config.

        Returns:
            String containing ANSI escape codes for terminal display.
        """
        summary, fixes = self._unpack(report)
        return self._rich.format(summary, config or self._config, fixes)

    def render_to_console(
        self,
        report: FinalSummaryReport | TestRunSummary,
        console: Any,
        config: SummaryRenderConfig | None = None,
    ) -> None:
        """Render directly to a rich Console instance.

        Args:
            report: A FinalSummaryReport or TestRunSummary to render.
            console: A ``rich.console.Console`` to write to.
            config: Optional override config.
        """
        summary, fixes = self._unpack(report)
        self._rich.render_to_console(
            summary, config or self._config, console, fixes
        )

    def render_structured(
        self,
        report: FinalSummaryReport | TestRunSummary,
        config: SummaryRenderConfig | None = None,
    ) -> dict[str, Any]:
        """Render as a structured dict for APIs and messaging platforms.

        Args:
            report: A FinalSummaryReport or TestRunSummary to render.
            config: Optional override config.

        Returns:
            Dict with all summary components and presentation fields.
        """
        summary, fixes = self._unpack(report)
        return self._structured.format(
            summary, config or self._config, fixes
        )
