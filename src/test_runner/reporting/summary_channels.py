"""Additional reporting channels for final summaries."""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .base import ReporterBase, StreamEvent


class JSONSummaryReporter(ReporterBase):
    """Emit the final run summary as formatted JSON."""

    def __init__(self, *, file: TextIO | None = None, indent: int = 2) -> None:
        self._file = file or sys.stdout
        self._indent = indent

    async def on_event(self, event: StreamEvent) -> None:
        """Ignore streaming events; this reporter only writes the final summary."""

    async def on_run_end(self, summary: dict[str, Any]) -> None:
        self._file.write(json.dumps(summary, indent=self._indent, sort_keys=True))
        self._file.write("\n")
        self._file.flush()


class MarkdownSummaryReporter(ReporterBase):
    """Emit the final run summary as simple Markdown."""

    def __init__(self, *, file: TextIO | None = None) -> None:
        self._file = file or sys.stdout

    async def on_event(self, event: StreamEvent) -> None:
        """Ignore streaming events; this reporter only writes the final summary."""

    async def on_run_end(self, summary: dict[str, Any]) -> None:
        lines = [
            "# Test Run Summary",
            "",
            f"- Total: {summary.get('total', 0)}",
            f"- Passed: {summary.get('passed', 0)}",
            f"- Failed: {summary.get('failed', 0)}",
            f"- Errors: {summary.get('errors', 0)}",
            f"- Skipped: {summary.get('skipped', 0)}",
            f"- Duration: {summary.get('duration', 0)}s",
        ]

        failure_details = summary.get("failure_details", [])
        if failure_details:
            lines.extend(["", "## Failure Details", ""])
            for detail in failure_details:
                test_name = detail.get("test_name", "(unknown)")
                status = detail.get("status", "unknown")
                lines.append(f"### {test_name}")
                lines.append(f"- Status: {status}")
                message = detail.get("message")
                if message:
                    lines.append(f"- Message: {message}")

        self._file.write("\n".join(lines))
        self._file.write("\n")
        self._file.flush()
