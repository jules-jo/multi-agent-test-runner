"""Failure analysis engine that aggregates, categorizes, and excerpts test failures.

Sits between raw FailureDetail records and fix proposal generation,
providing a structured view of what went wrong across a test run.

Key capabilities:
- **Aggregation**: Groups failures by category, file path, error type,
  and common error patterns to surface systemic issues
- **Categorization**: Leverages the analyzer's classify_failure() and
  adds higher-level pattern grouping for cross-failure insights
- **Log excerpting**: Extracts the most relevant portions of stdout,
  stderr, stack traces, and log output for each failure, with
  configurable line limits and keyword highlighting

Design decisions:
- Pure data transforms — no LLM calls, no side effects
- Frozen Pydantic models for safe sharing across agents via orchestrator
- FailureAnalysisReport is the top-level aggregate consumed by the
  reporter and troubleshooter agents
- Log excerpts are bounded to prevent bloated reports while preserving
  the most diagnostic information
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel, Field, computed_field

from test_runner.agents.troubleshooter.analyzer import classify_failure
from test_runner.agents.troubleshooter.models import FailureCategory
from test_runner.models.summary import FailureDetail, TestOutcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log excerpt extraction
# ---------------------------------------------------------------------------

# Keywords that signal high-value log lines
_IMPORTANT_KEYWORDS: list[re.Pattern[str]] = [
    re.compile(r"error", re.IGNORECASE),
    re.compile(r"exception", re.IGNORECASE),
    re.compile(r"traceback", re.IGNORECASE),
    re.compile(r"assert", re.IGNORECASE),
    re.compile(r"fail", re.IGNORECASE),
    re.compile(r"raise", re.IGNORECASE),
    re.compile(r"warning", re.IGNORECASE),
    re.compile(r"critical", re.IGNORECASE),
    re.compile(r"File \"", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"ImportError", re.IGNORECASE),
    re.compile(r"SyntaxError", re.IGNORECASE),
    re.compile(r"TypeError", re.IGNORECASE),
    re.compile(r"AttributeError", re.IGNORECASE),
]


@dataclass
class ExcerptConfig:
    """Configuration for log excerpt extraction.

    Attributes:
        max_lines_per_source: Maximum lines to keep from each source
            (stdout, stderr, log_output, stack_trace).
        max_total_lines: Maximum total excerpt lines per failure.
        context_lines: Lines of context to keep around important matches.
        include_line_numbers: Whether to prefix excerpt lines with numbers.
    """

    max_lines_per_source: int = 30
    max_total_lines: int = 80
    context_lines: int = 2
    include_line_numbers: bool = True


class LogExcerpt(BaseModel, frozen=True):
    """A bounded excerpt from a single log source.

    Attributes:
        source: The origin of the excerpt (stdout, stderr, traceback, logs).
        lines: The extracted lines (may be fewer than original).
        total_lines: Total lines in the original source.
        truncated: True if the excerpt was truncated.
        important_line_numbers: Line numbers (1-based) containing
            important keywords in the original source.
    """

    source: str
    lines: list[str] = Field(default_factory=list)
    total_lines: int = 0
    truncated: bool = False
    important_line_numbers: list[int] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def line_count(self) -> int:
        """Number of lines in the excerpt."""
        return len(self.lines)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def text(self) -> str:
        """The excerpt as a single string."""
        return "\n".join(self.lines)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_content(self) -> bool:
        """True if the excerpt contains any lines."""
        return len(self.lines) > 0


def _find_important_lines(lines: list[str]) -> list[int]:
    """Return 1-based line numbers containing important keywords."""
    important: list[int] = []
    for i, line in enumerate(lines, start=1):
        for pattern in _IMPORTANT_KEYWORDS:
            if pattern.search(line):
                important.append(i)
                break
    return important


def _extract_excerpt(
    text: str,
    source: str,
    config: ExcerptConfig,
) -> LogExcerpt:
    """Extract a bounded excerpt from a text source.

    Strategy:
    1. Split into lines
    2. Find lines containing important keywords
    3. Keep important lines plus context_lines before/after
    4. If no important lines found, keep the last max_lines_per_source lines
       (tail behavior — end of output is usually most relevant)
    5. Truncate to max_lines_per_source
    """
    if not text or not text.strip():
        return LogExcerpt(source=source)

    all_lines = text.splitlines()
    total = len(all_lines)
    important = _find_important_lines(all_lines)

    if important:
        # Build a set of line indices to keep (0-based)
        keep_indices: set[int] = set()
        for line_no in important:
            idx = line_no - 1  # Convert to 0-based
            start = max(0, idx - config.context_lines)
            end = min(total, idx + config.context_lines + 1)
            for i in range(start, end):
                keep_indices.add(i)

        # Sort and extract
        sorted_indices = sorted(keep_indices)
        selected = [all_lines[i] for i in sorted_indices]
    else:
        # No important lines — take the tail
        start = max(0, total - config.max_lines_per_source)
        selected = all_lines[start:]

    # Truncate to limit
    truncated = len(selected) > config.max_lines_per_source
    selected = selected[: config.max_lines_per_source]

    return LogExcerpt(
        source=source,
        lines=selected,
        total_lines=total,
        truncated=truncated or total > len(selected),
        important_line_numbers=important,
    )


def extract_failure_excerpts(
    failure: FailureDetail,
    config: ExcerptConfig | None = None,
) -> list[LogExcerpt]:
    """Extract log excerpts from all sources of a failure.

    Processes stdout, stderr, log_output, and stack_trace, returning
    a list of LogExcerpt objects (only non-empty sources).

    Args:
        failure: The failure detail to extract excerpts from.
        config: Excerpt extraction configuration. Uses defaults if None.

    Returns:
        List of LogExcerpt objects, one per non-empty source.
    """
    cfg = config or ExcerptConfig()
    sources = [
        (failure.stack_trace, "traceback"),
        (failure.stderr, "stderr"),
        (failure.stdout, "stdout"),
        (failure.log_output, "logs"),
    ]

    excerpts: list[LogExcerpt] = []
    total_lines = 0

    for text, source_name in sources:
        if not text or not text.strip():
            continue
        if total_lines >= cfg.max_total_lines:
            break

        excerpt = _extract_excerpt(text, source_name, cfg)
        if excerpt.has_content:
            # Enforce total line budget
            remaining = cfg.max_total_lines - total_lines
            if excerpt.line_count > remaining:
                excerpt = LogExcerpt(
                    source=excerpt.source,
                    lines=excerpt.lines[:remaining],
                    total_lines=excerpt.total_lines,
                    truncated=True,
                    important_line_numbers=excerpt.important_line_numbers,
                )
            excerpts.append(excerpt)
            total_lines += excerpt.line_count

    return excerpts


# ---------------------------------------------------------------------------
# Categorized failure record
# ---------------------------------------------------------------------------


class CategorizedFailure(BaseModel, frozen=True):
    """A failure enriched with category classification and log excerpts.

    Wraps a FailureDetail with analysis metadata.

    Attributes:
        failure: The original failure detail.
        category: The classified failure category.
        excerpts: Relevant log excerpts extracted from the failure.
        error_pattern: A normalized pattern extracted from the error
            message (for grouping similar failures).
    """

    failure: FailureDetail
    category: FailureCategory
    excerpts: list[LogExcerpt] = Field(default_factory=list)
    error_pattern: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_excerpts(self) -> bool:
        """True if any excerpts were extracted."""
        return len(self.excerpts) > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def excerpt_text(self) -> str:
        """Combined excerpt text with source headers."""
        parts: list[str] = []
        for exc in self.excerpts:
            if exc.has_content:
                parts.append(f"--- {exc.source} ---\n{exc.text}")
        return "\n\n".join(parts)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary_line(self) -> str:
        """One-line summary for display."""
        loc = ""
        if self.failure.file_path:
            loc = f" ({self.failure.file_path}"
            if self.failure.line_number is not None:
                loc += f":{self.failure.line_number}"
            loc += ")"
        prefix = "ERROR" if self.failure.outcome == TestOutcome.ERROR else "FAIL"
        msg = self.failure.error_message[:100] or "no message"
        return f"[{prefix}] [{self.category.value}] {self.failure.test_name}{loc}: {msg}"


# ---------------------------------------------------------------------------
# Failure group (aggregation unit)
# ---------------------------------------------------------------------------


class FailureGroup(BaseModel, frozen=True):
    """A group of related failures sharing a common characteristic.

    Groups can be by category, file, error type, or error pattern.

    Attributes:
        key: The grouping key (e.g. category name, file path, pattern).
        group_type: What dimension was used to group ("category", "file",
            "error_type", "error_pattern").
        failures: The categorized failures in this group.
        representative_excerpt: A single excerpt that best represents
            the group (from the failure with the most log data).
    """

    key: str
    group_type: str
    failures: list[CategorizedFailure] = Field(default_factory=list)
    representative_excerpt: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of failures in this group."""
        return len(self.failures)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def test_ids(self) -> list[str]:
        """Test IDs in this group."""
        return [f.failure.test_id for f in self.failures]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def affected_files(self) -> list[str]:
        """Unique file paths affected by failures in this group."""
        files: list[str] = []
        seen: set[str] = set()
        for f in self.failures:
            if f.failure.file_path and f.failure.file_path not in seen:
                files.append(f.failure.file_path)
                seen.add(f.failure.file_path)
        return files

    def summary_line(self) -> str:
        """One-line summary of this group."""
        files = ", ".join(self.affected_files[:3])
        extra = f" (+{len(self.affected_files) - 3} more)" if len(self.affected_files) > 3 else ""
        return (
            f"[{self.group_type}:{self.key}] "
            f"{self.count} failure(s) | files: {files}{extra or 'none'}"
        )


# ---------------------------------------------------------------------------
# Top-level analysis report
# ---------------------------------------------------------------------------


class FailureAnalysisReport(BaseModel, frozen=True):
    """Aggregated failure analysis report for a test run.

    This is the top-level output of the failure analysis engine,
    consumed by the reporter and troubleshooter agents. It provides
    multiple views into the failures:
    - All categorized failures (flat list)
    - Groups by category, file, error type, and error pattern
    - Aggregate statistics

    Attributes:
        categorized_failures: All failures with categories and excerpts.
        by_category: Failures grouped by FailureCategory.
        by_file: Failures grouped by source file path.
        by_error_type: Failures grouped by exception/error type.
        by_error_pattern: Failures grouped by normalized error pattern.
        total_failures: Total number of failures analyzed.
        category_counts: Mapping of category name to failure count.
        most_common_category: The most frequently occurring category.
        analysis_notes: Human-readable notes about the analysis.
    """

    categorized_failures: list[CategorizedFailure] = Field(default_factory=list)
    by_category: list[FailureGroup] = Field(default_factory=list)
    by_file: list[FailureGroup] = Field(default_factory=list)
    by_error_type: list[FailureGroup] = Field(default_factory=list)
    by_error_pattern: list[FailureGroup] = Field(default_factory=list)
    total_failures: int = 0
    category_counts: dict[str, int] = Field(default_factory=dict)
    most_common_category: str = ""
    analysis_notes: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_failures(self) -> bool:
        return self.total_failures > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def category_count(self) -> int:
        """Number of distinct failure categories seen."""
        return len(self.by_category)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def file_count(self) -> int:
        """Number of distinct files with failures."""
        return len(self.by_file)

    def get_category_group(self, category: FailureCategory) -> FailureGroup | None:
        """Return the FailureGroup for a specific category, or None."""
        for g in self.by_category:
            if g.key == category.value:
                return g
        return None

    def get_file_group(self, file_path: str) -> FailureGroup | None:
        """Return the FailureGroup for a specific file, or None."""
        for g in self.by_file:
            if g.key == file_path:
                return g
        return None

    def summary_lines(self) -> list[str]:
        """Multi-line summary for CLI or chat output."""
        lines: list[str] = []
        lines.append(f"Failure Analysis: {self.total_failures} failure(s)")
        lines.append(f"  Categories: {self.category_count} distinct")
        lines.append(f"  Files: {self.file_count} affected")

        if self.most_common_category:
            count = self.category_counts.get(self.most_common_category, 0)
            lines.append(f"  Most common: {self.most_common_category} ({count})")

        if self.by_category:
            lines.append("")
            lines.append("By category:")
            for group in self.by_category:
                lines.append(f"  {group.key}: {group.count} failure(s)")

        for note in self.analysis_notes:
            lines.append(f"  Note: {note}")

        return lines

    def to_report_dict(self) -> dict[str, Any]:
        """Serializable dict for JSON reporting."""
        return {
            "total_failures": self.total_failures,
            "category_counts": self.category_counts,
            "most_common_category": self.most_common_category,
            "categories": [
                {
                    "category": g.key,
                    "count": g.count,
                    "test_ids": g.test_ids,
                    "affected_files": g.affected_files,
                }
                for g in self.by_category
            ],
            "files": [
                {
                    "file": g.key,
                    "count": g.count,
                    "test_ids": g.test_ids,
                }
                for g in self.by_file
            ],
            "failures": [
                {
                    "test_id": cf.failure.test_id,
                    "test_name": cf.failure.test_name,
                    "category": cf.category.value,
                    "error_type": cf.failure.error_type,
                    "error_message": cf.failure.error_message[:300],
                    "error_pattern": cf.error_pattern,
                    "file_path": cf.failure.file_path,
                    "line_number": cf.failure.line_number,
                    "has_excerpts": cf.has_excerpts,
                    "excerpt_sources": [e.source for e in cf.excerpts if e.has_content],
                }
                for cf in self.categorized_failures
            ],
            "analysis_notes": self.analysis_notes,
        }


# ---------------------------------------------------------------------------
# Error pattern normalization
# ---------------------------------------------------------------------------

# Patterns to normalize error messages into groupable patterns
_NORMALIZATION_RULES: list[tuple[re.Pattern[str], str]] = [
    # Replace specific values with placeholders
    (re.compile(r"'[^']{1,100}'"), "'<value>'"),
    (re.compile(r'"[^"]{1,100}"'), '"<value>"'),
    (re.compile(r"\b\d+\b"), "<N>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<addr>"),
    # Collapse whitespace
    (re.compile(r"\s+"), " "),
]


def normalize_error_pattern(error_message: str) -> str:
    """Normalize an error message into a groupable pattern.

    Replaces specific values (strings, numbers, addresses) with
    placeholders so that similar errors can be grouped together.

    Args:
        error_message: The raw error message.

    Returns:
        A normalized pattern string.
    """
    if not error_message:
        return ""

    # Take the first line only (most distinctive)
    first_line = error_message.strip().split("\n")[0]

    # Truncate very long messages
    if len(first_line) > 200:
        first_line = first_line[:200]

    pattern = first_line
    for regex, replacement in _NORMALIZATION_RULES:
        pattern = regex.sub(replacement, pattern)

    return pattern.strip()


# ---------------------------------------------------------------------------
# Main analysis engine
# ---------------------------------------------------------------------------


@dataclass
class FailureAnalysisConfig:
    """Configuration for the failure analysis engine.

    Attributes:
        excerpt_config: Configuration for log excerpt extraction.
        max_failures: Maximum failures to process (budget cap).
        group_by_pattern: Whether to compute error pattern groups
            (slightly more expensive due to normalization).
        min_group_size: Minimum group size to include in results.
            Groups below this threshold are still in the flat list,
            just not surfaced as named groups.
    """

    excerpt_config: ExcerptConfig = field(default_factory=ExcerptConfig)
    max_failures: int = 50
    group_by_pattern: bool = True
    min_group_size: int = 1


def _select_representative_excerpt(failures: list[CategorizedFailure]) -> str:
    """Select the most informative excerpt from a group of failures.

    Picks the failure with the most excerpt content as representative.
    """
    best = ""
    best_len = 0
    for cf in failures:
        text = cf.excerpt_text
        if len(text) > best_len:
            best = text
            best_len = len(text)
    return best


def _build_groups(
    categorized: list[CategorizedFailure],
    key_fn: Any,
    group_type: str,
    min_group_size: int = 1,
) -> list[FailureGroup]:
    """Build FailureGroup objects from categorized failures.

    Args:
        categorized: The categorized failures to group.
        key_fn: A callable that extracts the grouping key from a
            CategorizedFailure. Should return a string.
        group_type: Label for the grouping dimension.
        min_group_size: Minimum failures to form a group.

    Returns:
        List of FailureGroup objects, sorted by count descending.
    """
    buckets: dict[str, list[CategorizedFailure]] = defaultdict(list)
    for cf in categorized:
        key = key_fn(cf)
        if key:  # Skip empty keys
            buckets[key].append(cf)

    groups: list[FailureGroup] = []
    for key, members in buckets.items():
        if len(members) < min_group_size:
            continue
        groups.append(
            FailureGroup(
                key=key,
                group_type=group_type,
                failures=members,
                representative_excerpt=_select_representative_excerpt(members),
            )
        )

    # Sort by count descending, then by key for stability
    groups.sort(key=lambda g: (-g.count, g.key))
    return groups


def analyze_failures(
    failures: Sequence[FailureDetail],
    config: FailureAnalysisConfig | None = None,
) -> FailureAnalysisReport:
    """Aggregate, categorize, and excerpt test failures.

    This is the main entry point for the failure analysis engine. It:
    1. Classifies each failure into a FailureCategory
    2. Extracts relevant log excerpts from each failure
    3. Normalizes error messages into groupable patterns
    4. Aggregates failures by category, file, error type, and pattern
    5. Produces a structured FailureAnalysisReport

    Args:
        failures: Test failure details from the reporter agent.
        config: Analysis configuration. Uses defaults if None.

    Returns:
        A FailureAnalysisReport with all categorized failures and groups.
    """
    cfg = config or FailureAnalysisConfig()
    logger.info("Starting failure analysis for %d failure(s)", len(failures))

    # Budget cap
    to_analyze = list(failures[: cfg.max_failures])
    budget_note = ""
    if len(failures) > cfg.max_failures:
        budget_note = (
            f"Budget cap reached: analyzed {cfg.max_failures} of "
            f"{len(failures)} failures"
        )
        logger.warning(budget_note)

    # Step 1+2+3: Classify, excerpt, and pattern-normalize each failure
    categorized: list[CategorizedFailure] = []
    for failure in to_analyze:
        category = classify_failure(failure)
        excerpts = extract_failure_excerpts(failure, cfg.excerpt_config)
        error_pattern = (
            normalize_error_pattern(failure.error_message)
            if cfg.group_by_pattern
            else ""
        )

        categorized.append(
            CategorizedFailure(
                failure=failure,
                category=category,
                excerpts=excerpts,
                error_pattern=error_pattern,
            )
        )

    # Step 4: Build groups along multiple dimensions
    by_category = _build_groups(
        categorized,
        key_fn=lambda cf: cf.category.value,
        group_type="category",
        min_group_size=cfg.min_group_size,
    )

    by_file = _build_groups(
        categorized,
        key_fn=lambda cf: cf.failure.file_path,
        group_type="file",
        min_group_size=cfg.min_group_size,
    )

    by_error_type = _build_groups(
        categorized,
        key_fn=lambda cf: cf.failure.error_type,
        group_type="error_type",
        min_group_size=cfg.min_group_size,
    )

    by_error_pattern: list[FailureGroup] = []
    if cfg.group_by_pattern:
        by_error_pattern = _build_groups(
            categorized,
            key_fn=lambda cf: cf.error_pattern,
            group_type="error_pattern",
            min_group_size=cfg.min_group_size,
        )

    # Step 5: Compute aggregate statistics
    category_counts: dict[str, int] = {}
    for group in by_category:
        category_counts[group.key] = group.count

    most_common = ""
    if category_counts:
        most_common = max(category_counts, key=category_counts.get)  # type: ignore[arg-type]

    # Analysis notes
    notes: list[str] = []
    if budget_note:
        notes.append(budget_note)

    # Detect systemic patterns
    if by_category:
        dominant = by_category[0]
        if dominant.count > len(categorized) * 0.5 and len(categorized) > 2:
            notes.append(
                f"Dominant failure category: {dominant.key} "
                f"({dominant.count}/{len(categorized)} = "
                f"{dominant.count * 100 // len(categorized)}%)"
            )

    if by_file:
        hotspot = by_file[0]
        if hotspot.count > 2:
            notes.append(
                f"Failure hotspot: {hotspot.key} with {hotspot.count} failure(s)"
            )

    # Check for recurring error patterns
    if by_error_pattern:
        for pg in by_error_pattern:
            if pg.count > 1:
                notes.append(
                    f"Recurring error pattern ({pg.count}x): {pg.key[:80]}"
                )
                break  # Only note the top one

    logger.info(
        "Failure analysis complete: %d failures, %d categories, %d files, %d notes",
        len(categorized),
        len(by_category),
        len(by_file),
        len(notes),
    )

    return FailureAnalysisReport(
        categorized_failures=categorized,
        by_category=by_category,
        by_file=by_file,
        by_error_type=by_error_type,
        by_error_pattern=by_error_pattern,
        total_failures=len(categorized),
        category_counts=category_counts,
        most_common_category=most_common,
        analysis_notes=notes,
    )
