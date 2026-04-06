"""Failure analysis engine for the troubleshooter agent.

Analyzes test failures from FailureDetail records and produces structured
FixProposal objects. The analyzer uses pattern-matching heuristics on
error messages, stack traces, and log output to classify failures and
generate fix suggestions.

The analyzer is purely programmatic (no LLM calls) — it applies
deterministic rules to produce proposals. An LLM-enhanced analyzer
can be layered on top in the future by extending FailureAnalyzer.

Design decisions:
- Strategy pattern: each FailureCategory has a dedicated analysis strategy
- Strategies are registered, not hardcoded — new ones can be plugged in
- Confidence scoring is explicit and auditable (rationale captured)
- Analysis respects a step budget to prevent runaway processing
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from test_runner.agents.troubleshooter.models import (
    FailureCategory,
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.models.summary import FailureDetail

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

# Ordered list of (pattern, category) — first match wins
_ERROR_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (re.compile(r"SyntaxError", re.IGNORECASE), FailureCategory.SYNTAX_ERROR),
    (re.compile(r"IndentationError", re.IGNORECASE), FailureCategory.SYNTAX_ERROR),
    (re.compile(r"ImportError|ModuleNotFoundError|No module named", re.IGNORECASE), FailureCategory.IMPORT_ERROR),
    (re.compile(r"TypeError", re.IGNORECASE), FailureCategory.TYPE_ERROR),
    (re.compile(r"AttributeError", re.IGNORECASE), FailureCategory.ATTRIBUTE_ERROR),
    (re.compile(r"fixture.*error|fixture.*not found|SetupError", re.IGNORECASE), FailureCategory.FIXTURE_ERROR),
    (re.compile(r"timeout|timed?\s*out|TimeoutError", re.IGNORECASE), FailureCategory.TIMEOUT),
    (re.compile(r"AssertionError|assert\s+.*==|assert\s+.*!=|Expected.*got", re.IGNORECASE), FailureCategory.ASSERTION),
    (re.compile(r"FileNotFoundError|PermissionError|OSError|IOError", re.IGNORECASE), FailureCategory.CONFIGURATION),
    (re.compile(r"pkg_resources|setuptools|pip|install", re.IGNORECASE), FailureCategory.DEPENDENCY),
]


def classify_failure(failure: FailureDetail) -> FailureCategory:
    """Classify a failure into a category based on error signals.

    Checks error_type, error_message, stack_trace, and stderr in
    priority order. Returns UNKNOWN if no pattern matches.
    """
    # Check all text sources for pattern matches
    texts = [
        failure.error_type,
        failure.error_message,
        failure.stack_trace,
        failure.stderr,
    ]
    combined = " ".join(t for t in texts if t)

    for pattern, category in _ERROR_PATTERNS:
        if pattern.search(combined):
            return category

    return FailureCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Analysis strategies (one per category)
# ---------------------------------------------------------------------------


class AnalysisStrategy(ABC):
    """Base class for category-specific failure analysis."""

    @property
    @abstractmethod
    def category(self) -> FailureCategory:
        """The failure category this strategy handles."""
        ...

    @abstractmethod
    def analyze(self, failure: FailureDetail) -> FixProposal | None:
        """Analyze a failure and return a fix proposal, or None if no fix found."""
        ...

    def _confidence_from_evidence(
        self,
        has_file_path: bool,
        has_stack_trace: bool,
        has_specific_error: bool,
    ) -> tuple[FixConfidence, float]:
        """Compute confidence based on available evidence.

        Returns (tier, score) tuple. More evidence = higher confidence.
        """
        evidence_count = sum([has_file_path, has_stack_trace, has_specific_error])
        if evidence_count >= 3:
            return FixConfidence.HIGH, 0.85
        if evidence_count >= 2:
            return FixConfidence.MEDIUM, 0.65
        if evidence_count >= 1:
            return FixConfidence.LOW, 0.40
        return FixConfidence.LOW, 0.20


class ImportErrorStrategy(AnalysisStrategy):
    """Analyzes import errors and suggests module installation or path fixes."""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.IMPORT_ERROR

    def analyze(self, failure: FailureDetail) -> FixProposal | None:
        combined = f"{failure.error_message} {failure.error_type} {failure.stderr}"
        module_match = re.search(
            r"No module named ['\"]?(\S+?)['\"]?(?:\s|$|;)",
            combined,
        )
        module_name = module_match.group(1).strip("'\"") if module_match else ""

        has_file = bool(failure.file_path)
        has_trace = bool(failure.stack_trace)
        has_module = bool(module_name)

        confidence, score = self._confidence_from_evidence(has_file, has_trace, has_module)

        changes: list[ProposedChange] = []
        affected: list[str] = []
        title = "Fix import error"
        description = "A module import failed during test execution."
        user_action = ""
        requires_user = False

        if module_name:
            title = f"Fix missing module: {module_name}"
            description = (
                f"The test failed because module '{module_name}' could not be imported. "
                f"This is likely a missing dependency or incorrect import path."
            )
            # Suggest pip install
            requires_user = True
            user_action = f"Install the missing module: pip install {module_name}"

            if failure.file_path:
                affected.append(failure.file_path)
                changes.append(
                    ProposedChange(
                        file_path=failure.file_path,
                        description=f"Verify import of '{module_name}' is correct",
                        change_type="config",
                    )
                )

        return FixProposal(
            failure_id=failure.test_id,
            title=title,
            description=description,
            category=self.category,
            confidence=confidence,
            confidence_score=score,
            affected_files=affected,
            proposed_changes=changes,
            rationale=f"Detected ImportError/ModuleNotFoundError for '{module_name}'. "
                      f"Evidence: file_path={'yes' if has_file else 'no'}, "
                      f"stack_trace={'yes' if has_trace else 'no'}, "
                      f"module_identified={'yes' if has_module else 'no'}.",
            requires_user_action=requires_user,
            user_action_description=user_action,
            alternative_fixes=[
                "Check if the module is installed in the correct virtual environment",
                "Verify PYTHONPATH includes the source directory",
            ],
        )


class AssertionErrorStrategy(AnalysisStrategy):
    """Analyzes assertion failures and suggests value corrections."""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.ASSERTION

    def analyze(self, failure: FailureDetail) -> FixProposal | None:
        combined = f"{failure.error_message} {failure.stack_trace}"
        # Try to extract expected vs actual
        expect_match = re.search(
            r"(?:expected|assert)\s+(.+?)\s*(?:==|!=)\s*(.+?)(?:\s|$)",
            combined,
            re.IGNORECASE,
        )

        has_file = bool(failure.file_path)
        has_trace = bool(failure.stack_trace)
        has_values = expect_match is not None

        confidence, score = self._confidence_from_evidence(has_file, has_trace, has_values)

        affected = [failure.file_path] if failure.file_path else []
        changes: list[ProposedChange] = []

        if failure.file_path and failure.line_number:
            changes.append(
                ProposedChange(
                    file_path=failure.file_path,
                    description="Review assertion — expected value may need updating or "
                                "the code under test may have a logic error",
                    line_start=failure.line_number,
                    change_type="modify",
                )
            )

        description = "An assertion failed during test execution."
        if has_values:
            description += (
                f" The comparison involved: {expect_match.group(1).strip()} vs "
                f"{expect_match.group(2).strip()}."
            )

        return FixProposal(
            failure_id=failure.test_id,
            title="Fix assertion failure" + (f" in {failure.test_name}" if failure.test_name else ""),
            description=description,
            category=self.category,
            confidence=confidence,
            confidence_score=score,
            affected_files=affected,
            proposed_changes=changes,
            rationale=f"Detected AssertionError. "
                      f"file_path={'yes' if has_file else 'no'}, "
                      f"stack_trace={'yes' if has_trace else 'no'}, "
                      f"expected_vs_actual={'yes' if has_values else 'no'}.",
            alternative_fixes=[
                "Check if the test expectation is outdated after a code change",
                "Verify test data/fixtures are set up correctly",
            ],
        )


class SyntaxErrorStrategy(AnalysisStrategy):
    """Analyzes syntax errors and suggests corrections."""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.SYNTAX_ERROR

    def analyze(self, failure: FailureDetail) -> FixProposal | None:
        has_file = bool(failure.file_path)
        has_trace = bool(failure.stack_trace)
        has_line = failure.line_number is not None

        confidence, score = self._confidence_from_evidence(has_file, has_trace, has_line)
        # Syntax errors with file+line are very actionable
        if has_file and has_line:
            confidence = FixConfidence.HIGH
            score = 0.90

        affected = [failure.file_path] if failure.file_path else []
        changes: list[ProposedChange] = []

        if failure.file_path:
            changes.append(
                ProposedChange(
                    file_path=failure.file_path,
                    description=f"Fix syntax error: {failure.error_message[:200]}",
                    line_start=failure.line_number,
                    change_type="modify",
                )
            )

        return FixProposal(
            failure_id=failure.test_id,
            title=f"Fix syntax error in {failure.file_path or 'unknown file'}",
            description=f"A syntax error prevents the file from being parsed. "
                        f"Error: {failure.error_message[:300]}",
            category=self.category,
            confidence=confidence,
            confidence_score=score,
            affected_files=affected,
            proposed_changes=changes,
            rationale=f"SyntaxError detected. Location: "
                      f"{'known' if has_file and has_line else 'partial'}.",
        )


class TypeErrorStrategy(AnalysisStrategy):
    """Analyzes type errors."""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.TYPE_ERROR

    def analyze(self, failure: FailureDetail) -> FixProposal | None:
        has_file = bool(failure.file_path)
        has_trace = bool(failure.stack_trace)
        confidence, score = self._confidence_from_evidence(has_file, has_trace, True)

        affected = [failure.file_path] if failure.file_path else []
        changes: list[ProposedChange] = []
        if failure.file_path:
            changes.append(
                ProposedChange(
                    file_path=failure.file_path,
                    description=f"Fix type mismatch: {failure.error_message[:200]}",
                    line_start=failure.line_number,
                    change_type="modify",
                )
            )

        return FixProposal(
            failure_id=failure.test_id,
            title="Fix TypeError" + (f" in {failure.test_name}" if failure.test_name else ""),
            description=f"A type error occurred: {failure.error_message[:300]}",
            category=self.category,
            confidence=confidence,
            confidence_score=score,
            affected_files=affected,
            proposed_changes=changes,
            rationale=f"TypeError detected with {'specific' if has_file else 'limited'} location info.",
            alternative_fixes=[
                "Check function signatures for parameter type mismatches",
                "Verify data types being passed to the failing function",
            ],
        )


class AttributeErrorStrategy(AnalysisStrategy):
    """Analyzes attribute errors (missing method/property)."""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.ATTRIBUTE_ERROR

    def analyze(self, failure: FailureDetail) -> FixProposal | None:
        combined = f"{failure.error_message} {failure.error_type}"
        attr_match = re.search(
            r"has no attribute ['\"](\w+)['\"]",
            combined,
        )
        attr_name = attr_match.group(1) if attr_match else ""

        has_file = bool(failure.file_path)
        has_trace = bool(failure.stack_trace)
        has_attr = bool(attr_name)

        confidence, score = self._confidence_from_evidence(has_file, has_trace, has_attr)
        affected = [failure.file_path] if failure.file_path else []

        changes: list[ProposedChange] = []
        if failure.file_path:
            desc = f"Fix missing attribute"
            if attr_name:
                desc += f" '{attr_name}'"
            changes.append(
                ProposedChange(
                    file_path=failure.file_path,
                    description=desc,
                    line_start=failure.line_number,
                    change_type="modify",
                )
            )

        title = "Fix AttributeError"
        if attr_name:
            title += f": missing '{attr_name}'"

        return FixProposal(
            failure_id=failure.test_id,
            title=title,
            description=f"An object is missing attribute '{attr_name}'. "
                        f"This may indicate a renamed method, removed property, or wrong object type.",
            category=self.category,
            confidence=confidence,
            confidence_score=score,
            affected_files=affected,
            proposed_changes=changes,
            rationale=f"AttributeError for '{attr_name}'. "
                      f"File known: {'yes' if has_file else 'no'}.",
            alternative_fixes=[
                "Check if the attribute was recently renamed or removed",
                "Verify the object type matches expectations",
            ],
        )


class TimeoutStrategy(AnalysisStrategy):
    """Analyzes timeout failures."""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.TIMEOUT

    def analyze(self, failure: FailureDetail) -> FixProposal | None:
        return FixProposal(
            failure_id=failure.test_id,
            title=f"Fix timeout in {failure.test_name or 'test'}",
            description="The test exceeded its time limit. This may indicate "
                        "an infinite loop, deadlock, or the need for a longer timeout.",
            category=self.category,
            confidence=FixConfidence.MEDIUM,
            confidence_score=0.50,
            affected_files=[failure.file_path] if failure.file_path else [],
            proposed_changes=[],
            rationale="Test timed out. Without more context, root cause is uncertain.",
            requires_user_action=True,
            user_action_description="Investigate whether the test has an infinite loop or "
                                    "if the timeout limit needs increasing.",
            alternative_fixes=[
                "Increase the test timeout setting",
                "Check for infinite loops or blocking I/O",
                "Mock slow external dependencies",
            ],
        )


class FixtureErrorStrategy(AnalysisStrategy):
    """Analyzes test fixture errors."""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.FIXTURE_ERROR

    def analyze(self, failure: FailureDetail) -> FixProposal | None:
        has_file = bool(failure.file_path)
        has_trace = bool(failure.stack_trace)
        confidence, score = self._confidence_from_evidence(has_file, has_trace, True)

        affected = [failure.file_path] if failure.file_path else []
        changes: list[ProposedChange] = []
        if failure.file_path:
            changes.append(
                ProposedChange(
                    file_path=failure.file_path,
                    description="Review fixture setup/teardown",
                    line_start=failure.line_number,
                    change_type="modify",
                )
            )

        return FixProposal(
            failure_id=failure.test_id,
            title="Fix fixture error" + (f" in {failure.test_name}" if failure.test_name else ""),
            description=f"A test fixture failed during setup or teardown: {failure.error_message[:200]}",
            category=self.category,
            confidence=confidence,
            confidence_score=score,
            affected_files=affected,
            proposed_changes=changes,
            rationale="Fixture error detected. Check conftest.py and fixture definitions.",
            alternative_fixes=[
                "Check conftest.py for fixture definition issues",
                "Verify fixture dependencies are available",
                "Check for fixture scope conflicts",
            ],
        )


class GenericStrategy(AnalysisStrategy):
    """Fallback strategy for unclassified failures."""

    @property
    def category(self) -> FailureCategory:
        return FailureCategory.UNKNOWN

    def analyze(self, failure: FailureDetail) -> FixProposal | None:
        has_file = bool(failure.file_path)
        has_trace = bool(failure.stack_trace)

        # Generic analysis is always low confidence
        confidence = FixConfidence.LOW
        score = 0.25 if has_file else 0.15

        affected = [failure.file_path] if failure.file_path else []

        return FixProposal(
            failure_id=failure.test_id,
            title=f"Investigate failure in {failure.test_name or 'unknown test'}",
            description=f"Test failed with: {failure.error_message[:300] or 'no error message'}",
            category=FailureCategory.RUNTIME if failure.error_type else FailureCategory.UNKNOWN,
            confidence=confidence,
            confidence_score=score,
            affected_files=affected,
            proposed_changes=[],
            rationale="Could not classify the failure into a known category. "
                      "Manual investigation recommended.",
            requires_user_action=True,
            user_action_description="Review the test output and stack trace to identify the root cause.",
        )


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


class StrategyRegistry:
    """Registry of analysis strategies keyed by FailureCategory.

    Strategies are pluggable — register new ones or override existing
    ones at runtime.
    """

    def __init__(self) -> None:
        self._strategies: dict[FailureCategory, AnalysisStrategy] = {}
        self._fallback: AnalysisStrategy = GenericStrategy()

    def register(self, strategy: AnalysisStrategy) -> None:
        """Register a strategy for its declared category."""
        self._strategies[strategy.category] = strategy

    def get(self, category: FailureCategory) -> AnalysisStrategy:
        """Get the strategy for a category, falling back to generic."""
        return self._strategies.get(category, self._fallback)

    @property
    def registered_categories(self) -> list[FailureCategory]:
        """List of categories with registered strategies."""
        return list(self._strategies.keys())


def create_default_registry() -> StrategyRegistry:
    """Create a registry with all built-in strategies."""
    registry = StrategyRegistry()
    for strategy_cls in [
        ImportErrorStrategy,
        AssertionErrorStrategy,
        SyntaxErrorStrategy,
        TypeErrorStrategy,
        AttributeErrorStrategy,
        TimeoutStrategy,
        FixtureErrorStrategy,
    ]:
        registry.register(strategy_cls())
    return registry


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


@dataclass
class AnalyzerConfig:
    """Configuration for the failure analyzer.

    Attributes:
        max_failures_to_analyze: Budget cap — stop after this many failures.
        include_low_confidence: Whether to include LOW confidence proposals.
        min_confidence_score: Skip proposals below this score.
    """

    max_failures_to_analyze: int = 20
    include_low_confidence: bool = True
    min_confidence_score: float = 0.0


class FailureAnalyzer:
    """Analyzes test failures and produces fix proposals.

    Uses category-specific strategies to generate structured fix
    suggestions. Respects a budget (max failures to analyze) to
    prevent runaway processing.

    Usage::

        analyzer = FailureAnalyzer()
        result = analyzer.analyze_failures(failures)
        for proposal in result.by_confidence():
            print(proposal.summary_line())
    """

    def __init__(
        self,
        *,
        registry: StrategyRegistry | None = None,
        config: AnalyzerConfig | None = None,
    ) -> None:
        self._registry = registry or create_default_registry()
        self._config = config or AnalyzerConfig()

    @property
    def config(self) -> AnalyzerConfig:
        return self._config

    @property
    def registry(self) -> StrategyRegistry:
        return self._registry

    def analyze_failures(
        self,
        failures: Sequence[FailureDetail],
    ) -> FixProposalSet:
        """Analyze a sequence of failures and produce fix proposals.

        Processes failures up to the configured budget. Each failure
        is classified, then the appropriate strategy generates a
        fix proposal.

        Args:
            failures: Test failure details from the reporter agent.

        Returns:
            A FixProposalSet with all generated proposals.
        """
        proposals: list[FixProposal] = []
        budget = self._config.max_failures_to_analyze
        analyzed = 0
        budget_exhausted = False

        for failure in failures:
            if analyzed >= budget:
                budget_exhausted = True
                logger.warning(
                    "Analyzer budget exhausted after %d failures "
                    "(total: %d)",
                    analyzed,
                    len(failures),
                )
                break

            analyzed += 1
            category = classify_failure(failure)
            strategy = self._registry.get(category)

            logger.debug(
                "Analyzing failure %s: category=%s strategy=%s",
                failure.test_id,
                category.value,
                type(strategy).__name__,
            )

            try:
                proposal = strategy.analyze(failure)
            except Exception:
                logger.exception(
                    "Strategy %s failed for %s",
                    type(strategy).__name__,
                    failure.test_id,
                )
                proposal = None

            if proposal is None:
                continue

            # Apply filters
            if proposal.confidence_score < self._config.min_confidence_score:
                logger.debug(
                    "Skipping proposal for %s: score %.2f < min %.2f",
                    failure.test_id,
                    proposal.confidence_score,
                    self._config.min_confidence_score,
                )
                continue

            if not self._config.include_low_confidence and proposal.confidence == FixConfidence.LOW:
                logger.debug(
                    "Skipping LOW confidence proposal for %s",
                    failure.test_id,
                )
                continue

            proposals.append(proposal)

        # Build summary
        categories_seen = set()
        for p in proposals:
            categories_seen.add(p.category.value)

        summary_parts = [
            f"Analyzed {analyzed} of {len(failures)} failure(s).",
            f"Generated {len(proposals)} fix proposal(s).",
        ]
        if categories_seen:
            summary_parts.append(
                f"Categories: {', '.join(sorted(categories_seen))}."
            )
        if budget_exhausted:
            summary_parts.append(
                f"Budget exhausted — {len(failures) - analyzed} failure(s) not analyzed."
            )

        return FixProposalSet(
            proposals=proposals,
            analysis_summary=" ".join(summary_parts),
            total_failures_analyzed=analyzed,
            total_proposals_generated=len(proposals),
            budget_exhausted=budget_exhausted,
        )

    def analyze_single(self, failure: FailureDetail) -> FixProposal | None:
        """Analyze a single failure and return its proposal (or None).

        Convenience method for one-off analysis without the full
        FixProposalSet wrapper.
        """
        category = classify_failure(failure)
        strategy = self._registry.get(category)
        try:
            return strategy.analyze(failure)
        except Exception:
            logger.exception(
                "Strategy %s failed for %s",
                type(strategy).__name__,
                failure.test_id,
            )
            return None
