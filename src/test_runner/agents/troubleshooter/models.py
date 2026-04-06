"""Data models for troubleshooter fix proposals.

Fix proposals are structured suggestions that the troubleshooter agent
generates after analyzing test failures. They are *diagnose-only* by
default: the agent proposes fixes but never auto-executes them. The
configurable autonomy policy architecture allows future auto-fix modes
to be plugged in without changing the model layer.

Design decisions:
- Pydantic BaseModel for validation, serialization, and schema generation
- Immutable (frozen) models for safe cross-agent sharing via orchestrator
- FixConfidence uses the same tier system as the confidence model
- Each FixProposal targets a specific failure and includes enough context
  for a human (or future auto-fix agent) to act on it
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field, computed_field


class FixConfidence(str, Enum):
    """Confidence level in a proposed fix.

    Maps to the project-wide confidence tier system but with
    fix-specific semantics:
    - HIGH: Strong evidence the fix addresses the root cause
    - MEDIUM: Plausible fix based on pattern matching
    - LOW: Speculative suggestion, needs human review
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailureCategory(str, Enum):
    """Broad classification of test failure root causes.

    Used by the analyzer to group failures and select appropriate
    fix strategies.
    """

    ASSERTION = "assertion"          # Expected vs actual mismatch
    IMPORT_ERROR = "import_error"    # Missing module / import failure
    SYNTAX_ERROR = "syntax_error"    # Python syntax issue
    TYPE_ERROR = "type_error"        # Type mismatch at runtime
    ATTRIBUTE_ERROR = "attribute_error"  # Missing attribute / method
    TIMEOUT = "timeout"              # Test exceeded time limit
    FIXTURE_ERROR = "fixture_error"  # Test fixture setup/teardown failure
    CONFIGURATION = "configuration"  # Config / environment issue
    DEPENDENCY = "dependency"        # Missing or incompatible dependency
    RUNTIME = "runtime"              # Generic runtime error
    UNKNOWN = "unknown"              # Could not classify


class ProposedChange(BaseModel, frozen=True):
    """A single proposed code or configuration change.

    Represents one atomic modification that would be part of a fix.
    Multiple ProposedChange objects can compose a single FixProposal.

    Attributes:
        file_path: Path to the file that should be modified.
        description: Human-readable description of what to change.
        original_snippet: The current code/text (for context).
        proposed_snippet: The suggested replacement code/text.
        line_start: Starting line number for the change, if known.
        line_end: Ending line number for the change, if known.
        change_type: Nature of the change (modify, add, delete, config).
    """

    file_path: str
    description: str
    original_snippet: str = ""
    proposed_snippet: str = ""
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    change_type: str = "modify"  # modify | add | delete | config

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_diff(self) -> bool:
        """True if both original and proposed snippets are provided."""
        return bool(self.original_snippet and self.proposed_snippet)


class FixProposal(BaseModel, frozen=True):
    """A structured fix suggestion for a specific test failure.

    The troubleshooter agent produces these after analyzing failures.
    They are presented to the user for approval — never auto-executed
    (unless the autonomy policy allows it in a future mode).

    Attributes:
        failure_id: Identifier of the failure this fix addresses
            (matches FailureDetail.test_id from the summary model).
        title: Short one-line summary of the proposed fix.
        description: Detailed explanation of the root cause analysis
            and why this fix should work.
        category: Classification of the failure's root cause.
        confidence: How confident the troubleshooter is in this fix.
        confidence_score: Numeric confidence in [0.0, 1.0].
        affected_files: List of file paths that would be modified.
        proposed_changes: Ordered list of specific changes to make.
        rationale: Step-by-step reasoning that led to this proposal.
        alternative_fixes: Brief descriptions of other approaches
            that were considered but ranked lower.
        requires_user_action: True if the fix needs manual steps
            beyond code changes (e.g. install a package, set env var).
        user_action_description: What the user needs to do manually.
        metadata: Extensible metadata for framework-specific context.
    """

    failure_id: str
    title: str
    description: str
    category: FailureCategory = FailureCategory.UNKNOWN
    confidence: FixConfidence = FixConfidence.LOW
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    affected_files: list[str] = Field(default_factory=list)
    proposed_changes: list[ProposedChange] = Field(default_factory=list)
    rationale: str = ""
    alternative_fixes: list[str] = Field(default_factory=list)
    requires_user_action: bool = False
    user_action_description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def change_count(self) -> int:
        """Number of individual changes in this proposal."""
        return len(self.proposed_changes)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_actionable(self) -> bool:
        """True if the proposal contains at least one concrete change."""
        return len(self.proposed_changes) > 0 or self.requires_user_action

    def summary_line(self) -> str:
        """One-line summary suitable for CLI or chat output."""
        conf = self.confidence.value.upper()
        files = ", ".join(self.affected_files[:3])
        if len(self.affected_files) > 3:
            files += f" (+{len(self.affected_files) - 3} more)"
        return f"[{conf}] {self.title} | files: {files or 'none'}"


class FixProposalSet(BaseModel, frozen=True):
    """Collection of fix proposals for a test run's failures.

    The top-level result returned by the troubleshooter agent to the
    orchestrator hub. Groups all proposals and provides aggregate
    statistics.

    Attributes:
        proposals: All generated fix proposals.
        analysis_summary: High-level summary of the failure analysis.
        total_failures_analyzed: Number of failures that were analyzed.
        total_proposals_generated: Number of fix proposals produced.
        budget_exhausted: True if the troubleshooter hit its step budget
            before analyzing all failures.
        metadata: Extensible metadata.
    """

    proposals: list[FixProposal] = Field(default_factory=list)
    analysis_summary: str = ""
    total_failures_analyzed: int = 0
    total_proposals_generated: int = 0
    budget_exhausted: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def high_confidence_count(self) -> int:
        """Number of proposals with HIGH confidence."""
        return sum(1 for p in self.proposals if p.confidence == FixConfidence.HIGH)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def actionable_count(self) -> int:
        """Number of proposals that are actionable."""
        return sum(1 for p in self.proposals if p.is_actionable)

    def by_confidence(self) -> list[FixProposal]:
        """Return proposals sorted by confidence (highest first)."""
        order = {FixConfidence.HIGH: 0, FixConfidence.MEDIUM: 1, FixConfidence.LOW: 2}
        return sorted(self.proposals, key=lambda p: (order.get(p.confidence, 3), -p.confidence_score))

    def for_failure(self, failure_id: str) -> list[FixProposal]:
        """Return all proposals targeting a specific failure."""
        return [p for p in self.proposals if p.failure_id == failure_id]

    def summary_lines(self) -> list[str]:
        """One-line summaries for all proposals, sorted by confidence."""
        return [p.summary_line() for p in self.by_confidence()]
