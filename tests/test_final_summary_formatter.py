"""Tests for the final summary report formatter.

Verifies that the FinalSummaryRenderer correctly composes all summary
sections (counts, failures, timing, analysis, fixes) into unified output
across all three output formats: plain text, rich text, and structured dict.
"""

from __future__ import annotations

import pytest

from test_runner.agents.troubleshooter.models import (
    FailureCategory,
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.models.summary import (
    FailureDetail,
    TestOutcome,
    TestRunSummary,
)
from test_runner.reporting.summary_renderer import (
    FinalSummaryReport,
    FinalSummaryRenderer,
    PlainTextFormatter,
    RichTextFormatter,
    StructuredDictFormatter,
    SummaryRenderConfig,
    _format_duration,
    _pass_rate_bar,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_failure(
    name: str = "test_fail",
    outcome: TestOutcome = TestOutcome.FAILED,
    msg: str = "assert 1 == 2",
    file_path: str = "tests/test_a.py",
    line_number: int = 10,
    stdout: str = "",
    stderr: str = "",
) -> FailureDetail:
    return FailureDetail(
        test_id=f"tests/test_a.py::{name}",
        test_name=name,
        outcome=outcome,
        error_message=msg,
        file_path=file_path,
        line_number=line_number,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.5,
        framework="pytest",
    )


def _make_summary(
    total: int = 10,
    passed: int = 7,
    failed: int = 2,
    errors: int = 1,
    skipped: int = 0,
    failures: list[FailureDetail] | None = None,
    ai_analysis: str = "",
    metadata: dict | None = None,
) -> TestRunSummary:
    if failures is None:
        failures = [
            _make_failure("test_alpha"),
            _make_failure("test_beta", TestOutcome.ERROR, "RuntimeError: boom"),
        ]
    return TestRunSummary(
        run_id="run-test-001",
        total=total,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        start_time=1700000000.0,
        end_time=1700000005.5,
        duration_seconds=5.5,
        success=(failed == 0 and errors == 0),
        framework="pytest",
        failures=failures,
        ai_analysis=ai_analysis,
        metadata=metadata or {},
    )


def _make_fix_proposal(
    failure_id: str = "tests/test_a.py::test_alpha",
    title: str = "Fix assertion in test_alpha",
    confidence: FixConfidence = FixConfidence.HIGH,
    score: float = 0.85,
    category: FailureCategory = FailureCategory.ASSERTION,
) -> FixProposal:
    return FixProposal(
        failure_id=failure_id,
        title=title,
        description="The assertion compares wrong values. Update expected to match actual.",
        category=category,
        confidence=confidence,
        confidence_score=score,
        affected_files=["tests/test_a.py"],
        proposed_changes=[
            ProposedChange(
                file_path="tests/test_a.py",
                description="Change expected value from 1 to 2",
                original_snippet="assert result == 1",
                proposed_snippet="assert result == 2",
                change_type="modify",
            ),
        ],
        rationale="Pattern analysis detected assertion mismatch.",
        alternative_fixes=["Check if the function implementation changed"],
    )


def _make_fix_proposals(
    proposals: list[FixProposal] | None = None,
    analysis_summary: str = "Analyzed 2 failures. Generated 2 fix proposals.",
    budget_exhausted: bool = False,
) -> FixProposalSet:
    if proposals is None:
        proposals = [
            _make_fix_proposal(),
            _make_fix_proposal(
                failure_id="tests/test_a.py::test_beta",
                title="Fix RuntimeError in test_beta",
                confidence=FixConfidence.MEDIUM,
                score=0.55,
                category=FailureCategory.RUNTIME,
            ),
        ]
    return FixProposalSet(
        proposals=proposals,
        analysis_summary=analysis_summary,
        total_failures_analyzed=2,
        total_proposals_generated=len(proposals),
        budget_exhausted=budget_exhausted,
    )


def _make_report(
    summary: TestRunSummary | None = None,
    fix_proposals: FixProposalSet | None = None,
) -> FinalSummaryReport:
    return FinalSummaryReport(
        summary=summary or _make_summary(),
        fix_proposals=fix_proposals,
    )


# ---------------------------------------------------------------------------
# FinalSummaryReport dataclass
# ---------------------------------------------------------------------------


class TestFinalSummaryReport:
    """Tests for the FinalSummaryReport data class."""

    def test_has_fixes_true(self):
        report = _make_report(fix_proposals=_make_fix_proposals())
        assert report.has_fixes is True

    def test_has_fixes_false_none(self):
        report = _make_report(fix_proposals=None)
        assert report.has_fixes is False

    def test_has_fixes_false_empty(self):
        report = _make_report(
            fix_proposals=FixProposalSet(proposals=[])
        )
        assert report.has_fixes is False

    def test_from_summary_factory(self):
        summary = _make_summary()
        fixes = _make_fix_proposals()
        report = FinalSummaryReport.from_summary(summary, fixes)
        assert report.summary is summary
        assert report.fix_proposals is fixes

    def test_from_summary_without_fixes(self):
        summary = _make_summary()
        report = FinalSummaryReport.from_summary(summary)
        assert report.summary is summary
        assert report.fix_proposals is None


# ---------------------------------------------------------------------------
# PlainTextFormatter — unified output
# ---------------------------------------------------------------------------


class TestPlainTextFormatterUnified:
    """Tests that PlainTextFormatter composes all sections."""

    def test_all_sections_present(self):
        """All sections appear when enabled and data is available."""
        summary = _make_summary(ai_analysis="Root cause is assertion drift.")
        fixes = _make_fix_proposals()
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)

        # Header
        assert "FAILED" in output
        # Counts
        assert "7 passed" in output
        assert "2 failed" in output
        # Timing
        assert "5.50s" in output
        # Failures
        assert "test_alpha" in output
        # AI Analysis
        assert "Root cause is assertion drift" in output
        # Fixes
        assert "Proposed Fixes" in output
        assert "Fix assertion in test_alpha" in output
        assert "Fix RuntimeError in test_beta" in output

    def test_error_only_header_uses_errored_wording(self):
        summary = _make_summary(
            total=1,
            passed=0,
            failed=0,
            errors=1,
            failures=[_make_failure("test_beta", TestOutcome.ERROR, "RuntimeError: boom")],
        )
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig())

        assert "1 TEST(S) ERRORED out of 1" in output

    def test_no_fixes_section_when_none(self):
        """Fixes section absent when no fix proposals provided."""
        summary = _make_summary()
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), None)
        assert "Proposed Fixes" not in output

    def test_no_fixes_section_when_empty(self):
        """Fixes section absent when fix proposals list is empty."""
        summary = _make_summary()
        fmt = PlainTextFormatter()
        output = fmt.format(
            summary, SummaryRenderConfig(), FixProposalSet(proposals=[])
        )
        assert "Proposed Fixes" not in output

    def test_fixes_section_disabled_by_config(self):
        """Fixes section absent when show_fixes is False."""
        summary = _make_summary()
        fixes = _make_fix_proposals()
        config = SummaryRenderConfig(show_fixes=False)
        fmt = PlainTextFormatter()
        output = fmt.format(summary, config, fixes)
        assert "Proposed Fixes" not in output

    def test_fix_details_rendered(self):
        """Fix proposal details are correctly rendered."""
        summary = _make_summary()
        fixes = _make_fix_proposals()
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)

        assert "[HIGH]" in output
        assert "[MEDIUM]" in output
        assert "Category: assertion" in output
        assert "Confidence: 85%" in output
        assert "tests/test_a.py" in output
        assert "1 proposed change(s)" in output
        assert "Change expected value" in output

    def test_budget_exhausted_note(self):
        """Budget exhaustion note appears when troubleshooter hit budget."""
        summary = _make_summary()
        fixes = _make_fix_proposals(budget_exhausted=True)
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)
        assert "budget exhausted" in output.lower()

    def test_max_fixes_shown(self):
        """Only max_fixes_shown proposals are rendered."""
        summary = _make_summary()
        fixes = _make_fix_proposals()
        config = SummaryRenderConfig(max_fixes_shown=1)
        fmt = PlainTextFormatter()
        output = fmt.format(summary, config, fixes)

        assert "Proposed Fixes (2)" in output
        assert "1 more fix proposal(s)" in output

    def test_user_action_required(self):
        """User action description appears for fixes that require it."""
        proposal = FixProposal(
            failure_id="test::id",
            title="Install missing package",
            description="Package xyz not found",
            category=FailureCategory.DEPENDENCY,
            confidence=FixConfidence.HIGH,
            confidence_score=0.9,
            requires_user_action=True,
            user_action_description="Run: pip install xyz",
        )
        fixes = _make_fix_proposals(proposals=[proposal])
        summary = _make_summary()
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)
        assert "User action required" in output
        assert "pip install xyz" in output

    def test_all_passed_no_failures_no_fixes(self):
        """When all tests pass, no failure or fix sections appear."""
        summary = _make_summary(
            total=5, passed=5, failed=0, errors=0, failures=[]
        )
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig())
        assert "ALL 5 TESTS PASSED" in output
        assert "Failures" not in output
        assert "Proposed Fixes" not in output

    def test_analysis_summary_in_fixes(self):
        """Analysis summary appears in fixes section."""
        summary = _make_summary()
        fixes = _make_fix_proposals(
            analysis_summary="Found 2 distinct root causes."
        )
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)
        assert "Found 2 distinct root causes" in output


# ---------------------------------------------------------------------------
# RichTextFormatter — unified output
# ---------------------------------------------------------------------------


class TestRichTextFormatterUnified:
    """Tests that RichTextFormatter composes all sections."""

    def test_all_sections_present(self):
        """All sections rendered with ANSI output."""
        summary = _make_summary(ai_analysis="Drift detected.")
        fixes = _make_fix_proposals()
        fmt = RichTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)

        # Rich output includes ANSI codes, so check for text content
        assert "failed" in output.lower()
        assert "passed" in output.lower() or "7" in output
        assert "test_alpha" in output
        assert "Drift detected" in output
        assert "Proposed Fixes" in output
        assert "Fix assertion" in output

    def test_no_fixes_when_none(self):
        """No fixes section in rich output when None."""
        summary = _make_summary()
        fmt = RichTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), None)
        assert "Proposed Fixes" not in output

    def test_confidence_levels_rendered(self):
        """Both HIGH and MEDIUM confidence markers appear."""
        summary = _make_summary()
        fixes = _make_fix_proposals()
        fmt = RichTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)
        assert "HIGH" in output
        assert "MEDIUM" in output

    def test_budget_exhausted_in_rich(self):
        """Budget exhaustion warning appears in rich output."""
        summary = _make_summary()
        fixes = _make_fix_proposals(budget_exhausted=True)
        fmt = RichTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)
        assert "budget exhausted" in output.lower()


# ---------------------------------------------------------------------------
# StructuredDictFormatter — unified output
# ---------------------------------------------------------------------------


class TestStructuredDictFormatterUnified:
    """Tests that StructuredDictFormatter composes all sections."""

    def test_all_sections_present(self):
        """Structured dict contains all expected keys."""
        summary = _make_summary(ai_analysis="Analysis here.")
        fixes = _make_fix_proposals()
        fmt = StructuredDictFormatter()
        result = fmt.format(summary, SummaryRenderConfig(), fixes)

        # Base summary fields
        assert result["total"] == 10
        assert result["passed"] == 7
        assert result["failed"] == 2
        assert "formatted_duration" in result
        assert "status_label" in result
        assert "pass_rate_bar" in result

        # Sections visibility
        assert result["sections"]["fixes"] is True

        # Fix proposals block
        assert "fix_proposals" in result
        fp = result["fix_proposals"]
        assert fp["total"] == 2
        assert fp["shown"] == 2
        assert fp["analysis_summary"] == "Analyzed 2 failures. Generated 2 fix proposals."
        assert fp["budget_exhausted"] is False
        assert fp["high_confidence_count"] == 1
        assert fp["actionable_count"] == 2
        assert len(fp["proposals"]) == 2
        assert len(fp["summary_lines"]) == 2

    def test_fix_proposal_fields(self):
        """Each proposal in structured dict has expected fields."""
        summary = _make_summary()
        fixes = _make_fix_proposals()
        fmt = StructuredDictFormatter()
        result = fmt.format(summary, SummaryRenderConfig(), fixes)

        proposal = result["fix_proposals"]["proposals"][0]
        assert "failure_id" in proposal
        assert "title" in proposal
        assert "description" in proposal
        assert "category" in proposal
        assert "confidence" in proposal
        assert "confidence_score" in proposal
        assert "affected_files" in proposal
        assert "change_count" in proposal
        assert "is_actionable" in proposal
        assert "requires_user_action" in proposal
        assert "summary_line" in proposal

    def test_no_fixes_key_when_none(self):
        """No fix_proposals key when no fixes provided."""
        summary = _make_summary()
        fmt = StructuredDictFormatter()
        result = fmt.format(summary, SummaryRenderConfig(), None)
        assert "fix_proposals" not in result
        assert result["sections"]["fixes"] is False

    def test_no_fixes_key_when_empty(self):
        """No fix_proposals key when fix proposals list is empty."""
        summary = _make_summary()
        fmt = StructuredDictFormatter()
        result = fmt.format(
            summary, SummaryRenderConfig(), FixProposalSet(proposals=[])
        )
        assert "fix_proposals" not in result
        assert result["sections"]["fixes"] is False

    def test_fixes_disabled_by_config(self):
        """No fix_proposals key when show_fixes is False."""
        summary = _make_summary()
        fixes = _make_fix_proposals()
        config = SummaryRenderConfig(show_fixes=False)
        fmt = StructuredDictFormatter()
        result = fmt.format(summary, config, fixes)
        assert "fix_proposals" not in result
        assert result["sections"]["fixes"] is False

    def test_max_fixes_shown_in_structured(self):
        """Only max_fixes_shown proposals rendered in structured."""
        summary = _make_summary()
        fixes = _make_fix_proposals()
        config = SummaryRenderConfig(max_fixes_shown=1)
        fmt = StructuredDictFormatter()
        result = fmt.format(summary, config, fixes)

        fp = result["fix_proposals"]
        assert fp["total"] == 2
        assert fp["shown"] == 1
        assert len(fp["proposals"]) == 1

    def test_budget_exhausted_in_structured(self):
        """Budget exhausted flag propagated to structured dict."""
        summary = _make_summary()
        fixes = _make_fix_proposals(budget_exhausted=True)
        fmt = StructuredDictFormatter()
        result = fmt.format(summary, SummaryRenderConfig(), fixes)
        assert result["fix_proposals"]["budget_exhausted"] is True


# ---------------------------------------------------------------------------
# FinalSummaryRenderer — unified entry point
# ---------------------------------------------------------------------------


class TestFinalSummaryRenderer:
    """Tests for the FinalSummaryRenderer unified entry point."""

    def test_render_text_with_report(self):
        """render_text accepts FinalSummaryReport."""
        report = _make_report(fix_proposals=_make_fix_proposals())
        renderer = FinalSummaryRenderer()
        output = renderer.render_text(report)
        assert "Proposed Fixes" in output
        assert "FAILED" in output

    def test_render_text_with_plain_summary(self):
        """render_text accepts plain TestRunSummary (backwards compat)."""
        summary = _make_summary()
        renderer = FinalSummaryRenderer()
        output = renderer.render_text(summary)
        assert "FAILED" in output
        assert "Proposed Fixes" not in output

    def test_render_rich_with_report(self):
        """render_rich accepts FinalSummaryReport."""
        report = _make_report(fix_proposals=_make_fix_proposals())
        renderer = FinalSummaryRenderer()
        output = renderer.render_rich(report)
        assert "Proposed Fixes" in output

    def test_render_rich_with_plain_summary(self):
        """render_rich accepts plain TestRunSummary."""
        summary = _make_summary()
        renderer = FinalSummaryRenderer()
        output = renderer.render_rich(summary)
        assert "failed" in output.lower()

    def test_render_structured_with_report(self):
        """render_structured accepts FinalSummaryReport."""
        report = _make_report(fix_proposals=_make_fix_proposals())
        renderer = FinalSummaryRenderer()
        result = renderer.render_structured(report)
        assert "fix_proposals" in result

    def test_render_structured_with_plain_summary(self):
        """render_structured accepts plain TestRunSummary."""
        summary = _make_summary()
        renderer = FinalSummaryRenderer()
        result = renderer.render_structured(summary)
        assert "fix_proposals" not in result

    def test_render_to_console_with_report(self):
        """render_to_console accepts FinalSummaryReport."""
        from io import StringIO
        from rich.console import Console

        report = _make_report(fix_proposals=_make_fix_proposals())
        renderer = FinalSummaryRenderer()

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)
        renderer.render_to_console(report, console)
        output = buf.getvalue()
        assert "Proposed Fixes" in output

    def test_config_override(self):
        """Config override works for individual render calls."""
        report = _make_report(fix_proposals=_make_fix_proposals())
        renderer = FinalSummaryRenderer()
        no_fixes_config = SummaryRenderConfig(show_fixes=False)
        output = renderer.render_text(report, config=no_fixes_config)
        assert "Proposed Fixes" not in output

    def test_config_setter(self):
        """Config setter updates instance config."""
        renderer = FinalSummaryRenderer()
        new_config = SummaryRenderConfig(show_header=False)
        renderer.config = new_config
        assert renderer.config.show_header is False

    def test_full_end_to_end_all_sections(self):
        """End-to-end: all sections composed into a single output."""
        summary = _make_summary(
            ai_analysis="Pattern: assertion drift in module A.",
            metadata={"target": "local", "python": "3.12"},
        )
        fixes = _make_fix_proposals()
        report = FinalSummaryReport(summary=summary, fix_proposals=fixes)
        config = SummaryRenderConfig(show_metadata=True)
        renderer = FinalSummaryRenderer(config=config)

        text = renderer.render_text(report)
        # Verify all five sections are present:
        # 1. Counts
        assert "7 passed" in text
        assert "2 failed" in text
        # 2. Failures
        assert "test_alpha" in text
        assert "test_beta" in text
        # 3. Timing
        assert "5.50s" in text
        assert "pytest" in text
        # 4. AI Analysis
        assert "assertion drift" in text
        # 5. Fixes
        assert "Proposed Fixes" in text
        assert "[HIGH]" in text
        assert "[MEDIUM]" in text
        # 6. Metadata
        assert "target: local" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the final summary formatter."""

    def test_empty_summary_with_fixes(self):
        """Fixes section renders even when summary has no tests."""
        summary = TestRunSummary()
        fixes = _make_fix_proposals()
        report = FinalSummaryReport(summary=summary, fix_proposals=fixes)
        renderer = FinalSummaryRenderer()
        output = renderer.render_text(report)
        assert "No tests were executed" in output
        # Fixes should still render because they may come from a
        # previous run's analysis
        assert "Proposed Fixes" in output

    def test_fix_with_no_changes(self):
        """Fix proposal with no proposed_changes renders without error."""
        proposal = FixProposal(
            failure_id="test::id",
            title="Investigate flaky test",
            description="Test appears flaky.",
            category=FailureCategory.UNKNOWN,
            confidence=FixConfidence.LOW,
            confidence_score=0.2,
        )
        fixes = _make_fix_proposals(proposals=[proposal])
        summary = _make_summary()
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)
        assert "Investigate flaky test" in output
        assert "[LOW]" in output
        # No "Changes:" line since no proposed changes
        assert "proposed change(s)" not in output

    def test_many_affected_files_truncated(self):
        """Many affected files are truncated in plain text output."""
        proposal = FixProposal(
            failure_id="test::id",
            title="Multi-file fix",
            description="Complex fix across many files",
            confidence=FixConfidence.MEDIUM,
            confidence_score=0.6,
            affected_files=[f"src/module_{i}.py" for i in range(10)],
        )
        fixes = _make_fix_proposals(proposals=[proposal])
        summary = _make_summary()
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)
        assert "(+5 more)" in output

    def test_sorted_by_confidence(self):
        """Proposals are sorted by confidence in output."""
        low = _make_fix_proposal(
            failure_id="t1", title="Low fix", confidence=FixConfidence.LOW, score=0.2
        )
        high = _make_fix_proposal(
            failure_id="t2", title="High fix", confidence=FixConfidence.HIGH, score=0.9
        )
        medium = _make_fix_proposal(
            failure_id="t3", title="Medium fix", confidence=FixConfidence.MEDIUM, score=0.5
        )
        fixes = _make_fix_proposals(proposals=[low, high, medium])
        summary = _make_summary()
        fmt = PlainTextFormatter()
        output = fmt.format(summary, SummaryRenderConfig(), fixes)

        # High should come before Medium, Medium before Low
        high_pos = output.index("High fix")
        medium_pos = output.index("Medium fix")
        low_pos = output.index("Low fix")
        assert high_pos < medium_pos < low_pos
