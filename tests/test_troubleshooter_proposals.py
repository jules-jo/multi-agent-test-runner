"""Tests for troubleshooter fix proposal generation.

Tests the full pipeline: failure classification → strategy selection →
fix proposal generation → proposal set aggregation.
"""

from __future__ import annotations

import pytest

from test_runner.agents.troubleshooter.analyzer import (
    AnalyzerConfig,
    FailureAnalyzer,
    StrategyRegistry,
    classify_failure,
    create_default_registry,
)
from test_runner.agents.troubleshooter.models import (
    FailureCategory,
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.agents.troubleshooter.agent import TroubleshooterAgent
from test_runner.models.summary import FailureDetail, TestOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure(
    *,
    test_id: str = "tests/test_foo.py::test_bar",
    test_name: str = "test_bar",
    error_message: str = "",
    error_type: str = "",
    stack_trace: str = "",
    stdout: str = "",
    stderr: str = "",
    file_path: str = "",
    line_number: int | None = None,
    outcome: TestOutcome = TestOutcome.FAILED,
) -> FailureDetail:
    return FailureDetail(
        test_id=test_id,
        test_name=test_name,
        outcome=outcome,
        error_message=error_message,
        error_type=error_type,
        stack_trace=stack_trace,
        stdout=stdout,
        stderr=stderr,
        file_path=file_path,
        line_number=line_number,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestProposedChange:
    def test_has_diff_true(self):
        change = ProposedChange(
            file_path="foo.py",
            description="Fix import",
            original_snippet="import foo",
            proposed_snippet="import bar",
        )
        assert change.has_diff is True

    def test_has_diff_false_no_original(self):
        change = ProposedChange(
            file_path="foo.py",
            description="Add import",
            proposed_snippet="import bar",
        )
        assert change.has_diff is False


class TestFixProposal:
    def test_summary_line(self):
        proposal = FixProposal(
            failure_id="test_a",
            title="Fix import error",
            description="desc",
            confidence=FixConfidence.HIGH,
            affected_files=["a.py", "b.py"],
        )
        line = proposal.summary_line()
        assert "[HIGH]" in line
        assert "Fix import error" in line
        assert "a.py" in line

    def test_is_actionable_with_changes(self):
        proposal = FixProposal(
            failure_id="test_a",
            title="Fix",
            description="desc",
            proposed_changes=[
                ProposedChange(file_path="a.py", description="fix")
            ],
        )
        assert proposal.is_actionable is True

    def test_is_actionable_with_user_action(self):
        proposal = FixProposal(
            failure_id="test_a",
            title="Fix",
            description="desc",
            requires_user_action=True,
        )
        assert proposal.is_actionable is True

    def test_not_actionable(self):
        proposal = FixProposal(
            failure_id="test_a",
            title="Investigate",
            description="desc",
        )
        assert proposal.is_actionable is False

    def test_change_count(self):
        proposal = FixProposal(
            failure_id="test_a",
            title="Fix",
            description="desc",
            proposed_changes=[
                ProposedChange(file_path="a.py", description="fix1"),
                ProposedChange(file_path="b.py", description="fix2"),
            ],
        )
        assert proposal.change_count == 2

    def test_summary_line_many_files(self):
        proposal = FixProposal(
            failure_id="test_a",
            title="Fix",
            description="desc",
            confidence=FixConfidence.MEDIUM,
            affected_files=["a.py", "b.py", "c.py", "d.py", "e.py"],
        )
        line = proposal.summary_line()
        assert "+2 more" in line


class TestFixProposalSet:
    def test_by_confidence_ordering(self):
        proposals = [
            FixProposal(failure_id="a", title="low", description="d", confidence=FixConfidence.LOW, confidence_score=0.2),
            FixProposal(failure_id="b", title="high", description="d", confidence=FixConfidence.HIGH, confidence_score=0.9),
            FixProposal(failure_id="c", title="med", description="d", confidence=FixConfidence.MEDIUM, confidence_score=0.6),
        ]
        pset = FixProposalSet(proposals=proposals)
        ordered = pset.by_confidence()
        assert ordered[0].confidence == FixConfidence.HIGH
        assert ordered[1].confidence == FixConfidence.MEDIUM
        assert ordered[2].confidence == FixConfidence.LOW

    def test_for_failure(self):
        proposals = [
            FixProposal(failure_id="test_a", title="fix a", description="d"),
            FixProposal(failure_id="test_b", title="fix b", description="d"),
            FixProposal(failure_id="test_a", title="alt fix a", description="d"),
        ]
        pset = FixProposalSet(proposals=proposals)
        matches = pset.for_failure("test_a")
        assert len(matches) == 2
        assert all(p.failure_id == "test_a" for p in matches)

    def test_high_confidence_count(self):
        proposals = [
            FixProposal(failure_id="a", title="t", description="d", confidence=FixConfidence.HIGH),
            FixProposal(failure_id="b", title="t", description="d", confidence=FixConfidence.HIGH),
            FixProposal(failure_id="c", title="t", description="d", confidence=FixConfidence.LOW),
        ]
        pset = FixProposalSet(proposals=proposals)
        assert pset.high_confidence_count == 2

    def test_actionable_count(self):
        proposals = [
            FixProposal(
                failure_id="a", title="t", description="d",
                proposed_changes=[ProposedChange(file_path="x.py", description="fix")],
            ),
            FixProposal(failure_id="b", title="t", description="d"),
        ]
        pset = FixProposalSet(proposals=proposals)
        assert pset.actionable_count == 1

    def test_summary_lines(self):
        proposals = [
            FixProposal(failure_id="a", title="Fix A", description="d", confidence=FixConfidence.HIGH),
        ]
        pset = FixProposalSet(proposals=proposals)
        lines = pset.summary_lines()
        assert len(lines) == 1
        assert "Fix A" in lines[0]


# ---------------------------------------------------------------------------
# Failure classification tests
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    def test_import_error(self):
        f = _make_failure(error_type="ModuleNotFoundError", error_message="No module named 'foo'")
        assert classify_failure(f) == FailureCategory.IMPORT_ERROR

    def test_syntax_error(self):
        f = _make_failure(error_type="SyntaxError", error_message="invalid syntax")
        assert classify_failure(f) == FailureCategory.SYNTAX_ERROR

    def test_assertion_error(self):
        f = _make_failure(error_type="AssertionError", error_message="assert 1 == 2")
        assert classify_failure(f) == FailureCategory.ASSERTION

    def test_type_error(self):
        f = _make_failure(error_type="TypeError", error_message="expected int got str")
        assert classify_failure(f) == FailureCategory.TYPE_ERROR

    def test_attribute_error(self):
        f = _make_failure(error_type="AttributeError", error_message="has no attribute 'foo'")
        assert classify_failure(f) == FailureCategory.ATTRIBUTE_ERROR

    def test_timeout(self):
        f = _make_failure(error_message="test timed out after 30s")
        assert classify_failure(f) == FailureCategory.TIMEOUT

    def test_fixture_error(self):
        f = _make_failure(error_message="fixture 'db' not found")
        assert classify_failure(f) == FailureCategory.FIXTURE_ERROR

    def test_unknown(self):
        f = _make_failure(error_message="something went wrong")
        assert classify_failure(f) == FailureCategory.UNKNOWN

    def test_stderr_classification(self):
        f = _make_failure(stderr="ImportError: No module named 'requests'")
        assert classify_failure(f) == FailureCategory.IMPORT_ERROR

    def test_stack_trace_classification(self):
        f = _make_failure(stack_trace="Traceback...\nTypeError: unsupported operand")
        assert classify_failure(f) == FailureCategory.TYPE_ERROR


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------


class TestFailureAnalyzer:
    def test_analyze_import_error(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(
            error_type="ModuleNotFoundError",
            error_message="No module named 'requests'",
            file_path="src/app.py",
            stack_trace="Traceback...\n  File src/app.py:10\nModuleNotFoundError",
        )
        result = analyzer.analyze_failures([failure])
        assert result.total_failures_analyzed == 1
        assert result.total_proposals_generated == 1

        proposal = result.proposals[0]
        assert proposal.category == FailureCategory.IMPORT_ERROR
        assert "requests" in proposal.title
        assert proposal.requires_user_action is True
        assert "pip install" in proposal.user_action_description

    def test_analyze_assertion_error(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(
            error_type="AssertionError",
            error_message="assert 1 == 2",
            file_path="tests/test_math.py",
            line_number=42,
            stack_trace="Traceback...\nassert 1 == 2",
        )
        result = analyzer.analyze_failures([failure])
        assert result.total_proposals_generated == 1
        proposal = result.proposals[0]
        assert proposal.category == FailureCategory.ASSERTION
        assert len(proposal.proposed_changes) > 0
        assert proposal.proposed_changes[0].line_start == 42

    def test_analyze_syntax_error_high_confidence(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(
            error_type="SyntaxError",
            error_message="invalid syntax",
            file_path="src/broken.py",
            line_number=15,
            stack_trace="File 'src/broken.py', line 15\nSyntaxError",
        )
        result = analyzer.analyze_failures([failure])
        proposal = result.proposals[0]
        assert proposal.category == FailureCategory.SYNTAX_ERROR
        assert proposal.confidence == FixConfidence.HIGH
        assert proposal.confidence_score >= 0.85

    def test_analyze_type_error(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(
            error_type="TypeError",
            error_message="expected int got str",
            file_path="src/calc.py",
            stack_trace="TypeError: expected int got str",
        )
        result = analyzer.analyze_failures([failure])
        assert result.total_proposals_generated == 1
        assert result.proposals[0].category == FailureCategory.TYPE_ERROR

    def test_analyze_attribute_error(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(
            error_type="AttributeError",
            error_message="'NoneType' object has no attribute 'split'",
            file_path="src/parser.py",
        )
        result = analyzer.analyze_failures([failure])
        proposal = result.proposals[0]
        assert proposal.category == FailureCategory.ATTRIBUTE_ERROR
        assert "split" in proposal.title

    def test_analyze_timeout(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(
            error_message="test timed out after 30s",
            file_path="tests/test_slow.py",
        )
        result = analyzer.analyze_failures([failure])
        proposal = result.proposals[0]
        assert proposal.category == FailureCategory.TIMEOUT
        assert proposal.requires_user_action is True

    def test_analyze_fixture_error(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(
            error_message="fixture 'database' not found",
            file_path="tests/conftest.py",
            stack_trace="SetupError...",
        )
        result = analyzer.analyze_failures([failure])
        assert result.proposals[0].category == FailureCategory.FIXTURE_ERROR

    def test_analyze_unknown_failure(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(
            error_message="something totally unexpected",
            file_path="src/mystery.py",
        )
        result = analyzer.analyze_failures([failure])
        proposal = result.proposals[0]
        assert proposal.confidence == FixConfidence.LOW
        assert proposal.requires_user_action is True

    def test_budget_exhaustion(self):
        config = AnalyzerConfig(max_failures_to_analyze=2)
        analyzer = FailureAnalyzer(config=config)
        failures = [
            _make_failure(test_id=f"test_{i}", error_type="TypeError", error_message="err")
            for i in range(5)
        ]
        result = analyzer.analyze_failures(failures)
        assert result.budget_exhausted is True
        assert result.total_failures_analyzed == 2
        assert result.total_proposals_generated == 2

    def test_min_confidence_filter(self):
        config = AnalyzerConfig(min_confidence_score=0.5)
        analyzer = FailureAnalyzer(config=config)
        failure = _make_failure(error_message="unknown error")
        result = analyzer.analyze_failures([failure])
        # Generic strategy yields LOW confidence (~0.25), should be filtered
        assert result.total_proposals_generated == 0

    def test_exclude_low_confidence(self):
        config = AnalyzerConfig(include_low_confidence=False)
        analyzer = FailureAnalyzer(config=config)
        failure = _make_failure(error_message="unknown error")
        result = analyzer.analyze_failures([failure])
        assert result.total_proposals_generated == 0

    def test_multiple_failures_mixed(self):
        analyzer = FailureAnalyzer()
        failures = [
            _make_failure(
                test_id="test_import",
                error_type="ModuleNotFoundError",
                error_message="No module named 'foo'",
                file_path="src/a.py",
                stack_trace="trace",
            ),
            _make_failure(
                test_id="test_syntax",
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/b.py",
                line_number=10,
                stack_trace="trace",
            ),
            _make_failure(
                test_id="test_assert",
                error_type="AssertionError",
                error_message="assert False",
                file_path="tests/test_c.py",
            ),
        ]
        result = analyzer.analyze_failures(failures)
        assert result.total_failures_analyzed == 3
        assert result.total_proposals_generated == 3

        categories = {p.category for p in result.proposals}
        assert FailureCategory.IMPORT_ERROR in categories
        assert FailureCategory.SYNTAX_ERROR in categories
        assert FailureCategory.ASSERTION in categories

    def test_analyze_single(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(
            error_type="TypeError",
            error_message="got int, expected str",
            file_path="src/x.py",
            stack_trace="trace",
        )
        proposal = analyzer.analyze_single(failure)
        assert proposal is not None
        assert proposal.category == FailureCategory.TYPE_ERROR

    def test_analysis_summary_text(self):
        analyzer = FailureAnalyzer()
        failure = _make_failure(error_type="TypeError", error_message="err")
        result = analyzer.analyze_failures([failure])
        assert "Analyzed 1" in result.analysis_summary
        assert "Generated 1" in result.analysis_summary


# ---------------------------------------------------------------------------
# Strategy registry tests
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    def test_default_registry_categories(self):
        registry = create_default_registry()
        cats = registry.registered_categories
        assert FailureCategory.IMPORT_ERROR in cats
        assert FailureCategory.ASSERTION in cats
        assert FailureCategory.SYNTAX_ERROR in cats
        assert FailureCategory.TYPE_ERROR in cats
        assert FailureCategory.ATTRIBUTE_ERROR in cats
        assert FailureCategory.TIMEOUT in cats
        assert FailureCategory.FIXTURE_ERROR in cats

    def test_fallback_to_generic(self):
        registry = create_default_registry()
        strategy = registry.get(FailureCategory.UNKNOWN)
        # Should return the generic fallback
        assert strategy is not None


# ---------------------------------------------------------------------------
# TroubleshooterAgent integration tests
# ---------------------------------------------------------------------------


class TestTroubleshooterAgentProposals:
    def test_generate_fix_proposals(self):
        agent = TroubleshooterAgent()
        failures = [
            _make_failure(
                test_id="test_import",
                error_type="ModuleNotFoundError",
                error_message="No module named 'pandas'",
                file_path="src/data.py",
                stack_trace="trace",
            ),
        ]
        result = agent.generate_fix_proposals(failures)
        assert isinstance(result, FixProposalSet)
        assert result.total_proposals_generated == 1
        assert agent.last_proposals is result
        assert agent.state.steps_taken == 1

    def test_proposals_update_state_findings(self):
        agent = TroubleshooterAgent()
        failures = [
            _make_failure(
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/bad.py",
                line_number=5,
                stack_trace="trace",
            ),
        ]
        agent.generate_fix_proposals(failures)
        assert len(agent.state.findings) == 1
        finding = agent.state.findings[0]
        assert finding["type"] == "fix_proposal"
        assert finding["category"] == "syntax_error"

    def test_proposals_in_handoff_summary(self):
        agent = TroubleshooterAgent()
        failures = [
            _make_failure(
                error_type="TypeError",
                error_message="err",
                file_path="src/x.py",
                stack_trace="trace",
            ),
        ]
        agent.generate_fix_proposals(failures)
        summary = agent.get_handoff_summary()
        assert "fix_proposals" in summary
        fp = summary["fix_proposals"]
        assert fp["total"] == 1
        assert len(fp["proposals"]) == 1

    def test_analyze_single_failure(self):
        agent = TroubleshooterAgent()
        failure = _make_failure(
            error_type="AttributeError",
            error_message="has no attribute 'run'",
            file_path="src/engine.py",
        )
        proposal = agent.analyze_single_failure(failure)
        assert proposal is not None
        assert proposal.category == FailureCategory.ATTRIBUTE_ERROR
        assert agent.state.steps_taken == 1

    def test_reset_clears_proposals(self):
        agent = TroubleshooterAgent()
        failures = [_make_failure(error_type="TypeError", error_message="err")]
        agent.generate_fix_proposals(failures)
        assert agent.last_proposals is not None
        agent.reset_state()
        assert agent.last_proposals is None
        assert agent.state.steps_taken == 0

    def test_escalation_returns_empty_proposals(self):
        agent = TroubleshooterAgent(hard_cap_steps=0)
        failures = [_make_failure(error_type="TypeError", error_message="err")]
        result = agent.generate_fix_proposals(failures)
        assert result.budget_exhausted is True
        assert result.total_proposals_generated == 0

    def test_confidence_update_after_proposals(self):
        agent = TroubleshooterAgent()
        failures = [
            _make_failure(
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/x.py",
                line_number=10,
                stack_trace="trace",
            ),
        ]
        agent.generate_fix_proposals(failures)
        # With a HIGH confidence proposal, agent confidence should be > 0.5
        assert agent.state.current_confidence > 0.5

    def test_custom_analyzer_config(self):
        config = AnalyzerConfig(max_failures_to_analyze=1)
        agent = TroubleshooterAgent(analyzer_config=config)
        failures = [
            _make_failure(test_id=f"t{i}", error_type="TypeError", error_message="err")
            for i in range(3)
        ]
        result = agent.generate_fix_proposals(failures)
        assert result.budget_exhausted is True
        assert result.total_failures_analyzed == 1

    def test_no_failures_produces_empty_set(self):
        agent = TroubleshooterAgent()
        result = agent.generate_fix_proposals([])
        assert result.total_proposals_generated == 0
        assert result.total_failures_analyzed == 0
        assert not result.budget_exhausted
