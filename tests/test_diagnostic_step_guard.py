"""Tests for the DiagnosticStepGuard and DiagnosisSummary.

Covers:
- Step counting and limit enforcement
- Lifecycle management (start, record, finalize)
- Structured DiagnosisSummary generation
- Warning thresholds and budget status
- Edge cases (zero steps, immediate limit, reset)
- Integration with TroubleshooterAgent
"""

from __future__ import annotations

import time

import pytest

from test_runner.agents.troubleshooter.step_guard import (
    DEFAULT_MAX_DIAGNOSTIC_STEPS,
    CompletionReason,
    DiagnosticStep,
    DiagnosticStepGuard,
    DiagnosisSummary,
    _DIAGNOSTIC_WARNING_FRACTION,
)


# ---------------------------------------------------------------------------
# DiagnosticStep model tests
# ---------------------------------------------------------------------------


class TestDiagnosticStep:
    """Test the DiagnosticStep model."""

    def test_basic_creation(self):
        step = DiagnosticStep(
            step_number=1,
            action="read_failure_log",
            target="test_foo.py",
            finding="ImportError on line 42",
        )
        assert step.step_number == 1
        assert step.action == "read_failure_log"
        assert step.target == "test_foo.py"
        assert step.finding == "ImportError on line 42"
        assert step.confidence_delta == 0.0
        assert step.timestamp > 0

    def test_frozen(self):
        step = DiagnosticStep(step_number=1, action="test")
        with pytest.raises(Exception):  # ValidationError for frozen model
            step.action = "changed"  # type: ignore[misc]

    def test_metadata(self):
        step = DiagnosticStep(
            step_number=1,
            action="test",
            metadata={"key": "value"},
        )
        assert step.metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# DiagnosisSummary model tests
# ---------------------------------------------------------------------------


class TestDiagnosisSummary:
    """Test the DiagnosisSummary model."""

    def test_basic_creation(self):
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=5,
            max_steps=10,
            root_cause="Missing dependency",
            confidence=0.85,
        )
        assert summary.completion_reason == CompletionReason.COMPLETED
        assert summary.total_steps == 5
        assert summary.max_steps == 10
        assert summary.root_cause == "Missing dependency"
        assert summary.confidence == 0.85

    def test_budget_used_fraction(self):
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=7,
            max_steps=10,
        )
        assert summary.budget_used_fraction == 0.7

    def test_budget_used_fraction_zero_max(self):
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=0,
            max_steps=0,
        )
        assert summary.budget_used_fraction == 1.0

    def test_is_conclusive_true(self):
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=5,
            max_steps=10,
            root_cause="Missing module 'requests'",
            confidence=0.85,
        )
        assert summary.is_conclusive is True

    def test_is_conclusive_false_no_root_cause(self):
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=5,
            max_steps=10,
            confidence=0.85,
        )
        assert summary.is_conclusive is False

    def test_is_conclusive_false_low_confidence(self):
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=5,
            max_steps=10,
            root_cause="Possible issue",
            confidence=0.3,
        )
        assert summary.is_conclusive is False

    def test_was_truncated(self):
        truncated = DiagnosisSummary(
            completion_reason=CompletionReason.LIMIT_REACHED,
            total_steps=10,
            max_steps=10,
        )
        assert truncated.was_truncated is True

        completed = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=5,
            max_steps=10,
        )
        assert completed.was_truncated is False

    def test_duration_seconds(self):
        now = time.time()
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=5,
            max_steps=10,
            start_time=now - 5.0,
            end_time=now,
        )
        assert 4.9 <= summary.duration_seconds <= 5.1

    def test_summary_line(self):
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=5,
            max_steps=10,
            root_cause="Test issue",
            confidence=0.85,
        )
        line = summary.summary_line()
        assert "CONCLUSIVE" in line
        assert "5/10" in line
        assert "85%" in line

    def test_summary_line_inconclusive(self):
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.LIMIT_REACHED,
            total_steps=10,
            max_steps=10,
            confidence=0.3,
        )
        line = summary.summary_line()
        assert "INCONCLUSIVE" in line
        assert "limit reached" in line

    def test_to_report_dict(self):
        summary = DiagnosisSummary(
            completion_reason=CompletionReason.COMPLETED,
            total_steps=3,
            max_steps=10,
            root_cause="Missing dep",
            confidence=0.9,
            evidence=["ImportError in logs"],
            proposed_fixes=["pip install foo"],
            steps=[
                DiagnosticStep(step_number=1, action="read_log", finding="ImportError"),
            ],
        )
        d = summary.to_report_dict()
        assert d["completion_reason"] == "completed"
        assert d["total_steps"] == 3
        assert d["max_steps"] == 10
        assert d["root_cause"] == "Missing dep"
        assert d["confidence"] == 0.9
        assert d["is_conclusive"] is True
        assert d["was_truncated"] is False
        assert len(d["steps"]) == 1
        assert d["steps"][0]["action"] == "read_log"


# ---------------------------------------------------------------------------
# DiagnosticStepGuard core tests
# ---------------------------------------------------------------------------


class TestDiagnosticStepGuardInit:
    """Test DiagnosticStepGuard initialization."""

    def test_default_max_steps(self):
        guard = DiagnosticStepGuard()
        assert guard.max_steps == DEFAULT_MAX_DIAGNOSTIC_STEPS
        assert guard.max_steps == 10

    def test_custom_max_steps(self):
        guard = DiagnosticStepGuard(max_steps=5)
        assert guard.max_steps == 5

    def test_invalid_max_steps(self):
        with pytest.raises(ValueError, match="max_steps must be >= 1"):
            DiagnosticStepGuard(max_steps=0)

    def test_initial_state(self):
        guard = DiagnosticStepGuard()
        assert guard.steps_taken == 0
        assert guard.remaining == 10
        assert guard.is_at_limit is False
        assert guard.is_started is False
        assert guard.is_finalized is False
        assert guard.current_confidence == 0.0


class TestDiagnosticStepGuardLifecycle:
    """Test the start/record/finalize lifecycle."""

    def test_start(self):
        guard = DiagnosticStepGuard()
        guard.start()
        assert guard.is_started is True
        assert guard.is_finalized is False
        assert guard.steps_taken == 0

    def test_record_step_without_start_raises(self):
        guard = DiagnosticStepGuard()
        with pytest.raises(RuntimeError, match="start.*must be called"):
            guard.record_step("test_action")

    def test_finalize_without_start_raises(self):
        guard = DiagnosticStepGuard()
        with pytest.raises(RuntimeError, match="start.*must be called"):
            guard.finalize()

    def test_double_finalize_raises(self):
        guard = DiagnosticStepGuard()
        guard.start()
        guard.finalize()
        with pytest.raises(RuntimeError, match="already been finalized"):
            guard.finalize()

    def test_record_after_finalize_raises(self):
        guard = DiagnosticStepGuard()
        guard.start()
        guard.finalize()
        with pytest.raises(RuntimeError, match="has been finalized"):
            guard.record_step("action")

    def test_can_proceed_before_start(self):
        guard = DiagnosticStepGuard()
        assert guard.can_proceed() is False

    def test_can_proceed_after_finalize(self):
        guard = DiagnosticStepGuard()
        guard.start()
        guard.finalize()
        assert guard.can_proceed() is False


class TestDiagnosticStepGuardStepCounting:
    """Test step counting and limit enforcement."""

    def test_record_steps(self):
        guard = DiagnosticStepGuard(max_steps=5)
        guard.start()
        assert guard.record_step("step1") is True
        assert guard.steps_taken == 1
        assert guard.remaining == 4

    def test_limit_enforcement(self):
        guard = DiagnosticStepGuard(max_steps=3)
        guard.start()
        assert guard.record_step("step1") is True
        assert guard.record_step("step2") is True
        assert guard.record_step("step3") is True
        # Limit reached
        assert guard.is_at_limit is True
        assert guard.can_proceed() is False
        assert guard.record_step("step4") is False
        # Only 3 steps recorded
        assert guard.steps_taken == 3

    def test_ten_step_default_limit(self):
        """Verify the ~10 step default limit works correctly."""
        guard = DiagnosticStepGuard()  # Default is 10
        guard.start()
        for i in range(10):
            assert guard.record_step(f"step_{i+1}") is True
        # 11th step should be rejected
        assert guard.record_step("step_11") is False
        assert guard.steps_taken == 10
        assert guard.is_at_limit is True

    def test_usage_fraction(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1")
        assert guard.usage_fraction == 0.1
        for _ in range(4):
            guard.record_step("s")
        assert guard.usage_fraction == 0.5

    def test_confidence_tracking(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1", confidence_delta=0.3)
        assert guard.current_confidence == pytest.approx(0.3)
        guard.record_step("s2", confidence_delta=0.2)
        assert guard.current_confidence == pytest.approx(0.5)
        guard.record_step("s3", confidence_delta=-0.1)
        assert guard.current_confidence == pytest.approx(0.4)

    def test_confidence_clamped_to_zero(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1", confidence_delta=-0.5)
        assert guard.current_confidence == 0.0

    def test_confidence_clamped_to_one(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1", confidence_delta=1.5)
        assert guard.current_confidence == 1.0

    def test_evidence_accumulation(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1", finding="Found ImportError")
        guard.record_step("s2", finding="Missing 'requests' module")
        guard.record_step("s3")  # No finding
        summary = guard.finalize()
        assert len(summary.evidence) == 2
        assert "Found ImportError" in summary.evidence
        assert "Missing 'requests' module" in summary.evidence


class TestDiagnosticStepGuardWarning:
    """Test warning threshold behavior."""

    def test_warning_at_threshold(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        # Warning at 70% = step 7
        for i in range(6):
            guard.record_step(f"s{i+1}")
        assert guard.is_warning is False
        guard.record_step("s7")
        assert guard.is_warning is True

    def test_budget_status_ok(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1")
        msg = guard.budget_status_message()
        assert "DIAGNOSTIC OK" in msg
        assert "1/10" in msg

    def test_budget_status_warning(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        for i in range(8):
            guard.record_step(f"s{i+1}")
        msg = guard.budget_status_message()
        assert "DIAGNOSTIC WARNING" in msg

    def test_budget_status_limit_reached(self):
        guard = DiagnosticStepGuard(max_steps=3)
        guard.start()
        for i in range(3):
            guard.record_step(f"s{i+1}")
        msg = guard.budget_status_message()
        assert "DIAGNOSTIC LIMIT REACHED" in msg


class TestDiagnosticStepGuardFinalize:
    """Test finalization and summary generation."""

    def test_finalize_completed(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1", finding="Found the bug")
        summary = guard.finalize(
            root_cause="Missing dependency",
            confidence=0.85,
            proposed_fixes=["pip install requests"],
        )
        assert summary.completion_reason == CompletionReason.COMPLETED
        assert summary.total_steps == 1
        assert summary.max_steps == 10
        assert summary.root_cause == "Missing dependency"
        assert summary.confidence == 0.85
        assert summary.proposed_fixes == ["pip install requests"]
        assert summary.evidence == ["Found the bug"]
        assert summary.is_conclusive is True
        assert summary.was_truncated is False

    def test_finalize_limit_reached_auto_detected(self):
        guard = DiagnosticStepGuard(max_steps=2)
        guard.start()
        guard.record_step("s1")
        guard.record_step("s2")
        summary = guard.finalize()
        assert summary.completion_reason == CompletionReason.LIMIT_REACHED
        assert summary.was_truncated is True

    def test_finalize_with_explicit_reason(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        summary = guard.finalize(reason=CompletionReason.ERROR)
        assert summary.completion_reason == CompletionReason.ERROR

    def test_finalize_manual_stop(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1")
        summary = guard.finalize(reason=CompletionReason.MANUAL_STOP)
        assert summary.completion_reason == CompletionReason.MANUAL_STOP

    def test_finalize_uses_running_confidence_by_default(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1", confidence_delta=0.6)
        summary = guard.finalize()
        assert summary.confidence == pytest.approx(0.6)

    def test_finalize_overrides_confidence(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1", confidence_delta=0.6)
        summary = guard.finalize(confidence=0.9)
        assert summary.confidence == 0.9

    def test_finalize_includes_all_steps(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("read_log", target="err.log", finding="Error found")
        guard.record_step("read_src", target="app.py", finding="Bug on line 10")
        summary = guard.finalize()
        assert len(summary.steps) == 2
        assert summary.steps[0].action == "read_log"
        assert summary.steps[0].target == "err.log"
        assert summary.steps[1].action == "read_src"

    def test_finalize_with_all_fields(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1", finding="evidence1")
        summary = guard.finalize(
            root_cause="Root cause",
            confidence=0.75,
            proposed_fixes=["fix1", "fix2"],
            alternative_causes=["alt1"],
            unresolved_questions=["q1"],
            metadata={"key": "value"},
        )
        assert summary.root_cause == "Root cause"
        assert summary.proposed_fixes == ["fix1", "fix2"]
        assert summary.alternative_causes == ["alt1"]
        assert summary.unresolved_questions == ["q1"]
        assert summary.metadata == {"key": "value"}

    def test_finalize_has_timestamps(self):
        guard = DiagnosticStepGuard(max_steps=10)
        before = time.time()
        guard.start()
        guard.record_step("s1")
        summary = guard.finalize()
        after = time.time()
        assert summary.start_time >= before
        assert summary.end_time <= after
        assert summary.duration_seconds >= 0


class TestDiagnosticStepGuardReset:
    """Test reset behavior."""

    def test_reset_clears_state(self):
        guard = DiagnosticStepGuard(max_steps=5)
        guard.start()
        guard.record_step("s1")
        guard.record_step("s2")
        guard.finalize()

        guard.reset()
        assert guard.steps_taken == 0
        assert guard.is_started is False
        assert guard.is_finalized is False
        assert guard.current_confidence == 0.0

    def test_can_reuse_after_reset(self):
        guard = DiagnosticStepGuard(max_steps=3)
        guard.start()
        guard.record_step("s1")
        guard.finalize()

        guard.reset()
        guard.start()
        assert guard.record_step("s1") is True
        assert guard.steps_taken == 1
        summary = guard.finalize(root_cause="New root cause")
        assert summary.root_cause == "New root cause"

    def test_start_acts_as_reset(self):
        guard = DiagnosticStepGuard(max_steps=5)
        guard.start()
        guard.record_step("s1")
        # Starting again resets
        guard.start()
        assert guard.steps_taken == 0
        assert guard.is_started is True
        assert guard.is_finalized is False


# ---------------------------------------------------------------------------
# Integration with TroubleshooterAgent
# ---------------------------------------------------------------------------


class TestTroubleshooterAgentDiagnosticGuard:
    """Test DiagnosticStepGuard integration with TroubleshooterAgent."""

    def test_agent_has_diagnostic_guard(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent()
        assert agent.diagnostic_guard is not None
        assert agent.diagnostic_guard.max_steps == DEFAULT_MAX_DIAGNOSTIC_STEPS

    def test_agent_custom_diagnostic_steps(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent(max_diagnostic_steps=5)
        assert agent.diagnostic_guard.max_steps == 5

    def test_agent_diagnostic_session_lifecycle(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent(max_diagnostic_steps=3)

        # Start session
        agent.start_diagnostic_session()
        assert agent.can_continue_diagnosis() is True

        # Record steps
        assert agent.record_diagnostic_step("read_log", finding="Error found") is True
        assert agent.record_diagnostic_step("read_src", finding="Bug on line 10") is True
        assert agent.record_diagnostic_step("check_env") is True

        # Limit reached
        assert agent.can_continue_diagnosis() is False
        assert agent.record_diagnostic_step("extra_step") is False

    def test_agent_finalize_diagnosis(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent(max_diagnostic_steps=10)
        agent.start_diagnostic_session()
        agent.record_diagnostic_step("read_log", finding="ImportError", confidence_delta=0.4)
        agent.record_diagnostic_step("read_src", finding="Missing import", confidence_delta=0.3)

        summary = agent.finalize_diagnosis(
            root_cause="Missing dependency 'requests'",
            confidence=0.85,
            proposed_fixes=["pip install requests"],
        )

        assert summary.completion_reason == CompletionReason.COMPLETED
        assert summary.total_steps == 2
        assert summary.root_cause == "Missing dependency 'requests'"
        assert summary.confidence == 0.85
        assert summary.is_conclusive is True

        # Also stored on agent
        assert agent.last_diagnosis_summary is summary

        # Legacy diagnosis also recorded
        assert agent.diagnosis is not None
        assert agent.diagnosis["root_cause"] == "Missing dependency 'requests'"

    def test_agent_finalize_at_limit(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent(max_diagnostic_steps=2)
        agent.start_diagnostic_session()
        agent.record_diagnostic_step("s1", confidence_delta=0.3)
        agent.record_diagnostic_step("s2", confidence_delta=0.2)

        summary = agent.finalize_diagnosis(
            root_cause="Partial diagnosis",
            unresolved_questions=["Need to check more files"],
        )

        assert summary.completion_reason == CompletionReason.LIMIT_REACHED
        assert summary.was_truncated is True
        assert summary.unresolved_questions == ["Need to check more files"]

    def test_agent_reset_clears_diagnostic_guard(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent(max_diagnostic_steps=10)
        agent.start_diagnostic_session()
        agent.record_diagnostic_step("s1")
        agent.finalize_diagnosis(root_cause="test")

        agent.reset_state()
        assert agent.last_diagnosis_summary is None
        assert agent.diagnostic_guard.steps_taken == 0
        assert agent.diagnostic_guard.is_started is False

    def test_agent_handoff_includes_diagnosis_summary(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent(max_diagnostic_steps=10)
        agent.start_diagnostic_session()
        agent.record_diagnostic_step("s1", finding="Evidence")
        agent.finalize_diagnosis(
            root_cause="Bug in code",
            confidence=0.8,
            proposed_fixes=["Fix the bug"],
        )

        handoff = agent.get_handoff_summary()
        assert "diagnosis_summary" in handoff
        ds = handoff["diagnosis_summary"]
        assert ds["root_cause"] == "Bug in code"
        assert ds["confidence"] == 0.8
        assert ds["total_steps"] == 1

    def test_agent_budget_status(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent(max_diagnostic_steps=10)
        agent.start_diagnostic_session()
        agent.record_diagnostic_step("s1")
        status = agent.diagnostic_budget_status
        assert "DIAGNOSTIC OK" in status
        assert "1/10" in status


# ---------------------------------------------------------------------------
# CompletionReason enum tests
# ---------------------------------------------------------------------------


class TestCompletionReason:
    """Test CompletionReason enum."""

    def test_values(self):
        assert CompletionReason.COMPLETED.value == "completed"
        assert CompletionReason.LIMIT_REACHED.value == "limit_reached"
        assert CompletionReason.ERROR.value == "error"
        assert CompletionReason.MANUAL_STOP.value == "manual_stop"

    def test_string_enum(self):
        assert str(CompletionReason.COMPLETED) == "CompletionReason.COMPLETED"
        assert CompletionReason.COMPLETED == "completed"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDiagnosticStepGuardEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_step_limit(self):
        guard = DiagnosticStepGuard(max_steps=1)
        guard.start()
        assert guard.record_step("only_step") is True
        assert guard.is_at_limit is True
        assert guard.can_proceed() is False
        summary = guard.finalize()
        assert summary.total_steps == 1
        assert summary.completion_reason == CompletionReason.LIMIT_REACHED

    def test_finalize_with_zero_steps(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        summary = guard.finalize(root_cause="Obvious issue")
        assert summary.total_steps == 0
        assert summary.completion_reason == CompletionReason.COMPLETED
        assert summary.root_cause == "Obvious issue"

    def test_steps_property_returns_copy(self):
        guard = DiagnosticStepGuard(max_steps=10)
        guard.start()
        guard.record_step("s1")
        steps = guard.steps
        assert len(steps) == 1
        # Modifying returned list doesn't affect guard
        steps.clear()
        assert guard.steps_taken == 1

    def test_large_max_steps(self):
        guard = DiagnosticStepGuard(max_steps=1000)
        guard.start()
        for i in range(100):
            guard.record_step(f"s{i}")
        assert guard.steps_taken == 100
        assert guard.remaining == 900
