"""Tests for post-execution evidence-based signal collectors.

Covers:
- ExecutionEvidence construction (including from_execution_result factory)
- ExitCodeSignalCollector
- OutputPatternSignalCollector
- TimingSignalCollector
- AssertionCountSignalCollector
- InfrastructureHealthSignalCollector
- collect_execution_signals convenience function
"""

from __future__ import annotations

import pytest

from test_runner.confidence.signals import (
    AssertionCountSignalCollector,
    ExecutionEvidence,
    ExecutionSignalCollector,
    ExitCodeSignalCollector,
    InfrastructureHealthSignalCollector,
    OutputPatternSignalCollector,
    TimingSignalCollector,
    _extract_counts,
    collect_execution_signals,
)
from test_runner.models.confidence import ConfidenceModel, ConfidenceSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_evidence(
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_seconds: float = 1.5,
    command: str = "pytest tests/",
    timed_out: bool = False,
    framework: str = "",
) -> ExecutionEvidence:
    return ExecutionEvidence(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration_seconds,
        command=command,
        timed_out=timed_out,
        framework=framework,
    )


# ---------------------------------------------------------------------------
# ExecutionEvidence
# ---------------------------------------------------------------------------


class TestExecutionEvidence:
    def test_basic_construction(self):
        ev = make_evidence(exit_code=0, stdout="3 passed", stderr="", duration_seconds=1.0)
        assert ev.exit_code == 0
        assert ev.stdout == "3 passed"
        assert ev.stderr == ""
        assert ev.duration_seconds == 1.0

    def test_combined_output_both(self):
        ev = make_evidence(stdout="stdout line", stderr="stderr line")
        assert "stdout line" in ev.combined_output
        assert "stderr line" in ev.combined_output

    def test_combined_output_stdout_only(self):
        ev = make_evidence(stdout="only stdout", stderr="")
        assert ev.combined_output == "only stdout"

    def test_combined_output_stderr_only(self):
        ev = make_evidence(stdout="", stderr="only stderr")
        assert ev.combined_output == "only stderr"

    def test_combined_output_both_empty(self):
        ev = make_evidence(stdout="", stderr="")
        assert ev.combined_output == ""

    def test_frozen(self):
        ev = make_evidence()
        with pytest.raises((AttributeError, TypeError)):
            ev.exit_code = 1  # type: ignore[misc]

    def test_from_execution_result_duck_typing(self):
        """Verify from_execution_result works with duck-typed objects."""

        class FakeStatus:
            value = "passed"

        class FakeResult:
            exit_code = 0
            stdout = "3 passed"
            stderr = ""
            duration_seconds = 1.23
            command_display = "pytest tests/"
            status = FakeStatus()
            metadata = {"target": "local"}

        ev = ExecutionEvidence.from_execution_result(FakeResult())
        assert ev.exit_code == 0
        assert ev.stdout == "3 passed"
        assert ev.duration_seconds == 1.23
        assert ev.command == "pytest tests/"
        assert ev.timed_out is False
        assert ev.metadata == {"target": "local"}

    def test_from_execution_result_timeout_detection(self):
        """Verify that timeout status is detected via from_execution_result."""

        class FakeStatus:
            value = "timeout"

        class FakeResult:
            exit_code = -1
            stdout = ""
            stderr = "timed out"
            duration_seconds = 300.0
            command_display = "pytest"
            status = FakeStatus()
            metadata = {}

        ev = ExecutionEvidence.from_execution_result(FakeResult())
        assert ev.timed_out is True


# ---------------------------------------------------------------------------
# ExitCodeSignalCollector
# ---------------------------------------------------------------------------


class TestExitCodeSignalCollector:
    def setup_method(self):
        self.collector = ExitCodeSignalCollector()

    def _collect_one(self, evidence: ExecutionEvidence) -> ConfidenceSignal:
        signals = self.collector.collect(evidence)
        assert len(signals) == 1
        return signals[0]

    def test_exit_zero_is_full_confidence(self):
        ev = make_evidence(exit_code=0)
        sig = self._collect_one(ev)
        assert sig.score == 1.0
        assert sig.weight > 0.8
        assert sig.name == "exit_code_success"

    def test_exit_one_is_test_failure(self):
        ev = make_evidence(exit_code=1)
        sig = self._collect_one(ev)
        assert sig.score < 0.5
        assert sig.name == "exit_code_test_failures"
        assert sig.evidence["exit_code"] == 1

    def test_exit_two_is_config_error(self):
        ev = make_evidence(exit_code=2)
        sig = self._collect_one(ev)
        assert sig.score <= 0.15
        assert sig.name == "exit_code_config_error"

    def test_exit_five_no_tests_collected(self):
        ev = make_evidence(exit_code=5)
        sig = self._collect_one(ev)
        assert sig.score == pytest.approx(0.20)
        assert sig.name == "exit_code_no_tests_collected"

    def test_exit_127_command_not_found(self):
        ev = make_evidence(exit_code=127)
        sig = self._collect_one(ev)
        assert sig.score == 0.0
        assert sig.name == "exit_code_command_not_found"

    def test_exit_neg_one_infra_error(self):
        ev = make_evidence(exit_code=-1)
        sig = self._collect_one(ev)
        assert sig.score == 0.0
        assert sig.name == "exit_code_infrastructure_error"

    def test_unknown_exit_code_soft_failure(self):
        ev = make_evidence(exit_code=42)
        sig = self._collect_one(ev)
        assert sig.score == pytest.approx(0.15)
        assert sig.name == "exit_code_unknown_failure"

    def test_timeout_overrides_exit_code(self):
        ev = make_evidence(exit_code=0, timed_out=True)
        sig = self._collect_one(ev)
        assert sig.score == 0.0
        assert sig.name == "exit_code_timeout"
        assert sig.evidence["timed_out"] is True

    def test_custom_code_scores(self):
        collector = ExitCodeSignalCollector(
            code_scores={99: ("custom_code", 0.6, 0.7)}
        )
        ev = make_evidence(exit_code=99)
        sig = collector.collect(ev)[0]
        assert sig.name == "exit_code_custom_code"
        assert sig.score == pytest.approx(0.6)
        assert sig.weight == pytest.approx(0.7)

    def test_evidence_metadata_populated(self):
        ev = make_evidence(exit_code=1, command="pytest -v tests/")
        sig = self._collect_one(ev)
        assert sig.evidence["exit_code"] == 1
        assert sig.evidence["command"] == "pytest -v tests/"

    def test_is_abstract_base_subclass(self):
        assert isinstance(self.collector, ExecutionSignalCollector)


# ---------------------------------------------------------------------------
# OutputPatternSignalCollector
# ---------------------------------------------------------------------------


class TestOutputPatternSignalCollector:
    def setup_method(self):
        self.collector = OutputPatternSignalCollector()

    def _by_name(self, signals: list[ConfidenceSignal], name: str) -> ConfidenceSignal:
        return next(s for s in signals if s.name == name)

    def test_detects_pytest_passed(self):
        ev = make_evidence(stdout="5 passed in 1.23s", framework="pytest")
        signals = self.collector.collect(ev)
        sig = self._by_name(signals, "output_tests_passed")
        assert sig.score == 1.0
        assert sig.evidence["matched"] is True

    def test_detects_pytest_failed(self):
        ev = make_evidence(stdout="2 passed, 1 failed in 0.5s", framework="pytest")
        signals = self.collector.collect(ev)
        fail_sig = self._by_name(signals, "output_tests_failed")
        assert fail_sig.score == pytest.approx(0.20)

    def test_detects_jest_passed(self):
        ev = make_evidence(
            stdout="Tests: 10 passed, 10 total",
            framework="jest",
        )
        signals = self.collector.collect(ev)
        sig = self._by_name(signals, "output_jest_tests_passed")
        assert sig.score == 1.0

    def test_detects_go_pass(self):
        ev = make_evidence(stdout="ok  github.com/myproject  0.123s", framework="go")
        signals = self.collector.collect(ev)
        sig = self._by_name(signals, "output_go_pass")
        assert sig.score == 1.0

    def test_detects_cargo_ok(self):
        ev = make_evidence(
            stdout="test result: ok. 5 passed; 0 failed; 0 ignored",
            framework="cargo",
        )
        signals = self.collector.collect(ev)
        sig = self._by_name(signals, "output_cargo_ok")
        assert sig.score == 1.0

    def test_absent_pattern_scores_neutral(self):
        ev = make_evidence(stdout="some other output")
        signals = self.collector.collect(ev)
        sig = self._by_name(signals, "output_tests_passed")
        assert sig.score == pytest.approx(OutputPatternSignalCollector.ABSENT_SCORE)
        assert sig.evidence["matched"] is False

    def test_import_error_scores_zero(self):
        ev = make_evidence(
            stderr="ModuleNotFoundError: No module named 'pytest'"
        )
        signals = self.collector.collect(ev)
        sig = self._by_name(signals, "output_module_not_found")
        assert sig.score == 0.0

    def test_command_not_found_scores_zero(self):
        ev = make_evidence(stderr="pytest: command not found")
        signals = self.collector.collect(ev)
        sig = self._by_name(signals, "output_command_not_found")
        assert sig.score == 0.0

    def test_framework_weight_boost(self):
        """Framework-matched patterns should get a weight boost."""
        ev_plain = make_evidence(stdout="5 passed in 1.23s", framework="")
        ev_pytest = make_evidence(stdout="5 passed in 1.23s", framework="pytest")

        plain_signals = self.collector.collect(ev_plain)
        pytest_signals = self.collector.collect(ev_pytest)

        plain_sig = self._by_name(plain_signals, "output_tests_passed")
        pytest_sig = self._by_name(pytest_signals, "output_tests_passed")

        assert pytest_sig.weight >= plain_sig.weight

    def test_syntax_error_scores_low(self):
        ev = make_evidence(stderr="SyntaxError: invalid syntax")
        signals = self.collector.collect(ev)
        sig = self._by_name(signals, "output_syntax_error")
        assert sig.score <= 0.10

    def test_all_signals_have_valid_ranges(self):
        ev = make_evidence(stdout="3 passed, 1 failed", stderr="some warning")
        for sig in self.collector.collect(ev):
            assert 0.0 <= sig.score <= 1.0
            assert 0.0 <= sig.weight <= 1.0

    def test_custom_patterns(self):
        from test_runner.confidence.signals import _OutputPattern

        custom = [
            _OutputPattern(
                pattern=r"CUSTOM_PASS",
                signal_name="my_custom_pass",
                weight=0.9,
                score_if_matched=1.0,
            )
        ]
        collector = OutputPatternSignalCollector(patterns=custom)
        ev = make_evidence(stdout="CUSTOM_PASS: 5 tests ok")
        signals = collector.collect(ev)
        assert len(signals) == 1
        assert signals[0].name == "my_custom_pass"
        assert signals[0].score == 1.0


# ---------------------------------------------------------------------------
# TimingSignalCollector
# ---------------------------------------------------------------------------


class TestTimingSignalCollector:
    def setup_method(self):
        self.collector = TimingSignalCollector()

    def _collect_one(self, evidence: ExecutionEvidence) -> ConfidenceSignal:
        signals = self.collector.collect(evidence)
        assert len(signals) == 1
        return signals[0]

    def test_normal_duration_full_confidence(self):
        ev = make_evidence(duration_seconds=2.5)
        sig = self._collect_one(ev)
        assert sig.score == 1.0
        assert sig.name == "timing_normal"

    def test_instant_exit_very_low_confidence(self):
        ev = make_evidence(duration_seconds=0.01)
        sig = self._collect_one(ev)
        assert sig.score == pytest.approx(0.10)
        assert sig.name == "timing_instant_exit"

    def test_fast_exit_slight_uncertainty(self):
        ev = make_evidence(duration_seconds=0.2)
        sig = self._collect_one(ev)
        assert sig.score == pytest.approx(0.55)
        assert sig.name == "timing_fast_exit"

    def test_slow_run_reduced_confidence(self):
        ev = make_evidence(duration_seconds=200.0)
        sig = self._collect_one(ev)
        assert sig.score == pytest.approx(0.75)
        assert sig.name == "timing_slow"

    def test_very_slow_run_low_confidence(self):
        ev = make_evidence(duration_seconds=700.0)
        sig = self._collect_one(ev)
        assert sig.score == pytest.approx(0.40)
        assert sig.name == "timing_very_slow"

    def test_timeout_zero_confidence(self):
        ev = make_evidence(duration_seconds=300.0, timed_out=True)
        sig = self._collect_one(ev)
        assert sig.score == 0.0
        assert sig.name == "timing_timed_out"

    def test_evidence_includes_duration(self):
        ev = make_evidence(duration_seconds=1.5)
        sig = self._collect_one(ev)
        assert sig.evidence["duration_seconds"] == pytest.approx(1.5)

    def test_evidence_includes_thresholds(self):
        ev = make_evidence(duration_seconds=1.0)
        sig = self._collect_one(ev)
        assert "thresholds" in sig.evidence
        assert "instant_max" in sig.evidence["thresholds"]

    def test_custom_thresholds(self):
        collector = TimingSignalCollector(
            instant_max=0.001,  # very tight
            fast_max=0.01,
            slow_warn=10.0,
            very_slow_warn=30.0,
        )
        ev = make_evidence(duration_seconds=0.005)
        sig = collector.collect(ev)[0]
        assert sig.name == "timing_fast_exit"

    def test_boundary_instant_to_fast(self):
        # Exactly at instant_max boundary → fast_exit
        collector = TimingSignalCollector(instant_max=0.05, fast_max=0.5)
        ev = make_evidence(duration_seconds=0.05)
        sig = collector.collect(ev)[0]
        assert sig.name == "timing_fast_exit"

    def test_weight_within_valid_range(self):
        ev = make_evidence(duration_seconds=1.0)
        sig = self._collect_one(ev)
        assert 0.0 <= sig.weight <= 1.0


# ---------------------------------------------------------------------------
# AssertionCountSignalCollector — count extraction
# ---------------------------------------------------------------------------


class TestExtractCounts:
    """Unit tests for the internal _extract_counts function."""

    def test_pytest_all_passed(self):
        counts = _extract_counts("5 passed in 1.23s")
        assert counts is not None
        assert counts.passed == 5
        assert counts.failed == 0
        assert counts.pass_rate == 1.0

    def test_pytest_mixed(self):
        counts = _extract_counts("3 passed, 2 failed in 0.5s")
        assert counts is not None
        assert counts.passed == 3
        assert counts.failed == 2
        assert counts.pass_rate == pytest.approx(0.6)

    def test_pytest_with_skipped(self):
        counts = _extract_counts("4 passed, 1 failed, 2 skipped in 1.0s")
        assert counts is not None
        assert counts.passed == 4
        assert counts.failed == 1
        assert counts.skipped == 2

    def test_pytest_only_failed(self):
        counts = _extract_counts("3 failed in 0.3s")
        assert counts is not None
        assert counts.failed == 3
        assert counts.passed == 0
        assert counts.pass_rate == 0.0

    def test_jest_passed(self):
        counts = _extract_counts("Tests: 10 passed, 10 total")
        assert counts is not None
        assert counts.passed == 10
        assert counts.total == 10

    def test_jest_mixed(self):
        counts = _extract_counts("Tests: 2 failed, 8 passed, 10 total")
        assert counts is not None
        assert counts.failed == 2
        assert counts.passed == 8

    def test_cargo_ok(self):
        counts = _extract_counts(
            "test result: ok. 7 passed; 0 failed; 1 ignored"
        )
        assert counts is not None
        assert counts.passed == 7
        assert counts.failed == 0
        assert counts.skipped == 1

    def test_cargo_failed(self):
        counts = _extract_counts(
            "test result: FAILED. 3 passed; 2 failed; 0 ignored"
        )
        assert counts is not None
        assert counts.passed == 3
        assert counts.failed == 2

    def test_go_test_cases(self):
        output = (
            "--- PASS: TestFoo (0.00s)\n"
            "--- PASS: TestBar (0.01s)\n"
            "--- FAIL: TestBaz (0.02s)\n"
        )
        counts = _extract_counts(output)
        assert counts is not None
        assert counts.passed == 2
        assert counts.failed == 1

    def test_junit_format(self):
        counts = _extract_counts(
            "Tests run: 10, Failures: 2, Errors: 0, Skipped: 1"
        )
        assert counts is not None
        assert counts.total == 10
        assert counts.failed == 2
        assert counts.skipped == 1

    def test_generic_passed(self):
        counts = _extract_counts("42 tests passed")
        assert counts is not None
        assert counts.passed == 42

    def test_no_match_returns_none(self):
        counts = _extract_counts("some random output with no test summary")
        assert counts is None

    def test_empty_output_returns_none(self):
        counts = _extract_counts("")
        assert counts is None


# ---------------------------------------------------------------------------
# AssertionCountSignalCollector — full collector
# ---------------------------------------------------------------------------


class TestAssertionCountSignalCollector:
    def setup_method(self):
        self.collector = AssertionCountSignalCollector()

    def _collect_one(self, evidence: ExecutionEvidence) -> ConfidenceSignal:
        signals = self.collector.collect(evidence)
        assert len(signals) == 1
        return signals[0]

    def test_all_passed_full_score(self):
        ev = make_evidence(stdout="5 passed in 1.23s")
        sig = self._collect_one(ev)
        assert sig.score == pytest.approx(1.0)
        assert sig.name == "assertion_count_pass_rate"
        assert sig.evidence["passed"] == 5
        assert sig.evidence["failed"] == 0

    def test_mixed_pass_fail_partial_score(self):
        ev = make_evidence(stdout="3 passed, 2 failed in 0.5s")
        sig = self._collect_one(ev)
        assert sig.score == pytest.approx(0.6)
        assert sig.name == "assertion_count_pass_rate"

    def test_all_failed_zero_score(self):
        ev = make_evidence(stdout="5 failed in 0.5s")
        sig = self._collect_one(ev)
        assert sig.score == pytest.approx(0.0)

    def test_no_counts_parseable(self):
        ev = make_evidence(stdout="some random output")
        sig = self._collect_one(ev)
        assert sig.name == "assertion_count_unparsed"
        assert sig.score == pytest.approx(AssertionCountSignalCollector.NO_COUNTS_SCORE)

    def test_zero_tests_ran(self):
        # Edge: parsed 0 passed, 0 failed, 0 skipped — nothing ran
        # Achieve by using a format that yields all zeros (unlikely, but test the path)
        # Use pytest "0 passed" edge
        ev = make_evidence(stdout="0 passed in 0.01s")
        sig = self._collect_one(ev)
        # 0 passed → ran == 0, skipped == 0 → zero_ran
        assert sig.name == "assertion_count_zero_ran"
        assert sig.score == pytest.approx(AssertionCountSignalCollector.ZERO_TESTS_SCORE)

    def test_all_skipped(self):
        ev = make_evidence(stdout="0 passed, 5 skipped in 0.1s")
        sig = self._collect_one(ev)
        assert sig.name == "assertion_count_all_skipped"
        assert sig.score == pytest.approx(AssertionCountSignalCollector.ALL_SKIPPED_SCORE)

    def test_evidence_includes_counts(self):
        ev = make_evidence(stdout="3 passed, 1 failed in 0.5s")
        sig = self._collect_one(ev)
        assert sig.evidence["passed"] == 3
        assert sig.evidence["failed"] == 1
        assert sig.evidence["parsed"] is True

    def test_weight_within_valid_range(self):
        ev = make_evidence(stdout="5 passed")
        sig = self._collect_one(ev)
        assert 0.0 <= sig.weight <= 1.0

    def test_custom_scores(self):
        collector = AssertionCountSignalCollector(
            no_counts_score=0.50,
            zero_tests_score=0.10,
            all_skipped_score=0.25,
        )
        ev = make_evidence(stdout="no summary here")
        sig = collector.collect(ev)[0]
        assert sig.score == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# InfrastructureHealthSignalCollector
# ---------------------------------------------------------------------------


class TestInfrastructureHealthSignalCollector:
    def setup_method(self):
        self.collector = InfrastructureHealthSignalCollector()

    def _by_name(self, signals: list[ConfidenceSignal], name: str) -> ConfidenceSignal | None:
        return next((s for s in signals if s.name == name), None)

    def test_clean_stderr_emits_positive_signal(self):
        ev = make_evidence(stderr="")
        signals = self.collector.collect(ev)
        clean = self._by_name(signals, "infra_stderr_clean")
        assert clean is not None
        assert clean.score == 1.0

    def test_module_not_found_emits_error_signal(self):
        ev = make_evidence(stderr="ModuleNotFoundError: No module named 'mylib'")
        signals = self.collector.collect(ev)
        err = self._by_name(signals, "infra_module_not_found")
        assert err is not None
        assert err.score == 0.0

    def test_module_not_found_also_taints_clean_signal(self):
        ev = make_evidence(stderr="ModuleNotFoundError: No module named 'mylib'")
        signals = self.collector.collect(ev)
        clean = self._by_name(signals, "infra_stderr_clean")
        assert clean is not None
        assert clean.score == 0.0  # tainted

    def test_command_not_found_emits_error_signal(self):
        ev = make_evidence(stderr="pytest: command not found")
        signals = self.collector.collect(ev)
        err = self._by_name(signals, "infra_command_not_found")
        assert err is not None
        assert err.score == 0.0
        assert err.weight > 0.8

    def test_permission_denied_emits_error_signal(self):
        ev = make_evidence(stderr="/workspace: Permission denied")
        signals = self.collector.collect(ev)
        err = self._by_name(signals, "infra_permission_denied")
        assert err is not None
        assert err.score == 0.0

    def test_import_error_emits_error_signal(self):
        ev = make_evidence(stderr="ImportError: cannot import name 'foo'")
        signals = self.collector.collect(ev)
        err = self._by_name(signals, "infra_import_error")
        assert err is not None
        assert err.score == 0.0

    def test_multiple_errors_all_emitted(self):
        ev = make_evidence(
            stderr="ImportError: foo\nPermission denied: /x"
        )
        signals = self.collector.collect(ev)
        names = {s.name for s in signals}
        assert "infra_import_error" in names
        assert "infra_permission_denied" in names

    def test_node_module_not_found(self):
        ev = make_evidence(stderr="Cannot find module 'jest'")
        signals = self.collector.collect(ev)
        err = self._by_name(signals, "infra_node_module_not_found")
        assert err is not None
        assert err.score == 0.0

    def test_npm_error(self):
        ev = make_evidence(stderr="npm ERR! code ENOENT")
        signals = self.collector.collect(ev)
        err = self._by_name(signals, "infra_npm_error")
        assert err is not None
        assert err.score < 0.5

    def test_custom_patterns(self):
        custom = [("CUSTOM_ERROR", "custom_infra_error", 0.9, 0.0)]
        collector = InfrastructureHealthSignalCollector(error_patterns=custom)
        ev = make_evidence(stderr="CUSTOM_ERROR: something broke")
        signals = collector.collect(ev)
        err = self._by_name(signals, "custom_infra_error")
        assert err is not None
        assert err.score == 0.0

    def test_snippet_in_evidence(self):
        ev = make_evidence(stderr="ImportError: cannot import name 'foo'")
        signals = self.collector.collect(ev)
        err = self._by_name(signals, "infra_import_error")
        assert err is not None
        assert "snippet" in err.evidence
        assert len(err.evidence["snippet"]) <= 200

    def test_all_signals_valid_ranges(self):
        ev = make_evidence(stderr="ModuleNotFoundError: x\nPermission denied: y")
        for sig in self.collector.collect(ev):
            assert 0.0 <= sig.score <= 1.0
            assert 0.0 <= sig.weight <= 1.0


# ---------------------------------------------------------------------------
# collect_execution_signals (integration / convenience function)
# ---------------------------------------------------------------------------


class TestCollectExecutionSignals:
    def test_returns_signals_from_all_collectors(self):
        ev = make_evidence(
            exit_code=0,
            stdout="5 passed in 1.23s",
            stderr="",
            duration_seconds=1.23,
            framework="pytest",
        )
        signals = collect_execution_signals(ev)
        names = {s.name for s in signals}

        # Exit code
        assert "exit_code_success" in names
        # Output patterns
        assert "output_tests_passed" in names
        # Timing
        assert "timing_normal" in names
        # Assertion counts
        assert "assertion_count_pass_rate" in names
        # Infrastructure health
        assert "infra_stderr_clean" in names

    def test_all_signals_have_valid_ranges(self):
        ev = make_evidence(
            exit_code=1,
            stdout="2 passed, 1 failed in 0.5s",
            stderr="some warning",
            duration_seconds=0.5,
        )
        for sig in collect_execution_signals(ev):
            assert isinstance(sig, ConfidenceSignal)
            assert 0.0 <= sig.score <= 1.0
            assert 0.0 <= sig.weight <= 1.0

    def test_custom_collector_list(self):
        ev = make_evidence(exit_code=0)
        signals = collect_execution_signals(ev, collectors=[ExitCodeSignalCollector()])
        assert len(signals) == 1
        assert signals[0].name == "exit_code_success"

    def test_feeds_confidence_model(self):
        """End-to-end: signals feed into ConfidenceModel and produce a decision."""
        ev = make_evidence(
            exit_code=0,
            stdout="10 passed in 2.0s",
            stderr="",
            duration_seconds=2.0,
            framework="pytest",
        )
        signals = collect_execution_signals(ev)
        model = ConfidenceModel()
        result = model.evaluate(signals)
        # A clean all-passed run should yield a reasonably high confidence
        assert result.score > 0.5

    def test_failing_run_lower_confidence(self):
        """A run with failures should have lower confidence than a passing run."""
        ev_pass = make_evidence(
            exit_code=0,
            stdout="10 passed in 2.0s",
            stderr="",
            duration_seconds=2.0,
        )
        ev_fail = make_evidence(
            exit_code=1,
            stdout="3 passed, 7 failed in 1.0s",
            stderr="",
            duration_seconds=1.0,
        )
        model = ConfidenceModel()
        pass_result = model.evaluate(collect_execution_signals(ev_pass))
        fail_result = model.evaluate(collect_execution_signals(ev_fail))
        assert pass_result.score > fail_result.score

    def test_infra_error_very_low_confidence(self):
        """Infrastructure errors should drive confidence near zero."""
        ev = make_evidence(
            exit_code=127,
            stdout="",
            stderr="pytest: command not found",
            duration_seconds=0.01,
        )
        model = ConfidenceModel()
        result = model.evaluate(collect_execution_signals(ev))
        assert result.score < 0.5


# ---------------------------------------------------------------------------
# Abstract base class contract
# ---------------------------------------------------------------------------


class TestExecutionSignalCollectorABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ExecutionSignalCollector()  # type: ignore[abstract]

    def test_subclass_must_implement_collect(self):
        class Incomplete(ExecutionSignalCollector):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class AlwaysHappy(ExecutionSignalCollector):
            def collect(self, evidence: ExecutionEvidence) -> list[ConfidenceSignal]:
                return [ConfidenceSignal(name="happy", weight=1.0, score=1.0)]

        collector = AlwaysHappy()
        signals = collector.collect(make_evidence())
        assert signals[0].name == "happy"
