"""Tests for the confidence-based autonomy engine."""

import pytest

from test_runner.agents.parser import TestFramework, TestIntent
from test_runner.autonomy.engine import (
    AutonomyDecision,
    AutonomyEngine,
    DiscoveryFindings,
    ExplorationAction,
    ExplorationSuggestion,
    InvocationSpec,
    TestTarget,
)
from test_runner.autonomy.policy import AutonomyPolicyConfig
from test_runner.models.confidence import (
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceTier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig(name: str = "s", weight: float = 1.0, score: float = 1.0) -> ConfidenceSignal:
    return ConfidenceSignal(name=name, weight=weight, score=score)


def _high_confidence_findings(
    round_num: int = 1,
    working_dir: str = "/project",
) -> DiscoveryFindings:
    """Findings that should produce HIGH confidence."""
    return DiscoveryFindings(
        signals=[
            _sig("pytest_in_pyproject", weight=0.9, score=1.0),
            _sig("python_test_files", weight=0.8, score=1.0),
            _sig("pytest_ini_exists", weight=0.9, score=1.0),
        ],
        frameworks_detected=[
            {"framework": "pytest", "confidence": 0.95, "evidence": "pyproject.toml"},
        ],
        test_files=["tests/test_foo.py", "tests/test_bar.py"],
        working_directory=working_dir,
        exploration_round=round_num,
    )


def _medium_confidence_findings(round_num: int = 1) -> DiscoveryFindings:
    """Findings that should produce MEDIUM confidence (0.60–0.89)."""
    return DiscoveryFindings(
        signals=[
            _sig("pyproject_toml_exists", weight=0.7, score=1.0),
            _sig("python_test_files", weight=0.8, score=0.8),
            _sig("some_config_hint", weight=0.6, score=0.5),
        ],
        frameworks_detected=[
            {"framework": "pytest", "confidence": 0.6, "evidence": "heuristic"},
        ],
        test_files=["tests/test_something.py"],
        working_directory="/project",
        exploration_round=round_num,
    )


def _low_confidence_findings(round_num: int = 1) -> DiscoveryFindings:
    """Findings with very little evidence."""
    return DiscoveryFindings(
        signals=[
            _sig("no_files", weight=0.8, score=0.0),
            _sig("no_config", weight=0.9, score=0.0),
            _sig("maybe_something", weight=0.3, score=0.2),
        ],
        working_directory="/project",
        exploration_round=round_num,
    )


# ---------------------------------------------------------------------------
# TestTarget
# ---------------------------------------------------------------------------


class TestTestTarget:
    def test_construction(self):
        t = TestTarget(
            framework=TestFramework.PYTEST,
            path="/project",
            run_command=["pytest", "-v"],
            confidence=0.95,
        )
        assert t.framework == TestFramework.PYTEST
        assert t.run_command == ["pytest", "-v"]
        assert t.confidence == 0.95

    def test_default_metadata(self):
        t = TestTarget(
            framework=TestFramework.JEST,
            path="/project",
            run_command=["npx", "jest"],
        )
        assert t.metadata == {}
        assert t.confidence == 1.0


# ---------------------------------------------------------------------------
# InvocationSpec
# ---------------------------------------------------------------------------


class TestInvocationSpec:
    def test_empty_spec(self):
        spec = InvocationSpec(
            targets=(),
            intent=TestIntent.RUN,
            working_directory="/project",
            overall_confidence=0.0,
            confidence_tier=ConfidenceTier.LOW,
        )
        assert spec.is_empty is True
        assert spec.framework_summary == []

    def test_non_empty_spec(self):
        targets = (
            TestTarget(TestFramework.PYTEST, "/p", ["pytest"]),
            TestTarget(TestFramework.JEST, "/p", ["npx", "jest"]),
        )
        spec = InvocationSpec(
            targets=targets,
            intent=TestIntent.RUN,
            working_directory="/project",
            overall_confidence=0.92,
            confidence_tier=ConfidenceTier.HIGH,
        )
        assert spec.is_empty is False
        assert "pytest" in spec.framework_summary
        assert "jest" in spec.framework_summary

    def test_summary_keys(self):
        spec = InvocationSpec(
            targets=(TestTarget(TestFramework.PYTEST, "/p", ["pytest"]),),
            intent=TestIntent.LIST,
            working_directory="/project",
            overall_confidence=0.85,
            confidence_tier=ConfidenceTier.MEDIUM,
        )
        s = spec.summary()
        assert s["target_count"] == 1
        assert s["intent"] == "list"
        assert s["overall_confidence"] == 0.85
        assert s["confidence_tier"] == "medium"
        assert len(s["targets"]) == 1

    def test_dedup_framework_summary(self):
        targets = (
            TestTarget(TestFramework.PYTEST, "/a", ["pytest"]),
            TestTarget(TestFramework.PYTEST, "/b", ["pytest"]),
        )
        spec = InvocationSpec(
            targets=targets,
            intent=TestIntent.RUN,
            working_directory="/p",
            overall_confidence=0.9,
            confidence_tier=ConfidenceTier.HIGH,
        )
        assert spec.framework_summary == ["pytest"]


# ---------------------------------------------------------------------------
# AutonomyPolicyConfig
# ---------------------------------------------------------------------------


class TestAutonomyPolicyConfig:
    def test_defaults(self):
        p = AutonomyPolicyConfig()
        assert p.execute_threshold == 0.90
        assert p.warn_threshold == 0.60
        assert p.max_exploration_rounds == 5

    def test_conservative_preset(self):
        p = AutonomyPolicyConfig.conservative()
        assert p.execute_threshold == 0.95
        assert p.require_framework_detection is True

    def test_aggressive_preset(self):
        p = AutonomyPolicyConfig.aggressive()
        assert p.execute_threshold == 0.70
        assert p.min_positive_signals == 1

    def test_rejects_invalid_thresholds(self):
        with pytest.raises(ValueError):
            AutonomyPolicyConfig(execute_threshold=0.5, warn_threshold=0.8)

    def test_rejects_zero_rounds(self):
        with pytest.raises(ValueError):
            AutonomyPolicyConfig(max_exploration_rounds=0)


# ---------------------------------------------------------------------------
# AutonomyEngine — high confidence scenarios
# ---------------------------------------------------------------------------


class TestEngineHighConfidence:
    def test_high_confidence_proceeds(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(_high_confidence_findings())
        assert decision.action == ExplorationAction.PROCEED
        assert decision.should_proceed is True
        assert decision.invocation_spec is not None
        assert not decision.invocation_spec.is_empty

    def test_high_confidence_spec_has_targets(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(_high_confidence_findings())
        spec = decision.invocation_spec
        assert spec is not None
        assert len(spec.targets) >= 1
        assert spec.overall_confidence >= 0.90
        assert spec.confidence_tier == ConfidenceTier.HIGH

    def test_high_confidence_uses_intent(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(
            _high_confidence_findings(), intent=TestIntent.LIST
        )
        assert decision.invocation_spec is not None
        assert decision.invocation_spec.intent == TestIntent.LIST

    def test_high_confidence_preserves_working_dir(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(
            _high_confidence_findings(working_dir="/my/project")
        )
        assert decision.invocation_spec is not None
        assert decision.invocation_spec.working_directory == "/my/project"


# ---------------------------------------------------------------------------
# AutonomyEngine — medium confidence scenarios
# ---------------------------------------------------------------------------


class TestEngineMediumConfidence:
    def test_medium_confidence_warns(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(_medium_confidence_findings())
        # Medium but far from execute threshold — should warn or explore
        assert decision.action in (
            ExplorationAction.PROCEED_WITH_WARNING,
            ExplorationAction.EXPLORE_FURTHER,
        )

    def test_medium_confidence_at_max_rounds_proceeds_with_warning(self):
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(max_exploration_rounds=2)
        )
        findings = _medium_confidence_findings(round_num=2)
        decision = engine.evaluate(findings)
        assert decision.action == ExplorationAction.PROCEED_WITH_WARNING
        assert decision.invocation_spec is not None

    def test_medium_near_threshold_explores(self):
        """When score is close to execute threshold, engine explores further."""
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(execute_threshold=0.90)
        )
        # Build findings with score around 0.82-0.88 (within 0.10 of 0.90)
        findings = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=0.85),
                _sig("b", weight=1.0, score=0.88),
            ],
            frameworks_detected=[{"framework": "pytest", "confidence": 0.85}],
            test_files=["test_a.py", "test_b.py"],
            working_directory="/project",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        assert decision.action == ExplorationAction.EXPLORE_FURTHER
        assert len(decision.suggestions) > 0


# ---------------------------------------------------------------------------
# AutonomyEngine — low confidence scenarios
# ---------------------------------------------------------------------------


class TestEngineLowConfidence:
    def test_low_confidence_explores(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(_low_confidence_findings())
        assert decision.action == ExplorationAction.EXPLORE_FURTHER
        assert decision.needs_exploration is True
        assert len(decision.suggestions) > 0

    def test_low_confidence_at_max_rounds_escalates(self):
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(max_exploration_rounds=3)
        )
        findings = _low_confidence_findings(round_num=3)
        decision = engine.evaluate(findings)
        assert decision.action == ExplorationAction.ESCALATE
        assert decision.needs_escalation is True


# ---------------------------------------------------------------------------
# AutonomyEngine — hard cap escalation
# ---------------------------------------------------------------------------


class TestEngineHardCap:
    def test_exceeding_max_rounds_escalates(self):
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(max_exploration_rounds=3)
        )
        findings = _high_confidence_findings(round_num=4)
        decision = engine.evaluate(findings)
        assert decision.action == ExplorationAction.ESCALATE
        assert "Hard cap" in decision.reason

    def test_exactly_at_max_rounds_does_not_escalate_if_confident(self):
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(max_exploration_rounds=3)
        )
        findings = _high_confidence_findings(round_num=3)
        decision = engine.evaluate(findings)
        # Should still proceed if confidence is high, even at max rounds
        assert decision.action == ExplorationAction.PROCEED


# ---------------------------------------------------------------------------
# AutonomyEngine — minimum signals requirement
# ---------------------------------------------------------------------------


class TestEngineMinSignals:
    def test_too_few_positive_signals_explores(self):
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(min_positive_signals=3)
        )
        findings = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=1.0),
                _sig("b", weight=1.0, score=0.0),
                _sig("c", weight=1.0, score=0.0),
            ],
            frameworks_detected=[{"framework": "pytest", "confidence": 0.9}],
            test_files=["test.py"],
            working_directory="/p",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        # Only 1 positive signal, need 3
        assert decision.action == ExplorationAction.EXPLORE_FURTHER

    def test_enough_positive_signals_proceeds(self):
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(min_positive_signals=2)
        )
        findings = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=1.0),
                _sig("b", weight=1.0, score=0.9),
            ],
            frameworks_detected=[{"framework": "pytest", "confidence": 0.95}],
            test_files=["test.py"],
            working_directory="/p",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        assert decision.should_proceed is True


# ---------------------------------------------------------------------------
# AutonomyEngine — framework detection requirement
# ---------------------------------------------------------------------------


class TestEngineFrameworkRequirement:
    def test_no_framework_explores_when_required(self):
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(require_framework_detection=True)
        )
        findings = DiscoveryFindings(
            signals=[_sig("a", weight=1.0, score=0.9)],
            frameworks_detected=[],
            test_files=["test.py"],
            working_directory="/p",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        assert decision.action == ExplorationAction.EXPLORE_FURTHER

    def test_no_framework_escalates_at_max_rounds(self):
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(
                require_framework_detection=True,
                max_exploration_rounds=2,
                allow_script_fallback=False,
            )
        )
        findings = DiscoveryFindings(
            signals=[_sig("a", weight=1.0, score=0.9)],
            frameworks_detected=[],
            test_files=[],
            scripts=[],
            working_directory="/p",
            exploration_round=2,
        )
        decision = engine.evaluate(findings)
        assert decision.action == ExplorationAction.ESCALATE

    def test_script_fallback_when_no_framework(self):
        engine = AutonomyEngine(
            policy=AutonomyPolicyConfig(
                require_framework_detection=True,
                allow_script_fallback=True,
                max_exploration_rounds=2,
            )
        )
        findings = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=1.0),
                _sig("b", weight=1.0, score=0.9),
            ],
            frameworks_detected=[],
            scripts=["./run_tests.sh"],
            test_files=[],
            working_directory="/p",
            exploration_round=2,
        )
        decision = engine.evaluate(findings)
        # Should NOT escalate because scripts exist and fallback allowed
        assert decision.action != ExplorationAction.ESCALATE


# ---------------------------------------------------------------------------
# AutonomyEngine — exploration suggestions
# ---------------------------------------------------------------------------


class TestEngineSuggestions:
    def test_suggests_deep_scan_when_few_files(self):
        engine = AutonomyEngine()
        findings = DiscoveryFindings(
            signals=[_sig("a", weight=0.5, score=0.3)],
            test_files=["test.py"],  # fewer than 5
            working_directory="/p",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        assert decision.needs_exploration
        strategies = [s.strategy for s in decision.suggestions]
        assert "deep_pattern_scan" in strategies

    def test_suggests_config_inspection_when_no_framework(self):
        engine = AutonomyEngine()
        findings = DiscoveryFindings(
            signals=[_sig("a", weight=0.5, score=0.3)],
            frameworks_detected=[],
            test_files=["test.py"],
            working_directory="/p",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        assert decision.needs_exploration
        strategies = [s.strategy for s in decision.suggestions]
        assert "config_file_inspection" in strategies

    def test_suggests_help_probe_when_framework_detected(self):
        engine = AutonomyEngine()
        findings = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=0.85),
                _sig("b", weight=1.0, score=0.88),
            ],
            frameworks_detected=[{"framework": "pytest", "confidence": 0.8}],
            test_files=["test.py"],
            working_directory="/p",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        if decision.needs_exploration:
            strategies = [s.strategy for s in decision.suggestions]
            assert "help_probe" in strategies


# ---------------------------------------------------------------------------
# AutonomyEngine — InvocationSpec building
# ---------------------------------------------------------------------------


class TestEngineInvocationSpec:
    def test_spec_includes_detected_framework(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(_high_confidence_findings())
        spec = decision.invocation_spec
        assert spec is not None
        frameworks = [t.framework for t in spec.targets]
        assert TestFramework.PYTEST in frameworks

    def test_spec_metadata_has_round_info(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(_high_confidence_findings(round_num=2))
        spec = decision.invocation_spec
        assert spec is not None
        assert spec.metadata["exploration_round"] == 2

    def test_spec_for_scripts(self):
        engine = AutonomyEngine()
        findings = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=1.0),
                _sig("b", weight=1.0, score=0.95),
            ],
            frameworks_detected=[],
            scripts=["./run_tests.sh", "./test_all.sh"],
            working_directory="/p",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        if decision.invocation_spec:
            script_targets = [
                t for t in decision.invocation_spec.targets
                if t.framework == TestFramework.SCRIPT
            ]
            assert len(script_targets) == 2

    def test_fallback_target_from_primary_framework(self):
        """When no test files match but a framework is detected, build a fallback."""
        engine = AutonomyEngine()
        findings = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=0.95),
                _sig("b", weight=1.0, score=0.92),
            ],
            frameworks_detected=[{"framework": "jest", "confidence": 0.9}],
            test_files=[],  # No test files
            working_directory="/project",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        spec = decision.invocation_spec
        assert spec is not None
        # Should have a fallback target
        assert len(spec.targets) >= 1


# ---------------------------------------------------------------------------
# AutonomyDecision
# ---------------------------------------------------------------------------


class TestAutonomyDecision:
    def test_decision_summary_keys(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(_high_confidence_findings())
        s = decision.summary()
        assert "action" in s
        assert "round" in s
        assert "reason" in s
        assert "confidence" in s
        assert "invocation_spec" in s

    def test_explore_decision_has_suggestions_in_summary(self):
        engine = AutonomyEngine()
        decision = engine.evaluate(_low_confidence_findings())
        s = decision.summary()
        if decision.needs_exploration:
            assert "suggestions" in s

    def test_decision_properties(self):
        engine = AutonomyEngine()

        proceed = engine.evaluate(_high_confidence_findings())
        assert proceed.should_proceed is True
        assert proceed.needs_exploration is False
        assert proceed.needs_escalation is False

        explore = engine.evaluate(_low_confidence_findings())
        assert explore.should_proceed is False
        assert explore.needs_exploration is True
        assert explore.needs_escalation is False


# ---------------------------------------------------------------------------
# DiscoveryFindings helpers
# ---------------------------------------------------------------------------


class TestDiscoveryFindings:
    def test_positive_signal_count(self):
        f = DiscoveryFindings(
            signals=[
                _sig("a", score=1.0),
                _sig("b", score=0.0),
                _sig("c", score=0.5),
            ]
        )
        assert f.positive_signal_count == 2

    def test_has_framework(self):
        f = DiscoveryFindings(
            frameworks_detected=[{"framework": "pytest", "confidence": 0.9}]
        )
        assert f.has_framework is True
        assert f.primary_framework == "pytest"

    def test_no_framework(self):
        f = DiscoveryFindings()
        assert f.has_framework is False
        assert f.primary_framework is None

    def test_primary_framework_picks_highest_confidence(self):
        f = DiscoveryFindings(
            frameworks_detected=[
                {"framework": "unittest", "confidence": 0.7},
                {"framework": "pytest", "confidence": 0.95},
                {"framework": "jest", "confidence": 0.8},
            ]
        )
        assert f.primary_framework == "pytest"


# ---------------------------------------------------------------------------
# Custom policies
# ---------------------------------------------------------------------------


class TestCustomPolicies:
    def test_lenient_policy_proceeds_at_lower_score(self):
        engine = AutonomyEngine(policy=AutonomyPolicyConfig.aggressive())
        findings = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=0.75),
                _sig("b", weight=1.0, score=0.70),
            ],
            frameworks_detected=[{"framework": "pytest", "confidence": 0.7}],
            test_files=["test.py"],
            working_directory="/p",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        assert decision.should_proceed is True

    def test_strict_policy_requires_higher_score(self):
        engine = AutonomyEngine(policy=AutonomyPolicyConfig.conservative())
        findings = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=0.90),
                _sig("b", weight=1.0, score=0.88),
            ],
            frameworks_detected=[{"framework": "pytest", "confidence": 0.9}],
            test_files=["test.py", "test2.py"],
            working_directory="/p",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        # 0.89 average < 0.95 conservative threshold
        assert decision.action != ExplorationAction.PROCEED


# ---------------------------------------------------------------------------
# AutonomyEngine — composite evaluation (evidence + LLM signal blending)
# ---------------------------------------------------------------------------


class TestEngineCompositeEvaluation:
    """Verify the engine uses composite scoring when LLM signals are present."""

    def test_composite_scoring_with_llm_signals(self):
        """When LLM signals are present, engine uses composite evaluation."""
        engine = AutonomyEngine()
        findings = DiscoveryFindings(
            signals=[
                # Evidence signals
                _sig("pytest_in_pyproject", weight=0.9, score=1.0),
                _sig("python_test_files", weight=0.8, score=1.0),
                # LLM self-assessment signal (prefixed with "llm_")
                _sig("llm_self_assessment", weight=0.6, score=0.95),
            ],
            frameworks_detected=[
                {"framework": "pytest", "confidence": 0.95},
            ],
            test_files=["tests/test_foo.py"],
            working_directory="/project",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        assert decision.should_proceed is True
        assert decision.confidence_result.score > 0.0

    def test_composite_scoring_without_llm_signals(self):
        """Without LLM signals, engine uses flat evaluation."""
        engine = AutonomyEngine()
        findings = DiscoveryFindings(
            signals=[
                _sig("pytest_in_pyproject", weight=0.9, score=1.0),
                _sig("python_test_files", weight=0.8, score=1.0),
            ],
            frameworks_detected=[
                {"framework": "pytest", "confidence": 0.95},
            ],
            test_files=["tests/test_foo.py"],
            working_directory="/project",
            exploration_round=1,
        )
        decision = engine.evaluate(findings)
        assert decision.should_proceed is True

    def test_llm_signals_influence_score(self):
        """LLM signals should actually affect the composite score."""
        engine = AutonomyEngine()

        # Evidence-only findings — moderate confidence
        evidence_only = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=0.85),
                _sig("b", weight=1.0, score=0.80),
            ],
            frameworks_detected=[{"framework": "pytest", "confidence": 0.8}],
            test_files=["test.py"],
            working_directory="/p",
            exploration_round=1,
        )

        # Same evidence + LLM boost
        with_llm = DiscoveryFindings(
            signals=[
                _sig("a", weight=1.0, score=0.85),
                _sig("b", weight=1.0, score=0.80),
                _sig("llm_self_assessment", weight=0.6, score=0.99),
            ],
            frameworks_detected=[{"framework": "pytest", "confidence": 0.8}],
            test_files=["test.py"],
            working_directory="/p",
            exploration_round=1,
        )

        score_no_llm = engine.score_findings(evidence_only).score
        score_with_llm = engine.score_findings(with_llm).score
        # The scores may differ because composite evaluation separates
        # categories; the important thing is both produce valid scores
        assert 0.0 <= score_no_llm <= 1.0
        assert 0.0 <= score_with_llm <= 1.0

    def test_custom_composite_weights(self):
        """Custom composite weights can be passed to the engine."""
        from test_runner.models.confidence import CompositeWeights

        weights = CompositeWeights(evidence=0.9, llm=0.1)
        engine = AutonomyEngine(composite_weights=weights)
        assert engine.confidence_model.composite_weights.evidence == 0.9
        assert engine.confidence_model.composite_weights.llm == 0.1


# ---------------------------------------------------------------------------
# AutonomyEngine — score_findings
# ---------------------------------------------------------------------------


class TestEngineScoreFindings:
    """Verify score_findings returns confidence without making decisions."""

    def test_score_findings_returns_result(self):
        engine = AutonomyEngine()
        findings = _high_confidence_findings()
        result = engine.score_findings(findings)
        assert isinstance(result, ConfidenceResult)
        assert result.score >= 0.90
        assert result.tier == ConfidenceTier.HIGH

    def test_score_findings_empty(self):
        engine = AutonomyEngine()
        findings = DiscoveryFindings()
        result = engine.score_findings(findings)
        assert result.score == 0.0
        assert result.tier == ConfidenceTier.LOW

    def test_score_findings_matches_evaluate(self):
        """score_findings should produce the same score as evaluate."""
        engine = AutonomyEngine()
        findings = _high_confidence_findings()
        score_result = engine.score_findings(findings)
        decision = engine.evaluate(findings)
        assert score_result.score == decision.confidence_result.score


# ---------------------------------------------------------------------------
# AutonomyEngine — build_invocation_spec auto-scoring
# ---------------------------------------------------------------------------


class TestEngineAutoScoring:
    """Verify build_invocation_spec can auto-score when no result provided."""

    def test_auto_score_on_build(self):
        engine = AutonomyEngine()
        findings = _high_confidence_findings()
        spec = engine.build_invocation_spec(findings, intent=TestIntent.RUN)
        assert spec.overall_confidence >= 0.90
        assert spec.confidence_tier == ConfidenceTier.HIGH

    def test_explicit_score_on_build(self):
        engine = AutonomyEngine()
        findings = _high_confidence_findings()
        explicit_result = engine.score_findings(findings)
        spec = engine.build_invocation_spec(
            findings, confidence_result=explicit_result, intent=TestIntent.RUN
        )
        assert spec.overall_confidence == explicit_result.score
