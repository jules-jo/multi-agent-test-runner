"""Tests for the three-tier decision engine with threshold-based routing.

Verifies:
- Tier 1: >=90% confidence => EXECUTE_IMMEDIATELY
- Tier 2: 60-90% confidence => EXECUTE_WITH_WARNING
- Tier 3: <60% confidence => CONTINUE_INVESTIGATING
- Budget exhaustion escalation
- Precondition checks (min signals, framework requirement)
- Custom policy thresholds
- Edge cases at tier boundaries
- DecisionResult properties and serialization
"""

from __future__ import annotations

import pytest

from test_runner.autonomy.decision_engine import (
    DecisionContext,
    DecisionEngine,
    DecisionResult,
    DecisionVerdict,
)
from test_runner.autonomy.policy import AutonomyPolicyConfig
from test_runner.models.confidence import (
    ConfidenceSignal,
    ConfidenceTier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signals(score: float, count: int = 3) -> list[ConfidenceSignal]:
    """Create uniform signals that aggregate to the given score."""
    return [
        ConfidenceSignal(name=f"sig_{i}", weight=1.0, score=score)
        for i in range(count)
    ]


def _ctx(**kwargs) -> DecisionContext:
    """Build a DecisionContext with sensible defaults, overridable."""
    defaults = {
        "exploration_round": 1,
        "max_exploration_rounds": 5,
        "positive_signal_count": 5,
        "min_positive_signals": 2,
        "has_framework_detected": True,
        "has_scripts": False,
        "require_framework": False,
        "allow_script_fallback": True,
    }
    defaults.update(kwargs)
    return DecisionContext(**defaults)


# ---------------------------------------------------------------------------
# DecisionVerdict enum
# ---------------------------------------------------------------------------


class TestDecisionVerdict:
    def test_three_verdicts_exist(self) -> None:
        assert len(DecisionVerdict) == 3

    def test_verdict_values(self) -> None:
        assert DecisionVerdict.EXECUTE_IMMEDIATELY.value == "execute_immediately"
        assert DecisionVerdict.EXECUTE_WITH_WARNING.value == "execute_with_warning"
        assert DecisionVerdict.CONTINUE_INVESTIGATING.value == "continue_investigating"


# ---------------------------------------------------------------------------
# DecisionContext properties
# ---------------------------------------------------------------------------


class TestDecisionContext:
    def test_default_context(self) -> None:
        ctx = DecisionContext()
        assert ctx.exploration_round == 1
        assert ctx.max_exploration_rounds == 5
        assert not ctx.is_at_budget_limit
        assert not ctx.has_exceeded_budget

    def test_at_budget_limit(self) -> None:
        ctx = DecisionContext(exploration_round=5, max_exploration_rounds=5)
        assert ctx.is_at_budget_limit is True
        assert ctx.has_exceeded_budget is False

    def test_exceeded_budget(self) -> None:
        ctx = DecisionContext(exploration_round=6, max_exploration_rounds=5)
        assert ctx.has_exceeded_budget is True
        assert ctx.is_at_budget_limit is True

    def test_has_enough_signals(self) -> None:
        ctx = DecisionContext(positive_signal_count=3, min_positive_signals=2)
        assert ctx.has_enough_signals is True

    def test_not_enough_signals(self) -> None:
        ctx = DecisionContext(positive_signal_count=1, min_positive_signals=2)
        assert ctx.has_enough_signals is False

    def test_framework_not_required(self) -> None:
        ctx = DecisionContext(require_framework=False, has_framework_detected=False)
        assert ctx.framework_requirement_met is True

    def test_framework_required_and_detected(self) -> None:
        ctx = DecisionContext(require_framework=True, has_framework_detected=True)
        assert ctx.framework_requirement_met is True

    def test_framework_required_not_detected(self) -> None:
        ctx = DecisionContext(
            require_framework=True,
            has_framework_detected=False,
            exploration_round=1,
            max_exploration_rounds=5,
        )
        assert ctx.framework_requirement_met is False

    def test_framework_script_fallback_at_limit(self) -> None:
        ctx = DecisionContext(
            require_framework=True,
            has_framework_detected=False,
            has_scripts=True,
            allow_script_fallback=True,
            exploration_round=5,
            max_exploration_rounds=5,
        )
        assert ctx.framework_requirement_met is True

    def test_framework_no_script_fallback_at_limit(self) -> None:
        ctx = DecisionContext(
            require_framework=True,
            has_framework_detected=False,
            has_scripts=True,
            allow_script_fallback=False,
            exploration_round=5,
            max_exploration_rounds=5,
        )
        assert ctx.framework_requirement_met is False


# ---------------------------------------------------------------------------
# Tier 1: EXECUTE_IMMEDIATELY (score >= 90%)
# ---------------------------------------------------------------------------


class TestTier1ExecuteImmediately:
    """>=90% confidence should result in EXECUTE_IMMEDIATELY."""

    def test_score_at_90_percent(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.90), _ctx())
        assert result.verdict == DecisionVerdict.EXECUTE_IMMEDIATELY
        assert result.can_execute is True
        assert result.needs_warning is False
        assert result.needs_investigation is False

    def test_score_at_95_percent(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.95), _ctx())
        assert result.verdict == DecisionVerdict.EXECUTE_IMMEDIATELY

    def test_score_at_100_percent(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(1.0), _ctx())
        assert result.verdict == DecisionVerdict.EXECUTE_IMMEDIATELY

    def test_high_confidence_no_escalation(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.95), _ctx())
        assert result.should_escalate is False
        assert result.escalation_reason == ""

    def test_high_confidence_reason_mentions_threshold(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.92), _ctx())
        assert "execute threshold" in result.reason.lower() or "0.90" in result.reason

    def test_high_confidence_tier_is_high(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.95), _ctx())
        assert result.confidence_tier == ConfidenceTier.HIGH


# ---------------------------------------------------------------------------
# Tier 2: EXECUTE_WITH_WARNING (60% <= score < 90%)
# ---------------------------------------------------------------------------


class TestTier2ExecuteWithWarning:
    """60-90% confidence should result in EXECUTE_WITH_WARNING."""

    def test_score_at_60_percent(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.60), _ctx())
        # 60% is in warn zone. Could be EXECUTE_WITH_WARNING or
        # CONTINUE_INVESTIGATING if near-threshold logic triggers.
        assert result.verdict in (
            DecisionVerdict.EXECUTE_WITH_WARNING,
            DecisionVerdict.CONTINUE_INVESTIGATING,
        )
        # But if we're at budget limit, it must execute with warning
        result2 = engine.decide(
            _signals(0.60),
            _ctx(exploration_round=5, max_exploration_rounds=5),
        )
        assert result2.verdict == DecisionVerdict.EXECUTE_WITH_WARNING

    def test_score_at_70_percent(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.70), _ctx())
        assert result.verdict == DecisionVerdict.EXECUTE_WITH_WARNING
        assert result.can_execute is True
        assert result.needs_warning is True

    def test_score_at_75_percent(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.75), _ctx())
        assert result.verdict == DecisionVerdict.EXECUTE_WITH_WARNING

    def test_medium_confidence_at_budget_limit(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.65),
            _ctx(exploration_round=5, max_exploration_rounds=5),
        )
        assert result.verdict == DecisionVerdict.EXECUTE_WITH_WARNING
        assert result.should_escalate is False

    def test_medium_confidence_tier(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.75), _ctx())
        assert result.confidence_tier == ConfidenceTier.MEDIUM

    def test_near_execute_threshold_explores_if_budget_allows(self) -> None:
        """Score in [0.80, 0.90) may trigger extra investigation."""
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.85),
            _ctx(exploration_round=1, max_exploration_rounds=5),
        )
        # Within 10% of 0.90, should continue investigating
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING
        assert result.should_escalate is False

    def test_near_threshold_at_budget_limit_warns(self) -> None:
        """At budget limit, even near-threshold scores proceed with warning."""
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.85),
            _ctx(exploration_round=5, max_exploration_rounds=5),
        )
        assert result.verdict == DecisionVerdict.EXECUTE_WITH_WARNING


# ---------------------------------------------------------------------------
# Tier 3: CONTINUE_INVESTIGATING (score < 60%)
# ---------------------------------------------------------------------------


class TestTier3ContinueInvestigating:
    """<60% confidence should result in CONTINUE_INVESTIGATING."""

    def test_score_at_50_percent(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.50), _ctx())
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING
        assert result.can_execute is False
        assert result.needs_investigation is True

    def test_score_at_30_percent(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.30), _ctx())
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING

    def test_score_at_0_percent(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.0), _ctx())
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING

    def test_empty_signals(self) -> None:
        engine = DecisionEngine()
        result = engine.decide([], _ctx())
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING
        assert result.confidence_score == 0.0

    def test_low_confidence_no_escalation_with_budget(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.30),
            _ctx(exploration_round=1, max_exploration_rounds=5),
        )
        assert result.should_escalate is False

    def test_low_confidence_escalates_at_budget_limit(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.30),
            _ctx(exploration_round=5, max_exploration_rounds=5),
        )
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING
        assert result.should_escalate is True
        assert result.escalation_reason != ""

    def test_low_confidence_tier(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.40), _ctx())
        assert result.confidence_tier == ConfidenceTier.LOW

    def test_just_below_60_investigates(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.59), _ctx())
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    """Test exact threshold boundaries."""

    def test_exactly_90_is_tier_1(self) -> None:
        engine = DecisionEngine()
        verdict = engine.classify_score(0.90)
        assert verdict == DecisionVerdict.EXECUTE_IMMEDIATELY

    def test_89_99_is_tier_2(self) -> None:
        engine = DecisionEngine()
        verdict = engine.classify_score(0.8999)
        assert verdict == DecisionVerdict.EXECUTE_WITH_WARNING

    def test_exactly_60_is_tier_2(self) -> None:
        engine = DecisionEngine()
        verdict = engine.classify_score(0.60)
        assert verdict == DecisionVerdict.EXECUTE_WITH_WARNING

    def test_59_99_is_tier_3(self) -> None:
        engine = DecisionEngine()
        verdict = engine.classify_score(0.5999)
        assert verdict == DecisionVerdict.CONTINUE_INVESTIGATING

    def test_exactly_0_is_tier_3(self) -> None:
        engine = DecisionEngine()
        verdict = engine.classify_score(0.0)
        assert verdict == DecisionVerdict.CONTINUE_INVESTIGATING

    def test_exactly_1_is_tier_1(self) -> None:
        engine = DecisionEngine()
        verdict = engine.classify_score(1.0)
        assert verdict == DecisionVerdict.EXECUTE_IMMEDIATELY


# ---------------------------------------------------------------------------
# Budget exhaustion and escalation
# ---------------------------------------------------------------------------


class TestBudgetExhaustion:
    def test_exceeded_budget_always_escalates(self) -> None:
        """Exceeding max rounds forces escalation regardless of score."""
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.95),  # High confidence
            _ctx(exploration_round=6, max_exploration_rounds=5),
        )
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING
        assert result.should_escalate is True

    def test_at_budget_limit_high_confidence_proceeds(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.95),
            _ctx(exploration_round=5, max_exploration_rounds=5),
        )
        assert result.verdict == DecisionVerdict.EXECUTE_IMMEDIATELY
        assert result.should_escalate is False

    def test_at_budget_limit_medium_confidence_warns(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.70),
            _ctx(exploration_round=5, max_exploration_rounds=5),
        )
        assert result.verdict == DecisionVerdict.EXECUTE_WITH_WARNING

    def test_at_budget_limit_low_confidence_escalates(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.30),
            _ctx(exploration_round=5, max_exploration_rounds=5),
        )
        assert result.should_escalate is True
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING


# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------


class TestPreconditions:
    def test_insufficient_signals_forces_investigation(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.95),  # High score
            _ctx(positive_signal_count=1, min_positive_signals=3),
        )
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING

    def test_insufficient_signals_at_limit_escalates(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.95),
            _ctx(
                positive_signal_count=1,
                min_positive_signals=3,
                exploration_round=5,
                max_exploration_rounds=5,
            ),
        )
        assert result.should_escalate is True

    def test_framework_required_not_met_investigates(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.95),
            _ctx(require_framework=True, has_framework_detected=False),
        )
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING

    def test_framework_required_met_proceeds(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.95),
            _ctx(require_framework=True, has_framework_detected=True),
        )
        assert result.verdict == DecisionVerdict.EXECUTE_IMMEDIATELY


# ---------------------------------------------------------------------------
# Custom policies
# ---------------------------------------------------------------------------


class TestCustomPolicies:
    def test_aggressive_policy_lower_thresholds(self) -> None:
        policy = AutonomyPolicyConfig.aggressive()
        engine = DecisionEngine(policy=policy)
        # 0.75 > aggressive execute_threshold of 0.70
        result = engine.decide(_signals(0.75), _ctx())
        assert result.verdict == DecisionVerdict.EXECUTE_IMMEDIATELY

    def test_conservative_policy_higher_thresholds(self) -> None:
        policy = AutonomyPolicyConfig.conservative()
        engine = DecisionEngine(policy=policy)
        # 0.90 < conservative execute_threshold of 0.95
        verdict = engine.classify_score(0.90)
        assert verdict == DecisionVerdict.EXECUTE_WITH_WARNING

    def test_custom_thresholds(self) -> None:
        policy = AutonomyPolicyConfig(
            execute_threshold=0.80,
            warn_threshold=0.50,
        )
        engine = DecisionEngine(policy=policy)
        assert engine.classify_score(0.80) == DecisionVerdict.EXECUTE_IMMEDIATELY
        assert engine.classify_score(0.79) == DecisionVerdict.EXECUTE_WITH_WARNING
        assert engine.classify_score(0.50) == DecisionVerdict.EXECUTE_WITH_WARNING
        assert engine.classify_score(0.49) == DecisionVerdict.CONTINUE_INVESTIGATING

    def test_engine_exposes_thresholds(self) -> None:
        engine = DecisionEngine()
        assert engine.execute_threshold == 0.90
        assert engine.warn_threshold == 0.60


# ---------------------------------------------------------------------------
# classify_score — pure threshold routing
# ---------------------------------------------------------------------------


class TestClassifyScore:
    def test_all_tiers(self) -> None:
        engine = DecisionEngine()
        assert engine.classify_score(1.0) == DecisionVerdict.EXECUTE_IMMEDIATELY
        assert engine.classify_score(0.90) == DecisionVerdict.EXECUTE_IMMEDIATELY
        assert engine.classify_score(0.89) == DecisionVerdict.EXECUTE_WITH_WARNING
        assert engine.classify_score(0.75) == DecisionVerdict.EXECUTE_WITH_WARNING
        assert engine.classify_score(0.60) == DecisionVerdict.EXECUTE_WITH_WARNING
        assert engine.classify_score(0.59) == DecisionVerdict.CONTINUE_INVESTIGATING
        assert engine.classify_score(0.30) == DecisionVerdict.CONTINUE_INVESTIGATING
        assert engine.classify_score(0.0) == DecisionVerdict.CONTINUE_INVESTIGATING


# ---------------------------------------------------------------------------
# decide_from_score convenience method
# ---------------------------------------------------------------------------


class TestDecideFromScore:
    def test_high_score(self) -> None:
        engine = DecisionEngine()
        result = engine.decide_from_score(0.95, _ctx())
        assert result.verdict == DecisionVerdict.EXECUTE_IMMEDIATELY

    def test_medium_score(self) -> None:
        engine = DecisionEngine()
        result = engine.decide_from_score(0.70, _ctx())
        assert result.can_execute is True

    def test_low_score(self) -> None:
        engine = DecisionEngine()
        result = engine.decide_from_score(0.30, _ctx())
        assert result.needs_investigation is True

    def test_invalid_score_raises(self) -> None:
        engine = DecisionEngine()
        with pytest.raises(ValueError, match="Score must be"):
            engine.decide_from_score(1.5)

    def test_negative_score_raises(self) -> None:
        engine = DecisionEngine()
        with pytest.raises(ValueError, match="Score must be"):
            engine.decide_from_score(-0.1)

    def test_with_context(self) -> None:
        engine = DecisionEngine()
        result = engine.decide_from_score(
            0.95,
            _ctx(positive_signal_count=0, min_positive_signals=2),
        )
        # Despite high score, insufficient signals force investigation
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING


# ---------------------------------------------------------------------------
# DecisionResult properties and serialization
# ---------------------------------------------------------------------------


class TestDecisionResult:
    def test_score_percentage(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.85), _ctx())
        assert result.score_percentage == 85.0

    def test_confidence_score_shortcut(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.75), _ctx())
        assert abs(result.confidence_score - 0.75) < 0.01

    def test_summary_keys(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.80), _ctx())
        s = result.summary()
        assert "verdict" in s
        assert "confidence_score" in s
        assert "confidence_tier" in s
        assert "score_percentage" in s
        assert "can_execute" in s
        assert "needs_warning" in s
        assert "needs_investigation" in s
        assert "should_escalate" in s
        assert "reason" in s
        assert "exploration_round" in s

    def test_summary_includes_escalation_reason_when_escalating(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(
            _signals(0.30),
            _ctx(exploration_round=5, max_exploration_rounds=5),
        )
        s = result.summary()
        assert "escalation_reason" in s

    def test_summary_excludes_escalation_reason_when_not_escalating(self) -> None:
        engine = DecisionEngine()
        result = engine.decide(_signals(0.95), _ctx())
        s = result.summary()
        assert "escalation_reason" not in s

    def test_summary_includes_metadata_when_present(self) -> None:
        engine = DecisionEngine()
        # Near-threshold produces metadata with "near_threshold" flag
        result = engine.decide(
            _signals(0.85),
            _ctx(exploration_round=1, max_exploration_rounds=5),
        )
        if result.metadata:
            s = result.summary()
            assert "metadata" in s


# ---------------------------------------------------------------------------
# Integration with AutonomyEngine
# ---------------------------------------------------------------------------


class TestIntegrationWithAutonomyEngine:
    """Verify DecisionEngine can be used alongside AutonomyEngine."""

    def test_same_policy_same_thresholds(self) -> None:
        from test_runner.autonomy.engine import AutonomyEngine

        policy = AutonomyPolicyConfig()
        ae = AutonomyEngine(policy=policy)
        de = DecisionEngine(policy=policy)
        assert ae.policy.execute_threshold == de.execute_threshold
        assert ae.policy.warn_threshold == de.warn_threshold

    def test_decision_engine_agrees_with_autonomy_engine_high(self) -> None:
        """Both engines should agree on high-confidence proceed."""
        from test_runner.autonomy.engine import (
            AutonomyEngine,
            DiscoveryFindings,
            ExplorationAction,
        )

        signals = _signals(0.95)
        findings = DiscoveryFindings(
            signals=signals,
            frameworks_detected=[{"framework": "pytest", "confidence": 0.95}],
            test_files=["test.py", "test2.py"],
            working_directory="/p",
            exploration_round=1,
        )

        ae = AutonomyEngine()
        de = DecisionEngine()

        ae_decision = ae.evaluate(findings)
        de_decision = de.decide(signals, _ctx())

        assert ae_decision.action == ExplorationAction.PROCEED
        assert de_decision.verdict == DecisionVerdict.EXECUTE_IMMEDIATELY

    def test_decision_engine_agrees_with_autonomy_engine_low(self) -> None:
        """Both engines should agree on low-confidence investigation."""
        from test_runner.autonomy.engine import (
            AutonomyEngine,
            DiscoveryFindings,
            ExplorationAction,
        )

        signals = _signals(0.20)
        findings = DiscoveryFindings(
            signals=signals,
            working_directory="/p",
            exploration_round=1,
        )

        ae = AutonomyEngine()
        de = DecisionEngine()

        ae_decision = ae.evaluate(findings)
        de_decision = de.decide(signals, _ctx(positive_signal_count=0))

        assert ae_decision.action == ExplorationAction.EXPLORE_FURTHER
        assert de_decision.verdict == DecisionVerdict.CONTINUE_INVESTIGATING


# ---------------------------------------------------------------------------
# Weighted signal scenarios
# ---------------------------------------------------------------------------


class TestWeightedSignals:
    def test_high_weight_low_score_signal_dominates(self) -> None:
        engine = DecisionEngine()
        signals = [
            ConfidenceSignal(name="strong_neg", weight=0.9, score=0.20),
            ConfidenceSignal(name="weak_pos", weight=0.1, score=0.95),
        ]
        result = engine.decide(signals, _ctx())
        # Weighted avg ≈ (0.9*0.2 + 0.1*0.95) / (0.9+0.1) = 0.275
        assert result.verdict == DecisionVerdict.CONTINUE_INVESTIGATING

    def test_high_weight_high_score_signal_dominates(self) -> None:
        engine = DecisionEngine()
        signals = [
            ConfidenceSignal(name="strong_pos", weight=0.9, score=0.95),
            ConfidenceSignal(name="weak_neg", weight=0.1, score=0.10),
        ]
        result = engine.decide(signals, _ctx())
        # Weighted avg ≈ (0.9*0.95 + 0.1*0.10) / (0.9+0.1) = 0.865
        # This is in the near-threshold zone (within 10% of 0.90),
        # so engine may choose to investigate further or warn
        assert result.confidence_score == pytest.approx(0.865, abs=0.01)
        assert result.confidence_tier == ConfidenceTier.MEDIUM

    def test_mixed_signals_in_warn_zone(self) -> None:
        engine = DecisionEngine()
        signals = [
            ConfidenceSignal(name="a", weight=1.0, score=0.80),
            ConfidenceSignal(name="b", weight=1.0, score=0.60),
            ConfidenceSignal(name="c", weight=1.0, score=0.70),
        ]
        result = engine.decide(signals, _ctx())
        # Weighted avg = 0.70, in warn zone
        assert result.confidence_score == pytest.approx(0.70, abs=0.01)
