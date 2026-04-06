"""Tests for the ConfidenceDecision enum and three-tier threshold-based routing.

Verifies:
- ConfidenceDecision enum values and properties
- Three-tier classification: >=90% EXECUTE, 60-90% WARN, <60% INVESTIGATE
- from_score() with default and custom thresholds
- from_tier() mapping
- Integration with ConfidenceModel.decide() and decide_from_score()
- Integration with ConfidenceResult.decision property
- Bidirectional mapping with DecisionVerdict
- Edge cases at tier boundaries
"""

from __future__ import annotations

import pytest

from test_runner.models.confidence import (
    ConfidenceDecision,
    ConfidenceModel,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceTier,
    EXECUTE_THRESHOLD,
    WARN_THRESHOLD,
)
from test_runner.autonomy.decision_engine import (
    DecisionContext,
    DecisionEngine,
    DecisionResult,
    DecisionVerdict,
)
from test_runner.autonomy.policy import AutonomyPolicyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signals(score: float, count: int = 3) -> list[ConfidenceSignal]:
    """Create uniform signals that aggregate to the given score."""
    return [
        ConfidenceSignal(name=f"sig_{i}", weight=1.0, score=score)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# ConfidenceDecision enum basics
# ---------------------------------------------------------------------------


class TestConfidenceDecisionEnum:
    def test_three_decisions_exist(self) -> None:
        assert len(ConfidenceDecision) == 3

    def test_values(self) -> None:
        assert ConfidenceDecision.EXECUTE.value == "execute"
        assert ConfidenceDecision.WARN.value == "warn"
        assert ConfidenceDecision.INVESTIGATE.value == "investigate"

    def test_is_str_enum(self) -> None:
        """ConfidenceDecision is a str enum for easy serialization."""
        assert isinstance(ConfidenceDecision.EXECUTE, str)
        assert ConfidenceDecision.EXECUTE == "execute"

    def test_can_execute_property(self) -> None:
        assert ConfidenceDecision.EXECUTE.can_execute is True
        assert ConfidenceDecision.WARN.can_execute is True
        assert ConfidenceDecision.INVESTIGATE.can_execute is False

    def test_needs_warning_property(self) -> None:
        assert ConfidenceDecision.EXECUTE.needs_warning is False
        assert ConfidenceDecision.WARN.needs_warning is True
        assert ConfidenceDecision.INVESTIGATE.needs_warning is False

    def test_needs_investigation_property(self) -> None:
        assert ConfidenceDecision.EXECUTE.needs_investigation is False
        assert ConfidenceDecision.WARN.needs_investigation is False
        assert ConfidenceDecision.INVESTIGATE.needs_investigation is True


# ---------------------------------------------------------------------------
# from_score — default thresholds (90% / 60%)
# ---------------------------------------------------------------------------


class TestFromScoreDefaultThresholds:
    """Three-tier classification with default 90%/60% boundaries."""

    def test_at_100_percent(self) -> None:
        assert ConfidenceDecision.from_score(1.0) == ConfidenceDecision.EXECUTE

    def test_at_95_percent(self) -> None:
        assert ConfidenceDecision.from_score(0.95) == ConfidenceDecision.EXECUTE

    def test_at_exactly_90_percent(self) -> None:
        """>=90% is EXECUTE (boundary inclusive)."""
        assert ConfidenceDecision.from_score(0.90) == ConfidenceDecision.EXECUTE

    def test_at_89_percent(self) -> None:
        """Just below 90% is WARN."""
        assert ConfidenceDecision.from_score(0.89) == ConfidenceDecision.WARN

    def test_at_75_percent(self) -> None:
        assert ConfidenceDecision.from_score(0.75) == ConfidenceDecision.WARN

    def test_at_exactly_60_percent(self) -> None:
        """>=60% is WARN (boundary inclusive)."""
        assert ConfidenceDecision.from_score(0.60) == ConfidenceDecision.WARN

    def test_at_59_percent(self) -> None:
        """Just below 60% is INVESTIGATE."""
        assert ConfidenceDecision.from_score(0.59) == ConfidenceDecision.INVESTIGATE

    def test_at_30_percent(self) -> None:
        assert ConfidenceDecision.from_score(0.30) == ConfidenceDecision.INVESTIGATE

    def test_at_0_percent(self) -> None:
        assert ConfidenceDecision.from_score(0.0) == ConfidenceDecision.INVESTIGATE

    def test_uses_correct_default_thresholds(self) -> None:
        """Verify the defaults match the module constants."""
        assert EXECUTE_THRESHOLD == 0.90
        assert WARN_THRESHOLD == 0.60


# ---------------------------------------------------------------------------
# from_score — custom thresholds
# ---------------------------------------------------------------------------


class TestFromScoreCustomThresholds:
    def test_custom_high_threshold(self) -> None:
        # Conservative: execute only at 95%+
        d = ConfidenceDecision.from_score(0.92, execute_threshold=0.95, warn_threshold=0.60)
        assert d == ConfidenceDecision.WARN

    def test_custom_low_threshold(self) -> None:
        # Aggressive: warn starts at 40%
        d = ConfidenceDecision.from_score(0.45, execute_threshold=0.90, warn_threshold=0.40)
        assert d == ConfidenceDecision.WARN

    def test_custom_both_thresholds(self) -> None:
        d = ConfidenceDecision.from_score(0.80, execute_threshold=0.80, warn_threshold=0.50)
        assert d == ConfidenceDecision.EXECUTE

    def test_invalid_score_above_1(self) -> None:
        with pytest.raises(ValueError, match="score must be in"):
            ConfidenceDecision.from_score(1.5)

    def test_invalid_score_below_0(self) -> None:
        with pytest.raises(ValueError, match="score must be in"):
            ConfidenceDecision.from_score(-0.1)

    def test_invalid_thresholds_warn_above_execute(self) -> None:
        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            ConfidenceDecision.from_score(0.5, execute_threshold=0.5, warn_threshold=0.8)


# ---------------------------------------------------------------------------
# from_tier mapping
# ---------------------------------------------------------------------------


class TestFromTier:
    def test_high_to_execute(self) -> None:
        assert ConfidenceDecision.from_tier(ConfidenceTier.HIGH) == ConfidenceDecision.EXECUTE

    def test_medium_to_warn(self) -> None:
        assert ConfidenceDecision.from_tier(ConfidenceTier.MEDIUM) == ConfidenceDecision.WARN

    def test_low_to_investigate(self) -> None:
        assert ConfidenceDecision.from_tier(ConfidenceTier.LOW) == ConfidenceDecision.INVESTIGATE


# ---------------------------------------------------------------------------
# ConfidenceResult.decision property
# ---------------------------------------------------------------------------


class TestConfidenceResultDecision:
    def test_high_tier_gives_execute_decision(self) -> None:
        model = ConfidenceModel()
        result = model.evaluate(_signals(0.95))
        assert result.decision == ConfidenceDecision.EXECUTE

    def test_medium_tier_gives_warn_decision(self) -> None:
        model = ConfidenceModel()
        result = model.evaluate(_signals(0.75))
        assert result.decision == ConfidenceDecision.WARN

    def test_low_tier_gives_investigate_decision(self) -> None:
        model = ConfidenceModel()
        result = model.evaluate(_signals(0.30))
        assert result.decision == ConfidenceDecision.INVESTIGATE

    def test_decision_in_summary(self) -> None:
        model = ConfidenceModel()
        result = model.evaluate(_signals(0.95))
        s = result.summary()
        assert "decision" in s
        assert s["decision"] == "execute"


# ---------------------------------------------------------------------------
# ConfidenceModel.decide() and decide_from_score()
# ---------------------------------------------------------------------------


class TestConfidenceModelDecide:
    def test_decide_high_confidence(self) -> None:
        model = ConfidenceModel()
        d = model.decide(_signals(0.95))
        assert d == ConfidenceDecision.EXECUTE

    def test_decide_medium_confidence(self) -> None:
        model = ConfidenceModel()
        d = model.decide(_signals(0.75))
        assert d == ConfidenceDecision.WARN

    def test_decide_low_confidence(self) -> None:
        model = ConfidenceModel()
        d = model.decide(_signals(0.30))
        assert d == ConfidenceDecision.INVESTIGATE

    def test_decide_empty_signals(self) -> None:
        model = ConfidenceModel()
        d = model.decide([])
        assert d == ConfidenceDecision.INVESTIGATE

    def test_decide_from_score_high(self) -> None:
        model = ConfidenceModel()
        assert model.decide_from_score(0.95) == ConfidenceDecision.EXECUTE

    def test_decide_from_score_medium(self) -> None:
        model = ConfidenceModel()
        assert model.decide_from_score(0.70) == ConfidenceDecision.WARN

    def test_decide_from_score_low(self) -> None:
        model = ConfidenceModel()
        assert model.decide_from_score(0.40) == ConfidenceDecision.INVESTIGATE

    def test_decide_from_score_respects_custom_thresholds(self) -> None:
        model = ConfidenceModel(execute_threshold=0.80, warn_threshold=0.50)
        assert model.decide_from_score(0.80) == ConfidenceDecision.EXECUTE
        assert model.decide_from_score(0.79) == ConfidenceDecision.WARN
        assert model.decide_from_score(0.50) == ConfidenceDecision.WARN
        assert model.decide_from_score(0.49) == ConfidenceDecision.INVESTIGATE


# ---------------------------------------------------------------------------
# Bidirectional mapping with DecisionVerdict
# ---------------------------------------------------------------------------


class TestDecisionVerdictMapping:
    def test_verdict_from_confidence_decision(self) -> None:
        assert (
            DecisionVerdict.from_confidence_decision(ConfidenceDecision.EXECUTE)
            == DecisionVerdict.EXECUTE_IMMEDIATELY
        )
        assert (
            DecisionVerdict.from_confidence_decision(ConfidenceDecision.WARN)
            == DecisionVerdict.EXECUTE_WITH_WARNING
        )
        assert (
            DecisionVerdict.from_confidence_decision(ConfidenceDecision.INVESTIGATE)
            == DecisionVerdict.CONTINUE_INVESTIGATING
        )

    def test_verdict_to_confidence_decision(self) -> None:
        assert (
            DecisionVerdict.EXECUTE_IMMEDIATELY.to_confidence_decision()
            == ConfidenceDecision.EXECUTE
        )
        assert (
            DecisionVerdict.EXECUTE_WITH_WARNING.to_confidence_decision()
            == ConfidenceDecision.WARN
        )
        assert (
            DecisionVerdict.CONTINUE_INVESTIGATING.to_confidence_decision()
            == ConfidenceDecision.INVESTIGATE
        )

    def test_roundtrip(self) -> None:
        """Converting to verdict and back yields the original decision."""
        for cd in ConfidenceDecision:
            verdict = DecisionVerdict.from_confidence_decision(cd)
            assert verdict.to_confidence_decision() == cd


# ---------------------------------------------------------------------------
# DecisionResult.decision property
# ---------------------------------------------------------------------------


class TestDecisionResultDecision:
    def test_execute_immediately_maps_to_execute(self) -> None:
        engine = DecisionEngine()
        ctx = DecisionContext(positive_signal_count=5, has_framework_detected=True)
        result = engine.decide(_signals(0.95), ctx)
        assert result.decision == ConfidenceDecision.EXECUTE

    def test_execute_with_warning_maps_to_warn(self) -> None:
        engine = DecisionEngine()
        ctx = DecisionContext(positive_signal_count=5, has_framework_detected=True)
        result = engine.decide(_signals(0.70), ctx)
        assert result.decision == ConfidenceDecision.WARN

    def test_continue_investigating_maps_to_investigate(self) -> None:
        engine = DecisionEngine()
        ctx = DecisionContext(positive_signal_count=5, has_framework_detected=True)
        result = engine.decide(_signals(0.30), ctx)
        assert result.decision == ConfidenceDecision.INVESTIGATE


# ---------------------------------------------------------------------------
# Boundary precision tests
# ---------------------------------------------------------------------------


class TestBoundaryPrecision:
    """Verify exact boundary behavior with floating point edge cases."""

    def test_execute_threshold_boundary(self) -> None:
        # Exactly at threshold: EXECUTE
        assert ConfidenceDecision.from_score(0.90) == ConfidenceDecision.EXECUTE
        # Just below: WARN
        assert ConfidenceDecision.from_score(0.8999999) == ConfidenceDecision.WARN

    def test_warn_threshold_boundary(self) -> None:
        # Exactly at threshold: WARN
        assert ConfidenceDecision.from_score(0.60) == ConfidenceDecision.WARN
        # Just below: INVESTIGATE
        assert ConfidenceDecision.from_score(0.5999999) == ConfidenceDecision.INVESTIGATE

    def test_zero_and_one(self) -> None:
        assert ConfidenceDecision.from_score(0.0) == ConfidenceDecision.INVESTIGATE
        assert ConfidenceDecision.from_score(1.0) == ConfidenceDecision.EXECUTE

    def test_equal_thresholds(self) -> None:
        """When execute == warn threshold, only EXECUTE and INVESTIGATE exist."""
        d_at = ConfidenceDecision.from_score(0.70, execute_threshold=0.70, warn_threshold=0.70)
        assert d_at == ConfidenceDecision.EXECUTE

        d_below = ConfidenceDecision.from_score(0.69, execute_threshold=0.70, warn_threshold=0.70)
        assert d_below == ConfidenceDecision.INVESTIGATE
