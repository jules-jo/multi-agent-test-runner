"""Tests for ConfidenceModel, ConfidenceResult, and CompositeWeights."""

import pytest

from test_runner.models.confidence import (
    CompositeWeights,
    ConfidenceModel,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceTier,
    DEFAULT_COMPOSITE_WEIGHTS,
    EXECUTE_THRESHOLD,
    LLM_SIGNAL_PREFIX,
    WARN_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig(name: str = "s", weight: float = 1.0, score: float = 1.0) -> ConfidenceSignal:
    return ConfidenceSignal(name=name, weight=weight, score=score)


# ---------------------------------------------------------------------------
# ConfidenceModel construction
# ---------------------------------------------------------------------------


class TestConfidenceModelInit:
    def test_default_thresholds(self):
        model = ConfidenceModel()
        assert model.execute_threshold == EXECUTE_THRESHOLD  # 0.90
        assert model.warn_threshold == WARN_THRESHOLD  # 0.60

    def test_custom_thresholds(self):
        model = ConfidenceModel(execute_threshold=0.95, warn_threshold=0.70)
        assert model.execute_threshold == 0.95
        assert model.warn_threshold == 0.70

    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            ConfidenceModel(execute_threshold=0.50, warn_threshold=0.80)

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            ConfidenceModel(warn_threshold=-0.1)

    def test_rejects_threshold_above_one(self):
        with pytest.raises(ValueError):
            ConfidenceModel(execute_threshold=1.1)

    def test_equal_thresholds_allowed(self):
        model = ConfidenceModel(execute_threshold=0.80, warn_threshold=0.80)
        assert model.execute_threshold == model.warn_threshold


# ---------------------------------------------------------------------------
# Tier classification (>=90% execute, 60-90% warn, <60% investigate)
# ---------------------------------------------------------------------------


class TestTierClassification:
    """Verify the canonical tier thresholds from the spec."""

    def test_score_100_executes(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=1.0)])
        assert result.tier == ConfidenceTier.HIGH
        assert result.should_execute is True

    def test_score_90_executes(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.90)])
        assert result.tier == ConfidenceTier.HIGH
        assert result.should_execute is True

    def test_score_89_warns(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.89)])
        assert result.tier == ConfidenceTier.MEDIUM
        assert result.should_warn is True
        assert result.should_execute is False

    def test_score_60_warns(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.60)])
        assert result.tier == ConfidenceTier.MEDIUM
        assert result.should_warn is True

    def test_score_59_investigates(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.59)])
        assert result.tier == ConfidenceTier.LOW
        assert result.should_investigate is True
        assert result.should_execute is False
        assert result.should_warn is False

    def test_score_0_investigates(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.0)])
        assert result.tier == ConfidenceTier.LOW
        assert result.should_investigate is True

    def test_empty_signals_investigates(self):
        model = ConfidenceModel()
        result = model.evaluate([])
        assert result.score == 0.0
        assert result.tier == ConfidenceTier.LOW
        assert result.should_investigate is True


# ---------------------------------------------------------------------------
# Weighted aggregation
# ---------------------------------------------------------------------------


class TestWeightedAggregation:
    def test_single_signal_passthrough(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.75, weight=1.0)])
        assert result.score == pytest.approx(0.75)

    def test_equal_weights(self):
        model = ConfidenceModel()
        signals = [_sig("a", weight=1.0, score=1.0), _sig("b", weight=1.0, score=0.5)]
        result = model.evaluate(signals)
        # (1.0*1.0 + 1.0*0.5) / (1.0+1.0) = 0.75
        assert result.score == pytest.approx(0.75)

    def test_unequal_weights(self):
        model = ConfidenceModel()
        signals = [_sig("a", weight=0.8, score=1.0), _sig("b", weight=0.2, score=0.0)]
        result = model.evaluate(signals)
        # (0.8*1.0 + 0.2*0.0) / (0.8+0.2) = 0.8
        assert result.score == pytest.approx(0.8)

    def test_heavy_weight_dominates(self):
        model = ConfidenceModel()
        signals = [
            _sig("high", weight=0.9, score=0.95),
            _sig("low", weight=0.1, score=0.10),
        ]
        result = model.evaluate(signals)
        expected = (0.9 * 0.95 + 0.1 * 0.10) / (0.9 + 0.1)
        assert result.score == pytest.approx(expected)

    def test_zero_weight_signals_ignored_in_score(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(weight=0.0, score=1.0)])
        assert result.score == 0.0

    def test_many_signals(self):
        model = ConfidenceModel()
        signals = [_sig(f"s{i}", weight=0.5, score=0.92) for i in range(20)]
        result = model.evaluate(signals)
        assert result.score == pytest.approx(0.92)
        assert result.tier == ConfidenceTier.HIGH


# ---------------------------------------------------------------------------
# ConfidenceResult properties
# ---------------------------------------------------------------------------


class TestConfidenceResult:
    def test_immutable(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.95)])
        with pytest.raises(AttributeError):
            result.score = 0.5  # type: ignore[misc]

    def test_signals_tuple(self):
        model = ConfidenceModel()
        sigs = [_sig("a"), _sig("b")]
        result = model.evaluate(sigs)
        assert isinstance(result.signals, tuple)
        assert len(result.signals) == 2

    def test_thresholds_stored(self):
        model = ConfidenceModel(execute_threshold=0.95, warn_threshold=0.70)
        result = model.evaluate([_sig(score=0.80)])
        assert result.execute_threshold == 0.95
        assert result.warn_threshold == 0.70

    def test_summary_keys(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig("x", score=0.92)])
        s = result.summary()
        assert set(s.keys()) == {
            "score",
            "tier",
            "decision",
            "action",
            "execute_threshold",
            "warn_threshold",
            "signal_count",
            "signals",
        }

    def test_summary_action_execute(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.95)])
        assert result.summary()["action"] == "execute"

    def test_summary_action_warn(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.75)])
        assert result.summary()["action"] == "warn"

    def test_summary_action_investigate(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.30)])
        assert result.summary()["action"] == "investigate"

    def test_summary_signal_detail(self):
        model = ConfidenceModel()
        result = model.evaluate([_sig("pytest_found", weight=0.8, score=0.9)])
        sig_detail = result.summary()["signals"][0]
        assert sig_detail["name"] == "pytest_found"
        assert sig_detail["weight"] == 0.8
        assert sig_detail["score"] == 0.9
        assert sig_detail["weighted_score"] == pytest.approx(0.72)


# ---------------------------------------------------------------------------
# Custom thresholds (autonomy policy)
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    def test_strict_policy(self):
        """Strict policy: execute only at 95%, warn at 75%."""
        model = ConfidenceModel(execute_threshold=0.95, warn_threshold=0.75)
        assert model.evaluate([_sig(score=0.94)]).tier == ConfidenceTier.MEDIUM
        assert model.evaluate([_sig(score=0.95)]).tier == ConfidenceTier.HIGH
        assert model.evaluate([_sig(score=0.74)]).tier == ConfidenceTier.LOW

    def test_lenient_policy(self):
        """Lenient policy: execute at 70%, warn at 40%."""
        model = ConfidenceModel(execute_threshold=0.70, warn_threshold=0.40)
        assert model.evaluate([_sig(score=0.70)]).tier == ConfidenceTier.HIGH
        assert model.evaluate([_sig(score=0.50)]).tier == ConfidenceTier.MEDIUM
        assert model.evaluate([_sig(score=0.39)]).tier == ConfidenceTier.LOW


# ---------------------------------------------------------------------------
# Edge cases: no signals, single signal, conflicting signals
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case scenarios required by the confidence_autonomy evaluation."""

    # -- No signals ----------------------------------------------------------

    def test_no_signals_returns_zero_score(self):
        """Empty signal list must yield score 0.0."""
        model = ConfidenceModel()
        result = model.evaluate([])
        assert result.score == 0.0

    def test_no_signals_tier_is_low(self):
        """With no evidence the agent must investigate (LOW tier)."""
        model = ConfidenceModel()
        result = model.evaluate([])
        assert result.tier == ConfidenceTier.LOW
        assert result.should_investigate is True
        assert result.should_execute is False
        assert result.should_warn is False

    def test_no_signals_summary_is_valid(self):
        """Summary should still serialize cleanly with zero signals."""
        model = ConfidenceModel()
        result = model.evaluate([])
        s = result.summary()
        assert s["signal_count"] == 0
        assert s["signals"] == []
        assert s["action"] == "investigate"

    # -- Single signal -------------------------------------------------------

    def test_single_signal_score_equals_signal_score(self):
        """With one signal, aggregated score == that signal's score."""
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.73, weight=0.5)])
        assert result.score == pytest.approx(0.73)

    def test_single_signal_zero_weight(self):
        """A single zero-weight signal yields 0.0 (no evidence)."""
        model = ConfidenceModel()
        result = model.evaluate([_sig(weight=0.0, score=1.0)])
        assert result.score == 0.0
        assert result.tier == ConfidenceTier.LOW

    def test_single_signal_zero_score(self):
        """A signal with score 0 contributes nothing."""
        model = ConfidenceModel()
        result = model.evaluate([_sig(weight=1.0, score=0.0)])
        assert result.score == 0.0
        assert result.tier == ConfidenceTier.LOW

    def test_single_signal_perfect(self):
        """A single perfect signal should execute."""
        model = ConfidenceModel()
        result = model.evaluate([_sig(weight=1.0, score=1.0)])
        assert result.score == 1.0
        assert result.tier == ConfidenceTier.HIGH

    # -- Conflicting signals -------------------------------------------------

    def test_conflicting_equal_weight_averages(self):
        """Two conflicting signals with equal weight should average to 0.5."""
        model = ConfidenceModel()
        signals = [
            _sig("positive", weight=1.0, score=1.0),
            _sig("negative", weight=1.0, score=0.0),
        ]
        result = model.evaluate(signals)
        assert result.score == pytest.approx(0.5)
        # 0.5 is below 0.60 warn threshold → LOW
        assert result.tier == ConfidenceTier.LOW

    def test_conflicting_positive_heavier(self):
        """Positive signal with higher weight should dominate."""
        model = ConfidenceModel()
        signals = [
            _sig("strong_positive", weight=0.9, score=1.0),
            _sig("weak_negative", weight=0.1, score=0.0),
        ]
        result = model.evaluate(signals)
        # (0.9*1.0 + 0.1*0.0) / (0.9+0.1) = 0.9
        assert result.score == pytest.approx(0.9)
        assert result.tier == ConfidenceTier.HIGH

    def test_conflicting_negative_heavier(self):
        """Negative signal with higher weight should dominate."""
        model = ConfidenceModel()
        signals = [
            _sig("weak_positive", weight=0.1, score=1.0),
            _sig("strong_negative", weight=0.9, score=0.0),
        ]
        result = model.evaluate(signals)
        # (0.1*1.0 + 0.9*0.0) / (0.1+0.9) = 0.1
        assert result.score == pytest.approx(0.1)
        assert result.tier == ConfidenceTier.LOW

    def test_conflicting_many_signals_cancel_out(self):
        """Many conflicting signals should average toward the middle."""
        model = ConfidenceModel()
        signals = [
            _sig(f"pos_{i}", weight=1.0, score=1.0) for i in range(5)
        ] + [
            _sig(f"neg_{i}", weight=1.0, score=0.0) for i in range(5)
        ]
        result = model.evaluate(signals)
        assert result.score == pytest.approx(0.5)
        assert result.tier == ConfidenceTier.LOW

    def test_conflicting_mixed_partial_scores(self):
        """Signals with partial scores in both directions."""
        model = ConfidenceModel()
        signals = [
            _sig("high_conf", weight=0.8, score=0.95),
            _sig("low_conf", weight=0.8, score=0.20),
            _sig("mid_conf", weight=0.4, score=0.60),
        ]
        result = model.evaluate(signals)
        # (0.8*0.95 + 0.8*0.20 + 0.4*0.60) / (0.8+0.8+0.4)
        expected = (0.76 + 0.16 + 0.24) / 2.0
        assert result.score == pytest.approx(expected)

    # -- All zero-weight signals ---------------------------------------------

    def test_all_zero_weight_signals(self):
        """Multiple signals all with zero weight should return 0.0."""
        model = ConfidenceModel()
        signals = [
            _sig("a", weight=0.0, score=1.0),
            _sig("b", weight=0.0, score=0.5),
            _sig("c", weight=0.0, score=0.9),
        ]
        result = model.evaluate(signals)
        assert result.score == 0.0
        assert result.tier == ConfidenceTier.LOW
        # Signals are still recorded even though they don't affect score
        assert len(result.signals) == 3

    # -- Boundary precision --------------------------------------------------

    def test_exact_execute_boundary(self):
        """Score exactly at execute threshold should be HIGH."""
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.90)])
        assert result.tier == ConfidenceTier.HIGH

    def test_just_below_execute_boundary(self):
        """Score epsilon below execute threshold should be MEDIUM."""
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.8999999)])
        assert result.tier == ConfidenceTier.MEDIUM

    def test_exact_warn_boundary(self):
        """Score exactly at warn threshold should be MEDIUM."""
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.60)])
        assert result.tier == ConfidenceTier.MEDIUM

    def test_just_below_warn_boundary(self):
        """Score epsilon below warn threshold should be LOW."""
        model = ConfidenceModel()
        result = model.evaluate([_sig(score=0.5999999)])
        assert result.tier == ConfidenceTier.LOW

    # -- Signal preservation -------------------------------------------------

    def test_signals_preserved_in_result(self):
        """All input signals should be available in the result."""
        model = ConfidenceModel()
        signals = [_sig(f"s{i}", score=0.5) for i in range(10)]
        result = model.evaluate(signals)
        assert len(result.signals) == 10
        assert all(s.name == f"s{i}" for i, s in enumerate(result.signals))

    def test_result_signals_are_immutable_tuple(self):
        """Result signals should be a tuple (immutable)."""
        model = ConfidenceModel()
        result = model.evaluate([_sig()])
        assert isinstance(result.signals, tuple)


# ---------------------------------------------------------------------------
# ConfidenceSignal validation edge cases
# ---------------------------------------------------------------------------


class TestConfidenceSignalEdgeCases:
    """Signal construction validation."""

    def test_weight_at_zero(self):
        sig = ConfidenceSignal(name="z", weight=0.0, score=0.5)
        assert sig.weighted_score == 0.0

    def test_score_at_zero(self):
        sig = ConfidenceSignal(name="z", weight=0.5, score=0.0)
        assert sig.weighted_score == 0.0

    def test_weight_at_one(self):
        sig = ConfidenceSignal(name="z", weight=1.0, score=0.7)
        assert sig.weighted_score == pytest.approx(0.7)

    def test_score_at_one(self):
        sig = ConfidenceSignal(name="z", weight=0.3, score=1.0)
        assert sig.weighted_score == pytest.approx(0.3)

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="weight"):
            ConfidenceSignal(name="bad", weight=-0.01, score=0.5)

    def test_rejects_weight_above_one(self):
        with pytest.raises(ValueError, match="weight"):
            ConfidenceSignal(name="bad", weight=1.01, score=0.5)

    def test_rejects_negative_score(self):
        with pytest.raises(ValueError, match="score"):
            ConfidenceSignal(name="bad", weight=0.5, score=-0.01)

    def test_rejects_score_above_one(self):
        with pytest.raises(ValueError, match="score"):
            ConfidenceSignal(name="bad", weight=0.5, score=1.01)


# ---------------------------------------------------------------------------
# CompositeWeights
# ---------------------------------------------------------------------------


class TestCompositeWeights:
    """Validate CompositeWeights construction and normalisation."""

    def test_default_weights(self):
        cw = CompositeWeights()
        assert cw.evidence == 0.7
        assert cw.llm == 0.3

    def test_custom_weights(self):
        cw = CompositeWeights(evidence=0.5, llm=0.5)
        assert cw.normalized_evidence == pytest.approx(0.5)
        assert cw.normalized_llm == pytest.approx(0.5)

    def test_normalization(self):
        cw = CompositeWeights(evidence=2.0, llm=1.0)
        assert cw.normalized_evidence == pytest.approx(2.0 / 3.0)
        assert cw.normalized_llm == pytest.approx(1.0 / 3.0)

    def test_evidence_only(self):
        cw = CompositeWeights(evidence=1.0, llm=0.0)
        assert cw.normalized_evidence == 1.0
        assert cw.normalized_llm == 0.0

    def test_llm_only(self):
        cw = CompositeWeights(evidence=0.0, llm=1.0)
        assert cw.normalized_evidence == 0.0
        assert cw.normalized_llm == 1.0

    def test_rejects_both_zero(self):
        with pytest.raises(ValueError, match="At least one"):
            CompositeWeights(evidence=0.0, llm=0.0)

    def test_rejects_negative_evidence(self):
        with pytest.raises(ValueError, match="non-negative"):
            CompositeWeights(evidence=-0.1, llm=0.5)

    def test_rejects_negative_llm(self):
        with pytest.raises(ValueError, match="non-negative"):
            CompositeWeights(evidence=0.5, llm=-0.1)

    def test_frozen(self):
        cw = CompositeWeights()
        with pytest.raises(AttributeError):
            cw.evidence = 0.5  # type: ignore[misc]

    def test_default_constant(self):
        assert DEFAULT_COMPOSITE_WEIGHTS.evidence == 0.7
        assert DEFAULT_COMPOSITE_WEIGHTS.llm == 0.3


# ---------------------------------------------------------------------------
# ConfidenceModel — composite evaluation
# ---------------------------------------------------------------------------


class TestCompositeEvaluation:
    """Tests for evaluate_composite which blends evidence + LLM categories."""

    def test_evidence_only_no_llm(self):
        """When only evidence signals present, LLM weight gracefully drops."""
        model = ConfidenceModel()
        signals = [_sig("file_check", weight=1.0, score=0.95)]
        result = model.evaluate_composite(signals)
        # Only evidence → 100% evidence score
        assert result.score == pytest.approx(0.95)
        assert result.tier == ConfidenceTier.HIGH

    def test_llm_only_no_evidence(self):
        """When only LLM signals present, evidence weight gracefully drops."""
        model = ConfidenceModel()
        signals = [_sig("llm_self_assessment", weight=1.0, score=0.80)]
        result = model.evaluate_composite(signals)
        assert result.score == pytest.approx(0.80)
        assert result.tier == ConfidenceTier.MEDIUM

    def test_both_categories_default_weights(self):
        """Composite with default weights (0.7 evidence, 0.3 LLM)."""
        model = ConfidenceModel()
        signals = [
            _sig("pytest_found", weight=1.0, score=1.0),
            _sig("llm_self_assessment", weight=1.0, score=0.80),
        ]
        result = model.evaluate_composite(signals)
        # 0.7 * 1.0 + 0.3 * 0.8 = 0.94
        assert result.score == pytest.approx(0.94)
        assert result.tier == ConfidenceTier.HIGH

    def test_both_categories_equal_weights(self):
        """50/50 blend between evidence and LLM."""
        cw = CompositeWeights(evidence=0.5, llm=0.5)
        model = ConfidenceModel(composite_weights=cw)
        signals = [
            _sig("file_check", weight=1.0, score=1.0),
            _sig("llm_assessment", weight=1.0, score=0.60),
        ]
        result = model.evaluate_composite(signals)
        # 0.5 * 1.0 + 0.5 * 0.6 = 0.80
        assert result.score == pytest.approx(0.80)
        assert result.tier == ConfidenceTier.MEDIUM

    def test_override_weights_per_call(self):
        """Per-call weight override doesn't change model defaults."""
        model = ConfidenceModel()
        signals = [
            _sig("evidence_a", weight=1.0, score=1.0),
            _sig("llm_score", weight=1.0, score=0.50),
        ]
        override = CompositeWeights(evidence=0.5, llm=0.5)
        result = model.evaluate_composite(signals, weights=override)
        # 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        assert result.score == pytest.approx(0.75)
        # Model defaults unchanged
        assert model.composite_weights is DEFAULT_COMPOSITE_WEIGHTS

    def test_multiple_evidence_multiple_llm(self):
        """Multiple signals in each category are averaged within category."""
        model = ConfidenceModel(
            composite_weights=CompositeWeights(evidence=0.6, llm=0.4)
        )
        signals = [
            # Evidence signals: avg = (1.0*0.9 + 0.5*0.8) / (1.0+0.5) = 1.3/1.5
            _sig("file_a", weight=1.0, score=0.9),
            _sig("file_b", weight=0.5, score=0.8),
            # LLM signals: avg = (0.6*0.7 + 0.4*0.6) / (0.6+0.4) = 0.66
            _sig("llm_primary", weight=0.6, score=0.7),
            _sig("llm_secondary", weight=0.4, score=0.6),
        ]
        result = model.evaluate_composite(signals)
        evidence_avg = (1.0 * 0.9 + 0.5 * 0.8) / (1.0 + 0.5)
        llm_avg = (0.6 * 0.7 + 0.4 * 0.6) / (0.6 + 0.4)
        expected = 0.6 * evidence_avg + 0.4 * llm_avg
        assert result.score == pytest.approx(expected)

    def test_empty_signals_returns_zero(self):
        """No signals at all yields 0.0."""
        model = ConfidenceModel()
        result = model.evaluate_composite([])
        assert result.score == 0.0
        assert result.tier == ConfidenceTier.LOW

    def test_all_zero_weight_returns_zero(self):
        """All signals with zero weight yield 0.0."""
        model = ConfidenceModel()
        signals = [
            _sig("evidence_a", weight=0.0, score=1.0),
            _sig("llm_zero", weight=0.0, score=1.0),
        ]
        result = model.evaluate_composite(signals)
        assert result.score == 0.0

    def test_llm_prefix_detection(self):
        """Only signals starting with 'llm_' go into the LLM category."""
        model = ConfidenceModel()
        signals = [
            _sig("llm_assessment", weight=1.0, score=0.50),
            _sig("llm_backup", weight=1.0, score=0.50),
            _sig("not_llm", weight=1.0, score=1.0),
        ]
        result = model.evaluate_composite(signals)
        # evidence: 1.0, llm: 0.5
        # 0.7*1.0 + 0.3*0.5 = 0.85
        assert result.score == pytest.approx(0.85)

    def test_custom_llm_prefix(self):
        """Model can use a custom prefix to identify LLM signals."""
        model = ConfidenceModel(llm_signal_prefix="ai_")
        signals = [
            _sig("ai_assessment", weight=1.0, score=0.60),
            _sig("file_check", weight=1.0, score=1.0),
        ]
        result = model.evaluate_composite(signals)
        # evidence=1.0, llm=0.6, default weights 0.7/0.3
        assert result.score == pytest.approx(0.7 * 1.0 + 0.3 * 0.6)

    def test_signals_preserved_in_composite_result(self):
        """All signals (both categories) appear in the result."""
        model = ConfidenceModel()
        signals = [
            _sig("evidence_a", weight=1.0, score=0.9),
            _sig("llm_check", weight=1.0, score=0.7),
        ]
        result = model.evaluate_composite(signals)
        assert len(result.signals) == 2
        names = {s.name for s in result.signals}
        assert names == {"evidence_a", "llm_check"}

    def test_composite_respects_custom_thresholds(self):
        """Custom thresholds affect tier classification of composite score."""
        model = ConfidenceModel(
            execute_threshold=0.95,
            warn_threshold=0.70,
        )
        signals = [
            _sig("evidence", weight=1.0, score=0.95),
            _sig("llm_check", weight=1.0, score=0.80),
        ]
        result = model.evaluate_composite(signals)
        # 0.7*0.95 + 0.3*0.80 = 0.665 + 0.24 = 0.905
        assert result.score == pytest.approx(0.905)
        # 0.905 < 0.95 execute threshold → MEDIUM
        assert result.tier == ConfidenceTier.MEDIUM

    def test_evidence_heavy_weight_minimises_llm_influence(self):
        """Evidence-heavy weighting reduces LLM impact on final score."""
        cw = CompositeWeights(evidence=0.9, llm=0.1)
        model = ConfidenceModel(composite_weights=cw)
        signals = [
            _sig("evidence", weight=1.0, score=0.95),
            _sig("llm_bad", weight=1.0, score=0.10),
        ]
        result = model.evaluate_composite(signals)
        expected = 0.9 * 0.95 + 0.1 * 0.10
        assert result.score == pytest.approx(expected)
        assert result.tier == ConfidenceTier.MEDIUM  # 0.865

    def test_llm_heavy_weight_amplifies_llm(self):
        """LLM-heavy weighting amplifies LLM influence."""
        cw = CompositeWeights(evidence=0.1, llm=0.9)
        model = ConfidenceModel(composite_weights=cw)
        signals = [
            _sig("evidence", weight=1.0, score=0.50),
            _sig("llm_confident", weight=1.0, score=0.95),
        ]
        result = model.evaluate_composite(signals)
        expected = 0.1 * 0.50 + 0.9 * 0.95
        assert result.score == pytest.approx(expected)
        assert result.tier == ConfidenceTier.HIGH  # 0.905

    def test_composite_weights_property(self):
        """Model exposes its configured composite weights."""
        cw = CompositeWeights(evidence=0.6, llm=0.4)
        model = ConfidenceModel(composite_weights=cw)
        assert model.composite_weights.evidence == 0.6
        assert model.composite_weights.llm == 0.4
