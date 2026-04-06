"""Tests for ConfidenceSignal, AggregatedConfidence, and tier logic."""

import pytest

from test_runner.models.confidence import (
    AggregatedConfidence,
    ConfidenceSignal,
    ConfidenceTier,
)


class TestConfidenceSignal:
    def test_basic_construction(self):
        sig = ConfidenceSignal(name="test_sig", weight=0.8, score=0.9)
        assert sig.name == "test_sig"
        assert sig.weight == 0.8
        assert sig.score == 0.9
        assert sig.evidence == {}

    def test_weighted_score(self):
        sig = ConfidenceSignal(name="x", weight=0.5, score=0.6)
        assert sig.weighted_score == pytest.approx(0.3)

    def test_evidence_metadata(self):
        sig = ConfidenceSignal(
            name="x", weight=1.0, score=1.0, evidence={"path": "/foo"}
        )
        assert sig.evidence["path"] == "/foo"

    def test_frozen(self):
        sig = ConfidenceSignal(name="x", weight=0.5, score=0.5)
        with pytest.raises(AttributeError):
            sig.name = "y"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "weight,score",
        [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)],
    )
    def test_validation_rejects_out_of_range(self, weight, score):
        with pytest.raises(ValueError):
            ConfidenceSignal(name="bad", weight=weight, score=score)

    def test_boundary_values(self):
        ConfidenceSignal(name="lo", weight=0.0, score=0.0)
        ConfidenceSignal(name="hi", weight=1.0, score=1.0)


class TestAggregatedConfidence:
    def test_empty_returns_zero(self):
        agg = AggregatedConfidence()
        assert agg.score == 0.0
        assert agg.tier == ConfidenceTier.LOW

    def test_single_signal(self):
        agg = AggregatedConfidence()
        agg.add(ConfidenceSignal(name="a", weight=1.0, score=0.9))
        assert agg.score == pytest.approx(0.9)
        assert agg.tier == ConfidenceTier.HIGH

    def test_weighted_average(self):
        agg = AggregatedConfidence()
        agg.add(ConfidenceSignal(name="a", weight=0.8, score=1.0))
        agg.add(ConfidenceSignal(name="b", weight=0.2, score=0.0))
        # (0.8*1.0 + 0.2*0.0) / (0.8+0.2) = 0.8
        assert agg.score == pytest.approx(0.8)
        assert agg.tier == ConfidenceTier.HIGH

    def test_medium_tier(self):
        agg = AggregatedConfidence()
        agg.add(ConfidenceSignal(name="a", weight=1.0, score=0.5))
        assert agg.tier == ConfidenceTier.MEDIUM

    def test_low_tier_and_escalation(self):
        agg = AggregatedConfidence()
        agg.add(ConfidenceSignal(name="a", weight=1.0, score=0.2))
        assert agg.tier == ConfidenceTier.LOW
        assert agg.should_escalate is True

    def test_custom_thresholds(self):
        agg = AggregatedConfidence(high_threshold=0.9, low_threshold=0.5)
        agg.add(ConfidenceSignal(name="a", weight=1.0, score=0.85))
        assert agg.tier == ConfidenceTier.MEDIUM  # below 0.9

    def test_should_not_escalate_medium(self):
        agg = AggregatedConfidence()
        agg.add(ConfidenceSignal(name="a", weight=1.0, score=0.5))
        assert agg.should_escalate is False

    def test_zero_weight_signals(self):
        agg = AggregatedConfidence()
        agg.add(ConfidenceSignal(name="a", weight=0.0, score=1.0))
        assert agg.score == 0.0

    def test_summary(self):
        agg = AggregatedConfidence()
        agg.add(ConfidenceSignal(name="a", weight=1.0, score=0.9))
        s = agg.summary()
        assert s["tier"] == "high"
        assert s["signal_count"] == 1
        assert s["should_escalate"] is False
        assert len(s["signals"]) == 1
