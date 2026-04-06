"""Confidence signal model for evidence-based discovery decisions.

The confidence system drives autonomous exploration: each signal collector
produces weighted evidence that the orchestrator aggregates into a tier
(HIGH / MEDIUM / LOW). Tier thresholds determine whether the discovery
agent proceeds autonomously, asks for confirmation, or escalates.

Two aggregation APIs are provided:

* **AggregatedConfidence** — simple mutable container (legacy, still used
  by existing tests).
* **ConfidenceModel / ConfidenceResult** — richer, immutable result with
  the canonical tier thresholds required by the orchestrator:
  - *execute* (>=90%): proceed autonomously
  - *warn*    (60–89%): ask for confirmation
  - *investigate* (<60%): escalate / gather more evidence
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Sequence


class ConfidenceTier(enum.Enum):
    """Autonomy tiers derived from aggregated confidence scores.

    HIGH   - agent proceeds autonomously (score >= high_threshold)
    MEDIUM - agent asks for confirmation (low_threshold <= score < high_threshold)
    LOW    - agent escalates to orchestrator (score < low_threshold)
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Default tier boundaries - configurable via autonomy policy
DEFAULT_HIGH_THRESHOLD = 0.80
DEFAULT_LOW_THRESHOLD = 0.40

# Canonical thresholds for ConfidenceModel (per spec)
EXECUTE_THRESHOLD = 0.90
WARN_THRESHOLD = 0.60


class ConfidenceDecision(str, enum.Enum):
    """Three-tier decision outcomes driven by confidence score thresholds.

    Maps directly to the canonical decision tiers:

    * **EXECUTE** — confidence >= 90% (configurable). The agent has
      gathered sufficient evidence and should proceed to test execution
      immediately without user intervention.

    * **WARN** — 60% <= confidence < 90% (configurable). Evidence is
      moderate. The agent proceeds to execution but surfaces a warning
      so the user is aware that results may be incomplete or imprecise.

    * **INVESTIGATE** — confidence < 60% (configurable). Insufficient
      evidence. The agent should keep exploring or, if the exploration
      budget is exhausted, escalate for guidance.

    This enum is the primary interface for threshold-based routing in
    the orchestrator hub. It is produced by :meth:`ConfidenceModel.decide`
    and consumed by the orchestrator's state machine.
    """

    EXECUTE = "execute"
    """Confidence >= execute_threshold. Proceed autonomously."""

    WARN = "warn"
    """warn_threshold <= confidence < execute_threshold. Proceed with caution."""

    INVESTIGATE = "investigate"
    """Confidence < warn_threshold. Gather more evidence or escalate."""

    @property
    def can_execute(self) -> bool:
        """True when the decision allows execution (EXECUTE or WARN)."""
        return self in (ConfidenceDecision.EXECUTE, ConfidenceDecision.WARN)

    @property
    def needs_warning(self) -> bool:
        """True when the decision is WARN tier."""
        return self == ConfidenceDecision.WARN

    @property
    def needs_investigation(self) -> bool:
        """True when more evidence is needed."""
        return self == ConfidenceDecision.INVESTIGATE

    @classmethod
    def from_tier(cls, tier: ConfidenceTier) -> ConfidenceDecision:
        """Convert a :class:`ConfidenceTier` to a :class:`ConfidenceDecision`.

        Mapping:
            HIGH   -> EXECUTE
            MEDIUM -> WARN
            LOW    -> INVESTIGATE
        """
        _tier_to_decision = {
            ConfidenceTier.HIGH: cls.EXECUTE,
            ConfidenceTier.MEDIUM: cls.WARN,
            ConfidenceTier.LOW: cls.INVESTIGATE,
        }
        return _tier_to_decision[tier]

    @classmethod
    def from_score(
        cls,
        score: float,
        execute_threshold: float = EXECUTE_THRESHOLD,
        warn_threshold: float = WARN_THRESHOLD,
    ) -> ConfidenceDecision:
        """Classify a raw score into a decision using threshold boundaries.

        Args:
            score: Confidence score in [0.0, 1.0].
            execute_threshold: Score at or above which -> EXECUTE (default 0.90).
            warn_threshold: Score at or above which -> WARN (default 0.60).

        Returns:
            The appropriate ConfidenceDecision.

        Raises:
            ValueError: If thresholds or score are out of range.
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {score}")
        if not (0.0 <= warn_threshold <= execute_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 <= warn ({warn_threshold}) "
                f"<= execute ({execute_threshold}) <= 1"
            )
        if score >= execute_threshold:
            return cls.EXECUTE
        if score >= warn_threshold:
            return cls.WARN
        return cls.INVESTIGATE


@dataclass(frozen=True)
class ConfidenceSignal:
    """A single piece of evidence produced by a signal collector.

    Attributes:
        name: Human-readable identifier for the signal (e.g. "pytest_ini_exists").
        weight: Relative importance of this signal in [0.0, 1.0].
                Higher weight means this signal contributes more to the
                aggregated score.
        score: Measured confidence in [0.0, 1.0].
               1.0 = certainty, 0.0 = no evidence found.
        evidence: Optional free-form metadata explaining how the score
                  was derived (file paths found, patterns matched, etc.).
    """

    name: str
    weight: float
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    @property
    def weighted_score(self) -> float:
        """Convenience: weight * score."""
        return self.weight * self.score


@dataclass
class AggregatedConfidence:
    """Combines multiple ConfidenceSignals into a single tier decision.

    The aggregated score is the weighted average of all signals:
        sum(weight_i * score_i) / sum(weight_i)

    Thresholds are configurable to support the autonomy policy architecture.
    """

    signals: list[ConfidenceSignal] = field(default_factory=list)
    high_threshold: float = DEFAULT_HIGH_THRESHOLD
    low_threshold: float = DEFAULT_LOW_THRESHOLD

    def add(self, signal: ConfidenceSignal) -> None:
        """Append a signal to the collection."""
        self.signals.append(signal)

    @property
    def score(self) -> float:
        """Weighted average across all signals. Returns 0.0 if empty."""
        if not self.signals:
            return 0.0
        total_weight = sum(s.weight for s in self.signals)
        if total_weight == 0.0:
            return 0.0
        return sum(s.weighted_score for s in self.signals) / total_weight

    @property
    def tier(self) -> ConfidenceTier:
        """Map the aggregated score to an autonomy tier."""
        s = self.score
        if s >= self.high_threshold:
            return ConfidenceTier.HIGH
        if s >= self.low_threshold:
            return ConfidenceTier.MEDIUM
        return ConfidenceTier.LOW

    @property
    def should_escalate(self) -> bool:
        """True when the score falls below the low threshold (hard cap)."""
        return self.tier == ConfidenceTier.LOW

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary of the aggregated confidence."""
        return {
            "score": round(self.score, 4),
            "tier": self.tier.value,
            "should_escalate": self.should_escalate,
            "signal_count": len(self.signals),
            "signals": [
                {"name": s.name, "weight": s.weight, "score": s.score}
                for s in self.signals
            ],
        }


# ---------------------------------------------------------------------------
# ConfidenceResult / ConfidenceModel — canonical orchestrator API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceResult:
    """Immutable outcome of :meth:`ConfidenceModel.evaluate`.

    Attributes:
        score: Final aggregated confidence in [0.0, 1.0].
        tier: One of ``ConfidenceTier.HIGH`` (execute),
              ``ConfidenceTier.MEDIUM`` (warn),
              ``ConfidenceTier.LOW`` (investigate).
        signals: The individual signals that were aggregated.
        execute_threshold: The threshold at or above which the tier is HIGH.
        warn_threshold: The threshold at or above which the tier is MEDIUM.
    """

    score: float
    tier: ConfidenceTier
    signals: tuple[ConfidenceSignal, ...]
    execute_threshold: float
    warn_threshold: float

    # Convenience helpers ------------------------------------------------

    @property
    def decision(self) -> ConfidenceDecision:
        """The three-tier decision derived from the confidence tier.

        Maps HIGH -> EXECUTE, MEDIUM -> WARN, LOW -> INVESTIGATE.
        """
        return ConfidenceDecision.from_tier(self.tier)

    @property
    def should_execute(self) -> bool:
        """True when the agent should proceed autonomously."""
        return self.tier == ConfidenceTier.HIGH

    @property
    def should_warn(self) -> bool:
        """True when the agent should ask for user confirmation."""
        return self.tier == ConfidenceTier.MEDIUM

    @property
    def should_investigate(self) -> bool:
        """True when the agent should escalate / gather more evidence."""
        return self.tier == ConfidenceTier.LOW

    def summary(self) -> dict[str, Any]:
        """Serializable representation for logging / reporting."""
        return {
            "score": round(self.score, 4),
            "tier": self.tier.value,
            "decision": self.decision.value,
            "action": (
                "execute"
                if self.should_execute
                else "warn"
                if self.should_warn
                else "investigate"
            ),
            "execute_threshold": self.execute_threshold,
            "warn_threshold": self.warn_threshold,
            "signal_count": len(self.signals),
            "signals": [
                {
                    "name": s.name,
                    "weight": s.weight,
                    "score": s.score,
                    "weighted_score": round(s.weighted_score, 4),
                }
                for s in self.signals
            ],
        }


@dataclass(frozen=True)
class CompositeWeights:
    """Configurable blend weights for the two signal categories.

    The composite score is computed as::

        composite = evidence_weight * evidence_avg + llm_weight * llm_avg

    where the weights are normalised so they sum to 1.0.

    Attributes:
        evidence: Relative weight for evidence-based signals (file checks,
                  pattern matches, framework detection).
        llm: Relative weight for LLM self-assessment signals.
    """

    evidence: float = 0.7
    llm: float = 0.3

    def __post_init__(self) -> None:
        if self.evidence < 0 or self.llm < 0:
            raise ValueError(
                f"Category weights must be non-negative, got "
                f"evidence={self.evidence}, llm={self.llm}"
            )
        if self.evidence == 0 and self.llm == 0:
            raise ValueError("At least one category weight must be > 0")

    @property
    def normalized_evidence(self) -> float:
        """Evidence weight normalised to [0, 1]."""
        total = self.evidence + self.llm
        return self.evidence / total

    @property
    def normalized_llm(self) -> float:
        """LLM weight normalised to [0, 1]."""
        total = self.evidence + self.llm
        return self.llm / total


# Default composite weights — evidence-heavy, LLM supplements.
DEFAULT_COMPOSITE_WEIGHTS = CompositeWeights(evidence=0.7, llm=0.3)

# Signal name prefix used to identify LLM self-assessment signals.
LLM_SIGNAL_PREFIX = "llm_"


class ConfidenceModel:
    """Aggregates weighted signals into a :class:`ConfidenceResult`.

    The model supports two evaluation modes:

    1. **Flat evaluation** (``evaluate``) — computes a single weighted
       average across all supplied signals::

           final_score = sum(w_i * s_i) / sum(w_i)

    2. **Composite evaluation** (``evaluate_composite``) — partitions
       signals into *evidence-based* and *LLM self-assessment* categories,
       computes the weighted average within each category, and blends
       them using configurable :class:`CompositeWeights`::

           composite = W_evidence * avg(evidence_signals)
                     + W_llm     * avg(llm_signals)

       This allows tuning how much the LLM's own opinion influences the
       final decision independently from the evidence signal weights.

    Tier classification uses configurable thresholds:

    * **execute** — ``score >= execute_threshold`` (default 0.90)
    * **warn**    — ``warn_threshold <= score < execute_threshold`` (default 0.60)
    * **investigate** — ``score < warn_threshold``

    Thresholds are configurable to support the autonomy-policy architecture.
    Changing them allows an admin to tighten or loosen the agent's freedom
    without touching any signal-collection logic.
    """

    def __init__(
        self,
        execute_threshold: float = EXECUTE_THRESHOLD,
        warn_threshold: float = WARN_THRESHOLD,
        composite_weights: CompositeWeights | None = None,
        llm_signal_prefix: str = LLM_SIGNAL_PREFIX,
    ) -> None:
        if not (0.0 <= warn_threshold <= execute_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 <= warn ({warn_threshold}) "
                f"<= execute ({execute_threshold}) <= 1"
            )
        self._execute_threshold = execute_threshold
        self._warn_threshold = warn_threshold
        self._composite_weights = composite_weights or DEFAULT_COMPOSITE_WEIGHTS
        self._llm_signal_prefix = llm_signal_prefix

    # -- Public properties -------------------------------------------------

    @property
    def execute_threshold(self) -> float:
        return self._execute_threshold

    @property
    def warn_threshold(self) -> float:
        return self._warn_threshold

    @property
    def composite_weights(self) -> CompositeWeights:
        """Current category blend weights."""
        return self._composite_weights

    # -- Decision API (three-tier routing) ---------------------------------

    def decide(self, signals: Sequence[ConfidenceSignal]) -> ConfidenceDecision:
        """Evaluate signals and return a three-tier decision.

        This is the primary decision interface for the orchestrator hub:

        * **EXECUTE** — score >= execute_threshold (default 0.90)
        * **WARN** — warn_threshold <= score < execute_threshold (default 0.60–0.90)
        * **INVESTIGATE** — score < warn_threshold (default < 0.60)

        For the full evaluation result including signals, use :meth:`evaluate`.

        Args:
            signals: Confidence signals to aggregate.

        Returns:
            A :class:`ConfidenceDecision` enum value.
        """
        result = self.evaluate(signals)
        return result.decision

    def decide_from_score(self, score: float) -> ConfidenceDecision:
        """Classify a pre-computed score into a three-tier decision.

        Args:
            score: Aggregated confidence score in [0.0, 1.0].

        Returns:
            A :class:`ConfidenceDecision` enum value.
        """
        return ConfidenceDecision.from_score(
            score,
            execute_threshold=self._execute_threshold,
            warn_threshold=self._warn_threshold,
        )

    # -- Core evaluation ---------------------------------------------------

    def evaluate(self, signals: Sequence[ConfidenceSignal]) -> ConfidenceResult:
        """Aggregate *signals* into a single :class:`ConfidenceResult`.

        Uses a flat weighted average across **all** signals regardless of
        category. Zero-weight signals are included in the result but do
        not affect the score. An empty signal list yields score 0.0
        (investigate).
        """
        score = self._weighted_average(signals)
        tier = self._classify(score)
        return ConfidenceResult(
            score=score,
            tier=tier,
            signals=tuple(signals),
            execute_threshold=self._execute_threshold,
            warn_threshold=self._warn_threshold,
        )

    def evaluate_composite(
        self,
        signals: Sequence[ConfidenceSignal],
        *,
        weights: CompositeWeights | None = None,
    ) -> ConfidenceResult:
        """Combine evidence-based and LLM signals with category weighting.

        Signals whose ``name`` starts with :attr:`llm_signal_prefix` are
        classified as *LLM self-assessment*; everything else is
        *evidence-based*. Each category's weighted average is computed
        independently, then blended using the provided (or model-default)
        :class:`CompositeWeights`.

        If one category has no signals, the other category receives 100%
        of the blend weight — this graceful degradation means the model
        works whether or not an LLM assessment was performed.

        Args:
            signals: All signals (both evidence and LLM).
            weights: Override blend weights for this call only.
                     Falls back to the model's configured weights.

        Returns:
            A :class:`ConfidenceResult` with the composite score.
        """
        cw = weights or self._composite_weights

        evidence_signals: list[ConfidenceSignal] = []
        llm_signals: list[ConfidenceSignal] = []
        for s in signals:
            if s.name.startswith(self._llm_signal_prefix):
                llm_signals.append(s)
            else:
                evidence_signals.append(s)

        evidence_avg = self._weighted_average(evidence_signals)
        llm_avg = self._weighted_average(llm_signals)

        # Graceful degradation: if one category is empty, give all
        # weight to the other.
        has_evidence = bool(evidence_signals) and any(
            s.weight > 0 for s in evidence_signals
        )
        has_llm = bool(llm_signals) and any(s.weight > 0 for s in llm_signals)

        if has_evidence and has_llm:
            score = (
                cw.normalized_evidence * evidence_avg
                + cw.normalized_llm * llm_avg
            )
        elif has_evidence:
            score = evidence_avg
        elif has_llm:
            score = llm_avg
        else:
            score = 0.0

        tier = self._classify(score)
        return ConfidenceResult(
            score=score,
            tier=tier,
            signals=tuple(signals),
            execute_threshold=self._execute_threshold,
            warn_threshold=self._warn_threshold,
        )

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _weighted_average(signals: Sequence[ConfidenceSignal]) -> float:
        """Compute weighted average; returns 0.0 for empty / zero-weight."""
        if not signals:
            return 0.0
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0.0:
            return 0.0
        return sum(s.weighted_score for s in signals) / total_weight

    def _classify(self, score: float) -> ConfidenceTier:
        """Map a score to the appropriate tier."""
        if score >= self._execute_threshold:
            return ConfidenceTier.HIGH
        if score >= self._warn_threshold:
            return ConfidenceTier.MEDIUM
        return ConfidenceTier.LOW
