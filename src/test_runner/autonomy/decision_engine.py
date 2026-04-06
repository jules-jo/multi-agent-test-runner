"""Three-tier decision engine with threshold-based routing.

Implements the core decision logic for the autonomy system:

* **Tier 1 — Execute Immediately** (confidence >= 90%):
  The discovery agent has gathered sufficient evidence. Proceed to
  test execution without user intervention.

* **Tier 2 — Execute with Warning** (60% <= confidence < 90%):
  Evidence is moderate. Proceed to execution but surface a warning
  so the user is aware that results may be incomplete or imprecise.

* **Tier 3 — Continue Investigating** (confidence < 60%):
  Insufficient evidence. The discovery agent should keep exploring
  or, if the exploration budget is exhausted, escalate to the
  orchestrator/user for guidance.

The engine is a pure function of (score, policy, context) -> decision,
with no side effects. All state management is handled by the
orchestrator hub.

Thresholds are fully configurable via :class:`AutonomyPolicyConfig`
to support the pluggable autonomy-policy architecture.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from test_runner.autonomy.policy import AutonomyPolicyConfig
from test_runner.models.confidence import (
    ConfidenceDecision,
    ConfidenceModel,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceTier,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


class DecisionVerdict(str, enum.Enum):
    """The three possible outcomes of a threshold-based decision.

    These map directly to the three tiers described in the module
    docstring and correspond to autonomy-engine actions.
    """

    EXECUTE_IMMEDIATELY = "execute_immediately"
    """Confidence >= execute_threshold (default 90%). No user confirmation."""

    EXECUTE_WITH_WARNING = "execute_with_warning"
    """warn_threshold <= confidence < execute_threshold (default 60-90%).
    Proceed but alert the user about reduced confidence."""

    CONTINUE_INVESTIGATING = "continue_investigating"
    """Confidence < warn_threshold (default 60%). Need more evidence."""

    @classmethod
    def from_confidence_decision(cls, cd: ConfidenceDecision) -> DecisionVerdict:
        """Convert a :class:`ConfidenceDecision` to a :class:`DecisionVerdict`.

        Mapping:
            EXECUTE     -> EXECUTE_IMMEDIATELY
            WARN        -> EXECUTE_WITH_WARNING
            INVESTIGATE -> CONTINUE_INVESTIGATING
        """
        _mapping = {
            ConfidenceDecision.EXECUTE: cls.EXECUTE_IMMEDIATELY,
            ConfidenceDecision.WARN: cls.EXECUTE_WITH_WARNING,
            ConfidenceDecision.INVESTIGATE: cls.CONTINUE_INVESTIGATING,
        }
        return _mapping[cd]

    def to_confidence_decision(self) -> ConfidenceDecision:
        """Convert this verdict back to a :class:`ConfidenceDecision`.

        Mapping:
            EXECUTE_IMMEDIATELY    -> EXECUTE
            EXECUTE_WITH_WARNING   -> WARN
            CONTINUE_INVESTIGATING -> INVESTIGATE
        """
        _mapping = {
            DecisionVerdict.EXECUTE_IMMEDIATELY: ConfidenceDecision.EXECUTE,
            DecisionVerdict.EXECUTE_WITH_WARNING: ConfidenceDecision.WARN,
            DecisionVerdict.CONTINUE_INVESTIGATING: ConfidenceDecision.INVESTIGATE,
        }
        return _mapping[self]


@dataclass(frozen=True)
class DecisionContext:
    """Contextual information fed into the decision engine.

    Captures the current state of exploration so the engine can
    make informed routing choices beyond raw confidence score.

    Attributes:
        exploration_round: Current exploration round number (1-based).
        max_exploration_rounds: Hard cap on rounds from the policy.
        positive_signal_count: Number of signals with score > 0.
        min_positive_signals: Policy minimum for proceeding.
        has_framework_detected: Whether a test framework was found.
        has_scripts: Whether executable scripts were found.
        require_framework: Whether policy requires framework detection.
        allow_script_fallback: Whether scripts can substitute for frameworks.
    """

    exploration_round: int = 1
    max_exploration_rounds: int = 5
    positive_signal_count: int = 0
    min_positive_signals: int = 2
    has_framework_detected: bool = False
    has_scripts: bool = False
    require_framework: bool = False
    allow_script_fallback: bool = True

    @property
    def is_at_budget_limit(self) -> bool:
        """True when exploration has reached the maximum allowed rounds."""
        return self.exploration_round >= self.max_exploration_rounds

    @property
    def has_exceeded_budget(self) -> bool:
        """True when exploration has exceeded the maximum allowed rounds."""
        return self.exploration_round > self.max_exploration_rounds

    @property
    def has_enough_signals(self) -> bool:
        """True when minimum positive signal count is met."""
        return self.positive_signal_count >= self.min_positive_signals

    @property
    def framework_requirement_met(self) -> bool:
        """True when framework requirement is satisfied or not required."""
        if not self.require_framework:
            return True
        if self.has_framework_detected:
            return True
        # At budget limit, scripts can substitute if allowed
        if self.is_at_budget_limit and self.allow_script_fallback and self.has_scripts:
            return True
        return False


@dataclass(frozen=True)
class DecisionResult:
    """The output of the three-tier decision engine.

    Attributes:
        verdict: Which tier the decision falls into.
        confidence_result: The full confidence evaluation that drove the decision.
        context: The exploration context used for the decision.
        reason: Human-readable explanation of why this verdict was chosen.
        should_escalate: True if exploration is exhausted and confidence
            remains in the investigate tier.
        escalation_reason: Why escalation is needed (empty if not escalating).
        metadata: Additional engine metadata for logging/debugging.
    """

    verdict: DecisionVerdict
    confidence_result: ConfidenceResult
    context: DecisionContext
    reason: str
    should_escalate: bool = False
    escalation_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Convenience properties --------------------------------------------

    @property
    def decision(self) -> ConfidenceDecision:
        """The three-tier :class:`ConfidenceDecision` for this result.

        Maps the verdict to the canonical EXECUTE / WARN / INVESTIGATE enum.
        """
        return self.verdict.to_confidence_decision()

    @property
    def can_execute(self) -> bool:
        """True when the verdict allows execution (tiers 1 or 2)."""
        return self.verdict in (
            DecisionVerdict.EXECUTE_IMMEDIATELY,
            DecisionVerdict.EXECUTE_WITH_WARNING,
        )

    @property
    def needs_warning(self) -> bool:
        """True when the verdict is tier 2 (execute with warning)."""
        return self.verdict == DecisionVerdict.EXECUTE_WITH_WARNING

    @property
    def needs_investigation(self) -> bool:
        """True when more evidence is needed (tier 3)."""
        return self.verdict == DecisionVerdict.CONTINUE_INVESTIGATING

    @property
    def confidence_score(self) -> float:
        """Shortcut to the underlying confidence score."""
        return self.confidence_result.score

    @property
    def confidence_tier(self) -> ConfidenceTier:
        """Shortcut to the underlying confidence tier."""
        return self.confidence_result.tier

    @property
    def score_percentage(self) -> float:
        """Confidence score as a percentage (0-100)."""
        return round(self.confidence_result.score * 100, 1)

    def summary(self) -> dict[str, Any]:
        """Serializable representation for logging / reporting."""
        result: dict[str, Any] = {
            "verdict": self.verdict.value,
            "confidence_score": round(self.confidence_result.score, 4),
            "confidence_tier": self.confidence_result.tier.value,
            "score_percentage": self.score_percentage,
            "can_execute": self.can_execute,
            "needs_warning": self.needs_warning,
            "needs_investigation": self.needs_investigation,
            "should_escalate": self.should_escalate,
            "reason": self.reason,
            "exploration_round": self.context.exploration_round,
        }
        if self.should_escalate:
            result["escalation_reason"] = self.escalation_reason
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------


class DecisionEngine:
    """Three-tier threshold-based routing engine.

    The engine evaluates a set of confidence signals against configurable
    thresholds and produces a :class:`DecisionResult` with one of three
    verdicts:

    1. **EXECUTE_IMMEDIATELY** — score >= execute_threshold (default 0.90)
    2. **EXECUTE_WITH_WARNING** — warn_threshold <= score < execute_threshold
    3. **CONTINUE_INVESTIGATING** — score < warn_threshold (default 0.60)

    The engine also considers exploration context (round budget, signal
    counts, framework requirements) to handle edge cases:

    * If the exploration budget is exhausted and confidence is in the
      "warn" zone, it upgrades to EXECUTE_WITH_WARNING rather than
      continuing to investigate.
    * If the budget is exhausted and confidence is in the "investigate"
      zone, it marks the decision as needing escalation.
    * If minimum signal requirements aren't met, it forces
      CONTINUE_INVESTIGATING regardless of score.

    The engine is stateless and deterministic — identical inputs always
    produce identical outputs.

    Example::

        engine = DecisionEngine()
        result = engine.decide(signals, context)

        match result.verdict:
            case DecisionVerdict.EXECUTE_IMMEDIATELY:
                # Hand off to executor
                ...
            case DecisionVerdict.EXECUTE_WITH_WARNING:
                # Execute but notify user
                ...
            case DecisionVerdict.CONTINUE_INVESTIGATING:
                if result.should_escalate:
                    # Budget exhausted, ask user
                    ...
                else:
                    # Run another discovery round
                    ...
    """

    def __init__(
        self,
        policy: AutonomyPolicyConfig | None = None,
    ) -> None:
        self._policy = policy or AutonomyPolicyConfig()
        self._confidence_model = ConfidenceModel(
            execute_threshold=self._policy.execute_threshold,
            warn_threshold=self._policy.warn_threshold,
        )

    # -- Public properties -------------------------------------------------

    @property
    def policy(self) -> AutonomyPolicyConfig:
        """The active autonomy policy."""
        return self._policy

    @property
    def execute_threshold(self) -> float:
        """Score at or above which execution proceeds immediately."""
        return self._policy.execute_threshold

    @property
    def warn_threshold(self) -> float:
        """Score at or above which execution proceeds with a warning."""
        return self._policy.warn_threshold

    @property
    def confidence_model(self) -> ConfidenceModel:
        """The underlying confidence model used for scoring."""
        return self._confidence_model

    # -- Core decision API -------------------------------------------------

    def decide(
        self,
        signals: Sequence[ConfidenceSignal],
        context: DecisionContext | None = None,
    ) -> DecisionResult:
        """Evaluate signals and produce a three-tier routing decision.

        This is the main entry point. It:
        1. Aggregates signals into a confidence score via ConfidenceModel
        2. Checks preconditions (budget, minimum signals, framework req)
        3. Routes to the appropriate tier based on thresholds

        Args:
            signals: Confidence signals from discovery collectors.
            context: Exploration context for budget/requirement checks.
                Uses defaults if not provided.

        Returns:
            A DecisionResult with the verdict and supporting details.
        """
        ctx = context or DecisionContext()
        confidence_result = self._confidence_model.evaluate(signals)
        score = confidence_result.score

        logger.info(
            "DecisionEngine: score=%.4f (%.1f%%) tier=%s round=%d/%d",
            score,
            score * 100,
            confidence_result.tier.value,
            ctx.exploration_round,
            ctx.max_exploration_rounds,
        )

        # -- Pre-routing checks --------------------------------------------

        # Hard budget exceeded: always escalate
        if ctx.has_exceeded_budget:
            return self._make_result(
                DecisionVerdict.CONTINUE_INVESTIGATING,
                confidence_result,
                ctx,
                reason=(
                    f"Hard cap exceeded: round {ctx.exploration_round} "
                    f"> max {ctx.max_exploration_rounds}"
                ),
                should_escalate=True,
                escalation_reason="exploration_budget_exceeded",
            )

        # Insufficient positive signals
        if not ctx.has_enough_signals:
            if ctx.is_at_budget_limit:
                return self._make_result(
                    DecisionVerdict.CONTINUE_INVESTIGATING,
                    confidence_result,
                    ctx,
                    reason=(
                        f"Only {ctx.positive_signal_count} positive signals "
                        f"(need {ctx.min_positive_signals}); budget exhausted"
                    ),
                    should_escalate=True,
                    escalation_reason="insufficient_signals_at_budget_limit",
                )
            return self._make_result(
                DecisionVerdict.CONTINUE_INVESTIGATING,
                confidence_result,
                ctx,
                reason=(
                    f"Only {ctx.positive_signal_count} positive signals "
                    f"(need {ctx.min_positive_signals}); continuing investigation"
                ),
            )

        # Framework requirement not met
        if not ctx.framework_requirement_met:
            if ctx.is_at_budget_limit:
                return self._make_result(
                    DecisionVerdict.CONTINUE_INVESTIGATING,
                    confidence_result,
                    ctx,
                    reason="No framework detected and policy requires it; budget exhausted",
                    should_escalate=True,
                    escalation_reason="framework_requirement_unmet",
                )
            return self._make_result(
                DecisionVerdict.CONTINUE_INVESTIGATING,
                confidence_result,
                ctx,
                reason="No framework detected; policy requires framework detection",
            )

        # -- Three-tier threshold routing ----------------------------------

        if score >= self._policy.execute_threshold:
            # TIER 1: High confidence — execute immediately
            return self._make_result(
                DecisionVerdict.EXECUTE_IMMEDIATELY,
                confidence_result,
                ctx,
                reason=(
                    f"Confidence {score:.2f} ({score*100:.1f}%) >= "
                    f"execute threshold {self._policy.execute_threshold:.2f} "
                    f"({self._policy.execute_threshold*100:.0f}%)"
                ),
            )

        if score >= self._policy.warn_threshold:
            # TIER 2: Medium confidence — execute with warning
            # At budget limit: proceed with warning (no more rounds available)
            if ctx.is_at_budget_limit:
                return self._make_result(
                    DecisionVerdict.EXECUTE_WITH_WARNING,
                    confidence_result,
                    ctx,
                    reason=(
                        f"Confidence {score:.2f} ({score*100:.1f}%) in warn zone "
                        f"[{self._policy.warn_threshold:.2f}, "
                        f"{self._policy.execute_threshold:.2f}); "
                        f"budget exhausted — proceeding with warning"
                    ),
                )

            # Check if score is close to execute threshold — might benefit
            # from one more investigation round
            gap = self._policy.execute_threshold - score
            if gap <= 0.10 and not ctx.is_at_budget_limit:
                return self._make_result(
                    DecisionVerdict.CONTINUE_INVESTIGATING,
                    confidence_result,
                    ctx,
                    reason=(
                        f"Confidence {score:.2f} ({score*100:.1f}%) is within "
                        f"10% of execute threshold {self._policy.execute_threshold:.2f}; "
                        f"one more round may reach it"
                    ),
                    metadata={"near_threshold": True, "gap": round(gap, 4)},
                )

            # Otherwise, proceed with warning
            return self._make_result(
                DecisionVerdict.EXECUTE_WITH_WARNING,
                confidence_result,
                ctx,
                reason=(
                    f"Confidence {score:.2f} ({score*100:.1f}%) in warn zone "
                    f"[{self._policy.warn_threshold:.2f}, "
                    f"{self._policy.execute_threshold:.2f})"
                ),
            )

        # TIER 3: Low confidence — continue investigating
        if ctx.is_at_budget_limit:
            return self._make_result(
                DecisionVerdict.CONTINUE_INVESTIGATING,
                confidence_result,
                ctx,
                reason=(
                    f"Confidence {score:.2f} ({score*100:.1f}%) < "
                    f"warn threshold {self._policy.warn_threshold:.2f} "
                    f"({self._policy.warn_threshold*100:.0f}%); "
                    f"budget exhausted"
                ),
                should_escalate=True,
                escalation_reason="low_confidence_at_budget_limit",
            )

        return self._make_result(
            DecisionVerdict.CONTINUE_INVESTIGATING,
            confidence_result,
            ctx,
            reason=(
                f"Confidence {score:.2f} ({score*100:.1f}%) < "
                f"warn threshold {self._policy.warn_threshold:.2f} "
                f"({self._policy.warn_threshold*100:.0f}%); "
                f"continuing investigation"
            ),
        )

    # -- Convenience API ---------------------------------------------------

    def decide_from_score(
        self,
        score: float,
        context: DecisionContext | None = None,
    ) -> DecisionResult:
        """Produce a decision from a pre-computed score.

        Useful when the caller has already aggregated signals and just
        needs the threshold routing. Creates a synthetic ConfidenceResult
        from the score.

        Args:
            score: Pre-computed confidence score in [0.0, 1.0].
            context: Exploration context. Uses defaults if not provided.

        Returns:
            A DecisionResult with the verdict.
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Score must be in [0, 1], got {score}")

        tier = self._confidence_model._classify(score)
        result = ConfidenceResult(
            score=score,
            tier=tier,
            signals=(),
            execute_threshold=self._policy.execute_threshold,
            warn_threshold=self._policy.warn_threshold,
        )
        ctx = context or DecisionContext()
        # Re-use decide logic by creating a synthetic signal
        synthetic = ConfidenceSignal(name="pre_computed", weight=1.0, score=score)
        return self.decide([synthetic], ctx)

    def classify_score(self, score: float) -> DecisionVerdict:
        """Pure threshold classification — no context checks.

        Returns the verdict based solely on the score vs thresholds,
        ignoring budget, signal count, and framework requirements.
        This is useful for quick categorization.

        Args:
            score: Confidence score in [0.0, 1.0].

        Returns:
            The DecisionVerdict for the score.
        """
        if score >= self._policy.execute_threshold:
            return DecisionVerdict.EXECUTE_IMMEDIATELY
        if score >= self._policy.warn_threshold:
            return DecisionVerdict.EXECUTE_WITH_WARNING
        return DecisionVerdict.CONTINUE_INVESTIGATING

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _make_result(
        verdict: DecisionVerdict,
        confidence_result: ConfidenceResult,
        context: DecisionContext,
        reason: str,
        should_escalate: bool = False,
        escalation_reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DecisionResult:
        """Construct a DecisionResult with logging."""
        logger.info(
            "Decision: %s (escalate=%s) — %s",
            verdict.value,
            should_escalate,
            reason,
        )
        return DecisionResult(
            verdict=verdict,
            confidence_result=confidence_result,
            context=context,
            reason=reason,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason,
            metadata=metadata or {},
        )
