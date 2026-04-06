"""Step counter for discovery agent investigation budget.

Tracks and enforces a hard cap on investigation steps per discovery session.
Each tool invocation counts as one step. When the budget is exhausted, tools
return an early-exit response instead of performing work, forcing the agent
to wrap up and report its findings.

The default hard cap is 20 steps, which provides enough room for a thorough
scan of typical projects while preventing unbounded exploration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default budget for a single discovery session.
DEFAULT_HARD_CAP: int = 20

# Warning is emitted when this fraction of the budget remains.
_WARNING_FRACTION: float = 0.20  # warn at 80% consumed


@dataclass
class StepCounter:
    """Tracks investigation steps and enforces a hard cap.

    Attributes:
        hard_cap: Maximum steps allowed before forced escalation.
        steps_taken: Number of steps consumed so far.
        step_log: Chronological record of each step for auditability.
    """

    hard_cap: int = DEFAULT_HARD_CAP
    steps_taken: int = 0
    step_log: list[dict[str, Any]] = field(default_factory=list)

    # -- Query helpers -----------------------------------------------------

    @property
    def remaining(self) -> int:
        """Steps remaining before the hard cap is hit."""
        return max(0, self.hard_cap - self.steps_taken)

    @property
    def is_exhausted(self) -> bool:
        """True when the step budget has been fully consumed."""
        return self.steps_taken >= self.hard_cap

    @property
    def is_warning(self) -> bool:
        """True when the remaining budget is at or below the warning threshold."""
        if self.hard_cap <= 0:
            return True
        return self.remaining <= max(1, int(self.hard_cap * _WARNING_FRACTION))

    @property
    def usage_fraction(self) -> float:
        """Fraction of the budget consumed, in [0.0, 1.0]."""
        if self.hard_cap <= 0:
            return 1.0
        return min(self.steps_taken / self.hard_cap, 1.0)

    # -- Mutation ----------------------------------------------------------

    def increment(self, tool_name: str, detail: str = "") -> bool:
        """Record a step and return True if the budget is still available.

        Args:
            tool_name: Name of the tool being invoked.
            detail: Optional human-readable context (e.g. arguments).

        Returns:
            True if the step was allowed (budget not yet exhausted at the
            time of the call). False if the cap has already been reached
            and the step should be rejected.
        """
        if self.is_exhausted:
            logger.warning(
                "Step budget exhausted (%d/%d). Rejecting %s.",
                self.steps_taken,
                self.hard_cap,
                tool_name,
            )
            return False

        self.steps_taken += 1
        entry: dict[str, Any] = {
            "step": self.steps_taken,
            "tool": tool_name,
        }
        if detail:
            entry["detail"] = detail
        self.step_log.append(entry)

        if self.is_warning:
            logger.info(
                "Step budget warning: %d/%d steps used (%d remaining).",
                self.steps_taken,
                self.hard_cap,
                self.remaining,
            )

        return True

    def reset(self) -> None:
        """Reset the counter for a new discovery session."""
        self.steps_taken = 0
        self.step_log.clear()

    # -- Serialization -----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary of step usage."""
        return {
            "steps_taken": self.steps_taken,
            "hard_cap": self.hard_cap,
            "remaining": self.remaining,
            "is_exhausted": self.is_exhausted,
            "usage_fraction": round(self.usage_fraction, 4),
            "step_log": list(self.step_log),
        }

    def budget_status_message(self) -> str:
        """Human-readable budget status for injection into agent context."""
        if self.is_exhausted:
            return (
                f"[BUDGET EXHAUSTED] You have used all {self.hard_cap} "
                f"investigation steps. Stop exploring and report your findings now."
            )
        if self.is_warning:
            return (
                f"[BUDGET WARNING] {self.remaining} of {self.hard_cap} steps "
                f"remaining. Wrap up your investigation soon."
            )
        return (
            f"[BUDGET OK] {self.steps_taken}/{self.hard_cap} steps used, "
            f"{self.remaining} remaining."
        )


# ---------------------------------------------------------------------------
# Budget-exceeded response for tools
# ---------------------------------------------------------------------------

BUDGET_EXCEEDED_RESPONSE: dict[str, Any] = {
    "error": "step_budget_exhausted",
    "message": (
        "Investigation step budget exhausted. You must stop exploring "
        "and report your findings to the orchestrator now. Summarize "
        "what you have discovered so far."
    ),
}
