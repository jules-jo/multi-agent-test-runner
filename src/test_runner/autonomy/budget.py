"""Autonomy budget configuration and enforcement.

Defines per-agent budgets (max iterations, max token spend, max wall-clock
time) and a tracker that monitors consumption against those limits.  The
orchestrator uses the tracker to enforce cutoff logic before and during
each delegation cycle.

Budget limits are *configurable*, not hardcoded:
- Defaults are generous enough for most runs.
- Per-agent overrides are supported via ``AgentBudgetConfig``.
- The ``BudgetTracker`` is the single enforcement point — the orchestrator
  queries it before delegating and after each cycle completes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from test_runner.agents.base import AgentRole


# ---------------------------------------------------------------------------
# Budget limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentBudget:
    """Resource budget for a single agent.

    All limits are *inclusive* — the agent may use up to (and including)
    the limit value.  A value of ``0`` disables that particular limit.

    Attributes:
        max_iterations: Maximum delegation cycles for this agent.
        max_token_spend: Maximum cumulative token count (prompt + completion).
        max_wall_clock_seconds: Maximum cumulative wall-clock seconds.
    """

    max_iterations: int = 0
    max_token_spend: int = 0
    max_wall_clock_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError(f"max_iterations must be >= 0, got {self.max_iterations}")
        if self.max_token_spend < 0:
            raise ValueError(f"max_token_spend must be >= 0, got {self.max_token_spend}")
        if self.max_wall_clock_seconds < 0:
            raise ValueError(
                f"max_wall_clock_seconds must be >= 0, got {self.max_wall_clock_seconds}"
            )

    @property
    def has_iteration_limit(self) -> bool:
        return self.max_iterations > 0

    @property
    def has_token_limit(self) -> bool:
        return self.max_token_spend > 0

    @property
    def has_time_limit(self) -> bool:
        return self.max_wall_clock_seconds > 0.0

    @property
    def is_unlimited(self) -> bool:
        """True if no limits are configured."""
        return not (self.has_iteration_limit or self.has_token_limit or self.has_time_limit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_token_spend": self.max_token_spend,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
        }


class BudgetExceededReason(str, Enum):
    """Why a budget cutoff was triggered."""

    ITERATIONS = "iterations_exceeded"
    TOKEN_SPEND = "token_spend_exceeded"
    WALL_CLOCK = "wall_clock_exceeded"


@dataclass(frozen=True)
class BudgetStatus:
    """Snapshot of an agent's budget consumption.

    Returned by :meth:`BudgetTracker.check` to let the orchestrator
    decide whether to proceed, warn, or cut off.

    Attributes:
        role: The agent role being tracked.
        budget: The configured budget limits.
        iterations_used: Delegation cycles completed so far.
        tokens_used: Cumulative tokens consumed so far.
        wall_clock_used: Cumulative wall-clock seconds consumed.
        exceeded: True if any limit has been breached.
        exceeded_reasons: Which limits were breached (empty if none).
    """

    role: AgentRole
    budget: AgentBudget
    iterations_used: int = 0
    tokens_used: int = 0
    wall_clock_used: float = 0.0
    exceeded: bool = False
    exceeded_reasons: tuple[BudgetExceededReason, ...] = ()

    @property
    def iterations_remaining(self) -> int | None:
        """Remaining iterations, or None if unlimited."""
        if not self.budget.has_iteration_limit:
            return None
        return max(0, self.budget.max_iterations - self.iterations_used)

    @property
    def tokens_remaining(self) -> int | None:
        """Remaining token budget, or None if unlimited."""
        if not self.budget.has_token_limit:
            return None
        return max(0, self.budget.max_token_spend - self.tokens_used)

    @property
    def wall_clock_remaining(self) -> float | None:
        """Remaining wall-clock seconds, or None if unlimited."""
        if not self.budget.has_time_limit:
            return None
        return max(0.0, self.budget.max_wall_clock_seconds - self.wall_clock_used)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role.value,
            "exceeded": self.exceeded,
            "iterations_used": self.iterations_used,
            "tokens_used": self.tokens_used,
            "wall_clock_used": round(self.wall_clock_used, 3),
        }
        if self.budget.has_iteration_limit:
            result["max_iterations"] = self.budget.max_iterations
        if self.budget.has_token_limit:
            result["max_token_spend"] = self.budget.max_token_spend
        if self.budget.has_time_limit:
            result["max_wall_clock_seconds"] = self.budget.max_wall_clock_seconds
        if self.exceeded_reasons:
            result["exceeded_reasons"] = [r.value for r in self.exceeded_reasons]
        return result


# ---------------------------------------------------------------------------
# Budget configuration (per-agent overrides)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentBudgetConfig:
    """Per-agent budget overrides layered on top of global defaults.

    The orchestrator resolves the effective budget for an agent as:
    1. Use the agent-specific override if present.
    2. Otherwise, fall back to the global default.

    Attributes:
        global_default: Default budget applied to all agents.
        per_agent: Agent-specific overrides keyed by AgentRole.
    """

    global_default: AgentBudget = field(default_factory=AgentBudget)
    per_agent: dict[AgentRole, AgentBudget] = field(default_factory=dict)

    def budget_for(self, role: AgentRole) -> AgentBudget:
        """Resolve the effective budget for *role*."""
        return self.per_agent.get(role, self.global_default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_default": self.global_default.to_dict(),
            "per_agent": {
                role.value: budget.to_dict()
                for role, budget in self.per_agent.items()
            },
        }


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


@dataclass
class _AgentUsage:
    """Internal mutable usage tracking for one agent."""

    iterations: int = 0
    tokens: int = 0
    wall_clock_seconds: float = 0.0
    cycle_start_time: float | None = None


class BudgetTracker:
    """Tracks resource consumption per agent and enforces budget limits.

    The orchestrator owns a single ``BudgetTracker`` instance and calls:
    - :meth:`begin_cycle` when starting a delegation to an agent.
    - :meth:`end_cycle` when the delegation completes.
    - :meth:`record_tokens` to report token usage during a cycle.
    - :meth:`check` at any time to query budget status.

    The tracker is passive — it records usage and reports status.  The
    orchestrator is responsible for acting on budget-exceeded signals
    (e.g. cutting off further delegation).

    Thread safety: not thread-safe.  The orchestrator is the sole writer.
    """

    def __init__(self, config: AgentBudgetConfig | None = None) -> None:
        self._config = config or AgentBudgetConfig()
        self._usage: dict[AgentRole, _AgentUsage] = {}

    @property
    def config(self) -> AgentBudgetConfig:
        return self._config

    def _ensure_usage(self, role: AgentRole) -> _AgentUsage:
        if role not in self._usage:
            self._usage[role] = _AgentUsage()
        return self._usage[role]

    # -- Cycle lifecycle -----------------------------------------------------

    def begin_cycle(self, role: AgentRole) -> None:
        """Mark the start of a delegation cycle for *role*.

        Records the start time so that wall-clock consumption can be
        computed when :meth:`end_cycle` is called.
        """
        usage = self._ensure_usage(role)
        usage.cycle_start_time = time.monotonic()

    def end_cycle(self, role: AgentRole, *, tokens: int = 0) -> None:
        """Mark the end of a delegation cycle for *role*.

        Increments the iteration counter and accumulates wall-clock
        time since :meth:`begin_cycle`.  Optionally records token usage.

        Args:
            role: The agent role whose cycle just completed.
            tokens: Token count consumed during this cycle (0 if unknown).
        """
        usage = self._ensure_usage(role)
        usage.iterations += 1
        if tokens > 0:
            usage.tokens += tokens
        if usage.cycle_start_time is not None:
            elapsed = time.monotonic() - usage.cycle_start_time
            usage.wall_clock_seconds += elapsed
            usage.cycle_start_time = None

    def record_tokens(self, role: AgentRole, tokens: int) -> None:
        """Incrementally record token usage for *role*.

        Can be called multiple times during a single cycle (e.g. after
        each LLM call).

        Args:
            role: The agent role.
            tokens: Additional tokens to add.
        """
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        usage = self._ensure_usage(role)
        usage.tokens += tokens

    # -- Querying budget status ----------------------------------------------

    def check(self, role: AgentRole) -> BudgetStatus:
        """Check the budget status for *role*.

        Returns a :class:`BudgetStatus` snapshot indicating whether the
        agent has exceeded any of its configured limits.

        If a cycle is currently in progress (started but not ended), the
        wall-clock time includes elapsed time up to *now*.

        Args:
            role: The agent role to check.

        Returns:
            A BudgetStatus with current consumption and exceeded flags.
        """
        budget = self._config.budget_for(role)
        usage = self._ensure_usage(role)

        # Include in-progress wall-clock time
        wall_clock = usage.wall_clock_seconds
        if usage.cycle_start_time is not None:
            wall_clock += time.monotonic() - usage.cycle_start_time

        # Check each limit
        reasons: list[BudgetExceededReason] = []
        if budget.has_iteration_limit and usage.iterations >= budget.max_iterations:
            reasons.append(BudgetExceededReason.ITERATIONS)
        if budget.has_token_limit and usage.tokens >= budget.max_token_spend:
            reasons.append(BudgetExceededReason.TOKEN_SPEND)
        if budget.has_time_limit and wall_clock >= budget.max_wall_clock_seconds:
            reasons.append(BudgetExceededReason.WALL_CLOCK)

        return BudgetStatus(
            role=role,
            budget=budget,
            iterations_used=usage.iterations,
            tokens_used=usage.tokens,
            wall_clock_used=round(wall_clock, 6),
            exceeded=len(reasons) > 0,
            exceeded_reasons=tuple(reasons),
        )

    def is_exceeded(self, role: AgentRole) -> bool:
        """Quick check: has *role* exceeded any budget limit?"""
        return self.check(role).exceeded

    def all_statuses(self) -> dict[AgentRole, BudgetStatus]:
        """Return budget status for all tracked agents."""
        return {role: self.check(role) for role in self._usage}

    def snapshot(self) -> dict[str, Any]:
        """Serialisable snapshot of all budget tracking state."""
        return {
            "config": self._config.to_dict(),
            "agents": {
                role.value: status.summary()
                for role, status in self.all_statuses().items()
            },
        }

    def reset(self, role: AgentRole | None = None) -> None:
        """Reset usage tracking.

        Args:
            role: If given, reset only this agent's usage.
                  If None, reset all agents.
        """
        if role is not None:
            self._usage.pop(role, None)
        else:
            self._usage.clear()


# ---------------------------------------------------------------------------
# Default budget presets
# ---------------------------------------------------------------------------


def default_budget_config() -> AgentBudgetConfig:
    """Return a sensible default budget configuration for all agent roles.

    These defaults are generous enough for typical test runs while
    providing safety rails against runaway agents.  Per-agent limits
    reflect the expected workload of each role:

    * **Discovery** — moderate iterations, generous tokens (LLM calls).
    * **Executor** — fewer iterations but long wall-clock (test suites).
    * **Reporter** — lightweight, few iterations.
    * **Troubleshooter** — limited iterations to bound diagnostic cost.
    """
    return AgentBudgetConfig(
        global_default=AgentBudget(
            max_iterations=20,
            max_token_spend=500_000,
            max_wall_clock_seconds=600.0,
        ),
        per_agent={
            AgentRole.DISCOVERY: AgentBudget(
                max_iterations=15,
                max_token_spend=300_000,
                max_wall_clock_seconds=300.0,
            ),
            AgentRole.EXECUTOR: AgentBudget(
                max_iterations=10,
                max_token_spend=100_000,
                max_wall_clock_seconds=900.0,
            ),
            AgentRole.REPORTER: AgentBudget(
                max_iterations=5,
                max_token_spend=50_000,
                max_wall_clock_seconds=120.0,
            ),
            AgentRole.TROUBLESHOOTER: AgentBudget(
                max_iterations=8,
                max_token_spend=200_000,
                max_wall_clock_seconds=300.0,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Budget guard — enforcement helper for the orchestrator loop
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised when an agent's budget has been exceeded.

    Contains the :class:`BudgetStatus` snapshot so the orchestrator
    can inspect which limits were breached and decide how to handle it.
    """

    def __init__(self, status: BudgetStatus) -> None:
        self.status = status
        reasons = ", ".join(r.value for r in status.exceeded_reasons)
        super().__init__(
            f"Budget exceeded for {status.role.value}: {reasons} "
            f"(iterations={status.iterations_used}, "
            f"tokens={status.tokens_used}, "
            f"wall_clock={status.wall_clock_used:.1f}s)"
        )


class BudgetGuard:
    """Orchestrator-side guard that checks budgets before and after delegation.

    The guard wraps a :class:`BudgetTracker` and provides a simple
    ``pre_check`` / ``post_check`` protocol that the orchestrator
    calls around each delegation cycle.

    Actions on budget exceeded:

    * ``pre_check`` raises :class:`BudgetExceededError` so the
      orchestrator can skip the delegation entirely and escalate.
    * ``post_check`` returns the :class:`BudgetStatus` with an
      ``exceeded`` flag.  The orchestrator can then decide to halt
      further delegations to that role or log a warning.

    Thread safety: not thread-safe (mirrors BudgetTracker).

    Example::

        guard = BudgetGuard(tracker)
        guard.pre_check(AgentRole.DISCOVERY)   # raises if already over
        tracker.begin_cycle(AgentRole.DISCOVERY)
        # ... run agent ...
        tracker.end_cycle(AgentRole.DISCOVERY, tokens=1234)
        status = guard.post_check(AgentRole.DISCOVERY)
        if status.exceeded:
            # halt or escalate
            ...
    """

    def __init__(self, tracker: BudgetTracker) -> None:
        self._tracker = tracker

    @property
    def tracker(self) -> BudgetTracker:
        return self._tracker

    def pre_check(self, role: AgentRole) -> BudgetStatus:
        """Check budget *before* starting a delegation cycle.

        Raises :class:`BudgetExceededError` if the agent has already
        exhausted any of its budget limits.

        Returns the current status if the budget is within limits.
        """
        status = self._tracker.check(role)
        if status.exceeded:
            raise BudgetExceededError(status)
        return status

    def post_check(self, role: AgentRole) -> BudgetStatus:
        """Check budget *after* a delegation cycle completes.

        Does not raise — returns the status so the orchestrator
        can decide how to handle budget exhaustion (log, halt, escalate).

        Returns:
            The current BudgetStatus for the role.
        """
        return self._tracker.check(role)

    def is_within_budget(self, role: AgentRole) -> bool:
        """Quick check: is the agent still within all budget limits?"""
        return not self._tracker.is_exceeded(role)

    def remaining_summary(self, role: AgentRole) -> dict[str, Any]:
        """Return a summary of remaining budget for *role*.

        Useful for passing to sub-agents so they can self-regulate.
        """
        status = self._tracker.check(role)
        result: dict[str, Any] = {"role": role.value}
        if status.iterations_remaining is not None:
            result["iterations_remaining"] = status.iterations_remaining
        if status.tokens_remaining is not None:
            result["tokens_remaining"] = status.tokens_remaining
        if status.wall_clock_remaining is not None:
            result["wall_clock_remaining"] = round(status.wall_clock_remaining, 3)
        return result
