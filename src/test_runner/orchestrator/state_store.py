"""Shared state store for the orchestrator hub.

The ``AgentStateStore`` is the single source of truth for per-agent status,
results, and metadata across delegation cycles.  The orchestrator owns this
store; sub-agents never access it directly.  After each delegation the
orchestrator merges the sub-agent's handoff summary into the store.

Design decisions
----------------
* **Dict-backed, dataclass-typed** — ``AgentRecord`` is a dataclass for
  type safety and serialisation; the store itself is a thin dict wrapper
  keyed on ``AgentRole`` for O(1) lookups.
* **Immutable history** — every delegation produces a ``DelegationCycle``
  snapshot appended to an append-only list so post-run analysis can
  reconstruct the full timeline.
* **No direct sub-agent references** — the store only holds *data*, not
  agent instances, enforcing the hub-spoke communication model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from test_runner.agents.base import AgentRole


class AgentStatus(str, Enum):
    """Lifecycle status of a sub-agent from the orchestrator's perspective."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"


@dataclass
class DelegationCycle:
    """Snapshot of one orchestrator → sub-agent delegation round-trip.

    Attributes:
        cycle_id: Monotonically increasing cycle counter.
        agent_role: Which sub-agent was invoked.
        started_at: Unix timestamp when delegation began.
        finished_at: Unix timestamp when delegation ended (0 if still running).
        status: Final status after the cycle.
        input_summary: Data the orchestrator sent to the sub-agent.
        output_summary: Handoff data returned by the sub-agent.
        confidence_before: Agent confidence at start of cycle.
        confidence_after: Agent confidence at end of cycle.
        steps_taken: Steps consumed during this cycle.
        error: Error message if the cycle failed.
    """

    cycle_id: int
    agent_role: AgentRole
    started_at: float
    finished_at: float = 0.0
    status: AgentStatus = AgentStatus.RUNNING
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    steps_taken: int = 0
    error: str = ""

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of this delegation cycle."""
        if self.finished_at <= 0:
            return 0.0
        return self.finished_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "agent_role": self.agent_role.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "status": self.status.value,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "confidence_before": round(self.confidence_before, 4),
            "confidence_after": round(self.confidence_after, 4),
            "steps_taken": self.steps_taken,
            "error": self.error,
        }


@dataclass
class AgentRecord:
    """Accumulated state for one sub-agent across all delegation cycles.

    The orchestrator updates this record after every delegation cycle to
    maintain a running view of each agent's status, cumulative results,
    and metadata.

    Attributes:
        role: The agent's role identifier.
        status: Current lifecycle status.
        total_steps: Cumulative steps across all cycles.
        total_cycles: How many times this agent has been invoked.
        current_confidence: Latest confidence score.
        results: Accumulated results/findings from all cycles.
        errors: All errors encountered across cycles.
        metadata: Arbitrary per-agent metadata (framework info, etc.).
        last_updated: Unix timestamp of the most recent update.
    """

    role: AgentRole
    status: AgentStatus = AgentStatus.IDLE
    total_steps: int = 0
    total_cycles: int = 0
    current_confidence: float = 0.0
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_updated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "status": self.status.value,
            "total_steps": self.total_steps,
            "total_cycles": self.total_cycles,
            "current_confidence": round(self.current_confidence, 4),
            "results_count": len(self.results),
            "errors_count": len(self.errors),
            "errors": self.errors,
            "metadata": self.metadata,
            "last_updated": self.last_updated,
        }


class AgentStateStore:
    """Shared state store tracking all sub-agents across delegation cycles.

    This is the orchestrator's single source of truth for agent state.
    It provides methods to:

    * Start / finish delegation cycles with proper bookkeeping.
    * Query any agent's current status and accumulated results.
    * Produce a full snapshot for reporting or serialisation.

    Usage (inside the orchestrator hub)::

        store = AgentStateStore()
        cycle = store.start_delegation(AgentRole.DISCOVERY, input_data)
        # ... invoke sub-agent ...
        store.finish_delegation(cycle.cycle_id, handoff_summary)

    Thread safety: not thread-safe. The orchestrator is expected to be
    the sole writer.
    """

    def __init__(self) -> None:
        self._agents: dict[AgentRole, AgentRecord] = {}
        self._cycles: list[DelegationCycle] = []
        self._next_cycle_id: int = 1

    # --- Agent record management ---

    def _ensure_record(self, role: AgentRole) -> AgentRecord:
        """Return the record for *role*, creating it lazily if needed."""
        if role not in self._agents:
            self._agents[role] = AgentRecord(role=role)
        return self._agents[role]

    def get_agent(self, role: AgentRole) -> AgentRecord | None:
        """Return the record for *role*, or ``None`` if never delegated."""
        return self._agents.get(role)

    def get_agent_status(self, role: AgentRole) -> AgentStatus:
        """Return the current status for *role* (IDLE if never delegated)."""
        rec = self._agents.get(role)
        return rec.status if rec else AgentStatus.IDLE

    def all_agents(self) -> dict[AgentRole, AgentRecord]:
        """Return a shallow copy of all agent records."""
        return dict(self._agents)

    # --- Delegation cycle management ---

    def start_delegation(
        self,
        role: AgentRole,
        input_summary: dict[str, Any] | None = None,
        confidence: float = 0.0,
    ) -> DelegationCycle:
        """Begin a new delegation cycle for *role*.

        Creates a ``DelegationCycle``, marks the agent as RUNNING, and
        returns the cycle object.  The caller must later call
        :meth:`finish_delegation` with the same ``cycle_id``.
        """
        record = self._ensure_record(role)
        record.status = AgentStatus.RUNNING
        record.last_updated = time.time()

        cycle = DelegationCycle(
            cycle_id=self._next_cycle_id,
            agent_role=role,
            started_at=time.time(),
            input_summary=input_summary or {},
            confidence_before=confidence,
        )
        self._next_cycle_id += 1
        self._cycles.append(cycle)
        return cycle

    def finish_delegation(
        self,
        cycle_id: int,
        output_summary: dict[str, Any] | None = None,
        *,
        status: AgentStatus = AgentStatus.COMPLETED,
        error: str = "",
    ) -> DelegationCycle:
        """Finalise a delegation cycle and merge results into the agent record.

        Parameters
        ----------
        cycle_id : int
            The cycle to finalise (from ``start_delegation``).
        output_summary : dict | None
            Handoff data from the sub-agent (typically from
            ``BaseSubAgent.get_handoff_summary()``).
        status : AgentStatus
            Final status of this cycle.
        error : str
            Error message if the cycle failed.

        Returns
        -------
        DelegationCycle
            The updated cycle object.

        Raises
        ------
        KeyError
            If *cycle_id* is not found.
        """
        cycle = self._find_cycle(cycle_id)
        now = time.time()

        output = output_summary or {}
        cycle.finished_at = now
        cycle.status = status
        cycle.output_summary = output
        cycle.error = error

        # Extract per-cycle stats from the handoff summary
        agent_state = output.get("state", {})
        cycle.steps_taken = agent_state.get("steps_taken", 0)
        cycle.confidence_after = agent_state.get("current_confidence", 0.0)

        # Merge into the running agent record
        record = self._ensure_record(cycle.agent_role)
        record.status = status
        record.total_steps += cycle.steps_taken
        record.total_cycles += 1
        record.current_confidence = cycle.confidence_after
        record.last_updated = now

        # Append findings as results
        findings = agent_state.get("findings", [])
        record.results.extend(findings)

        # Append errors
        errors = agent_state.get("errors", [])
        record.errors.extend(errors)
        if error:
            record.errors.append(error)

        # Store metadata from the handoff (agent name, role, escalated, etc.)
        for key in ("agent", "role", "confidence_tier", "escalated"):
            if key in output:
                record.metadata[key] = output[key]

        return cycle

    def fail_delegation(
        self, cycle_id: int, error: str,
    ) -> DelegationCycle:
        """Convenience wrapper to mark a delegation cycle as FAILED."""
        return self.finish_delegation(
            cycle_id, status=AgentStatus.FAILED, error=error,
        )

    # --- Query helpers ---

    @property
    def total_cycles(self) -> int:
        """Total number of delegation cycles (started or finished)."""
        return len(self._cycles)

    @property
    def active_cycles(self) -> list[DelegationCycle]:
        """Cycles that are currently RUNNING."""
        return [c for c in self._cycles if c.status == AgentStatus.RUNNING]

    def cycles_for(self, role: AgentRole) -> list[DelegationCycle]:
        """Return all cycles (in order) for the given agent role."""
        return [c for c in self._cycles if c.agent_role == role]

    def latest_cycle_for(self, role: AgentRole) -> DelegationCycle | None:
        """Return the most recent cycle for *role*, or ``None``."""
        cycles = self.cycles_for(role)
        return cycles[-1] if cycles else None

    # --- Snapshot / serialisation ---

    def snapshot(self) -> dict[str, Any]:
        """Produce a full serialisable snapshot of the store.

        Suitable for embedding in run reports or passing to the reporter
        sub-agent.
        """
        return {
            "agents": {
                role.value: record.to_dict()
                for role, record in self._agents.items()
            },
            "total_cycles": len(self._cycles),
            "cycles": [c.to_dict() for c in self._cycles],
            "active_cycles": len(self.active_cycles),
        }

    def agent_summary(self, role: AgentRole) -> dict[str, Any]:
        """Return a concise summary for one agent, suitable for handoff context."""
        record = self._agents.get(role)
        if record is None:
            return {"role": role.value, "status": AgentStatus.IDLE.value}
        return record.to_dict()

    # --- Internals ---

    def _find_cycle(self, cycle_id: int) -> DelegationCycle:
        """Look up a cycle by ID."""
        # Cycles are stored in order, so the ID-1 index is usually correct
        idx = cycle_id - 1
        if 0 <= idx < len(self._cycles) and self._cycles[idx].cycle_id == cycle_id:
            return self._cycles[idx]
        # Fallback linear scan
        for cycle in self._cycles:
            if cycle.cycle_id == cycle_id:
                return cycle
        msg = f"Delegation cycle {cycle_id} not found"
        raise KeyError(msg)

    def reset(self) -> None:
        """Clear all state (for testing or re-runs)."""
        self._agents.clear()
        self._cycles.clear()
        self._next_cycle_id = 1
