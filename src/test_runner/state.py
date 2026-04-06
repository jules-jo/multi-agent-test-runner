"""Shared orchestrator state store with thread-safe access.

The OrchestratorState is the single source of truth for all inter-agent
communication.  Sub-agents never talk to each other directly — they read
from and write to this store, and the orchestrator hub manages routing.

Thread safety is provided via a reentrant lock so the store can be used
safely from async tasks running on different threads (e.g. executor
callbacks).
"""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Agent identity ──────────────────────────────────────────────────────

class AgentRole(str, Enum):
    """Well-known sub-agent roles in the orchestrator hub."""

    ORCHESTRATOR = "orchestrator"
    DISCOVERY = "discovery"
    EXECUTOR = "executor"
    REPORTER = "reporter"
    TROUBLESHOOTER = "troubleshooter"


class AgentStatus(str, Enum):
    """Lifecycle status of a sub-agent."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"  # blocked on orchestrator decision


# ── Per-agent state ─────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Mutable state envelope for a single agent."""

    role: AgentRole
    status: AgentStatus = AgentStatus.IDLE
    result: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Bump the ``last_updated`` timestamp to *now*."""
        self.last_updated = time.time()


# Sentinel for "not provided" defaults (distinct from None)
_UNSET = object()


# ── Orchestrator-wide state ─────────────────────────────────────────────

@dataclass
class OrchestratorState:
    """Central, thread-safe state store shared across all agents.

    All public methods acquire the internal ``_lock`` before mutating or
    reading state so the object is safe to use from multiple threads.

    Usage::

        state = OrchestratorState()
        state.register_agent(AgentRole.DISCOVERY)
        state.update_agent(AgentRole.DISCOVERY, status=AgentStatus.RUNNING)
        snapshot = state.get_agent(AgentRole.DISCOVERY)
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    # Internal storage — keyed by AgentRole
    _agents: dict[AgentRole, AgentState] = field(default_factory=dict, repr=False)

    # Shared scratchpad for cross-agent data (e.g. discovered test files)
    _shared: dict[str, Any] = field(default_factory=dict, repr=False)

    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ── Agent registry ──────────────────────────────────────────────

    def register_agent(self, role: AgentRole) -> AgentState:
        """Register a new agent role, returning its initial state.

        Raises ``ValueError`` if the role is already registered.
        """
        with self._lock:
            if role in self._agents:
                raise ValueError(f"Agent role {role.value!r} is already registered")
            agent = AgentState(role=role)
            self._agents[role] = agent
            return deepcopy(agent)

    def get_agent(self, role: AgentRole) -> AgentState:
        """Return a *snapshot* (deep copy) of an agent's state.

        Callers receive a copy so they cannot accidentally mutate the
        canonical state without going through ``update_agent``.
        """
        with self._lock:
            if role not in self._agents:
                raise KeyError(f"Agent role {role.value!r} is not registered")
            return deepcopy(self._agents[role])

    def update_agent(
        self,
        role: AgentRole,
        *,
        status: Optional[AgentStatus] = None,
        result: Any = _UNSET,
        error: Any = _UNSET,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentState:
        """Atomically update one or more fields on an agent's state.

        Only provided keyword arguments are written; omitted fields are
        left untouched.  Returns a snapshot of the updated state.
        """
        with self._lock:
            if role not in self._agents:
                raise KeyError(f"Agent role {role.value!r} is not registered")
            agent = self._agents[role]
            if status is not None:
                agent.status = status
            if result is not _UNSET:
                agent.result = result
            if error is not _UNSET:
                agent.error = error
            if metadata is not None:
                agent.metadata.update(metadata)
            agent.touch()
            return deepcopy(agent)

    def all_agents(self) -> dict[AgentRole, AgentState]:
        """Return snapshots of every registered agent's state."""
        with self._lock:
            return {role: deepcopy(st) for role, st in self._agents.items()}

    # ── Shared scratchpad ───────────────────────────────────────────

    def set_shared(self, key: str, value: Any) -> None:
        """Write a value to the shared scratchpad."""
        with self._lock:
            self._shared[key] = value

    def get_shared(self, key: str, default: Any = None) -> Any:
        """Read a value from the shared scratchpad (returns a deep copy)."""
        with self._lock:
            if key not in self._shared:
                return default
            return deepcopy(self._shared[key])

    def delete_shared(self, key: str) -> None:
        """Remove a key from the shared scratchpad.

        Raises ``KeyError`` if the key does not exist.
        """
        with self._lock:
            del self._shared[key]

    # ── Convenience queries ─────────────────────────────────────────

    def is_any_running(self) -> bool:
        """Return ``True`` if at least one agent has ``RUNNING`` status."""
        with self._lock:
            return any(a.status == AgentStatus.RUNNING for a in self._agents.values())

    def agents_with_status(self, status: AgentStatus) -> list[AgentRole]:
        """Return the roles of all agents currently in the given status."""
        with self._lock:
            return [role for role, a in self._agents.items() if a.status == status]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the full state."""
        with self._lock:
            return {
                "run_id": self.run_id,
                "created_at": self.created_at,
                "agents": {
                    role.value: {
                        "status": a.status.value,
                        "error": a.error,
                        "last_updated": a.last_updated,
                        "metadata": a.metadata,
                    }
                    for role, a in self._agents.items()
                },
                "shared_keys": list(self._shared.keys()),
            }
