"""Base agent class providing common functionality for all sub-agents.

All sub-agents inherit from BaseSubAgent which provides:
- Confidence-based autonomous exploration with tier thresholds
- Step counting with hard-cap escalation
- Common state management
- Registration interface for the orchestrator hub
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from test_runner.models.confidence import AggregatedConfidence, ConfidenceTier


class AgentRole(str, Enum):
    """Roles for sub-agents in the orchestrator hub."""

    DISCOVERY = "discovery"
    EXECUTOR = "executor"
    REPORTER = "reporter"
    TROUBLESHOOTER = "troubleshooter"


@dataclass
class AgentState:
    """Mutable state tracked per agent invocation."""

    steps_taken: int = 0
    current_confidence: float = 0.5
    escalation_reason: str | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def record_step(self, confidence: float | None = None) -> None:
        """Record a step and optionally update confidence."""
        self.steps_taken += 1
        if confidence is not None:
            self.current_confidence = confidence

    def add_finding(self, finding: dict[str, Any]) -> None:
        self.findings.append(finding)

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_taken": self.steps_taken,
            "current_confidence": self.current_confidence,
            "escalation_reason": self.escalation_reason,
            "findings": self.findings,
            "errors": self.errors,
        }


class BaseSubAgent(ABC):
    """Abstract base class for all sub-agents.

    Sub-agents never communicate directly — the orchestrator hub manages
    all state and routing between agents. Each sub-agent:
    1. Receives a task from the orchestrator
    2. Uses its tools to accomplish the task
    3. Reports results back to the orchestrator

    Confidence-based exploration:
    - High confidence (>= threshold): proceed autonomously
    - Medium confidence: proceed with logging
    - Low confidence: escalate to orchestrator
    - Hard cap on steps: always escalate when reached
    """

    def __init__(
        self,
        role: AgentRole,
        hard_cap_steps: int = 50,
        high_threshold: float = 0.80,
        low_threshold: float = 0.40,
    ) -> None:
        self.role = role
        self.hard_cap_steps = hard_cap_steps
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.state = AgentState()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        ...

    @property
    @abstractmethod
    def instructions(self) -> str:
        """System prompt / instructions for this agent."""
        ...

    @abstractmethod
    def get_tools(self) -> list[Any]:
        """Return the list of tools this agent can use."""
        ...

    def confidence_tier(self) -> ConfidenceTier:
        """Return the current confidence tier."""
        c = self.state.current_confidence
        if c >= self.high_threshold:
            return ConfidenceTier.HIGH
        if c >= self.low_threshold:
            return ConfidenceTier.MEDIUM
        return ConfidenceTier.LOW

    def should_escalate(self) -> bool:
        """Check if agent should stop and escalate to orchestrator."""
        if self.state.steps_taken >= self.hard_cap_steps:
            return True
        return self.confidence_tier() == ConfidenceTier.LOW

    def reset_state(self) -> None:
        """Reset state for a new invocation."""
        self.state = AgentState()

    def get_handoff_summary(self) -> dict[str, Any]:
        """Generate a summary for the orchestrator after task completion."""
        return {
            "agent": self.name,
            "role": self.role.value,
            "state": self.state.to_dict(),
            "confidence_tier": self.confidence_tier().value,
            "escalated": self.state.escalation_reason is not None,
        }
