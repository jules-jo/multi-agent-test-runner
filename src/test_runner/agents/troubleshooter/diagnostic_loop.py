"""Diagnostic step loop for the troubleshooter agent.

Implements the core diagnostic iteration loop that:
1. Counts iterations via the DiagnosticStepGuard
2. Exits early when a root cause is found with sufficient confidence
3. Enforces read-only safety via the ReadOnlySafetyGuard
4. Produces a structured DiagnosisSummary on completion

The loop processes a sequence of DiagnosticAction objects — each
representing one logical investigation step (e.g. read a log file,
inspect source code, check environment). Actions are provided by the
caller (typically the orchestrator or the agent's reasoning engine).

Design decisions:
- Actions are data objects, not callables — the loop validates and
  executes them, maintaining full control over safety and counting
- Early exit is configurable: root_cause_confidence_threshold controls
  when the loop considers the root cause "found enough" to stop
- The loop never mutates external state — it only reads and records
- Frozen result model (DiagnosisSummary) for safe cross-agent sharing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from test_runner.agents.troubleshooter.safety_guard import (
    ReadOnlySafetyGuard,
    SafetyViolation,
    ViolationType,
)
from test_runner.agents.troubleshooter.step_guard import (
    CompletionReason,
    DiagnosticStepGuard,
    DiagnosisSummary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic action model
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """Types of diagnostic actions the loop can execute."""

    READ_FILE = "read_file"
    CHECK_LOGS = "check_logs"
    INSPECT_ENV = "inspect_env"
    LIST_PROCESSES = "list_processes"
    ANALYZE_OUTPUT = "analyze_output"
    CUSTOM = "custom"


@dataclass(frozen=True)
class DiagnosticAction:
    """A single diagnostic action to execute in the loop.

    Attributes:
        action_type: Classification of the action.
        description: Human-readable description of what this step does.
        target: The specific target (e.g. file path, test ID).
        tool_name: Name of the tool to invoke (for safety validation).
        tool_args: Arguments for the tool call.
        command: Shell command (if action involves command execution).
        expected_finding: What the action is looking for (for logging).
        metadata: Additional context.
    """

    action_type: ActionType
    description: str
    target: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    command: str = ""
    expected_finding: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    """Result of executing a single diagnostic action.

    Attributes:
        action: The action that was executed.
        success: Whether the action completed successfully.
        finding: What was discovered (empty if nothing found).
        confidence_delta: Change in diagnostic confidence from this step.
        blocked: Whether the action was blocked by safety guard.
        violation: Safety violation details if blocked.
        error: Error message if the action failed.
        data: Raw result data from the tool execution.
        iteration: The 1-based iteration number in the loop.
        elapsed_ms: How long the action took in milliseconds.
    """

    action: DiagnosticAction
    success: bool
    finding: str = ""
    confidence_delta: float = 0.0
    blocked: bool = False
    violation: SafetyViolation | None = None
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    elapsed_ms: float = 0.0


class LoopExitReason(str, Enum):
    """Why the diagnostic loop exited."""

    ROOT_CAUSE_FOUND = "root_cause_found"
    STEP_LIMIT_REACHED = "step_limit_reached"
    ALL_ACTIONS_PROCESSED = "all_actions_processed"
    SAFETY_VIOLATION = "safety_violation"
    ERROR = "error"
    MANUAL_STOP = "manual_stop"


@dataclass
class LoopResult:
    """Result of running the full diagnostic loop.

    Attributes:
        exit_reason: Why the loop stopped.
        iterations_completed: Number of actions that were executed.
        total_actions_provided: Total actions in the input sequence.
        results: Results for each executed action.
        root_cause: Identified root cause (if any).
        confidence: Final diagnostic confidence.
        safety_violations: All safety violations encountered.
        diagnosis_summary: The finalized DiagnosisSummary from the guard.
        elapsed_ms: Total loop execution time in milliseconds.
    """

    exit_reason: LoopExitReason
    iterations_completed: int = 0
    total_actions_provided: int = 0
    results: list[ActionResult] = field(default_factory=list)
    root_cause: str = ""
    confidence: float = 0.0
    safety_violations: list[SafetyViolation] = field(default_factory=list)
    diagnosis_summary: DiagnosisSummary | None = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Diagnostic loop configuration
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticLoopConfig:
    """Configuration for the diagnostic step loop.

    Attributes:
        root_cause_confidence_threshold: Confidence level at which
            the loop considers the root cause "found" and exits early.
            Default 0.80 — high confidence means we're quite sure.
        stop_on_safety_violation: Whether to abort the entire loop
            when a safety violation is detected (vs. skipping and
            continuing). Default True for strict safety.
        max_consecutive_failures: Number of consecutive action failures
            before the loop gives up. Default 3.
        record_blocked_as_step: Whether blocked actions count toward
            the step limit. Default False — we don't penalize the agent
            for the safety guard doing its job.
    """

    root_cause_confidence_threshold: float = 0.80
    stop_on_safety_violation: bool = True
    max_consecutive_failures: int = 3
    record_blocked_as_step: bool = False


# ---------------------------------------------------------------------------
# Action executor interface
# ---------------------------------------------------------------------------


# An action executor takes a DiagnosticAction and returns
# (finding: str, confidence_delta: float, data: dict)
ActionExecutor = Callable[
    [DiagnosticAction],
    tuple[str, float, dict[str, Any]],
]


# ---------------------------------------------------------------------------
# Diagnostic loop
# ---------------------------------------------------------------------------


class DiagnosticLoop:
    """Runs the diagnostic step loop with safety and counting enforcement.

    The loop processes a sequence of DiagnosticAction objects, validating
    each one through the ReadOnlySafetyGuard and tracking iterations via
    the DiagnosticStepGuard. It exits early when:

    1. Root cause confidence exceeds the threshold
    2. The step guard's limit is reached
    3. A safety violation is detected (configurable)
    4. All provided actions are processed
    5. Too many consecutive failures occur

    Usage::

        guard = DiagnosticStepGuard(max_steps=10)
        safety = ReadOnlySafetyGuard()
        loop = DiagnosticLoop(step_guard=guard, safety_guard=safety)

        actions = [
            DiagnosticAction(
                action_type=ActionType.READ_FILE,
                description="Read failure log",
                target="test_output.log",
                tool_name="read_file",
                tool_args={"path": "test_output.log"},
            ),
            ...
        ]

        result = loop.run(actions, executor=my_executor)
        print(result.exit_reason)
        print(result.diagnosis_summary.summary_line())

    Thread-safety: NOT thread-safe. One loop per diagnostic session.
    """

    def __init__(
        self,
        step_guard: DiagnosticStepGuard,
        safety_guard: ReadOnlySafetyGuard,
        config: DiagnosticLoopConfig | None = None,
    ) -> None:
        self._step_guard = step_guard
        self._safety_guard = safety_guard
        self._config = config or DiagnosticLoopConfig()
        self._running_confidence: float = 0.0
        self._root_cause: str = ""
        self._proposed_fixes: list[str] = []
        self._alternative_causes: list[str] = []

    # -- Properties -----------------------------------------------------------

    @property
    def step_guard(self) -> DiagnosticStepGuard:
        """The step counting guard."""
        return self._step_guard

    @property
    def safety_guard(self) -> ReadOnlySafetyGuard:
        """The read-only safety guard."""
        return self._safety_guard

    @property
    def config(self) -> DiagnosticLoopConfig:
        """Loop configuration."""
        return self._config

    @property
    def running_confidence(self) -> float:
        """Current diagnostic confidence."""
        return self._running_confidence

    @property
    def root_cause(self) -> str:
        """Currently identified root cause."""
        return self._root_cause

    @property
    def root_cause_found(self) -> bool:
        """True if confidence exceeds the early-exit threshold."""
        return (
            bool(self._root_cause)
            and self._running_confidence >= self._config.root_cause_confidence_threshold
        )

    # -- Mutation for root cause tracking -------------------------------------

    def set_root_cause(
        self,
        root_cause: str,
        confidence: float | None = None,
    ) -> None:
        """Update the identified root cause.

        Can be called by the executor or externally to signal that
        a root cause has been found.

        Args:
            root_cause: The identified root cause.
            confidence: Optional confidence override.
        """
        self._root_cause = root_cause
        if confidence is not None:
            self._running_confidence = max(0.0, min(1.0, confidence))

    def add_proposed_fix(self, fix: str) -> None:
        """Add a proposed fix to the accumulation list."""
        self._proposed_fixes.append(fix)

    def add_alternative_cause(self, cause: str) -> None:
        """Add an alternative cause to the accumulation list."""
        self._alternative_causes.append(cause)

    # -- Main loop ------------------------------------------------------------

    def run(
        self,
        actions: Sequence[DiagnosticAction],
        executor: ActionExecutor,
    ) -> LoopResult:
        """Execute the diagnostic step loop.

        Processes actions in order, validating each through the safety
        guard and recording steps in the step guard. Exits early based
        on the configured conditions.

        Args:
            actions: Ordered sequence of diagnostic actions to execute.
            executor: Callable that executes a single action and returns
                (finding, confidence_delta, data).

        Returns:
            A LoopResult with all results and the finalized summary.
        """
        loop_start = time.monotonic()
        results: list[ActionResult] = []
        violations: list[SafetyViolation] = []
        consecutive_failures = 0
        exit_reason = LoopExitReason.ALL_ACTIONS_PROCESSED

        # Ensure the step guard is started
        if not self._step_guard.is_started:
            self._step_guard.start()

        for i, action in enumerate(actions):
            iteration = i + 1

            # --- Check 1: Step guard budget ---
            if not self._step_guard.can_proceed():
                logger.info(
                    "Diagnostic loop: step limit reached at iteration %d",
                    iteration,
                )
                exit_reason = LoopExitReason.STEP_LIMIT_REACHED
                break

            # --- Check 2: Root cause early exit ---
            if self.root_cause_found:
                logger.info(
                    "Diagnostic loop: root cause found (confidence=%.2f) "
                    "at iteration %d, exiting early",
                    self._running_confidence,
                    iteration,
                )
                exit_reason = LoopExitReason.ROOT_CAUSE_FOUND
                break

            # --- Check 3: Safety validation ---
            action_start = time.monotonic()

            # Validate tool call
            if action.tool_name:
                safe, violation = self._safety_guard.validate_tool_call(
                    action.tool_name, action.tool_args,
                )
                if not safe:
                    violations.append(violation)  # type: ignore[arg-type]
                    results.append(ActionResult(
                        action=action,
                        success=False,
                        blocked=True,
                        violation=violation,
                        error=f"Blocked by safety guard: {violation.detail}" if violation else "Blocked",
                        iteration=iteration,
                        elapsed_ms=_elapsed_ms(action_start),
                    ))
                    if self._config.record_blocked_as_step:
                        self._step_guard.record_step(
                            action.description,
                            target=action.target,
                            finding=f"BLOCKED: {violation.detail}" if violation else "BLOCKED",
                        )
                    if self._config.stop_on_safety_violation:
                        exit_reason = LoopExitReason.SAFETY_VIOLATION
                        break
                    continue

            # Validate command
            if action.command:
                safe, violation = self._safety_guard.validate_command(action.command)
                if not safe:
                    violations.append(violation)  # type: ignore[arg-type]
                    results.append(ActionResult(
                        action=action,
                        success=False,
                        blocked=True,
                        violation=violation,
                        error=f"Blocked by safety guard: {violation.detail}" if violation else "Blocked",
                        iteration=iteration,
                        elapsed_ms=_elapsed_ms(action_start),
                    ))
                    if self._config.record_blocked_as_step:
                        self._step_guard.record_step(
                            action.description,
                            target=action.target,
                            finding=f"BLOCKED: {violation.detail}" if violation else "BLOCKED",
                        )
                    if self._config.stop_on_safety_violation:
                        exit_reason = LoopExitReason.SAFETY_VIOLATION
                        break
                    continue

            # --- Execute the action ---
            try:
                finding, confidence_delta, data = executor(action)
                consecutive_failures = 0

                # Update running confidence
                self._running_confidence = max(
                    0.0, min(1.0, self._running_confidence + confidence_delta),
                )

                # Record in step guard
                self._step_guard.record_step(
                    action.description,
                    target=action.target,
                    finding=finding,
                    confidence_delta=confidence_delta,
                    metadata=action.metadata,
                )

                # Check if finding contains root cause signal
                if data.get("root_cause"):
                    self.set_root_cause(
                        str(data["root_cause"]),
                        confidence=data.get("root_cause_confidence"),
                    )
                if data.get("proposed_fix"):
                    self.add_proposed_fix(str(data["proposed_fix"]))
                if data.get("alternative_cause"):
                    self.add_alternative_cause(str(data["alternative_cause"]))

                results.append(ActionResult(
                    action=action,
                    success=True,
                    finding=finding,
                    confidence_delta=confidence_delta,
                    data=data,
                    iteration=iteration,
                    elapsed_ms=_elapsed_ms(action_start),
                ))

            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    "Diagnostic action %d failed: %s — %s",
                    iteration,
                    action.description,
                    e,
                )

                # Record the failure in step guard
                self._step_guard.record_step(
                    action.description,
                    target=action.target,
                    finding=f"ERROR: {e}",
                )

                results.append(ActionResult(
                    action=action,
                    success=False,
                    error=str(e),
                    iteration=iteration,
                    elapsed_ms=_elapsed_ms(action_start),
                ))

                if consecutive_failures >= self._config.max_consecutive_failures:
                    logger.warning(
                        "Diagnostic loop: %d consecutive failures, aborting",
                        consecutive_failures,
                    )
                    exit_reason = LoopExitReason.ERROR
                    break

        # --- Finalize ---
        # Map exit reason to completion reason
        completion_reason = _map_exit_to_completion(exit_reason)

        # Finalize the step guard (only if not already finalized)
        diagnosis_summary: DiagnosisSummary | None = None
        if self._step_guard.is_started and not self._step_guard.is_finalized:
            diagnosis_summary = self._step_guard.finalize(
                reason=completion_reason,
                root_cause=self._root_cause,
                confidence=self._running_confidence if self._root_cause else None,
                proposed_fixes=self._proposed_fixes or None,
                alternative_causes=self._alternative_causes or None,
            )

        total_elapsed = _elapsed_ms(loop_start)
        iterations_completed = sum(1 for r in results if r.success)

        return LoopResult(
            exit_reason=exit_reason,
            iterations_completed=iterations_completed,
            total_actions_provided=len(actions),
            results=results,
            root_cause=self._root_cause,
            confidence=self._running_confidence,
            safety_violations=violations,
            diagnosis_summary=diagnosis_summary,
            elapsed_ms=total_elapsed,
        )

    def reset(self) -> None:
        """Reset loop state for reuse."""
        self._running_confidence = 0.0
        self._root_cause = ""
        self._proposed_fixes.clear()
        self._alternative_causes.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(start: float) -> float:
    """Compute elapsed milliseconds since start (monotonic)."""
    return round((time.monotonic() - start) * 1000, 2)


def _map_exit_to_completion(exit_reason: LoopExitReason) -> CompletionReason:
    """Map a LoopExitReason to a CompletionReason for the step guard."""
    mapping = {
        LoopExitReason.ROOT_CAUSE_FOUND: CompletionReason.COMPLETED,
        LoopExitReason.STEP_LIMIT_REACHED: CompletionReason.LIMIT_REACHED,
        LoopExitReason.ALL_ACTIONS_PROCESSED: CompletionReason.COMPLETED,
        LoopExitReason.SAFETY_VIOLATION: CompletionReason.ERROR,
        LoopExitReason.ERROR: CompletionReason.ERROR,
        LoopExitReason.MANUAL_STOP: CompletionReason.MANUAL_STOP,
    }
    return mapping.get(exit_reason, CompletionReason.COMPLETED)
