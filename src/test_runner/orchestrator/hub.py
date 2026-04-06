"""Orchestrator hub — central coordinator for all sub-agents.

The hub manages state and routes between sub-agents (discovery, executor,
reporter, troubleshooter). Sub-agents never communicate directly; the
orchestrator manages all state and routing.

Escalation handling:
When the discovery agent exhausts its step budget with confidence still
below 60%, the orchestrator receives an EscalationResult and routes it:
- To the troubleshooter if structural issues are detected
- Back to the user (via the orchestrator itself) for clarification otherwise

Execution failure handling:
When the executor reports test failures, the orchestrator automatically
triggers the troubleshooter agent, passing relevant error context
(stderr, stdout, exit codes, error messages) extracted from the
TaskAttemptRecord results. The troubleshooter diagnoses failures and
produces fix proposals without auto-executing them.

Periodic rollup:
During the execution phase, the orchestrator starts the reporter agent's
rollup generator so periodic progress summaries are emitted at a
configurable interval through all registered reporting channels. The
rollup is automatically stopped when execution completes or the run fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from test_runner.orchestrator.state_store import AgentStateStore, AgentStatus
from test_runner.agents.base import AgentRole
from test_runner.autonomy.budget import (
    AgentBudgetConfig,
    BudgetExceededError,
    BudgetGuard,
    BudgetStatus,
    BudgetTracker,
    default_budget_config,
)
from test_runner.agents.discovery.agent import DiscoveryAgent
from test_runner.agents.discovery.threshold_evaluator import (
    EscalationResult,
    EscalationTarget,
)
from test_runner.agents.intent_service import (
    IntentParserService,
    IntentResolution,
    IntentResolutionError,
    ParseMode,
)
from test_runner.agents.reporter.agent import ReporterAgent
from test_runner.agents.reporter.rollup import RollupConfig
from test_runner.agents.troubleshooter.agent import TroubleshooterAgent
from test_runner.agents.troubleshooter.models import FixProposalSet
from test_runner.config import Config
from test_runner.execution.command_translator import TestCommand
from test_runner.execution.executor import TaskAttemptRecord, TaskExecutor
from test_runner.execution.targets import ExecutionStatus, ExecutionTarget
from test_runner.models.progress import ProgressTracker
from test_runner.models.summary import FailureDetail, TestOutcome
from test_runner.reporting.base import ReporterBase

logger = logging.getLogger(__name__)


class RunPhase(str, Enum):
    """Phases of an orchestrated test run."""

    INIT = "init"
    PARSING = "parsing"
    DISCOVERY = "discovery"
    EXECUTION = "execution"
    REPORTING = "reporting"
    TROUBLESHOOTING = "troubleshooting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class EscalationRecord:
    """Record of an escalation handled by the orchestrator.

    Attributes:
        source_agent: Which agent produced the escalation.
        target: Where it was routed.
        reason: Why escalation was triggered.
        confidence_score: Score at escalation time.
        steps_taken: Steps consumed when escalation occurred.
        resolution: How the escalation was resolved (if at all).
        metadata: Additional context from the escalating agent.
    """

    source_agent: str
    target: str
    reason: str
    confidence_score: float
    steps_taken: int
    resolution: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_agent": self.source_agent,
            "target": self.target,
            "reason": self.reason,
            "confidence_score": round(self.confidence_score, 4),
            "steps_taken": self.steps_taken,
            "resolution": self.resolution,
            "metadata": self.metadata,
        }


@dataclass
class RunState:
    """Mutable state for a single orchestrator run.

    The orchestrator owns this state; sub-agents receive read-only slices
    and return updates that the orchestrator merges.
    """

    request: str = ""
    phase: RunPhase = RunPhase.INIT
    intent_resolution: dict[str, Any] = field(default_factory=dict)
    discovered_tests: list[dict[str, Any]] = field(default_factory=list)
    execution_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    escalations: list[EscalationRecord] = field(default_factory=list)
    progress: ProgressTracker = field(default_factory=ProgressTracker)
    failure_details: list[FailureDetail] = field(default_factory=list)
    troubleshooter_result: FixProposalSet | None = None
    agent_store: AgentStateStore = field(default_factory=AgentStateStore)


class OrchestratorHub:
    """Central orchestrator that delegates to specialized sub-agents.

    Parameters
    ----------
    config : Config
        Application configuration including LLM connection details and
        autonomy policy.
    parse_mode : ParseMode
        Parsing strategy for intent resolution (LLM, offline, or auto).
    """

    def __init__(
        self,
        config: Config,
        *,
        parse_mode: ParseMode = ParseMode.AUTO,
        discovery: DiscoveryAgent | None = None,
        executor: TaskExecutor | None = None,
        troubleshooter: TroubleshooterAgent | None = None,
        reporter: ReporterAgent | None = None,
        rollup_config: RollupConfig | None = None,
        channels: list[ReporterBase] | None = None,
        budget_config: AgentBudgetConfig | None = None,
        execution_target: ExecutionTarget | None = None,
        execution_target_name: str = "local",
        command_timeout: int | None = None,
        default_command_env: dict[str, str] | None = None,
        default_working_directory: str = "",
    ) -> None:
        self._config = config
        self._intent_service = IntentParserService(
            config, parse_mode=parse_mode,
        )
        self._discovery = discovery or DiscoveryAgent()
        self._executor = executor or TaskExecutor()
        self._troubleshooter = troubleshooter or TroubleshooterAgent()

        # Budget tracking — the orchestrator enforces budget limits
        # before and after each delegation cycle.
        self._budget_tracker = BudgetTracker(budget_config or default_budget_config())
        self._budget_guard = BudgetGuard(self._budget_tracker)
        self._execution_target = execution_target
        self._execution_target_name = execution_target_name
        self._command_timeout = command_timeout
        self._default_command_env = dict(default_command_env or {})
        self._default_working_directory = default_working_directory

        # Reporter agent — owns reporting channels and rollup lifecycle
        if reporter is not None:
            self._reporter = reporter
        else:
            self._reporter = ReporterAgent(
                rollup_config=rollup_config or RollupConfig(),
            )
        if channels:
            for ch in channels:
                self._reporter.add_channel(ch)
        if getattr(self._executor, "_on_attempt", None) is None:
            self._executor._on_attempt = self._reporter.on_execution_attempt

    @property
    def intent_service(self) -> IntentParserService:
        """The intent parser service used by this orchestrator."""
        return self._intent_service

    @property
    def discovery(self) -> DiscoveryAgent:
        """The discovery sub-agent managed by this orchestrator."""
        return self._discovery

    @property
    def executor(self) -> TaskExecutor:
        """The executor sub-agent managed by this orchestrator."""
        return self._executor

    @property
    def troubleshooter(self) -> TroubleshooterAgent:
        """The troubleshooter sub-agent managed by this orchestrator."""
        return self._troubleshooter

    @property
    def reporter(self) -> ReporterAgent:
        """The reporter sub-agent managed by this orchestrator."""
        return self._reporter

    @property
    def budget_tracker(self) -> BudgetTracker:
        """The budget tracker monitoring per-agent resource consumption."""
        return self._budget_tracker

    @property
    def budget_guard(self) -> BudgetGuard:
        """The budget guard enforcing limits before/after delegations."""
        return self._budget_guard

    async def run(self, request: str) -> RunState:
        """Execute a full orchestrated test run for the given natural language request.

        The orchestrator starts the reporter's periodic rollup generator
        before entering the execution phase so that progress summaries
        are emitted at regular intervals through all registered reporting
        channels.  The rollup is stopped when execution finishes (or on
        error) and a final summary is produced during the reporting phase.

        Parameters
        ----------
        request : str
            Natural language description of the tests to run.

        Returns
        -------
        RunState
            Final state after all phases complete (or fail).
        """
        state = RunState(request=request, phase=RunPhase.INIT)
        logger.info("Orchestrator received request: %s", request)

        try:
            # Phase: Parsing — resolve intent and build commands
            state.phase = RunPhase.PARSING
            logger.info("Phase: %s", state.phase.value)
            resolution = await self._resolve_intent(state, request)

            if resolution.needs_clarification:
                logger.warning(
                    "Low confidence (%.2f) — orchestrator should request "
                    "clarification in a future interactive mode.",
                    resolution.confidence,
                )

            # Phase: Discovery — with budget enforcement and state tracking
            state.phase = RunPhase.DISCOVERY
            logger.info("Phase: %s", state.phase.value)
            await self._delegate_to_discovery(state, resolution)

            # Phase: Execution — start reporter lifecycle (incl. rollup)
            state.phase = RunPhase.EXECUTION
            logger.info("Phase: %s", state.phase.value)
            self._enforce_budget_or_escalate(state, AgentRole.REPORTER)
            await self._start_reporter(state)
            try:
                await self._delegate_to_executor(state, resolution)
            finally:
                # Always finalize the reporter, even if execution raises
                await self._finalize_reporter(state)

            # Phase: Troubleshooting — auto-triggered on execution failures
            if self._has_execution_failures(state):
                state.phase = RunPhase.TROUBLESHOOTING
                logger.info("Phase: %s — execution failures detected", state.phase.value)
                self._enforce_budget_or_escalate(state, AgentRole.TROUBLESHOOTER)
                await self._invoke_troubleshooter(state)

            # Phase: Reporting — summary already captured in state.report
            state.phase = RunPhase.REPORTING
            logger.info("Phase: %s", state.phase.value)

            state.phase = RunPhase.COMPLETE
        except BudgetExceededError as exc:
            logger.warning(
                "Budget exceeded for %s: %s",
                exc.status.role.value,
                ", ".join(r.value for r in exc.status.exceeded_reasons),
            )
            state.errors.append(str(exc))
            state.escalations.append(
                EscalationRecord(
                    source_agent=exc.status.role.value,
                    target="orchestrator",
                    reason="budget_exceeded",
                    confidence_score=0.0,
                    steps_taken=exc.status.iterations_used,
                    metadata=exc.status.summary(),
                )
            )
            # Allow graceful completion with what we have so far
            if state.phase not in (RunPhase.COMPLETE, RunPhase.FAILED):
                state.phase = RunPhase.FAILED
            await self._stop_reporter_safely(state)
        except IntentResolutionError as exc:
            logger.error("Intent resolution failed: %s", exc)
            state.phase = RunPhase.FAILED
            state.errors.append(f"Intent resolution failed: {exc}")
            await self._stop_reporter_safely(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Orchestrator run failed")
            state.phase = RunPhase.FAILED
            state.errors.append(str(exc))
            await self._stop_reporter_safely(state)

        return state

    # ----- Budget enforcement -----

    def _enforce_budget_or_escalate(
        self,
        state: RunState,
        role: AgentRole,
    ) -> BudgetStatus:
        """Check an agent's budget before delegation and escalate if exceeded.

        This is the main enforcement point in the orchestrator loop.
        Called before each delegation to ensure the agent hasn't already
        exhausted its budget from previous cycles.

        If the budget is exceeded, raises :class:`BudgetExceededError`
        which is caught by the ``run()`` method and converted into a
        graceful escalation.

        Args:
            state: Current run state (for logging context).
            role: The agent role to check.

        Returns:
            The budget status if within limits.

        Raises:
            BudgetExceededError: If the agent has exceeded any budget limit.
        """
        status = self._budget_guard.pre_check(role)
        logger.debug(
            "Budget pre-check for %s: iterations=%d tokens=%d wall_clock=%.1fs",
            role.value,
            status.iterations_used,
            status.tokens_used,
            status.wall_clock_used,
        )
        return status

    def _begin_budget_cycle(self, role: AgentRole) -> None:
        """Start tracking a budget cycle for an agent delegation.

        Called at the start of each delegation to begin wall-clock timing.
        """
        self._budget_tracker.begin_cycle(role)

    def _end_budget_cycle(self, role: AgentRole, *, tokens: int = 0) -> BudgetStatus:
        """End a budget cycle and check for budget exhaustion.

        Called after each delegation completes. Records the cycle's
        resource consumption and returns the updated budget status.

        Args:
            role: The agent role that just completed.
            tokens: Tokens consumed during this cycle.

        Returns:
            The updated budget status after recording consumption.
        """
        self._budget_tracker.end_cycle(role, tokens=tokens)
        status = self._budget_guard.post_check(role)
        if status.exceeded:
            logger.warning(
                "Budget exhausted for %s after cycle: %s",
                role.value,
                ", ".join(r.value for r in status.exceeded_reasons),
            )
        return status

    def get_budget_snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of all budget tracking state.

        Useful for including in run reports or debugging.
        """
        return self._budget_tracker.snapshot()

    # ----- Discovery delegation -----

    async def _delegate_to_discovery(
        self,
        state: RunState,
        resolution: IntentResolution,
    ) -> None:
        """Delegate to the discovery sub-agent with full state and budget wiring.

        Follows the canonical delegation pattern:
        1. Pre-check budget (raises BudgetExceededError if exhausted)
        2. Start delegation cycle in agent state store
        3. Begin budget cycle for wall-clock tracking
        4. Execute the sub-agent
        5. End budget cycle and record resource consumption
        6. Finish delegation cycle with handoff summary

        If the discovery agent produces an escalation (e.g. step cap hit
        with low confidence), the orchestrator handles it via
        ``handle_escalation``.

        Args:
            state: Current run state — updated with discovered tests.
            resolution: Intent resolution from the parsing phase.
        """
        self._enforce_budget_or_escalate(state, AgentRole.DISCOVERY)

        # Reset discovery agent for a fresh session
        self._discovery.reset_state()

        # Begin tracking in the agent state store
        disc_cycle = state.agent_store.start_delegation(
            AgentRole.DISCOVERY,
            input_summary={
                "request": state.request,
                "intent": resolution.intent.value,
                "framework": resolution.framework.value,
                "commands": [c.display for c in resolution.commands],
            },
            confidence=self._discovery.state.current_confidence,
        )

        # Begin budget cycle for wall-clock and resource tracking
        self._begin_budget_cycle(AgentRole.DISCOVERY)

        try:
            # The discovery agent explores the project and produces findings.
            # In the current architecture the orchestrator uses the intent
            # resolution's commands as discovered tests. Future iterations
            # will invoke the discovery Agent SDK loop here.
            for cmd in resolution.commands:
                finding = {
                    "command": cmd.display,
                    "framework": cmd.framework.value if hasattr(cmd, "framework") else "unknown",
                    "source": "intent_resolution",
                }
                self._discovery.state.add_finding(finding)
                self._discovery.state.record_step(
                    confidence=resolution.confidence,
                )

            state.discovered_tests = list(self._discovery.state.findings)

            # Check for escalation after discovery completes
            if self._discovery.should_escalate():
                escalation = self._discovery.last_escalation
                if escalation is not None:
                    self.handle_escalation(state, escalation, source_agent="discovery")

            # End budget cycle — record consumption
            budget_status = self._end_budget_cycle(AgentRole.DISCOVERY)

            # Close the delegation cycle with handoff summary
            handoff = self._discovery.get_handoff_summary()
            state.agent_store.finish_delegation(
                disc_cycle.cycle_id,
                output_summary=handoff,
                status=AgentStatus.ESCALATED
                if self._discovery.state.escalation_reason
                else AgentStatus.COMPLETED,
            )

            logger.info(
                "Discovery completed — %d test(s) found, confidence=%.2f, "
                "budget iterations=%d",
                len(state.discovered_tests),
                self._discovery.state.current_confidence,
                budget_status.iterations_used,
            )

        except Exception as exc:
            # End budget cycle even on failure
            self._end_budget_cycle(AgentRole.DISCOVERY)
            state.agent_store.fail_delegation(
                disc_cycle.cycle_id,
                error=str(exc),
            )
            raise

    # ----- Executor delegation -----

    async def _delegate_to_executor(
        self,
        state: RunState,
        resolution: IntentResolution,
    ) -> None:
        """Delegate to the executor sub-agent with full state and budget wiring.

        Follows the same canonical delegation pattern as discovery:
        1. Pre-check budget
        2. Start delegation in state store
        3. Begin budget cycle
        4. Execute test commands
        5. End budget cycle
        6. Finish delegation in state store

        Execution results are stored in ``state.execution_results`` and
        streamed to the reporter via ``feed_execution_output``.

        Args:
            state: Current run state — updated with execution results.
            resolution: Intent resolution containing commands to execute.
        """
        self._enforce_budget_or_escalate(state, AgentRole.EXECUTOR)

        # Begin tracking in the agent state store
        exec_cycle = state.agent_store.start_delegation(
            AgentRole.EXECUTOR,
            input_summary={
                "command_count": len(resolution.commands),
                "commands": [c.display for c in resolution.commands],
            },
            confidence=0.0,  # executor starts with no confidence
        )

        # Begin budget cycle
        self._begin_budget_cycle(AgentRole.EXECUTOR)

        try:
            # Execute each command through the TaskExecutor
            records = await self._executor.execute_batch(
                resolution.commands,
                target=self._execution_target,
                target_name=self._execution_target_name,
            )

            # Store results as serialized summaries in state
            for record in records:
                summary = record.to_summary()
                state.execution_results.append(summary)

            # End budget cycle — record consumption
            budget_status = self._end_budget_cycle(AgentRole.EXECUTOR)

            # Build handoff summary for state store
            batch_summary = self._executor.batch_summary()
            exec_handoff = {
                "agent": "executor",
                "role": AgentRole.EXECUTOR.value,
                "state": {
                    "steps_taken": len(records),
                    "current_confidence": 1.0 if batch_summary["failed"] == 0 else 0.5,
                    "findings": [r.to_summary() for r in records],
                    "errors": [
                        r.to_summary()["command"]
                        for r in records
                        if r.final_status != ExecutionStatus.PASSED
                    ],
                },
            }

            state.agent_store.finish_delegation(
                exec_cycle.cycle_id,
                output_summary=exec_handoff,
                status=AgentStatus.COMPLETED,
            )

            logger.info(
                "Execution completed — %d task(s), passed=%d, failed=%d, "
                "budget iterations=%d",
                batch_summary["total_tasks"],
                batch_summary["passed"],
                batch_summary["failed"],
                budget_status.iterations_used,
            )

        except Exception as exc:
            self._end_budget_cycle(AgentRole.EXECUTOR)
            state.agent_store.fail_delegation(
                exec_cycle.cycle_id,
                error=str(exc),
            )
            raise

    # ----- Reporter wiring -----

    async def _start_reporter(self, state: RunState) -> None:
        """Start the reporter agent's run lifecycle.

        Resets the reporter for a fresh run, starts the delegation cycle
        in the agent state store, and signals ``start_run`` so reporting
        channels and the rollup generator are activated.

        Args:
            state: Current run state (agent_store is updated).
        """
        self._reporter.reset_state()
        self._begin_budget_cycle(AgentRole.REPORTER)
        self._reporter_cycle = state.agent_store.start_delegation(
            AgentRole.REPORTER,
            input_summary={"request": state.request},
            confidence=self._reporter.state.current_confidence,
        )
        logger.info(
            "Reporter started (cycle %d)", self._reporter_cycle.cycle_id,
        )
        await self._reporter.start_run()

    async def _finalize_reporter(self, state: RunState) -> None:
        """Finalize the reporter agent and store the summary.

        Calls ``end_run`` to stop rollup, emit summary events, and collect
        the final run summary.  The delegation cycle is closed in the
        agent state store.

        Args:
            state: Current run state — ``state.report`` is populated.
        """
        try:
            state.report = await self._reporter.end_run()
            # End budget cycle for reporter
            self._end_budget_cycle(AgentRole.REPORTER)
            handoff = self._reporter.get_handoff_summary()
            cycle_id = getattr(self, "_reporter_cycle", None)
            if cycle_id is not None:
                state.agent_store.finish_delegation(
                    self._reporter_cycle.cycle_id,
                    output_summary=handoff,
                    status=AgentStatus.COMPLETED,
                )
            logger.info(
                "Reporter finalized — total=%d passed=%d failed=%d",
                state.report.get("total", 0),
                state.report.get("passed", 0),
                state.report.get("failed", 0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reporter finalization failed")
            state.errors.append(f"Reporter error: {exc}")
            cycle_id = getattr(self, "_reporter_cycle", None)
            if cycle_id is not None:
                state.agent_store.fail_delegation(
                    self._reporter_cycle.cycle_id,
                    error=str(exc),
                )

    async def _stop_reporter_safely(self, state: RunState) -> None:
        """Stop the reporter if it is running, swallowing errors.

        Used in error-recovery paths where the run is already failing.
        """
        try:
            rg = self._reporter.rollup_generator
            if rg is not None and rg.is_running:
                await self._finalize_reporter(state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error stopping reporter during cleanup: %s", exc)

    async def feed_execution_output(
        self,
        output: str | list[str],
        framework: str = "generic",
    ) -> None:
        """Route executor output to the reporter for real-time streaming.

        The orchestrator calls this method during the execution phase to
        feed raw test output lines to the reporter.  The reporter parses
        each line through the appropriate ``OutputParser`` and emits
        ``TestResultEvent`` objects immediately to all registered
        reporting channels.

        This maintains the hub-spoke model: the executor never talks to
        the reporter directly — the orchestrator routes everything.

        Args:
            output: Raw output lines from the executor (string or list).
            framework: Framework identifier for parser selection.
        """
        await self._reporter.process_output(output, framework)

    def handle_escalation(
        self,
        state: RunState,
        escalation: EscalationResult,
        source_agent: str = "discovery",
    ) -> RunPhase:
        """Handle an escalation from a sub-agent.

        Routes the escalation based on its target:
        - TROUBLESHOOTER: transitions to troubleshooting phase
        - ORCHESTRATOR: records the escalation for user clarification

        The orchestrator never auto-fixes — it only records the
        escalation and transitions phases. The actual troubleshooting
        or user interaction happens in the respective phase handler.

        Args:
            state: Current run state to update.
            escalation: The escalation result from the sub-agent.
            source_agent: Name of the agent that triggered escalation.

        Returns:
            The new RunPhase after handling the escalation.
        """
        record = EscalationRecord(
            source_agent=source_agent,
            target=escalation.target.value,
            reason=escalation.reason.value,
            confidence_score=escalation.confidence_score,
            steps_taken=escalation.steps_taken,
            metadata=escalation.metadata,
        )
        state.escalations.append(record)

        logger.info(
            "Escalation from %s: target=%s reason=%s score=%.4f",
            source_agent,
            escalation.target.value,
            escalation.reason.value,
            escalation.confidence_score,
        )

        if escalation.target == EscalationTarget.TROUBLESHOOTER:
            state.phase = RunPhase.TROUBLESHOOTING
            logger.info(
                "Routing escalation to troubleshooter — "
                "structural issues detected"
            )
            return RunPhase.TROUBLESHOOTING

        # Default: stay with orchestrator for user clarification
        logger.info(
            "Escalation retained by orchestrator — "
            "needs user clarification (confidence %.1f%%)",
            escalation.confidence_score * 100,
        )
        return state.phase

    def get_escalation_summary(self, state: RunState) -> dict[str, Any]:
        """Produce a summary of all escalations in the current run.

        Args:
            state: The run state containing escalation records.

        Returns:
            A serializable summary for reporting.
        """
        return {
            "total_escalations": len(state.escalations),
            "escalations": [e.to_dict() for e in state.escalations],
            "routed_to_troubleshooter": any(
                e.target == EscalationTarget.TROUBLESHOOTER.value
                for e in state.escalations
            ),
            "needs_user_clarification": any(
                e.target == EscalationTarget.ORCHESTRATOR.value
                and not e.resolution
                for e in state.escalations
            ),
        }

    # ----- Execution failure → troubleshooter wiring -----

    def _has_execution_failures(self, state: RunState) -> bool:
        """Check if execution results contain any failures.

        Examines both ``state.execution_results`` (dict-based summaries
        from the executor) and ``state.failure_details`` (pre-built
        FailureDetail objects) to determine if troubleshooting is needed.

        Returns:
            True if there is at least one non-passing execution result.
        """
        # Check pre-built failure details first
        if state.failure_details:
            return True

        # Check dict-based execution results
        failure_statuses = {
            ExecutionStatus.FAILED.value,
            ExecutionStatus.ERROR.value,
            ExecutionStatus.TIMEOUT.value,
        }
        for result in state.execution_results:
            if result.get("final_status") in failure_statuses:
                return True
        return False

    def _create_llm_caller(self) -> Any:
        """Create an async LLM caller using the configured Dataiku LLM Mesh endpoint.

        Returns an async callable ``(system_prompt, user_prompt) -> str``
        suitable for the ``FixGenerator``, or ``None`` if the LLM connection
        is not configured.

        The caller uses the ``openai`` library's ``AsyncOpenAI`` client
        with ``base_url`` pointed at the Dataiku LLM Mesh endpoint, which
        is OpenAI-compatible.
        """
        config = self._config
        if not config.llm_base_url or not config.api_key:
            logger.info(
                "LLM connection not configured — troubleshooter will use "
                "pattern-only analysis"
            )
            return None

        async def _call_llm(system_prompt: str, user_prompt: str) -> str:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url=config.llm_base_url,
                api_key=config.api_key,
            )
            response = await client.chat.completions.create(
                model=config.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""

        return _call_llm

    async def _invoke_troubleshooter(self, state: RunState) -> None:
        """Invoke the troubleshooter agent on execution failures.

        This is the main wiring point: when the executor reports failures,
        the orchestrator:
        1. Extracts failure context from execution results
        2. Resets the troubleshooter for a fresh session
        3. Injects the LLM caller for AI-augmented analysis
        4. Delegates to the troubleshooter's AI-powered fix proposal generation
        5. Stores the result in state for the reporter

        The troubleshooter never auto-executes fixes — it only proposes
        them. The autonomy policy on the troubleshooter controls this.

        Args:
            state: Current run state with execution_results populated.
        """
        # Build FailureDetail objects from execution results if not already present
        failures = state.failure_details
        if not failures:
            failures = self._extract_failure_details(state)
            state.failure_details = failures

        if not failures:
            logger.info("No failure details to troubleshoot — skipping")
            return

        logger.info(
            "Invoking troubleshooter for %d failure(s)", len(failures),
        )

        # Reset troubleshooter for a fresh session
        self._troubleshooter.reset_state()

        # Begin budget tracking for this troubleshooter cycle
        self._begin_budget_cycle(AgentRole.TROUBLESHOOTER)

        # Track delegation in the agent state store
        ts_cycle = state.agent_store.start_delegation(
            AgentRole.TROUBLESHOOTER,
            input_summary={
                "failure_count": len(failures),
                "failure_ids": [f.test_id for f in failures],
            },
            confidence=self._troubleshooter.state.current_confidence,
        )

        # Inject LLM caller for AI-augmented analysis
        llm_caller = self._create_llm_caller()
        if llm_caller is not None:
            self._troubleshooter.set_llm_caller(llm_caller)

        # Generate fix proposals with AI augmentation (diagnose-only, never auto-executes)
        try:
            proposal_set = await self._troubleshooter.generate_fix_proposals_with_llm(
                failures,
            )
            state.troubleshooter_result = proposal_set

            llm_augmented_count = sum(
                1 for p in proposal_set.proposals
                if p.metadata.get("llm_augmented")
            )
            logger.info(
                "Troubleshooter produced %d proposal(s) "
                "(high=%d, actionable=%d, llm_augmented=%d)",
                proposal_set.total_proposals_generated,
                proposal_set.high_confidence_count,
                proposal_set.actionable_count,
                llm_augmented_count,
            )

            # End budget cycle for troubleshooter
            self._end_budget_cycle(AgentRole.TROUBLESHOOTER)

            # Close the delegation cycle successfully
            handoff = self._troubleshooter.get_handoff_summary()
            state.agent_store.finish_delegation(
                ts_cycle.cycle_id,
                output_summary=handoff,
                status=AgentStatus.COMPLETED,
            )

            # Record an escalation so the reporting phase knows
            # troubleshooting occurred
            state.escalations.append(
                EscalationRecord(
                    source_agent="executor",
                    target="troubleshooter",
                    reason="execution_failures",
                    confidence_score=self._troubleshooter.state.current_confidence,
                    steps_taken=self._troubleshooter.state.steps_taken,
                    metadata={
                        "failure_count": len(failures),
                        "proposals_generated": proposal_set.total_proposals_generated,
                        "high_confidence_proposals": proposal_set.high_confidence_count,
                        "llm_augmented_proposals": llm_augmented_count,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Troubleshooter invocation failed")
            state.errors.append(f"Troubleshooter error: {exc}")
            state.agent_store.fail_delegation(
                ts_cycle.cycle_id,
                error=str(exc),
            )

    @staticmethod
    def _extract_failure_details(state: RunState) -> list[FailureDetail]:
        """Extract FailureDetail objects from execution result dicts.

        Converts the dict-based execution summaries (produced by
        ``TaskAttemptRecord.to_summary()``) into structured
        ``FailureDetail`` objects suitable for the troubleshooter.

        Only non-passing results are converted. Each failure detail
        includes stderr, stdout, exit code, and timing from the
        *last* execution attempt.

        Args:
            state: Run state containing execution_results dicts.

        Returns:
            List of FailureDetail objects for failed/errored/timed-out tasks.
        """
        failure_statuses = {
            ExecutionStatus.FAILED.value,
            ExecutionStatus.ERROR.value,
            ExecutionStatus.TIMEOUT.value,
        }
        details: list[FailureDetail] = []

        for result in state.execution_results:
            final_status = result.get("final_status", "")
            if final_status not in failure_statuses:
                continue

            # Map execution status to test outcome
            outcome_map = {
                ExecutionStatus.FAILED.value: TestOutcome.FAILED,
                ExecutionStatus.ERROR.value: TestOutcome.ERROR,
                ExecutionStatus.TIMEOUT.value: TestOutcome.ERROR,
            }
            outcome = outcome_map.get(final_status, TestOutcome.ERROR)

            # Extract error context from the last attempt
            attempt_details = result.get("attempt_details", [])
            last_attempt = attempt_details[-1] if attempt_details else {}

            task_id = result.get("task_id", "unknown")
            command = result.get("command", "")
            framework = result.get("framework", "")

            # Build error message from available context
            error_message = (
                f"Command '{command}' {final_status} "
                f"(exit code: {last_attempt.get('exit_code', '?')})"
            )
            if final_status == ExecutionStatus.TIMEOUT.value:
                error_message = f"Command '{command}' timed out"

            details.append(
                FailureDetail(
                    test_id=task_id,
                    test_name=command,
                    outcome=outcome,
                    error_message=error_message,
                    error_type=final_status,
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", ""),
                    duration_seconds=result.get("total_duration_seconds", 0.0),
                    framework=framework,
                    metadata={
                        "attempts_made": result.get("attempts_made", 0),
                        "max_attempts": result.get("max_attempts", 0),
                        "budget_exhausted": result.get("budget_exhausted", False),
                        "attempt_details": attempt_details,
                    },
                )
            )

        return details

    @staticmethod
    def build_failure_details_from_records(
        records: list[TaskAttemptRecord],
    ) -> list[FailureDetail]:
        """Build FailureDetail objects directly from TaskAttemptRecord instances.

        This is the preferred path when the orchestrator has direct access
        to TaskAttemptRecord objects (rather than serialized dicts). It
        preserves full stdout/stderr from the last attempt.

        Args:
            records: Task attempt records from the executor.

        Returns:
            List of FailureDetail objects for failed tasks only.
        """
        passing = {ExecutionStatus.PASSED, ExecutionStatus.SKIPPED, ExecutionStatus.PENDING}
        details: list[FailureDetail] = []

        for record in records:
            if record.final_status in passing:
                continue

            last = record.latest_result
            outcome_map = {
                ExecutionStatus.FAILED: TestOutcome.FAILED,
                ExecutionStatus.ERROR: TestOutcome.ERROR,
                ExecutionStatus.TIMEOUT: TestOutcome.ERROR,
            }
            outcome = outcome_map.get(record.final_status, TestOutcome.ERROR)

            stdout = last.stdout if last else ""
            stderr = last.stderr if last else ""
            exit_code = last.exit_code if last else -1

            error_message = (
                f"Command '{record.command.display}' {record.final_status.value} "
                f"(exit code: {exit_code})"
            )
            if record.final_status == ExecutionStatus.TIMEOUT:
                error_message = f"Command '{record.command.display}' timed out"

            details.append(
                FailureDetail(
                    test_id=record.task_id,
                    test_name=record.command.display,
                    outcome=outcome,
                    error_message=error_message,
                    error_type=record.final_status.value,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=record.total_duration,
                    framework=record.command.framework.value,
                    metadata={
                        "attempts_made": record.attempt_count,
                        "max_attempts": record.max_attempts,
                        "budget_exhausted": record.budget_exhausted,
                    },
                )
            )

        return details

    async def _resolve_intent(
        self, state: RunState, request: str,
    ) -> IntentResolution:
        """Use the intent parser service to resolve the request.

        Updates state.intent_resolution with the summary.
        """
        resolution = await self._intent_service.resolve(
            request,
            timeout=self._command_timeout,
            env=self._default_command_env or None,
        )
        if self._default_working_directory:
            translated_commands = [
                replace(cmd, working_directory=self._default_working_directory)
                for cmd in resolution.commands
            ]
            resolution.translation.commands[:] = translated_commands
        state.intent_resolution = resolution.summary()
        logger.info(
            "Intent resolved: %s %s → %d command(s)",
            resolution.intent.value,
            resolution.framework.value,
            len(resolution.commands),
        )
        return resolution
