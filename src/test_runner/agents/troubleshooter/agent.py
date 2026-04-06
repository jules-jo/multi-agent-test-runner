"""Troubleshooter agent — diagnoses test failures and proposes fixes.

The troubleshooter agent is invoked when tests fail or when the discovery
agent escalates due to structural issues. It:

1. Inspects failure logs, error messages, and stack traces
2. Reads relevant source files and configuration
3. Checks environment variables and runtime state
4. Lists processes to detect resource conflicts
5. Analyzes failures and generates structured fix proposals

IMPORTANT: The troubleshooter is diagnose-only by default. It proposes
fixes but never auto-executes them. The autonomy policy controls whether
auto-fix is allowed (currently always requires user approval).

A step counter enforces a hard cap on investigation steps per session,
preventing unbounded troubleshooting exploration.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from agents import Agent, function_tool

from test_runner.agents.base import AgentRole, BaseSubAgent
from test_runner.agents.discovery.step_counter import (
    BUDGET_EXCEEDED_RESPONSE,
    DEFAULT_HARD_CAP,
    StepCounter,
)
from test_runner.agents.troubleshooter.step_guard import (
    CompletionReason,
    DEFAULT_MAX_DIAGNOSTIC_STEPS,
    DiagnosticStepGuard,
    DiagnosisSummary,
)
from test_runner.agents.troubleshooter.analyzer import (
    AnalyzerConfig,
    FailureAnalyzer,
    StrategyRegistry,
    create_default_registry,
)
from test_runner.agents.troubleshooter.fix_generator import (
    FixGenerator,
    FixGeneratorConfig,
)
from test_runner.agents.troubleshooter.models import (
    FixProposal,
    FixProposalSet,
)
from test_runner.config import Config
from test_runner.models.summary import FailureDetail
from test_runner.tools.troubleshooter_tools import (
    _check_logs_impl,
    _inspect_env_impl,
    _list_processes_impl,
    _read_file_impl,
)

logger = logging.getLogger(__name__)


# Default hard cap for troubleshooter — slightly more generous than
# discovery since diagnosing failures often requires reading multiple
# files and cross-referencing logs.
TROUBLESHOOTER_HARD_CAP = 30


TROUBLESHOOTER_INSTRUCTIONS = """\
You are the Troubleshooter Agent for a multi-agent test runner system.

Your job is to diagnose why tests failed and propose actionable fixes.
You have access to READ-ONLY tools — you MUST NOT modify any files,
run any test commands, or execute fixes yourself.

## Investigation Budget

You have a HARD CAP of {hard_cap} investigation steps for this session.
Each tool call counts as one step. Plan your diagnosis efficiently:
1. Start with the failure summary provided by the orchestrator
2. Read the most relevant log files and error output first
3. Check source files referenced in stack traces
4. Inspect environment only if failures suggest config issues
5. List processes only if failures suggest resource conflicts

When the budget is exhausted, tools will return an error and you MUST
immediately summarize your diagnosis.

## Diagnosis Strategy

Follow this priority order:
1. **Read failure logs**: Start with stderr/stdout from failed tests
2. **Inspect stack traces**: Read source files at the lines mentioned
3. **Check configuration**: Read test config files (pytest.ini, etc.)
4. **Environment check**: Only if failures suggest missing deps or config
5. **Process check**: Only if failures suggest port conflicts or OOM

## Output Format

Structure your diagnosis as:
- **Root Cause**: What is likely causing the failure (1-2 sentences)
- **Evidence**: What you found that supports this conclusion
- **Confidence**: How confident you are in the diagnosis (0.0-1.0)
- **Proposed Fix**: Step-by-step instructions for the user to fix the issue
- **Alternative Causes**: Other possible explanations if confidence < 0.8

## Safety Rules

- NEVER propose running arbitrary commands without user approval
- NEVER suggest modifying production files or databases
- NEVER expose sensitive environment variable values in your diagnosis
- Always explain WHY you recommend each fix step
- If you cannot determine the root cause, say so honestly and suggest
  what additional information would help
"""


def _make_tracked_tools(counter: StepCounter) -> list[Any]:
    """Create troubleshooter tools that track steps via the shared counter.

    Each tool checks the step budget before executing. If the budget is
    exhausted, the tool returns an error response instead of doing work.
    """

    @function_tool
    def read_file(path: str, max_lines: int = 500) -> dict[str, Any]:
        """Read file contents for troubleshooting (source, config, logs).

        Read-only inspection of any text file to help diagnose test failures.

        Args:
            path: File path to read.
            max_lines: Maximum number of lines to return.

        Returns:
            Dictionary with file content and metadata.
        """
        if not counter.increment("read_file", f"path={path}"):
            return {**BUDGET_EXCEEDED_RESPONSE, "budget": counter.summary()}
        result = _read_file_impl(path, max_lines)
        result["_budget"] = counter.budget_status_message()
        return result

    @function_tool
    def check_logs(
        path: str,
        tail_lines: int = 100,
        pattern: str = "",
    ) -> dict[str, Any]:
        """Read recent log entries from a file, optionally filtered by pattern.

        Reads from the end of the file (tail behavior) for efficiency.

        Args:
            path: Path to the log file.
            tail_lines: Number of lines to read from the end.
            pattern: Optional case-insensitive filter pattern.

        Returns:
            Dictionary with matching log lines and metadata.
        """
        if not counter.increment("check_logs", f"path={path} pattern={pattern}"):
            return {**BUDGET_EXCEEDED_RESPONSE, "budget": counter.summary()}
        result = _check_logs_impl(path, tail_lines, pattern or None)
        result["_budget"] = counter.budget_status_message()
        return result

    @function_tool
    def inspect_env(
        filter_prefix: str = "",
        include_python: bool = True,
    ) -> dict[str, Any]:
        """Inspect environment variables and Python runtime configuration.

        Sensitive values (SECRET, TOKEN, PASSWORD, KEY) are masked.

        Args:
            filter_prefix: Only return vars starting with this prefix.
            include_python: Include Python runtime info.

        Returns:
            Dictionary with environment variables and runtime details.
        """
        if not counter.increment("inspect_env", f"prefix={filter_prefix}"):
            return {**BUDGET_EXCEEDED_RESPONSE, "budget": counter.summary()}
        result = _inspect_env_impl(filter_prefix or None, include_python)
        result["_budget"] = counter.budget_status_message()
        return result

    @function_tool
    def list_processes(
        filter_pattern: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """List running processes to detect resource issues or conflicts.

        Useful for finding zombie test processes, port conflicts, etc.

        Args:
            filter_pattern: Optional filter to match against process commands.
            limit: Maximum number of processes to return.

        Returns:
            Dictionary with process list and metadata.
        """
        if not counter.increment("list_processes", f"filter={filter_pattern}"):
            return {**BUDGET_EXCEEDED_RESPONSE, "budget": counter.summary()}
        result = _list_processes_impl(filter_pattern or None, limit)
        result["_budget"] = counter.budget_status_message()
        return result

    return [read_file, check_logs, inspect_env, list_processes]


class TroubleshooterAgent(BaseSubAgent):
    """Troubleshooter sub-agent that diagnoses test failures.

    Uses four read-only tools:
    - read_file: inspect source files, configs, and logs
    - check_logs: tail log files with optional pattern filtering
    - inspect_env: check environment variables and runtime config
    - list_processes: detect resource conflicts and zombie processes

    Additionally provides programmatic fix proposal generation via
    the FailureAnalyzer, which classifies failures and produces
    structured FixProposal objects with confidence scores.

    A step counter enforces a hard cap on investigation steps.
    The default cap is 30 steps per session.

    The troubleshooter proposes fixes but never auto-executes them.
    The autonomy policy controls whether auto-fix could be enabled
    in future modes (currently always diagnose-only).

    Usage::

        troubleshooter = TroubleshooterAgent()

        # Programmatic fix proposal generation
        result = troubleshooter.generate_fix_proposals(failures)
        for proposal in result.by_confidence():
            print(proposal.summary_line())

        # Single failure analysis
        proposal = troubleshooter.analyze_single_failure(failure)
    """

    def __init__(
        self,
        hard_cap_steps: int = TROUBLESHOOTER_HARD_CAP,
        high_threshold: float = 0.80,
        low_threshold: float = 0.40,
        auto_fix_enabled: bool = False,
        analyzer: FailureAnalyzer | None = None,
        analyzer_config: AnalyzerConfig | None = None,
        max_diagnostic_steps: int = DEFAULT_MAX_DIAGNOSTIC_STEPS,
        fix_generator: FixGenerator | None = None,
        fix_generator_config: FixGeneratorConfig | None = None,
        llm_caller: Any = None,
    ) -> None:
        super().__init__(
            role=AgentRole.TROUBLESHOOTER,
            hard_cap_steps=hard_cap_steps,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
        self._step_counter = StepCounter(hard_cap=hard_cap_steps)
        self._auto_fix_enabled = auto_fix_enabled
        self._diagnosis: dict[str, Any] | None = None
        # Fix proposal analyzer
        if analyzer is not None:
            self._analyzer = analyzer
        else:
            self._analyzer = FailureAnalyzer(
                registry=create_default_registry(),
                config=analyzer_config or AnalyzerConfig(),
            )
        self._last_proposals: FixProposalSet | None = None
        # Diagnostic step guard — enforces ~10-step max on diagnostic iterations
        self._diagnostic_guard = DiagnosticStepGuard(max_steps=max_diagnostic_steps)
        self._last_diagnosis_summary: DiagnosisSummary | None = None

        # AI-powered fix generator (LLM-augmented analysis)
        if fix_generator is not None:
            self._fix_generator = fix_generator
        else:
            self._fix_generator = FixGenerator(
                analyzer=self._analyzer,
                llm_caller=llm_caller,
                config=fix_generator_config or FixGeneratorConfig(),
            )

    @property
    def step_counter(self) -> StepCounter:
        """The step counter for this troubleshooting session."""
        return self._step_counter

    @property
    def auto_fix_enabled(self) -> bool:
        """Whether auto-fix mode is enabled (currently always False)."""
        return self._auto_fix_enabled

    @property
    def diagnosis(self) -> dict[str, Any] | None:
        """The most recent diagnosis result, if any."""
        return self._diagnosis

    @property
    def analyzer(self) -> FailureAnalyzer:
        """The failure analyzer used for fix proposal generation."""
        return self._analyzer

    @property
    def last_proposals(self) -> FixProposalSet | None:
        """The most recent fix proposals, or None if not yet generated."""
        return self._last_proposals

    @property
    def fix_generator(self) -> FixGenerator:
        """The AI-powered fix generator for LLM-augmented analysis."""
        return self._fix_generator

    def set_llm_caller(self, llm_caller: Any) -> None:
        """Set or replace the LLM caller for AI-powered fix generation.

        This allows the orchestrator to inject the LLM caller after
        construction (e.g., when the Config becomes available).

        Args:
            llm_caller: An async callable (system_prompt, user_prompt) -> str.
        """
        self._fix_generator = FixGenerator(
            analyzer=self._analyzer,
            llm_caller=llm_caller,
            config=self._fix_generator.config,
        )

    @property
    def diagnostic_guard(self) -> DiagnosticStepGuard:
        """The diagnostic step guard for this session."""
        return self._diagnostic_guard

    @property
    def last_diagnosis_summary(self) -> DiagnosisSummary | None:
        """The most recent DiagnosisSummary, or None if not yet produced."""
        return self._last_diagnosis_summary

    @property
    def name(self) -> str:
        return "troubleshooter-agent"

    @property
    def instructions(self) -> str:
        return TROUBLESHOOTER_INSTRUCTIONS.format(
            hard_cap=self._step_counter.hard_cap,
        )

    def get_tools(self) -> list[Any]:
        """Return troubleshooter tools wired to the step counter."""
        return _make_tracked_tools(self._step_counter)

    def reset_state(self) -> None:
        """Reset state, step counter, guard, proposals, and LLM call counter."""
        super().reset_state()
        self._step_counter.reset()
        self._diagnosis = None
        self._last_proposals = None
        self._diagnostic_guard.reset()
        self._last_diagnosis_summary = None
        self._fix_generator.reset()

    def should_escalate(self) -> bool:
        """Check if agent should stop — includes step counter exhaustion."""
        if self._step_counter.is_exhausted:
            self.state.escalation_reason = (
                f"Step budget exhausted ({self._step_counter.hard_cap} steps)"
            )
            return True
        return super().should_escalate()

    def record_diagnosis(
        self,
        root_cause: str,
        evidence: list[str],
        confidence: float,
        proposed_fix: list[str],
        alternative_causes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record a structured diagnosis from the agent's analysis.

        Args:
            root_cause: The likely root cause of the failure.
            evidence: List of evidence supporting the diagnosis.
            confidence: Confidence in the diagnosis (0.0-1.0).
            proposed_fix: Step-by-step fix instructions for the user.
            alternative_causes: Other possible explanations.

        Returns:
            The structured diagnosis dict.
        """
        self._diagnosis = {
            "root_cause": root_cause,
            "evidence": evidence,
            "confidence": confidence,
            "proposed_fix": proposed_fix,
            "alternative_causes": alternative_causes or [],
            "auto_fix_enabled": self._auto_fix_enabled,
            "steps_used": self._step_counter.steps_taken,
            "steps_budget": self._step_counter.hard_cap,
        }
        self.state.current_confidence = confidence
        self.state.add_finding(self._diagnosis)
        return self._diagnosis

    # ----- Fix proposal generation -----

    def generate_fix_proposals(
        self,
        failures: Sequence[FailureDetail],
    ) -> FixProposalSet:
        """Analyze test failures and produce structured fix proposals.

        This is the main entry point for fix proposal generation, called
        by the orchestrator. It:
        1. Checks the step budget (counts as one step)
        2. Delegates to the FailureAnalyzer
        3. Updates agent state with findings
        4. Returns the FixProposalSet

        The troubleshooter never auto-executes fixes — proposals are
        returned to the orchestrator for user presentation.

        Args:
            failures: Test failure details from the reporter agent,
                      routed through the orchestrator.

        Returns:
            A FixProposalSet with all generated proposals.
        """
        logger.info(
            "Troubleshooter generating fix proposals for %d failure(s)",
            len(failures),
        )

        # Check if we should escalate before even starting
        if self.should_escalate():
            logger.warning(
                "Troubleshooter escalating before proposal generation: "
                "steps=%d confidence=%.2f",
                self.state.steps_taken,
                self.state.current_confidence,
            )
            self.state.escalation_reason = (
                "Step budget exhausted or confidence too low to analyze"
            )
            self._last_proposals = FixProposalSet(
                analysis_summary="Fix proposal generation skipped due to escalation.",
                budget_exhausted=True,
            )
            return self._last_proposals

        # Record as a step
        self.state.record_step()

        # Run the analyzer
        result = self._analyzer.analyze_failures(failures)
        self._last_proposals = result

        # Update agent state with proposal findings
        self.state.current_confidence = self._compute_proposal_confidence(result)

        for proposal in result.proposals:
            self.state.add_finding({
                "type": "fix_proposal",
                "failure_id": proposal.failure_id,
                "category": proposal.category.value,
                "confidence": proposal.confidence.value,
                "confidence_score": proposal.confidence_score,
                "title": proposal.title,
                "affected_files": proposal.affected_files,
                "change_count": proposal.change_count,
            })

        if result.budget_exhausted:
            self.state.add_error(
                f"Analyzer budget exhausted: analyzed {result.total_failures_analyzed} "
                f"of {len(failures)} failures"
            )

        logger.info(
            "Troubleshooter produced %d fix proposal(s) "
            "(high=%d, actionable=%d, budget_exhausted=%s)",
            result.total_proposals_generated,
            result.high_confidence_count,
            result.actionable_count,
            result.budget_exhausted,
        )

        return result

    def analyze_single_failure(
        self,
        failure: FailureDetail,
    ) -> FixProposal | None:
        """Analyze a single failure and return its fix proposal.

        Convenience method for one-off analysis. Updates state but does
        not replace last_proposals.

        Args:
            failure: A single test failure detail.

        Returns:
            A FixProposal, or None if no fix could be generated.
        """
        self.state.record_step()
        proposal = self._analyzer.analyze_single(failure)
        if proposal:
            self.state.add_finding({
                "type": "fix_proposal",
                "failure_id": proposal.failure_id,
                "category": proposal.category.value,
                "confidence": proposal.confidence.value,
                "title": proposal.title,
            })
        return proposal

    async def generate_fix_proposals_with_llm(
        self,
        failures: Sequence[FailureDetail],
    ) -> FixProposalSet:
        """Analyze test failures with AI-powered LLM augmentation.

        This is the preferred entry point for fix proposal generation when
        an LLM caller is available. It combines pattern-based classification
        with LLM-powered root-cause analysis for deeper, more actionable
        fix recommendations.

        Workflow:
        1. Checks step budget (counts as one step)
        2. Delegates to the FixGenerator's ``analyze_with_llm()``
        3. Falls back to pattern-only analysis if LLM is unavailable
        4. Updates agent state with findings
        5. Returns the FixProposalSet

        The troubleshooter never auto-executes fixes — proposals are
        returned to the orchestrator for user presentation.

        Args:
            failures: Test failure details from the reporter agent,
                      routed through the orchestrator.

        Returns:
            A FixProposalSet with all generated proposals (pattern + LLM).
        """
        logger.info(
            "Troubleshooter generating AI-augmented fix proposals for %d failure(s)",
            len(failures),
        )

        # Check if we should escalate before even starting
        if self.should_escalate():
            logger.warning(
                "Troubleshooter escalating before AI proposal generation: "
                "steps=%d confidence=%.2f",
                self.state.steps_taken,
                self.state.current_confidence,
            )
            self.state.escalation_reason = (
                "Step budget exhausted or confidence too low to analyze"
            )
            self._last_proposals = FixProposalSet(
                analysis_summary="AI fix proposal generation skipped due to escalation.",
                budget_exhausted=True,
            )
            return self._last_proposals

        # Record as a step
        self.state.record_step()

        # Run the AI-augmented analyzer
        try:
            result = await self._fix_generator.analyze_with_llm(failures)
        except Exception as exc:
            logger.exception("AI-augmented analysis failed, falling back to pattern-only")
            self.state.add_error(f"LLM analysis failed: {exc}")
            result = self._analyzer.analyze_failures(failures)

        self._last_proposals = result

        # Update agent state with proposal findings
        self.state.current_confidence = self._compute_proposal_confidence(result)

        for proposal in result.proposals:
            finding: dict[str, Any] = {
                "type": "fix_proposal",
                "failure_id": proposal.failure_id,
                "category": proposal.category.value,
                "confidence": proposal.confidence.value,
                "confidence_score": proposal.confidence_score,
                "title": proposal.title,
                "affected_files": proposal.affected_files,
                "change_count": proposal.change_count,
                "llm_augmented": proposal.metadata.get("llm_augmented", False),
            }
            self.state.add_finding(finding)

        if result.budget_exhausted:
            self.state.add_error(
                f"Analyzer budget exhausted: analyzed {result.total_failures_analyzed} "
                f"of {len(failures)} failures"
            )

        llm_augmented_count = sum(
            1 for p in result.proposals if p.metadata.get("llm_augmented")
        )

        logger.info(
            "Troubleshooter produced %d AI-augmented fix proposal(s) "
            "(high=%d, actionable=%d, llm_augmented=%d, budget_exhausted=%s)",
            result.total_proposals_generated,
            result.high_confidence_count,
            result.actionable_count,
            llm_augmented_count,
            result.budget_exhausted,
        )

        return result

    @staticmethod
    def _compute_proposal_confidence(result: FixProposalSet) -> float:
        """Compute overall confidence based on fix proposal results.

        Higher confidence when more proposals are high-confidence
        and actionable. Lower when budget was exhausted or many
        proposals are low-confidence.
        """
        if not result.proposals:
            return 0.3

        total = len(result.proposals)
        high_ratio = result.high_confidence_count / total
        actionable_ratio = result.actionable_count / total

        base = 0.4 + (high_ratio * 0.3) + (actionable_ratio * 0.2)

        if result.budget_exhausted:
            base *= 0.8  # Penalty for incomplete analysis

        return min(base, 1.0)

    # ----- Diagnostic step guard integration -----

    def start_diagnostic_session(self) -> None:
        """Start a new diagnostic session with the step guard.

        Call this before beginning diagnostic iterations.
        """
        self._diagnostic_guard.start()
        logger.info(
            "Diagnostic session started (max_steps=%d)",
            self._diagnostic_guard.max_steps,
        )

    def can_continue_diagnosis(self) -> bool:
        """Check if the diagnostic step guard allows another iteration.

        Returns False if the guard hasn't been started, is finalized,
        or the step limit has been reached.
        """
        return self._diagnostic_guard.can_proceed()

    def record_diagnostic_step(
        self,
        action: str,
        *,
        target: str = "",
        finding: str = "",
        confidence_delta: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Record a diagnostic iteration via the step guard.

        Args:
            action: What was investigated (e.g. "read_failure_log").
            target: The specific target (e.g. file path, test ID).
            finding: What was discovered.
            confidence_delta: Change in confidence from this step.
            metadata: Additional context.

        Returns:
            True if the step was recorded, False if the limit was reached.
        """
        return self._diagnostic_guard.record_step(
            action,
            target=target,
            finding=finding,
            confidence_delta=confidence_delta,
            metadata=metadata,
        )

    def finalize_diagnosis(
        self,
        *,
        reason: CompletionReason | None = None,
        root_cause: str = "",
        confidence: float | None = None,
        proposed_fixes: list[str] | None = None,
        alternative_causes: list[str] | None = None,
        unresolved_questions: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DiagnosisSummary:
        """Finalize the diagnostic session and produce a summary.

        Produces a structured DiagnosisSummary and stores it for handoff.
        Also records the diagnosis in the legacy format for backward
        compatibility.

        Args:
            reason: Why the session ended (auto-detected if None).
            root_cause: The identified root cause.
            confidence: Final confidence score.
            proposed_fixes: Fix suggestions for the user.
            alternative_causes: Other possible explanations.
            unresolved_questions: What could not be determined.
            metadata: Additional context.

        Returns:
            A frozen DiagnosisSummary.
        """
        summary = self._diagnostic_guard.finalize(
            reason=reason,
            root_cause=root_cause,
            confidence=confidence,
            proposed_fixes=proposed_fixes,
            alternative_causes=alternative_causes,
            unresolved_questions=unresolved_questions,
            metadata=metadata,
        )
        self._last_diagnosis_summary = summary

        # Also record in legacy diagnosis format for backward compat
        final_conf = confidence if confidence is not None else self._diagnostic_guard.current_confidence
        self.record_diagnosis(
            root_cause=root_cause,
            evidence=summary.evidence,
            confidence=final_conf,
            proposed_fix=proposed_fixes or [],
            alternative_causes=alternative_causes,
        )

        return summary

    @property
    def diagnostic_budget_status(self) -> str:
        """Human-readable diagnostic step budget status."""
        return self._diagnostic_guard.budget_status_message()

    # ----- Handoff -----

    def get_handoff_summary(self) -> dict[str, Any]:
        """Include step counter, diagnosis, fix proposals, and LLM stats in the handoff summary."""
        summary = super().get_handoff_summary()
        summary["step_budget"] = self._step_counter.summary()
        summary["auto_fix_enabled"] = self._auto_fix_enabled
        summary["llm_calls_made"] = self._fix_generator.llm_calls_made
        summary["has_llm_caller"] = self._fix_generator.has_llm_caller
        if self._diagnosis is not None:
            summary["diagnosis"] = self._diagnosis
        if self._last_diagnosis_summary is not None:
            summary["diagnosis_summary"] = self._last_diagnosis_summary.to_report_dict()
        if self._last_proposals is not None:
            llm_augmented_count = sum(
                1 for p in self._last_proposals.proposals
                if p.metadata.get("llm_augmented")
            )
            summary["fix_proposals"] = {
                "total": self._last_proposals.total_proposals_generated,
                "high_confidence": self._last_proposals.high_confidence_count,
                "actionable": self._last_proposals.actionable_count,
                "budget_exhausted": self._last_proposals.budget_exhausted,
                "analysis_summary": self._last_proposals.analysis_summary,
                "llm_augmented_count": llm_augmented_count,
                "proposals": [
                    {
                        "failure_id": p.failure_id,
                        "title": p.title,
                        "confidence": p.confidence.value,
                        "confidence_score": p.confidence_score,
                        "category": p.category.value,
                        "affected_files": p.affected_files,
                        "change_count": p.change_count,
                        "llm_augmented": p.metadata.get("llm_augmented", False),
                    }
                    for p in self._last_proposals.by_confidence()
                ],
            }
        return summary


def create_troubleshooter_agent(
    config: Config | None = None,
    hard_cap_steps: int = TROUBLESHOOTER_HARD_CAP,
    auto_fix_enabled: bool = False,
) -> Agent:
    """Create an OpenAI Agents SDK Agent instance for troubleshooting.

    This factory wires up the TroubleshooterAgent's tools and instructions
    into an ``agents.Agent`` that the orchestrator can delegate to.

    Args:
        config: Application configuration. Uses defaults if None.
        hard_cap_steps: Max investigation steps before escalation.
                        Defaults to 30.
        auto_fix_enabled: Whether auto-fix mode is enabled. Currently
                          always False (diagnose-only).

    Returns:
        An agents.Agent configured as the troubleshooter sub-agent.
    """
    troubleshooter = TroubleshooterAgent(
        hard_cap_steps=hard_cap_steps,
        auto_fix_enabled=auto_fix_enabled,
    )

    model = config.model_id if config else "gpt-4o"

    agent = Agent(
        name=troubleshooter.name,
        instructions=troubleshooter.instructions,
        tools=troubleshooter.get_tools(),
        model=model,
    )
    return agent
