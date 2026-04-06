"""Discovery agent — explores the project to find and classify tests.

The discovery agent uses confidence-based autonomous exploration:
- Scans directories for test files and scripts
- Reads configuration files to detect frameworks
- Executes --help on discovered tools/scripts
- Classifies findings with confidence scores

It reports back to the orchestrator hub with structured findings
that the executor agent can later use to run tests.

A step counter enforces a hard cap (~20 by default) on investigation
steps per session, preventing unbounded exploration.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, function_tool

from test_runner.agents.base import AgentRole, BaseSubAgent
from test_runner.agents.discovery.step_counter import (
    BUDGET_EXCEEDED_RESPONSE,
    DEFAULT_HARD_CAP,
    StepCounter,
)
from test_runner.agents.discovery.threshold_evaluator import (
    ConfidenceThresholdEvaluator,
    EscalationResult,
    ESCALATION_CONFIDENCE_THRESHOLD,
)
from test_runner.config import Config
from test_runner.models.confidence import ConfidenceModel, ConfidenceSignal
from test_runner.tools.discovery_tools import (
    _detect_frameworks_impl,
    _read_file_impl,
    _run_help_impl,
    _scan_directory_impl,
)


DISCOVERY_INSTRUCTIONS = """\
You are the Discovery Agent for a multi-agent test runner system.

Your job is to explore a project directory and discover all testable artifacts:
- Test files (e.g. test_*.py, *.spec.js, *_test.go)
- Test configuration files (pytest.ini, jest.config.js, etc.)
- Arbitrary test scripts (shell scripts, Makefiles with test targets)
- Documentation about how to run tests

## Investigation Budget

You have a HARD CAP of {hard_cap} investigation steps for this session.
Each tool call counts as one step. Plan your exploration efficiently:
1. Start broad (detect_frameworks, scan_directory on root)
2. Narrow down to specific files only when needed
3. Prioritize high-signal actions (framework detection > file reading)
4. If you approach the budget limit, stop and report what you have found

When the budget is exhausted, tools will return an error and you MUST
immediately summarize your findings.

## Exploration Strategy

Use confidence-based exploration:
1. Start by detecting frameworks in the project root using `detect_frameworks`
2. Scan for test files using `scan_directory` with common patterns
3. Read configuration files to understand test setup
4. Use `run_help` on discovered test commands to learn their options
5. Build a comprehensive map of all testable artifacts

## Confidence Guidelines

- If you find clear framework indicators (pytest.ini, package.json with jest), \
report HIGH confidence (0.8+)
- If you find test-like files but no clear framework config, \
report MEDIUM confidence (0.5-0.8)
- If the project structure is unclear, report LOW confidence (<0.5) \
and escalate to the orchestrator

## Output Format

Always structure your findings as a list of discovered test targets, each with:
- path: where the test lives
- framework: detected framework (or "unknown")
- confidence: your confidence in the classification (0.0-1.0)
- run_command: suggested command to execute the tests (if known)
- notes: any relevant context

Report your overall exploration confidence so the orchestrator can decide \
whether to proceed or request clarification from the user.
"""


def _make_tracked_tools(counter: StepCounter) -> list[Any]:
    """Create discovery tools that track steps via the shared counter.

    Each tool checks the step budget before executing. If the budget is
    exhausted, the tool returns an error response instead of doing work.
    """

    @function_tool
    def scan_directory(
        path: str,
        pattern: str = "*",
        recursive: bool = True,
        max_results: int = 200,
    ) -> dict[str, Any]:
        """Scan a directory for files matching a glob pattern.

        Args:
            path: Directory path to scan (absolute or relative to project root).
            pattern: Glob pattern to match files (e.g. 'test_*.py', '*.spec.js').
            recursive: Whether to scan subdirectories.
            max_results: Maximum number of results to return.

        Returns:
            Dictionary with matched files and metadata.
        """
        if not counter.increment("scan_directory", f"path={path} pattern={pattern}"):
            return {**BUDGET_EXCEEDED_RESPONSE, "budget": counter.summary()}
        result = _scan_directory_impl(path, pattern, recursive, max_results)
        result["_budget"] = counter.budget_status_message()
        return result

    @function_tool
    def read_file(path: str, max_lines: int = 500) -> dict[str, Any]:
        """Read file contents for inspection (docs, configs, test scripts).

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
    def run_help(command: str, timeout_seconds: int = 15) -> dict[str, Any]:
        """Execute a command with --help to discover its usage and options.

        Args:
            command: The command/script to run (e.g. 'pytest', 'npm test').
            timeout_seconds: Timeout for the command execution.

        Returns:
            Dictionary with stdout, stderr, and return code.
        """
        if not counter.increment("run_help", f"command={command}"):
            return {**BUDGET_EXCEEDED_RESPONSE, "budget": counter.summary()}
        result = _run_help_impl(command, timeout_seconds)
        result["_budget"] = counter.budget_status_message()
        return result

    @function_tool
    def detect_frameworks(project_path: str) -> dict[str, Any]:
        """Detect known test frameworks by inspecting project configuration files.

        Args:
            project_path: Root path of the project to inspect.

        Returns:
            Dictionary with detected frameworks and confidence scores.
        """
        if not counter.increment("detect_frameworks", f"path={project_path}"):
            return {**BUDGET_EXCEEDED_RESPONSE, "budget": counter.summary()}
        result = _detect_frameworks_impl(project_path)
        result["_budget"] = counter.budget_status_message()
        return result

    return [scan_directory, read_file, run_help, detect_frameworks]


class DiscoveryAgent(BaseSubAgent):
    """Discovery sub-agent that explores projects for test artifacts.

    Uses four tools:
    - scan_directory: find test files by glob pattern
    - read_file: inspect configs, docs, test contents
    - run_help: execute --help to discover CLI usage
    - detect_frameworks: auto-detect known test frameworks

    A step counter enforces a hard cap on investigation steps.
    The default cap is ~20 steps per session.
    """

    def __init__(
        self,
        hard_cap_steps: int = DEFAULT_HARD_CAP,
        high_threshold: float = 0.80,
        low_threshold: float = 0.40,
        escalation_threshold: float = ESCALATION_CONFIDENCE_THRESHOLD,
    ) -> None:
        super().__init__(
            role=AgentRole.DISCOVERY,
            hard_cap_steps=hard_cap_steps,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
        self._step_counter = StepCounter(hard_cap=hard_cap_steps)
        self._threshold_evaluator = ConfidenceThresholdEvaluator(
            step_counter=self._step_counter,
            confidence_model=ConfidenceModel(
                execute_threshold=high_threshold,
                warn_threshold=escalation_threshold,
            ),
            escalation_threshold=escalation_threshold,
        )
        self._last_escalation: EscalationResult | None = None

    @property
    def step_counter(self) -> StepCounter:
        """The step counter for this discovery session."""
        return self._step_counter

    @property
    def threshold_evaluator(self) -> ConfidenceThresholdEvaluator:
        """The confidence threshold evaluator."""
        return self._threshold_evaluator

    @property
    def last_escalation(self) -> EscalationResult | None:
        """The most recent escalation result, if any."""
        return self._last_escalation

    @property
    def name(self) -> str:
        return "discovery-agent"

    @property
    def instructions(self) -> str:
        return DISCOVERY_INSTRUCTIONS.format(hard_cap=self._step_counter.hard_cap)

    def get_tools(self) -> list[Any]:
        """Return discovery tools wired to the step counter."""
        return _make_tracked_tools(self._step_counter)

    def reset_state(self) -> None:
        """Reset state and step counter for a new session."""
        super().reset_state()
        self._step_counter.reset()

    def should_escalate(self) -> bool:
        """Check if agent should stop — includes step counter exhaustion."""
        if self._step_counter.is_exhausted:
            self.state.escalation_reason = (
                f"Step budget exhausted ({self._step_counter.hard_cap} steps)"
            )
            return True
        return super().should_escalate()

    def evaluate_confidence_at_cap(
        self,
        signals: list[ConfidenceSignal],
    ) -> EscalationResult | None:
        """Evaluate confidence when the step cap is reached.

        This is called by the orchestrator (or internally) when the
        discovery session ends due to budget exhaustion. If confidence
        is below 60%, an EscalationResult is produced targeting either
        the orchestrator or troubleshooter.

        Args:
            signals: All confidence signals collected during discovery.

        Returns:
            An EscalationResult if escalation is needed, None otherwise.
        """
        result = self._threshold_evaluator.check_at_step_cap(signals)
        if result is not None:
            self._last_escalation = result
            self.state.escalation_reason = result.message
        return result

    def check_threshold(
        self,
        signals: list[ConfidenceSignal],
    ):
        """Full threshold check including budget and confidence.

        Returns a ThresholdCheckResult with escalation details if
        the step cap has been reached and confidence is below 60%.
        """
        check = self._threshold_evaluator.evaluate(signals)
        if check.needs_escalation and check.escalation is not None:
            self._last_escalation = check.escalation
            self.state.escalation_reason = check.escalation.message
        return check

    def get_handoff_summary(self) -> dict[str, Any]:
        """Include step counter and escalation info in the handoff summary."""
        summary = super().get_handoff_summary()
        summary["step_budget"] = self._step_counter.summary()
        if self._last_escalation is not None:
            summary["escalation"] = self._last_escalation.summary()
        return summary


def create_discovery_agent(
    config: Config | None = None,
    hard_cap_steps: int = DEFAULT_HARD_CAP,
) -> Agent:
    """Create an OpenAI Agents SDK Agent instance for discovery.

    This factory wires up the DiscoveryAgent's tools and instructions
    into an `agents.Agent` that the orchestrator can delegate to.

    Args:
        config: Application configuration. Uses defaults if None.
        hard_cap_steps: Max autonomous exploration steps before escalation.
                        Defaults to ~20.

    Returns:
        An agents.Agent configured as the discovery sub-agent.
    """
    discovery = DiscoveryAgent(hard_cap_steps=hard_cap_steps)

    model = config.model_id if config else "gpt-4o"

    agent = Agent(
        name=discovery.name,
        instructions=discovery.instructions,
        tools=discovery.get_tools(),
        model=model,
    )
    return agent
