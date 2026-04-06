"""Confidence-based autonomy engine.

Scores discovery findings, decides whether enough information has been
gathered or whether further exploration is needed, and produces a
structured InvocationSpec for the executor agent.

The engine operates in rounds:
1. Collect signals from filesystem-based collectors
2. Aggregate into a confidence score via ConfidenceModel
3. Compare against policy thresholds to decide next action
4. If confident enough, produce an InvocationSpec
5. If not, suggest exploration actions for the next round
6. Hard-cap on rounds forces escalation to orchestrator/user

The engine is stateless between calls — all state is passed in and
returned, so the orchestrator hub can manage it centrally.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from test_runner.agents.parser import TestFramework, TestIntent
from test_runner.autonomy.policy import AutonomyPolicyConfig
from test_runner.models.confidence import (
    CompositeWeights,
    ConfidenceModel,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceTier,
    LLM_SIGNAL_PREFIX,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ExplorationAction(str, enum.Enum):
    """Actions the engine can recommend to the orchestrator."""

    PROCEED = "proceed"            # Enough info — produce invocation spec
    PROCEED_WITH_WARNING = "proceed_with_warning"  # Proceed but warn user
    EXPLORE_FURTHER = "explore_further"  # Need more signals
    ESCALATE = "escalate"          # Hard cap hit or too little info


@dataclass(frozen=True)
class TestTarget:
    """A single discovered test target ready for invocation.

    Attributes:
        framework: Detected test framework.
        path: Path to the test file, directory, or script.
        run_command: Suggested command tokens to execute.
        confidence: Per-target confidence in [0.0, 1.0].
        metadata: Additional context (e.g. config file, markers, notes).
    """

    framework: TestFramework
    path: str
    run_command: list[str]
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvocationSpec:
    """Structured specification for the executor agent.

    Produced by the autonomy engine when confidence is sufficient.
    Contains everything the executor needs to run the tests.

    Attributes:
        targets: Ordered list of test targets to execute.
        intent: The user's original intent (run, list, etc.).
        working_directory: Root directory for execution.
        overall_confidence: Aggregated confidence score.
        confidence_tier: The tier classification.
        environment: Extra environment variables to inject.
        timeout_seconds: Per-target execution timeout.
        metadata: Engine metadata (rounds taken, signal summary, etc.).
    """

    targets: tuple[TestTarget, ...]
    intent: TestIntent
    working_directory: str
    overall_confidence: float
    confidence_tier: ConfidenceTier
    environment: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True if there are no targets to execute."""
        return len(self.targets) == 0

    @property
    def framework_summary(self) -> list[str]:
        """Unique frameworks across all targets."""
        return list(dict.fromkeys(t.framework.value for t in self.targets))

    def summary(self) -> dict[str, Any]:
        """Serializable summary for logging / reporting."""
        return {
            "target_count": len(self.targets),
            "frameworks": self.framework_summary,
            "intent": self.intent.value,
            "working_directory": self.working_directory,
            "overall_confidence": round(self.overall_confidence, 4),
            "confidence_tier": self.confidence_tier.value,
            "timeout_seconds": self.timeout_seconds,
            "targets": [
                {
                    "framework": t.framework.value,
                    "path": t.path,
                    "command": " ".join(t.run_command),
                    "confidence": round(t.confidence, 4),
                }
                for t in self.targets
            ],
        }


@dataclass(frozen=True)
class ExplorationSuggestion:
    """A suggested exploration action for the next round.

    Attributes:
        strategy: Name of the strategy (matches policy exploration_strategies).
        description: Human-readable explanation of what to do.
        parameters: Strategy-specific parameters.
    """

    strategy: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutonomyDecision:
    """The engine's decision after evaluating discovery findings.

    This is the main return type of :meth:`AutonomyEngine.evaluate`.
    The orchestrator inspects `action` to decide the next step.

    Attributes:
        action: What to do next (proceed, explore, escalate).
        confidence_result: Full confidence evaluation details.
        invocation_spec: Populated when action is PROCEED or PROCEED_WITH_WARNING.
        suggestions: Exploration suggestions when action is EXPLORE_FURTHER.
        round_number: Which exploration round produced this decision.
        reason: Human-readable explanation of why this action was chosen.
    """

    action: ExplorationAction
    confidence_result: ConfidenceResult
    invocation_spec: InvocationSpec | None = None
    suggestions: tuple[ExplorationSuggestion, ...] = ()
    round_number: int = 1
    reason: str = ""

    @property
    def should_proceed(self) -> bool:
        """True if the engine decided there's enough information."""
        return self.action in (
            ExplorationAction.PROCEED,
            ExplorationAction.PROCEED_WITH_WARNING,
        )

    @property
    def needs_exploration(self) -> bool:
        """True if the engine wants more signals."""
        return self.action == ExplorationAction.EXPLORE_FURTHER

    @property
    def needs_escalation(self) -> bool:
        """True if the engine is giving up and needs human help."""
        return self.action == ExplorationAction.ESCALATE

    def summary(self) -> dict[str, Any]:
        """Serializable summary for logging / reporting."""
        result: dict[str, Any] = {
            "action": self.action.value,
            "round": self.round_number,
            "reason": self.reason,
            "confidence": self.confidence_result.summary(),
        }
        if self.invocation_spec is not None:
            result["invocation_spec"] = self.invocation_spec.summary()
        if self.suggestions:
            result["suggestions"] = [
                {"strategy": s.strategy, "description": s.description}
                for s in self.suggestions
            ]
        return result


# ---------------------------------------------------------------------------
# Discovery findings input
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryFindings:
    """Structured discovery output that the engine evaluates.

    Populated by the discovery agent or signal collectors and fed
    into the autonomy engine for scoring.

    Attributes:
        signals: Raw confidence signals from collectors.
        frameworks_detected: Framework names with per-framework confidence.
        test_files: Paths to discovered test files.
        config_files: Config files found (path -> framework hint).
        scripts: Executable scripts/commands found.
        working_directory: The scanned project root.
        exploration_round: Which round produced these findings.
        raw_agent_output: Optional free-form text from the discovery agent.
    """

    signals: list[ConfidenceSignal] = field(default_factory=list)
    frameworks_detected: list[dict[str, Any]] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    config_files: dict[str, str] = field(default_factory=dict)
    scripts: list[str] = field(default_factory=list)
    working_directory: str = ""
    exploration_round: int = 1
    raw_agent_output: str = ""

    @property
    def positive_signal_count(self) -> int:
        """Count of signals with score > 0."""
        return sum(1 for s in self.signals if s.score > 0)

    @property
    def has_framework(self) -> bool:
        """True if at least one framework was detected."""
        return len(self.frameworks_detected) > 0

    @property
    def primary_framework(self) -> str | None:
        """The framework with highest confidence, if any."""
        if not self.frameworks_detected:
            return None
        best = max(self.frameworks_detected, key=lambda f: f.get("confidence", 0))
        return best.get("framework")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AutonomyEngine:
    """Confidence-based autonomy engine for test discovery.

    Evaluates discovery findings against a configurable policy and
    produces either an InvocationSpec (when confident) or exploration
    suggestions (when more information is needed).

    The engine is stateless — the orchestrator feeds findings in and
    receives decisions back, managing round progression externally.

    Example::

        engine = AutonomyEngine()
        decision = engine.evaluate(findings)
        if decision.should_proceed:
            spec = decision.invocation_spec
            # hand off to executor
        elif decision.needs_exploration:
            # run suggested exploration actions, collect more findings
            ...
        else:
            # escalate to user
            ...
    """

    def __init__(
        self,
        policy: AutonomyPolicyConfig | None = None,
        *,
        composite_weights: CompositeWeights | None = None,
        llm_signal_prefix: str = LLM_SIGNAL_PREFIX,
    ) -> None:
        self._policy = policy or AutonomyPolicyConfig()
        self._composite_weights = composite_weights
        self._llm_signal_prefix = llm_signal_prefix
        self._confidence_model = ConfidenceModel(
            execute_threshold=self._policy.execute_threshold,
            warn_threshold=self._policy.warn_threshold,
            composite_weights=composite_weights,
            llm_signal_prefix=llm_signal_prefix,
        )

    @property
    def policy(self) -> AutonomyPolicyConfig:
        """The active autonomy policy."""
        return self._policy

    @property
    def confidence_model(self) -> ConfidenceModel:
        """The underlying confidence model."""
        return self._confidence_model

    # -- Signal scoring ----------------------------------------------------

    def _score_signals(
        self, signals: Sequence[ConfidenceSignal]
    ) -> ConfidenceResult:
        """Score signals using composite evaluation when LLM signals present.

        Automatically switches between flat and composite evaluation based
        on whether any signals carry the LLM signal prefix. This allows
        the engine to leverage the evidence/LLM blend weights when an LLM
        self-assessment has been performed, and fall back to flat averaging
        when only filesystem-based signals are available.
        """
        has_llm = any(
            s.name.startswith(self._llm_signal_prefix) for s in signals
        )
        if has_llm:
            return self._confidence_model.evaluate_composite(signals)
        return self._confidence_model.evaluate(signals)

    def score_findings(
        self, findings: DiscoveryFindings
    ) -> ConfidenceResult:
        """Score discovery findings without making a proceed/explore decision.

        Useful for the orchestrator to check progress mid-exploration
        without triggering the full decision logic.

        Args:
            findings: Discovery output to score.

        Returns:
            A ConfidenceResult with the aggregated score and tier.
        """
        return self._score_signals(findings.signals)

    # -- Main evaluation API -----------------------------------------------

    def evaluate(
        self,
        findings: DiscoveryFindings,
        *,
        intent: TestIntent = TestIntent.RUN,
    ) -> AutonomyDecision:
        """Evaluate discovery findings and decide the next action.

        Args:
            findings: Structured discovery output to score.
            intent: The user's intent (used when building InvocationSpec).

        Returns:
            An AutonomyDecision with the recommended action and
            either an InvocationSpec or exploration suggestions.
        """
        # 1. Score signals — use composite evaluation when LLM signals present
        confidence_result = self._score_signals(findings.signals)

        logger.info(
            "Autonomy engine round %d: score=%.4f tier=%s (%d signals, %d positive)",
            findings.exploration_round,
            confidence_result.score,
            confidence_result.tier.value,
            len(findings.signals),
            findings.positive_signal_count,
        )

        # 2. Check hard cap on exploration rounds
        if findings.exploration_round > self._policy.max_exploration_rounds:
            return self._make_escalation(
                confidence_result,
                findings,
                reason=(
                    f"Hard cap reached: {findings.exploration_round} rounds "
                    f"exceeds max {self._policy.max_exploration_rounds}"
                ),
            )

        # 3. Check minimum positive signals
        if findings.positive_signal_count < self._policy.min_positive_signals:
            if findings.exploration_round >= self._policy.max_exploration_rounds:
                return self._make_escalation(
                    confidence_result,
                    findings,
                    reason=(
                        f"Insufficient signals: {findings.positive_signal_count} "
                        f"positive signals found, need {self._policy.min_positive_signals}"
                    ),
                )
            return self._make_explore_further(
                confidence_result,
                findings,
                reason=(
                    f"Only {findings.positive_signal_count} positive signals; "
                    f"need at least {self._policy.min_positive_signals}"
                ),
            )

        # 4. Check framework requirement
        if self._policy.require_framework_detection and not findings.has_framework:
            if findings.exploration_round >= self._policy.max_exploration_rounds:
                if self._policy.allow_script_fallback and findings.scripts:
                    # Fall back to script execution
                    logger.info("No framework detected but scripts found; using script fallback")
                else:
                    return self._make_escalation(
                        confidence_result,
                        findings,
                        reason="No framework detected and policy requires framework detection",
                    )
            else:
                return self._make_explore_further(
                    confidence_result,
                    findings,
                    reason="Policy requires framework detection but none found yet",
                )

        # 5. Decide based on confidence tier
        if confidence_result.tier == ConfidenceTier.HIGH:
            return self._make_proceed(
                confidence_result,
                findings,
                intent=intent,
                warn=False,
            )
        elif confidence_result.tier == ConfidenceTier.MEDIUM:
            # Medium confidence: proceed with warning if we've explored enough,
            # otherwise try one more round
            if findings.exploration_round >= self._policy.max_exploration_rounds:
                return self._make_proceed(
                    confidence_result,
                    findings,
                    intent=intent,
                    warn=True,
                )
            # Check if score is close to execute threshold — if so, explore
            gap = self._policy.execute_threshold - confidence_result.score
            if gap <= 0.10:
                # Close enough — one more round might push it over
                return self._make_explore_further(
                    confidence_result,
                    findings,
                    reason=(
                        f"Score {confidence_result.score:.2f} is close to "
                        f"execute threshold {self._policy.execute_threshold:.2f}; "
                        f"one more round may suffice"
                    ),
                )
            else:
                return self._make_proceed(
                    confidence_result,
                    findings,
                    intent=intent,
                    warn=True,
                )

        else:
            # LOW confidence
            if findings.exploration_round >= self._policy.max_exploration_rounds:
                return self._make_escalation(
                    confidence_result,
                    findings,
                    reason=(
                        f"Low confidence ({confidence_result.score:.2f}) after "
                        f"{findings.exploration_round} rounds"
                    ),
                )
            return self._make_explore_further(
                confidence_result,
                findings,
                reason=f"Low confidence ({confidence_result.score:.2f}); exploring further",
            )

    # -- InvocationSpec construction ---------------------------------------

    def build_invocation_spec(
        self,
        findings: DiscoveryFindings,
        confidence_result: ConfidenceResult | None = None,
        intent: TestIntent = TestIntent.RUN,
    ) -> InvocationSpec:
        """Build a structured InvocationSpec from discovery findings.

        This is called internally by evaluate() when proceeding, but
        can also be called directly by the orchestrator.

        Args:
            findings: The discovery findings to convert.
            confidence_result: The confidence evaluation. If None, the
                engine will score the findings automatically.
            intent: The user's test intent.

        Returns:
            An InvocationSpec ready for the executor agent.
        """
        if confidence_result is None:
            confidence_result = self.score_findings(findings)

        targets = self._build_targets(findings)

        return InvocationSpec(
            targets=tuple(targets),
            intent=intent,
            working_directory=findings.working_directory,
            overall_confidence=confidence_result.score,
            confidence_tier=confidence_result.tier,
            metadata={
                "exploration_round": findings.exploration_round,
                "signal_count": len(findings.signals),
                "positive_signal_count": findings.positive_signal_count,
                "frameworks_detected": [
                    f.get("framework", "unknown")
                    for f in findings.frameworks_detected
                ],
                "confidence_summary": confidence_result.summary(),
            },
        )

    # -- Target building ---------------------------------------------------

    def _build_targets(self, findings: DiscoveryFindings) -> list[TestTarget]:
        """Convert discovery findings into executable TestTargets."""
        targets: list[TestTarget] = []

        # Group test files by framework
        framework_files: dict[str, list[str]] = {}
        for fw in findings.frameworks_detected:
            fw_name = fw.get("framework", "unknown")
            if fw_name not in framework_files:
                framework_files[fw_name] = []

        # Assign test files to detected frameworks
        for test_file in findings.test_files:
            assigned = False
            for fw_name in framework_files:
                if self._file_matches_framework(test_file, fw_name):
                    framework_files[fw_name].append(test_file)
                    assigned = True
                    break
            if not assigned:
                # Put unassigned files under the primary framework or unknown
                primary = findings.primary_framework or "unknown"
                framework_files.setdefault(primary, []).append(test_file)

        # Build per-framework targets
        for fw_name, files in framework_files.items():
            tf = self._map_framework(fw_name)
            fw_confidence = self._get_framework_confidence(fw_name, findings)
            cmd = self._build_framework_command(tf, findings.working_directory)

            targets.append(
                TestTarget(
                    framework=tf,
                    path=findings.working_directory,
                    run_command=cmd,
                    confidence=fw_confidence,
                    metadata={
                        "test_file_count": len(files),
                        "sample_files": files[:5],
                    },
                )
            )

        # Add script targets
        for script in findings.scripts:
            targets.append(
                TestTarget(
                    framework=TestFramework.SCRIPT,
                    path=script,
                    run_command=[script],
                    confidence=0.7,
                    metadata={"source": "script_discovery"},
                )
            )

        # If no targets found but we have a working directory, create a
        # best-effort target from the primary framework
        if not targets and findings.working_directory:
            primary = findings.primary_framework
            if primary:
                tf = self._map_framework(primary)
                targets.append(
                    TestTarget(
                        framework=tf,
                        path=findings.working_directory,
                        run_command=self._build_framework_command(
                            tf, findings.working_directory
                        ),
                        confidence=0.5,
                        metadata={"source": "fallback_from_primary_framework"},
                    )
                )

        return targets

    @staticmethod
    def _file_matches_framework(filepath: str, framework: str) -> bool:
        """Heuristic: does a test file likely belong to this framework?"""
        lower = filepath.lower()
        match framework:
            case "pytest" | "unittest":
                return lower.endswith(".py")
            case "jest" | "mocha" | "vitest":
                return lower.endswith((".js", ".ts", ".jsx", ".tsx"))
            case "go_test":
                return lower.endswith("_test.go")
            case "cargo_test":
                return lower.endswith(".rs")
            case _:
                return False

    @staticmethod
    def _map_framework(name: str) -> TestFramework:
        """Map a framework name string to a TestFramework enum."""
        mapping = {
            "pytest": TestFramework.PYTEST,
            "unittest": TestFramework.UNITTEST,
            "jest": TestFramework.JEST,
            "mocha": TestFramework.MOCHA,
            "vitest": TestFramework.JEST,  # vitest uses same runner pattern
            "go_test": TestFramework.GO_TEST,
            "cargo_test": TestFramework.CARGO_TEST,
            "dotnet_test": TestFramework.DOTNET_TEST,
            "shell_scripts": TestFramework.SCRIPT,
        }
        return mapping.get(name, TestFramework.AUTO_DETECT)

    @staticmethod
    def _get_framework_confidence(
        framework: str, findings: DiscoveryFindings
    ) -> float:
        """Look up the confidence for a specific framework from findings."""
        for fw in findings.frameworks_detected:
            if fw.get("framework") == framework:
                return float(fw.get("confidence", 0.5))
        return 0.5

    @staticmethod
    def _build_framework_command(
        framework: TestFramework, working_dir: str
    ) -> list[str]:
        """Build a default run command for a framework."""
        match framework:
            case TestFramework.PYTEST:
                return ["pytest"]
            case TestFramework.UNITTEST:
                return ["python", "-m", "unittest", "discover"]
            case TestFramework.JEST:
                return ["npx", "jest"]
            case TestFramework.MOCHA:
                return ["npx", "mocha"]
            case TestFramework.GO_TEST:
                return ["go", "test", "./..."]
            case TestFramework.CARGO_TEST:
                return ["cargo", "test"]
            case TestFramework.DOTNET_TEST:
                return ["dotnet", "test"]
            case TestFramework.SCRIPT:
                return ["bash", "-c", "echo 'no script specified'"]
            case _:
                return ["echo", "unknown framework"]

    # -- Decision builders -------------------------------------------------

    def _make_proceed(
        self,
        confidence_result: ConfidenceResult,
        findings: DiscoveryFindings,
        *,
        intent: TestIntent,
        warn: bool,
    ) -> AutonomyDecision:
        """Build a PROCEED or PROCEED_WITH_WARNING decision."""
        spec = self.build_invocation_spec(findings, confidence_result, intent)
        action = (
            ExplorationAction.PROCEED_WITH_WARNING
            if warn
            else ExplorationAction.PROCEED
        )
        reason = (
            f"Confidence {confidence_result.score:.2f} "
            f"({'≥' if not warn else '<'} execute threshold "
            f"{self._policy.execute_threshold:.2f})"
        )
        if warn:
            reason += "; proceeding with warning"

        logger.info("Decision: %s — %s", action.value, reason)
        return AutonomyDecision(
            action=action,
            confidence_result=confidence_result,
            invocation_spec=spec,
            round_number=findings.exploration_round,
            reason=reason,
        )

    def _make_explore_further(
        self,
        confidence_result: ConfidenceResult,
        findings: DiscoveryFindings,
        reason: str,
    ) -> AutonomyDecision:
        """Build an EXPLORE_FURTHER decision with suggestions."""
        suggestions = self._generate_suggestions(findings)

        logger.info(
            "Decision: explore_further — %s (%d suggestions)",
            reason,
            len(suggestions),
        )
        return AutonomyDecision(
            action=ExplorationAction.EXPLORE_FURTHER,
            confidence_result=confidence_result,
            suggestions=tuple(suggestions),
            round_number=findings.exploration_round,
            reason=reason,
        )

    def _make_escalation(
        self,
        confidence_result: ConfidenceResult,
        findings: DiscoveryFindings,
        reason: str,
    ) -> AutonomyDecision:
        """Build an ESCALATE decision."""
        logger.info("Decision: escalate — %s", reason)

        # Still build a best-effort spec if possible
        spec = None
        if findings.has_framework or findings.test_files:
            spec = self.build_invocation_spec(
                findings, confidence_result, TestIntent.RUN
            )

        return AutonomyDecision(
            action=ExplorationAction.ESCALATE,
            confidence_result=confidence_result,
            invocation_spec=spec,
            round_number=findings.exploration_round,
            reason=reason,
        )

    # -- Exploration suggestions -------------------------------------------

    def _generate_suggestions(
        self,
        findings: DiscoveryFindings,
    ) -> list[ExplorationSuggestion]:
        """Generate exploration suggestions based on what's missing."""
        suggestions: list[ExplorationSuggestion] = []
        used_strategies = set()

        # Suggest deep pattern scan if few test files found
        if len(findings.test_files) < 5 and "deep_pattern_scan" in self._policy.exploration_strategies:
            suggestions.append(
                ExplorationSuggestion(
                    strategy="deep_pattern_scan",
                    description="Scan deeper directories for test files using additional glob patterns",
                    parameters={
                        "patterns": [
                            "**/test_*.py", "**/*_test.py",
                            "**/*.spec.ts", "**/*.test.js",
                            "**/*_test.go", "**/Test*.java",
                        ],
                        "max_depth": 10,
                    },
                )
            )
            used_strategies.add("deep_pattern_scan")

        # Suggest config inspection if no framework detected
        if not findings.has_framework and "config_file_inspection" in self._policy.exploration_strategies:
            suggestions.append(
                ExplorationSuggestion(
                    strategy="config_file_inspection",
                    description="Read config files (pyproject.toml, package.json, etc.) for test framework clues",
                    parameters={
                        "files_to_check": [
                            "pyproject.toml", "package.json", "Makefile",
                            "tox.ini", "setup.cfg", "go.mod", "Cargo.toml",
                        ],
                    },
                )
            )
            used_strategies.add("config_file_inspection")

        # Suggest help probe if frameworks detected but commands unclear
        if findings.has_framework and "help_probe" in self._policy.exploration_strategies:
            for fw in findings.frameworks_detected:
                fw_name = fw.get("framework", "")
                cmd = self._help_command_for_framework(fw_name)
                if cmd:
                    suggestions.append(
                        ExplorationSuggestion(
                            strategy="help_probe",
                            description=f"Run '{cmd} --help' to learn available options",
                            parameters={"command": cmd, "framework": fw_name},
                        )
                    )
            used_strategies.add("help_probe")

        # Suggest Makefile scan
        if "makefile_target_scan" in self._policy.exploration_strategies and "makefile_target_scan" not in used_strategies:
            if not any(
                f.get("framework") == "make"
                for f in findings.frameworks_detected
            ):
                suggestions.append(
                    ExplorationSuggestion(
                        strategy="makefile_target_scan",
                        description="Scan Makefile/Justfile for test-related targets",
                        parameters={
                            "files": ["Makefile", "makefile", "Justfile", "justfile"],
                            "target_patterns": ["test", "check", "verify", "spec"],
                        },
                    )
                )

        return suggestions

    @staticmethod
    def _help_command_for_framework(framework: str) -> str | None:
        """Return the help command for a framework."""
        mapping = {
            "pytest": "pytest",
            "unittest": "python -m unittest",
            "jest": "npx jest",
            "mocha": "npx mocha",
            "vitest": "npx vitest",
            "go_test": "go test",
            "cargo_test": "cargo test",
            "dotnet_test": "dotnet test",
        }
        return mapping.get(framework)
