"""Discovery agent — signal collectors, tools, and autonomous exploration."""

from test_runner.agents.discovery.agent import (
    DiscoveryAgent,
    create_discovery_agent,
)
from test_runner.agents.discovery.confidence_tracker import (
    ConfidenceSnapshot,
    ConfidenceTracker,
    ConfidenceTrend,
    TrackingResult,
)
from test_runner.agents.discovery.invocation_confidence import (
    InvocationConfidence,
    InvocationConfidenceScorer,
)
from test_runner.agents.discovery.llm_confidence import (
    DiscoveryContext,
    assess_confidence,
)
from test_runner.agents.discovery.signals import (
    ConfidenceSignalCollector,
    FileExistenceCollector,
    FrameworkDetectionCollector,
    PatternMatchingCollector,
    collect_all_signals,
)
from test_runner.agents.discovery.step_counter import (
    StepCounter,
)
from test_runner.agents.discovery.threshold_evaluator import (
    ConfidenceThresholdEvaluator,
    EscalationResult,
    EscalationReason,
    EscalationTarget,
    ThresholdCheckResult,
    ESCALATION_CONFIDENCE_THRESHOLD,
)

__all__ = [
    "ConfidenceSignalCollector",
    "ConfidenceSnapshot",
    "ConfidenceThresholdEvaluator",
    "ConfidenceTracker",
    "ConfidenceTrend",
    "DiscoveryAgent",
    "DiscoveryContext",
    "EscalationReason",
    "EscalationResult",
    "EscalationTarget",
    "ESCALATION_CONFIDENCE_THRESHOLD",
    "FileExistenceCollector",
    "FrameworkDetectionCollector",
    "InvocationConfidence",
    "InvocationConfidenceScorer",
    "PatternMatchingCollector",
    "StepCounter",
    "ThresholdCheckResult",
    "TrackingResult",
    "assess_confidence",
    "collect_all_signals",
    "create_discovery_agent",
]
