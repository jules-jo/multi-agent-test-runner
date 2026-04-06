"""Troubleshooter sub-agent for diagnosing test failures and proposing fixes.

Provides both pattern-based and AI-powered (LLM-augmented) failure analysis:
- FailureAnalyzer: Deterministic pattern-based classification and proposals
- FixGenerator: LLM-augmented analysis for deeper root-cause investigation
- TroubleshooterAgent: Orchestrator-facing agent with step budget enforcement
- FailureAnalysisReport: Aggregated failure analysis with categorization and
  log excerpts for cross-failure insights
"""

from test_runner.agents.troubleshooter.agent import (
    TroubleshooterAgent,
    create_troubleshooter_agent,
)
from test_runner.agents.troubleshooter.analyzer import (
    AnalyzerConfig,
    FailureAnalyzer,
    StrategyRegistry,
    classify_failure,
    create_default_registry,
)
from test_runner.agents.troubleshooter.failure_analysis import (
    CategorizedFailure,
    ExcerptConfig,
    FailureAnalysisConfig,
    FailureAnalysisReport,
    FailureGroup,
    LogExcerpt,
    analyze_failures,
    extract_failure_excerpts,
    normalize_error_pattern,
)
from test_runner.agents.troubleshooter.fix_generator import (
    FixGenerator,
    FixGeneratorConfig,
    LLMAnalysisResult,
    build_analysis_prompt,
    merge_analysis,
    parse_llm_response,
)
from test_runner.agents.troubleshooter.models import (
    FailureCategory,
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.agents.troubleshooter.diagnostic_loop import (
    ActionResult,
    ActionType,
    DiagnosticAction,
    DiagnosticLoop,
    DiagnosticLoopConfig,
    LoopExitReason,
    LoopResult,
)
from test_runner.agents.troubleshooter.safety_guard import (
    MutationPolicy,
    ReadOnlySafetyGuard,
    SafetyGuardConfig,
    SafetyViolation,
    ViolationType,
)
from test_runner.agents.troubleshooter.step_guard import (
    CompletionReason,
    DiagnosticStep,
    DiagnosticStepGuard,
    DiagnosisSummary,
)

__all__ = [
    "ActionResult",
    "ActionType",
    "AnalyzerConfig",
    "CategorizedFailure",
    "CompletionReason",
    "DiagnosticAction",
    "DiagnosticLoop",
    "DiagnosticLoopConfig",
    "DiagnosticStep",
    "DiagnosticStepGuard",
    "DiagnosisSummary",
    "ExcerptConfig",
    "FailureAnalysisConfig",
    "FailureAnalysisReport",
    "FailureAnalyzer",
    "FailureCategory",
    "FailureGroup",
    "FixConfidence",
    "FixGenerator",
    "FixGeneratorConfig",
    "FixProposal",
    "FixProposalSet",
    "LLMAnalysisResult",
    "LogExcerpt",
    "LoopExitReason",
    "LoopResult",
    "MutationPolicy",
    "ProposedChange",
    "ReadOnlySafetyGuard",
    "SafetyGuardConfig",
    "SafetyViolation",
    "ViolationType",
    "StrategyRegistry",
    "TroubleshooterAgent",
    "analyze_failures",
    "build_analysis_prompt",
    "classify_failure",
    "create_default_registry",
    "create_troubleshooter_agent",
    "extract_failure_excerpts",
    "merge_analysis",
    "normalize_error_pattern",
    "parse_llm_response",
]
