"""Post-execution confidence modeling package.

Provides evidence-based signal collectors that analyze test execution
results (exit codes, output patterns, timing, assertion counts) and
translate them into :class:`~test_runner.models.confidence.ConfidenceSignal`
instances for the confidence model.

Primary entry point::

    from test_runner.confidence.signals import collect_execution_signals

    signals = collect_execution_signals(evidence)
    result = confidence_model.evaluate(signals)
"""

from test_runner.confidence.signals import (
    ExecutionEvidence,
    ExecutionSignalCollector,
    ExitCodeSignalCollector,
    OutputPatternSignalCollector,
    TimingSignalCollector,
    AssertionCountSignalCollector,
    InfrastructureHealthSignalCollector,
    collect_execution_signals,
)

__all__ = [
    "ExecutionEvidence",
    "ExecutionSignalCollector",
    "ExitCodeSignalCollector",
    "OutputPatternSignalCollector",
    "TimingSignalCollector",
    "AssertionCountSignalCollector",
    "InfrastructureHealthSignalCollector",
    "collect_execution_signals",
]
