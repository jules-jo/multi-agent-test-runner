"""Evidence-based signal collectors for post-execution confidence modeling.

These collectors analyze the raw evidence produced by a test execution
(exit codes, stdout/stderr output, timing, assertion counts) and translate
it into :class:`~test_runner.models.confidence.ConfidenceSignal` instances
that feed the confidence model.

The confidence model aggregates these signals into a tier decision (EXECUTE /
WARN / INVESTIGATE) that guides the orchestrator's next routing decision after
a test run completes.

Collector taxonomy
------------------
* **ExitCodeSignalCollector** — interprets exit codes (0 = pass, framework-
  specific codes for partial failure, error codes for infrastructure issues).
* **OutputPatternSignalCollector** — scans stdout/stderr for test outcome
  phrases recognized across major frameworks (pytest, jest, go test, etc.).
* **TimingSignalCollector** — uses wall-clock execution time as a signal
  (suspiciously fast = nothing ran; near/over timeout = reliability risk).
* **AssertionCountSignalCollector** — extracts test/assertion counts from
  output and computes a pass-rate-based confidence score.
* **InfrastructureHealthSignalCollector** — detects infrastructure-level
  errors (command not found, import errors, permission denied) in stderr.

All collectors accept an :class:`ExecutionEvidence` data object which wraps
the raw execution evidence in a framework-agnostic form.

Usage::

    from test_runner.confidence.signals import (
        ExecutionEvidence,
        collect_execution_signals,
    )

    evidence = ExecutionEvidence(
        exit_code=0,
        stdout="5 passed in 1.23s",
        stderr="",
        duration_seconds=1.23,
        command="pytest tests/",
        timed_out=False,
    )
    signals = collect_execution_signals(evidence)
    result = confidence_model.evaluate(signals)
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from test_runner.models.confidence import ConfidenceSignal


# ---------------------------------------------------------------------------
# ExecutionEvidence — framework-agnostic container for raw execution data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionEvidence:
    """Framework-agnostic container for raw test execution evidence.

    This lightweight value object wraps the essential evidence produced
    by any execution target (local, Docker, remote CI) and serves as the
    common input to all post-execution signal collectors.

    Attributes:
        exit_code: Process exit code.  0 indicates success; non-zero indicates
            failure.  -1 is used for infrastructure errors (command not found,
            OS errors) and timeouts.
        stdout: Full standard output captured from the test process.
        stderr: Full standard error captured from the test process.
        duration_seconds: Wall-clock execution time in seconds.
        command: Human-readable command string (for evidence metadata).
        timed_out: True if the process was killed due to a timeout.
        framework: Optional test framework hint (e.g. "pytest", "jest").
            Used by pattern-matching collectors to apply framework-specific
            patterns with higher priority.
        metadata: Any additional target-specific data (e.g. Docker image,
            CI run URL) forwarded verbatim into signal evidence dicts.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    command: str = ""
    timed_out: bool = False
    framework: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_execution_result(cls, result: Any) -> "ExecutionEvidence":
        """Construct from an ``ExecutionResult`` (execution.targets module).

        This factory avoids a hard import dependency on the execution layer,
        keeping the confidence package decoupled from execution internals.

        Args:
            result: An ``ExecutionResult`` instance.

        Returns:
            An ``ExecutionEvidence`` with data extracted from *result*.
        """
        timed_out = getattr(result, "status", None) is not None and str(
            getattr(result.status, "value", result.status)
        ) == "timeout"

        return cls(
            exit_code=getattr(result, "exit_code", -1),
            stdout=getattr(result, "stdout", ""),
            stderr=getattr(result, "stderr", ""),
            duration_seconds=getattr(result, "duration_seconds", 0.0),
            command=getattr(result, "command_display", ""),
            timed_out=timed_out,
            metadata=dict(getattr(result, "metadata", {})),
        )

    @property
    def combined_output(self) -> str:
        """Convenience: stdout + newline + stderr for single-pass scanning."""
        parts = [self.stdout, self.stderr]
        return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ExecutionSignalCollector(abc.ABC):
    """Base class for all post-execution signal collectors.

    Subclasses implement ``collect(evidence)`` which inspects the execution
    evidence and returns a list of :class:`ConfidenceSignal` instances.
    """

    @abc.abstractmethod
    def collect(self, evidence: ExecutionEvidence) -> list[ConfidenceSignal]:
        """Analyse *evidence* and return zero or more confidence signals."""


# ---------------------------------------------------------------------------
# Exit-code signal collector
# ---------------------------------------------------------------------------

# Exit-code taxonomy used by major test frameworks:
#
# 0  — All tests passed (universal).
# 1  — Tests ran, at least one failed (pytest, jest, go test, cargo test,
#       JUnit via Maven/Gradle).
# 2  — Usage / configuration error or interrupted (pytest).
# 3  — Internal pytest error.
# 4  — Command-line usage error (pytest).
# 5  — No tests collected (pytest).
# 127 — Command not found (shell).
# -1  — Infrastructure error or timeout (set by LocalTarget/DockerTarget).

_EXIT_CODE_SCORES: dict[int, tuple[str, float, float]] = {
    # (signal_suffix, score, weight)
    0: ("success", 1.0, 0.95),
    1: ("test_failures", 0.35, 0.90),
    2: ("config_error", 0.10, 0.85),
    3: ("internal_error", 0.05, 0.85),
    4: ("usage_error", 0.10, 0.80),
    5: ("no_tests_collected", 0.20, 0.75),
    127: ("command_not_found", 0.0, 0.95),
    -1: ("infrastructure_error", 0.0, 0.90),
}

# Score for any other non-zero exit code (e.g. segfault, OOM, etc.)
_UNKNOWN_NONZERO_SCORE = 0.15
_UNKNOWN_NONZERO_WEIGHT = 0.70


class ExitCodeSignalCollector(ExecutionSignalCollector):
    """Translate the process exit code into a confidence signal.

    The exit code is the most authoritative single indicator of test
    outcome.  It receives the highest weight among all post-execution
    signals.

    Scores:
        - 0   → 1.0 (all tests passed)
        - 1   → 0.35 (some tests failed — partial success)
        - 2-4 → 0.05–0.10 (configuration / usage errors)
        - 5   → 0.20 (no tests collected — low but recoverable)
        - 127 → 0.0 (command not found — likely misconfiguration)
        - -1  → 0.0 (infrastructure failure / timeout)

    Custom *code_scores* can be supplied to extend or override defaults,
    useful for framework-specific exit codes (e.g. PHPUnit returns 2 for
    a successful run with warnings).
    """

    def __init__(
        self,
        code_scores: dict[int, tuple[str, float, float]] | None = None,
    ) -> None:
        self._code_scores = dict(_EXIT_CODE_SCORES)
        if code_scores:
            self._code_scores.update(code_scores)

    def collect(self, evidence: ExecutionEvidence) -> list[ConfidenceSignal]:
        code = evidence.exit_code

        if code in self._code_scores:
            suffix, score, weight = self._code_scores[code]
        else:
            # Unknown non-zero code: treat as a soft failure
            suffix = "unknown_failure"
            score = _UNKNOWN_NONZERO_SCORE
            weight = _UNKNOWN_NONZERO_WEIGHT

        # Timeout always overrides the exit code — it's an infrastructure issue
        if evidence.timed_out:
            suffix = "timeout"
            score = 0.0
            weight = 0.95

        return [
            ConfidenceSignal(
                name=f"exit_code_{suffix}",
                weight=weight,
                score=score,
                evidence={
                    "exit_code": code,
                    "timed_out": evidence.timed_out,
                    "command": evidence.command,
                    "interpretation": suffix,
                },
            )
        ]


# ---------------------------------------------------------------------------
# Output-pattern signal collector
# ---------------------------------------------------------------------------

# Each entry: (pattern, signal_name, weight, score_if_matched)
# Patterns are applied to combined stdout+stderr.
# A positive match (evidence of passing tests) scores high;
# a negative match (evidence of errors) scores low.

@dataclass(frozen=True)
class _OutputPattern:
    """Descriptor for a single output pattern probe."""

    pattern: str
    signal_name: str
    weight: float
    score_if_matched: float
    frameworks: frozenset[str] = field(default_factory=frozenset)
    # If True, score_if_not_matched = 1 - score_if_matched
    invert_on_absence: bool = False

    def matches(self, text: str) -> bool:
        return bool(re.search(self.pattern, text, re.IGNORECASE | re.MULTILINE))


_DEFAULT_OUTPUT_PATTERNS: list[_OutputPattern] = [
    # ------------------------------------------------------------------ #
    # Positive — evidence of successful test execution                    #
    # ------------------------------------------------------------------ #

    # pytest: "X passed" or "X passed, Y skipped"
    _OutputPattern(
        pattern=r"\b\d+\s+passed\b",
        signal_name="output_tests_passed",
        weight=0.85,
        score_if_matched=1.0,
        frameworks=frozenset({"pytest"}),
    ),
    # jest / vitest: "Tests: N passed, N total" or "✓ N tests passed"
    _OutputPattern(
        pattern=r"(?:Tests|tests):\s*\d+\s+passed",
        signal_name="output_jest_tests_passed",
        weight=0.85,
        score_if_matched=1.0,
        frameworks=frozenset({"jest", "vitest"}),
    ),
    # go test: "ok  github.com/..." or "PASS" (allow trailing content like timing)
    _OutputPattern(
        pattern=r"^(?:ok\s+\S+|^PASS)\b",
        signal_name="output_go_pass",
        weight=0.85,
        score_if_matched=1.0,
        frameworks=frozenset({"go"}),
    ),
    # cargo test: "test result: ok."
    _OutputPattern(
        pattern=r"test result:\s+ok\b",
        signal_name="output_cargo_ok",
        weight=0.85,
        score_if_matched=1.0,
        frameworks=frozenset({"cargo"}),
    ),
    # JUnit / Maven: "BUILD SUCCESS"
    _OutputPattern(
        pattern=r"BUILD\s+SUCCESS",
        signal_name="output_maven_build_success",
        weight=0.80,
        score_if_matched=1.0,
        frameworks=frozenset({"maven", "gradle"}),
    ),
    # Generic: "All tests passed"
    _OutputPattern(
        pattern=r"all\s+tests?\s+passed",
        signal_name="output_all_passed",
        weight=0.75,
        score_if_matched=1.0,
    ),

    # ------------------------------------------------------------------ #
    # Partial — mixed pass/fail evidence                                  #
    # ------------------------------------------------------------------ #

    # pytest: "X failed"
    _OutputPattern(
        pattern=r"\b\d+\s+failed\b",
        signal_name="output_tests_failed",
        weight=0.80,
        score_if_matched=0.20,
        frameworks=frozenset({"pytest"}),
    ),
    # jest: "Tests: N failed"
    _OutputPattern(
        pattern=r"(?:Tests|tests):\s*\d+\s+failed",
        signal_name="output_jest_tests_failed",
        weight=0.80,
        score_if_matched=0.20,
        frameworks=frozenset({"jest", "vitest"}),
    ),
    # go test: "FAIL"
    _OutputPattern(
        pattern=r"^FAIL\b",
        signal_name="output_go_fail",
        weight=0.80,
        score_if_matched=0.20,
        frameworks=frozenset({"go"}),
    ),
    # cargo test: "test result: FAILED"
    _OutputPattern(
        pattern=r"test result:\s+FAILED\b",
        signal_name="output_cargo_failed",
        weight=0.80,
        score_if_matched=0.20,
        frameworks=frozenset({"cargo"}),
    ),
    # JUnit / Maven: "BUILD FAILURE"
    _OutputPattern(
        pattern=r"BUILD\s+FAILURE",
        signal_name="output_maven_build_failure",
        weight=0.80,
        score_if_matched=0.20,
        frameworks=frozenset({"maven", "gradle"}),
    ),

    # ------------------------------------------------------------------ #
    # Negative — evidence of infrastructure / configuration problems      #
    # ------------------------------------------------------------------ #

    # "No module named" — Python import failure
    _OutputPattern(
        pattern=r"no module named",
        signal_name="output_import_error",
        weight=0.90,
        score_if_matched=0.0,
    ),
    # "ModuleNotFoundError" / "ImportError"
    _OutputPattern(
        pattern=r"(?:ModuleNotFoundError|ImportError):",
        signal_name="output_module_not_found",
        weight=0.90,
        score_if_matched=0.0,
    ),
    # "command not found" — shell / path problem
    _OutputPattern(
        pattern=r"command not found",
        signal_name="output_command_not_found",
        weight=0.95,
        score_if_matched=0.0,
    ),
    # "Permission denied"
    _OutputPattern(
        pattern=r"permission denied",
        signal_name="output_permission_denied",
        weight=0.85,
        score_if_matched=0.0,
    ),
    # "SyntaxError" — broken source code
    _OutputPattern(
        pattern=r"SyntaxError:",
        signal_name="output_syntax_error",
        weight=0.80,
        score_if_matched=0.05,
    ),
    # "error: could not compile" — Rust compilation failure
    _OutputPattern(
        pattern=r"error\[E\d+\]|could not compile",
        signal_name="output_compile_error",
        weight=0.85,
        score_if_matched=0.0,
        frameworks=frozenset({"cargo"}),
    ),
]


class OutputPatternSignalCollector(ExecutionSignalCollector):
    """Scan stdout/stderr for well-known test outcome phrases.

    Each registered pattern produces one :class:`ConfidenceSignal` whose
    score is:

    * ``score_if_matched`` — when the pattern is found in the output.
    * ``0.5`` — when the pattern is *not* found (absence is neutral
      evidence, not negative evidence, for most patterns).

    Patterns can be scoped to specific frameworks via the ``frameworks``
    field.  When *evidence.framework* is set and a pattern's frameworks set
    is non-empty, framework-specific patterns receive a small weight boost
    (+0.05) because the framework context makes the match more meaningful.

    Custom patterns can be supplied to extend or replace the defaults.
    """

    # Score when a pattern is not found — neutral, not negative
    ABSENT_SCORE: float = 0.5

    def __init__(
        self,
        patterns: list[_OutputPattern] | None = None,
    ) -> None:
        self._patterns = patterns if patterns is not None else _DEFAULT_OUTPUT_PATTERNS

    def collect(self, evidence: ExecutionEvidence) -> list[ConfidenceSignal]:
        text = evidence.combined_output
        fw = evidence.framework.lower()
        signals: list[ConfidenceSignal] = []

        for pat in self._patterns:
            matched = pat.matches(text)

            # Adjust weight when framework context confirms relevance
            weight = pat.weight
            if fw and pat.frameworks and fw in pat.frameworks:
                weight = min(1.0, weight + 0.05)

            score = pat.score_if_matched if matched else self.ABSENT_SCORE

            signals.append(
                ConfidenceSignal(
                    name=pat.signal_name,
                    weight=weight,
                    score=score,
                    evidence={
                        "pattern": pat.pattern,
                        "matched": matched,
                        "framework_hint": fw or None,
                        "frameworks_targeted": sorted(pat.frameworks),
                    },
                )
            )

        return signals


# ---------------------------------------------------------------------------
# Timing signal collector
# ---------------------------------------------------------------------------

# Timing thresholds (seconds):
_TIMING_INSTANT_MAX = 0.05       # < 50ms — almost certainly nothing ran
_TIMING_FAST_MAX = 0.5           # 50ms–500ms — could be a trivial suite or empty
_TIMING_NORMAL_MIN = 0.5         # >= 500ms — plausible test suite ran
_TIMING_SLOW_WARN = 120.0        # > 2min — worth a warning (may be flaky)
_TIMING_VERY_SLOW_WARN = 600.0   # > 10min — very suspicious


class TimingSignalCollector(ExecutionSignalCollector):
    """Use wall-clock execution time as a proxy for reliability.

    Timing evidence is *weak* but useful for detecting common failure
    modes:

    - **Instant exit** (< 50ms): The process probably crashed at startup
      or ran no tests.  Score: 0.10 (very low confidence).
    - **Fast exit** (50ms–500ms): May indicate a very small suite or a
      startup failure.  Score: 0.55 (slight uncertainty).
    - **Normal range** (500ms–120s): Plausible test execution.  Score: 1.0.
    - **Slow run** (120s–600s): Possibly long integration tests.  Score: 0.75.
    - **Very slow** (> 600s): Risk of timeout / hanging.  Score: 0.40.
    - **Timed out**: Score 0.0 regardless of duration.

    Thresholds are configurable.
    """

    def __init__(
        self,
        instant_max: float = _TIMING_INSTANT_MAX,
        fast_max: float = _TIMING_FAST_MAX,
        slow_warn: float = _TIMING_SLOW_WARN,
        very_slow_warn: float = _TIMING_VERY_SLOW_WARN,
        weight: float = 0.55,
    ) -> None:
        self._instant_max = instant_max
        self._fast_max = fast_max
        self._slow_warn = slow_warn
        self._very_slow_warn = very_slow_warn
        self._weight = weight

    def collect(self, evidence: ExecutionEvidence) -> list[ConfidenceSignal]:
        dur = evidence.duration_seconds

        if evidence.timed_out:
            category = "timed_out"
            score = 0.0
        elif dur < self._instant_max:
            category = "instant_exit"
            score = 0.10
        elif dur < self._fast_max:
            category = "fast_exit"
            score = 0.55
        elif dur < self._slow_warn:
            category = "normal"
            score = 1.0
        elif dur < self._very_slow_warn:
            category = "slow"
            score = 0.75
        else:
            category = "very_slow"
            score = 0.40

        return [
            ConfidenceSignal(
                name=f"timing_{category}",
                weight=self._weight,
                score=score,
                evidence={
                    "duration_seconds": dur,
                    "category": category,
                    "thresholds": {
                        "instant_max": self._instant_max,
                        "fast_max": self._fast_max,
                        "slow_warn": self._slow_warn,
                        "very_slow_warn": self._very_slow_warn,
                    },
                },
            )
        ]


# ---------------------------------------------------------------------------
# Assertion-count signal collector
# ---------------------------------------------------------------------------

# Patterns for extracting passed / failed / skipped / total counts.
# Each tuple: (pattern, group_name_passed, group_name_failed, group_name_total)
# Groups that don't apply use None.

@dataclass(frozen=True)
class _ParsedCounts:
    """Parsed test/assertion counts from output."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    source_pattern: str = ""

    @property
    def ran(self) -> int:
        """Total that actually ran (passed + failed)."""
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float:
        """Pass rate as [0, 1]; 0.0 if nothing ran."""
        if self.ran == 0:
            return 0.0
        return self.passed / self.ran


def _try_int(s: str | None) -> int:
    """Convert a string to int, returning 0 on failure."""
    try:
        return int(s or "0")
    except (TypeError, ValueError):
        return 0


# Framework-specific count extraction patterns.
# Order matters: first match wins.
# More-specific / longer patterns must come BEFORE the short generic ones
# (e.g. cargo and jest before the bare pytest "N passed" pattern).
_COUNT_PATTERNS: list[tuple[str, str]] = [
    # cargo test: "test result: ok. N passed; M failed; K ignored"
    # Must be before pytest because it also contains the word "passed"
    (
        r"test result:\s+(?:ok|FAILED)\.\s+(?P<passed>\d+)\s+passed;\s*(?P<failed>\d+)\s+failed;\s*(?P<skipped>\d+)\s+ignored",
        "cargo",
    ),
    # jest / vitest: "Tests: N passed, M failed, K total" (order of fields varies)
    # Must be before pytest to capture the "Tests:" prefix form
    (
        r"Tests:\s+(?:(?P<failed>\d+)\s+failed,\s*)?(?:(?P<skipped>\d+)\s+skipped,\s*)?(?P<passed>\d+)\s+passed,\s*(?P<total>\d+)\s+total",
        "jest",
    ),
    # JUnit XML summary: "Tests run: N, Failures: M, Errors: K, Skipped: J"
    (
        r"Tests run:\s*(?P<total>\d+),\s*Failures:\s*(?P<failed>\d+),\s*Errors:\s*\d+,\s*Skipped:\s*(?P<skipped>\d+)",
        "junit",
    ),
    # go test: "ok  package  0.123s" or "--- PASS: TestFoo (0.00s)"
    (
        r"^---\s+(?P<status>PASS|FAIL):\s+\S+\s+\([\d.]+s\)",
        "go_test_case",
    ),
    # pytest: "3 passed, 1 failed, 2 skipped in 1.23s"
    (
        r"(?P<passed>\d+)\s+passed"
        r"(?:,\s*(?P<failed>\d+)\s+failed)?"
        r"(?:,\s*(?P<skipped>\d+)\s+skipped)?",
        "pytest",
    ),
    # pytest failures only: "2 failed in 0.5s"
    (
        r"(?P<failed>\d+)\s+failed(?:\s+in\s+[\d.]+s)?",
        "pytest_fail_only",
    ),
    # Generic: "N tests passed" or "N/M tests passed"
    (
        r"(?P<passed>\d+)(?:/(?P<total>\d+))?\s+tests?\s+passed",
        "generic_passed",
    ),
]


def _extract_counts(text: str) -> _ParsedCounts | None:
    """Attempt to extract test/assertion counts from *text*.

    Tries each pattern in ``_COUNT_PATTERNS`` in order, returning the first
    match.  Returns ``None`` if no pattern matches.

    For the ``go_test_case`` pattern, accumulates PASS/FAIL lines rather
    than parsing a single summary line.
    """
    for pattern, source in _COUNT_PATTERNS:
        # Special case: go test accumulates per-test lines
        if source == "go_test_case":
            passed = len(re.findall(r"^---\s+PASS:", text, re.MULTILINE))
            failed = len(re.findall(r"^---\s+FAIL:", text, re.MULTILINE))
            if passed + failed > 0:
                return _ParsedCounts(
                    passed=passed,
                    failed=failed,
                    total=passed + failed,
                    source_pattern=source,
                )
            continue

        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            groups = m.groupdict()
            passed = _try_int(groups.get("passed"))
            failed = _try_int(groups.get("failed"))
            skipped = _try_int(groups.get("skipped"))
            total_raw = _try_int(groups.get("total"))
            total = total_raw or (passed + failed + skipped)

            # Accept the match whenever any named group was actually captured,
            # including the edge case where all counts are explicitly 0
            # (e.g. "0 passed in 0.01s" or "0 failed; 0 passed").
            if any(v is not None for v in groups.values()):
                return _ParsedCounts(
                    passed=passed,
                    failed=failed,
                    skipped=skipped,
                    total=total,
                    source_pattern=source,
                )

    return None


class AssertionCountSignalCollector(ExecutionSignalCollector):
    """Extract test/assertion counts and compute a pass-rate confidence signal.

    The collector parses the combined output for known test summary formats
    (pytest, jest, go test, cargo, JUnit) and computes:

        score = pass_rate = passed / (passed + failed)

    Where ``passed`` and ``failed`` are the parsed test counts.

    Special cases:
    - Zero tests ran (no counts found, or both 0): score 0.20 (low but
      not zero — absence of count lines is sometimes normal for scripts).
    - All skipped (passed == 0, failed == 0, skipped > 0): score 0.40.
    - Parse failure (no pattern matched): score 0.30 (weak uncertainty).

    The weight of this signal is moderate (0.75) because count parsing can
    be fragile across frameworks — it supplements rather than replaces the
    exit-code signal.
    """

    # Defaults for edge cases
    NO_COUNTS_SCORE = 0.30       # No parseable counts in output
    ZERO_TESTS_SCORE = 0.20      # Counts parsed but 0 tests ran
    ALL_SKIPPED_SCORE = 0.40     # Tests were collected but all skipped
    DEFAULT_WEIGHT = 0.75

    def __init__(
        self,
        weight: float = DEFAULT_WEIGHT,
        no_counts_score: float = NO_COUNTS_SCORE,
        zero_tests_score: float = ZERO_TESTS_SCORE,
        all_skipped_score: float = ALL_SKIPPED_SCORE,
    ) -> None:
        self._weight = weight
        self._no_counts_score = no_counts_score
        self._zero_tests_score = zero_tests_score
        self._all_skipped_score = all_skipped_score

    def collect(self, evidence: ExecutionEvidence) -> list[ConfidenceSignal]:
        text = evidence.combined_output
        counts = _extract_counts(text)

        if counts is None:
            return [
                ConfidenceSignal(
                    name="assertion_count_unparsed",
                    weight=self._weight,
                    score=self._no_counts_score,
                    evidence={
                        "parsed": False,
                        "reason": "No recognized test summary pattern found",
                        "output_length": len(text),
                    },
                )
            ]

        if counts.ran == 0 and counts.skipped > 0:
            score = self._all_skipped_score
            signal_name = "assertion_count_all_skipped"
        elif counts.ran == 0:
            score = self._zero_tests_score
            signal_name = "assertion_count_zero_ran"
        else:
            score = counts.pass_rate
            signal_name = "assertion_count_pass_rate"

        return [
            ConfidenceSignal(
                name=signal_name,
                weight=self._weight,
                score=score,
                evidence={
                    "parsed": True,
                    "source_pattern": counts.source_pattern,
                    "passed": counts.passed,
                    "failed": counts.failed,
                    "skipped": counts.skipped,
                    "total": counts.total,
                    "pass_rate": round(counts.pass_rate, 4),
                },
            )
        ]


# ---------------------------------------------------------------------------
# Infrastructure-health signal collector
# ---------------------------------------------------------------------------

# Stderr patterns that indicate infrastructure / environment problems.
# (pattern, signal_name, weight, score_if_matched)
_INFRA_ERROR_PATTERNS: list[tuple[str, str, float, float]] = [
    # Python import failures
    (r"ModuleNotFoundError:", "infra_module_not_found", 0.90, 0.0),
    (r"ImportError:", "infra_import_error", 0.85, 0.0),
    # Command / binary not found
    (r"command not found", "infra_command_not_found", 0.95, 0.0),
    (r"No such file or directory", "infra_file_not_found", 0.80, 0.05),
    # Permission problems
    (r"Permission denied", "infra_permission_denied", 0.85, 0.0),
    # Python environment / version issues
    (r"SyntaxError:", "infra_syntax_error", 0.80, 0.05),
    (r"IndentationError:", "infra_indentation_error", 0.75, 0.05),
    # Docker / container issues
    (r"docker:\s+Error", "infra_docker_error", 0.85, 0.0),
    (r"Unable to find image", "infra_docker_image_missing", 0.85, 0.05),
    # Out of memory / resource exhaustion
    (r"(?:MemoryError|OOMKilled|out of memory)", "infra_oom", 0.85, 0.0),
    # Network issues
    (r"(?:Connection refused|Network unreachable)", "infra_network_error", 0.75, 0.05),
    # Node.js / npm errors
    (r"Cannot find module", "infra_node_module_not_found", 0.85, 0.0),
    (r"npm ERR!", "infra_npm_error", 0.80, 0.05),
]

# Score when none of the error patterns are found (clean stderr)
_CLEAN_STDERR_SCORE = 1.0
_CLEAN_STDERR_WEIGHT = 0.65


class InfrastructureHealthSignalCollector(ExecutionSignalCollector):
    """Detect infrastructure-level errors in stderr.

    This collector acts as a safety net that catches problems *upstream*
    of the test runner itself — broken environments, missing dependencies,
    permission issues — that would make any test output unreliable.

    Two types of signals are produced:

    1. **Per-error signals** — one signal per matched error pattern, each
       scored 0.0 (or near-0) and with a high weight.  These are the
       primary escalation signals for the troubleshooter.

    2. **Clean-stderr signal** — emitted when *none* of the error patterns
       matched, scored 1.0.  This positive evidence that the environment
       is healthy moderately boosts overall confidence.

    Custom *error_patterns* can be supplied to extend or replace defaults.
    """

    def __init__(
        self,
        error_patterns: list[tuple[str, str, float, float]] | None = None,
        clean_stderr_weight: float = _CLEAN_STDERR_WEIGHT,
    ) -> None:
        self._patterns = error_patterns or _INFRA_ERROR_PATTERNS
        self._clean_stderr_weight = clean_stderr_weight

    def collect(self, evidence: ExecutionEvidence) -> list[ConfidenceSignal]:
        stderr = evidence.stderr
        signals: list[ConfidenceSignal] = []
        any_error_matched = False

        for pattern, signal_name, weight, score in self._patterns:
            m = re.search(pattern, stderr, re.IGNORECASE | re.MULTILINE)
            matched = bool(m)
            if matched:
                any_error_matched = True
                signals.append(
                    ConfidenceSignal(
                        name=signal_name,
                        weight=weight,
                        score=score,
                        evidence={
                            "pattern": pattern,
                            "matched": True,
                            "snippet": (m.group(0) if m else "")[:200],
                        },
                    )
                )

        # Emit a positive signal when stderr is clean
        signals.append(
            ConfidenceSignal(
                name="infra_stderr_clean",
                weight=self._clean_stderr_weight,
                score=_CLEAN_STDERR_SCORE if not any_error_matched else 0.0,
                evidence={
                    "any_error_matched": any_error_matched,
                    "stderr_length": len(stderr),
                },
            )
        )

        return signals


# ---------------------------------------------------------------------------
# Convenience: run all default collectors at once
# ---------------------------------------------------------------------------


def collect_execution_signals(
    evidence: ExecutionEvidence,
    collectors: Sequence[ExecutionSignalCollector] | None = None,
) -> list[ConfidenceSignal]:
    """Run all post-execution signal collectors against *evidence*.

    This is the primary entry point for the orchestrator hub after a test
    run completes.  It runs every built-in collector (or a custom list) and
    returns a flat list of :class:`ConfidenceSignal` instances ready to be
    fed into the :class:`~test_runner.models.confidence.ConfidenceModel`.

    Args:
        evidence: The raw execution evidence (exit code, output, timing).
        collectors: Optional custom list of collectors.  Defaults to all
            built-in collectors.

    Returns:
        Flat list of confidence signals from all collectors.

    Example::

        evidence = ExecutionEvidence(
            exit_code=0,
            stdout="3 passed in 1.23s",
            stderr="",
            duration_seconds=1.23,
        )
        signals = collect_execution_signals(evidence)
        # Feed into the confidence model:
        result = ConfidenceModel().evaluate(signals)
    """
    all_collectors: Sequence[ExecutionSignalCollector] = collectors or [
        ExitCodeSignalCollector(),
        OutputPatternSignalCollector(),
        TimingSignalCollector(),
        AssertionCountSignalCollector(),
        InfrastructureHealthSignalCollector(),
    ]
    signals: list[ConfidenceSignal] = []
    for collector in all_collectors:
        signals.extend(collector.collect(evidence))
    return signals
