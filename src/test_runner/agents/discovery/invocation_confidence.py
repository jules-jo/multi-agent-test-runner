"""Per-file invocation confidence scoring.

Assigns an invocation confidence level to each discovered test script based
on three orthogonal evidence dimensions:

1. **File type** — derived from the file extension. Determines whether we
   know *how* to run the file (e.g. ``.py`` → pytest/python, ``.sh`` → bash).

2. **Naming conventions** — derived from the filename. Strong test-naming
   patterns (``test_*.py``, ``*_test.go``, ``*.spec.ts``) indicate the file
   is *intended* to be run as a test.

3. **Framework markers** — derived from the file's content. Actual import
   statements and function definitions (e.g. ``import pytest``, ``def test_``,
   ``describe(``) confirm the file *contains* test logic.

Each dimension is represented by one or more :class:`ConfidenceSignal`
instances that feed into an :class:`~test_runner.models.confidence.AggregatedConfidence`
or :class:`~test_runner.models.confidence.ConfidenceModel` to produce the
final :class:`InvocationConfidence` result.

Typical usage::

    scorer = InvocationConfidenceScorer()
    result = scorer.score_file(Path("tests/test_math.py"))
    print(result.score)          # 0.93
    print(result.tier)           # ConfidenceTier.HIGH
    print(result.framework)      # "pytest"
    print(result.suggested_command)  # "pytest tests/test_math.py"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from test_runner.models.confidence import (
    AggregatedConfidence,
    ConfidenceModel,
    ConfidenceResult,
    ConfidenceSignal,
    ConfidenceTier,
    EXECUTE_THRESHOLD,
    WARN_THRESHOLD,
)


# ---------------------------------------------------------------------------
# File-type taxonomy
# ---------------------------------------------------------------------------

# Mapping of file extension → (framework_hint, base_confidence)
# base_confidence: how sure we are we can invoke this file type at all.
_FILE_TYPE_MAP: dict[str, tuple[str, float]] = {
    # Python — high invocability; clear runner (pytest / python -m unittest)
    ".py": ("python", 0.80),
    # JavaScript
    ".js": ("javascript", 0.75),
    # TypeScript (needs compilation or ts-node/vitest)
    ".ts": ("typescript", 0.70),
    # Shell scripts
    ".sh": ("shell", 0.70),
    ".bash": ("shell", 0.70),
    ".zsh": ("shell", 0.65),
    # Go — standard `go test`
    ".go": ("go", 0.75),
    # Rust — `cargo test`
    ".rs": ("rust", 0.65),
    # Java/Kotlin — build-tool invocation
    ".java": ("java", 0.65),
    ".kt": ("kotlin", 0.60),
    # Ruby
    ".rb": ("ruby", 0.65),
    # Makefile — may contain a `test` target
    "Makefile": ("make", 0.50),
    ".mk": ("make", 0.45),
}

# Extensions for which we have no runner inference.
_UNKNOWN_FILE_TYPE_CONFIDENCE = 0.15


# ---------------------------------------------------------------------------
# Naming-convention patterns
# ---------------------------------------------------------------------------

# Each entry: (regex_on_filename, signal_name, weight, score)
# Patterns are matched against the filename only (not the full path).
_NAMING_PATTERNS: list[tuple[str, str, float, float]] = [
    # Python — pytest conventions
    (r"^test_.+\.py$", "naming_python_test_prefix", 0.90, 1.0),
    (r"^.+_test\.py$", "naming_python_test_suffix", 0.90, 1.0),
    # JavaScript/TypeScript — Jest/Vitest/Mocha conventions
    (r"^.+\.spec\.(js|jsx|ts|tsx|mjs|cjs)$", "naming_spec_file", 0.90, 1.0),
    (r"^.+\.test\.(js|jsx|ts|tsx|mjs|cjs)$", "naming_test_file_js", 0.90, 1.0),
    # Go — built-in convention
    (r"^.+_test\.go$", "naming_go_test_suffix", 0.95, 1.0),
    # Rust — tests often in files named test_* or alongside impl
    (r"^test_.+\.rs$", "naming_rust_test_prefix", 0.80, 1.0),
    # Java/Kotlin — JUnit convention
    (r"^Test.+\.(java|kt)$", "naming_java_test_prefix", 0.85, 1.0),
    (r"^.+Test\.(java|kt)$", "naming_java_test_suffix", 0.85, 1.0),
    (r"^.+Tests?\.(java|kt)$", "naming_java_tests_suffix", 0.85, 1.0),
    # Ruby — RSpec/Minitest
    (r"^.+_spec\.rb$", "naming_ruby_spec", 0.85, 1.0),
    (r"^test_.+\.rb$", "naming_ruby_test_prefix", 0.80, 1.0),
    # Generic: filename contains "test" anywhere (lower confidence)
    (r"(?i)test", "naming_contains_test", 0.40, 0.50),
]

# Penalty for filenames that look like conftest/fixture helpers (not directly runnable).
_CONFTEST_PATTERN = re.compile(r"^conftest\.py$", re.IGNORECASE)

# Weight for the "no test naming" signal (negative evidence).
_NO_NAMING_SIGNAL_WEIGHT = 0.50


# ---------------------------------------------------------------------------
# Framework-marker content patterns
# ---------------------------------------------------------------------------

# Each entry: (compiled_regex, signal_name, weight, framework_hint)
# Matched against the first N lines of file content (see _MAX_MARKER_LINES).
_MARKER_PATTERNS: list[tuple[re.Pattern[str], str, float, str]] = [
    # Python — pytest
    (re.compile(r"^\s*import\s+pytest\b", re.MULTILINE), "marker_pytest_import", 0.95, "pytest"),
    (re.compile(r"^\s*from\s+pytest\b", re.MULTILINE), "marker_pytest_from_import", 0.90, "pytest"),
    (re.compile(r"^\s*def\s+test_\w+\s*\(", re.MULTILINE), "marker_pytest_test_function", 0.85, "pytest"),
    (re.compile(r"^\s*class\s+Test\w+\s*[:\(]", re.MULTILINE), "marker_test_class", 0.75, "pytest"),
    # Python — unittest
    (re.compile(r"^\s*import\s+unittest\b", re.MULTILINE), "marker_unittest_import", 0.90, "unittest"),
    (re.compile(r"unittest\.TestCase", re.MULTILINE), "marker_unittest_testcase", 0.90, "unittest"),
    # JavaScript/TypeScript — Jest, Mocha, Jasmine, Vitest
    (re.compile(r"\bdescribe\s*\(", re.MULTILINE), "marker_js_describe", 0.80, "jest"),
    (re.compile(r"\bit\s*\(\s*['\"]", re.MULTILINE), "marker_js_it", 0.75, "jest"),
    (re.compile(r"\btest\s*\(\s*['\"]", re.MULTILINE), "marker_js_test_call", 0.75, "jest"),
    (re.compile(r"\bexpect\s*\(", re.MULTILINE), "marker_js_expect", 0.70, "jest"),
    (re.compile(r"^import\s+.*\bvitest\b", re.MULTILINE), "marker_vitest_import", 0.90, "vitest"),
    (re.compile(r"^import\s+.*\bjest\b", re.MULTILINE), "marker_jest_import", 0.85, "jest"),
    (re.compile(r"^(?:const|import)\s+.*\bmocha\b", re.MULTILINE), "marker_mocha_import", 0.80, "mocha"),
    # Go — standard testing package
    (re.compile(r"^\s*func\s+Test\w+\s*\(t\s+\*testing\.T\)", re.MULTILINE), "marker_go_test_func", 0.98, "go_test"),
    (re.compile(r'^\s*"testing"', re.MULTILINE), "marker_go_testing_import", 0.85, "go_test"),
    # Rust — built-in test infrastructure
    (re.compile(r"#\[test\]", re.MULTILINE), "marker_rust_test_attr", 0.95, "cargo_test"),
    (re.compile(r"#\[cfg\(test\)\]", re.MULTILINE), "marker_rust_cfg_test", 0.90, "cargo_test"),
    # Java/Kotlin — JUnit
    (re.compile(r"@Test\b", re.MULTILINE), "marker_junit_annotation", 0.90, "junit"),
    (re.compile(r"import\s+org\.junit\.", re.MULTILINE), "marker_junit_import", 0.90, "junit"),
    # Shell — executable script with test-like structure
    (re.compile(r"^#!/(?:bin/bash|bin/sh|usr/bin/env\s+bash)", re.MULTILINE), "marker_shebang", 0.60, "shell"),
    (re.compile(r"\bassert\b", re.MULTILINE), "marker_shell_assert", 0.45, "shell"),
]

# Maximum number of lines to scan for framework markers (avoids reading huge files).
_MAX_MARKER_LINES = 200

# Maximum bytes to read from a file for content analysis.
_MAX_READ_BYTES = 16_384


# ---------------------------------------------------------------------------
# Suggested-command inference
# ---------------------------------------------------------------------------

# Priority-ordered list of (framework_hint, command_template).
# The first match for the detected framework is used.
_COMMAND_TEMPLATES: dict[str, str] = {
    "pytest": "pytest {path}",
    "unittest": "python -m unittest {path_module}",
    "jest": "npx jest {path}",
    "vitest": "npx vitest run {path}",
    "mocha": "npx mocha {path}",
    "go_test": "go test {dir}/...",
    "cargo_test": "cargo test",
    "junit": "mvn test",
    "shell": "bash {path}",
    "make": "make test",
    "ruby": "ruby {path}",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvocationConfidence:
    """Invocation confidence result for a single discovered test script.

    Attributes:
        path: Absolute path to the test script.
        score: Aggregated confidence score in [0.0, 1.0].
        tier: Confidence tier (HIGH / MEDIUM / LOW).
        signals: All individual signals that contributed to the score.
        framework: Best-guess test framework, or ``None`` if undetected.
        suggested_command: Best-effort invocation command, or ``None``.
        evidence: Free-form metadata about the analysis.
    """

    path: Path
    score: float
    tier: ConfidenceTier
    signals: tuple[ConfidenceSignal, ...]
    framework: str | None = None
    suggested_command: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def can_invoke(self) -> bool:
        """True when confidence is HIGH or MEDIUM (score >= warn threshold)."""
        return self.tier in (ConfidenceTier.HIGH, ConfidenceTier.MEDIUM)

    @property
    def needs_investigation(self) -> bool:
        """True when confidence is LOW (score < warn threshold)."""
        return self.tier == ConfidenceTier.LOW

    def summary(self) -> dict[str, Any]:
        """Serializable representation for logging / handoff."""
        return {
            "path": str(self.path),
            "score": round(self.score, 4),
            "tier": self.tier.value,
            "can_invoke": self.can_invoke,
            "framework": self.framework,
            "suggested_command": self.suggested_command,
            "signal_count": len(self.signals),
            "signals": [
                {
                    "name": s.name,
                    "weight": s.weight,
                    "score": s.score,
                    "weighted_score": round(s.weighted_score, 4),
                }
                for s in self.signals
            ],
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Signal collectors
# ---------------------------------------------------------------------------


def _collect_file_type_signals(path: Path) -> tuple[list[ConfidenceSignal], str | None]:
    """Emit signals based solely on file extension.

    Returns a list of signals *and* the framework hint (if any) derived from
    the file type.
    """
    # Special-case Makefile (no extension)
    if path.name == "Makefile" or path.suffix == "":
        if path.name in _FILE_TYPE_MAP:
            framework_hint, confidence = _FILE_TYPE_MAP[path.name]
        else:
            # Unknown extension-less file
            framework_hint = None
            confidence = _UNKNOWN_FILE_TYPE_CONFIDENCE

        return (
            [
                ConfidenceSignal(
                    name="file_type_known" if framework_hint else "file_type_unknown",
                    weight=0.50,
                    score=confidence,
                    evidence={
                        "extension": "",
                        "filename": path.name,
                        "framework_hint": framework_hint,
                    },
                )
            ],
            framework_hint,
        )

    ext = path.suffix.lower()
    if ext in _FILE_TYPE_MAP:
        framework_hint, confidence = _FILE_TYPE_MAP[ext]
        signal_name = f"file_type_{ext.lstrip('.')}"
    else:
        framework_hint = None
        confidence = _UNKNOWN_FILE_TYPE_CONFIDENCE
        signal_name = "file_type_unknown"

    return (
        [
            ConfidenceSignal(
                name=signal_name,
                weight=0.50,
                score=confidence,
                evidence={
                    "extension": ext,
                    "filename": path.name,
                    "framework_hint": framework_hint,
                },
            )
        ],
        framework_hint,
    )


def _collect_naming_convention_signals(path: Path) -> list[ConfidenceSignal]:
    """Emit signals based on filename naming conventions."""
    filename = path.name
    signals: list[ConfidenceSignal] = []
    matched_any_strong_pattern = False

    # Penalise conftest.py — it's a pytest fixture file, not a runnable test
    if _CONFTEST_PATTERN.match(filename):
        signals.append(
            ConfidenceSignal(
                name="naming_conftest_helper",
                weight=0.80,
                score=0.10,
                evidence={
                    "filename": filename,
                    "reason": "conftest.py is a fixture helper, not a runnable test",
                },
            )
        )
        return signals

    for pattern_str, signal_name, weight, score in _NAMING_PATTERNS:
        if re.search(pattern_str, filename, re.IGNORECASE):
            if signal_name != "naming_contains_test":
                # Strong, specific match
                matched_any_strong_pattern = True
                signals.append(
                    ConfidenceSignal(
                        name=signal_name,
                        weight=weight,
                        score=score,
                        evidence={"filename": filename, "pattern": pattern_str},
                    )
                )
            elif not matched_any_strong_pattern:
                # Only emit the weak "contains test" signal if no strong pattern hit
                signals.append(
                    ConfidenceSignal(
                        name=signal_name,
                        weight=weight,
                        score=score,
                        evidence={"filename": filename, "pattern": pattern_str},
                    )
                )

    if not signals:
        # Explicit negative evidence: file name gives no test indication
        signals.append(
            ConfidenceSignal(
                name="naming_no_test_pattern",
                weight=_NO_NAMING_SIGNAL_WEIGHT,
                score=0.0,
                evidence={
                    "filename": filename,
                    "reason": "Filename does not match any known test naming convention",
                },
            )
        )

    return signals


def _read_file_head(path: Path) -> str | None:
    """Read up to _MAX_READ_BYTES of a file for content analysis.

    Returns None if the file cannot be read (binary, missing, permission error).
    """
    try:
        with path.open("rb") as fh:
            raw = fh.read(_MAX_READ_BYTES)

        # Reject binary files (heuristic: high proportion of non-text bytes)
        text_bytes = sum(1 for b in raw if b < 128 or b >= 160)
        if len(raw) > 0 and text_bytes / len(raw) < 0.70:
            return None

        content = raw.decode("utf-8", errors="replace")
        # Limit to first _MAX_MARKER_LINES lines
        lines = content.splitlines()[:_MAX_MARKER_LINES]
        return "\n".join(lines)
    except (OSError, PermissionError):
        return None


def _collect_framework_marker_signals(
    path: Path,
) -> tuple[list[ConfidenceSignal], list[str]]:
    """Emit signals based on framework-specific markers in the file content.

    Returns a list of signals *and* a list of detected framework hints ordered
    by total matched signal weight (highest-confidence framework first).
    """
    content = _read_file_head(path)
    if content is None:
        return (
            [
                ConfidenceSignal(
                    name="marker_unreadable",
                    weight=0.30,
                    score=0.0,
                    evidence={"path": str(path), "reason": "file unreadable or binary"},
                )
            ],
            [],
        )

    signals: list[ConfidenceSignal] = []
    # Accumulate total weight per framework for ranking
    framework_weights: dict[str, float] = {}

    for pattern, signal_name, weight, framework_hint in _MARKER_PATTERNS:
        matched = bool(pattern.search(content))
        if matched:
            signals.append(
                ConfidenceSignal(
                    name=signal_name,
                    weight=weight,
                    score=1.0,
                    evidence={
                        "pattern": pattern.pattern,
                        "matched": True,
                        "framework": framework_hint,
                    },
                )
            )
            framework_weights[framework_hint] = (
                framework_weights.get(framework_hint, 0.0) + weight
            )

    if not signals:
        # No markers found — modest negative evidence
        signals.append(
            ConfidenceSignal(
                name="marker_none_found",
                weight=0.35,
                score=0.0,
                evidence={
                    "path": str(path),
                    "reason": "No framework markers found in file content",
                },
            )
        )
        return signals, []

    # Order frameworks by total weight descending (most-confident framework first)
    detected_frameworks = sorted(
        framework_weights.keys(),
        key=lambda fw: framework_weights[fw],
        reverse=True,
    )

    return signals, detected_frameworks


# ---------------------------------------------------------------------------
# Command suggestion
# ---------------------------------------------------------------------------


def _suggest_command(
    path: Path,
    framework: str | None,
) -> str | None:
    """Build a best-effort invocation command for the given file.

    Selects a template from :data:`_COMMAND_TEMPLATES` using the detected
    framework, substituting ``{path}`` with the file's string representation.
    """
    if framework is None:
        return None

    template = _COMMAND_TEMPLATES.get(framework)
    if template is None:
        return None

    path_str = str(path)
    # For Python unittest, convert path to dotted module notation
    if "{path_module}" in template:
        # Strip .py and replace separators with dots
        module_path = path_str.replace("/", ".").replace("\\", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]
        template = template.replace("{path_module}", module_path)

    # For Go, use the directory (not the file)
    if "{dir}" in template:
        dir_str = str(path.parent)
        template = template.replace("{dir}", dir_str)

    return template.replace("{path}", path_str)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class InvocationConfidenceScorer:
    """Scores per-file invocation confidence for discovered test scripts.

    For each file, three signal collectors are run in sequence:

    1. **FileType** — extension-based confidence (do we know how to run it?).
    2. **NamingConvention** — filename-pattern confidence (is it a test?).
    3. **FrameworkMarker** — content-based confidence (does it contain tests?).

    The signals are aggregated by a :class:`~test_runner.models.confidence.ConfidenceModel`
    using a weighted average, and the result is classified into
    HIGH / MEDIUM / LOW tiers.

    Args:
        execute_threshold: Score at or above which the tier is HIGH.
                           Default 0.90 (per canonical spec).
        warn_threshold: Score at or above which the tier is MEDIUM.
                        Default 0.60 (per canonical spec).
        read_content: Whether to read file contents for marker detection.
                      Set to ``False`` to skip I/O-intensive content scanning
                      (useful for very large file trees or quick mode).
    """

    def __init__(
        self,
        execute_threshold: float = EXECUTE_THRESHOLD,
        warn_threshold: float = WARN_THRESHOLD,
        read_content: bool = True,
    ) -> None:
        self._model = ConfidenceModel(
            execute_threshold=execute_threshold,
            warn_threshold=warn_threshold,
        )
        self._read_content = read_content

    # -- Properties -----------------------------------------------------------

    @property
    def execute_threshold(self) -> float:
        """Score threshold for HIGH (execute) tier."""
        return self._model.execute_threshold

    @property
    def warn_threshold(self) -> float:
        """Score threshold for MEDIUM (warn) tier."""
        return self._model.warn_threshold

    # -- Primary API ----------------------------------------------------------

    def score_file(self, path: Path) -> InvocationConfidence:
        """Compute invocation confidence for a single test script.

        Runs all three signal collectors, aggregates the results, and
        returns a fully populated :class:`InvocationConfidence`.

        Args:
            path: Path to the test script (need not exist for unit tests,
                  but content signals will be empty if it doesn't).

        Returns:
            :class:`InvocationConfidence` with score, tier, framework, and
            suggested invocation command.
        """
        all_signals: list[ConfidenceSignal] = []

        # 1. File-type signals
        file_type_signals, file_type_framework = _collect_file_type_signals(path)
        all_signals.extend(file_type_signals)

        # 2. Naming-convention signals
        naming_signals = _collect_naming_convention_signals(path)
        all_signals.extend(naming_signals)

        # 3. Framework-marker signals (content-based)
        detected_frameworks: list[str] = []
        if self._read_content and path.exists() and path.is_file():
            marker_signals, detected_frameworks = _collect_framework_marker_signals(path)
            all_signals.extend(marker_signals)

        # Determine best framework: content markers > file type hint
        framework = (
            detected_frameworks[0]
            if detected_frameworks
            else file_type_framework
        )

        # Aggregate
        result: ConfidenceResult = self._model.evaluate(all_signals)

        # Suggest invocation command
        suggested = _suggest_command(path, framework)

        return InvocationConfidence(
            path=path,
            score=result.score,
            tier=result.tier,
            signals=result.signals,
            framework=framework,
            suggested_command=suggested,
            evidence={
                "file_type_framework": file_type_framework,
                "detected_frameworks": detected_frameworks,
                "content_scanned": (
                    self._read_content and path.exists() and path.is_file()
                ),
            },
        )

    def score_files(
        self,
        paths: Sequence[Path],
    ) -> list[InvocationConfidence]:
        """Score multiple test scripts in one call.

        Args:
            paths: Iterable of file paths to score.

        Returns:
            List of :class:`InvocationConfidence` results in the same order
            as the input paths.
        """
        return [self.score_file(p) for p in paths]

    def score_files_sorted(
        self,
        paths: Sequence[Path],
        *,
        descending: bool = True,
    ) -> list[InvocationConfidence]:
        """Score and sort test scripts by confidence score.

        Args:
            paths: Iterable of file paths to score.
            descending: If True (default), highest confidence first.

        Returns:
            List sorted by :attr:`InvocationConfidence.score`.
        """
        results = self.score_files(paths)
        return sorted(results, key=lambda r: r.score, reverse=descending)
