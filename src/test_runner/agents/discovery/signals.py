"""Evidence-based signal collectors for test discovery.

Each collector implements `ConfidenceSignalCollector.collect()` and returns
one or more `ConfidenceSignal` instances. Collectors are composable — the
discovery agent feeds them a scan root and aggregates the results.

Collector taxonomy
------------------
* **FileExistenceCollector** — checks for well-known config/marker files
  (e.g. pytest.ini, package.json with "test" script, Makefile with test target).
* **PatternMatchingCollector** — scans file trees for naming patterns
  (e.g. test_*.py, *_test.go, *.spec.ts).
* **FrameworkDetectionCollector** — inspects dependency manifests and imports
  to identify installed/used test frameworks.
"""

from __future__ import annotations

import abc
import re
from pathlib import Path
from typing import Sequence

from test_runner.models.confidence import ConfidenceSignal


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ConfidenceSignalCollector(abc.ABC):
    """Base class for all signal collectors.

    Subclasses implement ``collect(root)`` which scans a directory tree
    and returns a list of :class:`ConfidenceSignal` describing what was
    (or was not) found.
    """

    @abc.abstractmethod
    def collect(self, root: Path) -> list[ConfidenceSignal]:
        """Scan *root* and return zero or more confidence signals."""


# ---------------------------------------------------------------------------
# File-existence collector
# ---------------------------------------------------------------------------

# Mapping: filename -> (signal_name, weight)
_FILE_MARKERS: dict[str, tuple[str, float]] = {
    # Python
    "pytest.ini": ("pytest_ini_exists", 0.9),
    "pyproject.toml": ("pyproject_toml_exists", 0.7),
    "setup.cfg": ("setup_cfg_exists", 0.5),
    "tox.ini": ("tox_ini_exists", 0.7),
    # JavaScript / TypeScript
    "package.json": ("package_json_exists", 0.6),
    "jest.config.js": ("jest_config_exists", 0.9),
    "jest.config.ts": ("jest_config_ts_exists", 0.9),
    "vitest.config.ts": ("vitest_config_exists", 0.9),
    "karma.conf.js": ("karma_conf_exists", 0.8),
    ".mocharc.yml": ("mocharc_exists", 0.8),
    # Go
    "go.mod": ("go_mod_exists", 0.6),
    # Rust
    "Cargo.toml": ("cargo_toml_exists", 0.6),
    # JVM
    "build.gradle": ("gradle_build_exists", 0.6),
    "pom.xml": ("pom_xml_exists", 0.6),
    # Generic
    "Makefile": ("makefile_exists", 0.4),
}


class FileExistenceCollector(ConfidenceSignalCollector):
    """Check for well-known test-related configuration / marker files.

    For each marker file, emits a signal with score 1.0 if the file
    exists and 0.0 otherwise. The weight reflects how strongly the
    file's presence indicates a runnable test suite.

    Custom markers can be supplied to extend or replace the defaults.
    """

    def __init__(
        self,
        markers: dict[str, tuple[str, float]] | None = None,
    ) -> None:
        self._markers = markers if markers is not None else _FILE_MARKERS

    def collect(self, root: Path) -> list[ConfidenceSignal]:
        signals: list[ConfidenceSignal] = []
        for filename, (signal_name, weight) in self._markers.items():
            target = root / filename
            exists = target.is_file()
            signals.append(
                ConfidenceSignal(
                    name=signal_name,
                    weight=weight,
                    score=1.0 if exists else 0.0,
                    evidence={"path": str(target), "exists": exists},
                )
            )
        return signals


# ---------------------------------------------------------------------------
# Pattern-matching collector
# ---------------------------------------------------------------------------


class _PatternSpec:
    """Internal descriptor for a file-name pattern."""

    __slots__ = ("glob", "signal_name", "weight")

    def __init__(self, glob: str, signal_name: str, weight: float) -> None:
        self.glob = glob
        self.signal_name = signal_name
        self.weight = weight


# Default patterns recognised across common ecosystems.
_DEFAULT_PATTERNS: list[_PatternSpec] = [
    # Python
    _PatternSpec("**/test_*.py", "python_test_files", 0.8),
    _PatternSpec("**/*_test.py", "python_test_files_alt", 0.8),
    _PatternSpec("**/tests/**/*.py", "python_tests_dir", 0.7),
    # JavaScript / TypeScript
    _PatternSpec("**/*.spec.ts", "ts_spec_files", 0.8),
    _PatternSpec("**/*.test.ts", "ts_test_files", 0.8),
    _PatternSpec("**/*.spec.js", "js_spec_files", 0.8),
    _PatternSpec("**/*.test.js", "js_test_files", 0.8),
    # Go
    _PatternSpec("**/*_test.go", "go_test_files", 0.8),
    # Rust
    _PatternSpec("**/test_*.rs", "rust_test_files", 0.7),
    # Java
    _PatternSpec("**/Test*.java", "java_test_files", 0.7),
]

# Maximum files to scan per pattern before stopping (keeps discovery fast).
_MAX_SCAN_FILES = 50_000


class PatternMatchingCollector(ConfidenceSignalCollector):
    """Walk the file tree looking for test-file naming conventions.

    For each pattern, the score is proportional to the number of
    matching files found: ``min(count / 5, 1.0)``. This means
    five or more matches yield full confidence; fewer matches
    scale linearly.

    Custom patterns and the match ceiling can be overridden.
    """

    def __init__(
        self,
        patterns: Sequence[_PatternSpec] | None = None,
        max_files: int = _MAX_SCAN_FILES,
    ) -> None:
        self._patterns = list(patterns or _DEFAULT_PATTERNS)
        self._max_files = max_files

    def collect(self, root: Path) -> list[ConfidenceSignal]:
        signals: list[ConfidenceSignal] = []
        for spec in self._patterns:
            matches = list(_iglob_limited(root, spec.glob, self._max_files))
            count = len(matches)
            # Score: 1.0 if >= 5 files, proportional below that.
            score = min(count / 5.0, 1.0) if count > 0 else 0.0
            signals.append(
                ConfidenceSignal(
                    name=spec.signal_name,
                    weight=spec.weight,
                    score=score,
                    evidence={
                        "pattern": spec.glob,
                        "matched_count": count,
                        "sample_files": [str(p) for p in matches[:5]],
                    },
                )
            )
        return signals


def _iglob_limited(root: Path, pattern: str, limit: int):
    """Yield up to *limit* matches for *pattern* under *root*."""
    count = 0
    for match in root.glob(pattern):
        if count >= limit:
            break
        yield match
        count += 1


# ---------------------------------------------------------------------------
# Framework-detection collector
# ---------------------------------------------------------------------------

# Each entry: (file_to_inspect, regex, signal_name, weight)
_FRAMEWORK_PROBES: list[tuple[str, str, str, float]] = [
    # Python
    ("pyproject.toml", r"\bpytest\b", "pytest_in_pyproject", 0.9),
    ("pyproject.toml", r"\bunittest\b", "unittest_in_pyproject", 0.7),
    ("setup.cfg", r"\bpytest\b", "pytest_in_setup_cfg", 0.8),
    ("tox.ini", r"\bpytest\b", "pytest_in_tox", 0.8),
    ("requirements.txt", r"\bpytest\b", "pytest_in_requirements", 0.85),
    ("requirements-dev.txt", r"\bpytest\b", "pytest_in_dev_requirements", 0.85),
    # JavaScript / TypeScript
    ("package.json", r"\"jest\"", "jest_in_package_json", 0.9),
    ("package.json", r"\"vitest\"", "vitest_in_package_json", 0.9),
    ("package.json", r"\"mocha\"", "mocha_in_package_json", 0.85),
    ("package.json", r"\"karma\"", "karma_in_package_json", 0.8),
    ("package.json", r"\"test\"\s*:", "npm_test_script", 0.7),
    # Go — go test is built-in, just check for go.mod presence
    ("go.mod", r"^module\s+", "go_module_detected", 0.6),
    # Rust
    ("Cargo.toml", r"\[dev-dependencies\]", "cargo_dev_deps", 0.6),
    # JVM
    ("pom.xml", r"<artifactId>junit", "junit_in_pom", 0.85),
    ("build.gradle", r"testImplementation", "junit_in_gradle", 0.85),
]


class FrameworkDetectionCollector(ConfidenceSignalCollector):
    """Inspect dependency manifests / config files for framework references.

    For each probe, reads the target file (caching to avoid redundant I/O)
    and checks whether the regex pattern matches. Score is 1.0 on match,
    0.0 otherwise.

    Custom probes can be supplied to extend or replace the defaults.
    """

    def __init__(
        self,
        probes: list[tuple[str, str, str, float]] | None = None,
    ) -> None:
        self._probes = probes if probes is not None else _FRAMEWORK_PROBES

    def collect(self, root: Path) -> list[ConfidenceSignal]:
        # Cache file contents to avoid redundant reads.
        cache: dict[str, str | None] = {}
        signals: list[ConfidenceSignal] = []

        for filename, regex, signal_name, weight in self._probes:
            if filename not in cache:
                target = root / filename
                try:
                    cache[filename] = target.read_text(errors="replace")
                except (OSError, FileNotFoundError):
                    cache[filename] = None

            content = cache[filename]
            matched = bool(content and re.search(regex, content))
            signals.append(
                ConfidenceSignal(
                    name=signal_name,
                    weight=weight,
                    score=1.0 if matched else 0.0,
                    evidence={
                        "file": filename,
                        "pattern": regex,
                        "matched": matched,
                    },
                )
            )
        return signals


# ---------------------------------------------------------------------------
# Convenience: run all default collectors at once
# ---------------------------------------------------------------------------


def collect_all_signals(root: Path) -> list[ConfidenceSignal]:
    """Run every built-in collector and return a flat list of signals."""
    collectors: list[ConfidenceSignalCollector] = [
        FileExistenceCollector(),
        PatternMatchingCollector(),
        FrameworkDetectionCollector(),
    ]
    signals: list[ConfidenceSignal] = []
    for c in collectors:
        signals.extend(c.collect(root))
    return signals
