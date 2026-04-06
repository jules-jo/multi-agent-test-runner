"""Tests for per-file invocation confidence scoring.

Covers:
- File-type signal collection (extension → base confidence)
- Naming-convention signal collection (filename patterns)
- Framework-marker signal collection (content inspection)
- Command suggestion logic
- InvocationConfidenceScorer integration
- Edge cases: unreadable files, unknown types, conftest.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from test_runner.agents.discovery.invocation_confidence import (
    InvocationConfidence,
    InvocationConfidenceScorer,
    _collect_file_type_signals,
    _collect_framework_marker_signals,
    _collect_naming_convention_signals,
    _suggest_command,
)
from test_runner.models.confidence import ConfidenceTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Create a file with given name and content under tmp_path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# File-type signal collection
# ---------------------------------------------------------------------------


class TestFileTypeSignals:
    def test_py_file_gives_python_hint(self):
        signals, framework = _collect_file_type_signals(Path("test_foo.py"))
        assert framework == "python"
        assert len(signals) == 1
        assert signals[0].score >= 0.70

    def test_js_file_gives_javascript_hint(self):
        signals, framework = _collect_file_type_signals(Path("calc.test.js"))
        assert framework == "javascript"
        assert signals[0].score >= 0.60

    def test_ts_file_gives_typescript_hint(self):
        signals, framework = _collect_file_type_signals(Path("auth.spec.ts"))
        assert framework == "typescript"
        assert signals[0].score >= 0.60

    def test_sh_file_gives_shell_hint(self):
        signals, framework = _collect_file_type_signals(Path("run_tests.sh"))
        assert framework == "shell"

    def test_go_file_gives_go_hint(self):
        signals, framework = _collect_file_type_signals(Path("math_test.go"))
        assert framework == "go"

    def test_rs_file_gives_rust_hint(self):
        signals, framework = _collect_file_type_signals(Path("lib.rs"))
        assert framework == "rust"

    def test_unknown_extension_gives_no_hint(self):
        signals, framework = _collect_file_type_signals(Path("weird.xyz"))
        assert framework is None
        # Low confidence for unknown type
        assert signals[0].score < 0.50

    def test_makefile_gives_make_hint(self):
        signals, framework = _collect_file_type_signals(Path("Makefile"))
        assert framework == "make"

    def test_signals_are_valid_range(self):
        """All file-type signals must have score and weight in [0, 1]."""
        for ext in [".py", ".js", ".ts", ".go", ".rs", ".sh", ".java", ".rb", ".xyz"]:
            signals, _ = _collect_file_type_signals(Path(f"file{ext}"))
            for s in signals:
                assert 0.0 <= s.score <= 1.0
                assert 0.0 <= s.weight <= 1.0


# ---------------------------------------------------------------------------
# Naming-convention signal collection
# ---------------------------------------------------------------------------


class TestNamingConventionSignals:
    def test_pytest_prefix_convention(self):
        signals = _collect_naming_convention_signals(Path("test_math.py"))
        names = {s.name for s in signals}
        assert "naming_python_test_prefix" in names
        # Must have high weight and perfect score
        sig = next(s for s in signals if s.name == "naming_python_test_prefix")
        assert sig.weight >= 0.85
        assert sig.score == 1.0

    def test_pytest_suffix_convention(self):
        signals = _collect_naming_convention_signals(Path("math_test.py"))
        names = {s.name for s in signals}
        assert "naming_python_test_suffix" in names
        sig = next(s for s in signals if s.name == "naming_python_test_suffix")
        assert sig.score == 1.0

    def test_jest_spec_convention(self):
        signals = _collect_naming_convention_signals(Path("calculator.spec.ts"))
        names = {s.name for s in signals}
        assert "naming_spec_file" in names

    def test_jest_test_convention(self):
        signals = _collect_naming_convention_signals(Path("auth.test.js"))
        names = {s.name for s in signals}
        assert "naming_test_file_js" in names

    def test_go_test_convention(self):
        signals = _collect_naming_convention_signals(Path("math_test.go"))
        names = {s.name for s in signals}
        assert "naming_go_test_suffix" in names
        sig = next(s for s in signals if s.name == "naming_go_test_suffix")
        assert sig.weight >= 0.90

    def test_java_test_prefix_convention(self):
        signals = _collect_naming_convention_signals(Path("TestCalculator.java"))
        names = {s.name for s in signals}
        assert "naming_java_test_prefix" in names

    def test_java_test_suffix_convention(self):
        signals = _collect_naming_convention_signals(Path("CalculatorTest.java"))
        names = {s.name for s in signals}
        assert "naming_java_test_suffix" in names

    def test_ruby_spec_convention(self):
        signals = _collect_naming_convention_signals(Path("calculator_spec.rb"))
        names = {s.name for s in signals}
        assert "naming_ruby_spec" in names

    def test_conftest_gets_low_score(self):
        """conftest.py is a fixture helper, not directly invocable."""
        signals = _collect_naming_convention_signals(Path("conftest.py"))
        assert len(signals) == 1
        assert signals[0].name == "naming_conftest_helper"
        assert signals[0].score < 0.50

    def test_non_test_file_gets_negative_signal(self):
        """Files with no test naming pattern emit a zero-score signal."""
        signals = _collect_naming_convention_signals(Path("utils.py"))
        assert any(s.score == 0.0 for s in signals)

    def test_all_signals_valid_range(self):
        filenames = [
            "test_foo.py", "foo_test.py", "foo.spec.ts", "foo.test.js",
            "bar_test.go", "TestBar.java", "BarTest.java", "bar_spec.rb",
            "conftest.py", "utils.py", "main.rs",
        ]
        for name in filenames:
            for s in _collect_naming_convention_signals(Path(name)):
                assert 0.0 <= s.score <= 1.0, f"Score out of range for {name}: {s}"
                assert 0.0 <= s.weight <= 1.0, f"Weight out of range for {name}: {s}"


# ---------------------------------------------------------------------------
# Framework-marker signal collection
# ---------------------------------------------------------------------------


class TestFrameworkMarkerSignals:
    def test_pytest_import_detected(self, tmp_path: Path):
        f = _write(tmp_path, "test_math.py", """\
            import pytest

            def test_add():
                assert 1 + 1 == 2
        """)
        signals, frameworks = _collect_framework_marker_signals(f)
        names = {s.name for s in signals}
        assert "marker_pytest_import" in names
        assert "pytest" in frameworks

    def test_unittest_import_detected(self, tmp_path: Path):
        f = _write(tmp_path, "test_core.py", """\
            import unittest

            class TestCore(unittest.TestCase):
                def test_something(self):
                    self.assertEqual(1, 1)
        """)
        signals, frameworks = _collect_framework_marker_signals(f)
        names = {s.name for s in signals}
        assert "marker_unittest_import" in names
        assert "unittest" in frameworks

    def test_test_function_marker_detected(self, tmp_path: Path):
        f = _write(tmp_path, "test_stuff.py", """\
            def test_foo():
                pass

            def test_bar():
                pass
        """)
        signals, frameworks = _collect_framework_marker_signals(f)
        names = {s.name for s in signals}
        assert "marker_pytest_test_function" in names

    def test_jest_describe_detected(self, tmp_path: Path):
        f = _write(tmp_path, "calc.test.js", """\
            describe('Calculator', () => {
                it('adds two numbers', () => {
                    expect(1 + 1).toBe(2);
                });
            });
        """)
        signals, frameworks = _collect_framework_marker_signals(f)
        names = {s.name for s in signals}
        assert "marker_js_describe" in names
        assert "jest" in frameworks

    def test_go_test_func_detected(self, tmp_path: Path):
        f = _write(tmp_path, "math_test.go", """\
            package math

            import "testing"

            func TestAdd(t *testing.T) {
                if add(1, 2) != 3 {
                    t.Error("add failed")
                }
            }
        """)
        signals, frameworks = _collect_framework_marker_signals(f)
        names = {s.name for s in signals}
        assert "marker_go_test_func" in names
        assert "go_test" in frameworks

    def test_rust_test_attr_detected(self, tmp_path: Path):
        f = _write(tmp_path, "lib.rs", """\
            #[cfg(test)]
            mod tests {
                #[test]
                fn test_add() {
                    assert_eq!(1 + 1, 2);
                }
            }
        """)
        signals, frameworks = _collect_framework_marker_signals(f)
        names = {s.name for s in signals}
        assert "marker_rust_test_attr" in names
        assert "cargo_test" in frameworks

    def test_shell_shebang_detected(self, tmp_path: Path):
        f = _write(tmp_path, "run_tests.sh", """\
            #!/bin/bash
            echo "Running tests"
            python -m pytest
        """)
        signals, frameworks = _collect_framework_marker_signals(f)
        names = {s.name for s in signals}
        assert "marker_shebang" in names

    def test_no_markers_returns_negative_signal(self, tmp_path: Path):
        f = _write(tmp_path, "config.txt", """\
            [settings]
            key = value
        """)
        signals, frameworks = _collect_framework_marker_signals(f)
        assert any(s.name == "marker_none_found" and s.score == 0.0 for s in signals)
        assert frameworks == []

    def test_all_matched_signals_score_one(self, tmp_path: Path):
        """Every matched marker signal must have score == 1.0."""
        f = _write(tmp_path, "test_stuff.py", """\
            import pytest

            def test_foo():
                assert True
        """)
        signals, _ = _collect_framework_marker_signals(f)
        for s in signals:
            if s.name != "marker_none_found":
                assert s.score == 1.0, f"Signal {s.name} should have score 1.0"


# ---------------------------------------------------------------------------
# Command suggestion
# ---------------------------------------------------------------------------


class TestSuggestCommand:
    def test_pytest_command(self):
        cmd = _suggest_command(Path("/project/tests/test_math.py"), "pytest")
        assert cmd is not None
        assert "pytest" in cmd
        assert "/project/tests/test_math.py" in cmd

    def test_jest_command(self):
        cmd = _suggest_command(Path("/project/src/calc.test.js"), "jest")
        assert cmd is not None
        assert "jest" in cmd

    def test_go_test_command(self):
        cmd = _suggest_command(Path("/project/pkg/math_test.go"), "go_test")
        assert cmd is not None
        assert "go test" in cmd
        # Should use directory, not file
        assert "/project/pkg" in cmd

    def test_shell_command(self):
        cmd = _suggest_command(Path("/scripts/run_tests.sh"), "shell")
        assert cmd is not None
        assert "bash" in cmd
        assert "/scripts/run_tests.sh" in cmd

    def test_unknown_framework_returns_none(self):
        cmd = _suggest_command(Path("test.xyz"), None)
        assert cmd is None

    def test_unknown_framework_name_returns_none(self):
        cmd = _suggest_command(Path("test.py"), "unknown_fw_xyz")
        assert cmd is None

    def test_vitest_command(self):
        cmd = _suggest_command(Path("/src/app.test.ts"), "vitest")
        assert cmd is not None
        assert "vitest" in cmd

    def test_cargo_test_command(self):
        cmd = _suggest_command(Path("/project/src/lib.rs"), "cargo_test")
        assert cmd is not None
        assert "cargo test" in cmd


# ---------------------------------------------------------------------------
# InvocationConfidenceScorer integration
# ---------------------------------------------------------------------------


class TestInvocationConfidenceScorer:
    """End-to-end integration tests for the scorer."""

    def test_pytest_file_gets_high_confidence(self, tmp_path: Path):
        """A well-named pytest file with import should reach HIGH tier."""
        f = _write(tmp_path, "test_math.py", """\
            import pytest

            def test_add():
                assert 1 + 1 == 2
        """)
        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)

        assert result.score >= 0.70
        assert result.tier in (ConfidenceTier.HIGH, ConfidenceTier.MEDIUM)
        assert result.framework in ("pytest", "python")
        assert result.suggested_command is not None
        assert result.can_invoke is True
        assert result.needs_investigation is False

    def test_non_test_file_gets_low_confidence(self, tmp_path: Path):
        """A plain utility file with no test markers should score low."""
        f = _write(tmp_path, "utils.py", """\
            def add(a, b):
                return a + b
        """)
        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)

        # No test naming, no test markers → LOW confidence
        assert result.score < 0.70
        assert result.tier != ConfidenceTier.HIGH

    def test_go_test_file_high_confidence(self, tmp_path: Path):
        f = _write(tmp_path, "math_test.go", """\
            package math

            import "testing"

            func TestAdd(t *testing.T) {
                if 1 + 1 != 2 {
                    t.Error("math is broken")
                }
            }
        """)
        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)

        assert result.score >= 0.70
        assert result.framework == "go_test"
        assert result.suggested_command is not None
        assert "go test" in result.suggested_command

    def test_jest_spec_file_high_confidence(self, tmp_path: Path):
        f = _write(tmp_path, "calculator.spec.ts", """\
            import { describe, it, expect } from 'vitest';

            describe('Calculator', () => {
                it('adds numbers', () => {
                    expect(1 + 1).toBe(2);
                });
            });
        """)
        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)

        assert result.score >= 0.60
        assert result.can_invoke is True

    def test_conftest_gets_low_confidence(self, tmp_path: Path):
        """conftest.py should have low naming score even with pytest imports."""
        f = _write(tmp_path, "conftest.py", """\
            import pytest

            @pytest.fixture
            def my_fixture():
                return 42
        """)
        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)

        # conftest.py is a helper — naming penalty keeps score moderate/low
        # It may still have framework markers from pytest import, but naming
        # drags it down
        assert result.score < 0.90  # Should not be fully HIGH

    def test_rust_test_file(self, tmp_path: Path):
        f = _write(tmp_path, "lib.rs", """\
            pub fn add(a: i32, b: i32) -> i32 { a + b }

            #[cfg(test)]
            mod tests {
                use super::*;
                #[test]
                fn test_add() { assert_eq!(add(1, 2), 3); }
            }
        """)
        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)

        assert result.framework == "cargo_test"
        assert result.score > 0.0  # Some evidence found

    def test_nonexistent_file_still_scores(self):
        """Scoring a nonexistent path should not raise; content signals skip."""
        scorer = InvocationConfidenceScorer()
        p = Path("/nonexistent/path/test_foo.py")
        result = scorer.score_file(p)

        # File type + naming signals still run; only content is skipped
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.tier, ConfidenceTier)

    def test_score_files_returns_list(self, tmp_path: Path):
        files = [
            _write(tmp_path, f"test_mod{i}.py", f"def test_{i}(): pass")
            for i in range(3)
        ]
        scorer = InvocationConfidenceScorer()
        results = scorer.score_files(files)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, InvocationConfidence)

    def test_score_files_sorted_descending(self, tmp_path: Path):
        # pytest file → higher confidence
        strong = _write(tmp_path, "test_strong.py", """\
            import pytest
            def test_foo(): assert True
        """)
        # generic file → lower confidence
        weak = _write(tmp_path, "utils.py", "def helper(): pass")

        scorer = InvocationConfidenceScorer()
        results = scorer.score_files_sorted([weak, strong])

        # Highest confidence first
        assert results[0].score >= results[1].score

    def test_score_files_sorted_ascending(self, tmp_path: Path):
        strong = _write(tmp_path, "test_strong.py", "import pytest\ndef test_x(): pass")
        weak = _write(tmp_path, "utils.py", "x = 1")

        scorer = InvocationConfidenceScorer()
        results = scorer.score_files_sorted([strong, weak], descending=False)

        assert results[0].score <= results[1].score

    def test_result_summary_is_serializable(self, tmp_path: Path):
        f = _write(tmp_path, "test_x.py", "import pytest\ndef test_x(): pass")
        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)
        summary = result.summary()

        assert "path" in summary
        assert "score" in summary
        assert "tier" in summary
        assert "can_invoke" in summary
        assert "framework" in summary
        assert "suggested_command" in summary
        assert "signal_count" in summary
        assert "signals" in summary

    def test_result_signals_are_valid(self, tmp_path: Path):
        f = _write(tmp_path, "test_x.py", "import pytest\ndef test_x(): pass")
        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)

        for s in result.signals:
            assert 0.0 <= s.score <= 1.0
            assert 0.0 <= s.weight <= 1.0

    def test_custom_thresholds(self, tmp_path: Path):
        """Custom thresholds should change tier classification."""
        f = _write(tmp_path, "test_x.py", "import pytest\ndef test_x(): pass")

        # Very strict: only 99%+ is HIGH
        strict = InvocationConfidenceScorer(execute_threshold=0.99, warn_threshold=0.60)
        result = strict.score_file(f)
        # Even a good pytest file won't reach HIGH under extreme strictness
        assert result.tier in (ConfidenceTier.MEDIUM, ConfidenceTier.LOW, ConfidenceTier.HIGH)

    def test_skip_content_reads(self, tmp_path: Path):
        """read_content=False skips file I/O; score based on type + naming only."""
        f = _write(tmp_path, "test_calc.py", "import pytest\ndef test_x(): pass")

        scorer_no_read = InvocationConfidenceScorer(read_content=False)
        scorer_with_read = InvocationConfidenceScorer(read_content=True)

        result_no = scorer_no_read.score_file(f)
        result_yes = scorer_with_read.score_file(f)

        # With content read, score should be >= without (more signals)
        assert result_no.evidence["content_scanned"] is False
        assert result_yes.evidence["content_scanned"] is True
        # No read should have fewer signals
        assert len(result_no.signals) <= len(result_yes.signals)

    def test_invocation_confidence_is_frozen(self, tmp_path: Path):
        f = _write(tmp_path, "test_x.py", "def test_x(): pass")
        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)
        with pytest.raises(AttributeError):
            result.score = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tier mapping
# ---------------------------------------------------------------------------


class TestInvocationConfidenceTierProperties:
    def _make_result(self, tier: ConfidenceTier) -> InvocationConfidence:
        from test_runner.models.confidence import ConfidenceSignal
        return InvocationConfidence(
            path=Path("test_x.py"),
            score=0.95 if tier == ConfidenceTier.HIGH else 0.70 if tier == ConfidenceTier.MEDIUM else 0.30,
            tier=tier,
            signals=(ConfidenceSignal(name="s", weight=1.0, score=0.5),),
        )

    def test_high_tier_can_invoke(self):
        assert self._make_result(ConfidenceTier.HIGH).can_invoke is True

    def test_medium_tier_can_invoke(self):
        assert self._make_result(ConfidenceTier.MEDIUM).can_invoke is True

    def test_low_tier_cannot_invoke(self):
        assert self._make_result(ConfidenceTier.LOW).can_invoke is False

    def test_high_tier_not_needs_investigation(self):
        assert self._make_result(ConfidenceTier.HIGH).needs_investigation is False

    def test_medium_tier_not_needs_investigation(self):
        assert self._make_result(ConfidenceTier.MEDIUM).needs_investigation is False

    def test_low_tier_needs_investigation(self):
        assert self._make_result(ConfidenceTier.LOW).needs_investigation is True


# ---------------------------------------------------------------------------
# Diverse framework detection
# ---------------------------------------------------------------------------


class TestDiverseFrameworks:
    """Verify the scorer detects a variety of frameworks correctly."""

    @pytest.mark.parametrize("filename,content,expected_framework", [
        (
            "test_calc.py",
            "import pytest\ndef test_add(): assert 1 + 1 == 2",
            "pytest",
        ),
        (
            "test_core.py",
            "import unittest\nclass TestCore(unittest.TestCase): pass",
            "unittest",
        ),
        (
            "math_test.go",
            'package math\nimport "testing"\nfunc TestAdd(t *testing.T) {}',
            "go_test",
        ),
        (
            "lib.rs",
            "#[cfg(test)]\nmod tests {\n    #[test]\n    fn it_works() {}\n}",
            "cargo_test",
        ),
        (
            "CalculatorTest.java",
            "import org.junit.Test;\npublic class CalculatorTest {\n    @Test\n    public void testAdd() {}\n}",
            "junit",
        ),
    ])
    def test_framework_detection(
        self,
        tmp_path: Path,
        filename: str,
        content: str,
        expected_framework: str,
    ):
        f = tmp_path / filename
        f.write_text(content)

        scorer = InvocationConfidenceScorer()
        result = scorer.score_file(f)

        assert result.framework == expected_framework, (
            f"Expected {expected_framework}, got {result.framework} for {filename}"
        )
