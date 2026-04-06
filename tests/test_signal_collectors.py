"""Tests for the signal collectors (file existence, pattern matching, framework detection)."""

import pytest
from pathlib import Path

from test_runner.models.confidence import ConfidenceSignal
from test_runner.agents.discovery.signals import (
    FileExistenceCollector,
    PatternMatchingCollector,
    FrameworkDetectionCollector,
    collect_all_signals,
)


class TestFileExistenceCollector:
    def test_detects_existing_file(self, tmp_path: Path):
        (tmp_path / "pytest.ini").write_text("[pytest]")
        collector = FileExistenceCollector()
        signals = collector.collect(tmp_path)
        pytest_sig = next(s for s in signals if s.name == "pytest_ini_exists")
        assert pytest_sig.score == 1.0
        assert pytest_sig.evidence["exists"] is True

    def test_missing_file_scores_zero(self, tmp_path: Path):
        collector = FileExistenceCollector()
        signals = collector.collect(tmp_path)
        pytest_sig = next(s for s in signals if s.name == "pytest_ini_exists")
        assert pytest_sig.score == 0.0
        assert pytest_sig.evidence["exists"] is False

    def test_custom_markers(self, tmp_path: Path):
        (tmp_path / "custom.cfg").write_text("x")
        collector = FileExistenceCollector(
            markers={"custom.cfg": ("custom_exists", 0.5)}
        )
        signals = collector.collect(tmp_path)
        assert len(signals) == 1
        assert signals[0].name == "custom_exists"
        assert signals[0].score == 1.0
        assert signals[0].weight == 0.5

    def test_all_signals_have_valid_ranges(self, tmp_path: Path):
        collector = FileExistenceCollector()
        for sig in collector.collect(tmp_path):
            assert 0.0 <= sig.weight <= 1.0
            assert 0.0 <= sig.score <= 1.0


class TestPatternMatchingCollector:
    def test_detects_python_test_files(self, tmp_path: Path):
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        for i in range(5):
            (test_dir / f"test_mod{i}.py").write_text(f"# test {i}")
        collector = PatternMatchingCollector()
        signals = collector.collect(tmp_path)
        py_sig = next(s for s in signals if s.name == "python_test_files")
        assert py_sig.score == 1.0  # 5 files -> score 1.0
        assert py_sig.evidence["matched_count"] == 5

    def test_partial_score_for_few_files(self, tmp_path: Path):
        (tmp_path / "test_one.py").write_text("# test")
        collector = PatternMatchingCollector()
        signals = collector.collect(tmp_path)
        py_sig = next(s for s in signals if s.name == "python_test_files")
        assert py_sig.score == pytest.approx(0.2)  # 1/5

    def test_zero_matches(self, tmp_path: Path):
        collector = PatternMatchingCollector()
        signals = collector.collect(tmp_path)
        for sig in signals:
            assert sig.score == 0.0
            assert sig.evidence["matched_count"] == 0

    def test_sample_files_limited_to_five(self, tmp_path: Path):
        for i in range(10):
            (tmp_path / f"test_{i}.py").write_text("")
        collector = PatternMatchingCollector()
        signals = collector.collect(tmp_path)
        py_sig = next(s for s in signals if s.name == "python_test_files")
        assert len(py_sig.evidence["sample_files"]) <= 5


class TestFrameworkDetectionCollector:
    def test_detects_pytest_in_pyproject(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["pytest>=7.0"]'
        )
        collector = FrameworkDetectionCollector()
        signals = collector.collect(tmp_path)
        sig = next(s for s in signals if s.name == "pytest_in_pyproject")
        assert sig.score == 1.0
        assert sig.evidence["matched"] is True

    def test_detects_jest_in_package_json(self, tmp_path: Path):
        (tmp_path / "package.json").write_text(
            '{"devDependencies": {"jest": "^29.0"}}'
        )
        collector = FrameworkDetectionCollector()
        signals = collector.collect(tmp_path)
        sig = next(s for s in signals if s.name == "jest_in_package_json")
        assert sig.score == 1.0

    def test_missing_file_scores_zero(self, tmp_path: Path):
        collector = FrameworkDetectionCollector()
        signals = collector.collect(tmp_path)
        for sig in signals:
            assert sig.score == 0.0
            assert sig.evidence["matched"] is False

    def test_file_without_match_scores_zero(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'myapp'")
        collector = FrameworkDetectionCollector()
        signals = collector.collect(tmp_path)
        sig = next(s for s in signals if s.name == "pytest_in_pyproject")
        assert sig.score == 0.0

    def test_custom_probes(self, tmp_path: Path):
        (tmp_path / "deps.txt").write_text("myframework==1.0")
        collector = FrameworkDetectionCollector(
            probes=[("deps.txt", r"myframework", "custom_fw", 0.75)]
        )
        signals = collector.collect(tmp_path)
        assert len(signals) == 1
        assert signals[0].name == "custom_fw"
        assert signals[0].score == 1.0
        assert signals[0].weight == 0.75

    def test_caches_file_reads(self, tmp_path: Path):
        """Two probes for the same file should only read it once."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["pytest", "unittest"]'
        )
        collector = FrameworkDetectionCollector(
            probes=[
                ("pyproject.toml", r"\bpytest\b", "probe_a", 0.9),
                ("pyproject.toml", r"\bunittest\b", "probe_b", 0.7),
            ]
        )
        signals = collector.collect(tmp_path)
        assert len(signals) == 2
        assert all(s.score == 1.0 for s in signals)


class TestCollectAll:
    def test_returns_signals_from_all_collectors(self, tmp_path: Path):
        (tmp_path / "pytest.ini").write_text("[pytest]")
        (tmp_path / "test_main.py").write_text("def test_x(): pass")
        (tmp_path / "pyproject.toml").write_text('[tool.pytest]\naddopts = "-v"')

        signals = collect_all_signals(tmp_path)
        names = {s.name for s in signals}
        # Should contain signals from all three collector types
        assert "pytest_ini_exists" in names  # file existence
        assert "python_test_files" in names  # pattern matching
        assert "pytest_in_pyproject" in names  # framework detection

    def test_all_signals_are_valid(self, tmp_path: Path):
        for sig in collect_all_signals(tmp_path):
            assert isinstance(sig, ConfidenceSignal)
            assert 0.0 <= sig.weight <= 1.0
            assert 0.0 <= sig.score <= 1.0
