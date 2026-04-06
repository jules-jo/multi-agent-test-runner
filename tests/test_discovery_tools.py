"""Tests for discovery agent tools.

Tests call the raw _impl functions directly for unit testing.
The @function_tool wrappers are tested via the tool list validation.
"""

from __future__ import annotations

from pathlib import Path

from test_runner.tools.discovery_tools import (
    _scan_directory_impl as scan_directory,
    _read_file_impl as read_file,
    _run_help_impl as run_help,
    _detect_frameworks_impl as detect_frameworks,
    DISCOVERY_TOOLS,
)


class TestScanDirectory:
    def test_scan_existing_directory(self, tmp_path: Path) -> None:
        (tmp_path / "test_foo.py").write_text("pass")
        (tmp_path / "test_bar.py").write_text("pass")
        (tmp_path / "other.txt").write_text("hello")

        result = scan_directory(path=str(tmp_path), pattern="test_*.py")
        assert result["total_found"] == 2
        names = {f["name"] for f in result["files"]}
        assert names == {"test_foo.py", "test_bar.py"}

    def test_scan_nonexistent_directory(self) -> None:
        result = scan_directory(path="/nonexistent/path")
        assert "error" in result

    def test_scan_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "test_deep.py").write_text("pass")
        (tmp_path / "test_top.py").write_text("pass")

        result = scan_directory(path=str(tmp_path), pattern="test_*.py", recursive=True)
        assert result["total_found"] == 2

    def test_scan_non_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "test_deep.py").write_text("pass")
        (tmp_path / "test_top.py").write_text("pass")

        result = scan_directory(path=str(tmp_path), pattern="test_*.py", recursive=False)
        assert result["total_found"] == 1

    def test_max_results_cap(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"test_{i}.py").write_text("pass")

        result = scan_directory(path=str(tmp_path), pattern="test_*.py", max_results=3)
        assert result["total_found"] == 3

    def test_returns_file_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "test_meta.py").write_text("x = 1")
        result = scan_directory(path=str(tmp_path), pattern="test_*.py")
        f = result["files"][0]
        assert "path" in f
        assert "name" in f
        assert "size_bytes" in f
        assert "relative_path" in f


class TestReadFile:
    def test_read_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.md"
        f.write_text("# Test\nHello world")

        result = read_file(path=str(f))
        assert "# Test" in result["content"]
        assert result["total_lines"] == 2

    def test_read_nonexistent_file(self) -> None:
        result = read_file(path="/no/such/file.txt")
        assert "error" in result

    def test_read_truncation(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1000)))

        result = read_file(path=str(f), max_lines=10)
        assert result["truncated"] is True
        assert result["returned_lines"] == 10

    def test_read_not_a_file(self, tmp_path: Path) -> None:
        result = read_file(path=str(tmp_path))
        assert "error" in result


class TestRunHelp:
    def test_run_help_on_python(self) -> None:
        result = run_help(command="python3")
        assert result["return_code"] == 0
        assert "usage" in result["stdout"].lower() or "options" in result["stdout"].lower()

    def test_run_help_nonexistent_command(self) -> None:
        result = run_help(command="nonexistent_command_xyz_123")
        assert "error" in result or result.get("return_code", 1) != 0

    def test_run_help_appends_help_flag(self) -> None:
        result = run_help(command="echo")
        assert result["command"] == "echo --help"


class TestDetectFrameworks:
    def test_detect_pytest(self, tmp_path: Path) -> None:
        (tmp_path / "pytest.ini").write_text("[pytest]\n")

        result = detect_frameworks(project_path=str(tmp_path))
        frameworks = [f["framework"] for f in result["frameworks_detected"]]
        assert "pytest" in frameworks

    def test_detect_pytest_from_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
        )

        result = detect_frameworks(project_path=str(tmp_path))
        frameworks = [f["framework"] for f in result["frameworks_detected"]]
        assert "pytest" in frameworks

    def test_detect_jest(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            '{"devDependencies": {"jest": "^29.0.0"}}'
        )

        result = detect_frameworks(project_path=str(tmp_path))
        frameworks = [f["framework"] for f in result["frameworks_detected"]]
        assert "jest" in frameworks

    def test_detect_nothing(self, tmp_path: Path) -> None:
        result = detect_frameworks(project_path=str(tmp_path))
        assert result["total_detected"] == 0

    def test_detect_shell_scripts(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        (test_dir / "run_tests.sh").write_text("#!/bin/bash\necho test")

        result = detect_frameworks(project_path=str(tmp_path))
        frameworks = [f["framework"] for f in result["frameworks_detected"]]
        assert "shell_scripts" in frameworks

    def test_nonexistent_path(self) -> None:
        result = detect_frameworks(project_path="/nonexistent/path")
        assert "error" in result

    def test_confidence_scores_in_range(self, tmp_path: Path) -> None:
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        result = detect_frameworks(project_path=str(tmp_path))
        for fw in result["frameworks_detected"]:
            assert 0.0 <= fw["confidence"] <= 1.0


class TestDiscoveryToolsList:
    def test_all_tools_registered(self) -> None:
        assert len(DISCOVERY_TOOLS) == 4
        names = {t.name for t in DISCOVERY_TOOLS}
        assert names == {"scan_directory", "read_file", "run_help", "detect_frameworks"}
