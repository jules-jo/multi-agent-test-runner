"""Tool definitions for the Discovery agent.

These tools enable exploring test scripts and project structure:
- scan_directory: recursive directory scanning for test files
- read_file: read file contents (docs, configs, test files)
- run_help: execute a script/tool with --help to discover usage
- detect_frameworks: detect known test frameworks from project files

Each tool is defined as a plain function (for testability) and then
wrapped with @function_tool for the OpenAI Agents SDK registration.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from agents import function_tool


# ---------------------------------------------------------------------------
# Raw implementations (directly testable)
# ---------------------------------------------------------------------------


def _scan_directory_impl(
    path: str,
    pattern: str = "*",
    recursive: bool = True,
    max_results: int = 200,
) -> dict[str, Any]:
    """Scan a directory for files matching a glob pattern."""
    target = Path(path).resolve()
    if not target.exists():
        return {"error": f"Path does not exist: {path}", "files": []}
    if not target.is_dir():
        return {"error": f"Path is not a directory: {path}", "files": []}

    glob_method = target.rglob if recursive else target.glob
    files: list[dict[str, Any]] = []
    for f in glob_method(pattern):
        if f.is_file() and len(files) < max_results:
            try:
                stat = f.stat()
                files.append(
                    {
                        "path": str(f),
                        "name": f.name,
                        "size_bytes": stat.st_size,
                        "relative_path": str(f.relative_to(target)),
                    }
                )
            except (OSError, ValueError):
                continue

    return {
        "base_path": str(target),
        "pattern": pattern,
        "recursive": recursive,
        "total_found": len(files),
        "files": files,
    }


def _read_file_impl(path: str, max_lines: int = 500) -> dict[str, Any]:
    """Read file contents for inspection."""
    target = Path(path).resolve()
    if not target.exists():
        return {"error": f"File does not exist: {path}"}
    if not target.is_file():
        return {"error": f"Path is not a file: {path}"}

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        truncated = len(lines) > max_lines
        if truncated:
            lines = lines[:max_lines]
        return {
            "path": str(target),
            "total_lines": len(content.splitlines()),
            "returned_lines": len(lines),
            "truncated": truncated,
            "content": "\n".join(lines),
        }
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}


def _run_help_impl(command: str, timeout_seconds: int = 15) -> dict[str, Any]:
    """Execute a command with --help to discover usage."""
    full_cmd = f"{command} --help"

    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ, "NO_COLOR": "1"},
        )
        return {
            "command": full_cmd,
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "command": full_cmd,
            "error": f"Command timed out after {timeout_seconds}s",
        }
    except Exception as e:
        return {"command": full_cmd, "error": f"Failed to execute: {e}"}


def _detect_frameworks_impl(project_path: str) -> dict[str, Any]:
    """Detect known test frameworks by inspecting project configuration files."""
    root = Path(project_path).resolve()
    if not root.exists():
        return {"error": f"Path does not exist: {project_path}"}

    detected: list[dict[str, Any]] = []

    # Python: pytest
    indicators: dict[str, float | None] = {
        "pytest.ini": 0.95,
        "pyproject.toml": None,
        "setup.cfg": None,
        "conftest.py": 0.9,
        "tox.ini": 0.7,
    }
    for name, conf in indicators.items():
        found = list(root.rglob(name))
        if found:
            if conf is not None:
                detected.append(
                    {
                        "framework": "pytest",
                        "confidence": conf,
                        "evidence": str(found[0]),
                    }
                )
            else:
                try:
                    content = found[0].read_text(errors="replace")
                    if "pytest" in content.lower():
                        detected.append(
                            {
                                "framework": "pytest",
                                "confidence": 0.85,
                                "evidence": f"{found[0]} contains pytest reference",
                            }
                        )
                except OSError:
                    pass

    # Python: unittest
    test_files = list(root.rglob("test_*.py")) + list(root.rglob("*_test.py"))
    if test_files:
        for tf in test_files[:5]:
            try:
                content = tf.read_text(errors="replace")
                if "import unittest" in content:
                    detected.append(
                        {
                            "framework": "unittest",
                            "confidence": 0.9,
                            "evidence": f"{tf} imports unittest",
                        }
                    )
                    break
            except OSError:
                pass

    # JavaScript: jest / mocha / vitest
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            content = pkg_json.read_text(errors="replace")
            if "jest" in content:
                detected.append(
                    {"framework": "jest", "confidence": 0.9, "evidence": "package.json references jest"}
                )
            if "mocha" in content:
                detected.append(
                    {"framework": "mocha", "confidence": 0.85, "evidence": "package.json references mocha"}
                )
            if "vitest" in content:
                detected.append(
                    {"framework": "vitest", "confidence": 0.9, "evidence": "package.json references vitest"}
                )
        except OSError:
            pass

    # Go test
    go_mod = root / "go.mod"
    if go_mod.exists():
        go_test_files = list(root.rglob("*_test.go"))
        if go_test_files:
            detected.append(
                {
                    "framework": "go_test",
                    "confidence": 0.95,
                    "evidence": f"go.mod + {len(go_test_files)} test files",
                }
            )

    # Rust: cargo test
    cargo_toml = root / "Cargo.toml"
    if cargo_toml.exists():
        detected.append(
            {"framework": "cargo_test", "confidence": 0.85, "evidence": "Cargo.toml found"}
        )

    # Shell scripts in test directories
    test_dirs_found = [
        d for d in root.iterdir()
        if d.is_dir() and d.name.lower() in ("test", "tests", "spec", "specs")
    ]
    shell_scripts: list[Path] = []
    for d in test_dirs_found:
        shell_scripts.extend(d.rglob("*.sh"))
    if shell_scripts:
        detected.append(
            {
                "framework": "shell_scripts",
                "confidence": 0.7,
                "evidence": f"{len(shell_scripts)} .sh files in test directories",
            }
        )

    return {
        "project_path": str(root),
        "frameworks_detected": detected,
        "total_detected": len(detected),
    }


# ---------------------------------------------------------------------------
# SDK-wrapped tools (registered with OpenAI Agents SDK)
# ---------------------------------------------------------------------------


@function_tool
def scan_directory(
    path: str,
    pattern: str = "*",
    recursive: bool = True,
    max_results: int = 200,
) -> dict[str, Any]:
    """Scan a directory for files matching a glob pattern.

    Args:
        path: Directory path to scan (absolute or relative to project root).
        pattern: Glob pattern to match files (e.g. 'test_*.py', '*.spec.js').
        recursive: Whether to scan subdirectories.
        max_results: Maximum number of results to return.

    Returns:
        Dictionary with matched files and metadata.
    """
    return _scan_directory_impl(path, pattern, recursive, max_results)


@function_tool
def read_file(path: str, max_lines: int = 500) -> dict[str, Any]:
    """Read file contents for inspection (docs, configs, test scripts).

    Args:
        path: File path to read.
        max_lines: Maximum number of lines to return.

    Returns:
        Dictionary with file content and metadata.
    """
    return _read_file_impl(path, max_lines)


@function_tool
def run_help(command: str, timeout_seconds: int = 15) -> dict[str, Any]:
    """Execute a command with --help to discover its usage and options.

    Args:
        command: The command/script to run (e.g. 'pytest', 'npm test', './run_tests.sh').
        timeout_seconds: Timeout for the command execution.

    Returns:
        Dictionary with stdout, stderr, and return code.
    """
    return _run_help_impl(command, timeout_seconds)


@function_tool
def detect_frameworks(project_path: str) -> dict[str, Any]:
    """Detect known test frameworks by inspecting project configuration files.

    Checks for pytest, unittest, jest, mocha, go test, cargo test, etc.

    Args:
        project_path: Root path of the project to inspect.

    Returns:
        Dictionary with detected frameworks and confidence scores.
    """
    return _detect_frameworks_impl(project_path)


# Collect all discovery tools for registration
DISCOVERY_TOOLS = [scan_directory, read_file, run_help, detect_frameworks]
