"""Read-only tool definitions for the Troubleshooter agent.

These tools enable diagnosing test failures without modifying anything:
- read_file: read file contents (source files, configs, logs)
- check_logs: read recent log files or stderr output
- inspect_env: inspect environment variables and runtime config
- list_processes: list running processes to detect resource issues

Each tool is defined as a plain function (for testability) and then
wrapped with @function_tool for the OpenAI Agents SDK registration.

IMPORTANT: All tools are strictly read-only. The troubleshooter agent
proposes fixes but never auto-executes them.
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


def _read_file_impl(path: str, max_lines: int = 500) -> dict[str, Any]:
    """Read file contents for troubleshooting inspection.

    Reads source files, configs, logs, or any text file to help
    diagnose test failures. Read-only — never modifies the file.
    """
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


def _check_logs_impl(
    path: str,
    tail_lines: int = 100,
    pattern: str | None = None,
) -> dict[str, Any]:
    """Read recent log entries from a file.

    Optionally filters lines by a grep-like pattern. Reads from the
    end of the file (tail behavior) for efficiency with large logs.
    """
    target = Path(path).resolve()
    if not target.exists():
        return {"error": f"Log file does not exist: {path}"}
    if not target.is_file():
        return {"error": f"Path is not a file: {path}"}

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        # Take the last N lines (tail behavior)
        selected = all_lines[-tail_lines:] if len(all_lines) > tail_lines else all_lines

        if pattern:
            pattern_lower = pattern.lower()
            selected = [line for line in selected if pattern_lower in line.lower()]

        return {
            "path": str(target),
            "total_lines": len(all_lines),
            "returned_lines": len(selected),
            "tail_lines_requested": tail_lines,
            "pattern": pattern,
            "content": "\n".join(selected),
        }
    except Exception as e:
        return {"error": f"Failed to read log file: {e}"}


def _inspect_env_impl(
    filter_prefix: str | None = None,
    include_python: bool = True,
) -> dict[str, Any]:
    """Inspect environment variables and runtime configuration.

    Returns environment variables, optionally filtered by prefix.
    Sensitive values (keys containing SECRET, TOKEN, PASSWORD, KEY)
    are masked for safety.
    """
    env_vars: dict[str, str] = {}
    sensitive_keywords = {"secret", "token", "password", "key", "credential", "auth"}

    for key, value in sorted(os.environ.items()):
        if filter_prefix and not key.lower().startswith(filter_prefix.lower()):
            continue
        # Mask sensitive values
        if any(kw in key.lower() for kw in sensitive_keywords):
            env_vars[key] = "****MASKED****"
        else:
            env_vars[key] = value

    result: dict[str, Any] = {
        "filter_prefix": filter_prefix,
        "total_vars": len(env_vars),
        "variables": env_vars,
    }

    if include_python:
        import sys

        result["python"] = {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path[:10],  # Limit to avoid huge output
            "platform": sys.platform,
        }

    return result


def _list_processes_impl(
    filter_pattern: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List running processes to detect resource issues.

    Optionally filters by a command name pattern. Useful for detecting
    zombie test processes, port conflicts, or resource exhaustion.
    """
    try:
        # Use ps for cross-platform process listing
        cmd = ["ps", "aux", "--no-headers"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            # Fallback for systems without --no-headers
            cmd_fallback = ["ps", "aux"]
            result = subprocess.run(
                cmd_fallback,
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = result.stdout.strip().splitlines()[1:]  # Skip header
        else:
            lines = result.stdout.strip().splitlines()

        processes: list[dict[str, str]] = []
        for line in lines:
            if filter_pattern and filter_pattern.lower() not in line.lower():
                continue
            parts = line.split(None, 10)
            if len(parts) >= 11:
                processes.append({
                    "user": parts[0],
                    "pid": parts[1],
                    "cpu": parts[2],
                    "mem": parts[3],
                    "command": parts[10],
                })
            if len(processes) >= limit:
                break

        return {
            "filter_pattern": filter_pattern,
            "total_shown": len(processes),
            "processes": processes,
        }
    except FileNotFoundError:
        return {"error": "ps command not found on this system"}
    except subprocess.TimeoutExpired:
        return {"error": "Process listing timed out"}
    except Exception as e:
        return {"error": f"Failed to list processes: {e}"}


# ---------------------------------------------------------------------------
# SDK-wrapped tools (registered with OpenAI Agents SDK)
# ---------------------------------------------------------------------------


@function_tool
def read_file(path: str, max_lines: int = 500) -> dict[str, Any]:
    """Read file contents for troubleshooting (source, config, logs).

    Read-only inspection of any text file to help diagnose test failures.

    Args:
        path: File path to read.
        max_lines: Maximum number of lines to return.

    Returns:
        Dictionary with file content and metadata.
    """
    return _read_file_impl(path, max_lines)


@function_tool
def check_logs(
    path: str,
    tail_lines: int = 100,
    pattern: str = "",
) -> dict[str, Any]:
    """Read recent log entries from a file, optionally filtered by pattern.

    Reads from the end of the file (tail behavior) for efficiency.
    Useful for inspecting test output logs, stderr captures, or
    application logs to understand failure context.

    Args:
        path: Path to the log file.
        tail_lines: Number of lines to read from the end.
        pattern: Optional case-insensitive filter pattern.

    Returns:
        Dictionary with matching log lines and metadata.
    """
    return _check_logs_impl(path, tail_lines, pattern or None)


@function_tool
def inspect_env(
    filter_prefix: str = "",
    include_python: bool = True,
) -> dict[str, Any]:
    """Inspect environment variables and Python runtime configuration.

    Returns environment variables (optionally filtered by prefix).
    Sensitive values (SECRET, TOKEN, PASSWORD, KEY) are masked.
    Includes Python version, executable path, and sys.path.

    Args:
        filter_prefix: Only return vars starting with this prefix.
        include_python: Include Python runtime info.

    Returns:
        Dictionary with environment variables and runtime details.
    """
    return _inspect_env_impl(filter_prefix or None, include_python)


@function_tool
def list_processes(
    filter_pattern: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """List running processes to detect resource issues or conflicts.

    Useful for finding zombie test processes, port conflicts, or
    resource exhaustion. Optionally filters by command name pattern.

    Args:
        filter_pattern: Optional filter to match against process commands.
        limit: Maximum number of processes to return.

    Returns:
        Dictionary with process list and metadata.
    """
    return _list_processes_impl(filter_pattern or None, limit)


# Collect all troubleshooter tools for registration
TROUBLESHOOTER_TOOLS = [read_file, check_logs, inspect_env, list_processes]
