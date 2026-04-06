"""Read-only safety enforcement for the troubleshooter agent.

Prevents any mutating operations during diagnostic investigation.
The guard validates tool calls and file system operations against
a blocklist of mutating patterns, ensuring the troubleshooter
remains strictly diagnose-only.

Design decisions:
- Blocklist approach: explicitly block known mutating patterns rather
  than allowlisting, so new read-only tools don't need registration
- Two-layer validation: command-level and path-level checks
- Immutable violation records for audit trail
- Configurable strictness via MutationPolicy
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mutating operation patterns
# ---------------------------------------------------------------------------

# Shell commands that mutate state
_MUTATING_COMMANDS: list[re.Pattern[str]] = [
    re.compile(r"\brm\b", re.IGNORECASE),
    re.compile(r"\bmv\b", re.IGNORECASE),
    re.compile(r"\bcp\b", re.IGNORECASE),
    re.compile(r"\bchmod\b", re.IGNORECASE),
    re.compile(r"\bchown\b", re.IGNORECASE),
    re.compile(r"\bmkdir\b", re.IGNORECASE),
    re.compile(r"\brmdir\b", re.IGNORECASE),
    re.compile(r"\btouch\b", re.IGNORECASE),
    re.compile(r"\btruncate\b", re.IGNORECASE),
    re.compile(r"\btee\b", re.IGNORECASE),
    re.compile(r"\bsed\s+-i\b", re.IGNORECASE),
    re.compile(r"\bawk\b.*\binplace\b", re.IGNORECASE),
    re.compile(r"\bgit\s+(commit|push|merge|rebase|reset|checkout|branch\s+-[dD])\b", re.IGNORECASE),
    re.compile(r"\bgit\s+rm\b", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bpip\s+uninstall\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+(install|uninstall|update)\b", re.IGNORECASE),
    re.compile(r"\byarn\s+(add|remove|upgrade)\b", re.IGNORECASE),
    re.compile(r"\bapt(-get)?\s+(install|remove|purge)\b", re.IGNORECASE),
    re.compile(r"\bdocker\s+(rm|rmi|kill|stop|exec|run)\b", re.IGNORECASE),
    re.compile(r"\bkubectl\s+(delete|apply|patch|edit|exec)\b", re.IGNORECASE),
    re.compile(r">>?\s", re.IGNORECASE),  # Shell redirects (> and >>)
    re.compile(r"\|\s*tee\b", re.IGNORECASE),
    re.compile(r"\bwrite\b", re.IGNORECASE),
    re.compile(r"\bdelete\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\binsert\s+into\b", re.IGNORECASE),
    re.compile(r"\bupdate\s+\w+\s+set\b", re.IGNORECASE),
    re.compile(r"\balter\s+table\b", re.IGNORECASE),
]

# Tool names that indicate mutation
_MUTATING_TOOL_NAMES: set[str] = {
    "write_file",
    "edit_file",
    "delete_file",
    "create_file",
    "rename_file",
    "move_file",
    "execute_command",
    "run_command",
    "shell_exec",
    "apply_fix",
    "auto_fix",
    "modify_config",
    "patch_file",
}

# File extensions that should never be written during diagnosis
_PROTECTED_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh", ".bat", ".ps1",
    ".toml", ".yaml", ".yml", ".json", ".xml", ".ini", ".cfg",
    ".env", ".conf",
}


class ViolationType(str, Enum):
    """Classification of safety violations."""

    MUTATING_COMMAND = "mutating_command"
    MUTATING_TOOL = "mutating_tool"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    PROTECTED_PATH = "protected_path"
    SHELL_REDIRECT = "shell_redirect"


class SafetyViolation(BaseModel, frozen=True):
    """Record of a blocked mutating operation.

    Attributes:
        violation_type: Classification of the violation.
        operation: The operation that was attempted.
        detail: Human-readable explanation of why it was blocked.
        matched_pattern: The pattern or rule that triggered the block.
        context: Additional context (e.g. tool name, file path).
    """

    violation_type: ViolationType
    operation: str
    detail: str
    matched_pattern: str = ""
    context: dict[str, Any] = Field(default_factory=dict)


class MutationPolicy(str, Enum):
    """How the safety guard handles detected mutations.

    BLOCK: Reject the operation and record a violation (default).
    WARN: Allow the operation but record a warning.
    AUDIT: Silently record the operation (for monitoring only).
    """

    BLOCK = "block"
    WARN = "warn"
    AUDIT = "audit"


@dataclass
class SafetyGuardConfig:
    """Configuration for the read-only safety guard.

    Attributes:
        policy: How to handle detected mutations.
        extra_blocked_commands: Additional regex patterns to block.
        extra_blocked_tools: Additional tool names to block.
        extra_protected_extensions: Additional file extensions to protect.
        allow_temp_writes: Whether to allow writes to /tmp and temp dirs.
    """

    policy: MutationPolicy = MutationPolicy.BLOCK
    extra_blocked_commands: list[str] = field(default_factory=list)
    extra_blocked_tools: list[str] = field(default_factory=list)
    extra_protected_extensions: list[str] = field(default_factory=list)
    allow_temp_writes: bool = False


class ReadOnlySafetyGuard:
    """Enforces read-only safety during troubleshooting diagnosis.

    Validates tool calls, commands, and file operations against a
    comprehensive blocklist of mutating patterns. Any detected mutation
    is blocked (by default) and recorded as a SafetyViolation.

    The guard maintains an audit trail of all violations for reporting
    and can be configured to warn-only or audit-only mode via the
    MutationPolicy.

    Usage::

        guard = ReadOnlySafetyGuard()

        # Check a tool call
        ok, violation = guard.validate_tool_call("read_file", {"path": "foo.py"})
        assert ok is True

        # Check a mutating tool call
        ok, violation = guard.validate_tool_call("write_file", {"path": "foo.py"})
        assert ok is False
        assert violation is not None

        # Check a shell command
        ok, violation = guard.validate_command("cat foo.py")
        assert ok is True

        ok, violation = guard.validate_command("rm -rf /")
        assert ok is False

    Thread-safety: NOT thread-safe. Use one guard per diagnostic session.
    """

    def __init__(self, config: SafetyGuardConfig | None = None) -> None:
        self._config = config or SafetyGuardConfig()
        self._violations: list[SafetyViolation] = []
        self._checks_performed: int = 0

        # Build effective blocked patterns
        self._command_patterns = list(_MUTATING_COMMANDS)
        for pattern_str in self._config.extra_blocked_commands:
            self._command_patterns.append(re.compile(pattern_str, re.IGNORECASE))

        self._blocked_tools = set(_MUTATING_TOOL_NAMES)
        self._blocked_tools.update(self._config.extra_blocked_tools)

        self._protected_extensions = set(_PROTECTED_EXTENSIONS)
        self._protected_extensions.update(self._config.extra_protected_extensions)

    # -- Properties -----------------------------------------------------------

    @property
    def config(self) -> SafetyGuardConfig:
        """Current guard configuration."""
        return self._config

    @property
    def policy(self) -> MutationPolicy:
        """Active mutation policy."""
        return self._config.policy

    @property
    def violations(self) -> list[SafetyViolation]:
        """All recorded safety violations (read-only copy)."""
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        """Number of safety violations recorded."""
        return len(self._violations)

    @property
    def checks_performed(self) -> int:
        """Total number of validation checks performed."""
        return self._checks_performed

    @property
    def has_violations(self) -> bool:
        """True if any violations have been recorded."""
        return len(self._violations) > 0

    # -- Validation methods ---------------------------------------------------

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> tuple[bool, SafetyViolation | None]:
        """Validate a tool call for read-only safety.

        Checks the tool name against the blocklist and validates
        arguments for file mutation patterns.

        Args:
            tool_name: Name of the tool being called.
            arguments: Tool call arguments (optional).

        Returns:
            (True, None) if the call is safe.
            (False, violation) if the call is blocked.
            In WARN/AUDIT mode, always returns (True, ...) but may
            record a violation.
        """
        self._checks_performed += 1
        args = arguments or {}

        # Check tool name
        tool_lower = tool_name.lower()
        if tool_lower in self._blocked_tools:
            violation = SafetyViolation(
                violation_type=ViolationType.MUTATING_TOOL,
                operation=f"tool:{tool_name}",
                detail=f"Tool '{tool_name}' is a mutating operation and is blocked "
                       f"during read-only diagnosis.",
                matched_pattern=tool_lower,
                context={"tool_name": tool_name, "arguments": args},
            )
            return self._handle_violation(violation)

        # Check for file path arguments that suggest writing
        for key in ("path", "file_path", "target", "destination", "output"):
            if key in args and isinstance(args[key], str):
                path = args[key]
                # Check if combined with a write-like action
                if any(w in tool_lower for w in ("write", "edit", "create", "delete", "modify", "patch")):
                    violation = SafetyViolation(
                        violation_type=ViolationType.FILE_WRITE,
                        operation=f"tool:{tool_name}({key}={path})",
                        detail=f"Tool '{tool_name}' appears to modify file '{path}'. "
                               f"File modifications are blocked during diagnosis.",
                        matched_pattern=f"tool_name contains write/edit/create/delete",
                        context={"tool_name": tool_name, "path": path},
                    )
                    return self._handle_violation(violation)

        return True, None

    def validate_command(self, command: str) -> tuple[bool, SafetyViolation | None]:
        """Validate a shell command for read-only safety.

        Checks the command against known mutating command patterns.

        Args:
            command: The shell command string to validate.

        Returns:
            (True, None) if the command is safe.
            (False, violation) if the command is blocked.
        """
        self._checks_performed += 1

        if not command or not command.strip():
            return True, None

        for pattern in self._command_patterns:
            match = pattern.search(command)
            if match:
                violation = SafetyViolation(
                    violation_type=ViolationType.MUTATING_COMMAND,
                    operation=f"command:{command[:200]}",
                    detail=f"Command contains mutating operation '{match.group()}'. "
                           f"Shell mutations are blocked during read-only diagnosis.",
                    matched_pattern=pattern.pattern,
                    context={"command": command[:500]},
                )
                return self._handle_violation(violation)

        return True, None

    def validate_file_write(
        self,
        file_path: str,
        operation: str = "write",
    ) -> tuple[bool, SafetyViolation | None]:
        """Validate a file write/delete operation.

        Always blocks file mutations during diagnosis unless the file
        is in a temp directory and allow_temp_writes is enabled.

        Args:
            file_path: Path to the file being modified.
            operation: Type of operation (write, delete, create, etc.).

        Returns:
            (True, None) if allowed (only in temp dirs with config).
            (False, violation) if blocked.
        """
        self._checks_performed += 1

        # Allow temp writes if configured
        if self._config.allow_temp_writes:
            import tempfile
            temp_dir = tempfile.gettempdir()
            resolved = os.path.abspath(file_path)
            if resolved.startswith(temp_dir):
                return True, None

        vtype = ViolationType.FILE_DELETE if operation == "delete" else ViolationType.FILE_WRITE
        violation = SafetyViolation(
            violation_type=vtype,
            operation=f"file_{operation}:{file_path}",
            detail=f"File {operation} on '{file_path}' is blocked during "
                   f"read-only diagnosis. The troubleshooter may only read files.",
            matched_pattern=f"file_{operation}",
            context={"file_path": file_path, "operation": operation},
        )
        return self._handle_violation(violation)

    # -- Internal helpers -----------------------------------------------------

    def _handle_violation(
        self,
        violation: SafetyViolation,
    ) -> tuple[bool, SafetyViolation | None]:
        """Handle a detected violation per the active policy.

        Returns:
            (False, violation) in BLOCK mode.
            (True, violation) in WARN mode (logs warning).
            (True, violation) in AUDIT mode (silent recording).
        """
        self._violations.append(violation)

        if self._config.policy == MutationPolicy.BLOCK:
            logger.warning(
                "SAFETY BLOCKED: %s — %s",
                violation.violation_type.value,
                violation.detail,
            )
            return False, violation

        if self._config.policy == MutationPolicy.WARN:
            logger.warning(
                "SAFETY WARNING (allowed): %s — %s",
                violation.violation_type.value,
                violation.detail,
            )
            return True, violation

        # AUDIT mode — silent recording
        logger.debug(
            "SAFETY AUDIT: %s — %s",
            violation.violation_type.value,
            violation.detail,
        )
        return True, violation

    # -- Reporting ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Summary of safety guard activity.

        Returns:
            Dict with check counts, violation counts, and violation details.
        """
        by_type: dict[str, int] = {}
        for v in self._violations:
            by_type[v.violation_type.value] = by_type.get(v.violation_type.value, 0) + 1

        return {
            "policy": self._config.policy.value,
            "checks_performed": self._checks_performed,
            "violations_total": len(self._violations),
            "violations_by_type": by_type,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "operation": v.operation,
                    "detail": v.detail,
                }
                for v in self._violations
            ],
        }

    def reset(self) -> None:
        """Reset violation records and check counter."""
        self._violations.clear()
        self._checks_performed = 0
