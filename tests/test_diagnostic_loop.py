"""Tests for the diagnostic step loop and read-only safety enforcement.

Covers:
- DiagnosticLoop iteration counting and step guard integration
- Early-exit on root cause found with confidence threshold
- ReadOnlySafetyGuard blocking of mutating operations
- Safety violation handling (block, warn, audit policies)
- Consecutive failure handling
- Integration of loop + guard + safety
"""

from __future__ import annotations

from typing import Any

import pytest

from test_runner.agents.troubleshooter.diagnostic_loop import (
    ActionResult,
    ActionType,
    DiagnosticAction,
    DiagnosticLoop,
    DiagnosticLoopConfig,
    LoopExitReason,
    LoopResult,
)
from test_runner.agents.troubleshooter.safety_guard import (
    MutationPolicy,
    ReadOnlySafetyGuard,
    SafetyGuardConfig,
    SafetyViolation,
    ViolationType,
)
from test_runner.agents.troubleshooter.step_guard import (
    CompletionReason,
    DiagnosticStepGuard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action(
    desc: str = "test_action",
    target: str = "test.py",
    tool_name: str = "read_file",
    tool_args: dict[str, Any] | None = None,
    command: str = "",
    action_type: ActionType = ActionType.READ_FILE,
) -> DiagnosticAction:
    """Create a test DiagnosticAction."""
    return DiagnosticAction(
        action_type=action_type,
        description=desc,
        target=target,
        tool_name=tool_name,
        tool_args=tool_args or {},
        command=command,
    )


def _simple_executor(action: DiagnosticAction) -> tuple[str, float, dict[str, Any]]:
    """Simple executor that returns a finding with small confidence delta."""
    return f"Found something in {action.target}", 0.1, {}


def _root_cause_executor(action: DiagnosticAction) -> tuple[str, float, dict[str, Any]]:
    """Executor that signals a root cause on the 2nd call."""
    if not hasattr(_root_cause_executor, "_count"):
        _root_cause_executor._count = 0  # type: ignore[attr-defined]
    _root_cause_executor._count += 1  # type: ignore[attr-defined]
    if _root_cause_executor._count >= 2:  # type: ignore[attr-defined]
        return "Found root cause!", 0.5, {
            "root_cause": "Missing dependency 'requests'",
            "root_cause_confidence": 0.9,
            "proposed_fix": "pip install requests",
        }
    return "Checking logs...", 0.15, {}


def _failing_executor(action: DiagnosticAction) -> tuple[str, float, dict[str, Any]]:
    """Executor that always raises an exception."""
    raise RuntimeError("Tool execution failed")


def _make_loop(
    max_steps: int = 10,
    config: DiagnosticLoopConfig | None = None,
    safety_config: SafetyGuardConfig | None = None,
) -> DiagnosticLoop:
    """Create a DiagnosticLoop with fresh guards."""
    guard = DiagnosticStepGuard(max_steps=max_steps)
    safety = ReadOnlySafetyGuard(config=safety_config)
    return DiagnosticLoop(step_guard=guard, safety_guard=safety, config=config)


# ---------------------------------------------------------------------------
# ReadOnlySafetyGuard tests
# ---------------------------------------------------------------------------


class TestReadOnlySafetyGuardToolValidation:
    """Test tool call validation."""

    def test_read_file_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_tool_call("read_file", {"path": "foo.py"})
        assert ok is True
        assert violation is None

    def test_check_logs_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_tool_call("check_logs", {"path": "app.log"})
        assert ok is True

    def test_inspect_env_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_tool_call("inspect_env", {})
        assert ok is True

    def test_list_processes_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_tool_call("list_processes", {})
        assert ok is True

    def test_write_file_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_tool_call("write_file", {"path": "foo.py"})
        assert ok is False
        assert violation is not None
        assert violation.violation_type == ViolationType.MUTATING_TOOL

    def test_edit_file_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_tool_call("edit_file", {"path": "foo.py"})
        assert ok is False
        assert violation.violation_type == ViolationType.MUTATING_TOOL

    def test_delete_file_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_tool_call("delete_file", {"path": "foo.py"})
        assert ok is False
        assert violation.violation_type == ViolationType.MUTATING_TOOL

    def test_execute_command_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_tool_call("execute_command", {"cmd": "ls"})
        assert ok is False
        assert violation.violation_type == ViolationType.MUTATING_TOOL

    def test_apply_fix_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_tool_call("apply_fix", {})
        assert ok is False
        assert violation.violation_type == ViolationType.MUTATING_TOOL

    def test_custom_blocked_tool(self):
        config = SafetyGuardConfig(extra_blocked_tools=["my_dangerous_tool"])
        guard = ReadOnlySafetyGuard(config=config)
        ok, violation = guard.validate_tool_call("my_dangerous_tool", {})
        assert ok is False

    def test_checks_performed_counter(self):
        guard = ReadOnlySafetyGuard()
        assert guard.checks_performed == 0
        guard.validate_tool_call("read_file", {})
        assert guard.checks_performed == 1
        guard.validate_tool_call("write_file", {})
        assert guard.checks_performed == 2


class TestReadOnlySafetyGuardCommandValidation:
    """Test shell command validation."""

    def test_cat_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_command("cat foo.py")
        assert ok is True

    def test_ls_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_command("ls -la")
        assert ok is True

    def test_head_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_command("head -n 50 file.log")
        assert ok is True

    def test_grep_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_command("grep -n 'error' file.log")
        assert ok is True

    def test_rm_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_command("rm -rf /tmp/test")
        assert ok is False
        assert violation.violation_type == ViolationType.MUTATING_COMMAND

    def test_mv_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_command("mv foo.py bar.py")
        assert ok is False

    def test_chmod_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_command("chmod +x script.sh")
        assert ok is False

    def test_git_commit_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_command("git commit -m 'fix'")
        assert ok is False

    def test_git_push_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_command("git push origin main")
        assert ok is False

    def test_git_log_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_command("git log --oneline")
        assert ok is True

    def test_git_status_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_command("git status")
        assert ok is True

    def test_git_diff_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_command("git diff HEAD")
        assert ok is True

    def test_pip_install_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_command("pip install requests")
        assert ok is False

    def test_shell_redirect_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_command("echo 'test' > file.txt")
        assert ok is False

    def test_append_redirect_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_command("echo 'test' >> file.txt")
        assert ok is False

    def test_docker_run_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_command("docker run alpine")
        assert ok is False

    def test_empty_command_allowed(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_command("")
        assert ok is True

    def test_custom_blocked_pattern(self):
        config = SafetyGuardConfig(extra_blocked_commands=[r"\bmy_cmd\b"])
        guard = ReadOnlySafetyGuard(config=config)
        ok, _ = guard.validate_command("my_cmd --force")
        assert ok is False


class TestReadOnlySafetyGuardFileWrite:
    """Test file write validation."""

    def test_write_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_file_write("foo.py", "write")
        assert ok is False
        assert violation.violation_type == ViolationType.FILE_WRITE

    def test_delete_blocked(self):
        guard = ReadOnlySafetyGuard()
        ok, violation = guard.validate_file_write("foo.py", "delete")
        assert ok is False
        assert violation.violation_type == ViolationType.FILE_DELETE

    def test_temp_write_blocked_by_default(self):
        guard = ReadOnlySafetyGuard()
        ok, _ = guard.validate_file_write("/tmp/test.txt", "write")
        assert ok is False

    def test_temp_write_allowed_when_configured(self):
        import tempfile
        config = SafetyGuardConfig(allow_temp_writes=True)
        guard = ReadOnlySafetyGuard(config=config)
        temp_path = f"{tempfile.gettempdir()}/test_file.txt"
        ok, _ = guard.validate_file_write(temp_path, "write")
        assert ok is True

    def test_non_temp_write_blocked_even_when_temp_allowed(self):
        config = SafetyGuardConfig(allow_temp_writes=True)
        guard = ReadOnlySafetyGuard(config=config)
        ok, _ = guard.validate_file_write("/home/user/foo.py", "write")
        assert ok is False


class TestReadOnlySafetyGuardPolicies:
    """Test different mutation policies."""

    def test_block_policy_returns_false(self):
        config = SafetyGuardConfig(policy=MutationPolicy.BLOCK)
        guard = ReadOnlySafetyGuard(config=config)
        ok, violation = guard.validate_tool_call("write_file", {})
        assert ok is False
        assert violation is not None

    def test_warn_policy_returns_true(self):
        config = SafetyGuardConfig(policy=MutationPolicy.WARN)
        guard = ReadOnlySafetyGuard(config=config)
        ok, violation = guard.validate_tool_call("write_file", {})
        assert ok is True
        assert violation is not None
        assert guard.has_violations is True

    def test_audit_policy_returns_true(self):
        config = SafetyGuardConfig(policy=MutationPolicy.AUDIT)
        guard = ReadOnlySafetyGuard(config=config)
        ok, violation = guard.validate_tool_call("write_file", {})
        assert ok is True
        assert violation is not None
        assert guard.has_violations is True


class TestReadOnlySafetyGuardSummary:
    """Test summary and reporting."""

    def test_summary_empty(self):
        guard = ReadOnlySafetyGuard()
        s = guard.summary()
        assert s["checks_performed"] == 0
        assert s["violations_total"] == 0

    def test_summary_with_violations(self):
        guard = ReadOnlySafetyGuard()
        guard.validate_tool_call("write_file", {})
        guard.validate_command("rm -rf /")
        guard.validate_tool_call("read_file", {})  # allowed
        s = guard.summary()
        assert s["checks_performed"] == 3
        assert s["violations_total"] == 2
        assert ViolationType.MUTATING_TOOL.value in s["violations_by_type"]
        assert ViolationType.MUTATING_COMMAND.value in s["violations_by_type"]

    def test_reset(self):
        guard = ReadOnlySafetyGuard()
        guard.validate_tool_call("write_file", {})
        assert guard.violation_count == 1
        guard.reset()
        assert guard.violation_count == 0
        assert guard.checks_performed == 0


# ---------------------------------------------------------------------------
# DiagnosticLoop iteration counting tests
# ---------------------------------------------------------------------------


class TestDiagnosticLoopIterationCounting:
    """Test that the loop correctly counts iterations."""

    def test_counts_all_actions(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(5)]
        result = loop.run(actions, _simple_executor)
        assert result.iterations_completed == 5
        assert result.total_actions_provided == 5
        assert len(result.results) == 5

    def test_step_guard_tracks_steps(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(3)]
        loop.run(actions, _simple_executor)
        assert loop.step_guard.steps_taken == 3

    def test_each_result_has_iteration_number(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(3)]
        result = loop.run(actions, _simple_executor)
        for i, r in enumerate(result.results):
            assert r.iteration == i + 1

    def test_step_limit_stops_loop(self):
        loop = _make_loop(max_steps=3)
        actions = [_make_action(desc=f"step_{i}") for i in range(10)]
        result = loop.run(actions, _simple_executor)
        assert result.exit_reason == LoopExitReason.STEP_LIMIT_REACHED
        assert result.iterations_completed == 3
        assert len(result.results) == 3

    def test_empty_actions(self):
        loop = _make_loop(max_steps=10)
        result = loop.run([], _simple_executor)
        assert result.exit_reason == LoopExitReason.ALL_ACTIONS_PROCESSED
        assert result.iterations_completed == 0
        assert len(result.results) == 0

    def test_single_action(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc="only_step")]
        result = loop.run(actions, _simple_executor)
        assert result.iterations_completed == 1
        assert result.exit_reason == LoopExitReason.ALL_ACTIONS_PROCESSED

    def test_step_limit_of_one(self):
        loop = _make_loop(max_steps=1)
        actions = [_make_action(desc=f"step_{i}") for i in range(5)]
        result = loop.run(actions, _simple_executor)
        assert result.iterations_completed == 1
        assert result.exit_reason == LoopExitReason.STEP_LIMIT_REACHED


# ---------------------------------------------------------------------------
# DiagnosticLoop early-exit on root cause tests
# ---------------------------------------------------------------------------


class TestDiagnosticLoopEarlyExit:
    """Test early exit when root cause is found."""

    def test_early_exit_on_root_cause(self):
        """Loop exits early when executor signals root cause with high confidence."""
        # Reset the stateful executor
        if hasattr(_root_cause_executor, "_count"):
            del _root_cause_executor._count

        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(10)]
        result = loop.run(actions, _root_cause_executor)

        # Should exit after finding root cause (step 2 signals it,
        # loop checks before step 3)
        assert result.exit_reason == LoopExitReason.ROOT_CAUSE_FOUND
        assert result.root_cause == "Missing dependency 'requests'"
        assert result.confidence >= 0.8
        assert result.iterations_completed < 10

    def test_early_exit_respects_threshold(self):
        """Loop only exits early when confidence >= threshold."""
        config = DiagnosticLoopConfig(root_cause_confidence_threshold=0.95)
        loop = _make_loop(max_steps=10, config=config)

        # Executor that gives moderate confidence root cause
        def moderate_executor(action: DiagnosticAction) -> tuple[str, float, dict[str, Any]]:
            return "Found something", 0.2, {
                "root_cause": "Possible issue",
                "root_cause_confidence": 0.7,
            }

        actions = [_make_action(desc=f"step_{i}") for i in range(5)]
        result = loop.run(actions, moderate_executor)

        # Should NOT early-exit because confidence < 0.95 threshold
        assert result.exit_reason == LoopExitReason.ALL_ACTIONS_PROCESSED
        assert result.iterations_completed == 5

    def test_set_root_cause_externally(self):
        """Root cause can be set externally before or during the loop."""
        loop = _make_loop(max_steps=10)
        loop.set_root_cause("Known issue", confidence=0.9)

        actions = [_make_action(desc=f"step_{i}") for i in range(5)]
        result = loop.run(actions, _simple_executor)

        # Should exit immediately because root cause already set
        assert result.exit_reason == LoopExitReason.ROOT_CAUSE_FOUND
        assert result.iterations_completed == 0

    def test_root_cause_found_property(self):
        loop = _make_loop(max_steps=10)
        assert loop.root_cause_found is False

        loop.set_root_cause("An issue", confidence=0.5)
        # Below default threshold of 0.80
        assert loop.root_cause_found is False

        loop.set_root_cause("An issue", confidence=0.85)
        assert loop.root_cause_found is True

    def test_root_cause_requires_both_cause_and_confidence(self):
        loop = _make_loop(max_steps=10)
        # High confidence but no root cause text
        loop.set_root_cause("", confidence=0.9)
        assert loop.root_cause_found is False

    def test_early_exit_produces_diagnosis_summary(self):
        """When early-exiting, a DiagnosisSummary is produced."""
        if hasattr(_root_cause_executor, "_count"):
            del _root_cause_executor._count

        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(10)]
        result = loop.run(actions, _root_cause_executor)

        assert result.diagnosis_summary is not None
        assert result.diagnosis_summary.root_cause == "Missing dependency 'requests'"
        assert result.diagnosis_summary.completion_reason == CompletionReason.COMPLETED

    def test_proposed_fixes_accumulated(self):
        """Proposed fixes from executor are accumulated."""
        if hasattr(_root_cause_executor, "_count"):
            del _root_cause_executor._count

        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(10)]
        result = loop.run(actions, _root_cause_executor)

        assert result.diagnosis_summary is not None
        assert "pip install requests" in result.diagnosis_summary.proposed_fixes


# ---------------------------------------------------------------------------
# DiagnosticLoop read-only safety enforcement tests
# ---------------------------------------------------------------------------


class TestDiagnosticLoopSafetyEnforcement:
    """Test that the loop enforces read-only safety."""

    def test_mutating_tool_blocked(self):
        loop = _make_loop(max_steps=10)
        actions = [
            _make_action(desc="write something", tool_name="write_file"),
        ]
        result = loop.run(actions, _simple_executor)
        assert result.exit_reason == LoopExitReason.SAFETY_VIOLATION
        assert len(result.safety_violations) == 1
        assert result.results[0].blocked is True
        assert result.iterations_completed == 0

    def test_mutating_command_blocked(self):
        loop = _make_loop(max_steps=10)
        actions = [
            _make_action(desc="delete files", tool_name="", command="rm -rf /tmp/test"),
        ]
        result = loop.run(actions, _simple_executor)
        assert result.exit_reason == LoopExitReason.SAFETY_VIOLATION
        assert len(result.safety_violations) == 1
        assert result.results[0].blocked is True

    def test_stop_on_safety_violation_default(self):
        """Default behavior: stop the loop on safety violation."""
        loop = _make_loop(max_steps=10)
        actions = [
            _make_action(desc="read first", tool_name="read_file"),
            _make_action(desc="write bad", tool_name="write_file"),
            _make_action(desc="read third", tool_name="read_file"),
        ]
        result = loop.run(actions, _simple_executor)
        assert result.exit_reason == LoopExitReason.SAFETY_VIOLATION
        # First action succeeds, second is blocked and stops the loop
        assert len(result.results) == 2
        assert result.results[0].success is True
        assert result.results[1].blocked is True

    def test_continue_on_safety_violation(self):
        """When configured, skip blocked actions and continue."""
        config = DiagnosticLoopConfig(stop_on_safety_violation=False)
        loop = _make_loop(max_steps=10, config=config)
        actions = [
            _make_action(desc="read first", tool_name="read_file"),
            _make_action(desc="write bad", tool_name="write_file"),
            _make_action(desc="read third", tool_name="read_file"),
        ]
        result = loop.run(actions, _simple_executor)
        assert result.exit_reason == LoopExitReason.ALL_ACTIONS_PROCESSED
        assert len(result.results) == 3
        assert result.results[0].success is True
        assert result.results[1].blocked is True
        assert result.results[2].success is True
        # 2 successful iterations (blocked doesn't count by default)
        assert result.iterations_completed == 2

    def test_blocked_actions_dont_count_as_steps_by_default(self):
        """Blocked actions should not consume step budget by default."""
        config = DiagnosticLoopConfig(
            stop_on_safety_violation=False,
            record_blocked_as_step=False,
        )
        loop = _make_loop(max_steps=3, config=config)
        actions = [
            _make_action(desc="read", tool_name="read_file"),
            _make_action(desc="bad", tool_name="write_file"),
            _make_action(desc="read2", tool_name="read_file"),
            _make_action(desc="read3", tool_name="read_file"),
        ]
        result = loop.run(actions, _simple_executor)
        # 3 successful reads + 1 blocked = 4 results
        assert len(result.results) == 4
        assert loop.step_guard.steps_taken == 3

    def test_blocked_actions_count_when_configured(self):
        """Blocked actions consume step budget when configured."""
        config = DiagnosticLoopConfig(
            stop_on_safety_violation=False,
            record_blocked_as_step=True,
        )
        loop = _make_loop(max_steps=3, config=config)
        actions = [
            _make_action(desc="read", tool_name="read_file"),
            _make_action(desc="bad", tool_name="write_file"),
            _make_action(desc="read2", tool_name="read_file"),
            _make_action(desc="read3", tool_name="read_file"),
        ]
        result = loop.run(actions, _simple_executor)
        # 3 steps consumed (1 read + 1 blocked + 1 read), then limit reached
        assert loop.step_guard.steps_taken == 3
        assert result.exit_reason == LoopExitReason.STEP_LIMIT_REACHED

    def test_safety_violation_produces_diagnosis_summary(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc="bad", tool_name="write_file")]
        result = loop.run(actions, _simple_executor)
        assert result.diagnosis_summary is not None
        assert result.diagnosis_summary.completion_reason == CompletionReason.ERROR


# ---------------------------------------------------------------------------
# DiagnosticLoop error handling tests
# ---------------------------------------------------------------------------


class TestDiagnosticLoopErrorHandling:
    """Test error handling and consecutive failure limits."""

    def test_single_failure_continues(self):
        call_count = 0

        def mixed_executor(action: DiagnosticAction) -> tuple[str, float, dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("One-off failure")
            return "ok", 0.1, {}

        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(4)]
        result = loop.run(actions, mixed_executor)
        assert result.exit_reason == LoopExitReason.ALL_ACTIONS_PROCESSED
        assert result.iterations_completed == 3  # 3 success + 1 failure
        assert result.results[1].success is False
        assert "One-off failure" in result.results[1].error

    def test_consecutive_failures_abort(self):
        config = DiagnosticLoopConfig(max_consecutive_failures=2)
        loop = _make_loop(max_steps=10, config=config)
        actions = [_make_action(desc=f"step_{i}") for i in range(5)]
        result = loop.run(actions, _failing_executor)
        assert result.exit_reason == LoopExitReason.ERROR
        assert result.iterations_completed == 0
        # Should have 2 failed results (the limit)
        assert len(result.results) == 2

    def test_consecutive_counter_resets_on_success(self):
        call_count = 0

        def alternating_executor(action: DiagnosticAction) -> tuple[str, float, dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("Intermittent failure")
            return "ok", 0.1, {}

        config = DiagnosticLoopConfig(max_consecutive_failures=2)
        loop = _make_loop(max_steps=10, config=config)
        actions = [_make_action(desc=f"step_{i}") for i in range(6)]
        result = loop.run(actions, alternating_executor)
        # Alternating success/fail should not trigger consecutive limit
        assert result.exit_reason == LoopExitReason.ALL_ACTIONS_PROCESSED


# ---------------------------------------------------------------------------
# DiagnosticLoop confidence tracking tests
# ---------------------------------------------------------------------------


class TestDiagnosticLoopConfidenceTracking:
    """Test confidence accumulation across iterations."""

    def test_confidence_accumulates(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(3)]
        result = loop.run(actions, _simple_executor)
        # Each step adds 0.1 confidence
        assert result.confidence == pytest.approx(0.3, abs=0.01)

    def test_confidence_clamped_to_one(self):
        def high_conf_executor(action: DiagnosticAction) -> tuple[str, float, dict[str, Any]]:
            return "big finding", 0.5, {}

        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(5)]
        result = loop.run(actions, high_conf_executor)
        assert result.confidence == 1.0

    def test_confidence_from_root_cause_signal(self):
        if hasattr(_root_cause_executor, "_count"):
            del _root_cause_executor._count

        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc=f"step_{i}") for i in range(5)]
        result = loop.run(actions, _root_cause_executor)
        assert result.confidence >= 0.8


# ---------------------------------------------------------------------------
# DiagnosticLoop finalization tests
# ---------------------------------------------------------------------------


class TestDiagnosticLoopFinalization:
    """Test that the loop produces proper finalized summaries."""

    def test_all_processed_produces_summary(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc="step_1")]
        result = loop.run(actions, _simple_executor)
        assert result.diagnosis_summary is not None
        assert result.diagnosis_summary.completion_reason == CompletionReason.COMPLETED
        assert result.diagnosis_summary.total_steps == 1

    def test_step_limit_produces_summary(self):
        loop = _make_loop(max_steps=2)
        actions = [_make_action(desc=f"step_{i}") for i in range(5)]
        result = loop.run(actions, _simple_executor)
        assert result.diagnosis_summary is not None
        assert result.diagnosis_summary.completion_reason == CompletionReason.LIMIT_REACHED
        assert result.diagnosis_summary.was_truncated is True

    def test_error_produces_summary(self):
        config = DiagnosticLoopConfig(max_consecutive_failures=1)
        loop = _make_loop(max_steps=10, config=config)
        actions = [_make_action(desc="fail")]
        result = loop.run(actions, _failing_executor)
        assert result.diagnosis_summary is not None
        assert result.diagnosis_summary.completion_reason == CompletionReason.ERROR

    def test_summary_has_timestamps(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc="step")]
        result = loop.run(actions, _simple_executor)
        assert result.diagnosis_summary is not None
        assert result.diagnosis_summary.start_time > 0
        assert result.diagnosis_summary.end_time > 0
        assert result.diagnosis_summary.duration_seconds >= 0

    def test_loop_elapsed_ms(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc="step")]
        result = loop.run(actions, _simple_executor)
        assert result.elapsed_ms >= 0

    def test_action_results_have_elapsed_ms(self):
        loop = _make_loop(max_steps=10)
        actions = [_make_action(desc="step")]
        result = loop.run(actions, _simple_executor)
        assert result.results[0].elapsed_ms >= 0


# ---------------------------------------------------------------------------
# DiagnosticLoop reset and reuse tests
# ---------------------------------------------------------------------------


class TestDiagnosticLoopReset:
    """Test loop reset and reuse."""

    def test_reset_clears_state(self):
        loop = _make_loop(max_steps=10)
        loop.set_root_cause("test", confidence=0.9)
        loop.add_proposed_fix("fix1")
        loop.add_alternative_cause("alt1")
        assert loop.root_cause_found is True

        loop.reset()
        assert loop.running_confidence == 0.0
        assert loop.root_cause == ""
        assert loop.root_cause_found is False


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDiagnosticLoopIntegration:
    """End-to-end integration tests for the diagnostic loop."""

    def test_full_diagnosis_flow(self):
        """Simulate a complete diagnostic session."""
        loop = _make_loop(max_steps=10)
        step = 0

        def diagnosis_executor(action: DiagnosticAction) -> tuple[str, float, dict[str, Any]]:
            nonlocal step
            step += 1
            if step == 1:
                return "Found ImportError in test output", 0.2, {}
            if step == 2:
                return "Module 'requests' not installed", 0.3, {}
            if step == 3:
                return "requirements.txt missing 'requests'", 0.2, {
                    "root_cause": "Missing dependency 'requests' in requirements.txt",
                    "root_cause_confidence": 0.85,
                    "proposed_fix": "Add 'requests' to requirements.txt",
                }
            return "No additional info", 0.0, {}

        actions = [
            _make_action(
                desc="read_test_output",
                target="test_output.log",
                tool_name="check_logs",
            ),
            _make_action(
                desc="check_installed_packages",
                target="pip_list",
                tool_name="inspect_env",
            ),
            _make_action(
                desc="read_requirements",
                target="requirements.txt",
                tool_name="read_file",
            ),
            _make_action(desc="extra_check_1", tool_name="read_file"),
            _make_action(desc="extra_check_2", tool_name="read_file"),
        ]

        result = loop.run(actions, diagnosis_executor)

        # Should exit early after step 3 (root cause found at step 3,
        # early-exit check at start of step 4)
        assert result.exit_reason == LoopExitReason.ROOT_CAUSE_FOUND
        assert result.iterations_completed == 3
        assert result.root_cause == "Missing dependency 'requests' in requirements.txt"
        assert result.confidence >= 0.80
        assert result.diagnosis_summary is not None
        assert result.diagnosis_summary.is_conclusive is True
        assert "requests" in result.diagnosis_summary.root_cause

    def test_mixed_safe_and_unsafe_actions(self):
        """Loop handles a mix of safe and unsafe actions."""
        config = DiagnosticLoopConfig(stop_on_safety_violation=False)
        loop = _make_loop(max_steps=10, config=config)

        actions = [
            _make_action(desc="read_logs", tool_name="check_logs"),
            _make_action(desc="try_write", tool_name="write_file"),
            _make_action(desc="read_src", tool_name="read_file"),
            _make_action(desc="try_delete", tool_name="delete_file"),
            _make_action(desc="read_env", tool_name="inspect_env"),
        ]

        result = loop.run(actions, _simple_executor)
        assert result.exit_reason == LoopExitReason.ALL_ACTIONS_PROCESSED
        assert result.iterations_completed == 3  # 3 safe, 2 blocked
        assert len(result.safety_violations) == 2
        assert result.results[0].success is True
        assert result.results[1].blocked is True
        assert result.results[2].success is True
        assert result.results[3].blocked is True
        assert result.results[4].success is True

    def test_step_guard_auto_starts(self):
        """Loop auto-starts the step guard if not already started."""
        guard = DiagnosticStepGuard(max_steps=10)
        safety = ReadOnlySafetyGuard()
        loop = DiagnosticLoop(step_guard=guard, safety_guard=safety)
        assert guard.is_started is False

        actions = [_make_action(desc="step_1")]
        loop.run(actions, _simple_executor)
        # Guard should have been auto-started and finalized
        assert guard.is_finalized is True
