"""Tests for the discovery agent step counter mechanism."""

from __future__ import annotations

from test_runner.agents.discovery.step_counter import (
    BUDGET_EXCEEDED_RESPONSE,
    DEFAULT_HARD_CAP,
    StepCounter,
)


class TestStepCounterDefaults:
    def test_default_hard_cap_is_20(self) -> None:
        assert DEFAULT_HARD_CAP == 20

    def test_initial_state(self) -> None:
        counter = StepCounter()
        assert counter.hard_cap == 20
        assert counter.steps_taken == 0
        assert counter.remaining == 20
        assert counter.is_exhausted is False
        assert counter.is_warning is False
        assert counter.step_log == []

    def test_custom_hard_cap(self) -> None:
        counter = StepCounter(hard_cap=10)
        assert counter.hard_cap == 10
        assert counter.remaining == 10


class TestStepCounterIncrement:
    def test_increment_returns_true_when_budget_available(self) -> None:
        counter = StepCounter(hard_cap=5)
        assert counter.increment("scan_directory") is True
        assert counter.steps_taken == 1
        assert counter.remaining == 4

    def test_increment_records_tool_name(self) -> None:
        counter = StepCounter(hard_cap=5)
        counter.increment("read_file", "path=/foo/bar.py")
        assert len(counter.step_log) == 1
        assert counter.step_log[0]["tool"] == "read_file"
        assert counter.step_log[0]["detail"] == "path=/foo/bar.py"
        assert counter.step_log[0]["step"] == 1

    def test_increment_without_detail(self) -> None:
        counter = StepCounter(hard_cap=5)
        counter.increment("detect_frameworks")
        assert "detail" not in counter.step_log[0]

    def test_increment_returns_false_when_exhausted(self) -> None:
        counter = StepCounter(hard_cap=2)
        assert counter.increment("step1") is True
        assert counter.increment("step2") is True
        assert counter.increment("step3") is False
        # steps_taken stays at 2 — rejected step is not counted
        assert counter.steps_taken == 2

    def test_multiple_increments(self) -> None:
        counter = StepCounter(hard_cap=20)
        for i in range(20):
            assert counter.increment(f"tool_{i}") is True
        assert counter.steps_taken == 20
        assert counter.is_exhausted is True
        assert counter.increment("one_more") is False


class TestStepCounterExhaustion:
    def test_exhaustion_at_exact_cap(self) -> None:
        counter = StepCounter(hard_cap=3)
        counter.increment("a")
        counter.increment("b")
        counter.increment("c")
        assert counter.is_exhausted is True
        assert counter.remaining == 0

    def test_not_exhausted_below_cap(self) -> None:
        counter = StepCounter(hard_cap=3)
        counter.increment("a")
        counter.increment("b")
        assert counter.is_exhausted is False
        assert counter.remaining == 1

    def test_zero_cap_is_immediately_exhausted(self) -> None:
        counter = StepCounter(hard_cap=0)
        assert counter.is_exhausted is True
        assert counter.increment("anything") is False


class TestStepCounterWarning:
    def test_warning_at_80_percent(self) -> None:
        counter = StepCounter(hard_cap=20)
        # 80% of 20 = 16 steps consumed, 4 remaining
        # Warning threshold = 20% of 20 = 4
        for _ in range(16):
            counter.increment("tool")
        assert counter.is_warning is True

    def test_no_warning_below_80_percent(self) -> None:
        counter = StepCounter(hard_cap=20)
        for _ in range(15):
            counter.increment("tool")
        assert counter.is_warning is False

    def test_warning_with_small_cap(self) -> None:
        # With cap=5, 20% = 1 step remaining triggers warning
        counter = StepCounter(hard_cap=5)
        for _ in range(4):
            counter.increment("tool")
        assert counter.is_warning is True


class TestStepCounterUsageFraction:
    def test_zero_usage(self) -> None:
        counter = StepCounter(hard_cap=20)
        assert counter.usage_fraction == 0.0

    def test_half_usage(self) -> None:
        counter = StepCounter(hard_cap=20)
        for _ in range(10):
            counter.increment("tool")
        assert counter.usage_fraction == 0.5

    def test_full_usage(self) -> None:
        counter = StepCounter(hard_cap=20)
        for _ in range(20):
            counter.increment("tool")
        assert counter.usage_fraction == 1.0

    def test_zero_cap_returns_one(self) -> None:
        counter = StepCounter(hard_cap=0)
        assert counter.usage_fraction == 1.0


class TestStepCounterReset:
    def test_reset_clears_state(self) -> None:
        counter = StepCounter(hard_cap=5)
        counter.increment("a")
        counter.increment("b")
        counter.reset()
        assert counter.steps_taken == 0
        assert counter.step_log == []
        assert counter.remaining == 5
        assert counter.is_exhausted is False

    def test_reset_preserves_hard_cap(self) -> None:
        counter = StepCounter(hard_cap=10)
        counter.increment("tool")
        counter.reset()
        assert counter.hard_cap == 10


class TestStepCounterSummary:
    def test_summary_structure(self) -> None:
        counter = StepCounter(hard_cap=20)
        counter.increment("scan_directory", "path=/app")
        s = counter.summary()
        assert s["steps_taken"] == 1
        assert s["hard_cap"] == 20
        assert s["remaining"] == 19
        assert s["is_exhausted"] is False
        assert 0.0 < s["usage_fraction"] < 1.0
        assert len(s["step_log"]) == 1

    def test_budget_status_message_ok(self) -> None:
        counter = StepCounter(hard_cap=20)
        counter.increment("tool")
        msg = counter.budget_status_message()
        assert "[BUDGET OK]" in msg
        assert "1/20" in msg

    def test_budget_status_message_warning(self) -> None:
        counter = StepCounter(hard_cap=5)
        for _ in range(4):
            counter.increment("tool")
        msg = counter.budget_status_message()
        assert "[BUDGET WARNING]" in msg

    def test_budget_status_message_exhausted(self) -> None:
        counter = StepCounter(hard_cap=2)
        counter.increment("a")
        counter.increment("b")
        msg = counter.budget_status_message()
        assert "[BUDGET EXHAUSTED]" in msg


class TestBudgetExceededResponse:
    def test_response_has_error_key(self) -> None:
        assert "error" in BUDGET_EXCEEDED_RESPONSE
        assert BUDGET_EXCEEDED_RESPONSE["error"] == "step_budget_exhausted"

    def test_response_has_message(self) -> None:
        assert "message" in BUDGET_EXCEEDED_RESPONSE
        assert "exhausted" in BUDGET_EXCEEDED_RESPONSE["message"].lower()
