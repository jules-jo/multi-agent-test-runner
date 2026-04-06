"""Tests for the progress tracking data model and shared state."""

from __future__ import annotations

import threading
import time

import pytest

from test_runner.models.progress import (
    ProgressSnapshot,
    ProgressTracker,
    TestResult,
    TestStatus,
)
from test_runner.orchestrator.hub import RunPhase, RunState


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------


class TestTestResult:
    """Tests for the immutable TestResult data class."""

    def test_passed_property(self):
        r = TestResult(test_id="t1", name="test_a", status=TestStatus.PASSED)
        assert r.passed is True
        assert r.failed is False

    def test_failed_property(self):
        r = TestResult(test_id="t1", name="test_a", status=TestStatus.FAILED)
        assert r.passed is False
        assert r.failed is True

    def test_error_counts_as_failed(self):
        r = TestResult(test_id="t1", name="test_a", status=TestStatus.ERROR)
        assert r.failed is True

    def test_skipped_is_neither_passed_nor_failed(self):
        r = TestResult(test_id="t1", name="test_a", status=TestStatus.SKIPPED)
        assert r.passed is False
        assert r.failed is False

    def test_default_fields(self):
        r = TestResult(test_id="t1", name="test_a", status=TestStatus.PASSED)
        assert r.duration_seconds == 0.0
        assert r.output == ""
        assert r.error_message == ""
        assert r.framework == ""
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# ProgressSnapshot
# ---------------------------------------------------------------------------


class TestProgressSnapshot:
    """Tests for the immutable ProgressSnapshot."""

    def _make_snapshot(self, **kwargs) -> ProgressSnapshot:
        defaults = dict(
            total=10,
            completed=5,
            passed=3,
            failed=1,
            skipped=1,
            errored=0,
            start_time=1000.0,
            elapsed_seconds=2.5,
            is_running=True,
            is_complete=False,
            results=(),
        )
        defaults.update(kwargs)
        return ProgressSnapshot(**defaults)

    def test_pending(self):
        snap = self._make_snapshot(total=10, completed=4)
        assert snap.pending == 6

    def test_pass_rate(self):
        snap = self._make_snapshot(completed=4, passed=3)
        assert snap.pass_rate == pytest.approx(0.75)

    def test_pass_rate_zero_completed(self):
        snap = self._make_snapshot(completed=0, passed=0)
        assert snap.pass_rate == 0.0

    def test_success_when_all_passed(self):
        snap = self._make_snapshot(
            completed=5, passed=5, failed=0, errored=0, is_complete=True
        )
        assert snap.success is True

    def test_not_success_with_failures(self):
        snap = self._make_snapshot(
            completed=5, passed=4, failed=1, errored=0, is_complete=True
        )
        assert snap.success is False

    def test_not_success_when_still_running(self):
        snap = self._make_snapshot(
            completed=5, passed=5, failed=0, errored=0, is_complete=False
        )
        assert snap.success is False

    def test_summary_serializable(self):
        snap = self._make_snapshot()
        s = snap.summary()
        assert isinstance(s, dict)
        assert s["total"] == 10
        assert s["completed"] == 5
        assert s["passed"] == 3
        assert s["failed"] == 1
        assert s["skipped"] == 1
        assert s["errored"] == 0
        assert s["pending"] == 5
        assert "pass_rate" in s
        assert "elapsed_seconds" in s
        assert "is_running" in s
        assert "is_complete" in s


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------


class TestProgressTracker:
    """Tests for the mutable, thread-safe ProgressTracker."""

    def test_initial_state(self):
        tracker = ProgressTracker(total=10)
        snap = tracker.snapshot()
        assert snap.total == 10
        assert snap.completed == 0
        assert snap.passed == 0
        assert snap.failed == 0
        assert snap.skipped == 0
        assert snap.errored == 0
        assert snap.is_running is False
        assert snap.is_complete is False

    def test_set_total(self):
        tracker = ProgressTracker()
        tracker.set_total(42)
        assert tracker.total == 42
        assert tracker.snapshot().total == 42

    def test_start_marks_running(self):
        tracker = ProgressTracker(total=5)
        tracker.start()
        snap = tracker.snapshot()
        assert snap.is_running is True
        assert snap.is_complete is False
        assert snap.start_time is not None

    def test_finish_marks_complete(self):
        tracker = ProgressTracker(total=5)
        tracker.start()
        tracker.finish()
        snap = tracker.snapshot()
        assert snap.is_running is False
        assert snap.is_complete is True

    def test_elapsed_time_increases(self):
        tracker = ProgressTracker(total=1)
        tracker.start()
        time.sleep(0.05)
        snap = tracker.snapshot()
        assert snap.elapsed_seconds > 0.0

    def test_record_passed(self):
        tracker = ProgressTracker(total=3)
        tracker.start()
        tracker.record_result(
            TestResult(test_id="t1", name="test_a", status=TestStatus.PASSED)
        )
        snap = tracker.snapshot()
        assert snap.completed == 1
        assert snap.passed == 1
        assert snap.failed == 0

    def test_record_failed(self):
        tracker = ProgressTracker(total=3)
        tracker.start()
        tracker.record_result(
            TestResult(test_id="t1", name="test_a", status=TestStatus.FAILED)
        )
        snap = tracker.snapshot()
        assert snap.completed == 1
        assert snap.passed == 0
        assert snap.failed == 1

    def test_record_skipped(self):
        tracker = ProgressTracker(total=3)
        tracker.start()
        tracker.record_result(
            TestResult(test_id="t1", name="test_a", status=TestStatus.SKIPPED)
        )
        snap = tracker.snapshot()
        assert snap.completed == 1
        assert snap.skipped == 1

    def test_record_error(self):
        tracker = ProgressTracker(total=3)
        tracker.start()
        tracker.record_result(
            TestResult(test_id="t1", name="test_a", status=TestStatus.ERROR)
        )
        snap = tracker.snapshot()
        assert snap.completed == 1
        assert snap.errored == 1

    def test_record_multiple_results(self):
        tracker = ProgressTracker(total=4)
        tracker.start()
        tracker.record_results([
            TestResult(test_id="t1", name="a", status=TestStatus.PASSED),
            TestResult(test_id="t2", name="b", status=TestStatus.FAILED),
            TestResult(test_id="t3", name="c", status=TestStatus.SKIPPED),
            TestResult(test_id="t4", name="d", status=TestStatus.ERROR),
        ])
        snap = tracker.snapshot()
        assert snap.completed == 4
        assert snap.passed == 1
        assert snap.failed == 1
        assert snap.skipped == 1
        assert snap.errored == 1

    def test_results_stored_in_snapshot(self):
        tracker = ProgressTracker(total=2)
        tracker.start()
        r1 = TestResult(test_id="t1", name="a", status=TestStatus.PASSED)
        r2 = TestResult(test_id="t2", name="b", status=TestStatus.FAILED)
        tracker.record_result(r1)
        tracker.record_result(r2)
        snap = tracker.snapshot()
        assert len(snap.results) == 2
        assert snap.results[0] == r1
        assert snap.results[1] == r2

    def test_callback_invoked_on_record(self):
        snapshots: list[ProgressSnapshot] = []
        tracker = ProgressTracker(total=2)
        tracker.on_progress(lambda s: snapshots.append(s))
        tracker.start()  # triggers callback
        tracker.record_result(
            TestResult(test_id="t1", name="a", status=TestStatus.PASSED)
        )
        # start() + record_result() = 2 callbacks
        assert len(snapshots) == 2
        assert snapshots[-1].passed == 1

    def test_callback_on_set_total(self):
        snapshots: list[ProgressSnapshot] = []
        tracker = ProgressTracker()
        tracker.on_progress(lambda s: snapshots.append(s))
        tracker.set_total(10)
        assert len(snapshots) == 1
        assert snapshots[0].total == 10

    def test_callback_on_finish(self):
        snapshots: list[ProgressSnapshot] = []
        tracker = ProgressTracker(total=1)
        tracker.on_progress(lambda s: snapshots.append(s))
        tracker.start()
        tracker.finish()
        assert any(s.is_complete for s in snapshots)

    def test_bad_callback_does_not_break_tracker(self):
        def bad_cb(_snap):
            raise RuntimeError("boom")

        tracker = ProgressTracker(total=1)
        tracker.on_progress(bad_cb)
        # Should not raise
        tracker.start()
        tracker.record_result(
            TestResult(test_id="t1", name="a", status=TestStatus.PASSED)
        )
        snap = tracker.snapshot()
        assert snap.passed == 1

    def test_reset_clears_everything(self):
        tracker = ProgressTracker(total=5)
        tracker.start()
        tracker.record_result(
            TestResult(test_id="t1", name="a", status=TestStatus.PASSED)
        )
        tracker.finish()
        tracker.reset()
        snap = tracker.snapshot()
        assert snap.total == 0
        assert snap.completed == 0
        assert snap.is_running is False
        assert snap.is_complete is False
        assert len(snap.results) == 0

    def test_thread_safety(self):
        """Concurrent record_result calls should not corrupt counters."""
        tracker = ProgressTracker(total=100)
        tracker.start()

        def record_batch(start: int) -> None:
            for i in range(start, start + 25):
                tracker.record_result(
                    TestResult(
                        test_id=f"t{i}",
                        name=f"test_{i}",
                        status=TestStatus.PASSED,
                    )
                )

        threads = [threading.Thread(target=record_batch, args=(i * 25,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = tracker.snapshot()
        assert snap.completed == 100
        assert snap.passed == 100
        assert len(snap.results) == 100


# ---------------------------------------------------------------------------
# RunState integration
# ---------------------------------------------------------------------------


class TestRunStateProgress:
    """Tests that ProgressTracker is properly integrated into RunState."""

    def test_run_state_has_progress_tracker(self):
        state = RunState(request="run all tests")
        assert isinstance(state.progress, ProgressTracker)

    def test_run_state_progress_is_usable(self):
        state = RunState(request="run all tests")
        state.progress.set_total(5)
        state.progress.start()
        state.progress.record_result(
            TestResult(test_id="t1", name="a", status=TestStatus.PASSED)
        )
        snap = state.progress.snapshot()
        assert snap.total == 5
        assert snap.passed == 1

    def test_each_run_state_gets_own_tracker(self):
        s1 = RunState(request="run tests a")
        s2 = RunState(request="run tests b")
        s1.progress.set_total(10)
        assert s2.progress.total == 0
