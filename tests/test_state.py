"""Tests for the shared orchestrator state store."""

from __future__ import annotations

import threading
import time

import pytest

from test_runner.state import (
    AgentRole,
    AgentState,
    AgentStatus,
    OrchestratorState,
)


# ── AgentState unit tests ──────────────────────────────────────────────


class TestAgentState:
    def test_defaults(self):
        state = AgentState(role=AgentRole.DISCOVERY)
        assert state.status == AgentStatus.IDLE
        assert state.result is None
        assert state.error is None
        assert state.metadata == {}
        assert state.last_updated > 0

    def test_touch_updates_timestamp(self):
        state = AgentState(role=AgentRole.EXECUTOR)
        old_ts = state.last_updated
        time.sleep(0.01)
        state.touch()
        assert state.last_updated > old_ts


# ── OrchestratorState – registration ───────────────────────────────────


class TestRegistration:
    def test_register_and_get(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.DISCOVERY)
        snap = store.get_agent(AgentRole.DISCOVERY)
        assert snap.role == AgentRole.DISCOVERY
        assert snap.status == AgentStatus.IDLE

    def test_duplicate_register_raises(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.EXECUTOR)
        with pytest.raises(ValueError, match="already registered"):
            store.register_agent(AgentRole.EXECUTOR)

    def test_get_unregistered_raises(self):
        store = OrchestratorState()
        with pytest.raises(KeyError, match="not registered"):
            store.get_agent(AgentRole.REPORTER)

    def test_get_returns_snapshot_not_reference(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.DISCOVERY)
        snap = store.get_agent(AgentRole.DISCOVERY)
        snap.status = AgentStatus.FAILED  # mutate snapshot
        # Canonical state should be unaffected
        assert store.get_agent(AgentRole.DISCOVERY).status == AgentStatus.IDLE


# ── OrchestratorState – update_agent ───────────────────────────────────


class TestUpdateAgent:
    def test_update_status(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.EXECUTOR)
        updated = store.update_agent(AgentRole.EXECUTOR, status=AgentStatus.RUNNING)
        assert updated.status == AgentStatus.RUNNING
        assert store.get_agent(AgentRole.EXECUTOR).status == AgentStatus.RUNNING

    def test_update_result(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.EXECUTOR)
        store.update_agent(AgentRole.EXECUTOR, result={"passed": 5, "failed": 1})
        snap = store.get_agent(AgentRole.EXECUTOR)
        assert snap.result == {"passed": 5, "failed": 1}

    def test_update_result_to_none(self):
        """Explicitly setting result=None should store None."""
        store = OrchestratorState()
        store.register_agent(AgentRole.EXECUTOR)
        store.update_agent(AgentRole.EXECUTOR, result="initial")
        store.update_agent(AgentRole.EXECUTOR, result=None)
        assert store.get_agent(AgentRole.EXECUTOR).result is None

    def test_update_error(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.EXECUTOR)
        store.update_agent(AgentRole.EXECUTOR, error="segfault")
        assert store.get_agent(AgentRole.EXECUTOR).error == "segfault"

    def test_update_metadata_merges(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.DISCOVERY)
        store.update_agent(AgentRole.DISCOVERY, metadata={"files": 10})
        store.update_agent(AgentRole.DISCOVERY, metadata={"frameworks": ["pytest"]})
        snap = store.get_agent(AgentRole.DISCOVERY)
        assert snap.metadata == {"files": 10, "frameworks": ["pytest"]}

    def test_update_unregistered_raises(self):
        store = OrchestratorState()
        with pytest.raises(KeyError, match="not registered"):
            store.update_agent(AgentRole.TROUBLESHOOTER, status=AgentStatus.RUNNING)

    def test_omitted_fields_unchanged(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.REPORTER)
        store.update_agent(AgentRole.REPORTER, result="report", status=AgentStatus.COMPLETED)
        # Update only status, result should stay
        store.update_agent(AgentRole.REPORTER, status=AgentStatus.IDLE)
        snap = store.get_agent(AgentRole.REPORTER)
        assert snap.result == "report"
        assert snap.status == AgentStatus.IDLE


# ── OrchestratorState – all_agents ─────────────────────────────────────


class TestAllAgents:
    def test_all_agents_returns_all(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.DISCOVERY)
        store.register_agent(AgentRole.EXECUTOR)
        agents = store.all_agents()
        assert set(agents.keys()) == {AgentRole.DISCOVERY, AgentRole.EXECUTOR}

    def test_all_agents_empty(self):
        store = OrchestratorState()
        assert store.all_agents() == {}


# ── OrchestratorState – shared scratchpad ──────────────────────────────


class TestSharedScratchpad:
    def test_set_and_get(self):
        store = OrchestratorState()
        store.set_shared("discovered_files", ["/a.py", "/b.py"])
        assert store.get_shared("discovered_files") == ["/a.py", "/b.py"]

    def test_get_default(self):
        store = OrchestratorState()
        assert store.get_shared("missing") is None
        assert store.get_shared("missing", 42) == 42

    def test_get_returns_copy(self):
        store = OrchestratorState()
        store.set_shared("data", [1, 2, 3])
        copy = store.get_shared("data")
        copy.append(4)
        assert store.get_shared("data") == [1, 2, 3]

    def test_delete(self):
        store = OrchestratorState()
        store.set_shared("key", "value")
        store.delete_shared("key")
        assert store.get_shared("key") is None

    def test_delete_missing_raises(self):
        store = OrchestratorState()
        with pytest.raises(KeyError):
            store.delete_shared("nope")


# ── OrchestratorState – convenience queries ────────────────────────────


class TestConvenienceQueries:
    def test_is_any_running_true(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.DISCOVERY)
        store.update_agent(AgentRole.DISCOVERY, status=AgentStatus.RUNNING)
        assert store.is_any_running() is True

    def test_is_any_running_false(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.DISCOVERY)
        assert store.is_any_running() is False

    def test_agents_with_status(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.DISCOVERY)
        store.register_agent(AgentRole.EXECUTOR)
        store.register_agent(AgentRole.REPORTER)
        store.update_agent(AgentRole.DISCOVERY, status=AgentStatus.COMPLETED)
        store.update_agent(AgentRole.EXECUTOR, status=AgentStatus.COMPLETED)
        completed = store.agents_with_status(AgentStatus.COMPLETED)
        assert set(completed) == {AgentRole.DISCOVERY, AgentRole.EXECUTOR}

    def test_summary_structure(self):
        store = OrchestratorState()
        store.register_agent(AgentRole.ORCHESTRATOR)
        store.set_shared("plan", "run all")
        s = store.summary()
        assert "run_id" in s
        assert "agents" in s
        assert "orchestrator" in s["agents"]
        assert s["shared_keys"] == ["plan"]


# ── Thread-safety ──────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_updates_no_crash(self):
        """Hammer the store from many threads — no exceptions should escape."""
        store = OrchestratorState()
        for role in AgentRole:
            store.register_agent(role)

        errors: list[str] = []
        barrier = threading.Barrier(10)

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                role = list(AgentRole)[idx % len(AgentRole)]
                for i in range(50):
                    store.update_agent(role, status=AgentStatus.RUNNING, metadata={"i": i})
                    store.get_agent(role)
                    store.is_any_running()
                    store.set_shared(f"t{idx}_{i}", i)
                    store.get_shared(f"t{idx}_{i}")
                    store.all_agents()
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent errors: {errors}"

    def test_concurrent_register_rejects_duplicates(self):
        """Only one thread should succeed at registering a given role."""
        store = OrchestratorState()
        successes = []
        failures = []
        barrier = threading.Barrier(5)

        def register_discovery() -> None:
            barrier.wait(timeout=5)
            try:
                store.register_agent(AgentRole.DISCOVERY)
                successes.append(True)
            except ValueError:
                failures.append(True)

        threads = [threading.Thread(target=register_discovery) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(successes) == 1
        assert len(failures) == 4
