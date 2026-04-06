"""Tests for the orchestrator shared state store."""

from __future__ import annotations

import time

import pytest

from test_runner.agents.base import AgentRole
from test_runner.orchestrator.state_store import (
    AgentRecord,
    AgentStateStore,
    AgentStatus,
    DelegationCycle,
)


class TestAgentRecord:
    """Tests for the AgentRecord dataclass."""

    def test_default_values(self):
        rec = AgentRecord(role=AgentRole.DISCOVERY)
        assert rec.status == AgentStatus.IDLE
        assert rec.total_steps == 0
        assert rec.total_cycles == 0
        assert rec.current_confidence == 0.0
        assert rec.results == []
        assert rec.errors == []
        assert rec.metadata == {}

    def test_to_dict_roundtrip(self):
        rec = AgentRecord(
            role=AgentRole.EXECUTOR,
            status=AgentStatus.COMPLETED,
            total_steps=10,
            total_cycles=2,
            current_confidence=0.85,
        )
        d = rec.to_dict()
        assert d["role"] == "executor"
        assert d["status"] == "completed"
        assert d["total_steps"] == 10
        assert d["total_cycles"] == 2
        assert d["current_confidence"] == 0.85


class TestDelegationCycle:
    """Tests for the DelegationCycle dataclass."""

    def test_duration_when_finished(self):
        cycle = DelegationCycle(
            cycle_id=1,
            agent_role=AgentRole.DISCOVERY,
            started_at=100.0,
            finished_at=105.5,
        )
        assert cycle.duration_seconds == pytest.approx(5.5)

    def test_duration_zero_when_not_finished(self):
        cycle = DelegationCycle(
            cycle_id=1,
            agent_role=AgentRole.DISCOVERY,
            started_at=100.0,
        )
        assert cycle.duration_seconds == 0.0

    def test_to_dict_contains_all_fields(self):
        cycle = DelegationCycle(
            cycle_id=3,
            agent_role=AgentRole.REPORTER,
            started_at=200.0,
            finished_at=210.0,
            status=AgentStatus.COMPLETED,
            steps_taken=5,
            confidence_after=0.9,
        )
        d = cycle.to_dict()
        assert d["cycle_id"] == 3
        assert d["agent_role"] == "reporter"
        assert d["duration_seconds"] == 10.0
        assert d["status"] == "completed"


class TestAgentStateStore:
    """Tests for the AgentStateStore."""

    def test_initial_state_is_empty(self):
        store = AgentStateStore()
        assert store.total_cycles == 0
        assert store.all_agents() == {}
        assert store.active_cycles == []

    def test_get_agent_returns_none_for_unknown(self):
        store = AgentStateStore()
        assert store.get_agent(AgentRole.DISCOVERY) is None

    def test_get_agent_status_idle_for_unknown(self):
        store = AgentStateStore()
        assert store.get_agent_status(AgentRole.DISCOVERY) == AgentStatus.IDLE

    def test_start_delegation_creates_running_cycle(self):
        store = AgentStateStore()
        cycle = store.start_delegation(AgentRole.DISCOVERY, {"query": "find tests"})

        assert cycle.cycle_id == 1
        assert cycle.agent_role == AgentRole.DISCOVERY
        assert cycle.status == AgentStatus.RUNNING
        assert cycle.input_summary == {"query": "find tests"}
        assert store.total_cycles == 1
        assert store.get_agent_status(AgentRole.DISCOVERY) == AgentStatus.RUNNING

    def test_finish_delegation_updates_record(self):
        store = AgentStateStore()
        cycle = store.start_delegation(AgentRole.DISCOVERY)

        handoff = {
            "agent": "discovery",
            "role": "discovery",
            "confidence_tier": "high",
            "escalated": False,
            "state": {
                "steps_taken": 7,
                "current_confidence": 0.92,
                "findings": [{"file": "test_foo.py", "framework": "pytest"}],
                "errors": [],
            },
        }
        finished = store.finish_delegation(cycle.cycle_id, handoff)

        assert finished.status == AgentStatus.COMPLETED
        assert finished.steps_taken == 7
        assert finished.confidence_after == 0.92
        assert finished.finished_at > 0

        record = store.get_agent(AgentRole.DISCOVERY)
        assert record is not None
        assert record.status == AgentStatus.COMPLETED
        assert record.total_steps == 7
        assert record.total_cycles == 1
        assert record.current_confidence == 0.92
        assert len(record.results) == 1
        assert record.metadata["agent"] == "discovery"

    def test_multiple_cycles_accumulate(self):
        store = AgentStateStore()

        # Cycle 1
        c1 = store.start_delegation(AgentRole.EXECUTOR)
        store.finish_delegation(c1.cycle_id, {
            "state": {"steps_taken": 3, "current_confidence": 0.7, "findings": [{"result": "pass"}], "errors": []},
        })

        # Cycle 2
        c2 = store.start_delegation(AgentRole.EXECUTOR)
        store.finish_delegation(c2.cycle_id, {
            "state": {"steps_taken": 5, "current_confidence": 0.6, "findings": [{"result": "fail"}], "errors": ["timeout"]},
        })

        record = store.get_agent(AgentRole.EXECUTOR)
        assert record.total_steps == 8  # 3 + 5
        assert record.total_cycles == 2
        assert record.current_confidence == 0.6  # latest
        assert len(record.results) == 2
        assert len(record.errors) == 1

    def test_fail_delegation(self):
        store = AgentStateStore()
        cycle = store.start_delegation(AgentRole.TROUBLESHOOTER)
        store.fail_delegation(cycle.cycle_id, "LLM timeout")

        record = store.get_agent(AgentRole.TROUBLESHOOTER)
        assert record.status == AgentStatus.FAILED
        assert "LLM timeout" in record.errors

    def test_cycles_for_filters_by_role(self):
        store = AgentStateStore()
        c1 = store.start_delegation(AgentRole.DISCOVERY)
        c2 = store.start_delegation(AgentRole.EXECUTOR)
        c3 = store.start_delegation(AgentRole.DISCOVERY)
        store.finish_delegation(c1.cycle_id)
        store.finish_delegation(c2.cycle_id)
        store.finish_delegation(c3.cycle_id)

        discovery_cycles = store.cycles_for(AgentRole.DISCOVERY)
        assert len(discovery_cycles) == 2
        assert all(c.agent_role == AgentRole.DISCOVERY for c in discovery_cycles)

    def test_latest_cycle_for(self):
        store = AgentStateStore()
        assert store.latest_cycle_for(AgentRole.DISCOVERY) is None

        c1 = store.start_delegation(AgentRole.DISCOVERY)
        store.finish_delegation(c1.cycle_id)
        c2 = store.start_delegation(AgentRole.DISCOVERY)
        store.finish_delegation(c2.cycle_id)

        latest = store.latest_cycle_for(AgentRole.DISCOVERY)
        assert latest.cycle_id == c2.cycle_id

    def test_active_cycles(self):
        store = AgentStateStore()
        c1 = store.start_delegation(AgentRole.DISCOVERY)
        c2 = store.start_delegation(AgentRole.EXECUTOR)

        assert len(store.active_cycles) == 2

        store.finish_delegation(c1.cycle_id)
        assert len(store.active_cycles) == 1
        assert store.active_cycles[0].cycle_id == c2.cycle_id

    def test_snapshot_structure(self):
        store = AgentStateStore()
        c = store.start_delegation(AgentRole.DISCOVERY)
        store.finish_delegation(c.cycle_id, {
            "state": {"steps_taken": 2, "current_confidence": 0.8, "findings": [], "errors": []},
        })

        snap = store.snapshot()
        assert "agents" in snap
        assert "discovery" in snap["agents"]
        assert snap["total_cycles"] == 1
        assert snap["active_cycles"] == 0
        assert len(snap["cycles"]) == 1

    def test_agent_summary_for_unknown_role(self):
        store = AgentStateStore()
        summary = store.agent_summary(AgentRole.REPORTER)
        assert summary["role"] == "reporter"
        assert summary["status"] == "idle"

    def test_reset_clears_everything(self):
        store = AgentStateStore()
        c = store.start_delegation(AgentRole.DISCOVERY)
        store.finish_delegation(c.cycle_id)

        store.reset()
        assert store.total_cycles == 0
        assert store.all_agents() == {}
        assert store.get_agent(AgentRole.DISCOVERY) is None

    def test_find_cycle_raises_on_invalid_id(self):
        store = AgentStateStore()
        with pytest.raises(KeyError, match="not found"):
            store._find_cycle(999)

    def test_cycle_ids_are_monotonic(self):
        store = AgentStateStore()
        c1 = store.start_delegation(AgentRole.DISCOVERY)
        c2 = store.start_delegation(AgentRole.EXECUTOR)
        c3 = store.start_delegation(AgentRole.REPORTER)
        assert c1.cycle_id < c2.cycle_id < c3.cycle_id


class TestRunStateIntegration:
    """Test that RunState includes the agent_store."""

    def test_runstate_has_agent_store(self):
        from test_runner.orchestrator.hub import RunState

        state = RunState(request="run all tests")
        assert isinstance(state.agent_store, AgentStateStore)
        assert state.agent_store.total_cycles == 0

    def test_runstate_agent_store_is_independent(self):
        from test_runner.orchestrator.hub import RunState

        s1 = RunState()
        s2 = RunState()
        s1.agent_store.start_delegation(AgentRole.DISCOVERY)
        assert s1.agent_store.total_cycles == 1
        assert s2.agent_store.total_cycles == 0
