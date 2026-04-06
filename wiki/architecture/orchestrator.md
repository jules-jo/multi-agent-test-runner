# Orchestrator

## Primary Files

- `src/test_runner/orchestrator/hub.py`
- `src/test_runner/orchestrator/state_store.py`

## Role

The orchestrator is the control-plane center of the application. It coordinates specialized sub-agents and owns the authoritative run state.

## Main Concepts

- `RunPhase`: run lifecycle state machine
- `RunState`: mutable state for one run
- `EscalationRecord`: structured escalation history
- `AgentStateStore`: per-agent lifecycle and delegation tracking

## Responsibilities

- Resolve intent from natural-language input
- Run discovery before execution
- Execute commands with policy-aware handling
- Trigger troubleshooting on failures or escalations
- Coordinate reporter rollups and final summaries
- Enforce delegation budgets

## Design Strength

The hub-and-spoke model is clear: sub-agents do not communicate directly, which keeps workflow logic centralized and auditable.

## Current Risk

The orchestrator exists, but the CLI does not yet drive it in production flow.
