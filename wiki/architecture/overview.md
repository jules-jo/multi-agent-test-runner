# Architecture Overview

## Product Shape

The system is designed as an orchestrator-centered test runner with specialized sub-agents and pluggable execution/reporting layers.

## Major Subsystems

- CLI entrypoint
- Config loading and validation
- Intent parsing
- Orchestrator hub
- Discovery agent
- Execution layer
- Reporter agent and reporting channels
- Troubleshooter agent
- Autonomy and approval policy
- Shared models, events, and framework adapters

## Two Architectural Layers

The system has two distinct layers that should not be conflated.

### Local Python Orchestration Layer

This is the deterministic control plane. It should own:

- workflow sequencing
- phase transitions
- state tracking
- retries and timeouts
- execution targets
- approval and budget enforcement
- reporting lifecycle
- exit status decisions

In this repo, the main examples are the orchestrator, executor, config, autonomy, and reporting modules.

### SDK-Agent Layer

This is the LLM-driven reasoning layer. It should own:

- natural-language interpretation
- exploratory repository discovery
- heuristic troubleshooting
- synthesis of findings into structured outputs

In this repo, the main examples are the parser, discovery agent, and troubleshooter agent.

## Boundary Rule

Use local Python orchestration when behavior must be predictable, policy-constrained, and reproducible.

Use SDK agents when the task is ambiguous, language-heavy, exploratory, or diagnosis-oriented.

The preferred interaction pattern is:

1. Local orchestration frames the task and constraints.
2. The SDK agent works inside a bounded role with limited tools.
3. The agent returns structured output.
4. Local orchestration validates that output and decides what happens next.

## Control Plane

`OrchestratorHub` is the central coordinator. Sub-agents do not communicate directly with each other. They hand information back to the orchestrator, which owns:

- lifecycle phases
- state accumulation
- escalation routing
- budget enforcement
- reporting coordination

## Data Plane

The execution layer translates structured requests into concrete commands and sends them to pluggable targets such as the local machine. Framework adapters normalize outputs into shared result models.

## Current Integration Status

The internal architecture is more complete than the user-facing entry flow. Most subsystems exist, but end-to-end wiring from CLI to orchestrator is incomplete.
