# Architecture Overview

## Product Shape

The system is designed as an orchestrator-centered test runner with specialized sub-agents and pluggable execution/reporting layers.

The evolving target shape is a chat-based test operations agent with:

- a conversational front end
- a deterministic catalog of approved tests and systems
- controlled execution across local and remote targets
- streamed status updates and final summaries back into the conversation

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
- registry enforcement
- host and target selection
- refusal of unknown or unapproved test definitions

In this repo, the main examples are the orchestrator, executor, config, autonomy, and reporting modules.

### SDK-Agent Layer

This is the LLM-driven reasoning layer. It should own:

- natural-language interpretation
- alias and intent matching against saved tests
- clarification generation when references are ambiguous or incomplete
- conversational follow-up handling
- heuristic troubleshooting
- synthesis of findings into structured outputs

In this repo, the main examples are the parser, discovery agent, and troubleshooter agent.

## Boundary Rule

Use local Python orchestration when behavior must be predictable, policy-constrained, and reproducible.

Use SDK agents when the task is ambiguous, language-heavy, exploratory, or diagnosis-oriented.

For the intended chatbot product, the LLM layer should help map user language onto saved test definitions, but it should not be the authority that invents arbitrary commands or target systems.

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

The expected safety boundary is closed-world execution:

- if a requested test alias is not saved in the catalog, do not run it
- ask for clarification or registration instead
- only confirmed catalog entries become runnable definitions

The current implementation now includes a deterministic JSON-backed catalog resolver that can be enabled through configuration. In catalog mode, intent resolution matches saved aliases and keywords locally and only emits runnable commands for saved `python_script` and `executable` entries. The catalog schema now also includes named execution systems, with `local` systems executable today and non-local systems failing closed until the corresponding target is implemented.

## Current Integration Status

The internal architecture is more complete than the final product shape. Core orchestration exists, and a first machine-readable catalog layer now exists, but the full chatbot product still needs default-on registry adoption, richer clarification flows, remote-target execution, and conversation-driven catalog teaching.
