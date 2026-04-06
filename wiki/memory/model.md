# Memory Model

## Purpose

The wiki acts as durable memory for both humans and agents. Its job is to reduce repeated explanations and preserve working context across sessions.

## Memory Scopes

### User Memory

Stable user preferences and constraints, such as:

- preferred tools
- approval requirements
- communication style

### Repo Memory

Facts about this codebase, such as:

- architecture
- test commands
- conventions
- recurring failure patterns

### Runtime Memory

Environment facts that may change, such as:

- active virtualenv requirements
- CI connectivity
- host-specific behavior

### Run Memory

Temporary notes tied to a specific investigation or execution.

### Agent Memory

Reusable heuristics for a sub-agent, subject to review before being promoted to repo memory.

## Storage Rule

Durable facts should be written in markdown with metadata:

- Fact
- Scope
- Confidence
- Source
- Last verified
- Notes

## Governance

- Agents may propose memory.
- The orchestrator or supervising workflow should decide what becomes durable.
- Low-confidence notes should stay in run memory until confirmed.
