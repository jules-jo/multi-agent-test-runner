# Invariants

These are the current assumptions the system appears to rely on.

## Architectural Invariants

- `OrchestratorHub` is the central coordinator; sub-agents should not communicate directly.
- Shared agent state belongs to the orchestrator, not to individual agents.
- Discovery and troubleshooting are budget-limited.
- Troubleshooter behavior is diagnose-first and read-only by default.
- Command translation is separate from command execution.
- Reporting is a separate concern from execution and orchestration.

## Operational Invariants

- Natural-language requests must be validated before resolution.
- Low-confidence intent resolution should surface a clarification path.
- Retry policy should distinguish between deterministic test failures and infrastructure failures.
- Durable wiki facts should be inspectable and sourceable.
