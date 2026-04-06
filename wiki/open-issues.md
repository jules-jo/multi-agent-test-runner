# Open Issues

## High Priority

### Catalog Population Workflow Needs Expansion

- A deterministic JSON-backed catalog resolver now auto-loads `registry/catalog.json` in this repo, so closed-world execution is now the default behavior here.
- The shipped default catalog is intentionally empty, which is safe but not yet user-friendly.
- The interactive CLI now has a first-pass registration dialogue for unknown requests, and the CLI now supports basic entry management commands.
- The next product gap is richer teaching and management of saved definitions, not basic enforcement.

### Dataiku Mesh Integration Is Only Partially Proven

- The parser path is clearly designed for an OpenAI-compatible backend such as Dataiku LLM Mesh.
- The one-shot and piped CLI-to-orchestrator path now exists, but Mesh-backed operation is still not validated end to end against a real Dataiku backend.
- Discovery and troubleshooting should be checked to ensure they use the same backend/provider story consistently.

## Medium Priority

### Repo Hygiene

- Generated artifacts like `__pycache__` and `.egg-info` are checked into the workspace snapshot.
- This makes exploration noisier and should be cleaned up when convenient.

### Missing Top-Level Product Docs

- The wiki now captures architecture, but a user-facing `README.md` is still absent.

### Interactive Conversation Depth Is Still Thin

- Interactive mode now routes requests through the orchestrator, but it still behaves as a sequence of independent requests.
- The CLI now preserves lightweight session state for last-alias follow-ups and ambiguous alias clarification, but there is still no deeper threaded conversation model for approvals, edits, or longer-running multi-turn plans.
- Broader conversational intents are still not modeled separately from task requests.

### Catalog Teaching Workflow Is Still Thin

- The current interactive CLI can now collect and persist a new saved test definition after an unknown request, including a new system definition when needed.
- The flow is still form-like and local to the terminal REPL rather than a richer chatbot/session capability.
- There is now basic `list/show/edit/delete` management for both catalog entries and saved systems from the CLI front door, but the UX is still prompt-form based rather than conversational.

### Remote Execution Needs Hardening

- The authoritative catalog schema now includes named `ssh` systems and references from test entries to those systems.
- The execution layer now includes an `SSHTarget`, so cataloged remote commands can execute through the local `ssh` client.
- Basic deterministic SSH preflight now exists, including local `ssh` client availability checks and short remote reachability checks.
- What is still missing is broader operational hardening: better remote environment/bootstrap handling, artifact collection, richer remediation guidance, and more real-world validation against remote hosts.

## Tracking Note

When these issues are addressed, update:

- `wiki/overview.md`
- the relevant subsystem pages
- `wiki/log.md`
