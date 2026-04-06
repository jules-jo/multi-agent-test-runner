# Open Issues

## High Priority

### Catalog Population Workflow Needs Expansion

- A deterministic JSON-backed catalog resolver now auto-loads `registry/catalog.json` in this repo, so closed-world execution is now the default behavior here.
- The shipped default catalog is intentionally empty, which is safe but not yet user-friendly.
- The interactive CLI now has a first-pass registration dialogue for unknown requests, but the broader management story is still thin.
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
- There is no higher-level conversational session layer for preserving turn-local clarifications, approvals, or threaded follow-up state inside the CLI itself.
- Only a lightweight front door exists so far; broader conversational intents are still not modeled separately from task requests.

### Catalog Teaching Workflow Is Still Thin

- The current interactive CLI can now collect and persist a new saved test definition after an unknown request, including a new system definition when needed.
- The flow is still form-like and local to the terminal REPL rather than a richer chatbot/session capability.
- There is still no edit/delete/list management flow for catalog entries from the chat surface.

### Remote Execution Needs Hardening

- The authoritative catalog schema now includes named `ssh` systems and references from test entries to those systems.
- The execution layer now includes an `SSHTarget`, so cataloged remote commands can execute through the local `ssh` client.
- What is still missing is operational hardening: preflight SSH checks, better remote environment/bootstrap handling, artifact collection, and clearer user-facing remediation when SSH is unavailable or misconfigured.

## Tracking Note

When these issues are addressed, update:

- `wiki/overview.md`
- the relevant subsystem pages
- `wiki/log.md`
