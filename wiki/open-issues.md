# Open Issues

## High Priority

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

## Tracking Note

When these issues are addressed, update:

- `wiki/overview.md`
- the relevant subsystem pages
- `wiki/log.md`
