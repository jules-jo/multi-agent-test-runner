# Open Issues

## High Priority

### Catalog Enforcement Is Not Default Yet

- A deterministic JSON-backed catalog resolver now exists and blocks unknown or ambiguous requests when `TEST_CATALOG_PATH` is configured.
- The catalog schema now includes named execution systems, but the repo still does not ship a default repo-local catalog file or enable catalog mode automatically.
- The repo does not yet ship a default catalog, so legacy freeform translation still remains the default behavior in unconfigured environments.
- If the target product is truly closed-world by default, the next step is to decide on a repo-local catalog location and make registry enforcement the normal path.

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

### Catalog Teaching Workflow Is Still Missing

- The current catalog is machine-readable and executable, but there is no implemented conversational flow for proposing a new test, confirming alias/path/system details, and persisting the approved definition.
- The wiki captures the rule that new runnable tests require explicit confirmation, but the product flow to do that has not been built yet.

### Remote Execution Needs Hardening

- The authoritative catalog schema now includes named `ssh` systems and references from test entries to those systems.
- The execution layer now includes an `SSHTarget`, so cataloged remote commands can execute through the local `ssh` client.
- What is still missing is operational hardening: preflight SSH checks, better remote environment/bootstrap handling, artifact collection, and clearer user-facing remediation when SSH is unavailable or misconfigured.

## Tracking Note

When these issues are addressed, update:

- `wiki/overview.md`
- the relevant subsystem pages
- `wiki/log.md`
