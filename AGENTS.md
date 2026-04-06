# Agent Memory And Wiki Rules

This repository uses a local markdown wiki as a persistent memory layer for both humans and agents.

## Goals

- Preserve repo architecture and conventions across sessions.
- Reduce repeated user instructions.
- Keep operational knowledge inspectable in plain text.
- Separate stable facts from temporary run context.

## Wiki Layout

- `wiki/index.md`: top-level navigation
- `wiki/log.md`: append-only change log for wiki updates
- `wiki/overview.md`: current repo summary
- `wiki/architecture/`: subsystem pages
- `wiki/memory/`: rules, schema, and durable facts
- `wiki/runs/`: run-specific notes and investigations
- `wiki/open-issues.md`: unresolved gaps and follow-ups
- `wiki/invariants.md`: assumptions that should remain true unless intentionally changed

## Memory Rules

- Prefer writing facts that are stable, reusable, and actionable.
- Record the source of each fact when possible.
- Record confidence as `high`, `medium`, or `low`.
- Add a verification date for environment-dependent facts.
- Do not store secrets, tokens, passwords, or sensitive raw logs.
- Do not store speculative diagnoses as facts without labeling them clearly.
- Do not overwrite user intent; user instructions in the current session take precedence.

## Memory Scopes

- `user`: stable user preferences and constraints
- `repo`: architecture, commands, conventions, and known quirks for this codebase
- `runtime`: environment facts that may change and should be revalidated
- `run`: temporary facts tied to a specific execution or investigation
- `agent`: reusable heuristics for a specific sub-agent

## Write Pattern

When adding durable knowledge, prefer this structure:

```md
## Fact Title
- Fact: ...
- Scope: repo
- Confidence: high
- Source: observed in code or test output
- Last verified: YYYY-MM-DD
- Notes: optional
```

## Update Pattern

- Update the most specific page possible.
- Add a short line to `wiki/log.md`.
- If the new fact changes an existing assumption, update `wiki/invariants.md` or `wiki/open-issues.md`.

## Current Priority

Until the CLI is fully wired to the orchestrator, architecture and integration gaps should be documented aggressively so context is not lost between sessions.
