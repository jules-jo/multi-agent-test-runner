# Repo Overview

## Summary

This repository is a Python project named `multi-agent-test-runner`. Its goal is to accept a natural-language test request, resolve it into executable test commands, run those tests through an orchestrated multi-agent workflow, and produce progress and final summaries.

The current product direction is narrowing from a generic natural-language test runner toward a chat-based test operations agent backed by a controlled test catalog.

## Current State

- The codebase is substantial and modular.
- The orchestrator-centered architecture is present.
- The one-shot, piped, and terminal interactive CLI paths are now integrated with the orchestrator.
- A deterministic JSON-backed test catalog now exists, and this repo now auto-loads `registry/catalog.json` by default for closed-world execution.
- The catalog schema now models named execution systems as well as runnable test entries.
- Cataloged `ssh` systems can now execute through a dedicated SSH target instead of failing closed during translation.
- Interactive CLI mode can now teach a new catalog entry after an unknown request and optionally rerun the saved alias immediately.
- The CLI can now list, show, edit, and delete saved catalog entries without going through the orchestrator.
- The CLI can now list, show, edit, and delete saved execution systems without going through the orchestrator.
- Interactive CLI mode now keeps lightweight session memory for saved-test follow-ups and ambiguity resolution.
- SSH execution now does a basic preflight so missing local `ssh` binaries and unreachable destinations fail earlier with clearer error messages.
- The test suite is large and mostly green.
- Persistent wiki memory is now enabled through the markdown scaffold in `wiki/`.

## Intended Request Lifecycle

1. User submits a natural-language request through the CLI.
2. Config is loaded and validated.
3. Intent parsing resolves the request into structured intent and candidate commands.
4. The orchestrator delegates discovery, execution, reporting, and troubleshooting as needed.
5. Results are rendered into summaries and surfaced through reporting channels.
6. Durable knowledge is written back into the wiki when useful.

## Target Product Direction

- The final UX is intended to be chatbot-style rather than only a one-shot CLI utility.
- Users should be able to refer to approved tests by saved aliases or keywords such as `lt`.
- The system should only execute tests that exist in a saved catalog or wiki-backed registry.
- Unknown or ambiguous test references should trigger clarification instead of freeform command synthesis.
- New tests may be taught through conversation, but only after the user confirms the file path, host/system, and other execution details.
- The system should remember confirmed test definitions and grow its knowledge base over time.
- Python scripts and binary executables are the immediate execution targets to support.

## Current Verification Snapshot

- Date: 2026-04-06
- Command: `./.venv/bin/pytest -q`
- Result: `2129 passed, 55 warnings`

## Recent Implementation Notes

- The CLI now creates and runs an `OrchestratorHub` for non-dry requests.
- Intent resolution now supports a deterministic catalog/registry path for saved test aliases and keywords, with immediate execution support for Python scripts and binary executables.
- The catalog now supports named `local` and `ssh` systems; `ssh` entries resolve into a dedicated remote execution target rather than failing closed at translation time.
- Catalog translation now carries full system metadata into command metadata so execution can resolve per-command targets deterministically.
- The execution layer now resolves cataloged `ssh` commands into an `SSHTarget`, while preserving local-only PATH and working-directory defaults for local commands only.
- Unknown requests in interactive mode can now branch into a deterministic catalog registration dialogue that persists approved entry/system details into `registry/catalog.json`.
- Explicit catalog-management commands like `list saved tests`, `edit test <alias>`, `list systems`, and `edit system <alias>` are now handled locally at the CLI front door.
- Interactive CLI sessions now remember the last matched saved alias, can resolve ambiguous catalog matches with a short alias/number reply, and can pass a saved system override such as `on lab-a` into catalog resolution.
- Closed-world catalog mode now auto-enables in this repo when `registry/catalog.json` exists. `TEST_CATALOG_PATH` remains an override, and `TEST_CATALOG_PATH=""` explicitly disables repo auto-discovery.
- The shipped default catalog file is intentionally empty, so unknown requests now fail safely and point the user to the catalog file instead of falling back to freeform execution.
- CLI dry runs now surface catalog clarification failures as non-zero results instead of pretending the request is runnable.
- SSH execution now performs a basic deterministic preflight of the local `ssh` client and saved destination before the real remote command runs.
- Missing LLM configuration no longer blocks local runs; the CLI uses offline parsing in that case.
- Local CLI runs now inject the active virtualenv's `bin` directory into `PATH` so commands like `pytest` resolve correctly.
- Reporter summaries now receive task-level execution results through the executor callback path, which fixes the previous `0 total` summary problem for runs whose raw output did not yield per-test parse events.
- Terminal interactive mode now reuses the same request-handling path as one-shot mode instead of the placeholder REPL.
- Interactive mode now handles obvious greetings and help-style prompts locally instead of routing them into the test parser.
- The parser's Agents SDK provider is now forced onto Chat Completions mode (`use_responses=False`) because the Dataiku Mesh OpenAI-compatible endpoint used here supports chat completions but not the Responses API.
- Local CLI PATH injection is now cross-platform: it prepends the active interpreter directory plus repo `.venv/Scripts` on Windows and `.venv/bin` on Unix-like systems.
- Local execution now normalizes bare `pytest` commands to `python -m pytest`, which is more reliable on Windows when no standalone `pytest.exe` is present on `PATH`.
- Reporter fallback events no longer count retried infrastructure attempts as separate failed tests; intermediate retried attempts are emitted as log events instead.
- Placeholder values copied from `.env.example` are now treated as unset, so a fresh stub `.env` no longer masquerades as a valid LLM configuration.
- Serialized execution summaries now preserve the last attempt's stdout/stderr so launch failures retain useful context during troubleshooting.
- The main regression bundle covering config, catalog, parser, CLI, execution targets, attempts, delegation wiring, and troubleshooter wiring now passes at `329 passed, 16 warnings`.
- The current main regression bundle now passes at `333 passed, 16 warnings` after adding session follow-ups, ambiguity clarification replies, and saved system overrides for catalog runs.

## Current Known Gaps

- Repo-level documentation was absent before this wiki scaffold.
