# Repo Overview

## Summary

This repository is a Python project named `multi-agent-test-runner`. Its goal is to accept a natural-language test request, resolve it into executable test commands, run those tests through an orchestrated multi-agent workflow, and produce progress and final summaries.

## Current State

- The codebase is substantial and modular.
- The orchestrator-centered architecture is present.
- The one-shot, piped, and terminal interactive CLI paths are now integrated with the orchestrator.
- The test suite is large and mostly green.
- Persistent wiki memory is now enabled through the markdown scaffold in `wiki/`.

## Intended Request Lifecycle

1. User submits a natural-language request through the CLI.
2. Config is loaded and validated.
3. Intent parsing resolves the request into structured intent and candidate commands.
4. The orchestrator delegates discovery, execution, reporting, and troubleshooting as needed.
5. Results are rendered into summaries and surfaced through reporting channels.
6. Durable knowledge is written back into the wiki when useful.

## Current Verification Snapshot

- Date: 2026-04-06
- Command: `./.venv/bin/pytest -q`
- Result: `2129 passed, 55 warnings`

## Recent Implementation Notes

- The CLI now creates and runs an `OrchestratorHub` for non-dry requests.
- Missing LLM configuration no longer blocks local runs; the CLI uses offline parsing in that case.
- Local CLI runs now inject the active virtualenv's `bin` directory into `PATH` so commands like `pytest` resolve correctly.
- Reporter summaries now receive task-level execution results through the executor callback path, which fixes the previous `0 total` summary problem for runs whose raw output did not yield per-test parse events.
- Terminal interactive mode now reuses the same request-handling path as one-shot mode instead of the placeholder REPL.
- Interactive mode now handles obvious greetings and help-style prompts locally instead of routing them into the test parser.
- The parser's Agents SDK provider is now forced onto Chat Completions mode (`use_responses=False`) because the Dataiku Mesh OpenAI-compatible endpoint used here supports chat completions but not the Responses API.
- Local CLI PATH injection is now cross-platform: it prepends the active interpreter directory plus repo `.venv/Scripts` on Windows and `.venv/bin` on Unix-like systems.
- Reporter fallback events no longer count retried infrastructure attempts as separate failed tests; intermediate retried attempts are emitted as log events instead.
- Placeholder values copied from `.env.example` are now treated as unset, so a fresh stub `.env` no longer masquerades as a valid LLM configuration.

## Current Known Gaps

- Repo-level documentation was absent before this wiki scaffold.
