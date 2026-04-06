# Repo Facts

## Test Invocation Default

- Fact: `pytest` is not available directly on `PATH` in this workspace, but `./.venv/bin/pytest` works.
- Scope: runtime
- Confidence: high
- Source: observed during local verification
- Last verified: 2026-04-06

## Test Suite Status

- Fact: The suite is fully passing, with 2122 passed tests and 54 warnings during the latest verification run.
- Scope: runtime
- Confidence: high
- Source: `./.venv/bin/pytest -q`
- Last verified: 2026-04-06

## CLI Integration Status

- Fact: The one-shot, piped, and terminal interactive CLI paths now dispatch real runs through `OrchestratorHub`.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/cli.py`
- Last verified: 2026-04-06

## Interactive Session Isolation

- Fact: Interactive CLI mode creates a fresh orchestrator per request rather than reusing one across the whole REPL session.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/cli.py`
- Last verified: 2026-04-06

## Interactive Front Door

- Fact: Interactive CLI mode now handles common greetings and help-style prompts locally before validation or orchestration.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/cli.py` and interactive smoke test
- Last verified: 2026-04-06

## CLI Prompt Ordering Nuance

- Fact: In redirected or piped interactive sessions, prompt text can interleave with rich console output because `input()` writes to stdout while rich output is sent to stderr.
- Scope: runtime
- Confidence: medium
- Source: interactive CLI smoke test
- Last verified: 2026-04-06

## CLI Offline Fallback

- Fact: Missing LLM configuration no longer blocks local CLI runs; the CLI selects offline parsing when no OpenAI-compatible backend is configured.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/cli.py`
- Last verified: 2026-04-06

## Local Virtualenv PATH Injection

- Fact: Local CLI runs prepend the active virtualenv `bin` directory and repo `.venv/bin` to `PATH` so translated commands such as `pytest` resolve correctly.
- Scope: runtime
- Confidence: high
- Source: CLI smoke test and `src/test_runner/cli.py`
- Last verified: 2026-04-06

## Reporter Callback Wiring

- Fact: The executor now reports task attempts to the reporter through the task-level callback path, which gives the reporter a fallback event even when raw output parsing finds no per-test events.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/orchestrator/hub.py` and CLI smoke test
- Last verified: 2026-04-06

## Architecture Layering

- Fact: The intended design has two layers: a deterministic local Python orchestration layer and an SDK-agent layer for reasoning-heavy tasks.
- Scope: repo
- Confidence: high
- Source: architecture review and current module structure
- Last verified: 2026-04-06

## Local Orchestration Responsibilities

- Fact: Workflow sequencing, policy enforcement, retries, execution targets, state tracking, reporting lifecycle, and exit behavior should remain in local Python orchestration.
- Scope: repo
- Confidence: high
- Source: architecture review
- Last verified: 2026-04-06

## SDK-Agent Responsibilities

- Fact: Natural-language parsing, exploratory discovery, and troubleshooting are the main areas that should be SDK-agent driven.
- Scope: repo
- Confidence: high
- Source: architecture review
- Last verified: 2026-04-06

## Dataiku Mesh Status

- Fact: Dataiku LLM Mesh support is partially implemented through OpenAI-compatible client configuration, but it is not yet proven end-to-end through the full CLI-to-orchestrator flow.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/config.py`, `src/test_runner/agents/parser.py`, and current integration status
- Last verified: 2026-04-06

## Current Mesh Config Availability

- Fact: In the current workspace, no real Dataiku/OpenAI-compatible backend configuration is present yet; `.env` is absent and the `DATAIKU_*` and `LLM_*` environment variables are unset.
- Scope: runtime
- Confidence: high
- Source: environment inspection and repo file inspection
- Last verified: 2026-04-06
