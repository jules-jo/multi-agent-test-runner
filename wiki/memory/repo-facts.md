# Repo Facts

## Test Invocation Default

- Fact: `pytest` is not available directly on `PATH` in this workspace, but `./.venv/bin/pytest` works.
- Scope: runtime
- Confidence: high
- Source: observed during local verification
- Last verified: 2026-04-06

## Test Suite Status

- Fact: The suite is fully passing, with 2129 passed tests and 55 warnings during the latest verification run.
- Scope: runtime
- Confidence: high
- Source: `./.venv/bin/pytest -q`
- Last verified: 2026-04-06

## Main Regression Bundle Status

- Fact: The main regression bundle covering config, catalog, intent service, CLI, parser, execution targets, execution attempts, orchestrator delegation wiring, and troubleshooter wiring is passing at `333 passed, 16 warnings`.
- Scope: runtime
- Confidence: high
- Source: `./.venv/bin/pytest -q tests/test_config.py tests/test_catalog.py tests/test_intent_service.py tests/test_cli.py tests/test_parser.py tests/test_execution_targets.py tests/test_execution_attempts.py tests/test_orchestrator_delegation_wiring.py tests/test_troubleshooter_wiring.py`
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

## Interactive Session Memory

- Fact: Interactive CLI sessions now keep lightweight turn-local memory for the last matched saved alias, pending ambiguous alias choices, and the last selected saved system. This enables follow-ups like `rerun that` and short clarification replies such as `2`.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/cli.py` and `tests/test_cli.py`
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

- Fact: Local CLI runs prepend the active interpreter directory and repo virtualenv script directory to `PATH`, using `.venv/Scripts` on Windows and `.venv/bin` on Unix-like systems, so translated commands such as `pytest` resolve correctly.
- Scope: runtime
- Confidence: high
- Source: CLI smoke test and `src/test_runner/cli.py`
- Last verified: 2026-04-06

## Local Pytest Execution

- Fact: Local execution rewrites bare `pytest` invocations to `python -m pytest`, which avoids Windows launch failures when `pytest` is not separately discoverable on `PATH`.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/execution/targets.py` and Windows failure log analysis
- Last verified: 2026-04-06

## Reporter Callback Wiring

- Fact: The executor reports task attempts to the reporter through the task-level callback path, but retried infrastructure attempts are now logged as retry events instead of being counted as separate failed tests.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/orchestrator/hub.py` and CLI smoke test
- Last verified: 2026-04-06

## Failure Context Preservation

- Fact: Serialized execution summaries now preserve stdout/stderr from the last attempt, which keeps infrastructure launch errors visible to the troubleshooter.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/execution/executor.py` and Windows failure log analysis
- Last verified: 2026-04-06

## Dataiku Mesh Parser Compatibility

- Fact: The parser's OpenAI Agents SDK provider must set `use_responses=False` for the current Dataiku Mesh OpenAI-compatible endpoint, because the endpoint supports Chat Completions but not the Responses API.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/agents/parser.py` and live validation findings
- Last verified: 2026-04-06

## Placeholder Env Safety

- Fact: Placeholder values copied from `.env.example` are treated as unset, so a fresh stub `.env` does not enable LLM mode accidentally.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/config.py`, `.env.example`, and test suite behavior
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

## Product Direction

- Fact: The intended end state is a chatbot-style test operations agent rather than only a one-shot natural-language CLI runner.
- Scope: repo
- Confidence: high
- Source: product-direction discussion
- Last verified: 2026-04-06

## Closed-World Execution

- Fact: The desired execution model is closed-world: the system should only run tests that are saved in an approved catalog or wiki-backed registry, and it should ask for clarification instead of inventing unknown test runs.
- Scope: repo
- Confidence: high
- Source: product-direction discussion
- Last verified: 2026-04-06

## Catalog Registry Implementation

- Fact: The repo now has a deterministic JSON-backed catalog registry in `src/test_runner/catalog.py` that matches saved aliases/keywords locally and builds runnable commands only from saved entries.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/catalog.py`
- Last verified: 2026-04-06

## Catalog Mode Activation

- Fact: Closed-world catalog enforcement now auto-loads `registry/catalog.json` in this repo when `TEST_CATALOG_PATH` is absent. `TEST_CATALOG_PATH` still overrides the location, and setting it to an empty string explicitly disables repo auto-discovery.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/config.py`, `src/test_runner/agents/intent_service.py`, and `src/test_runner/orchestrator/hub.py`
- Last verified: 2026-04-06

## Default Catalog File

- Fact: The repo now ships an authoritative default catalog file at `registry/catalog.json`. It is intentionally empty except for a `local` system definition, so the default behavior is safe closed-world refusal until the catalog is populated.
- Scope: repo
- Confidence: high
- Source: `registry/catalog.json`
- Last verified: 2026-04-06

## Interactive Catalog Teaching

- Fact: Interactive CLI mode now offers a first-pass catalog teaching workflow when a request fails because no saved alias matches. The user can confirm a new entry, optionally create a new `local` or `ssh` system, and then rerun the new alias immediately.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/cli.py` and `tests/test_cli.py`
- Last verified: 2026-04-06

## CLI Catalog Management

- Fact: The CLI front door now supports deterministic local catalog-management commands for entries: `list saved tests`, `show test <alias>`, `edit test <alias>`, and `delete test <alias>`. These commands bypass the orchestrator and operate directly on the JSON catalog.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/cli.py`, `src/test_runner/catalog.py`, and `tests/test_cli.py`
- Last verified: 2026-04-06

## CLI System Management

- Fact: The CLI front door now also supports deterministic local system-management commands: `list systems`, `show system <alias>`, `edit system <alias>`, and `delete system <alias>`. These commands bypass the orchestrator and operate directly on the JSON catalog.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/cli.py`, `src/test_runner/catalog.py`, and `tests/test_cli.py`
- Last verified: 2026-04-06

## Catalog Execution Types

- Fact: The current machine-readable catalog supports saved `python_script` and `executable` definitions, and catalog mode ignores ad hoc extra args so execution stays bounded to the saved definition.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/catalog.py`
- Last verified: 2026-04-06

## Catalog System Schema

- Fact: The authoritative catalog schema now includes named execution systems with `local` and `ssh` transports; test entries reference those systems by alias instead of embedding host details ad hoc.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/catalog.py` and `registry/catalog.example.json`
- Last verified: 2026-04-06

## Remote Entries Execute Via SSH

- Fact: Catalog entries targeting saved `ssh` systems now translate into runnable commands and are executed through `SSHTarget`, which shells out through the local `ssh` client using catalog system metadata.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/catalog.py`, `src/test_runner/execution/targets.py`, and `src/test_runner/execution/executor.py`
- Last verified: 2026-04-06

## Remote Commands Avoid Local Defaults

- Fact: Local-only PATH injection and local default working-directory fallbacks are not applied to cataloged `ssh` commands; remote commands keep only the saved system and entry configuration.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/catalog.py` and `src/test_runner/orchestrator/hub.py`
- Last verified: 2026-04-06

## Saved System Overrides

- Fact: Catalog-backed requests can now use a saved per-run system override such as `on lab-a`. The override must resolve to a saved system alias in the catalog and changes the selected execution system without leaving closed-world mode.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/catalog.py`, `src/test_runner/agents/intent_service.py`, `src/test_runner/orchestrator/hub.py`, and `src/test_runner/cli.py`
- Last verified: 2026-04-06

## SSH Preflight

- Fact: `SSHTarget` now performs a basic deterministic preflight before remote execution by checking that the local `ssh` client is available and that the saved destination is reachable in batch mode with a short timeout.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/execution/targets.py` and `tests/test_execution_targets.py`
- Last verified: 2026-04-06

## Catalog Dry-Run Behavior

- Fact: In catalog mode, CLI dry runs now return a non-zero result when a request does not resolve to a runnable saved alias, instead of reporting the request as accepted.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/cli.py` and `tests/test_cli.py`
- Last verified: 2026-04-06

## Catalog Growth Rule

- Fact: New tests may be added through conversation, but only after the user confirms the alias, file path, execution type, and target system information.
- Scope: repo
- Confidence: high
- Source: product-direction discussion
- Last verified: 2026-04-06

## Immediate Execution Types

- Fact: The immediate execution targets to support are Python scripts and binary executables.
- Scope: repo
- Confidence: high
- Source: product-direction discussion
- Last verified: 2026-04-06

## Dataiku Mesh Status

- Fact: Dataiku LLM Mesh support is partially implemented through OpenAI-compatible client configuration, but it is not yet proven end-to-end through the full CLI-to-orchestrator flow.
- Scope: repo
- Confidence: high
- Source: `src/test_runner/config.py`, `src/test_runner/agents/parser.py`, and current integration status
- Last verified: 2026-04-06

## Current Mesh Config Availability

- Fact: In the current workspace, `.env` exists only as a placeholder stub. Its example values are intentionally treated as unset until replaced with real backend credentials.
- Scope: runtime
- Confidence: high
- Source: environment inspection and repo file inspection
- Last verified: 2026-04-06
