# Execution

## Primary Files

- `src/test_runner/execution/command_translator.py`
- `src/test_runner/execution/executor.py`
- `src/test_runner/execution/targets.py`
- `src/test_runner/execution/factory.py`
- `src/test_runner/execution/script_executor.py`
- `src/test_runner/execution/remote_ci.py`

## Flow

1. Structured test intent is translated into one or more `TestCommand` values.
2. Commands are executed through an `ExecutionTarget`.
3. Results are recorded in `TaskAttemptRecord`.
4. Retry policy determines whether to reattempt on errors or timeouts.

## Command Translation

The command translator uses framework-specific strategies for:

- pytest
- unittest
- jest
- mocha
- go test
- cargo test
- dotnet test
- scripts

## Execution Targets

`ExecutionTarget` is an abstraction for where commands run.

Current concrete targets:

- `LocalTarget`
- `SSHTarget`

Design intent:

- support local, Docker, and remote CI targets behind the same interface

Current catalog-driven behavior:

- local catalog entries execute through `LocalTarget`
- catalog entries with `catalog_system_transport=ssh` are resolved per command into `SSHTarget`
- explicit CLI target overrides still take precedence over automatic catalog-based target selection

## Policy

Execution retry logic is governed by `ExecutionPolicy`, which distinguishes between:

- passed
- failed test assertions
- infrastructure errors
- timeouts

## Current Integration Note

The orchestrator now passes CLI-selected execution parameters into the execution path:

- target selection
- timeout override
- working-directory override
- default command environment for local runs

Remote nuance:

- local-only PATH injection is not applied to cataloged `ssh` commands
- local default working-directory fallback is not injected into cataloged `ssh` commands
- remote commands run through the local `ssh` client using saved host metadata from the catalog
- `SSHTarget` now performs a basic deterministic preflight before remote execution:
  - verify that the local `ssh` client is available
  - verify that the saved destination is reachable in batch mode with a short timeout
- preflight failures are surfaced as infrastructure errors with explicit SSH context instead of opaque subprocess launch failures
