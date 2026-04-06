# Catalog

## Purpose

The catalog is the authoritative machine-readable source of runnable test definitions.

When catalog mode is enabled, the system should:

- match user requests only against saved aliases and keywords
- refuse unknown or ambiguous requests
- build commands only from saved definitions
- fail closed when a saved definition references an unknown execution system

## Current File Shape

The current on-disk format is JSON. The loader uses:

- `TEST_CATALOG_PATH` when explicitly set
- otherwise `registry/catalog.json` in the repo root when it exists

Example location:

- `registry/catalog.json`
- `registry/catalog.example.json`

Top-level fields:

- `version`: catalog document version
- `systems`: named execution systems
- `entries`: saved runnable test definitions

## Systems

Each system describes where a saved test is allowed to run.

Current fields:

- `alias`
- `description`
- `transport`
- `hostname`
- `username`
- `port`
- `ssh_config_host`
- `working_directory`
- `env`
- `credential_ref`
- `enabled`

Current transports:

- `local`
- `ssh`

Important rule:

- `credential_ref` is only a pointer to external credentials such as SSH config or a secret manager key. Secrets do not belong in the catalog.

Current implementation status:

- `local` systems are executable now
- `ssh` systems are executable through `SSHTarget`, which shells out through the local `ssh` client using saved host metadata

## Entries

Each entry is one approved runnable test definition.

Current fields:

- `alias`
- `description`
- `execution_type`
- `target`
- `system`
- `args`
- `keywords`
- `working_directory`
- `env`
- `timeout`
- `enabled`

Current execution types:

- `python_script`
- `executable`

Behavior rules:

- `system` defaults to `local`
- ad hoc user-provided extra args are ignored in catalog mode
- entry-level `working_directory` overrides the system default
- entry-level `env` overrides the system `env`
- runtime-injected env overrides both for local execution
- local-only runtime env injection is intentionally not copied onto `ssh` commands

## Resolution Rules

Matching is deterministic:

1. exact alias phrase matches win
2. otherwise keyword phrase matches are considered
3. multiple matches require clarification
4. no match means the request is not runnable

The catalog is the execution authority. LLM-based parsing may help interpret the request, but it should not invent runnable commands outside the saved catalog.

## Current Gaps

- there is no conversational flow yet for teaching a new test and persisting it safely
- remote host lifecycle is still thin: no preflight connectivity checks, artifact transfer, or richer SSH/session management yet
- the default shipped catalog is intentionally empty, so the product still needs a user-facing flow for populating real runnable definitions
