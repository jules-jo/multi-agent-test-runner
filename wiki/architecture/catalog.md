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
- `auth_method`
- `password_env_var`
- `python_command`
- `working_directory`
- `env`
- `credential_ref`
- `enabled`

Current transports:

- `local`
- `ssh`

Important rule:

- `credential_ref` is only a pointer to external credentials such as SSH config or a secret manager key. Secrets do not belong in the catalog.
- for password-based SSH, store the password in an environment variable and save only the env-var name in `password_env_var`
- `python_command` lets saved `python_script` entries use a system-specific interpreter such as `python3.8` while keeping the script path in `target`

Current implementation status:

- `local` systems are executable now
- `ssh` systems are executable through `SSHTarget`, which shells out through the local `ssh` client using saved host metadata
- password-based `ssh` systems are also supported; they use a Python SSH backend and read the password from the configured environment variable instead of prompting interactively

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
- a request may use a saved per-run system override such as `on lab-a`, but the override must still resolve to a saved system alias in the catalog
- `python_script` entries use `system.python_command` when present, otherwise they default to `python`

## Resolution Rules

Matching is deterministic:

1. exact alias phrase matches win
2. otherwise keyword phrase matches are considered
3. multiple matches require clarification
4. no match means the request is not runnable

The catalog is the execution authority. LLM-based parsing may help interpret the request, but it should not invent runnable commands outside the saved catalog.

## Current Gaps

- the current teaching flow is a first-pass interactive CLI dialogue, not a richer multi-turn chatbot/session feature
- remote host lifecycle is still thin: no preflight connectivity checks, artifact transfer, or richer SSH/session management yet
- the default shipped catalog is intentionally empty, so the product still needs a user-facing flow for populating real runnable definitions

## Teaching Flow

The interactive CLI now includes a deterministic first-pass registration flow:

1. user issues a request that does not match any saved alias
2. the CLI offers to register a new catalog entry
3. the user confirms alias, execution type, target path, system alias, and optional metadata
4. if the referenced system does not exist yet, the CLI collects a minimal `local` or `ssh` system definition
5. the entry is saved into the catalog
6. the CLI can optionally rerun the new saved alias immediately

Current limitations:

- the prompts are form-like rather than natural-language extraction
- there is no wiki page generation tied to catalog teaching yet

## Management Commands

The CLI now includes deterministic local catalog-management commands:

- `list saved tests`
- `show test <alias>`
- `edit test <alias>`
- `delete test <alias>`
- `list systems`
- `show system <alias>`
- `edit system <alias>`
- `delete system <alias>`

These commands operate directly on the JSON catalog and do not go through the orchestrator or intent parser.

System-management safety rules:

- deleting a system is refused while any saved test still references it
- system edits currently update the saved definition in place rather than renaming aliases across dependent entries
