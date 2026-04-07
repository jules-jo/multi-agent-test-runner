# CLI

## Primary Files

- `src/test_runner/__main__.py`
- `src/test_runner/cli.py`

## Responsibilities

- Parse command-line arguments
- Support one-shot, piped, and interactive input
- Load configuration
- Validate user requests
- Present initial request metadata

## Current Behavior

The one-shot, piped, and terminal interactive CLI paths now:

- validates the request
- chooses parse mode based on LLM availability
- constructs reporting channels
- creates an execution target
- instantiates `OrchestratorHub`
- awaits `hub.run(request)`
- returns a process exit code based on orchestrator state

## Interactive Mode

Interactive mode now loops over validated terminal input and dispatches each request through the same orchestrator-backed handler as one-shot mode.

The orchestrator instance is created per request rather than reused across the entire REPL session. That avoids stale executor history, reporter state, and budget counters leaking between requests.

Interactive mode is still task-oriented, not general chat-oriented, but it now has a lightweight front door:

- greetings like `hello` are answered locally
- help-style prompts like `help` or `what can you do` print usage guidance locally
- actual test requests continue through the orchestrator path
- the session now remembers the last resolved saved alias so follow-ups like `rerun that` can be rewritten into a concrete saved test request
- when a saved-test request is ambiguous, the session can hold pending alias choices and accept a short reply such as `2` or `beta`
- when a saved test has no default system, the session can hold pending saved-system choices and accept a short reply such as `2` or `lab-a`
- a request can now include a saved system alias such as `on lab-a` or `in lab-a`; the CLI passes that as a controlled per-run override into catalog resolution
- when a request fails because no saved catalog alias matches, the CLI can now offer an interactive registration flow that collects entry details, persists them to the catalog, and optionally reruns the saved alias immediately
- explicit catalog-management commands such as `list saved tests`, `show test lt`, `edit test lt`, `delete test lt`, `list systems`, `show system lab-a`, `edit system lab-a`, and `delete system lab-a` are now handled locally by the CLI and bypass the orchestrator entirely

## Operational Detail

For local execution, the CLI prepends the active interpreter directory and the repo virtualenv script directory to `PATH`. On Windows that means `.venv/Scripts`; on Unix-like systems that means `.venv/bin`. This allows translated commands such as `pytest ...` to resolve correctly when the CLI itself is launched outside the repo virtualenv.

CLI startup now also disables OpenAI Agents SDK trace export automatically when `OPENAI_API_KEY` is not configured. This suppresses the non-fatal `skipping trace export` warning for Dataiku-backed runs that use the Agents SDK only for local parsing/tracing and do not intend to export traces to OpenAI.

In redirected or piped sessions, prompt text may interleave slightly with rich console output because `input()` writes prompts to stdout while the rich console writes to stderr.
