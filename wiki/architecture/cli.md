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

## Operational Detail

For local execution, the CLI prepends the active interpreter's `bin` directory and the repo `.venv/bin` directory to `PATH`. This allows translated commands such as `pytest ...` to resolve correctly when the CLI itself is launched through the project virtualenv.

In redirected or piped sessions, prompt text may interleave slightly with rich console output because `input()` writes prompts to stdout while the rich console writes to stderr.
