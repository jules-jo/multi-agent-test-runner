# Wiki Log

## 2026-04-06

- Created initial wiki scaffold for repo architecture and persistent agent memory.
- Seeded subsystem pages for CLI, orchestrator, agents, intent parsing, execution, reporting, autonomy, and shared models.
- Recorded current integration gap: CLI still acknowledges requests without dispatching them to the orchestrator.
- Recorded current verification status: `./.venv/bin/pytest -q` passed 2113 tests and failed 2 intent-service tests.
- Added architectural guidance distinguishing deterministic local orchestration from SDK-agent responsibilities.
- Recorded that Dataiku LLM Mesh support is partially built via OpenAI-compatible configuration but not yet validated end to end.
- Wired the one-shot and piped CLI path into `OrchestratorHub`.
- Switched CLI behavior to offline parsing when no OpenAI-compatible backend is configured.
- Added JSON and Markdown final-summary reporting channels.
- Injected virtualenv-aware `PATH` handling for local CLI runs so translated console-script commands resolve correctly.
- Wired executor attempts to reporter callbacks so final summaries reflect task results even without per-test output parsing.
- Verified a manual CLI smoke run with `./.venv/bin/python -m test_runner "run pytest tests/test_cli.py"`.
- Routed terminal interactive mode through the same orchestrator-backed request handler as one-shot mode.
- Verified an interactive CLI smoke run with `printf 'run pytest tests/test_cli.py\nquit\n' | ./.venv/bin/python -m test_runner -i`.
- Re-ran the full suite successfully: `2117 passed, 54 warnings`.
- Re-ran the full suite successfully after interactive-mode wiring: `2119 passed, 54 warnings`.
- Recorded that interactive mode still treats casual chat like `hello` as a test request instead of a greeting/help intent.
- Added a lightweight CLI front door so greetings and help-style prompts are handled locally before orchestration.
- Verified an interactive CLI smoke run with `printf 'hello\nhelp\nrun pytest tests/test_cli.py\nquit\n' | ./.venv/bin/python -m test_runner -i`.
- Re-ran the full suite successfully after front-door handling: `2122 passed, 54 warnings`.
- Checked for a real Dataiku Mesh validation path and found no live backend configuration in the current workspace yet.
