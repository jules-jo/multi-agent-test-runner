# Reporting

## Primary Files

- `src/test_runner/reporting/base.py`
- `src/test_runner/reporting/cli_streaming.py`
- `src/test_runner/reporting/events.py`
- `src/test_runner/reporting/summary_renderer.py`
- `src/test_runner/agents/reporter/`

## Role

Reporting is both streaming and final. It is not just post-run printing.

## Responsibilities

- emit progress events during execution
- support periodic rollups
- write channel-specific output
- render final summaries with failures, timing, and optional AI analysis

## Final Summary

`FinalSummaryRenderer` is the main summary formatting surface. It can fold together:

- run counts
- timing
- failure details
- AI analysis
- troubleshooter fix proposals

## Design Strength

Reporting is separated from execution and orchestration, which makes output formatting easier to evolve without changing control flow.

## Current Integration Note

The executor-to-reporter path now relies on the executor's task-level callback wiring rather than ad hoc post-execution stdout forwarding. This matters because the reporter can emit a fallback task-level event even when framework output parsing does not produce individual test events. That fixes misleading summaries such as `0 total` for successful task executions.
