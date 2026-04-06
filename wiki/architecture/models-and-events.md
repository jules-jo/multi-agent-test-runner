# Models And Events

## Primary Files

- `src/test_runner/models/progress.py`
- `src/test_runner/models/summary.py`
- `src/test_runner/models/confidence.py`
- `src/test_runner/events/models.py`
- `src/test_runner/events/callbacks.py`
- `src/test_runner/frameworks/`

## Role

These modules define the typed contracts shared across the system.

## Main Categories

- progress tracking
- confidence modeling
- summary rendering inputs
- event schemas for reporting
- framework-specific output normalization

## Why This Matters

The codebase is modular enough that shared models act as subsystem boundaries. If these contracts drift, the orchestrator and reporter layers will become brittle quickly.
