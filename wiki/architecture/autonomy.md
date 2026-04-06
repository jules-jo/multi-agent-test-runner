# Autonomy

## Primary Files

- `src/test_runner/autonomy/approval.py`
- `src/test_runner/autonomy/budget.py`
- `src/test_runner/autonomy/decision_engine.py`
- `src/test_runner/autonomy/engine.py`
- `src/test_runner/autonomy/fix_executor.py`
- `src/test_runner/autonomy/policy.py`

## Role

The autonomy layer defines how far the system can act without user intervention.

## Concerns

- step and budget limits
- approval requirements
- fix execution boundaries
- policy interpretation
- escalation behavior

## Cross-Cutting Nature

Autonomy is not a single pipeline stage. It affects discovery, execution retries, troubleshooting, and any future auto-fix behavior.
