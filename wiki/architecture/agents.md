# Agents

## Primary Files

- `src/test_runner/agents/base.py`
- `src/test_runner/agents/discovery/`
- `src/test_runner/agents/troubleshooter/`
- `src/test_runner/agents/reporter/`

## Base Model

All sub-agents derive from `BaseSubAgent`, which provides:

- role identity
- confidence thresholds
- step tracking
- escalation checks
- handoff summaries to the orchestrator

## Discovery Agent

Purpose:

- inspect the repository
- detect frameworks
- find test files and configs
- propose runnable test targets

Signals and controls:

- step counter
- confidence tracking
- escalation threshold evaluation

## Troubleshooter Agent

Purpose:

- diagnose failed test runs
- inspect logs, environment, files, and processes
- generate structured fix proposals

Safety posture:

- diagnose-only by default
- does not auto-apply fixes
- uses read-only tools

## Reporter Agent

Purpose:

- own reporting channels
- emit periodic rollups
- support final result presentation

## Agent Memory Direction

Agents should propose durable knowledge for the wiki, but the orchestrator should remain the authority that decides what becomes long-lived repo memory.

## Agent Scope Guidance

The current intended use of SDK-backed agents is narrow and role-specific.

- Parser or conversational agent: interpret natural-language test requests and map them onto saved aliases
- Discovery agent: explore unknown repos and infer test structure when useful during development, not as the authority for production execution
- Troubleshooter agent: diagnose failures and propose fixes

SDK agents should not be the primary owners of:

- lifecycle control
- retry policy
- process execution policy
- final exit semantics
- durable memory promotion decisions
- approval of new runnable tests without user confirmation
- invention of arbitrary commands, paths, or systems outside the saved catalog

Those concerns belong in local orchestration.

## Chatbot Direction

For the intended chatbot product, the most useful role for SDK-backed agents is:

- understand what the user means by a test alias or request
- ask clarifying questions when the alias is unknown or ambiguous
- explain progress, failures, and next steps conversationally
- help turn confirmed user input into proposed catalog entries

They should not be trusted to bypass the catalog and synthesize runnable commands from scratch.
