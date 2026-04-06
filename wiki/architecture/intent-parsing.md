# Intent Parsing

## Primary Files

- `src/test_runner/agents/parser.py`
- `src/test_runner/agents/intent_service.py`

## Role

Convert natural-language user requests into structured intent and executable commands.

## Layers

### Parser Layer

`NaturalLanguageParser` defines:

- `TestIntent`
- `TestFramework`
- `ParsedTestRequest`

It supports:

- LLM-backed parsing through the OpenAI Agents SDK
- offline heuristic parsing when the LLM is unavailable or disabled

### Service Layer

`IntentParserService` wraps parsing and command translation into one cohesive integration surface.

It returns `IntentResolution`, which contains:

- parsed request
- translated commands
- warnings
- parse mode used
- clarification requirement

## Current Known Issue

There is a mismatch between config fallback behavior and test expectations. Generic `LLM_*` env values can make the service believe LLM config is available even when `DATAIKU_*` vars are cleared. This currently breaks two tests.

## Backend Note

The parser is the clearest implemented OpenAI-compatible backend integration point in the repo today. It constructs an `AsyncOpenAI` client using configured `base_url` and `api_key`, which means an OpenAI-compatible backend such as Dataiku LLM Mesh can sit behind it.
