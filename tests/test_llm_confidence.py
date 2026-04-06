"""Tests for LLM self-assessment confidence signal collector."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_runner.agents.discovery.llm_confidence import (
    DEFAULT_LLM_CONFIDENCE_WEIGHT,
    DiscoveryContext,
    _parse_llm_response,
    assess_confidence,
)
from test_runner.models.confidence import ConfidenceSignal


# ---------------------------------------------------------------------------
# DiscoveryContext tests
# ---------------------------------------------------------------------------


class TestDiscoveryContext:
    def test_empty_context_prompt(self):
        ctx = DiscoveryContext()
        prompt = ctx.to_prompt()
        assert "No evidence" in prompt

    def test_full_context_prompt(self):
        ctx = DiscoveryContext(
            project_path="/my/project",
            files_found=["test_foo.py", "test_bar.py"],
            frameworks_detected=["pytest"],
            config_snippets={"pytest.ini": "[pytest]\naddopts = -v"},
            raw_signals=[
                {"name": "pytest_ini_exists", "score": 1.0, "weight": 0.9},
                {"name": "no_match", "score": 0.0, "weight": 0.5},
            ],
            extra="This is a Python project",
        )
        prompt = ctx.to_prompt()
        assert "/my/project" in prompt
        assert "pytest" in prompt
        assert "test_foo.py" in prompt
        assert "test_bar.py" in prompt
        assert "[pytest]" in prompt
        assert "pytest_ini_exists" in prompt
        assert "Python project" in prompt

    def test_prompt_truncates_long_snippets(self):
        ctx = DiscoveryContext(
            config_snippets={"big.cfg": "x" * 1000},
        )
        prompt = ctx.to_prompt()
        assert "truncated" in prompt

    def test_prompt_limits_file_listing(self):
        ctx = DiscoveryContext(
            files_found=[f"test_{i}.py" for i in range(50)],
        )
        prompt = ctx.to_prompt()
        assert "50 total" in prompt
        # Should show up to 20
        assert "test_19.py" in prompt

    def test_prompt_filters_zero_score_signals(self):
        ctx = DiscoveryContext(
            raw_signals=[
                {"name": "found_it", "score": 0.8, "weight": 0.9},
                {"name": "nope", "score": 0.0, "weight": 0.5},
            ],
        )
        prompt = ctx.to_prompt()
        assert "found_it" in prompt
        assert "nope" not in prompt

    def test_only_project_path(self):
        ctx = DiscoveryContext(project_path="/some/path")
        prompt = ctx.to_prompt()
        assert "/some/path" in prompt
        assert "No evidence" not in prompt


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    def test_valid_json(self):
        raw = json.dumps({"confidence": 0.85, "reasoning": "Clear pytest setup"})
        score, reasoning = _parse_llm_response(raw)
        assert score == 0.85
        assert "pytest" in reasoning

    def test_json_with_code_fence(self):
        raw = '```json\n{"confidence": 0.7, "reasoning": "some tests"}\n```'
        score, reasoning = _parse_llm_response(raw)
        assert score == 0.7
        assert "some tests" in reasoning

    def test_clamps_score_above_one(self):
        raw = json.dumps({"confidence": 1.5, "reasoning": "too high"})
        score, _ = _parse_llm_response(raw)
        assert score == 1.0

    def test_clamps_score_below_zero(self):
        raw = json.dumps({"confidence": -0.3, "reasoning": "negative"})
        score, _ = _parse_llm_response(raw)
        assert score == 0.0

    def test_missing_confidence_defaults(self):
        raw = json.dumps({"reasoning": "no score field"})
        score, _ = _parse_llm_response(raw)
        assert score == 0.5

    def test_fallback_regex_extraction(self):
        raw = 'Some text confidence: 0.65 and reasoning: "found tests"'
        score, reasoning = _parse_llm_response(raw)
        assert score == 0.65
        assert "found tests" in reasoning

    def test_bare_float_extraction(self):
        raw = "I think the confidence is about 0.72 because reasons"
        score, reasoning = _parse_llm_response(raw)
        assert score == 0.72
        assert "bare float" in reasoning

    def test_completely_unparseable(self):
        raw = "I have no idea what format you want"
        score, reasoning = _parse_llm_response(raw)
        assert score == 0.5
        assert "default" in reasoning.lower()

    def test_integer_one(self):
        raw = json.dumps({"confidence": 1, "reasoning": "perfect"})
        score, _ = _parse_llm_response(raw)
        assert score == 1.0

    def test_integer_zero(self):
        raw = json.dumps({"confidence": 0, "reasoning": "nothing"})
        score, _ = _parse_llm_response(raw)
        assert score == 0.0

    def test_whitespace_handling(self):
        raw = '  \n  {"confidence": 0.55, "reasoning": "meh"}  \n  '
        score, _ = _parse_llm_response(raw)
        assert score == 0.55


# ---------------------------------------------------------------------------
# assess_confidence integration tests (with mocked OpenAI client)
# ---------------------------------------------------------------------------


def _make_mock_client(response_content: str) -> AsyncMock:
    """Create a mock AsyncOpenAI client that returns the given content."""
    client = AsyncMock()
    message = MagicMock()
    message.content = response_content
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


class TestAssessConfidence:
    @pytest.mark.asyncio
    async def test_returns_confidence_signal(self):
        client = _make_mock_client(
            json.dumps({"confidence": 0.85, "reasoning": "Clear pytest setup"})
        )
        ctx = DiscoveryContext(
            project_path="/proj",
            frameworks_detected=["pytest"],
            files_found=["test_main.py"],
        )
        signal = await assess_confidence(client, ctx, model="test-model")

        assert isinstance(signal, ConfidenceSignal)
        assert signal.name == "llm_self_assessment"
        assert signal.score == 0.85
        assert signal.weight == DEFAULT_LLM_CONFIDENCE_WEIGHT
        assert "pytest" in signal.evidence["reasoning"]
        assert signal.evidence["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_custom_weight(self):
        client = _make_mock_client(
            json.dumps({"confidence": 0.5, "reasoning": "ok"})
        )
        ctx = DiscoveryContext()
        signal = await assess_confidence(client, ctx, weight=0.9)
        assert signal.weight == 0.9

    @pytest.mark.asyncio
    async def test_sends_correct_messages(self):
        client = _make_mock_client(
            json.dumps({"confidence": 0.7, "reasoning": "fine"})
        )
        ctx = DiscoveryContext(
            project_path="/proj",
            frameworks_detected=["jest"],
        )
        await assess_confidence(client, ctx, model="my-model")

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "my-model"
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "confidence evaluator" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert "jest" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_low_temperature(self):
        client = _make_mock_client(
            json.dumps({"confidence": 0.5, "reasoning": "ok"})
        )
        await assess_confidence(client, DiscoveryContext())
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_handles_api_error_gracefully(self):
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("API timeout")
        )
        ctx = DiscoveryContext(project_path="/proj")
        signal = await assess_confidence(client, ctx)

        assert isinstance(signal, ConfidenceSignal)
        assert signal.name == "llm_self_assessment"
        assert signal.score == 0.5  # neutral default
        assert signal.weight == DEFAULT_LLM_CONFIDENCE_WEIGHT * 0.5  # reduced
        assert "error" in signal.evidence
        assert "API timeout" in signal.evidence["error"]

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        client = _make_mock_client("")
        ctx = DiscoveryContext()
        signal = await assess_confidence(client, ctx)

        assert isinstance(signal, ConfidenceSignal)
        # Empty string -> unparseable -> defaults to 0.5
        assert signal.score == 0.5

    @pytest.mark.asyncio
    async def test_handles_none_content(self):
        """When the LLM returns None content."""
        client = AsyncMock()
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        completion = MagicMock()
        completion.choices = [choice]
        client.chat.completions.create = AsyncMock(return_value=completion)

        ctx = DiscoveryContext()
        signal = await assess_confidence(client, ctx)
        assert isinstance(signal, ConfidenceSignal)
        assert signal.score == 0.5

    @pytest.mark.asyncio
    async def test_evidence_contains_context_summary(self):
        client = _make_mock_client(
            json.dumps({"confidence": 0.9, "reasoning": "excellent"})
        )
        ctx = DiscoveryContext(
            files_found=["a.py", "b.py"],
            frameworks_detected=["pytest", "unittest"],
        )
        signal = await assess_confidence(client, ctx)

        summary = signal.evidence["context_summary"]
        assert summary["files_count"] == 2
        assert "pytest" in summary["frameworks"]
        assert "unittest" in summary["frameworks"]

    @pytest.mark.asyncio
    async def test_signal_passes_validation(self):
        """The returned ConfidenceSignal must have valid weight/score ranges."""
        client = _make_mock_client(
            json.dumps({"confidence": 0.75, "reasoning": "good"})
        )
        signal = await assess_confidence(client, DiscoveryContext())
        assert 0.0 <= signal.weight <= 1.0
        assert 0.0 <= signal.score <= 1.0

    @pytest.mark.asyncio
    async def test_error_signal_passes_validation(self):
        """Even on error, the signal must have valid ranges."""
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            side_effect=Exception("boom")
        )
        signal = await assess_confidence(client, DiscoveryContext())
        assert 0.0 <= signal.weight <= 1.0
        assert 0.0 <= signal.score <= 1.0
