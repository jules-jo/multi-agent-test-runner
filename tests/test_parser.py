"""Tests for the natural language parser."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from test_runner.agents.parser import (
    NaturalLanguageParser,
    ParsedTestRequest,
    ParserError,
    TestFramework,
    TestIntent,
    _PARSER_SYSTEM_PROMPT,
)
from test_runner.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(monkeypatch):
    """Provide a valid Config for tests."""
    monkeypatch.setenv("DATAIKU_LLM_MESH_URL", "https://mesh.example.com/v1")
    monkeypatch.setenv("DATAIKU_API_KEY", "test-key-123")
    monkeypatch.setenv("DATAIKU_MODEL_ID", "test-model")
    return Config.load()


# ---------------------------------------------------------------------------
# ParsedTestRequest model tests
# ---------------------------------------------------------------------------


class TestParsedTestRequest:
    """Tests for the Pydantic model itself."""

    def test_defaults(self):
        req = ParsedTestRequest(
            intent=TestIntent.RUN,
            framework=TestFramework.PYTEST,
        )
        assert req.scope == ""
        assert req.working_directory == ""
        assert req.extra_args == []
        assert req.confidence == 1.0
        assert req.raw_request == ""
        assert req.reasoning == ""

    def test_full_construction(self):
        req = ParsedTestRequest(
            intent=TestIntent.RUN_SPECIFIC,
            framework=TestFramework.JEST,
            scope="src/components/",
            working_directory="/app",
            extra_args=["--coverage", "--verbose"],
            confidence=0.85,
            raw_request="run jest tests in src/components/ with coverage",
            reasoning="User wants jest tests in a specific directory.",
        )
        assert req.intent == TestIntent.RUN_SPECIFIC
        assert req.framework == TestFramework.JEST
        assert req.scope == "src/components/"
        assert req.confidence == 0.85
        assert len(req.extra_args) == 2

    def test_confidence_clamped(self):
        """Confidence must be between 0 and 1."""
        with pytest.raises(Exception):  # Pydantic validation
            ParsedTestRequest(
                intent=TestIntent.RUN,
                framework=TestFramework.PYTEST,
                confidence=1.5,
            )
        with pytest.raises(Exception):
            ParsedTestRequest(
                intent=TestIntent.RUN,
                framework=TestFramework.PYTEST,
                confidence=-0.1,
            )

    def test_serialization_roundtrip(self):
        original = ParsedTestRequest(
            intent=TestIntent.RUN,
            framework=TestFramework.PYTEST,
            scope="tests/",
            confidence=0.9,
            raw_request="run pytest on tests/",
        )
        data = original.model_dump()
        restored = ParsedTestRequest.model_validate(data)
        assert restored == original

    def test_json_roundtrip(self):
        original = ParsedTestRequest(
            intent=TestIntent.LIST,
            framework=TestFramework.AUTO_DETECT,
            scope="",
            confidence=0.7,
        )
        json_str = original.model_dump_json()
        restored = ParsedTestRequest.model_validate_json(json_str)
        assert restored == original


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_intent_values(self):
        assert TestIntent.RUN.value == "run"
        assert TestIntent.LIST.value == "list"
        assert TestIntent.RERUN_FAILED.value == "rerun_failed"
        assert TestIntent.RUN_SPECIFIC.value == "run_specific"
        assert TestIntent.UNKNOWN.value == "unknown"

    def test_framework_values(self):
        assert TestFramework.PYTEST.value == "pytest"
        assert TestFramework.AUTO_DETECT.value == "auto_detect"
        assert TestFramework.SCRIPT.value == "script"
        assert TestFramework.GO_TEST.value == "go_test"


# ---------------------------------------------------------------------------
# Offline (heuristic) parser tests
# ---------------------------------------------------------------------------


class TestParseOffline:
    """Tests for the keyword-based offline parser fallback."""

    def test_basic_run_pytest(self):
        result = NaturalLanguageParser.parse_offline("run pytest")
        assert result.intent == TestIntent.RUN
        assert result.framework == TestFramework.PYTEST
        assert result.confidence < 1.0  # heuristic = lower confidence
        assert result.raw_request == "run pytest"

    def test_list_intent(self):
        result = NaturalLanguageParser.parse_offline("list all tests in src/")
        assert result.intent == TestIntent.LIST
        assert result.scope == "src/"

    def test_discover_intent(self):
        result = NaturalLanguageParser.parse_offline("discover tests")
        assert result.intent == TestIntent.LIST

    def test_rerun_failed(self):
        result = NaturalLanguageParser.parse_offline("rerun failed tests")
        assert result.intent == TestIntent.RERUN_FAILED

    def test_retry_failed(self):
        result = NaturalLanguageParser.parse_offline("retry the failed tests")
        assert result.intent == TestIntent.RERUN_FAILED

    def test_jest_detection(self):
        result = NaturalLanguageParser.parse_offline("run jest tests")
        assert result.framework == TestFramework.JEST

    def test_mocha_detection(self):
        result = NaturalLanguageParser.parse_offline("run mocha suite")
        assert result.framework == TestFramework.MOCHA

    def test_go_test_detection(self):
        result = NaturalLanguageParser.parse_offline("run go test ./...")
        assert result.framework == TestFramework.GO_TEST

    def test_cargo_test_detection(self):
        result = NaturalLanguageParser.parse_offline("run cargo test")
        assert result.framework == TestFramework.CARGO_TEST

    def test_dotnet_test_detection(self):
        result = NaturalLanguageParser.parse_offline("run dotnet test")
        assert result.framework == TestFramework.DOTNET_TEST

    def test_unittest_detection(self):
        result = NaturalLanguageParser.parse_offline("run unittest")
        assert result.framework == TestFramework.UNITTEST

    def test_script_detection_bash(self):
        result = NaturalLanguageParser.parse_offline("run bash test_script.sh")
        assert result.framework == TestFramework.SCRIPT

    def test_script_detection_executable(self):
        result = NaturalLanguageParser.parse_offline("run ./run_tests.sh")
        assert result.framework == TestFramework.SCRIPT

    def test_auto_detect_when_no_framework(self):
        result = NaturalLanguageParser.parse_offline("run all unit tests")
        assert result.framework == TestFramework.AUTO_DETECT

    def test_scope_file_path(self):
        result = NaturalLanguageParser.parse_offline("run pytest tests/test_foo.py")
        assert result.scope == "tests/test_foo.py"

    def test_scope_test_name(self):
        result = NaturalLanguageParser.parse_offline("run test_something")
        assert result.scope == "test_something"

    def test_scope_class_name(self):
        result = NaturalLanguageParser.parse_offline("run TestMyClass with pytest")
        assert result.scope == "TestMyClass"

    def test_no_scope(self):
        result = NaturalLanguageParser.parse_offline("run all tests")
        assert result.scope == ""

    def test_reasoning_mentions_heuristic(self):
        result = NaturalLanguageParser.parse_offline("run pytest")
        assert "heuristic" in result.reasoning.lower()

    def test_empty_request(self):
        result = NaturalLanguageParser.parse_offline("")
        assert result.intent == TestIntent.RUN
        assert result.framework == TestFramework.AUTO_DETECT

    def test_js_file_scope(self):
        result = NaturalLanguageParser.parse_offline("run jest src/app.test.js")
        assert result.framework == TestFramework.JEST
        assert result.scope == "src/app.test.js"

    def test_ts_file_scope(self):
        result = NaturalLanguageParser.parse_offline("run jest src/app.test.ts")
        assert result.framework == TestFramework.JEST
        assert result.scope == "src/app.test.ts"


# ---------------------------------------------------------------------------
# NaturalLanguageParser construction tests
# ---------------------------------------------------------------------------


class TestParserInit:
    """Tests for parser initialization."""

    def test_creates_client_with_config(self, config):
        parser = NaturalLanguageParser(config)
        assert parser._client.base_url is not None
        assert parser._client.api_key == "test-key-123"

    def test_creates_agent_with_model(self, config):
        parser = NaturalLanguageParser(config)
        assert parser._agent.name == "test-request-parser"
        assert parser._agent.model == "test-model"

    def test_agent_has_structured_output(self, config):
        parser = NaturalLanguageParser(config)
        assert parser._agent.output_type == ParsedTestRequest

    def test_system_prompt_is_set(self, config):
        parser = NaturalLanguageParser(config)
        assert parser._agent.instructions == _PARSER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# LLM-backed parse (mocked) tests
# ---------------------------------------------------------------------------


class TestParseWithLLM:
    """Tests for the LLM-backed parse method with mocked Runner."""

    @pytest.mark.asyncio
    async def test_parse_returns_structured_output(self, config):
        expected = ParsedTestRequest(
            intent=TestIntent.RUN,
            framework=TestFramework.PYTEST,
            scope="tests/",
            confidence=0.95,
            reasoning="User wants to run all pytest tests in tests/ directory.",
        )

        mock_result = MagicMock()
        mock_result.final_output = expected

        with patch("test_runner.agents.parser.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            parser = NaturalLanguageParser(config)
            result = await parser.parse("run all pytest tests in tests/")

        assert result.intent == TestIntent.RUN
        assert result.framework == TestFramework.PYTEST
        assert result.scope == "tests/"
        assert result.confidence == 0.95
        assert result.raw_request == "run all pytest tests in tests/"

    @pytest.mark.asyncio
    async def test_parse_stamps_raw_request(self, config):
        """parse() always stamps the raw_request field."""
        expected = ParsedTestRequest(
            intent=TestIntent.RUN,
            framework=TestFramework.AUTO_DETECT,
        )

        mock_result = MagicMock()
        mock_result.final_output = expected

        with patch("test_runner.agents.parser.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            parser = NaturalLanguageParser(config)
            result = await parser.parse("some request")

        assert result.raw_request == "some request"

    @pytest.mark.asyncio
    async def test_parse_calls_runner_with_agent(self, config):
        mock_result = MagicMock()
        mock_result.final_output = ParsedTestRequest(
            intent=TestIntent.RUN,
            framework=TestFramework.PYTEST,
        )

        with patch("test_runner.agents.parser.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            parser = NaturalLanguageParser(config)
            await parser.parse("test request")

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args.kwargs.get("input") == "test request" or call_args.args[1] == "test request"

    @pytest.mark.asyncio
    async def test_parse_raises_parser_error_on_failure(self, config):
        with patch("test_runner.agents.parser.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = RuntimeError("LLM endpoint unavailable")
            parser = NaturalLanguageParser(config)

            with pytest.raises(ParserError, match="LLM endpoint unavailable"):
                await parser.parse("run tests")

    @pytest.mark.asyncio
    async def test_parse_passes_run_config_with_provider(self, config):
        mock_result = MagicMock()
        mock_result.final_output = ParsedTestRequest(
            intent=TestIntent.RUN,
            framework=TestFramework.PYTEST,
        )

        with patch("test_runner.agents.parser.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            parser = NaturalLanguageParser(config)
            await parser.parse("run tests")

            call_kwargs = mock_run.call_args
            # Should have a run_config argument
            assert call_kwargs.kwargs.get("run_config") is not None


# ---------------------------------------------------------------------------
# ParserError tests
# ---------------------------------------------------------------------------


class TestParserError:
    def test_parser_error_is_exception(self):
        err = ParserError("something went wrong")
        assert isinstance(err, Exception)
        assert str(err) == "something went wrong"

    def test_parser_error_chaining(self):
        cause = ValueError("bad value")
        err = ParserError("parse failed")
        err.__cause__ = cause
        assert err.__cause__ is cause
