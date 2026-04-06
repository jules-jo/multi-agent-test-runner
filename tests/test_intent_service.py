"""Tests for the IntentParserService — end-to-end NL → commands pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from test_runner.agents.intent_service import (
    IntentParserService,
    IntentResolution,
    IntentResolutionError,
    ParseMode,
)
from test_runner.agents.parser import (
    NaturalLanguageParser,
    ParsedTestRequest,
    ParserError,
    TestFramework,
    TestIntent,
)
from test_runner.config import Config
from test_runner.execution.command_translator import (
    CommandTranslator,
    TestCommand,
    TranslationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_with_llm(monkeypatch):
    """Config with all LLM fields populated."""
    monkeypatch.setenv("DATAIKU_LLM_MESH_URL", "https://mesh.example.com/v1")
    monkeypatch.setenv("DATAIKU_API_KEY", "test-key-123")
    monkeypatch.setenv("DATAIKU_MODEL_ID", "test-model")
    return Config.load()


@pytest.fixture
def config_no_llm(monkeypatch):
    """Config with no LLM fields — forces offline mode."""
    for var in (
        "DATAIKU_LLM_MESH_URL",
        "DATAIKU_API_KEY",
        "DATAIKU_MODEL_ID",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    return Config.load()


def _make_parsed(
    intent=TestIntent.RUN,
    framework=TestFramework.PYTEST,
    scope="tests/",
    confidence=0.9,
    raw_request="run pytest tests/",
    reasoning="Test parse",
) -> ParsedTestRequest:
    return ParsedTestRequest(
        intent=intent,
        framework=framework,
        scope=scope,
        confidence=confidence,
        raw_request=raw_request,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestServiceInit:
    def test_offline_mode_no_llm_parser(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        assert service.parse_mode == ParseMode.OFFLINE
        assert service._llm_parser is None

    def test_auto_mode_with_llm_config(self, config_with_llm):
        service = IntentParserService(config_with_llm, parse_mode=ParseMode.AUTO)
        assert service._llm_parser is not None

    def test_auto_mode_without_llm_config(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.AUTO)
        assert service._llm_parser is None

    def test_custom_clarification_threshold(self, config_no_llm):
        service = IntentParserService(
            config_no_llm, parse_mode=ParseMode.OFFLINE, clarification_threshold=0.7
        )
        assert service.clarification_threshold == 0.7

    def test_custom_command_translator(self, config_no_llm):
        custom_translator = CommandTranslator()
        service = IntentParserService(
            config_no_llm, parse_mode=ParseMode.OFFLINE,
            command_translator=custom_translator,
        )
        assert service.translator is custom_translator

    def test_default_translator_created(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        assert isinstance(service.translator, CommandTranslator)


# ---------------------------------------------------------------------------
# resolve_offline (synchronous) tests
# ---------------------------------------------------------------------------


class TestResolveOffline:
    def test_basic_pytest_run(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = service.resolve_offline("run pytest tests/")
        assert isinstance(result, IntentResolution)
        assert result.intent == TestIntent.RUN
        assert result.framework == TestFramework.PYTEST
        assert result.parse_mode_used == ParseMode.OFFLINE
        assert len(result.commands) >= 1
        assert "pytest" in result.commands[0].command

    def test_jest_list(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = service.resolve_offline("list jest tests")
        assert result.intent == TestIntent.LIST
        assert result.framework == TestFramework.JEST

    def test_rerun_failed(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = service.resolve_offline("rerun failed tests")
        assert result.intent == TestIntent.RERUN_FAILED

    def test_script_detection(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = service.resolve_offline("run bash ./test.sh")
        assert result.framework == TestFramework.SCRIPT

    def test_auto_detect_produces_warning(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = service.resolve_offline("run all unit tests")
        # Auto-detect falls back to pytest, which adds a warning
        assert any("auto-detect" in w.lower() for w in result.warnings)

    def test_low_confidence_flags_clarification(self, config_no_llm):
        # Offline parser always returns 0.6 confidence, set threshold high
        service = IntentParserService(
            config_no_llm, parse_mode=ParseMode.OFFLINE, clarification_threshold=0.8
        )
        result = service.resolve_offline("run tests")
        assert result.needs_clarification is True

    def test_normal_confidence_no_clarification(self, config_no_llm):
        # Offline parser returns 0.6, set threshold low
        service = IntentParserService(
            config_no_llm, parse_mode=ParseMode.OFFLINE, clarification_threshold=0.3
        )
        result = service.resolve_offline("run pytest")
        assert result.needs_clarification is False

    def test_summary_dict(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = service.resolve_offline("run pytest tests/unit")
        summary = result.summary()
        assert summary["intent"] == "run"
        assert summary["framework"] == "pytest"
        assert summary["scope"] == "tests/unit"
        assert summary["parse_mode"] == "offline"
        assert isinstance(summary["commands"], list)
        assert isinstance(summary["warnings"], list)

    def test_unknown_framework_raises_via_translator(self, config_no_llm):
        """Direct translator call with UNKNOWN framework raises UnsupportedFrameworkError."""
        from test_runner.execution.command_translator import UnsupportedFrameworkError

        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        parsed = ParsedTestRequest(
            intent=TestIntent.RUN,
            framework=TestFramework.UNKNOWN,
            raw_request="something unknown",
        )
        with pytest.raises(UnsupportedFrameworkError):
            service._translator.translate(parsed)

    def test_scope_preserved(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = service.resolve_offline("run pytest tests/test_foo.py")
        assert result.parsed_request.scope == "tests/test_foo.py"

    def test_timeout_and_env_passed_through(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = service.resolve_offline(
            "run pytest", timeout=60, env={"CI": "true"}
        )
        cmd = result.commands[0]
        assert cmd.timeout == 60
        assert cmd.env == {"CI": "true"}


# ---------------------------------------------------------------------------
# resolve (async) tests
# ---------------------------------------------------------------------------


class TestResolveAsync:
    @pytest.mark.asyncio
    async def test_offline_mode_uses_heuristics(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = await service.resolve("run pytest tests/")
        assert result.parse_mode_used == ParseMode.OFFLINE
        assert result.intent == TestIntent.RUN
        assert result.framework == TestFramework.PYTEST

    @pytest.mark.asyncio
    async def test_llm_mode_calls_parser(self, config_with_llm):
        expected_parsed = _make_parsed(confidence=0.95)
        mock_result = MagicMock()
        mock_result.final_output = expected_parsed

        with patch("test_runner.agents.parser.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            service = IntentParserService(config_with_llm, parse_mode=ParseMode.LLM)
            result = await service.resolve("run pytest tests/")

        assert result.parse_mode_used == ParseMode.LLM
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_auto_mode_tries_llm_first(self, config_with_llm):
        expected_parsed = _make_parsed(confidence=0.92)
        mock_result = MagicMock()
        mock_result.final_output = expected_parsed

        with patch("test_runner.agents.parser.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            service = IntentParserService(config_with_llm, parse_mode=ParseMode.AUTO)
            result = await service.resolve("run pytest tests/")

        assert result.parse_mode_used == ParseMode.LLM

    @pytest.mark.asyncio
    async def test_auto_mode_falls_back_to_offline_on_llm_failure(self, config_with_llm):
        with patch("test_runner.agents.parser.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = ParserError("LLM unavailable")
            service = IntentParserService(config_with_llm, parse_mode=ParseMode.AUTO)
            result = await service.resolve("run pytest")

        assert result.parse_mode_used == ParseMode.OFFLINE
        assert any("failed" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_auto_mode_no_llm_config_goes_offline(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.AUTO)
        result = await service.resolve("run pytest")
        assert result.parse_mode_used == ParseMode.OFFLINE

    @pytest.mark.asyncio
    async def test_llm_mode_without_config_raises(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.LLM)
        with pytest.raises(IntentResolutionError, match="LLM config is incomplete"):
            await service.resolve("run tests")

    @pytest.mark.asyncio
    async def test_resolution_has_commands(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = await service.resolve("run pytest tests/unit -v")
        assert len(result.commands) >= 1
        assert "pytest" in result.commands[0].command

    @pytest.mark.asyncio
    async def test_needs_clarification_at_threshold(self, config_no_llm):
        # Offline confidence is 0.6, threshold at 0.7 should flag clarification
        service = IntentParserService(
            config_no_llm, parse_mode=ParseMode.OFFLINE, clarification_threshold=0.7,
        )
        result = await service.resolve("run tests")
        assert result.needs_clarification is True

    @pytest.mark.asyncio
    async def test_resolve_passes_timeout_and_env(self, config_no_llm):
        service = IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)
        result = await service.resolve("run pytest", timeout=120, env={"FOO": "bar"})
        assert result.commands[0].timeout == 120
        assert result.commands[0].env == {"FOO": "bar"}


# ---------------------------------------------------------------------------
# IntentResolution model tests
# ---------------------------------------------------------------------------


class TestIntentResolution:
    def test_confidence_shortcut(self):
        parsed = _make_parsed(confidence=0.85)
        translation = TranslationResult(commands=[])
        resolution = IntentResolution(
            parsed_request=parsed,
            translation=translation,
            parse_mode_used=ParseMode.OFFLINE,
        )
        assert resolution.confidence == 0.85

    def test_commands_shortcut(self):
        parsed = _make_parsed()
        cmd = TestCommand(
            command=["pytest", "tests/"],
            display="pytest tests/",
            framework=TestFramework.PYTEST,
        )
        translation = TranslationResult(commands=[cmd])
        resolution = IntentResolution(
            parsed_request=parsed,
            translation=translation,
            parse_mode_used=ParseMode.LLM,
        )
        assert len(resolution.commands) == 1
        assert resolution.commands[0].command == ["pytest", "tests/"]

    def test_framework_shortcut(self):
        parsed = _make_parsed(framework=TestFramework.JEST)
        resolution = IntentResolution(
            parsed_request=parsed,
            translation=TranslationResult(commands=[]),
            parse_mode_used=ParseMode.OFFLINE,
        )
        assert resolution.framework == TestFramework.JEST

    def test_intent_shortcut(self):
        parsed = _make_parsed(intent=TestIntent.LIST)
        resolution = IntentResolution(
            parsed_request=parsed,
            translation=TranslationResult(commands=[]),
            parse_mode_used=ParseMode.OFFLINE,
        )
        assert resolution.intent == TestIntent.LIST

    def test_summary_keys(self):
        parsed = _make_parsed()
        resolution = IntentResolution(
            parsed_request=parsed,
            translation=TranslationResult(commands=[]),
            parse_mode_used=ParseMode.LLM,
            warnings=["some warning"],
            needs_clarification=True,
        )
        summary = resolution.summary()
        assert set(summary.keys()) == {
            "intent", "framework", "scope", "confidence",
            "parse_mode", "commands", "warnings",
            "needs_clarification", "reasoning",
        }
        assert summary["needs_clarification"] is True
        assert summary["warnings"] == ["some warning"]


# ---------------------------------------------------------------------------
# IntentResolutionError tests
# ---------------------------------------------------------------------------


class TestIntentResolutionError:
    def test_basic_error(self):
        err = IntentResolutionError("something failed")
        assert str(err) == "something failed"
        assert err.parsed_request is None

    def test_error_with_parsed_request(self):
        parsed = _make_parsed()
        err = IntentResolutionError("bad translation", parsed_request=parsed)
        assert err.parsed_request is parsed
        assert err.parsed_request.framework == TestFramework.PYTEST

    def test_is_exception(self):
        assert issubclass(IntentResolutionError, Exception)


# ---------------------------------------------------------------------------
# Cross-framework integration tests
# ---------------------------------------------------------------------------


class TestCrossFrameworkIntegration:
    """Verify the full pipeline works for each framework via offline parsing."""

    @pytest.fixture
    def service(self, config_no_llm):
        return IntentParserService(config_no_llm, parse_mode=ParseMode.OFFLINE)

    def test_pytest(self, service):
        result = service.resolve_offline("run pytest tests/unit")
        assert "pytest" in result.commands[0].command
        assert "tests/unit" in result.commands[0].command

    def test_jest(self, service):
        result = service.resolve_offline("run jest src/app.test.js")
        assert "jest" in result.commands[0].command

    def test_mocha(self, service):
        result = service.resolve_offline("run mocha tests/")
        assert "mocha" in result.commands[0].command

    def test_go_test(self, service):
        result = service.resolve_offline("run go test ./...")
        cmd = result.commands[0].command
        assert "go" in cmd
        assert "test" in cmd

    def test_cargo_test(self, service):
        result = service.resolve_offline("run cargo test")
        cmd = result.commands[0].command
        assert "cargo" in cmd
        assert "test" in cmd

    def test_dotnet_test(self, service):
        result = service.resolve_offline("run dotnet test")
        cmd = result.commands[0].command
        assert "dotnet" in cmd
        assert "test" in cmd

    def test_unittest(self, service):
        result = service.resolve_offline("run unittest")
        cmd = result.commands[0].command
        assert "unittest" in cmd

    def test_script(self, service):
        result = service.resolve_offline("run bash ./test.sh")
        assert result.framework == TestFramework.SCRIPT

    def test_list_intent_produces_list_command(self, service):
        result = service.resolve_offline("list pytest tests")
        cmd = result.commands[0].command
        assert "--collect-only" in cmd

    def test_rerun_intent_produces_rerun_command(self, service):
        result = service.resolve_offline("rerun failed pytest tests")
        cmd = result.commands[0].command
        assert "--lf" in cmd
