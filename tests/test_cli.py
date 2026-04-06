"""Tests for the CLI interface."""

from __future__ import annotations

import io
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from test_runner.catalog import (
    CatalogDocument,
    CatalogEntry,
    CatalogExecutionType,
    CatalogRepository,
)
from test_runner.cli import (
    InputValidationError,
    build_parser,
    validate_request,
    async_main,
    interactive_loop_async,
    _resolve_request,
    _build_default_command_env,
    MIN_REQUEST_LENGTH,
    MAX_REQUEST_LENGTH,
    EXIT_COMMANDS,
)
from test_runner.orchestrator.hub import RunPhase, RunState


# ---------------------------------------------------------------------------
# validate_request
# ---------------------------------------------------------------------------

class TestValidateRequest:
    """Tests for validate_request()."""

    def test_valid_request_returned_stripped(self):
        assert validate_request("  run pytest  ") == "run pytest"

    def test_empty_string_raises(self):
        with pytest.raises(InputValidationError, match="empty"):
            validate_request("")

    def test_whitespace_only_raises(self):
        with pytest.raises(InputValidationError, match="empty"):
            validate_request("   \n\t  ")

    def test_too_short_raises(self):
        with pytest.raises(InputValidationError, match="too short"):
            validate_request("ab")

    def test_minimum_length_accepted(self):
        result = validate_request("a" * MIN_REQUEST_LENGTH)
        assert len(result) == MIN_REQUEST_LENGTH

    def test_too_long_raises(self):
        with pytest.raises(InputValidationError, match="too long"):
            validate_request("a" * (MAX_REQUEST_LENGTH + 1))

    def test_max_length_accepted(self):
        result = validate_request("a" * MAX_REQUEST_LENGTH)
        assert len(result) == MAX_REQUEST_LENGTH

    def test_multiline_input_accepted(self):
        text = "run pytest\nwith verbose output"
        result = validate_request(text)
        assert result == text


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    """Tests for the argparse configuration."""

    def test_parser_accepts_request_words(self):
        parser = build_parser()
        args = parser.parse_args(["run", "all", "unit", "tests"])
        assert args.request == ["run", "all", "unit", "tests"]

    def test_one_shot_quoted_request(self):
        parser = build_parser()
        args = parser.parse_args(["run all unit tests"])
        assert "run all unit tests" in " ".join(args.request)

    def test_interactive_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-i"])
        assert args.interactive is True
        assert args.request == []

    def test_working_dir(self):
        parser = build_parser()
        args = parser.parse_args(["-w", "/tmp/project", "run tests"])
        assert args.working_dir == "/tmp/project"

    def test_parser_env_file_option(self):
        parser = build_parser()
        args = parser.parse_args(["--env-file", "/tmp/.env", "run tests"])
        assert args.env_file == "/tmp/.env"

    def test_timeout_override(self):
        parser = build_parser()
        args = parser.parse_args(["--timeout", "60", "run tests"])
        assert args.timeout == 60

    def test_parser_target_choices(self):
        parser = build_parser()
        for target in ("local", "docker", "ci"):
            args = parser.parse_args(["--target", target, "run tests"])
            assert args.target == target

    def test_invalid_target_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--target", "kubernetes", "run tests"])

    def test_parser_multiple_report_channels(self):
        parser = build_parser()
        args = parser.parse_args([
            "--report-channel", "cli",
            "--report-channel", "json",
            "run tests",
        ])
        assert args.report_channel == ["cli", "json"]

    def test_parser_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--dry-run", "run tests"])
        assert args.dry_run is True

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-v", "run tests"])
        assert args.verbose is True

    def test_parser_default_target_is_local(self):
        parser = build_parser()
        args = parser.parse_args(["run tests"])
        assert args.target == "local"

    def test_no_args_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.request == []
        assert args.interactive is False
        assert args.dry_run is False
        assert args.verbose is False


# ---------------------------------------------------------------------------
# _resolve_request
# ---------------------------------------------------------------------------

class TestResolveRequest:
    """Tests for request source resolution."""

    def test_interactive_flag_returns_none(self):
        parser = build_parser()
        args = parser.parse_args(["-i"])
        assert _resolve_request(args) is None

    def test_positional_args_joined(self):
        parser = build_parser()
        args = parser.parse_args(["run", "all", "tests"])
        assert _resolve_request(args) == "run all tests"

    def test_empty_request_no_tty_reads_stdin(self, monkeypatch):
        """When no args and stdin is not a tty, read from stdin."""
        parser = build_parser()
        args = parser.parse_args([])
        fake_stdin = io.StringIO("piped request\n")
        monkeypatch.setattr("sys.stdin", fake_stdin)
        result = _resolve_request(args)
        assert result == "piped request\n"

    def test_empty_request_tty_returns_none(self, monkeypatch):
        """When no args and stdin IS a tty, return None (interactive)."""
        parser = build_parser()
        args = parser.parse_args([])
        fake_stdin = io.StringIO("")
        fake_stdin.isatty = lambda: True  # type: ignore[attr-defined]
        monkeypatch.setattr("sys.stdin", fake_stdin)
        result = _resolve_request(args)
        assert result is None


# ---------------------------------------------------------------------------
# _build_default_command_env
# ---------------------------------------------------------------------------


class TestBuildDefaultCommandEnv:
    def test_non_local_target_returns_empty(self):
        assert _build_default_command_env("docker") == {}

    def test_local_target_includes_repo_venv_scripts_on_windows_layout(
        self, monkeypatch, tmp_path
    ):
        exec_dir = tmp_path / "host-python"
        exec_dir.mkdir()
        fake_python = exec_dir / "python.exe"
        fake_python.write_text("")

        scripts_dir = tmp_path / ".venv" / "Scripts"
        scripts_dir.mkdir(parents=True)

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("PATH", "existing-path")

        env = _build_default_command_env("local", working_dir=str(tmp_path))

        assert "PATH" in env
        parts = env["PATH"].split(os.pathsep)
        assert parts[0] == str(exec_dir)
        assert str(scripts_dir) in parts[:2]


# ---------------------------------------------------------------------------
# async_main integration tests
# ---------------------------------------------------------------------------

class TestAsyncMain:
    """Integration tests for the async entry point."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        """Ensure no leftover env vars affect tests."""
        for key in (
            "DATAIKU_LLM_MESH_URL",
            "DATAIKU_API_KEY",
            "DATAIKU_MODEL_ID",
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "LLM_MODEL",
            "TEST_CATALOG_PATH",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TEST_CATALOG_PATH", "")

    @pytest.mark.asyncio
    async def test_dry_run_succeeds(self):
        code = await async_main(["--dry-run", "run all pytest tests"])
        assert code == 0

    @pytest.mark.asyncio
    async def test_short_request_validation_error(self):
        code = await async_main(["ab"])
        assert code == 1

    @pytest.mark.asyncio
    async def test_dry_run_with_missing_config_still_works(self, monkeypatch):
        """Dry run does not need LLM config."""
        result = await async_main(["--dry-run", "run all unit tests"])
        assert result == 0

    @pytest.mark.asyncio
    async def test_dry_run_catalog_mode_requires_saved_alias(
        self, monkeypatch, tmp_path
    ):
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(
            CatalogDocument(
                entries=[
                    CatalogEntry(
                        alias="lt",
                        execution_type=CatalogExecutionType.PYTHON_SCRIPT,
                        target="scripts/local_smoke.py",
                    )
                ]
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        monkeypatch.setenv("TEST_CATALOG_PATH", str(catalog_path))

        result = await async_main(["--dry-run", "run missing suite"])

        assert result == 1

    @pytest.mark.asyncio
    async def test_dry_run_catalog_mode_accepts_saved_alias(
        self, monkeypatch, tmp_path
    ):
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(
            CatalogDocument(
                entries=[
                    CatalogEntry(
                        alias="lt",
                        execution_type=CatalogExecutionType.PYTHON_SCRIPT,
                        target="scripts/local_smoke.py",
                    )
                ]
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        monkeypatch.setenv("TEST_CATALOG_PATH", str(catalog_path))

        result = await async_main(["--dry-run", "run lt"])

        assert result == 0

    @pytest.mark.asyncio
    async def test_dry_run_uses_repo_default_catalog_when_present(
        self, monkeypatch, tmp_path
    ):
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()
        catalog_path = registry_dir / "catalog.json"
        catalog_path.write_text(
            CatalogDocument(
                entries=[
                    CatalogEntry(
                        alias="lt",
                        execution_type=CatalogExecutionType.PYTHON_SCRIPT,
                        target="scripts/local_smoke.py",
                    )
                ]
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TEST_CATALOG_PATH", raising=False)

        result = await async_main(["--dry-run", "run missing suite"])

        assert result == 1

    @pytest.mark.asyncio
    async def test_real_run_without_config_uses_offline_mode(self, monkeypatch):
        """A real run can proceed offline when no LLM config is present."""
        state = RunState(request="run all unit tests", phase=RunPhase.COMPLETE)
        fake_hub = SimpleNamespace(run=AsyncMock(return_value=state))
        monkeypatch.setattr("test_runner.cli._create_orchestrator", lambda config, args: fake_hub)

        code = await async_main(["run all unit tests"])
        assert code == 0
        fake_hub.run.assert_awaited_once_with("run all unit tests")

    @pytest.mark.asyncio
    async def test_one_shot_with_config_accepted(self, monkeypatch):
        monkeypatch.setenv("DATAIKU_LLM_MESH_URL", "https://mesh.example.com")
        monkeypatch.setenv("DATAIKU_API_KEY", "sk-test")
        monkeypatch.setenv("DATAIKU_MODEL_ID", "gpt-4")
        state = RunState(request="run pytest in tests/", phase=RunPhase.COMPLETE)
        fake_hub = SimpleNamespace(run=AsyncMock(return_value=state))
        monkeypatch.setattr("test_runner.cli._create_orchestrator", lambda config, args: fake_hub)
        code = await async_main(["run pytest in tests/"])
        assert code == 0
        fake_hub.run.assert_awaited_once_with("run pytest in tests/")

    @pytest.mark.asyncio
    async def test_failed_orchestrator_run_returns_nonzero(self, monkeypatch):
        state = RunState(request="run tests", phase=RunPhase.FAILED)
        state.errors.append("boom")
        fake_hub = SimpleNamespace(run=AsyncMock(return_value=state))
        monkeypatch.setattr("test_runner.cli._create_orchestrator", lambda config, args: fake_hub)

        code = await async_main(["run tests"])
        assert code == 1

    @pytest.mark.asyncio
    async def test_greeting_does_not_invoke_orchestrator(self, monkeypatch):
        fake_hub = SimpleNamespace(run=AsyncMock())
        monkeypatch.setattr("test_runner.cli._create_orchestrator", lambda config, args: fake_hub)

        code = await async_main(["hello"])

        assert code == 0
        fake_hub.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_help_does_not_invoke_orchestrator(self, monkeypatch):
        fake_hub = SimpleNamespace(run=AsyncMock())
        monkeypatch.setattr("test_runner.cli._create_orchestrator", lambda config, args: fake_hub)

        code = await async_main(["help"])

        assert code == 0
        fake_hub.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_version_flag(self):
        with pytest.raises(SystemExit) as exc_info:
            await async_main(["-V"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

class TestInteractiveMode:
    """Tests for the interactive REPL."""

    def test_exit_commands_defined(self):
        assert "quit" in EXIT_COMMANDS
        assert "exit" in EXIT_COMMANDS
        assert "q" in EXIT_COMMANDS
        assert ":q" in EXIT_COMMANDS

    @pytest.mark.asyncio
    async def test_async_interactive_loop_dispatches_requests(self, monkeypatch):
        seen: list[str] = []
        inputs = iter(["run tests", "quit"])

        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        async def dispatch(request: str) -> int:
            seen.append(request)
            return 0

        code = await interactive_loop_async(dispatch_fn=dispatch)
        assert code == 0
        assert seen == ["run tests"]

    @pytest.mark.asyncio
    async def test_async_main_interactive_mode_uses_orchestrator(self, monkeypatch):
        state = RunState(request="run tests", phase=RunPhase.COMPLETE)
        fake_hub = SimpleNamespace(run=AsyncMock(return_value=state))
        monkeypatch.setattr("test_runner.cli._create_orchestrator", lambda config, args: fake_hub)

        inputs = iter(["run tests", "quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        code = await async_main(["-i"])

        assert code == 0
        fake_hub.run.assert_awaited_once_with("run tests")

    @pytest.mark.asyncio
    async def test_interactive_greeting_stays_local_then_real_request_runs(self, monkeypatch):
        state = RunState(request="run tests", phase=RunPhase.COMPLETE)
        fake_hub = SimpleNamespace(run=AsyncMock(return_value=state))
        monkeypatch.setattr("test_runner.cli._create_orchestrator", lambda config, args: fake_hub)

        inputs = iter(["hello", "run tests", "quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        code = await async_main(["-i"])

        assert code == 0
        fake_hub.run.assert_awaited_once_with("run tests")

    @pytest.mark.asyncio
    async def test_interactive_unknown_request_can_register_and_rerun_saved_alias(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()
        catalog_path = registry_dir / "catalog.json"
        catalog_path.write_text(
            CatalogDocument().model_dump_json(indent=2),
            encoding="utf-8",
        )
        monkeypatch.delenv("TEST_CATALOG_PATH", raising=False)

        failed_state = RunState(request="run lt test", phase=RunPhase.FAILED)
        failed_state.errors.append("Request did not match any cataloged test definition.")
        success_state = RunState(request="run lt", phase=RunPhase.COMPLETE)
        fake_hub = SimpleNamespace(
            run=AsyncMock(side_effect=[failed_state, success_state])
        )
        monkeypatch.setattr(
            "test_runner.cli._create_orchestrator",
            lambda config, args: fake_hub,
        )

        inputs = iter([
            "run lt test",
            "y",
            "lt",
            "python",
            "scripts/local_smoke.py",
            "",
            "",
            "--quick",
            "lt test, local smoke",
            "",
            "",
            "y",
            "y",
            "quit",
        ])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        code = await async_main(["-i"])

        assert code == 0
        assert fake_hub.run.await_args_list[0].args == ("run lt test",)
        assert fake_hub.run.await_args_list[1].args == ("run lt",)

        document = CatalogRepository(catalog_path).load_document()
        assert len(document.entries) == 1
        assert document.entries[0].alias == "lt"
        assert document.entries[0].target == "scripts/local_smoke.py"
        assert document.entries[0].args == ["--quick"]
