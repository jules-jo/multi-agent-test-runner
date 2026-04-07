"""Tests for runtime argument resolution of catalog-backed commands."""

from __future__ import annotations

import pytest

from test_runner.agents.parser import ParsedTestRequest, TestFramework, TestIntent
from test_runner.catalog_arguments import CatalogArgumentResolver
from test_runner.execution.command_translator import TestCommand


def _make_request(raw_request: str) -> ParsedTestRequest:
    return ParsedTestRequest(
        intent=TestIntent.RUN,
        framework=TestFramework.SCRIPT,
        raw_request=raw_request,
        confidence=0.9,
    )


def _make_command() -> TestCommand:
    return TestCommand(
        command=["python3.8", "agent_test.py"],
        display="python3.8 agent_test.py",
        framework=TestFramework.SCRIPT,
        working_directory="/repo",
        metadata={
            "catalog_alias": "agent test",
            "catalog_execution_type": "python_script",
            "catalog_target": "agent_test.py",
            "catalog_system_transport": "local",
            "catalog_system_config": {
                "python_command": "python3.8",
            },
        },
    )


class TestCatalogArgumentResolver:
    @pytest.mark.asyncio
    async def test_maps_runtime_iteration_request_from_help_text(self, monkeypatch):
        resolver = CatalogArgumentResolver()

        async def fake_probe(command):
            return (
                "Usage: agent_test.py [options]\n"
                "  -l, --loops LOOPS  Number of iterations to run\n"
                "  --mode MODE        Execution mode\n",
                (),
            )

        monkeypatch.setattr(resolver, "_probe_help_text", fake_probe)

        result = await resolver.resolve(
            _make_command(),
            request=_make_request("run agent test for 10 iterations"),
        )

        assert result.command is not None
        assert result.command.command == [
            "python3.8",
            "agent_test.py",
            "-l",
            "10",
        ]
        assert result.command.metadata["catalog_runtime_args"] == ["-l", "10"]
        assert result.needs_clarification is False

    @pytest.mark.asyncio
    async def test_unmapped_runtime_value_requires_clarification(self, monkeypatch):
        resolver = CatalogArgumentResolver()

        async def fake_probe(command):
            return (
                "Usage: agent_test.py [options]\n"
                "  --mode MODE        Execution mode\n",
                (),
            )

        monkeypatch.setattr(resolver, "_probe_help_text", fake_probe)

        result = await resolver.resolve(
            _make_command(),
            request=_make_request("run agent test for 10 iterations"),
        )

        assert result.command is None
        assert result.needs_clarification is True
        assert any("Could not map requested" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_request_without_runtime_value_keeps_original_command(self):
        resolver = CatalogArgumentResolver()
        async def fake_probe(command):
            return (
                "Usage: agent_test.py [options]\n"
                "  --mode MODE        Execution mode\n",
                (),
            )

        resolver._probe_help_text = fake_probe  # type: ignore[method-assign]

        command = _make_command()
        result = await resolver.resolve(
            command,
            request=_make_request("run agent test"),
        )

        assert result.command is not None
        assert result.command.command == command.command
        assert result.command.metadata["catalog_runtime_args"] == []
        assert result.needs_clarification is False
        assert result.warnings == ()

    @pytest.mark.asyncio
    async def test_missing_required_option_requires_clarification(self, monkeypatch):
        resolver = CatalogArgumentResolver()

        async def fake_probe(command):
            return (
                "usage: agent_test.py [-h] -l LOOPS\n"
                "  -l, --loops LOOPS  Number of iterations to run\n",
                (),
            )

        monkeypatch.setattr(resolver, "_probe_help_text", fake_probe)

        result = await resolver.resolve(
            _make_command(),
            request=_make_request("run agent test"),
        )

        assert result.command is None
        assert result.needs_clarification is True
        assert any("--loops" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_missing_required_positional_requires_clarification(self, monkeypatch):
        resolver = CatalogArgumentResolver()

        async def fake_probe(command):
            return (
                "usage: agent_test.py [-h] target\n"
                "positional arguments:\n"
                "  target              Target host\n",
                (),
            )

        monkeypatch.setattr(resolver, "_probe_help_text", fake_probe)

        result = await resolver.resolve(
            _make_command(),
            request=_make_request("run agent test"),
        )

        assert result.command is None
        assert result.needs_clarification is True
        assert any("target" in warning for warning in result.warnings)
