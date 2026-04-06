"""Tests for the closed-world test catalog."""

from __future__ import annotations

import pytest

from test_runner.agents.parser import ParsedTestRequest, TestFramework, TestIntent
from test_runner.catalog import (
    CatalogRegistry,
    CatalogDocument,
    CatalogEntry,
    CatalogExecutionType,
    CatalogMatchStatus,
    CatalogSystem,
    CatalogSystemTransport,
)


def _make_request(
    *,
    intent: TestIntent = TestIntent.RUN,
    raw_request: str = "run lt",
    extra_args: list[str] | None = None,
) -> ParsedTestRequest:
    return ParsedTestRequest(
        intent=intent,
        framework=TestFramework.UNKNOWN,
        confidence=0.4,
        raw_request=raw_request,
        extra_args=extra_args or [],
    )


def _make_registry() -> CatalogRegistry:
    return CatalogRegistry(
        entries=[
            CatalogEntry(
                alias="lt",
                description="Run local smoke tests",
                execution_type=CatalogExecutionType.PYTHON_SCRIPT,
                target="scripts/local_smoke.py",
                args=["--quick"],
                keywords=["local smoke", "smoke suite"],
                system="devbox",
                env={"CATALOG": "1"},
                timeout=90,
            ),
            CatalogEntry(
                alias="device-check",
                description="Run hardware validation binary",
                execution_type=CatalogExecutionType.EXECUTABLE,
                target="./bin/device-check",
                keywords=["hardware validation", "device validation"],
            ),
        ],
        systems=[
            CatalogSystem(
                alias="devbox",
                description="Primary local development environment",
                transport=CatalogSystemTransport.LOCAL,
                working_directory="/repo",
                env={"SYSTEM_ENV": "devbox"},
            )
        ],
    )


class TestCatalogRegistryMatching:
    def test_loads_from_json_path(self, tmp_path) -> None:
        path = tmp_path / "catalog.json"
        document = CatalogDocument(
            systems=[
                CatalogSystem(
                    alias="devbox",
                    transport=CatalogSystemTransport.LOCAL,
                )
            ],
            entries=[
                CatalogEntry(
                    alias="lt",
                    execution_type=CatalogExecutionType.PYTHON_SCRIPT,
                    target="scripts/local_smoke.py",
                )
            ]
        )
        path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

        registry = CatalogRegistry.from_path(path)

        assert registry.aliases == ("lt",)
        assert tuple(system.alias for system in registry.systems) == ("devbox", "local")

    def test_duplicate_aliases_raise(self) -> None:
        with pytest.raises(ValueError, match="Duplicate catalog alias"):
            CatalogRegistry(
                [
                    CatalogEntry(
                        alias="lt",
                        execution_type=CatalogExecutionType.PYTHON_SCRIPT,
                        target="a.py",
                    ),
                    CatalogEntry(
                        alias="lt",
                        execution_type=CatalogExecutionType.EXECUTABLE,
                        target="./bin/lt",
                    ),
                ]
            )

    def test_matches_exact_alias(self) -> None:
        registry = _make_registry()

        match = registry.match_request("please run lt now")

        assert match.status == CatalogMatchStatus.MATCHED
        assert match.entry is not None
        assert match.entry.alias == "lt"

    def test_matches_keyword_when_alias_missing(self) -> None:
        registry = _make_registry()

        match = registry.match_request("run the hardware validation")

        assert match.status == CatalogMatchStatus.MATCHED
        assert match.entry is not None
        assert match.entry.alias == "device-check"

    def test_unknown_request_is_missing(self) -> None:
        registry = _make_registry()

        match = registry.match_request("run the secret suite")

        assert match.status == CatalogMatchStatus.MISSING
        assert "Known aliases" in match.message

    def test_multiple_keyword_matches_require_clarification(self) -> None:
        registry = CatalogRegistry(
            [
                CatalogEntry(
                    alias="alpha",
                    execution_type=CatalogExecutionType.EXECUTABLE,
                    target="./alpha",
                    keywords=["smoke"],
                ),
                CatalogEntry(
                    alias="beta",
                    execution_type=CatalogExecutionType.EXECUTABLE,
                    target="./beta",
                    keywords=["smoke"],
                ),
            ]
        )

        match = registry.match_request("run smoke")

        assert match.status == CatalogMatchStatus.AMBIGUOUS
        assert "clarification" in match.message


class TestCatalogRegistryTranslation:
    def test_python_script_entry_builds_saved_command(self) -> None:
        registry = _make_registry()
        match = registry.match_request("run lt")

        result = registry.translate_match(match, _make_request())

        assert len(result.commands) == 1
        command = result.commands[0]
        assert command.command == ["python", "scripts/local_smoke.py", "--quick"]
        assert command.framework == TestFramework.SCRIPT
        assert command.working_directory == "/repo"
        assert command.env == {"CATALOG": "1", "SYSTEM_ENV": "devbox"}
        assert command.timeout == 90
        assert command.metadata["catalog_alias"] == "lt"
        assert command.metadata["catalog_system"] == "devbox"

    def test_executable_entry_builds_saved_command(self) -> None:
        registry = _make_registry()
        match = registry.match_request("run device-check")

        result = registry.translate_match(match, _make_request(raw_request="run device-check"))

        assert result.commands[0].command == ["./bin/device-check"]
        assert result.commands[0].metadata["catalog_system"] == "local"

    def test_runtime_env_overrides_catalog_env(self) -> None:
        registry = _make_registry()
        match = registry.match_request("run lt")

        result = registry.translate_match(
            match,
            _make_request(),
            env={"PATH": "/venv/bin", "CATALOG": "override"},
        )

        assert result.commands[0].env == {
            "CATALOG": "override",
            "PATH": "/venv/bin",
            "SYSTEM_ENV": "devbox",
        }

    def test_extra_args_are_ignored_in_catalog_mode(self) -> None:
        registry = _make_registry()
        match = registry.match_request("run lt -x")

        result = registry.translate_match(
            match,
            _make_request(extra_args=["-x"]),
        )

        assert result.commands[0].command == ["python", "scripts/local_smoke.py", "--quick"]
        assert any("Ignoring ad hoc extra arguments" in warning for warning in result.warnings)

    def test_non_run_intents_require_clarification(self) -> None:
        registry = _make_registry()
        match = registry.match_request("list lt")

        result = registry.translate_match(
            match,
            _make_request(intent=TestIntent.LIST, raw_request="list lt"),
        )

        assert result.commands == []
        assert any("not implemented" in warning for warning in result.warnings)

    def test_missing_match_returns_no_commands(self) -> None:
        registry = _make_registry()
        match = registry.match_request("run missing")

        result = registry.translate_match(
            match,
            _make_request(raw_request="run missing"),
        )

        assert result.commands == []
        assert result.warnings == [match.message]

    def test_unknown_system_blocks_execution(self) -> None:
        registry = CatalogRegistry(
            entries=[
                CatalogEntry(
                    alias="lt",
                    execution_type=CatalogExecutionType.PYTHON_SCRIPT,
                    target="scripts/local_smoke.py",
                    system="lab-a",
                )
            ]
        )

        match = registry.match_request("run lt")
        result = registry.translate_match(match, _make_request())

        assert result.commands == []
        assert any("unknown system" in warning for warning in result.warnings)

    def test_remote_system_builds_saved_command_with_system_metadata(self) -> None:
        registry = CatalogRegistry(
            entries=[
                CatalogEntry(
                    alias="device-check",
                    execution_type=CatalogExecutionType.EXECUTABLE,
                    target="./bin/device-check",
                    system="lab-a",
                )
            ],
            systems=[
                CatalogSystem(
                    alias="lab-a",
                    transport=CatalogSystemTransport.SSH,
                    hostname="lab-a.internal.example",
                    username="runner",
                    credential_ref="ssh-config:lab-a",
                )
            ],
        )

        match = registry.match_request("run device-check")
        result = registry.translate_match(
            match,
            _make_request(raw_request="run device-check"),
        )

        assert len(result.commands) == 1
        command = result.commands[0]
        assert command.command == ["./bin/device-check"]
        assert command.metadata["catalog_system"] == "lab-a"
        assert command.metadata["catalog_system_transport"] == "ssh"
        assert command.metadata["catalog_system_config"]["hostname"] == "lab-a.internal.example"

    def test_remote_system_ignores_local_runtime_env_overrides(self) -> None:
        registry = CatalogRegistry(
            entries=[
                CatalogEntry(
                    alias="device-check",
                    execution_type=CatalogExecutionType.EXECUTABLE,
                    target="./bin/device-check",
                    system="lab-a",
                )
            ],
            systems=[
                CatalogSystem(
                    alias="lab-a",
                    transport=CatalogSystemTransport.SSH,
                    hostname="lab-a.internal.example",
                    username="runner",
                    env={"REMOTE_ONLY": "1"},
                )
            ],
        )

        match = registry.match_request("run device-check")
        result = registry.translate_match(
            match,
            _make_request(raw_request="run device-check"),
            env={"PATH": "/local/venv/bin"},
        )

        assert len(result.commands) == 1
        assert result.commands[0].env == {"REMOTE_ONLY": "1"}
