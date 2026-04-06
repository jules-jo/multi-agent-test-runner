"""Tests for configuration loading.

Covers:
  - Loading credentials from environment variables
  - Loading credentials from a .env file
  - Env var override: when both .env and env var exist, env var wins
  - Fallback from DATAIKU_* vars to generic LLM_* vars
  - Validation of required fields
  - Default values for optional fields
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from test_runner.config import AutonomyPolicy, Config, _resolve_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_env_file(tmp_path: Path, content: str) -> str:
    """Write a .env file and return its path as a string."""
    env_file = tmp_path / ".env"
    env_file.write_text(textwrap.dedent(content))
    return str(env_file)


# ---------------------------------------------------------------------------
# _resolve_env unit tests
# ---------------------------------------------------------------------------

class TestResolveEnv:
    """Tests for the _resolve_env helper."""

    def test_first_match_wins(self, monkeypatch):
        monkeypatch.setenv("A", "alpha")
        monkeypatch.setenv("B", "bravo")
        assert _resolve_env("A", "B") == "alpha"

    def test_skips_empty_to_fallback(self, monkeypatch):
        monkeypatch.setenv("A", "")
        monkeypatch.setenv("B", "bravo")
        assert _resolve_env("A", "B") == "bravo"

    def test_returns_default_when_none_set(self, monkeypatch):
        monkeypatch.delenv("A", raising=False)
        monkeypatch.delenv("B", raising=False)
        assert _resolve_env("A", "B", default="fallback") == "fallback"

    def test_skips_placeholder_values(self, monkeypatch):
        monkeypatch.setenv("A", "https://your-dataiku-instance.com/api/v1")
        monkeypatch.setenv("B", "real-value")
        assert _resolve_env("A", "B") == "real-value"


# ---------------------------------------------------------------------------
# Config.load – environment variable tests
# ---------------------------------------------------------------------------

class TestConfigLoadEnvVars:
    """Tests for Config.load() using only environment variables."""

    def test_env_vars_loaded(self, monkeypatch):
        monkeypatch.setenv("DATAIKU_LLM_MESH_URL", "https://mesh.example.com")
        monkeypatch.setenv("DATAIKU_API_KEY", "sk-test")
        monkeypatch.setenv("DATAIKU_MODEL_ID", "gpt-4")
        config = Config.load()
        assert config.llm_base_url == "https://mesh.example.com"
        assert config.api_key == "sk-test"
        assert config.model_id == "gpt-4"

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("DATAIKU_LLM_MESH_URL", raising=False)
        monkeypatch.delenv("DATAIKU_API_KEY", raising=False)
        monkeypatch.delenv("DATAIKU_MODEL_ID", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("AUTONOMY_POLICY", raising=False)
        config = Config.load()
        assert config.autonomy_policy == AutonomyPolicy.CONSERVATIVE
        assert config.log_level == "INFO"

    def test_autonomy_policy_from_env(self, monkeypatch):
        monkeypatch.setenv("AUTONOMY_POLICY", "aggressive")
        config = Config.load()
        assert config.autonomy_policy == AutonomyPolicy.AGGRESSIVE

    def test_invalid_autonomy_falls_back(self, monkeypatch):
        monkeypatch.setenv("AUTONOMY_POLICY", "yolo")
        config = Config.load()
        assert config.autonomy_policy == AutonomyPolicy.CONSERVATIVE

    def test_default_catalog_path_autodiscovered(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TEST_CATALOG_PATH", raising=False)
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()
        catalog_path = registry_dir / "catalog.json"
        catalog_path.write_text('{"version": 1, "entries": []}', encoding="utf-8")

        config = Config.load()

        assert config.test_catalog_path == str(catalog_path)

    def test_env_catalog_path_overrides_repo_default(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()
        (registry_dir / "catalog.json").write_text(
            '{"version": 1, "entries": []}',
            encoding="utf-8",
        )
        monkeypatch.setenv("TEST_CATALOG_PATH", "/tmp/override-catalog.json")

        config = Config.load()

        assert config.test_catalog_path == "/tmp/override-catalog.json"

    def test_empty_catalog_env_disables_repo_autodiscovery(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()
        (registry_dir / "catalog.json").write_text(
            '{"version": 1, "entries": []}',
            encoding="utf-8",
        )
        monkeypatch.setenv("TEST_CATALOG_PATH", "")

        config = Config.load()

        assert config.test_catalog_path == ""


# ---------------------------------------------------------------------------
# Config.load – .env file tests
# ---------------------------------------------------------------------------

class TestConfigLoadDotEnvFile:
    """Tests for Config.load() reading values from a .env file."""

    def test_loads_credentials_from_env_file(self, tmp_path, monkeypatch):
        """Credentials are read from the .env file when no env vars are set."""
        # Clear any existing env vars so .env file values are used
        for var in (
            "DATAIKU_LLM_MESH_URL", "DATAIKU_API_KEY", "DATAIKU_MODEL_ID",
            "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)

        env_file = _write_env_file(tmp_path, """\
            DATAIKU_LLM_MESH_URL=https://from-dotenv.example.com/api/v1
            DATAIKU_API_KEY=sk-from-dotenv
            DATAIKU_MODEL_ID=model-from-dotenv
        """)

        config = Config.load(env_file=env_file)
        assert config.llm_base_url == "https://from-dotenv.example.com/api/v1"
        assert config.api_key == "sk-from-dotenv"
        assert config.model_id == "model-from-dotenv"

    def test_env_var_overrides_dotenv_file(self, tmp_path, monkeypatch):
        """When both .env file and env vars exist, env vars take precedence."""
        # Set env var for API key only
        monkeypatch.setenv("DATAIKU_API_KEY", "sk-from-environment")
        # Clear others so they come from file
        monkeypatch.delenv("DATAIKU_LLM_MESH_URL", raising=False)
        monkeypatch.delenv("DATAIKU_MODEL_ID", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        env_file = _write_env_file(tmp_path, """\
            DATAIKU_LLM_MESH_URL=https://from-dotenv.example.com/api/v1
            DATAIKU_API_KEY=sk-from-dotenv-SHOULD-BE-OVERRIDDEN
            DATAIKU_MODEL_ID=model-from-dotenv
        """)

        config = Config.load(env_file=env_file)
        # URL and model come from .env file
        assert config.llm_base_url == "https://from-dotenv.example.com/api/v1"
        assert config.model_id == "model-from-dotenv"
        # API key comes from env var, NOT from .env file
        assert config.api_key == "sk-from-environment"

    def test_all_env_vars_override_all_dotenv(self, tmp_path, monkeypatch):
        """Every field can be overridden by its corresponding env var."""
        monkeypatch.setenv("DATAIKU_LLM_MESH_URL", "https://env-override.com")
        monkeypatch.setenv("DATAIKU_API_KEY", "sk-env-override")
        monkeypatch.setenv("DATAIKU_MODEL_ID", "model-env-override")
        monkeypatch.setenv("AUTONOMY_POLICY", "moderate")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        env_file = _write_env_file(tmp_path, """\
            DATAIKU_LLM_MESH_URL=https://file.com
            DATAIKU_API_KEY=sk-file
            DATAIKU_MODEL_ID=model-file
            AUTONOMY_POLICY=aggressive
            LOG_LEVEL=ERROR
        """)

        config = Config.load(env_file=env_file)
        assert config.llm_base_url == "https://env-override.com"
        assert config.api_key == "sk-env-override"
        assert config.model_id == "model-env-override"
        assert config.autonomy_policy == AutonomyPolicy.MODERATE
        assert config.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Config.load – generic LLM_* fallback tests
# ---------------------------------------------------------------------------

class TestConfigGenericFallback:
    """Tests for fallback from DATAIKU_* to generic LLM_* variable names."""

    def test_llm_fallback_vars(self, tmp_path, monkeypatch):
        """Generic LLM_* vars are used when DATAIKU_* vars are absent."""
        for var in (
            "DATAIKU_LLM_MESH_URL", "DATAIKU_API_KEY", "DATAIKU_MODEL_ID",
        ):
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("LLM_BASE_URL", "https://generic-llm.example.com")
        monkeypatch.setenv("LLM_API_KEY", "sk-generic")
        monkeypatch.setenv("LLM_MODEL", "generic-model")

        config = Config.load()
        assert config.llm_base_url == "https://generic-llm.example.com"
        assert config.api_key == "sk-generic"
        assert config.model_id == "generic-model"

    def test_dataiku_vars_take_priority_over_generic(self, monkeypatch):
        """DATAIKU_* vars win over LLM_* vars."""
        monkeypatch.setenv("DATAIKU_LLM_MESH_URL", "https://dataiku.com")
        monkeypatch.setenv("DATAIKU_API_KEY", "sk-dataiku")
        monkeypatch.setenv("DATAIKU_MODEL_ID", "dataiku-model")
        monkeypatch.setenv("LLM_BASE_URL", "https://generic.com")
        monkeypatch.setenv("LLM_API_KEY", "sk-generic")
        monkeypatch.setenv("LLM_MODEL", "generic-model")

        config = Config.load()
        assert config.llm_base_url == "https://dataiku.com"
        assert config.api_key == "sk-dataiku"
        assert config.model_id == "dataiku-model"

    def test_generic_vars_from_dotenv_file(self, tmp_path, monkeypatch):
        """Generic LLM_* vars in .env file are used as fallback."""
        for var in (
            "DATAIKU_LLM_MESH_URL", "DATAIKU_API_KEY", "DATAIKU_MODEL_ID",
            "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)

        env_file = _write_env_file(tmp_path, """\
            LLM_BASE_URL=https://generic-from-file.example.com
            LLM_API_KEY=sk-generic-file
            LLM_MODEL=generic-file-model
        """)

        config = Config.load(env_file=env_file)
        assert config.llm_base_url == "https://generic-from-file.example.com"
        assert config.api_key == "sk-generic-file"
        assert config.model_id == "generic-file-model"


# ---------------------------------------------------------------------------
# Config.validate
# ---------------------------------------------------------------------------

class TestConfigValidate:
    """Tests for Config.validate()."""

    def test_valid_config_no_errors(self, monkeypatch):
        monkeypatch.setenv("DATAIKU_LLM_MESH_URL", "https://mesh.example.com")
        monkeypatch.setenv("DATAIKU_API_KEY", "sk-test")
        monkeypatch.setenv("DATAIKU_MODEL_ID", "gpt-4")
        config = Config.load()
        assert config.validate() == []

    def test_missing_fields_reported(self, monkeypatch):
        monkeypatch.delenv("DATAIKU_LLM_MESH_URL", raising=False)
        monkeypatch.delenv("DATAIKU_API_KEY", raising=False)
        monkeypatch.delenv("DATAIKU_MODEL_ID", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        config = Config.load()
        errors = config.validate()
        assert len(errors) == 3
