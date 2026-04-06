"""Configuration management with .env file support and env var fallback.

Loading precedence:
  1. Environment variables (highest priority — override everything)
  2. Values from .env file (loaded via python-dotenv)
  3. Built-in defaults (lowest priority)

Variable resolution supports both Dataiku-specific names and generic LLM names:
  - DATAIKU_LLM_MESH_URL  →  falls back to  LLM_BASE_URL
  - DATAIKU_API_KEY        →  falls back to  LLM_API_KEY
  - DATAIKU_MODEL_ID       →  falls back to  LLM_MODEL
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PLACEHOLDER_ENV_VALUES = frozenset(
    {
        "https://your-dataiku-instance.com/api/v1",
        "your-api-key-here",
        "your-model-id",
    }
)

_DEFAULT_CATALOG_RELATIVE_PATH = Path("registry") / "catalog.json"


class AutonomyPolicy(str, Enum):
    """Configurable autonomy levels for agent behavior."""

    CONSERVATIVE = "conservative"  # Always ask before acting
    MODERATE = "moderate"          # Act autonomously within safe bounds
    AGGRESSIVE = "aggressive"      # Act autonomously, only ask for destructive ops


def _resolve_env(*names: str, default: str = "") -> str:
    """Return the first non-empty value found across *names* in ``os.environ``.

    This allows Dataiku-specific var names to take priority while falling back
    to generic LLM var names, and finally to *default*. Example placeholder
    values from ``.env.example`` are treated as unset so copied stubs do not
    masquerade as valid live credentials.
    """
    for name in names:
        value = os.environ.get(name, "")
        if value and value not in _PLACEHOLDER_ENV_VALUES:
            return value
    return default


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment.

    Use :meth:`Config.load` to create an instance — it handles reading the
    ``.env`` file and resolving variable precedence automatically.
    """

    # Dataiku LLM Mesh connection
    llm_base_url: str
    api_key: str
    model_id: str

    # Behavior
    autonomy_policy: AutonomyPolicy = AutonomyPolicy.CONSERVATIVE
    log_level: str = "INFO"

    # Execution defaults
    working_directory: str = field(default_factory=os.getcwd)
    timeout_seconds: int = 300
    test_catalog_path: str = ""

    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "Config":
        """Load configuration from .env file and environment variables.

        Environment variables always take precedence over .env file values
        because ``load_dotenv`` is called with ``override=False``.
        """
        # Resolve .env file path
        env_path = Path(env_file) if env_file else Path.cwd() / ".env"

        # load_dotenv with override=False will NOT overwrite env vars that
        # already exist in the process environment — this is what guarantees
        # "env var takes precedence over .env file".
        load_dotenv(dotenv_path=env_path, override=False)

        # Resolve credentials with fallback chain:
        #   DATAIKU_LLM_MESH_URL  →  LLM_BASE_URL  →  ""
        #   DATAIKU_API_KEY       →  LLM_API_KEY    →  ""
        #   DATAIKU_MODEL_ID      →  LLM_MODEL      →  ""
        llm_base_url = _resolve_env("DATAIKU_LLM_MESH_URL", "LLM_BASE_URL")
        api_key = _resolve_env("DATAIKU_API_KEY", "LLM_API_KEY")
        model_id = _resolve_env("DATAIKU_MODEL_ID", "LLM_MODEL")

        autonomy_raw = os.environ.get("AUTONOMY_POLICY", "conservative").lower()
        try:
            autonomy = AutonomyPolicy(autonomy_raw)
        except ValueError:
            autonomy = AutonomyPolicy.CONSERVATIVE

        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        catalog_path_env = os.environ.get("TEST_CATALOG_PATH")
        if catalog_path_env is not None:
            test_catalog_path = catalog_path_env.strip()
        else:
            test_catalog_path = ""
            default_catalog_path = (Path.cwd() / _DEFAULT_CATALOG_RELATIVE_PATH)
            if default_catalog_path.exists():
                test_catalog_path = str(default_catalog_path)

        return cls(
            llm_base_url=llm_base_url,
            api_key=api_key,
            model_id=model_id,
            autonomy_policy=autonomy,
            log_level=log_level,
            test_catalog_path=test_catalog_path,
        )

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        if not self.llm_base_url:
            errors.append("DATAIKU_LLM_MESH_URL is required")
        if not self.api_key:
            errors.append("DATAIKU_API_KEY is required")
        if not self.model_id:
            errors.append("DATAIKU_MODEL_ID is required")
        return errors
