"""Natural language parser for test requests.

Uses the OpenAI Agents SDK (via Dataiku LLM Mesh endpoint) to interpret
natural language test requests and extract structured intent, framework,
and scope information.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings

from test_runner.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class TestIntent(str, Enum):
    """The user's high-level intent."""

    RUN = "run"                   # Execute tests
    LIST = "list"                 # Discover / enumerate tests without running
    RERUN_FAILED = "rerun_failed" # Re-run only previously failed tests
    RUN_SPECIFIC = "run_specific" # Run a specific named test / file
    UNKNOWN = "unknown"


class TestFramework(str, Enum):
    """Known test frameworks the runner supports."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    GO_TEST = "go_test"
    CARGO_TEST = "cargo_test"
    DOTNET_TEST = "dotnet_test"
    SCRIPT = "script"            # Arbitrary script / command
    AUTO_DETECT = "auto_detect"  # Let discovery agent decide
    UNKNOWN = "unknown"


class ParsedTestRequest(BaseModel):
    """Structured representation of a parsed natural language test request."""

    intent: TestIntent = Field(
        description="The user's high-level intent (run, list, rerun_failed, etc.)",
    )
    framework: TestFramework = Field(
        description=(
            "Detected or specified test framework. Use auto_detect when the "
            "user doesn't specify a framework explicitly."
        ),
    )
    scope: str = Field(
        default="",
        description=(
            "The scope / filter for tests — e.g. a directory path, file glob, "
            "test name pattern, or marker/tag expression. Empty string means "
            "'all tests'."
        ),
    )
    working_directory: str = Field(
        default="",
        description=(
            "Explicit working directory if the user specified one, otherwise empty."
        ),
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description=(
            "Any extra CLI flags or arguments the user wants passed through "
            "to the test command (e.g. --verbose, -k expression)."
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "The model's confidence in its parsing (0.0-1.0). Low confidence "
            "signals the orchestrator should ask for clarification."
        ),
    )
    raw_request: str = Field(
        default="",
        description="The original natural language request text.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of how the request was interpreted.",
    )


# ---------------------------------------------------------------------------
# System prompt for the parser agent
# ---------------------------------------------------------------------------

_PARSER_SYSTEM_PROMPT = """\
You are a test-request parser. Given a natural language request from a \
developer, extract structured information about what tests they want to run.

Rules:
1. Identify the INTENT: run tests, list/discover tests, rerun failed tests, \
or run a specific test.
2. Identify the FRAMEWORK if mentioned (pytest, unittest, jest, mocha, go test, \
cargo test, dotnet test). If they mention running a script or command, use "script". \
If they don't specify, use "auto_detect".
3. Identify the SCOPE: file paths, directories, glob patterns, test name patterns, \
markers/tags. If they want all tests, leave scope empty.
4. Extract any EXTRA ARGS the user wants passed through (e.g., --verbose, -x, \
--coverage).
5. If the user mentions a specific directory to work in, set working_directory.
6. Set confidence between 0.0 and 1.0 based on how clear the request is. \
Use lower confidence when the request is ambiguous.
7. Provide brief reasoning explaining your interpretation.

Always respond with the structured output format. Never refuse a request — \
if it's unclear, set confidence low and do your best interpretation.
"""


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------


class NaturalLanguageParser:
    """Parses natural language test requests into structured ParsedTestRequest.

    Uses an OpenAI Agents SDK Agent backed by the Dataiku LLM Mesh to
    perform the interpretation.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = self._build_client()
        self._agent = self._build_agent()

    def _build_client(self) -> AsyncOpenAI:
        """Create the OpenAI client pointing at the Dataiku LLM Mesh."""
        return AsyncOpenAI(
            base_url=self._config.llm_base_url,
            api_key=self._config.api_key,
        )

    def _build_agent(self) -> Agent:
        """Create the parser agent with structured output."""
        return Agent(
            name="test-request-parser",
            instructions=_PARSER_SYSTEM_PROMPT,
            model=self._config.model_id,
            model_settings=ModelSettings(temperature=0.0),
            output_type=ParsedTestRequest,
        )

    async def parse(self, request: str) -> ParsedTestRequest:
        """Parse a natural language test request.

        Args:
            request: Free-form text describing what the user wants to test.

        Returns:
            A structured ParsedTestRequest with intent, framework, scope, etc.

        Raises:
            ParserError: If the LLM call fails or returns unparseable output.
        """
        logger.info("Parsing request: %s", request)

        try:
            result = await Runner.run(
                self._agent,
                input=request,
                run_config=self._make_run_config(),
            )
            parsed: ParsedTestRequest = result.final_output
            # Always stamp the raw request
            parsed = parsed.model_copy(update={"raw_request": request})
            logger.info(
                "Parsed → intent=%s framework=%s scope=%r confidence=%.2f",
                parsed.intent.value,
                parsed.framework.value,
                parsed.scope,
                parsed.confidence,
            )
            return parsed

        except Exception as exc:
            logger.error("Parser failed: %s", exc)
            raise ParserError(str(exc)) from exc

    def _make_run_config(self):
        """Build a RunConfig pointing the SDK at the Dataiku mesh endpoint."""
        from agents import RunConfig
        from agents.models.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            openai_client=self._client,
        )
        return RunConfig(model_provider=provider)

    @staticmethod
    def parse_offline(request: str) -> ParsedTestRequest:
        """Best-effort heuristic parse without calling the LLM.

        Useful as a fallback when the LLM endpoint is unavailable, or
        for fast-path obvious commands.

        Args:
            request: Free-form text.

        Returns:
            A ParsedTestRequest based on keyword heuristics.
        """
        lower = request.lower().strip()

        # --- Intent detection ---
        intent = TestIntent.RUN
        if any(kw in lower for kw in ("list", "discover", "find", "enumerate", "show")):
            intent = TestIntent.LIST
        elif any(kw in lower for kw in ("rerun", "re-run", "retry", "failed")):
            intent = TestIntent.RERUN_FAILED

        # --- Framework detection ---
        # Check named frameworks first (order matters — longer matches first)
        framework = TestFramework.AUTO_DETECT
        framework_keywords = [
            ("cargo test", TestFramework.CARGO_TEST),
            ("dotnet test", TestFramework.DOTNET_TEST),
            ("go test", TestFramework.GO_TEST),
            ("pytest", TestFramework.PYTEST),
            ("py.test", TestFramework.PYTEST),
            ("unittest", TestFramework.UNITTEST),
            ("jest", TestFramework.JEST),
            ("mocha", TestFramework.MOCHA),
        ]
        for keyword, fw in framework_keywords:
            if keyword in lower:
                framework = fw
                break

        # Script detection — only if no named framework was found
        if framework == TestFramework.AUTO_DETECT and any(
            kw in lower for kw in ("script", "bash", "sh ", "./", "python ")
        ):
            framework = TestFramework.SCRIPT

        # --- Scope extraction (simple heuristic) ---
        scope = ""
        # Look for path-like tokens
        tokens = request.split()
        for token in tokens:
            if "/" in token or token.endswith(".py") or token.endswith(".js") or token.endswith(".ts"):
                scope = token
                break
            if token.startswith("test_") or token.startswith("Test"):
                scope = token
                break

        # --- Confidence ---
        confidence = 0.6  # heuristic parse is inherently lower confidence

        return ParsedTestRequest(
            intent=intent,
            framework=framework,
            scope=scope,
            confidence=confidence,
            raw_request=request,
            reasoning="Parsed via offline keyword heuristics (no LLM call).",
        )


class ParserError(Exception):
    """Raised when the natural language parser fails."""
