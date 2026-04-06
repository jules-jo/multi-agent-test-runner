"""Intent parser service — cohesive pipeline from natural language to test commands.

Combines the NaturalLanguageParser (LLM-backed or offline heuristic) with the
CommandTranslator to provide a single entry point:

    request (str) → ParsedTestRequest → TranslationResult (executable commands)

The orchestrator hub delegates here during its init/discovery phase. The service
decides whether to call the LLM or fall back to offline heuristics based on
config availability and caller preference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from test_runner.agents.parser import (
    NaturalLanguageParser,
    ParsedTestRequest,
    ParserError,
    TestFramework,
    TestIntent,
)
from test_runner.catalog import CatalogRegistry
from test_runner.config import Config
from test_runner.execution.command_translator import (
    CommandTranslator,
    TranslationResult,
    UnsupportedFrameworkError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parse mode
# ---------------------------------------------------------------------------


class ParseMode(str, Enum):
    """How the intent parser resolves intent."""

    LLM = "llm"            # Use the LLM-backed parser (Dataiku Mesh)
    OFFLINE = "offline"      # Use keyword heuristics only
    AUTO = "auto"            # Try LLM first, fall back to offline on failure


# ---------------------------------------------------------------------------
# Intent resolution result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentResolution:
    """Full result of resolving a natural language request to executable commands.

    Attributes:
        parsed_request: The structured representation extracted from the NL input.
        translation: The translation result containing executable TestCommands.
        parse_mode_used: Which parse mode was actually used (LLM vs offline).
        warnings: Aggregated warnings from both parsing and translation.
        needs_clarification: True if confidence is too low for autonomous action.
    """

    parsed_request: ParsedTestRequest
    translation: TranslationResult
    parse_mode_used: ParseMode
    warnings: list[str] = field(default_factory=list)
    needs_clarification: bool = False

    @property
    def confidence(self) -> float:
        """Shortcut to the parser's confidence score."""
        return self.parsed_request.confidence

    @property
    def commands(self) -> list:
        """Shortcut to the translated commands."""
        return self.translation.commands

    @property
    def framework(self) -> TestFramework:
        """The resolved framework."""
        return self.parsed_request.framework

    @property
    def intent(self) -> TestIntent:
        """The resolved intent."""
        return self.parsed_request.intent

    def summary(self) -> dict[str, Any]:
        """Return a dict summary suitable for orchestrator state."""
        return {
            "intent": self.parsed_request.intent.value,
            "framework": self.parsed_request.framework.value,
            "scope": self.parsed_request.scope,
            "confidence": self.parsed_request.confidence,
            "parse_mode": self.parse_mode_used.value,
            "commands": [c.display for c in self.translation.commands],
            "warnings": self.warnings,
            "needs_clarification": self.needs_clarification,
            "reasoning": self.parsed_request.reasoning,
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


# Default confidence threshold below which we flag needs_clarification
_DEFAULT_CLARIFICATION_THRESHOLD = 0.4


class IntentParserService:
    """End-to-end service: natural language → structured intent → test commands.

    This is the single integration point the orchestrator calls. It:
    1. Parses the natural language request (via LLM or offline heuristics).
    2. Translates the parsed intent into concrete CLI commands.
    3. Returns an ``IntentResolution`` with commands, confidence, and warnings.

    Parameters
    ----------
    config : Config
        Application configuration (LLM connection, autonomy policy, etc.).
    parse_mode : ParseMode
        Which parsing strategy to use. ``AUTO`` tries LLM first, falls back
        to offline on failure.
    clarification_threshold : float
        Confidence below this triggers ``needs_clarification=True``.
    command_translator : CommandTranslator | None
        Optional pre-configured translator. A default is created if None.
    """

    def __init__(
        self,
        config: Config,
        *,
        parse_mode: ParseMode = ParseMode.AUTO,
        clarification_threshold: float = _DEFAULT_CLARIFICATION_THRESHOLD,
        command_translator: CommandTranslator | None = None,
        catalog_registry: CatalogRegistry | None = None,
    ) -> None:
        self._config = config
        self._parse_mode = parse_mode
        self._clarification_threshold = clarification_threshold
        self._translator = command_translator or CommandTranslator()
        self._catalog_registry = catalog_registry or self._load_catalog_registry()

        # Only build the LLM-backed parser if we might need it
        self._llm_parser: NaturalLanguageParser | None = None
        if parse_mode in (ParseMode.LLM, ParseMode.AUTO):
            if self._config_has_llm():
                self._llm_parser = NaturalLanguageParser(config)
            else:
                logger.warning(
                    "LLM config incomplete — LLM parser unavailable, "
                    "will use offline heuristics."
                )

    # -- Public API ----------------------------------------------------------

    async def resolve(
        self,
        request: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> IntentResolution:
        """Resolve a natural language test request into executable commands.

        Parameters
        ----------
        request : str
            Free-form natural language description of the desired test action.
        timeout : int | None
            Optional per-command timeout override (seconds).
        env : dict[str, str] | None
            Optional extra environment variables to inject into commands.

        Returns
        -------
        IntentResolution
            Complete resolution including parsed request, commands, and metadata.

        Raises
        ------
        IntentResolutionError
            If both LLM and offline parsing fail, or translation is impossible.
        """
        warnings: list[str] = []
        parsed: ParsedTestRequest
        mode_used: ParseMode

        # --- Step 1: Parse ---
        parsed, mode_used, parse_warnings = await self._parse(request)
        warnings.extend(parse_warnings)

        # --- Step 2: Translate ---
        translation, translation_needs_clarification = self._translate(
            parsed,
            timeout=timeout,
            env=env,
        )
        warnings.extend(translation.warnings)

        # --- Step 3: Assess confidence ---
        needs_clarification = translation_needs_clarification or (
            self._catalog_registry is None
            and parsed.confidence < self._clarification_threshold
        )

        if needs_clarification:
            logger.info(
                "Low confidence (%.2f < %.2f) — flagging for clarification",
                parsed.confidence,
                self._clarification_threshold,
            )

        resolution = IntentResolution(
            parsed_request=parsed,
            translation=translation,
            parse_mode_used=mode_used,
            warnings=warnings,
            needs_clarification=needs_clarification,
        )

        logger.info(
            "Intent resolved: %s %s scope=%r confidence=%.2f mode=%s cmds=%d",
            resolution.intent.value,
            resolution.framework.value,
            parsed.scope,
            parsed.confidence,
            mode_used.value,
            len(resolution.commands),
        )

        return resolution

    def resolve_offline(
        self,
        request: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> IntentResolution:
        """Synchronous offline-only resolution (no LLM call).

        Useful for dry-run mode, fast-path obvious commands, or when
        the LLM endpoint is known to be unavailable.
        """
        parsed = NaturalLanguageParser.parse_offline(request)
        warnings: list[str] = []

        translation, translation_needs_clarification = self._translate(
            parsed,
            timeout=timeout,
            env=env,
        )
        warnings.extend(translation.warnings)

        needs_clarification = translation_needs_clarification or (
            self._catalog_registry is None
            and parsed.confidence < self._clarification_threshold
        )

        return IntentResolution(
            parsed_request=parsed,
            translation=translation,
            parse_mode_used=ParseMode.OFFLINE,
            warnings=warnings,
            needs_clarification=needs_clarification,
        )

    @property
    def parse_mode(self) -> ParseMode:
        """Current parse mode."""
        return self._parse_mode

    @property
    def translator(self) -> CommandTranslator:
        """The underlying command translator."""
        return self._translator

    @property
    def clarification_threshold(self) -> float:
        """Confidence threshold below which clarification is flagged."""
        return self._clarification_threshold

    @property
    def catalog_registry(self) -> CatalogRegistry | None:
        """Loaded catalog registry, if closed-world mode is enabled."""
        return self._catalog_registry

    # -- Internal ------------------------------------------------------------

    async def _parse(
        self, request: str
    ) -> tuple[ParsedTestRequest, ParseMode, list[str]]:
        """Parse the request using the configured strategy.

        Returns (parsed_request, mode_used, warnings).
        """
        warnings: list[str] = []

        if self._parse_mode == ParseMode.OFFLINE:
            return (
                NaturalLanguageParser.parse_offline(request),
                ParseMode.OFFLINE,
                warnings,
            )

        if self._parse_mode == ParseMode.LLM:
            if self._llm_parser is None:
                raise IntentResolutionError(
                    "LLM parse mode requested but LLM config is incomplete. "
                    "Set DATAIKU_LLM_MESH_URL, DATAIKU_API_KEY, and DATAIKU_MODEL_ID.",
                )
            return (
                await self._llm_parser.parse(request),
                ParseMode.LLM,
                warnings,
            )

        # AUTO mode: try LLM, fall back to offline
        if self._llm_parser is not None:
            try:
                parsed = await self._llm_parser.parse(request)
                return parsed, ParseMode.LLM, warnings
            except ParserError as exc:
                logger.warning("LLM parser failed, falling back to offline: %s", exc)
                warnings.append(f"LLM parser failed ({exc}), used offline heuristics.")

        # Offline fallback
        parsed = NaturalLanguageParser.parse_offline(request)
        return parsed, ParseMode.OFFLINE, warnings

    def _config_has_llm(self) -> bool:
        """Check if config has all required LLM connection fields."""
        return bool(
            self._config.llm_base_url
            and self._config.api_key
            and self._config.model_id
        )

    def _load_catalog_registry(self) -> CatalogRegistry | None:
        """Load the configured test catalog, if any."""
        if not self._config.test_catalog_path:
            return None
        return CatalogRegistry.from_path(self._config.test_catalog_path)

    def _translate(
        self,
        parsed: ParsedTestRequest,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[TranslationResult, bool]:
        """Translate a parsed request into commands.

        Returns ``(translation, needs_clarification)``.
        """
        if self._catalog_registry is not None:
            match = self._catalog_registry.match_request(parsed.raw_request)
            translation = self._catalog_registry.translate_match(
                match,
                parsed,
                timeout=timeout,
                env=env,
            )
            if (
                not translation.commands
                and self._config.test_catalog_path
                and not any(self._config.test_catalog_path in warning for warning in translation.warnings)
            ):
                translation.warnings.append(
                    "Update saved test definitions in "
                    f"{self._config.test_catalog_path}."
                )
            return translation, not bool(translation.commands)

        try:
            translation = self._translator.translate(
                parsed, timeout=timeout, env=env,
            )
        except UnsupportedFrameworkError as exc:
            raise IntentResolutionError(
                f"Cannot translate: {exc}",
                parsed_request=parsed,
            ) from exc
        return translation, False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IntentResolutionError(Exception):
    """Raised when intent resolution fails completely.

    Attributes:
        parsed_request: The partially parsed request, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        parsed_request: ParsedTestRequest | None = None,
    ) -> None:
        super().__init__(message)
        self.parsed_request = parsed_request
