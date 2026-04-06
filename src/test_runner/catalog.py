"""Machine-readable catalog for closed-world test execution.

The catalog is the authoritative source of runnable test definitions when
closed-world mode is enabled. Requests are matched deterministically against
saved aliases and keywords; unknown or ambiguous requests do not synthesize
new commands.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from test_runner.agents.parser import ParsedTestRequest, TestFramework, TestIntent
from test_runner.execution.command_translator import TestCommand, TranslationResult

logger = logging.getLogger(__name__)


class CatalogExecutionType(str, Enum):
    """Execution forms currently supported by the catalog."""

    PYTHON_SCRIPT = "python_script"
    EXECUTABLE = "executable"


class CatalogSystemTransport(str, Enum):
    """Execution transports supported by the catalog."""

    LOCAL = "local"
    SSH = "ssh"


class CatalogSystem(BaseModel):
    """One saved execution system definition."""

    alias: str = Field(min_length=1)
    description: str = ""
    transport: CatalogSystemTransport = CatalogSystemTransport.LOCAL
    hostname: str = ""
    username: str = ""
    port: int | None = Field(default=None, ge=1, le=65535)
    ssh_config_host: str = ""
    working_directory: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    credential_ref: str = ""
    enabled: bool = True

    @field_validator("alias")
    @classmethod
    def _strip_alias(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("hostname", "username", "ssh_config_host", "credential_ref")
    @classmethod
    def _strip_optional_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _validate_ssh_requirements(self) -> "CatalogSystem":
        if self.transport == CatalogSystemTransport.SSH:
            if not (self.hostname or self.ssh_config_host):
                raise ValueError(
                    "ssh systems require either hostname or ssh_config_host",
                )
        return self


class CatalogEntry(BaseModel):
    """One saved runnable test definition."""

    alias: str = Field(min_length=1)
    description: str = ""
    execution_type: CatalogExecutionType
    target: str = Field(min_length=1)
    system: str = "local"
    args: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    working_directory: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    timeout: int | None = Field(default=None, ge=1)
    enabled: bool = True

    @field_validator("alias", "target")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("system")
    @classmethod
    def _normalize_system(cls, value: str) -> str:
        stripped = value.strip()
        return stripped or "local"

    @field_validator("keywords", mode="after")
    @classmethod
    def _normalize_keywords(cls, value: list[str]) -> list[str]:
        return [keyword.strip() for keyword in value if keyword.strip()]


class CatalogDocument(BaseModel):
    """Top-level on-disk catalog format."""

    version: int = 1
    systems: list[CatalogSystem] = Field(default_factory=list)
    entries: list[CatalogEntry] = Field(default_factory=list)


class CatalogMatchStatus(str, Enum):
    """Deterministic outcomes for matching a request against the catalog."""

    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


@dataclass(frozen=True)
class CatalogMatch:
    """Result of resolving a natural-language request against the catalog."""

    status: CatalogMatchStatus
    entries: tuple[CatalogEntry, ...] = ()
    matched_terms: tuple[str, ...] = ()
    message: str = ""

    @property
    def entry(self) -> CatalogEntry | None:
        if len(self.entries) == 1:
            return self.entries[0]
        return None


class CatalogRegistry:
    """Deterministic registry of approved runnable test definitions."""

    def __init__(
        self,
        entries: list[CatalogEntry],
        systems: list[CatalogSystem] | None = None,
    ) -> None:
        self._entries = [entry for entry in entries if entry.enabled]
        aliases: dict[str, CatalogEntry] = {}
        for entry in self._entries:
            alias_key = self._normalize_phrase(entry.alias)
            if alias_key in aliases:
                raise ValueError(
                    f"Duplicate catalog alias {entry.alias!r} "
                    f"for targets {aliases[alias_key].target!r} and {entry.target!r}"
                )
            aliases[alias_key] = entry
        self._aliases = aliases
        self._systems = self._build_system_index(systems or [])

    @classmethod
    def from_path(cls, path: str | Path) -> "CatalogRegistry":
        """Load a registry from a JSON catalog file."""
        catalog_path = Path(path)
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        document = CatalogDocument.model_validate(raw)
        logger.info(
            "Loaded test catalog from %s with %d enabled entry(ies) and %d system(s)",
            catalog_path,
            sum(1 for entry in document.entries if entry.enabled),
            sum(1 for system in document.systems if system.enabled),
        )
        return cls(document.entries, systems=document.systems)

    @property
    def entries(self) -> tuple[CatalogEntry, ...]:
        """Return all enabled catalog entries."""
        return tuple(self._entries)

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return the enabled aliases in sorted order."""
        return tuple(sorted(entry.alias for entry in self._entries))

    @property
    def systems(self) -> tuple[CatalogSystem, ...]:
        """Return all enabled execution systems."""
        return tuple(
            sorted(
                self._systems.values(),
                key=lambda system: system.alias.lower(),
            )
        )

    def match_request(self, request: str) -> CatalogMatch:
        """Match a request against saved aliases and keywords.

        Matching is deterministic and closed-world:
        1. Exact alias phrase matches win.
        2. If no alias matches, keyword phrase matches are considered.
        3. Multiple matches require clarification.
        4. No match means the request is not runnable.
        """
        normalized_request = self._normalize_phrase(request)
        if not normalized_request:
            return CatalogMatch(
                status=CatalogMatchStatus.MISSING,
                message="Request did not include a cataloged test alias.",
            )

        alias_matches = [
            entry for entry in self._entries
            if self._contains_phrase(normalized_request, self._normalize_phrase(entry.alias))
        ]
        if alias_matches:
            return self._build_match_result(
                alias_matches,
                match_kind="alias",
            )

        keyword_hits: list[tuple[CatalogEntry, str]] = []
        for entry in self._entries:
            for keyword in entry.keywords:
                normalized_keyword = self._normalize_phrase(keyword)
                if normalized_keyword and self._contains_phrase(
                    normalized_request, normalized_keyword,
                ):
                    keyword_hits.append((entry, keyword))
                    break

        if keyword_hits:
            return self._build_match_result(
                [entry for entry, _ in keyword_hits],
                match_kind="keyword",
                matched_terms=[term for _, term in keyword_hits],
            )

        aliases = ", ".join(self.aliases) or "(none configured)"
        return CatalogMatch(
            status=CatalogMatchStatus.MISSING,
            message=(
                "Request did not match any cataloged test definition. "
                f"Known aliases: {aliases}."
            ),
        )

    def translate_match(
        self,
        match: CatalogMatch,
        request: ParsedTestRequest,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> TranslationResult:
        """Build runnable commands for a catalog match."""
        if match.status != CatalogMatchStatus.MATCHED or match.entry is None:
            return TranslationResult(
                commands=[],
                warnings=[match.message] if match.message else [],
                source_request=request,
            )

        entry = match.entry
        warnings: list[str] = []

        if request.intent in {TestIntent.LIST, TestIntent.RERUN_FAILED}:
            warnings.append(
                f"Catalog entry {entry.alias!r} matched, but intent "
                f"{request.intent.value!r} is not implemented for saved "
                "script/executable definitions yet."
            )
            return TranslationResult(
                commands=[],
                warnings=warnings,
                source_request=request,
            )

        if request.extra_args:
            warnings.append(
                "Ignoring ad hoc extra arguments because catalog mode only "
                "runs saved definitions."
            )

        system = self._resolve_system(entry)
        if system is None:
            warnings.append(
                f"Catalog entry {entry.alias!r} references unknown system "
                f"{entry.system!r}.",
            )
            return TranslationResult(
                commands=[],
                warnings=warnings,
                source_request=request,
            )

        command_tokens = self._build_command_tokens(entry)
        merged_env = dict(system.env)
        merged_env.update(entry.env)
        if env and system.transport == CatalogSystemTransport.LOCAL:
            merged_env.update(env)

        command = TestCommand(
            command=command_tokens,
            display=" ".join(command_tokens),
            framework=TestFramework.SCRIPT,
            working_directory=entry.working_directory or system.working_directory,
            env=merged_env,
            timeout=timeout if timeout is not None else entry.timeout,
            metadata={
                "catalog_alias": entry.alias,
                "catalog_description": entry.description,
                "catalog_execution_type": entry.execution_type.value,
                "catalog_system": system.alias,
                "catalog_system_transport": system.transport.value,
                "catalog_system_config": system.model_dump(mode="python"),
                "intent": request.intent.value,
                "scope": request.scope,
                "confidence": request.confidence,
            },
        )

        return TranslationResult(
            commands=[command],
            warnings=warnings,
            source_request=request,
        )

    def _build_match_result(
        self,
        entries: list[CatalogEntry],
        *,
        match_kind: str,
        matched_terms: list[str] | None = None,
    ) -> CatalogMatch:
        unique_entries = sorted(
            {entry.alias: entry for entry in entries}.values(),
            key=lambda entry: entry.alias.lower(),
        )
        if len(unique_entries) == 1:
            entry = unique_entries[0]
            return CatalogMatch(
                status=CatalogMatchStatus.MATCHED,
                entries=(entry,),
                matched_terms=tuple(matched_terms or [entry.alias]),
                message=(
                    f"Matched catalog entry {entry.alias!r} via {match_kind}."
                ),
            )

        aliases = ", ".join(entry.alias for entry in unique_entries)
        return CatalogMatch(
            status=CatalogMatchStatus.AMBIGUOUS,
            entries=tuple(unique_entries),
            matched_terms=tuple(matched_terms or [entry.alias for entry in unique_entries]),
            message=(
                "Request matched multiple catalog entries and needs clarification: "
                f"{aliases}."
            ),
        )

    @staticmethod
    def _build_command_tokens(entry: CatalogEntry) -> list[str]:
        if entry.execution_type == CatalogExecutionType.PYTHON_SCRIPT:
            return ["python", entry.target, *entry.args]
        return [entry.target, *entry.args]

    def _build_system_index(
        self,
        systems: list[CatalogSystem],
    ) -> dict[str, CatalogSystem]:
        resolved: dict[str, CatalogSystem] = {}
        for system in systems:
            if not system.enabled:
                continue
            system_key = self._normalize_phrase(system.alias)
            if system_key in resolved:
                raise ValueError(
                    f"Duplicate catalog system alias {system.alias!r}.",
                )
            resolved[system_key] = system

        local_key = self._normalize_phrase("local")
        if local_key not in resolved:
            resolved[local_key] = CatalogSystem(
                alias="local",
                description="Current process host",
                transport=CatalogSystemTransport.LOCAL,
            )

        return resolved

    def _resolve_system(self, entry: CatalogEntry) -> CatalogSystem | None:
        return self._systems.get(self._normalize_phrase(entry.system))

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _contains_phrase(normalized_request: str, normalized_phrase: str) -> bool:
        if not normalized_phrase:
            return False
        return f" {normalized_phrase} " in f" {normalized_request} "


class CatalogRepository:
    """Persistence helper for the JSON-backed catalog document."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_document(self) -> CatalogDocument:
        """Load the current catalog document, or return an empty one."""
        if not self.path.exists():
            return CatalogDocument()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return CatalogDocument.model_validate(raw)

    def save_document(self, document: CatalogDocument) -> CatalogDocument:
        """Validate and persist a catalog document."""
        validated = CatalogDocument.model_validate(document.model_dump(mode="python"))
        CatalogRegistry(validated.entries, systems=validated.systems)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            validated.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return validated

    def has_entry_alias(self, alias: str) -> bool:
        """Return True when the alias already exists."""
        normalized = CatalogRegistry._normalize_phrase(alias)
        document = self.load_document()
        return any(
            CatalogRegistry._normalize_phrase(entry.alias) == normalized
            for entry in document.entries
        )

    def get_system(self, alias: str) -> CatalogSystem | None:
        """Return the saved system with the given alias, if any."""
        normalized = CatalogRegistry._normalize_phrase(alias)
        document = self.load_document()
        for system in document.systems:
            if CatalogRegistry._normalize_phrase(system.alias) == normalized:
                return system
        if normalized == CatalogRegistry._normalize_phrase("local"):
            return CatalogSystem(
                alias="local",
                description="Current process host",
                transport=CatalogSystemTransport.LOCAL,
            )
        return None

    def list_entries(self) -> tuple[CatalogEntry, ...]:
        """Return saved entries sorted by alias."""
        document = self.load_document()
        return tuple(
            sorted(
                (entry for entry in document.entries if entry.enabled),
                key=lambda entry: entry.alias.lower(),
            )
        )

    def get_entry(self, alias: str) -> CatalogEntry | None:
        """Return the saved entry with the given alias, if any."""
        normalized = CatalogRegistry._normalize_phrase(alias)
        document = self.load_document()
        for entry in document.entries:
            if CatalogRegistry._normalize_phrase(entry.alias) == normalized:
                return entry
        return None

    def add_system(self, system: CatalogSystem) -> CatalogDocument:
        """Persist a new system definition."""
        document = self.load_document()
        if self.get_system(system.alias) is not None:
            raise ValueError(f"Catalog system alias {system.alias!r} already exists.")
        updated = CatalogDocument(
            version=document.version,
            systems=[*document.systems, system],
            entries=document.entries,
        )
        return self.save_document(updated)

    def add_entry(self, entry: CatalogEntry) -> CatalogDocument:
        """Persist a new runnable entry."""
        document = self.load_document()
        if self.has_entry_alias(entry.alias):
            raise ValueError(f"Catalog alias {entry.alias!r} already exists.")
        if self.get_system(entry.system) is None:
            raise ValueError(
                f"Catalog entry {entry.alias!r} references unknown system "
                f"{entry.system!r}.",
            )
        updated = CatalogDocument(
            version=document.version,
            systems=document.systems,
            entries=[*document.entries, entry],
        )
        return self.save_document(updated)

    def update_entry(
        self,
        existing_alias: str,
        updated_entry: CatalogEntry,
    ) -> CatalogDocument:
        """Replace an existing entry with an updated definition."""
        document = self.load_document()
        existing_key = CatalogRegistry._normalize_phrase(existing_alias)
        updated_key = CatalogRegistry._normalize_phrase(updated_entry.alias)
        replacement_index: int | None = None

        for index, entry in enumerate(document.entries):
            entry_key = CatalogRegistry._normalize_phrase(entry.alias)
            if entry_key == existing_key:
                replacement_index = index
                continue
            if entry_key == updated_key:
                raise ValueError(
                    f"Catalog alias {updated_entry.alias!r} already exists.",
                )

        if replacement_index is None:
            raise ValueError(f"Catalog alias {existing_alias!r} does not exist.")

        if self.get_system(updated_entry.system) is None:
            raise ValueError(
                f"Catalog entry {updated_entry.alias!r} references unknown system "
                f"{updated_entry.system!r}.",
            )

        updated_entries = list(document.entries)
        updated_entries[replacement_index] = updated_entry
        updated_document = CatalogDocument(
            version=document.version,
            systems=document.systems,
            entries=updated_entries,
        )
        return self.save_document(updated_document)

    def delete_entry(self, alias: str) -> CatalogEntry | None:
        """Delete one entry by alias and return it."""
        document = self.load_document()
        normalized = CatalogRegistry._normalize_phrase(alias)
        kept_entries: list[CatalogEntry] = []
        deleted_entry: CatalogEntry | None = None

        for entry in document.entries:
            if CatalogRegistry._normalize_phrase(entry.alias) == normalized:
                deleted_entry = entry
                continue
            kept_entries.append(entry)

        if deleted_entry is None:
            return None

        updated_document = CatalogDocument(
            version=document.version,
            systems=document.systems,
            entries=kept_entries,
        )
        self.save_document(updated_document)
        return deleted_entry
