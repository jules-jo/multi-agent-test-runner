"""Specialized sub-agents for the test runner."""

from test_runner.agents.base import AgentRole, AgentState, BaseSubAgent
from test_runner.agents.parser import (
    NaturalLanguageParser,
    ParsedTestRequest,
    ParserError,
    TestFramework,
    TestIntent,
)

# Intent service imports are deferred to break the circular dependency:
#   agents.__init__ → intent_service → execution.command_translator
#   → agents.parser → agents.__init__  (cycle)
try:
    from test_runner.agents.intent_service import (
        IntentParserService,
        IntentResolution,
        IntentResolutionError,
        ParseMode,
    )
except ImportError:
    IntentParserService = None  # type: ignore[assignment,misc]
    IntentResolution = None  # type: ignore[assignment,misc]
    IntentResolutionError = None  # type: ignore[assignment,misc]
    ParseMode = None  # type: ignore[assignment,misc]

# Discovery agent exports — imported conditionally since
# DiscoveryAgent may not be available yet during incremental builds.
try:
    from test_runner.agents.discovery import DiscoveryAgent, create_discovery_agent
except ImportError:
    DiscoveryAgent = None  # type: ignore[assignment,misc]
    create_discovery_agent = None  # type: ignore[assignment]

# Reporter agent exports
try:
    from test_runner.agents.reporter import ReporterAgent, create_reporter_agent
except ImportError:
    ReporterAgent = None  # type: ignore[assignment,misc]
    create_reporter_agent = None  # type: ignore[assignment]

# Troubleshooter agent exports
try:
    from test_runner.agents.troubleshooter import (
        TroubleshooterAgent,
        create_troubleshooter_agent,
    )
except ImportError:
    TroubleshooterAgent = None  # type: ignore[assignment,misc]
    create_troubleshooter_agent = None  # type: ignore[assignment]

__all__ = [
    "AgentRole",
    "AgentState",
    "BaseSubAgent",
    "DiscoveryAgent",
    "IntentParserService",
    "IntentResolution",
    "IntentResolutionError",
    "NaturalLanguageParser",
    "ParsedTestRequest",
    "ParseMode",
    "ParserError",
    "ReporterAgent",
    "TestFramework",
    "TestIntent",
    "create_discovery_agent",
    "create_reporter_agent",
    "TroubleshooterAgent",
    "create_troubleshooter_agent",
]
