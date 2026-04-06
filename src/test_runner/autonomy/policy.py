"""Configurable autonomy policy for the engine.

The policy defines thresholds, budgets, and exploration limits that the
autonomy engine uses to decide when to proceed, explore further, or
escalate. The architecture is pluggable — different policies can be
loaded from config or injected at runtime for different autonomy modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AutonomyPolicyConfig:
    """Configurable policy controlling the autonomy engine's behaviour.

    Attributes:
        execute_threshold: Confidence at or above which the engine proceeds
            autonomously (default 0.90).
        warn_threshold: Confidence at or above which the engine proceeds
            with a warning/confirmation request (default 0.60).
        max_exploration_rounds: Hard cap on how many exploration rounds
            the engine will attempt before forcing escalation (default 5).
        max_signals_per_round: Max new signals collected per round to
            bound resource usage (default 50).
        min_positive_signals: Minimum number of positive (score > 0)
            signals required before the engine considers proceeding.
        require_framework_detection: If True, at least one known framework
            must be detected before proceeding autonomously.
        allow_script_fallback: If True, the engine can fall back to
            script-based execution when no framework is detected.
        exploration_strategies: Ordered list of strategy names to try
            during further exploration (e.g. deeper scans, help probing).
    """

    execute_threshold: float = 0.90
    warn_threshold: float = 0.60
    max_exploration_rounds: int = 5
    max_signals_per_round: int = 50
    min_positive_signals: int = 2
    require_framework_detection: bool = False
    allow_script_fallback: bool = True
    exploration_strategies: tuple[str, ...] = field(
        default_factory=lambda: (
            "deep_pattern_scan",
            "config_file_inspection",
            "help_probe",
            "makefile_target_scan",
        )
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.warn_threshold <= self.execute_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 <= warn ({self.warn_threshold}) "
                f"<= execute ({self.execute_threshold}) <= 1"
            )
        if self.max_exploration_rounds < 1:
            raise ValueError("max_exploration_rounds must be >= 1")

    @classmethod
    def conservative(cls) -> AutonomyPolicyConfig:
        """Pre-built conservative policy — high bar, few rounds."""
        return cls(
            execute_threshold=0.95,
            warn_threshold=0.75,
            max_exploration_rounds=3,
            require_framework_detection=True,
        )

    @classmethod
    def moderate(cls) -> AutonomyPolicyConfig:
        """Pre-built moderate policy — default settings."""
        return cls()

    @classmethod
    def aggressive(cls) -> AutonomyPolicyConfig:
        """Pre-built aggressive policy — low bar, more exploration budget."""
        return cls(
            execute_threshold=0.70,
            warn_threshold=0.40,
            max_exploration_rounds=8,
            min_positive_signals=1,
            require_framework_detection=False,
        )
