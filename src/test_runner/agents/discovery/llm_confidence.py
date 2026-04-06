"""LLM self-assessment signal collector.

Prompts the LLM to rate its own confidence about a discovery context
and returns a normalized score as a ConfidenceSignal.

This collector differs from file-based collectors: instead of scanning
the filesystem directly, it sends a structured prompt to the LLM asking
it to evaluate the evidence gathered so far and produce a confidence
rating. The response is parsed and clamped to [0.0, 1.0].
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from test_runner.models.confidence import ConfidenceSignal

logger = logging.getLogger(__name__)

# Default weight for the LLM self-assessment signal.
# Kept moderate so it doesn't dominate file-based evidence.
DEFAULT_LLM_CONFIDENCE_WEIGHT = 0.6

_ASSESSMENT_SYSTEM_PROMPT = """\
You are a confidence evaluator for a test discovery system.

Given a summary of evidence about a software project's test setup,
rate your confidence that the project has a runnable test suite.

You MUST respond with ONLY a JSON object in this exact format:
{
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}

Scoring guidelines:
- 0.9-1.0: Clear framework config + test files + runnable commands found
- 0.7-0.9: Strong indicators present but minor ambiguity
- 0.5-0.7: Some test artifacts found but unclear how to run them
- 0.3-0.5: Weak signals, might be test-like files but no framework
- 0.0-0.3: No meaningful test evidence found

Do NOT include any text outside the JSON object.
"""


@dataclass
class DiscoveryContext:
    """Evidence gathered during discovery, to be assessed by the LLM.

    Attributes:
        project_path: Root directory being scanned.
        files_found: List of test-related file paths discovered.
        frameworks_detected: Framework names found (e.g. ["pytest", "jest"]).
        config_snippets: Relevant excerpts from config files.
        raw_signals: Summary of signals from other collectors.
        extra: Any additional context for the LLM.
    """

    project_path: str = ""
    files_found: list[str] = field(default_factory=list)
    frameworks_detected: list[str] = field(default_factory=list)
    config_snippets: dict[str, str] = field(default_factory=dict)
    raw_signals: list[dict[str, Any]] = field(default_factory=list)
    extra: str = ""

    def to_prompt(self) -> str:
        """Serialize the context into a human-readable prompt string."""
        parts: list[str] = []

        if self.project_path:
            parts.append(f"Project root: {self.project_path}")

        if self.frameworks_detected:
            parts.append(
                f"Frameworks detected: {', '.join(self.frameworks_detected)}"
            )

        if self.files_found:
            sample = self.files_found[:20]
            parts.append(
                f"Test-related files ({len(self.files_found)} total, "
                f"showing up to 20):\n"
                + "\n".join(f"  - {f}" for f in sample)
            )

        if self.config_snippets:
            parts.append("Configuration excerpts:")
            for name, snippet in self.config_snippets.items():
                # Truncate long snippets
                truncated = snippet[:500]
                if len(snippet) > 500:
                    truncated += "\n  ... (truncated)"
                parts.append(f"  [{name}]:\n  {truncated}")

        if self.raw_signals:
            non_zero = [s for s in self.raw_signals if s.get("score", 0) > 0]
            if non_zero:
                parts.append(
                    f"Positive signals from other collectors "
                    f"({len(non_zero)} of {len(self.raw_signals)}):"
                )
                for sig in non_zero[:10]:
                    parts.append(
                        f"  - {sig.get('name', '?')}: "
                        f"score={sig.get('score', 0):.2f}, "
                        f"weight={sig.get('weight', 0):.2f}"
                    )

        if self.extra:
            parts.append(f"Additional context: {self.extra}")

        if not parts:
            parts.append("No evidence has been gathered yet.")

        return "\n\n".join(parts)


def _parse_llm_response(raw: str) -> tuple[float, str]:
    """Extract confidence score and reasoning from LLM response.

    Attempts JSON parsing first, then falls back to regex extraction.
    Returns (score, reasoning) with score clamped to [0.0, 1.0].
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try JSON parse
    try:
        data = json.loads(cleaned)
        score = float(data.get("confidence", 0.5))
        reasoning = str(data.get("reasoning", ""))
        return max(0.0, min(1.0, score)), reasoning
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: regex for a float after "confidence"
    match = re.search(
        r'"?confidence"?\s*[:=]\s*([0-9]*\.?[0-9]+)', cleaned, re.IGNORECASE
    )
    if match:
        score = float(match.group(1))
        # Try to find reasoning too
        reason_match = re.search(
            r'"?reasoning"?\s*[:=]\s*"([^"]*)"', cleaned, re.IGNORECASE
        )
        reasoning = reason_match.group(1) if reason_match else "parsed via fallback"
        return max(0.0, min(1.0, score)), reasoning

    # Last resort: look for any float in the response
    float_match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", cleaned)
    if float_match:
        score = float(float_match.group(1))
        return max(0.0, min(1.0, score)), "extracted bare float from response"

    logger.warning("Could not parse LLM confidence response: %s", raw[:200])
    return 0.5, "unparseable response, defaulting to 0.5"


async def assess_confidence(
    client: AsyncOpenAI,
    context: DiscoveryContext,
    *,
    model: str = "gpt-4o",
    weight: float = DEFAULT_LLM_CONFIDENCE_WEIGHT,
    timeout: float = 30.0,
) -> ConfidenceSignal:
    """Prompt the LLM to self-assess confidence and return a ConfidenceSignal.

    Args:
        client: An AsyncOpenAI client (configured for Dataiku Mesh or direct).
        context: Discovery evidence to evaluate.
        model: Model ID to use for the assessment.
        weight: Weight of the returned signal in [0.0, 1.0].
        timeout: Request timeout in seconds.

    Returns:
        A ConfidenceSignal with the LLM's self-assessed score, normalized
        to [0.0, 1.0].
    """
    user_prompt = (
        "Evaluate the following discovery evidence and rate your confidence "
        "that this project has a runnable test suite.\n\n"
        + context.to_prompt()
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ASSESSMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temp for consistent scoring
            max_tokens=256,
            timeout=timeout,
        )

        raw_content = response.choices[0].message.content or ""
        score, reasoning = _parse_llm_response(raw_content)

        return ConfidenceSignal(
            name="llm_self_assessment",
            weight=weight,
            score=score,
            evidence={
                "reasoning": reasoning,
                "raw_response": raw_content[:500],
                "model": model,
                "context_summary": {
                    "files_count": len(context.files_found),
                    "frameworks": context.frameworks_detected,
                },
            },
        )

    except Exception as exc:
        logger.error("LLM self-assessment failed: %s", exc)
        # On failure, return a neutral signal so it doesn't block the pipeline
        return ConfidenceSignal(
            name="llm_self_assessment",
            weight=weight * 0.5,  # Reduce weight on error
            score=0.5,
            evidence={
                "error": str(exc),
                "reasoning": "LLM call failed; using neutral default",
                "model": model,
            },
        )
