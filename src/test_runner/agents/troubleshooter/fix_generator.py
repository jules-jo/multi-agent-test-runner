"""AI-powered fix generation using LLM analysis.

The FixGenerator enhances the pattern-based FailureAnalyzer with LLM-powered
root-cause analysis and fix suggestions. It uses the troubleshooter agent's
LLM connection (via Dataiku LLM Mesh) to provide deeper analysis when
pattern-based confidence is insufficient.

Architecture:
- FixGenerator wraps FailureAnalyzer and adds LLM augmentation
- It constructs structured prompts from failure data
- The LLM response is parsed into enhanced FixProposal objects
- Budget-aware: respects the diagnostic step guard
- Diagnose-only: proposes fixes but never executes them

The generator operates in two modes:
1. **Pattern-first**: Always runs pattern-based analysis first
2. **LLM-augmented**: Invokes LLM only when pattern confidence is below
   the augmentation threshold (configurable, default 0.65)

This layered approach minimizes LLM calls while maximizing analysis quality.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from test_runner.agents.troubleshooter.analyzer import (
    AnalyzerConfig,
    FailureAnalyzer,
    classify_failure,
    create_default_registry,
)
from test_runner.agents.troubleshooter.models import (
    FailureCategory,
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.models.summary import FailureDetail

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert test failure analyst. Given a test failure, analyze the \
root cause and suggest fixes. You MUST respond with valid JSON only.

Output format:
{
  "root_cause": "One-line root cause description",
  "confidence": 0.85,
  "category": "import_error|assertion|syntax_error|type_error|attribute_error|\
timeout|fixture_error|configuration|dependency|runtime|unknown",
  "explanation": "Detailed explanation of why this failure occurred",
  "proposed_fixes": [
    {
      "description": "What to change",
      "file_path": "path/to/file.py",
      "original_snippet": "current code",
      "proposed_snippet": "fixed code",
      "change_type": "modify|add|delete|config"
    }
  ],
  "alternative_causes": ["Other possible explanation 1"],
  "requires_user_action": false,
  "user_action_description": ""
}

Rules:
- Confidence must be between 0.0 and 1.0
- Be specific about file paths and line numbers when available
- Never suggest changes to production databases or infrastructure
- Explain WHY each fix addresses the root cause
- If unsure, lower your confidence and list alternative causes
"""

_ANALYSIS_USER_TEMPLATE = """\
Analyze this test failure:

Test: {test_id}
Name: {test_name}
Framework: {framework}
File: {file_path}
Line: {line_number}

Error Type: {error_type}
Error Message:
{error_message}

Stack Trace:
{stack_trace}

Captured stdout:
{stdout}

Captured stderr:
{stderr}

Pattern-based category: {pattern_category}
Pattern-based confidence: {pattern_confidence}
"""


_BATCH_SUMMARY_SYSTEM_PROMPT = """\
You are an expert test failure analyst. Given multiple test failures, \
provide a high-level analysis summary identifying common themes, root \
causes, and prioritized fix recommendations. Respond with valid JSON only.

Output format:
{
  "summary": "High-level analysis of all failures",
  "common_themes": ["theme1", "theme2"],
  "priority_order": ["failure_id_1", "failure_id_2"],
  "cross_cutting_fixes": [
    {
      "description": "A fix that addresses multiple failures",
      "affected_failure_ids": ["id1", "id2"],
      "confidence": 0.8
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMAnalysisResult:
    """Parsed result from the LLM's failure analysis.

    Attributes:
        root_cause: The LLM's root cause assessment.
        confidence: LLM confidence in the analysis (0.0-1.0).
        category: Failure category assigned by the LLM.
        explanation: Detailed explanation of the failure.
        proposed_fixes: Structured fix suggestions from the LLM.
        alternative_causes: Other possible explanations.
        requires_user_action: Whether manual steps are needed.
        user_action_description: What the user needs to do.
        raw_response: The original LLM response text.
    """

    root_cause: str = ""
    confidence: float = 0.5
    category: str = "unknown"
    explanation: str = ""
    proposed_fixes: list[dict[str, Any]] = field(default_factory=list)
    alternative_causes: list[str] = field(default_factory=list)
    requires_user_action: bool = False
    user_action_description: str = ""
    raw_response: str = ""


def parse_llm_response(response_text: str) -> LLMAnalysisResult:
    """Parse the LLM's JSON response into a structured result.

    Handles malformed JSON gracefully — extracts what it can and
    falls back to defaults for missing fields.

    Args:
        response_text: Raw text from the LLM response.

    Returns:
        A parsed LLMAnalysisResult. If parsing fails completely,
        returns a result with the raw response and low confidence.
    """
    # Try to extract JSON from the response (may be wrapped in markdown)
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if not json_match:
        logger.warning("No JSON found in LLM response, using raw text")
        return LLMAnalysisResult(
            root_cause=response_text[:200],
            confidence=0.3,
            explanation=response_text,
            raw_response=response_text,
        )

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON: %s", e)
        return LLMAnalysisResult(
            root_cause=response_text[:200],
            confidence=0.25,
            explanation=response_text,
            raw_response=response_text,
        )

    # Extract with safe defaults
    confidence = data.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))

    return LLMAnalysisResult(
        root_cause=str(data.get("root_cause", "")),
        confidence=confidence,
        category=str(data.get("category", "unknown")),
        explanation=str(data.get("explanation", "")),
        proposed_fixes=data.get("proposed_fixes", []),
        alternative_causes=[
            str(c) for c in data.get("alternative_causes", [])
        ],
        requires_user_action=bool(data.get("requires_user_action", False)),
        user_action_description=str(
            data.get("user_action_description", "")
        ),
        raw_response=response_text,
    )


def build_analysis_prompt(
    failure: FailureDetail,
    pattern_category: FailureCategory,
    pattern_confidence: float,
) -> str:
    """Build the user prompt for LLM failure analysis.

    Formats the failure data into the structured prompt template.

    Args:
        failure: The test failure detail.
        pattern_category: Category from pattern-based classification.
        pattern_confidence: Confidence from pattern-based analysis.

    Returns:
        Formatted prompt string.
    """
    return _ANALYSIS_USER_TEMPLATE.format(
        test_id=failure.test_id,
        test_name=failure.test_name,
        framework=failure.framework or "unknown",
        file_path=failure.file_path or "unknown",
        line_number=failure.line_number or "unknown",
        error_type=failure.error_type or "unknown",
        error_message=failure.error_message[:1000] or "no message",
        stack_trace=failure.stack_trace[:2000] or "no stack trace",
        stdout=failure.stdout[:500] or "none",
        stderr=failure.stderr[:500] or "none",
        pattern_category=pattern_category.value,
        pattern_confidence=f"{pattern_confidence:.2f}",
    )


# ---------------------------------------------------------------------------
# Merge logic: combine pattern + LLM analysis into a FixProposal
# ---------------------------------------------------------------------------

_CATEGORY_MAP: dict[str, FailureCategory] = {
    c.value: c for c in FailureCategory
}


def merge_analysis(
    failure: FailureDetail,
    pattern_proposal: FixProposal | None,
    llm_result: LLMAnalysisResult,
    llm_weight: float = 0.4,
) -> FixProposal:
    """Merge pattern-based and LLM analyses into a single FixProposal.

    The LLM result enhances the pattern-based proposal with:
    - More detailed root cause explanation
    - Additional proposed changes (code diffs)
    - Better confidence when LLM agrees with pattern analysis
    - Alternative causes from LLM reasoning

    Args:
        failure: The original failure detail.
        pattern_proposal: The pattern-based proposal (may be None).
        llm_result: Parsed LLM analysis result.
        llm_weight: Weight for LLM confidence in blending (0.0-1.0).

    Returns:
        A merged FixProposal combining both analyses.
    """
    # Determine category
    llm_category = _CATEGORY_MAP.get(
        llm_result.category, FailureCategory.UNKNOWN,
    )
    if pattern_proposal:
        # Prefer pattern category unless LLM is very confident
        if llm_result.confidence >= 0.8:
            category = llm_category
        else:
            category = pattern_proposal.category
    else:
        category = llm_category

    # Blend confidence scores
    pattern_score = pattern_proposal.confidence_score if pattern_proposal else 0.3
    combined_score = (1 - llm_weight) * pattern_score + llm_weight * llm_result.confidence
    combined_score = round(min(combined_score, 1.0), 4)

    # Map to confidence tier
    if combined_score >= 0.75:
        confidence_tier = FixConfidence.HIGH
    elif combined_score >= 0.45:
        confidence_tier = FixConfidence.MEDIUM
    else:
        confidence_tier = FixConfidence.LOW

    # Build proposed changes from LLM suggestions
    llm_changes: list[ProposedChange] = []
    for fix_dict in llm_result.proposed_fixes:
        if not isinstance(fix_dict, dict):
            continue
        file_path = fix_dict.get("file_path", "")
        if not file_path:
            file_path = failure.file_path or ""
        llm_changes.append(ProposedChange(
            file_path=file_path,
            description=str(fix_dict.get("description", "LLM-suggested fix")),
            original_snippet=str(fix_dict.get("original_snippet", "")),
            proposed_snippet=str(fix_dict.get("proposed_snippet", "")),
            change_type=str(fix_dict.get("change_type", "modify")),
        ))

    # Merge changes: pattern changes first, then LLM additions
    all_changes = list(
        pattern_proposal.proposed_changes if pattern_proposal else []
    ) + llm_changes

    # Merge affected files
    affected_files: list[str] = []
    seen: set[str] = set()
    for fp in (
        (pattern_proposal.affected_files if pattern_proposal else [])
        + [c.file_path for c in llm_changes if c.file_path]
    ):
        if fp and fp not in seen:
            affected_files.append(fp)
            seen.add(fp)

    # Merge alternative causes
    alternatives = list(
        pattern_proposal.alternative_fixes if pattern_proposal else []
    ) + llm_result.alternative_causes

    # Build title — prefer LLM root cause if available
    title = llm_result.root_cause or (
        pattern_proposal.title if pattern_proposal else f"Fix {category.value} in {failure.test_name}"
    )

    # Build description — combine both
    desc_parts: list[str] = []
    if pattern_proposal and pattern_proposal.description:
        desc_parts.append(pattern_proposal.description)
    if llm_result.explanation:
        desc_parts.append(f"AI Analysis: {llm_result.explanation}")
    description = "\n\n".join(desc_parts) or title

    # Build rationale
    rationale_parts: list[str] = []
    if pattern_proposal and pattern_proposal.rationale:
        rationale_parts.append(f"Pattern analysis: {pattern_proposal.rationale}")
    rationale_parts.append(
        f"LLM analysis (confidence={llm_result.confidence:.2f}): {llm_result.root_cause}"
    )
    rationale = " | ".join(rationale_parts)

    # Determine user action
    requires_user = llm_result.requires_user_action or (
        pattern_proposal.requires_user_action if pattern_proposal else False
    )
    user_action = llm_result.user_action_description or (
        pattern_proposal.user_action_description if pattern_proposal else ""
    )

    return FixProposal(
        failure_id=failure.test_id,
        title=title[:200],  # Truncate long titles
        description=description,
        category=category,
        confidence=confidence_tier,
        confidence_score=combined_score,
        affected_files=affected_files,
        proposed_changes=all_changes,
        rationale=rationale,
        alternative_fixes=alternatives[:5],  # Limit to 5
        requires_user_action=requires_user,
        user_action_description=user_action,
        metadata={
            "llm_augmented": True,
            "pattern_confidence": pattern_score,
            "llm_confidence": llm_result.confidence,
            "llm_category": llm_result.category,
        },
    )


# ---------------------------------------------------------------------------
# FixGenerator — orchestrates pattern + LLM analysis
# ---------------------------------------------------------------------------


@dataclass
class FixGeneratorConfig:
    """Configuration for the AI-powered fix generator.

    Attributes:
        llm_augmentation_threshold: Pattern confidence below which LLM
            analysis is triggered (default 0.65).
        llm_confidence_weight: How much LLM confidence contributes to
            the blended score (0.0-1.0, default 0.4).
        max_llm_calls: Maximum number of LLM calls per batch to limit
            cost/latency (default 5).
        always_augment_unknown: Always use LLM for UNKNOWN category
            failures regardless of confidence.
    """

    llm_augmentation_threshold: float = 0.65
    llm_confidence_weight: float = 0.4
    max_llm_calls: int = 5
    always_augment_unknown: bool = True


class FixGenerator:
    """AI-powered fix generator combining pattern-based and LLM analysis.

    The FixGenerator wraps the FailureAnalyzer and adds LLM augmentation
    for failures where pattern-based confidence is insufficient. It respects
    a call budget and the troubleshooter's diagnose-only policy.

    Usage::

        generator = FixGenerator(
            analyzer=FailureAnalyzer(),
            llm_caller=my_llm_function,  # async (system, user) -> str
        )

        # Analyze with optional LLM augmentation
        result = await generator.analyze_with_llm(failures)

        # Or just check which failures need LLM help
        needs_llm = generator.identify_llm_candidates(failures)

    The ``llm_caller`` is an async callable that takes two strings
    (system_prompt, user_prompt) and returns the LLM's text response.
    This abstraction keeps the generator decoupled from specific LLM
    clients and works with the Dataiku LLM Mesh or any OpenAI-compatible
    endpoint.
    """

    def __init__(
        self,
        analyzer: FailureAnalyzer | None = None,
        llm_caller: Any = None,
        config: FixGeneratorConfig | None = None,
    ) -> None:
        self._analyzer = analyzer or FailureAnalyzer()
        self._llm_caller = llm_caller  # async (system: str, user: str) -> str
        self._config = config or FixGeneratorConfig()
        self._llm_calls_made = 0

    @property
    def analyzer(self) -> FailureAnalyzer:
        """The underlying pattern-based analyzer."""
        return self._analyzer

    @property
    def config(self) -> FixGeneratorConfig:
        """Current generator configuration."""
        return self._config

    @property
    def llm_calls_made(self) -> int:
        """Number of LLM calls made in this session."""
        return self._llm_calls_made

    @property
    def has_llm_caller(self) -> bool:
        """Whether an LLM caller is configured."""
        return self._llm_caller is not None

    def reset(self) -> None:
        """Reset the LLM call counter."""
        self._llm_calls_made = 0

    def identify_llm_candidates(
        self,
        failures: Sequence[FailureDetail],
    ) -> list[tuple[FailureDetail, FixProposal | None, FailureCategory]]:
        """Identify failures that would benefit from LLM analysis.

        Runs pattern-based analysis and returns failures where:
        - Pattern confidence is below the augmentation threshold, OR
        - Category is UNKNOWN and always_augment_unknown is True

        Args:
            failures: Test failure details to evaluate.

        Returns:
            List of (failure, pattern_proposal, category) tuples for
            failures that should be sent to the LLM.
        """
        candidates: list[tuple[FailureDetail, FixProposal | None, FailureCategory]] = []

        for failure in failures:
            category = classify_failure(failure)
            proposal = self._analyzer.analyze_single(failure)

            needs_augmentation = False
            if proposal is None:
                needs_augmentation = True
            elif proposal.confidence_score < self._config.llm_augmentation_threshold:
                needs_augmentation = True
            elif (
                category == FailureCategory.UNKNOWN
                and self._config.always_augment_unknown
            ):
                needs_augmentation = True

            if needs_augmentation:
                candidates.append((failure, proposal, category))

        return candidates

    async def analyze_single_with_llm(
        self,
        failure: FailureDetail,
        pattern_proposal: FixProposal | None = None,
    ) -> FixProposal:
        """Analyze a single failure with LLM augmentation.

        If no LLM caller is configured, returns the pattern proposal
        (or a low-confidence fallback).

        Args:
            failure: The test failure to analyze.
            pattern_proposal: Optional pattern-based proposal to enhance.

        Returns:
            An enhanced FixProposal combining pattern + LLM analysis.
        """
        category = classify_failure(failure)
        if pattern_proposal is None:
            pattern_proposal = self._analyzer.analyze_single(failure)

        # If no LLM caller, return pattern analysis only
        if not self.has_llm_caller:
            if pattern_proposal is not None:
                return pattern_proposal
            return _fallback_proposal(failure, category)

        # Check LLM call budget
        if self._llm_calls_made >= self._config.max_llm_calls:
            logger.warning(
                "LLM call budget exhausted (%d/%d)",
                self._llm_calls_made, self._config.max_llm_calls,
            )
            if pattern_proposal is not None:
                return pattern_proposal
            return _fallback_proposal(failure, category)

        # Build prompt and call LLM
        pattern_confidence = (
            pattern_proposal.confidence_score if pattern_proposal else 0.0
        )
        user_prompt = build_analysis_prompt(
            failure, category, pattern_confidence,
        )

        try:
            response_text = await self._llm_caller(
                _ANALYSIS_SYSTEM_PROMPT, user_prompt,
            )
            self._llm_calls_made += 1
            logger.info(
                "LLM analysis completed for %s (call %d/%d)",
                failure.test_id,
                self._llm_calls_made,
                self._config.max_llm_calls,
            )
        except Exception as e:
            logger.error("LLM call failed for %s: %s", failure.test_id, e)
            if pattern_proposal is not None:
                return pattern_proposal
            return _fallback_proposal(failure, category)

        # Parse and merge
        llm_result = parse_llm_response(response_text)
        return merge_analysis(
            failure,
            pattern_proposal,
            llm_result,
            llm_weight=self._config.llm_confidence_weight,
        )

    async def analyze_with_llm(
        self,
        failures: Sequence[FailureDetail],
    ) -> FixProposalSet:
        """Analyze failures with pattern-based + LLM augmentation.

        Workflow:
        1. Run pattern-based analysis on all failures
        2. Identify candidates for LLM augmentation
        3. Call LLM for each candidate (up to budget)
        4. Merge results into a unified FixProposalSet

        Args:
            failures: Test failure details to analyze.

        Returns:
            A FixProposalSet with all proposals (pattern-only + LLM-augmented).
        """
        if not failures:
            return FixProposalSet(
                analysis_summary="No failures to analyze.",
                total_failures_analyzed=0,
            )

        # Step 1: Pattern-based analysis for all failures
        pattern_result = self._analyzer.analyze_failures(failures)
        pattern_map: dict[str, FixProposal] = {
            p.failure_id: p for p in pattern_result.proposals
        }

        # If no LLM caller, return pattern analysis only
        if not self.has_llm_caller:
            return pattern_result

        # Step 2: Identify candidates
        candidates = self.identify_llm_candidates(failures)
        logger.info(
            "%d of %d failures need LLM augmentation",
            len(candidates), len(failures),
        )

        # Step 3: LLM augmentation for candidates
        enhanced_proposals: dict[str, FixProposal] = {}
        for failure, proposal, category in candidates:
            if self._llm_calls_made >= self._config.max_llm_calls:
                logger.info("LLM call budget reached, stopping augmentation")
                break

            enhanced = await self.analyze_single_with_llm(failure, proposal)
            enhanced_proposals[failure.test_id] = enhanced

        # Step 4: Merge — enhanced proposals override pattern proposals
        final_proposals: list[FixProposal] = []
        seen: set[str] = set()

        for failure in failures:
            fid = failure.test_id
            if fid in seen:
                continue
            seen.add(fid)

            if fid in enhanced_proposals:
                final_proposals.append(enhanced_proposals[fid])
            elif fid in pattern_map:
                final_proposals.append(pattern_map[fid])
            # else: failure was not analyzed (budget exhaustion)

        llm_augmented_count = len(enhanced_proposals)
        summary_parts = [
            f"Analyzed {len(failures)} failure(s).",
            f"Generated {len(final_proposals)} fix proposal(s).",
            f"LLM-augmented: {llm_augmented_count}.",
            f"Pattern-only: {len(final_proposals) - llm_augmented_count}.",
        ]

        return FixProposalSet(
            proposals=final_proposals,
            analysis_summary=" ".join(summary_parts),
            total_failures_analyzed=len(failures),
            total_proposals_generated=len(final_proposals),
            budget_exhausted=pattern_result.budget_exhausted,
            metadata={
                "llm_calls_made": self._llm_calls_made,
                "llm_augmented_count": llm_augmented_count,
            },
        )


def _fallback_proposal(
    failure: FailureDetail,
    category: FailureCategory,
) -> FixProposal:
    """Create a minimal fallback proposal when analysis fails."""
    return FixProposal(
        failure_id=failure.test_id,
        title=f"Investigate {category.value} in {failure.test_name}",
        description=f"Test failed with: {failure.error_message[:300] or 'no message'}",
        category=category,
        confidence=FixConfidence.LOW,
        confidence_score=0.15,
        affected_files=[failure.file_path] if failure.file_path else [],
        rationale="Fallback proposal — both pattern and LLM analysis unavailable.",
        requires_user_action=True,
        user_action_description="Review test output manually to identify root cause.",
    )
