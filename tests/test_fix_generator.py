"""Tests for AI-powered failure analysis and fix generation.

Tests the FixGenerator, LLM response parsing, prompt building,
merge logic, and the full pattern + LLM analysis pipeline.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from test_runner.agents.troubleshooter.analyzer import (
    AnalyzerConfig,
    FailureAnalyzer,
    classify_failure,
)
from test_runner.agents.troubleshooter.fix_generator import (
    FixGenerator,
    FixGeneratorConfig,
    LLMAnalysisResult,
    _fallback_proposal,
    build_analysis_prompt,
    merge_analysis,
    parse_llm_response,
)
from test_runner.agents.troubleshooter.models import (
    FailureCategory,
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.models.summary import FailureDetail, TestOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure(
    *,
    test_id: str = "tests/test_foo.py::test_bar",
    test_name: str = "test_bar",
    error_message: str = "",
    error_type: str = "",
    stack_trace: str = "",
    stdout: str = "",
    stderr: str = "",
    file_path: str = "",
    line_number: int | None = None,
    outcome: TestOutcome = TestOutcome.FAILED,
    framework: str = "pytest",
) -> FailureDetail:
    return FailureDetail(
        test_id=test_id,
        test_name=test_name,
        outcome=outcome,
        error_message=error_message,
        error_type=error_type,
        stack_trace=stack_trace,
        stdout=stdout,
        stderr=stderr,
        file_path=file_path,
        line_number=line_number,
        framework=framework,
    )


def _make_llm_response(
    root_cause: str = "Missing dependency",
    confidence: float = 0.85,
    category: str = "import_error",
    explanation: str = "The module is not installed.",
    proposed_fixes: list[dict[str, Any]] | None = None,
    alternative_causes: list[str] | None = None,
    requires_user_action: bool = True,
    user_action_description: str = "pip install requests",
) -> str:
    """Create a valid JSON LLM response string."""
    data = {
        "root_cause": root_cause,
        "confidence": confidence,
        "category": category,
        "explanation": explanation,
        "proposed_fixes": proposed_fixes or [
            {
                "description": "Install missing package",
                "file_path": "requirements.txt",
                "original_snippet": "",
                "proposed_snippet": "requests>=2.28.0",
                "change_type": "add",
            }
        ],
        "alternative_causes": alternative_causes or [],
        "requires_user_action": requires_user_action,
        "user_action_description": user_action_description,
    }
    return json.dumps(data)


# ---------------------------------------------------------------------------
# LLM response parsing tests
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    """Test parse_llm_response with various input formats."""

    def test_valid_json(self):
        response = _make_llm_response()
        result = parse_llm_response(response)
        assert result.root_cause == "Missing dependency"
        assert result.confidence == 0.85
        assert result.category == "import_error"
        assert len(result.proposed_fixes) == 1
        assert result.requires_user_action is True

    def test_json_in_markdown_codeblock(self):
        json_str = _make_llm_response(root_cause="Syntax issue")
        response = f"```json\n{json_str}\n```"
        result = parse_llm_response(response)
        assert result.root_cause == "Syntax issue"
        assert result.confidence == 0.85

    def test_json_with_surrounding_text(self):
        json_str = _make_llm_response(root_cause="Type mismatch")
        response = f"Here is my analysis:\n{json_str}\nHope that helps!"
        result = parse_llm_response(response)
        assert result.root_cause == "Type mismatch"

    def test_no_json_returns_raw_text(self):
        response = "This is just plain text without any JSON."
        result = parse_llm_response(response)
        assert result.confidence == 0.3
        assert "plain text" in result.root_cause
        assert result.raw_response == response

    def test_invalid_json(self):
        response = "{invalid json content here...}"
        result = parse_llm_response(response)
        assert result.confidence == 0.25
        assert result.raw_response == response

    def test_missing_fields_uses_defaults(self):
        response = json.dumps({"root_cause": "something"})
        result = parse_llm_response(response)
        assert result.root_cause == "something"
        assert result.confidence == 0.5  # default
        assert result.category == "unknown"  # default
        assert result.proposed_fixes == []
        assert result.alternative_causes == []

    def test_confidence_clamped_high(self):
        response = json.dumps({"confidence": 1.5, "root_cause": "test"})
        result = parse_llm_response(response)
        assert result.confidence == 1.0

    def test_confidence_clamped_low(self):
        response = json.dumps({"confidence": -0.5, "root_cause": "test"})
        result = parse_llm_response(response)
        assert result.confidence == 0.0

    def test_non_numeric_confidence(self):
        response = json.dumps({"confidence": "high", "root_cause": "test"})
        result = parse_llm_response(response)
        assert result.confidence == 0.5  # fallback to default

    def test_alternative_causes_converted_to_strings(self):
        response = json.dumps({
            "root_cause": "test",
            "alternative_causes": [42, True, "a real cause"],
        })
        result = parse_llm_response(response)
        assert all(isinstance(c, str) for c in result.alternative_causes)
        assert len(result.alternative_causes) == 3

    def test_empty_response(self):
        result = parse_llm_response("")
        assert result.confidence == 0.3
        assert result.raw_response == ""


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class TestBuildAnalysisPrompt:
    def test_contains_failure_details(self):
        failure = _make_failure(
            test_id="tests/test_x.py::test_y",
            error_type="ImportError",
            error_message="No module named 'numpy'",
            file_path="src/data.py",
            line_number=42,
        )
        prompt = build_analysis_prompt(failure, FailureCategory.IMPORT_ERROR, 0.70)
        assert "tests/test_x.py::test_y" in prompt
        assert "ImportError" in prompt
        assert "No module named 'numpy'" in prompt
        assert "src/data.py" in prompt
        assert "42" in prompt
        assert "import_error" in prompt
        assert "0.70" in prompt

    def test_missing_fields_show_defaults(self):
        failure = _make_failure(error_message="something broke")
        prompt = build_analysis_prompt(failure, FailureCategory.UNKNOWN, 0.15)
        assert "unknown" in prompt

    def test_truncates_long_messages(self):
        long_msg = "x" * 5000
        failure = _make_failure(error_message=long_msg)
        prompt = build_analysis_prompt(failure, FailureCategory.RUNTIME, 0.5)
        # Should be truncated to 1000 chars
        assert len(prompt) < 5000


# ---------------------------------------------------------------------------
# Merge analysis tests
# ---------------------------------------------------------------------------


class TestMergeAnalysis:
    def test_merge_with_both_analyses(self):
        failure = _make_failure(
            test_id="test_a",
            error_type="ImportError",
            error_message="No module named 'foo'",
            file_path="src/app.py",
        )
        pattern_proposal = FixProposal(
            failure_id="test_a",
            title="Fix missing module: foo",
            description="Module foo could not be imported.",
            category=FailureCategory.IMPORT_ERROR,
            confidence=FixConfidence.MEDIUM,
            confidence_score=0.60,
            affected_files=["src/app.py"],
            proposed_changes=[
                ProposedChange(
                    file_path="src/app.py",
                    description="Verify import",
                    change_type="config",
                )
            ],
            rationale="Pattern match on ImportError",
            alternative_fixes=["Check PYTHONPATH"],
        )
        llm_result = LLMAnalysisResult(
            root_cause="Module 'foo' is not in requirements.txt",
            confidence=0.90,
            category="import_error",
            explanation="The package needs to be added to project dependencies.",
            proposed_fixes=[{
                "description": "Add foo to requirements.txt",
                "file_path": "requirements.txt",
                "original_snippet": "",
                "proposed_snippet": "foo>=1.0",
                "change_type": "add",
            }],
            requires_user_action=True,
            user_action_description="Run pip install -r requirements.txt",
        )

        merged = merge_analysis(failure, pattern_proposal, llm_result, llm_weight=0.4)

        assert merged.failure_id == "test_a"
        # LLM confidence 0.90 is high, so LLM category should be used
        assert merged.category == FailureCategory.IMPORT_ERROR
        # Blended confidence: 0.6 * 0.60 + 0.4 * 0.90 = 0.36 + 0.36 = 0.72
        assert 0.70 <= merged.confidence_score <= 0.74
        assert merged.confidence == FixConfidence.MEDIUM  # 0.72 < 0.75
        # Changes from both analyses
        assert len(merged.proposed_changes) == 2
        assert "src/app.py" in merged.affected_files
        assert "requirements.txt" in merged.affected_files
        assert merged.requires_user_action is True
        assert merged.metadata.get("llm_augmented") is True

    def test_merge_without_pattern_proposal(self):
        failure = _make_failure(test_id="test_b", error_message="unknown")
        llm_result = LLMAnalysisResult(
            root_cause="Database connection timeout",
            confidence=0.70,
            category="configuration",
            explanation="DB is unreachable.",
        )

        merged = merge_analysis(failure, None, llm_result)

        assert merged.failure_id == "test_b"
        assert merged.category == FailureCategory.CONFIGURATION
        assert merged.title == "Database connection timeout"
        assert "AI Analysis" in merged.description

    def test_merge_with_high_llm_confidence_overrides_category(self):
        failure = _make_failure(test_id="test_c")
        pattern_proposal = FixProposal(
            failure_id="test_c",
            title="Fix runtime error",
            description="desc",
            category=FailureCategory.RUNTIME,
            confidence_score=0.40,
        )
        llm_result = LLMAnalysisResult(
            root_cause="Actual issue is a fixture error",
            confidence=0.85,  # High enough to override
            category="fixture_error",
        )

        merged = merge_analysis(failure, pattern_proposal, llm_result)
        assert merged.category == FailureCategory.FIXTURE_ERROR

    def test_merge_with_low_llm_confidence_keeps_pattern_category(self):
        failure = _make_failure(test_id="test_d")
        pattern_proposal = FixProposal(
            failure_id="test_d",
            title="Fix type error",
            description="desc",
            category=FailureCategory.TYPE_ERROR,
            confidence_score=0.65,
        )
        llm_result = LLMAnalysisResult(
            root_cause="Maybe a config issue",
            confidence=0.40,  # Too low to override
            category="configuration",
        )

        merged = merge_analysis(failure, pattern_proposal, llm_result)
        assert merged.category == FailureCategory.TYPE_ERROR

    def test_merge_limits_alternatives_to_five(self):
        failure = _make_failure(test_id="test_e")
        llm_result = LLMAnalysisResult(
            root_cause="test",
            alternative_causes=[f"cause_{i}" for i in range(10)],
        )
        pattern_proposal = FixProposal(
            failure_id="test_e",
            title="t",
            description="d",
            alternative_fixes=["p1", "p2"],
        )

        merged = merge_analysis(failure, pattern_proposal, llm_result)
        assert len(merged.alternative_fixes) <= 5


# ---------------------------------------------------------------------------
# Fallback proposal tests
# ---------------------------------------------------------------------------


class TestFallbackProposal:
    def test_creates_low_confidence_proposal(self):
        failure = _make_failure(
            test_id="test_fb",
            error_message="something broke",
            file_path="src/x.py",
        )
        proposal = _fallback_proposal(failure, FailureCategory.RUNTIME)

        assert proposal.failure_id == "test_fb"
        assert proposal.confidence == FixConfidence.LOW
        assert proposal.confidence_score == 0.15
        assert proposal.requires_user_action is True
        assert "src/x.py" in proposal.affected_files

    def test_fallback_without_file_path(self):
        failure = _make_failure(test_id="test_fb2", error_message="err")
        proposal = _fallback_proposal(failure, FailureCategory.UNKNOWN)
        assert proposal.affected_files == []


# ---------------------------------------------------------------------------
# FixGenerator tests
# ---------------------------------------------------------------------------


class TestFixGeneratorInit:
    def test_default_init(self):
        gen = FixGenerator()
        assert gen.has_llm_caller is False
        assert gen.llm_calls_made == 0
        assert gen.analyzer is not None

    def test_init_with_llm_caller(self):
        caller = AsyncMock(return_value='{"root_cause": "test"}')
        gen = FixGenerator(llm_caller=caller)
        assert gen.has_llm_caller is True

    def test_init_with_config(self):
        config = FixGeneratorConfig(
            llm_augmentation_threshold=0.80,
            max_llm_calls=3,
        )
        gen = FixGenerator(config=config)
        assert gen.config.llm_augmentation_threshold == 0.80
        assert gen.config.max_llm_calls == 3

    def test_reset_clears_call_count(self):
        gen = FixGenerator()
        gen._llm_calls_made = 5
        gen.reset()
        assert gen.llm_calls_made == 0


class TestFixGeneratorIdentifyCandidates:
    def test_low_confidence_identified(self):
        gen = FixGenerator(config=FixGeneratorConfig(
            llm_augmentation_threshold=0.70,
        ))
        failures = [
            _make_failure(
                test_id="test_low",
                error_message="something odd happened",
            ),
        ]
        candidates = gen.identify_llm_candidates(failures)
        assert len(candidates) == 1
        assert candidates[0][0].test_id == "test_low"

    def test_high_confidence_not_identified(self):
        gen = FixGenerator(config=FixGeneratorConfig(
            llm_augmentation_threshold=0.70,
            always_augment_unknown=False,
        ))
        failures = [
            _make_failure(
                test_id="test_high",
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/x.py",
                line_number=10,
                stack_trace="File 'src/x.py', line 10\nSyntaxError",
            ),
        ]
        candidates = gen.identify_llm_candidates(failures)
        assert len(candidates) == 0

    def test_unknown_always_identified(self):
        gen = FixGenerator(config=FixGeneratorConfig(
            always_augment_unknown=True,
        ))
        failures = [
            _make_failure(
                test_id="test_unknown",
                error_message="totally obscure error",
            ),
        ]
        candidates = gen.identify_llm_candidates(failures)
        assert len(candidates) == 1

    def test_unknown_not_identified_when_disabled(self):
        gen = FixGenerator(config=FixGeneratorConfig(
            llm_augmentation_threshold=0.10,  # Very low
            always_augment_unknown=False,
        ))
        failures = [
            _make_failure(
                test_id="test_unknown",
                error_message="random",
                file_path="src/x.py",
            ),
        ]
        candidates = gen.identify_llm_candidates(failures)
        # Generic strategy gives score ~0.25 which is > 0.10
        assert len(candidates) == 0


class TestFixGeneratorAnalyzeSingle:
    @pytest.mark.asyncio
    async def test_with_llm_caller(self):
        llm_response = _make_llm_response(
            root_cause="Module numpy missing from venv",
            confidence=0.90,
            category="import_error",
            explanation="numpy is not installed in the virtual environment",
        )
        caller = AsyncMock(return_value=llm_response)
        gen = FixGenerator(llm_caller=caller)

        failure = _make_failure(
            test_id="test_llm",
            error_type="ModuleNotFoundError",
            error_message="No module named 'numpy'",
            file_path="src/data.py",
        )

        proposal = await gen.analyze_single_with_llm(failure)

        assert proposal.failure_id == "test_llm"
        assert proposal.metadata.get("llm_augmented") is True
        assert gen.llm_calls_made == 1
        caller.assert_called_once()

    @pytest.mark.asyncio
    async def test_without_llm_caller_returns_pattern(self):
        gen = FixGenerator()  # No LLM caller

        failure = _make_failure(
            test_id="test_no_llm",
            error_type="TypeError",
            error_message="expected int",
            file_path="src/calc.py",
            stack_trace="trace",
        )

        proposal = await gen.analyze_single_with_llm(failure)
        assert proposal.failure_id == "test_no_llm"
        assert proposal.category == FailureCategory.TYPE_ERROR
        assert gen.llm_calls_made == 0

    @pytest.mark.asyncio
    async def test_llm_call_failure_falls_back_to_pattern(self):
        caller = AsyncMock(side_effect=Exception("LLM unavailable"))
        gen = FixGenerator(llm_caller=caller)

        failure = _make_failure(
            test_id="test_err",
            error_type="ImportError",
            error_message="No module named 'foo'",
            file_path="src/x.py",
            stack_trace="trace",
        )

        proposal = await gen.analyze_single_with_llm(failure)
        assert proposal.failure_id == "test_err"
        assert proposal.category == FailureCategory.IMPORT_ERROR
        # Pattern analysis should still work
        assert gen.llm_calls_made == 0  # call failed, counter not incremented

    @pytest.mark.asyncio
    async def test_budget_exhaustion_returns_pattern(self):
        caller = AsyncMock(return_value='{"root_cause": "test", "confidence": 0.9}')
        config = FixGeneratorConfig(max_llm_calls=0)  # Zero budget
        gen = FixGenerator(llm_caller=caller, config=config)

        failure = _make_failure(
            test_id="test_budget",
            error_type="TypeError",
            error_message="err",
            file_path="src/x.py",
            stack_trace="trace",
        )

        proposal = await gen.analyze_single_with_llm(failure)
        caller.assert_not_called()
        assert gen.llm_calls_made == 0


class TestFixGeneratorBatchAnalysis:
    @pytest.mark.asyncio
    async def test_batch_without_llm(self):
        gen = FixGenerator()
        failures = [
            _make_failure(
                test_id="test_1",
                error_type="TypeError",
                error_message="err",
                file_path="src/a.py",
            ),
            _make_failure(
                test_id="test_2",
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/b.py",
                line_number=10,
                stack_trace="trace",
            ),
        ]

        result = await gen.analyze_with_llm(failures)
        assert isinstance(result, FixProposalSet)
        assert result.total_failures_analyzed == 2
        assert result.total_proposals_generated == 2

    @pytest.mark.asyncio
    async def test_batch_with_llm_augmentation(self):
        llm_response = _make_llm_response(
            root_cause="Config file missing",
            confidence=0.80,
            category="configuration",
        )
        caller = AsyncMock(return_value=llm_response)
        gen = FixGenerator(
            llm_caller=caller,
            config=FixGeneratorConfig(
                llm_augmentation_threshold=0.70,
                max_llm_calls=5,
            ),
        )

        failures = [
            # Low confidence — should get LLM augmentation
            _make_failure(
                test_id="test_low",
                error_message="something odd",
            ),
            # High confidence — pattern analysis should suffice
            _make_failure(
                test_id="test_high",
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/x.py",
                line_number=5,
                stack_trace="File 'src/x.py', line 5\nSyntaxError",
            ),
        ]

        result = await gen.analyze_with_llm(failures)
        assert result.total_failures_analyzed == 2
        assert result.total_proposals_generated == 2
        assert gen.llm_calls_made >= 1  # At least the low-confidence one

        # Check that LLM-augmented proposal has metadata
        llm_augmented = [
            p for p in result.proposals
            if p.metadata.get("llm_augmented")
        ]
        assert len(llm_augmented) >= 1

    @pytest.mark.asyncio
    async def test_batch_empty_failures(self):
        gen = FixGenerator()
        result = await gen.analyze_with_llm([])
        assert result.total_failures_analyzed == 0
        assert result.total_proposals_generated == 0

    @pytest.mark.asyncio
    async def test_batch_respects_llm_budget(self):
        caller = AsyncMock(return_value='{"root_cause": "test", "confidence": 0.8}')
        config = FixGeneratorConfig(
            max_llm_calls=1,
            llm_augmentation_threshold=0.99,  # Everything needs LLM
        )
        gen = FixGenerator(llm_caller=caller, config=config)

        failures = [
            _make_failure(test_id=f"test_{i}", error_message="err")
            for i in range(5)
        ]

        result = await gen.analyze_with_llm(failures)
        assert gen.llm_calls_made <= 1
        # All should still get proposals (pattern fallback)
        assert result.total_proposals_generated == 5

    @pytest.mark.asyncio
    async def test_batch_metadata_includes_counts(self):
        caller = AsyncMock(return_value=_make_llm_response())
        config = FixGeneratorConfig(llm_augmentation_threshold=0.99)
        gen = FixGenerator(llm_caller=caller, config=config)

        failures = [
            _make_failure(
                test_id="test_meta",
                error_message="odd error",
            ),
        ]

        result = await gen.analyze_with_llm(failures)
        assert "llm_calls_made" in result.metadata
        assert "llm_augmented_count" in result.metadata


# ---------------------------------------------------------------------------
# Integration: Agent + FixGenerator
# ---------------------------------------------------------------------------


class TestAgentFixGeneratorIntegration:
    """Test that TroubleshooterAgent works with the FixGenerator ecosystem."""

    def test_agent_uses_analyzer(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent()
        failures = [
            _make_failure(
                test_id="test_int",
                error_type="ModuleNotFoundError",
                error_message="No module named 'pandas'",
                file_path="src/data.py",
                stack_trace="trace",
            ),
        ]
        result = agent.generate_fix_proposals(failures)
        assert isinstance(result, FixProposalSet)
        assert result.total_proposals_generated >= 1
        assert agent.last_proposals is result

    def test_agent_proposals_in_handoff(self):
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        agent = TroubleshooterAgent()
        failures = [
            _make_failure(
                test_id="test_ho",
                error_type="TypeError",
                error_message="err",
                file_path="src/x.py",
                stack_trace="trace",
            ),
        ]
        agent.generate_fix_proposals(failures)
        summary = agent.get_handoff_summary()
        assert "fix_proposals" in summary
        assert summary["fix_proposals"]["total"] >= 1

    def test_fix_generator_importable_from_package(self):
        """Verify FixGenerator is exported from the troubleshooter package."""
        from test_runner.agents.troubleshooter import (
            FixGenerator,
            FixGeneratorConfig,
            LLMAnalysisResult,
            build_analysis_prompt,
            merge_analysis,
            parse_llm_response,
        )
        assert FixGenerator is not None
        assert FixGeneratorConfig is not None
        assert LLMAnalysisResult is not None

    def test_full_pipeline_pattern_to_proposal(self):
        """End-to-end: failure → classify → analyze → propose."""
        from test_runner.agents.troubleshooter.agent import TroubleshooterAgent

        failure = _make_failure(
            test_id="tests/test_api.py::test_get_user",
            test_name="test_get_user",
            error_type="AttributeError",
            error_message="'NoneType' object has no attribute 'json'",
            file_path="tests/test_api.py",
            line_number=55,
            stack_trace=(
                'File "tests/test_api.py", line 55, in test_get_user\n'
                "    result = response.json()\n"
                "AttributeError: 'NoneType' object has no attribute 'json'"
            ),
            stderr="AttributeError: 'NoneType' object has no attribute 'json'",
        )

        # Classify
        category = classify_failure(failure)
        assert category == FailureCategory.ATTRIBUTE_ERROR

        # Analyze via agent
        agent = TroubleshooterAgent()
        result = agent.generate_fix_proposals([failure])

        assert result.total_proposals_generated == 1
        proposal = result.proposals[0]
        assert proposal.category == FailureCategory.ATTRIBUTE_ERROR
        assert "json" in proposal.title
        assert proposal.failure_id == "tests/test_api.py::test_get_user"
        assert len(proposal.affected_files) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_with_llm(self):
        """End-to-end: failure → classify → pattern → LLM → merge."""
        llm_response = _make_llm_response(
            root_cause="The API endpoint returns None when auth fails",
            confidence=0.88,
            category="attribute_error",
            explanation="The response is None because authentication failed silently.",
            proposed_fixes=[{
                "description": "Add auth token to test request",
                "file_path": "tests/test_api.py",
                "original_snippet": "response = client.get('/user')",
                "proposed_snippet": "response = client.get('/user', headers={'Authorization': 'Bearer test-token'})",
                "change_type": "modify",
            }],
            alternative_causes=["API server not running"],
            requires_user_action=False,
            user_action_description="",
        )
        caller = AsyncMock(return_value=llm_response)
        gen = FixGenerator(
            llm_caller=caller,
            config=FixGeneratorConfig(
                llm_augmentation_threshold=0.90,  # Force LLM augmentation
            ),
        )

        failure = _make_failure(
            test_id="tests/test_api.py::test_get_user",
            error_type="AttributeError",
            error_message="'NoneType' object has no attribute 'json'",
            file_path="tests/test_api.py",
            line_number=55,
        )

        result = await gen.analyze_with_llm([failure])
        assert result.total_proposals_generated == 1

        proposal = result.proposals[0]
        assert proposal.metadata.get("llm_augmented") is True
        # Should have both pattern and LLM changes
        assert len(proposal.proposed_changes) >= 1
        assert proposal.confidence_score > 0.5
