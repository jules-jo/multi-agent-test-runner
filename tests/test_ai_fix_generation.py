"""Tests for AI-suggested fixes generation via the troubleshooter agent.

Sub-AC 3 of AC 9: Verifies that the troubleshooter agent produces actionable
fix recommendations for each failure using LLM-augmented analysis, including:
- FixGenerator integration with TroubleshooterAgent
- LLM caller injection via set_llm_caller
- Async generate_fix_proposals_with_llm method
- Orchestrator hub wiring with _create_llm_caller
- Fallback to pattern-only analysis when LLM is unavailable
- Handoff summary includes LLM augmentation metadata
- Budget enforcement with LLM calls
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from test_runner.agents.troubleshooter.agent import TroubleshooterAgent
from test_runner.agents.troubleshooter.fix_generator import (
    FixGenerator,
    FixGeneratorConfig,
)
from test_runner.agents.troubleshooter.models import (
    FailureCategory,
    FixConfidence,
    FixProposal,
    FixProposalSet,
    ProposedChange,
)
from test_runner.config import Config
from test_runner.models.summary import FailureDetail, TestOutcome
from test_runner.orchestrator.hub import OrchestratorHub, RunPhase, RunState


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


def _make_config(
    llm_base_url: str = "https://llm.dataiku.example.com/v1",
    api_key: str = "test-key-123",
    model_id: str = "gpt-4o",
) -> Config:
    return Config(
        llm_base_url=llm_base_url,
        api_key=api_key,
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# TroubleshooterAgent + FixGenerator integration
# ---------------------------------------------------------------------------


class TestTroubleshooterAgentFixGeneratorInit:
    """Test FixGenerator integration in TroubleshooterAgent constructor."""

    def test_default_agent_has_fix_generator(self):
        agent = TroubleshooterAgent()
        assert agent.fix_generator is not None
        assert isinstance(agent.fix_generator, FixGenerator)
        assert agent.fix_generator.has_llm_caller is False

    def test_agent_with_llm_caller(self):
        caller = AsyncMock(return_value='{"root_cause": "test"}')
        agent = TroubleshooterAgent(llm_caller=caller)
        assert agent.fix_generator.has_llm_caller is True

    def test_agent_with_custom_fix_generator(self):
        custom_gen = FixGenerator(
            config=FixGeneratorConfig(max_llm_calls=10),
        )
        agent = TroubleshooterAgent(fix_generator=custom_gen)
        assert agent.fix_generator is custom_gen
        assert agent.fix_generator.config.max_llm_calls == 10

    def test_agent_with_fix_generator_config(self):
        config = FixGeneratorConfig(
            llm_augmentation_threshold=0.80,
            max_llm_calls=3,
        )
        agent = TroubleshooterAgent(fix_generator_config=config)
        assert agent.fix_generator.config.llm_augmentation_threshold == 0.80
        assert agent.fix_generator.config.max_llm_calls == 3

    def test_fix_generator_shares_analyzer(self):
        """FixGenerator should use the same FailureAnalyzer as the agent."""
        agent = TroubleshooterAgent()
        assert agent.fix_generator.analyzer is agent.analyzer


class TestSetLLMCaller:
    """Test set_llm_caller for post-construction LLM injection."""

    def test_set_llm_caller_enables_llm(self):
        agent = TroubleshooterAgent()
        assert agent.fix_generator.has_llm_caller is False

        caller = AsyncMock(return_value='{"root_cause": "test"}')
        agent.set_llm_caller(caller)
        assert agent.fix_generator.has_llm_caller is True

    def test_set_llm_caller_preserves_config(self):
        config = FixGeneratorConfig(max_llm_calls=7)
        agent = TroubleshooterAgent(fix_generator_config=config)

        caller = AsyncMock()
        agent.set_llm_caller(caller)
        assert agent.fix_generator.config.max_llm_calls == 7

    def test_set_llm_caller_replaces_previous(self):
        caller1 = AsyncMock(return_value='{"root_cause": "v1"}')
        caller2 = AsyncMock(return_value='{"root_cause": "v2"}')

        agent = TroubleshooterAgent(llm_caller=caller1)
        agent.set_llm_caller(caller2)
        assert agent.fix_generator.has_llm_caller is True


class TestResetState:
    """Test that reset_state clears fix generator call counter."""

    def test_reset_clears_llm_call_count(self):
        caller = AsyncMock(return_value='{"root_cause": "test"}')
        agent = TroubleshooterAgent(llm_caller=caller)
        agent.fix_generator._llm_calls_made = 5
        agent.reset_state()
        assert agent.fix_generator.llm_calls_made == 0


# ---------------------------------------------------------------------------
# Async generate_fix_proposals_with_llm
# ---------------------------------------------------------------------------


class TestGenerateFixProposalsWithLLM:
    """Test the async AI-augmented fix proposal generation."""

    @pytest.mark.asyncio
    async def test_with_llm_caller_produces_augmented_proposals(self):
        llm_response = _make_llm_response(
            root_cause="Module 'numpy' not in virtualenv",
            confidence=0.92,
            category="import_error",
            explanation="numpy is missing from the project dependencies",
            proposed_fixes=[{
                "description": "Add numpy to requirements.txt",
                "file_path": "requirements.txt",
                "original_snippet": "",
                "proposed_snippet": "numpy>=1.24.0",
                "change_type": "add",
            }],
        )
        caller = AsyncMock(return_value=llm_response)
        agent = TroubleshooterAgent(
            llm_caller=caller,
            fix_generator_config=FixGeneratorConfig(
                llm_augmentation_threshold=0.99,  # Force LLM augmentation
            ),
        )

        failures = [
            _make_failure(
                test_id="tests/test_data.py::test_load",
                error_type="ModuleNotFoundError",
                error_message="No module named 'numpy'",
                file_path="src/data.py",
                stack_trace="Traceback\nModuleNotFoundError",
            ),
        ]

        result = await agent.generate_fix_proposals_with_llm(failures)

        assert isinstance(result, FixProposalSet)
        assert result.total_proposals_generated >= 1
        assert agent.last_proposals is result

        # Verify LLM was called
        assert agent.fix_generator.llm_calls_made >= 1

        # Verify augmented proposals have metadata
        augmented = [p for p in result.proposals if p.metadata.get("llm_augmented")]
        assert len(augmented) >= 1

    @pytest.mark.asyncio
    async def test_without_llm_caller_falls_back_to_pattern(self):
        agent = TroubleshooterAgent()  # No LLM caller

        failures = [
            _make_failure(
                test_id="tests/test_x.py::test_y",
                error_type="TypeError",
                error_message="expected int",
                file_path="src/calc.py",
                stack_trace="trace",
            ),
        ]

        result = await agent.generate_fix_proposals_with_llm(failures)

        assert result.total_proposals_generated >= 1
        assert agent.fix_generator.llm_calls_made == 0
        # Pattern-only proposals should not have llm_augmented metadata
        for p in result.proposals:
            assert p.metadata.get("llm_augmented") is not True

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_pattern(self):
        caller = AsyncMock(side_effect=Exception("LLM service unavailable"))
        agent = TroubleshooterAgent(
            llm_caller=caller,
            fix_generator_config=FixGeneratorConfig(
                llm_augmentation_threshold=0.99,
            ),
        )

        failures = [
            _make_failure(
                test_id="tests/test_x.py::test_y",
                error_type="ImportError",
                error_message="No module named 'foo'",
                file_path="src/x.py",
                stack_trace="trace",
            ),
        ]

        result = await agent.generate_fix_proposals_with_llm(failures)

        # Should still produce proposals (pattern-based fallback within FixGenerator)
        assert result.total_proposals_generated >= 1
        assert agent.last_proposals is result

    @pytest.mark.asyncio
    async def test_empty_failures_returns_empty_set(self):
        agent = TroubleshooterAgent()
        result = await agent.generate_fix_proposals_with_llm([])
        assert result.total_failures_analyzed == 0
        assert result.total_proposals_generated == 0

    @pytest.mark.asyncio
    async def test_escalation_before_llm_analysis(self):
        """Agent should skip LLM analysis if step budget is exhausted."""
        agent = TroubleshooterAgent(hard_cap_steps=1)
        # Exhaust the step counter
        agent._step_counter.increment("warmup", "")

        failures = [
            _make_failure(
                test_id="test_esc",
                error_message="fail",
            ),
        ]

        result = await agent.generate_fix_proposals_with_llm(failures)
        assert result.budget_exhausted is True
        assert "skipped" in result.analysis_summary.lower()

    @pytest.mark.asyncio
    async def test_updates_agent_state_findings(self):
        agent = TroubleshooterAgent()

        failures = [
            _make_failure(
                test_id="tests/test_state.py::test_a",
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/broken.py",
                line_number=10,
                stack_trace="SyntaxError trace",
            ),
        ]

        await agent.generate_fix_proposals_with_llm(failures)

        # Agent state should have findings recorded
        findings = agent.state.findings
        assert len(findings) >= 1
        assert findings[0]["type"] == "fix_proposal"
        assert findings[0]["failure_id"] == "tests/test_state.py::test_a"

    @pytest.mark.asyncio
    async def test_multiple_failures_with_mixed_llm(self):
        """Test batch analysis with some failures needing LLM, others not."""
        llm_response = _make_llm_response(
            root_cause="Configuration error",
            confidence=0.80,
            category="configuration",
        )
        caller = AsyncMock(return_value=llm_response)
        agent = TroubleshooterAgent(
            llm_caller=caller,
            fix_generator_config=FixGeneratorConfig(
                llm_augmentation_threshold=0.70,
                max_llm_calls=5,
            ),
        )

        failures = [
            # Low confidence — should get LLM augmentation
            _make_failure(
                test_id="test_low",
                error_message="something odd happened",
            ),
            # High confidence (SyntaxError with file+line) — pattern-only
            _make_failure(
                test_id="test_high",
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/x.py",
                line_number=5,
                stack_trace="File 'src/x.py', line 5\nSyntaxError",
            ),
        ]

        result = await agent.generate_fix_proposals_with_llm(failures)

        assert result.total_proposals_generated == 2
        assert result.total_failures_analyzed == 2

    @pytest.mark.asyncio
    async def test_llm_budget_respected(self):
        caller = AsyncMock(return_value=_make_llm_response())
        agent = TroubleshooterAgent(
            llm_caller=caller,
            fix_generator_config=FixGeneratorConfig(
                max_llm_calls=1,
                llm_augmentation_threshold=0.99,
            ),
        )

        failures = [
            _make_failure(test_id=f"test_{i}", error_message="error")
            for i in range(5)
        ]

        result = await agent.generate_fix_proposals_with_llm(failures)

        # Only 1 LLM call should be made
        assert agent.fix_generator.llm_calls_made <= 1
        # All failures should still get proposals (pattern fallback)
        assert result.total_proposals_generated == 5


# ---------------------------------------------------------------------------
# Handoff summary with LLM info
# ---------------------------------------------------------------------------


class TestHandoffSummaryWithLLM:
    """Test that handoff summary includes LLM augmentation metadata."""

    def test_handoff_includes_llm_fields(self):
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

        assert "llm_calls_made" in summary
        assert "has_llm_caller" in summary
        assert summary["llm_calls_made"] == 0
        assert summary["has_llm_caller"] is False

    @pytest.mark.asyncio
    async def test_handoff_after_llm_augmented_proposals(self):
        llm_response = _make_llm_response()
        caller = AsyncMock(return_value=llm_response)
        agent = TroubleshooterAgent(
            llm_caller=caller,
            fix_generator_config=FixGeneratorConfig(
                llm_augmentation_threshold=0.99,
            ),
        )

        failures = [
            _make_failure(
                test_id="test_llm_ho",
                error_message="obscure error",
            ),
        ]

        await agent.generate_fix_proposals_with_llm(failures)
        summary = agent.get_handoff_summary()

        assert summary["has_llm_caller"] is True
        assert summary["llm_calls_made"] >= 1
        assert "fix_proposals" in summary
        assert summary["fix_proposals"]["llm_augmented_count"] >= 1

        # Each proposal should have llm_augmented field
        for p in summary["fix_proposals"]["proposals"]:
            assert "llm_augmented" in p

    def test_handoff_proposals_sorted_by_confidence(self):
        agent = TroubleshooterAgent()
        failures = [
            _make_failure(
                test_id="test_low",
                error_message="random",
            ),
            _make_failure(
                test_id="test_high",
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/x.py",
                line_number=5,
                stack_trace="trace",
            ),
        ]
        agent.generate_fix_proposals(failures)
        summary = agent.get_handoff_summary()

        proposals = summary["fix_proposals"]["proposals"]
        assert len(proposals) == 2
        # Should be sorted: high confidence first
        scores = [p["confidence_score"] for p in proposals]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Orchestrator hub LLM caller creation
# ---------------------------------------------------------------------------


class TestOrchestratorLLMCaller:
    """Test orchestrator hub's _create_llm_caller method."""

    def test_create_llm_caller_with_config(self):
        config = _make_config()
        hub = OrchestratorHub(config)
        caller = hub._create_llm_caller()
        assert caller is not None
        assert callable(caller)

    def test_create_llm_caller_without_url(self):
        config = _make_config(llm_base_url="")
        hub = OrchestratorHub(config)
        caller = hub._create_llm_caller()
        assert caller is None

    def test_create_llm_caller_without_api_key(self):
        config = _make_config(api_key="")
        hub = OrchestratorHub(config)
        caller = hub._create_llm_caller()
        assert caller is None


class TestOrchestratorInvokeTroubleshooter:
    """Test orchestrator hub's troubleshooter invocation with LLM."""

    @pytest.mark.asyncio
    async def test_invoke_troubleshooter_uses_llm_when_configured(self):
        config = _make_config()
        troubleshooter = TroubleshooterAgent()
        hub = OrchestratorHub(config, troubleshooter=troubleshooter)

        state = RunState()
        state.failure_details = [
            _make_failure(
                test_id="test_invoke",
                error_type="ImportError",
                error_message="No module named 'pandas'",
                file_path="src/data.py",
                stack_trace="trace",
            ),
        ]

        # Mock _create_llm_caller to avoid actual OpenAI calls
        llm_response = _make_llm_response(
            root_cause="pandas not installed",
            confidence=0.88,
        )
        mock_caller = AsyncMock(return_value=llm_response)
        with patch.object(hub, "_create_llm_caller", return_value=mock_caller):
            await hub._invoke_troubleshooter(state)

        assert state.troubleshooter_result is not None
        assert state.troubleshooter_result.total_proposals_generated >= 1
        assert len(state.escalations) == 1
        assert "llm_augmented_proposals" in state.escalations[0].metadata

    @pytest.mark.asyncio
    async def test_invoke_troubleshooter_pattern_only_when_no_llm(self):
        config = _make_config(llm_base_url="", api_key="")
        troubleshooter = TroubleshooterAgent()
        hub = OrchestratorHub(config, troubleshooter=troubleshooter)

        state = RunState()
        state.failure_details = [
            _make_failure(
                test_id="test_no_llm",
                error_type="TypeError",
                error_message="err",
                file_path="src/x.py",
                stack_trace="trace",
            ),
        ]

        await hub._invoke_troubleshooter(state)

        assert state.troubleshooter_result is not None
        assert state.troubleshooter_result.total_proposals_generated >= 1

    @pytest.mark.asyncio
    async def test_invoke_troubleshooter_skips_empty_failures(self):
        config = _make_config()
        hub = OrchestratorHub(config)

        state = RunState()
        state.failure_details = []

        await hub._invoke_troubleshooter(state)

        assert state.troubleshooter_result is None

    @pytest.mark.asyncio
    async def test_invoke_troubleshooter_handles_error_gracefully(self):
        config = _make_config()
        troubleshooter = TroubleshooterAgent()
        hub = OrchestratorHub(config, troubleshooter=troubleshooter)

        state = RunState()
        state.failure_details = [
            _make_failure(test_id="test_err", error_message="fail"),
        ]

        # Make the troubleshooter raise an error
        with patch.object(
            troubleshooter, "generate_fix_proposals_with_llm",
            side_effect=RuntimeError("Unexpected error"),
        ):
            await hub._invoke_troubleshooter(state)

        assert state.troubleshooter_result is None
        assert any("Troubleshooter error" in e for e in state.errors)


# ---------------------------------------------------------------------------
# End-to-end: failure → troubleshooter → AI-augmented proposals
# ---------------------------------------------------------------------------


class TestEndToEndAIFixGeneration:
    """End-to-end tests for the AI-suggested fix generation pipeline."""

    @pytest.mark.asyncio
    async def test_import_error_gets_actionable_fix(self):
        """ImportError should produce actionable fix with pip install suggestion."""
        llm_response = _make_llm_response(
            root_cause="numpy package is not installed",
            confidence=0.95,
            category="import_error",
            explanation="The project depends on numpy but it's not in the venv",
            proposed_fixes=[{
                "description": "Add numpy to requirements.txt",
                "file_path": "requirements.txt",
                "original_snippet": "",
                "proposed_snippet": "numpy>=1.24.0",
                "change_type": "add",
            }],
            requires_user_action=True,
            user_action_description="pip install numpy",
        )
        caller = AsyncMock(return_value=llm_response)
        agent = TroubleshooterAgent(
            llm_caller=caller,
            fix_generator_config=FixGeneratorConfig(
                llm_augmentation_threshold=0.99,
            ),
        )

        failure = _make_failure(
            test_id="tests/test_data.py::test_load_csv",
            test_name="test_load_csv",
            error_type="ModuleNotFoundError",
            error_message="No module named 'numpy'",
            file_path="tests/test_data.py",
            line_number=10,
            stack_trace=(
                'File "tests/test_data.py", line 10, in test_load_csv\n'
                "    import numpy\n"
                "ModuleNotFoundError: No module named 'numpy'"
            ),
            stderr="ModuleNotFoundError: No module named 'numpy'",
        )

        result = await agent.generate_fix_proposals_with_llm([failure])

        assert result.total_proposals_generated == 1
        proposal = result.proposals[0]
        assert proposal.failure_id == "tests/test_data.py::test_load_csv"
        assert proposal.is_actionable is True
        assert proposal.metadata.get("llm_augmented") is True
        # Should have concrete changes
        assert proposal.change_count >= 1
        # Should mention user action
        assert proposal.requires_user_action is True

    @pytest.mark.asyncio
    async def test_assertion_error_gets_review_fix(self):
        """AssertionError should produce a review-oriented fix suggestion."""
        llm_response = _make_llm_response(
            root_cause="Test expected 42 but code returns 41 due to off-by-one",
            confidence=0.82,
            category="assertion",
            explanation="The function has an off-by-one error in the loop boundary",
            proposed_fixes=[{
                "description": "Fix loop boundary in calculate()",
                "file_path": "src/calc.py",
                "original_snippet": "for i in range(n - 1):",
                "proposed_snippet": "for i in range(n):",
                "change_type": "modify",
            }],
            requires_user_action=False,
        )
        caller = AsyncMock(return_value=llm_response)
        agent = TroubleshooterAgent(
            llm_caller=caller,
            fix_generator_config=FixGeneratorConfig(
                llm_augmentation_threshold=0.99,
            ),
        )

        failure = _make_failure(
            test_id="tests/test_calc.py::test_sum",
            test_name="test_sum",
            error_type="AssertionError",
            error_message="assert 41 == 42",
            file_path="tests/test_calc.py",
            line_number=15,
            stack_trace='assert 41 == 42',
        )

        result = await agent.generate_fix_proposals_with_llm([failure])

        assert result.total_proposals_generated == 1
        proposal = result.proposals[0]
        assert proposal.metadata.get("llm_augmented") is True
        # Should have a code change suggestion
        code_changes = [c for c in proposal.proposed_changes if c.has_diff]
        assert len(code_changes) >= 1

    @pytest.mark.asyncio
    async def test_multiple_failures_produce_per_failure_proposals(self):
        """Each failure should get its own actionable fix proposal."""
        responses = [
            _make_llm_response(
                root_cause="Missing import",
                confidence=0.90,
                category="import_error",
            ),
            _make_llm_response(
                root_cause="Type mismatch",
                confidence=0.75,
                category="type_error",
            ),
        ]
        call_count = 0

        async def _multi_caller(system: str, user: str) -> str:
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return responses[idx]

        agent = TroubleshooterAgent(
            llm_caller=_multi_caller,
            fix_generator_config=FixGeneratorConfig(
                llm_augmentation_threshold=0.99,
                max_llm_calls=10,
            ),
        )

        failures = [
            _make_failure(
                test_id="test_a",
                error_type="ImportError",
                error_message="No module named 'foo'",
                file_path="src/a.py",
                stack_trace="trace",
            ),
            _make_failure(
                test_id="test_b",
                error_type="TypeError",
                error_message="expected str, got int",
                file_path="src/b.py",
                stack_trace="trace",
            ),
        ]

        result = await agent.generate_fix_proposals_with_llm(failures)

        assert result.total_proposals_generated == 2
        ids = {p.failure_id for p in result.proposals}
        assert "test_a" in ids
        assert "test_b" in ids

    @pytest.mark.asyncio
    async def test_proposals_are_diagnose_only(self):
        """Verify proposals are diagnose-only — never auto-execute."""
        agent = TroubleshooterAgent()
        assert agent.auto_fix_enabled is False

        failures = [
            _make_failure(
                test_id="test_safe",
                error_type="SyntaxError",
                error_message="invalid syntax",
                file_path="src/broken.py",
                line_number=5,
                stack_trace="trace",
            ),
        ]

        result = await agent.generate_fix_proposals_with_llm(failures)

        # Proposals exist but are just suggestions
        assert result.total_proposals_generated >= 1
        for p in result.proposals:
            # Proposals have confidence tiers to help user prioritize
            assert p.confidence in (FixConfidence.HIGH, FixConfidence.MEDIUM, FixConfidence.LOW)
            # All have a title and description
            assert p.title
            assert p.description
