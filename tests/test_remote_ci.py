"""Tests for RemoteCIExecutionTarget, CIProviders, and related models."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from test_runner.execution.remote_ci import (
    CIJobResult,
    CIJobStatus,
    CIProvider,
    CIProviderConfig,
    CIProviderError,
    GitHubActionsProvider,
    JenkinsProvider,
    RemoteCIExecutionTarget,
)
from test_runner.execution.targets import ExecutionResult, ExecutionStatus


# ---------------------------------------------------------------------------
# CIJobResult model tests
# ---------------------------------------------------------------------------


class TestCIJobResult:
    def test_success_when_completed_and_success(self):
        r = CIJobResult(job_id="1", status=CIJobStatus.COMPLETED, conclusion="success")
        assert r.success is True

    def test_not_success_when_failed(self):
        r = CIJobResult(job_id="1", status=CIJobStatus.FAILED, conclusion="failure")
        assert r.success is False

    def test_not_success_when_completed_with_failure(self):
        r = CIJobResult(
            job_id="1", status=CIJobStatus.COMPLETED, conclusion="failure"
        )
        assert r.success is False

    def test_defaults(self):
        r = CIJobResult(job_id="42", status=CIJobStatus.QUEUED)
        assert r.conclusion == ""
        assert r.logs == ""
        assert r.artifacts == {}
        assert r.duration_seconds == 0.0
        assert r.url == ""
        assert r.raw_response == {}


# ---------------------------------------------------------------------------
# CIProviderConfig tests
# ---------------------------------------------------------------------------


class TestCIProviderConfig:
    def test_defaults(self):
        c = CIProviderConfig()
        assert c.base_url == ""
        assert c.api_token == ""
        assert c.default_branch == "main"
        assert c.poll_interval_seconds == 10.0
        assert c.max_poll_attempts == 180
        assert c.verify_ssl is True

    def test_custom_values(self):
        c = CIProviderConfig(
            base_url="https://ci.example.com",
            api_token="tok_123",
            organization="my-org",
            repository="my-repo",
            default_branch="develop",
            workflow_id="test.yml",
            poll_interval_seconds=5.0,
            max_poll_attempts=60,
        )
        assert c.base_url == "https://ci.example.com"
        assert c.api_token == "tok_123"
        assert c.organization == "my-org"
        assert c.repository == "my-repo"
        assert c.default_branch == "develop"
        assert c.workflow_id == "test.yml"
        assert c.poll_interval_seconds == 5.0
        assert c.max_poll_attempts == 60


# ---------------------------------------------------------------------------
# Mock CI provider for testing the target
# ---------------------------------------------------------------------------


class MockCIProvider(CIProvider):
    """Test double for CIProvider."""

    def __init__(
        self,
        config: CIProviderConfig | None = None,
        *,
        trigger_job_id: str = "mock-job-123",
        statuses: list[CIJobResult] | None = None,
        logs: str = "test output\nall passed",
        trigger_error: Exception | None = None,
        health: bool = True,
    ) -> None:
        super().__init__(config or CIProviderConfig())
        self._trigger_job_id = trigger_job_id
        self._statuses = statuses or [
            CIJobResult(
                job_id=trigger_job_id,
                status=CIJobStatus.COMPLETED,
                conclusion="success",
                url="https://ci.example.com/runs/123",
            )
        ]
        self._status_idx = 0
        self._logs = logs
        self._trigger_error = trigger_error
        self._health = health
        self.trigger_calls: list[tuple[list[str], dict]] = []
        self.cancel_calls: list[str] = []
        self.log_calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock-ci"

    async def trigger_job(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        workflow_id: str = "",
        branch: str = "",
    ) -> str:
        self.trigger_calls.append((command, {"env": env, "workflow_id": workflow_id, "branch": branch}))
        if self._trigger_error:
            raise self._trigger_error
        return self._trigger_job_id

    async def get_job_status(self, job_id: str) -> CIJobResult:
        if self._status_idx < len(self._statuses):
            result = self._statuses[self._status_idx]
            self._status_idx += 1
            return result
        return self._statuses[-1]

    async def get_job_logs(self, job_id: str) -> str:
        self.log_calls.append(job_id)
        return self._logs

    async def cancel_job(self, job_id: str) -> bool:
        self.cancel_calls.append(job_id)
        return True

    async def health_check(self) -> bool:
        return self._health

    async def get_artifacts(self, job_id: str) -> dict[str, str]:
        return {"test-report": "https://ci.example.com/artifacts/report.xml"}


# ---------------------------------------------------------------------------
# RemoteCIExecutionTarget tests
# ---------------------------------------------------------------------------


class TestRemoteCIExecutionTarget:
    """Tests for the main RemoteCIExecutionTarget class."""

    @pytest.mark.asyncio
    async def test_name_includes_provider(self):
        provider = MockCIProvider()
        target = RemoteCIExecutionTarget(provider=provider)
        assert "mock-ci" in target.name
        assert target.name == "remote-ci:mock-ci"

    @pytest.mark.asyncio
    async def test_setup_runs_health_check(self):
        provider = MockCIProvider(health=True)
        target = RemoteCIExecutionTarget(provider=provider)
        await target.setup()
        assert target._setup_complete is True

    @pytest.mark.asyncio
    async def test_setup_with_unhealthy_provider(self):
        provider = MockCIProvider(health=False)
        target = RemoteCIExecutionTarget(provider=provider)
        await target.setup()
        # Setup completes but logs a warning
        assert target._setup_complete is True

    @pytest.mark.asyncio
    async def test_execute_success(self):
        provider = MockCIProvider(
            statuses=[
                CIJobResult(
                    job_id="mock-job-123",
                    status=CIJobStatus.COMPLETED,
                    conclusion="success",
                    url="https://ci.example.com/runs/123",
                )
            ],
            logs="PASSED: 5 tests",
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        result = await target.execute(["pytest", "-v", "tests/"])

        assert result.success
        assert result.status == ExecutionStatus.PASSED
        assert result.exit_code == 0
        assert "PASSED: 5 tests" in result.stdout
        assert result.command_display == "pytest -v tests/"
        assert result.metadata["provider"] == "mock-ci"
        assert result.metadata["job_id"] == "mock-job-123"
        assert result.metadata["job_url"] == "https://ci.example.com/runs/123"

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        provider = MockCIProvider(
            statuses=[
                CIJobResult(
                    job_id="mock-job-123",
                    status=CIJobStatus.FAILED,
                    conclusion="failure",
                )
            ],
            logs="FAILED: 2 tests",
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        result = await target.execute(["pytest", "tests/"])

        assert not result.success
        assert result.status == ExecutionStatus.FAILED
        assert result.exit_code == 1
        assert "FAILED: 2 tests" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_with_polling(self):
        """Test that the target polls through in_progress statuses."""
        provider = MockCIProvider(
            statuses=[
                CIJobResult(job_id="j1", status=CIJobStatus.QUEUED),
                CIJobResult(job_id="j1", status=CIJobStatus.IN_PROGRESS),
                CIJobResult(
                    job_id="j1",
                    status=CIJobStatus.COMPLETED,
                    conclusion="success",
                ),
            ],
            logs="all done",
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        result = await target.execute(["npm", "test"])

        assert result.success
        assert result.status == ExecutionStatus.PASSED
        assert "all done" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_trigger_error(self):
        provider = MockCIProvider(
            trigger_error=CIProviderError("auth failed", provider="mock-ci")
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        result = await target.execute(["pytest"])

        assert result.status == ExecutionStatus.ERROR
        assert "auth failed" in result.stderr
        assert result.metadata["phase"] == "trigger"

    @pytest.mark.asyncio
    async def test_execute_trigger_unexpected_error(self):
        provider = MockCIProvider(
            trigger_error=RuntimeError("network down")
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        result = await target.execute(["pytest"])

        assert result.status == ExecutionStatus.ERROR
        assert "network down" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test that execute respects timeout and cancels the job."""
        # Provider returns in_progress forever
        provider = MockCIProvider(
            statuses=[
                CIJobResult(job_id="j1", status=CIJobStatus.IN_PROGRESS),
            ]
            * 100,
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01, max_poll_attempts=5
        )

        result = await target.execute(["pytest"], timeout=1)

        assert result.status == ExecutionStatus.TIMEOUT
        assert result.metadata["job_id"] == "mock-job-123"

    @pytest.mark.asyncio
    async def test_execute_poll_exhausted(self):
        """Test behavior when max poll attempts are exhausted."""
        provider = MockCIProvider(
            statuses=[
                CIJobResult(job_id="j1", status=CIJobStatus.IN_PROGRESS),
            ]
            * 10,
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01, max_poll_attempts=3
        )

        result = await target.execute(["pytest"])

        assert result.status == ExecutionStatus.TIMEOUT
        assert "exhausted" in result.stderr.lower()
        # Should have attempted to cancel
        assert len(provider.cancel_calls) > 0

    @pytest.mark.asyncio
    async def test_execute_cancelled_job(self):
        provider = MockCIProvider(
            statuses=[
                CIJobResult(
                    job_id="j1",
                    status=CIJobStatus.CANCELLED,
                    conclusion="cancelled",
                )
            ],
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        result = await target.execute(["pytest"])

        assert result.status == ExecutionStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_execute_passes_env(self):
        provider = MockCIProvider()
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        await target.execute(
            ["pytest"], env={"CI": "true", "TOKEN": "secret"}
        )

        assert len(provider.trigger_calls) == 1
        _, kwargs = provider.trigger_calls[0]
        assert kwargs["env"] == {"CI": "true", "TOKEN": "secret"}

    @pytest.mark.asyncio
    async def test_collect_results_enriches_with_artifacts(self):
        provider = MockCIProvider()
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        result = ExecutionResult(
            status=ExecutionStatus.PASSED,
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.0,
            metadata={"job_id": "mock-job-123"},
        )
        enriched = await target.collect_results(result)
        assert "artifacts" in enriched.metadata
        assert "test-report" in enriched.metadata["artifacts"]

    @pytest.mark.asyncio
    async def test_teardown_cancels_active_jobs(self):
        provider = MockCIProvider(
            statuses=[
                CIJobResult(job_id="j1", status=CIJobStatus.IN_PROGRESS),
            ]
            * 100,
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01, max_poll_attempts=2
        )

        # Start a job that won't complete
        await target.execute(["pytest"])
        # Add a manual active job
        target._active_jobs.append("manual-job-456")

        await target.teardown()

        assert "manual-job-456" in provider.cancel_calls
        assert target._active_jobs == []
        assert target._setup_complete is False

    @pytest.mark.asyncio
    async def test_health_check_delegates_to_provider(self):
        provider = MockCIProvider(health=True)
        target = RemoteCIExecutionTarget(provider=provider)
        assert await target.health_check() is True

        provider2 = MockCIProvider(health=False)
        target2 = RemoteCIExecutionTarget(provider=provider2)
        assert await target2.health_check() is False

    @pytest.mark.asyncio
    async def test_stream_execute_yields_updates(self):
        provider = MockCIProvider(
            statuses=[
                CIJobResult(
                    job_id="j1",
                    status=CIJobStatus.COMPLETED,
                    conclusion="success",
                    url="https://ci.example.com/runs/1",
                )
            ],
            logs="line1\nline2\nline3",
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        lines = []
        async for line in target.stream_execute(["pytest"]):
            lines.append(line)

        # Should have trigger message, job ID, completion, URL, and log lines
        assert any("Triggering" in l for l in lines)
        assert any("triggered" in l.lower() for l in lines)
        assert any("finished" in l.lower() for l in lines)
        assert "line1" in lines
        assert "line2" in lines
        assert "line3" in lines

    @pytest.mark.asyncio
    async def test_stream_execute_trigger_error(self):
        provider = MockCIProvider(
            trigger_error=CIProviderError("boom", provider="mock-ci")
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        lines = []
        async for line in target.stream_execute(["pytest"]):
            lines.append(line)

        assert any("error" in l.lower() and "boom" in l for l in lines)

    @pytest.mark.asyncio
    async def test_execute_metadata_completeness(self):
        """Verify all expected metadata fields are present."""
        provider = MockCIProvider(
            statuses=[
                CIJobResult(
                    job_id="j1",
                    status=CIJobStatus.COMPLETED,
                    conclusion="success",
                    url="https://ci.example.com/runs/1",
                    duration_seconds=42.5,
                )
            ]
        )
        target = RemoteCIExecutionTarget(
            provider=provider, poll_interval=0.01
        )

        result = await target.execute(["pytest"])

        assert result.metadata["target"] == "remote-ci:mock-ci"
        assert result.metadata["provider"] == "mock-ci"
        assert result.metadata["job_id"] == "mock-job-123"
        assert result.metadata["conclusion"] == "success"
        assert result.metadata["ci_status"] == "completed"
        assert result.metadata["ci_duration_seconds"] == 42.5


# ---------------------------------------------------------------------------
# GitHubActionsProvider tests
# ---------------------------------------------------------------------------


class TestGitHubActionsProvider:
    def _make_provider(self, **kwargs) -> GitHubActionsProvider:
        config = CIProviderConfig(
            base_url="https://api.github.com",
            api_token="ghp_test123",
            organization="test-org",
            repository="test-repo",
            workflow_id="tests.yml",
            **kwargs,
        )
        return GitHubActionsProvider(config)

    def test_provider_name(self):
        p = self._make_provider()
        assert p.provider_name == "github-actions"

    def test_api_url(self):
        p = self._make_provider()
        url = p._api_url("/actions/runs/123")
        assert url == "https://api.github.com/repos/test-org/test-repo/actions/runs/123"

    def test_headers_include_auth(self):
        p = self._make_provider()
        headers = p._headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer ghp_test123"
        assert "X-GitHub-Api-Version" in headers

    def test_headers_without_token(self):
        config = CIProviderConfig(
            base_url="https://api.github.com",
            organization="org",
            repository="repo",
        )
        p = GitHubActionsProvider(config)
        headers = p._headers()
        assert "Authorization" not in headers

    def test_headers_with_extra_headers(self):
        p = self._make_provider(extra_headers={"X-Custom": "value"})
        headers = p._headers()
        assert headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_trigger_job_requires_workflow_id(self):
        config = CIProviderConfig(
            base_url="https://api.github.com",
            api_token="ghp_test",
            organization="org",
            repository="repo",
            workflow_id="",  # No workflow
        )
        p = GitHubActionsProvider(config)
        with pytest.raises(CIProviderError, match="No workflow_id"):
            await p.trigger_job(["pytest"])


# ---------------------------------------------------------------------------
# JenkinsProvider tests
# ---------------------------------------------------------------------------


class TestJenkinsProvider:
    def _make_provider(self, **kwargs) -> JenkinsProvider:
        config = CIProviderConfig(
            base_url="https://jenkins.example.com",
            api_token="user:api_token",
            repository="my-test-job",
            **kwargs,
        )
        return JenkinsProvider(config)

    def test_provider_name(self):
        p = self._make_provider()
        assert p.provider_name == "jenkins"

    def test_job_url_simple(self):
        p = self._make_provider()
        url = p._job_url("/api/json")
        assert url == "https://jenkins.example.com/job/my-test-job/api/json"

    def test_job_url_nested(self):
        config = CIProviderConfig(
            base_url="https://jenkins.example.com",
            api_token="user:api_token",
            repository="folder/subfolder/job",
        )
        p = JenkinsProvider(config)
        url = p._job_url("/api/json")
        assert url == "https://jenkins.example.com/job/folder/job/subfolder/job/job/api/json"

    def test_job_url_build(self):
        p = self._make_provider()
        url = p._job_url("/42/api/json")
        assert url == "https://jenkins.example.com/job/my-test-job/42/api/json"

    def test_headers_include_basic_auth(self):
        p = self._make_provider()
        headers = p._headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")

    def test_headers_without_token(self):
        config = CIProviderConfig(
            base_url="https://jenkins.example.com",
            repository="job",
        )
        p = JenkinsProvider(config)
        headers = p._headers()
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# CIProviderError tests
# ---------------------------------------------------------------------------


class TestCIProviderError:
    def test_basic(self):
        err = CIProviderError("something broke")
        assert str(err) == "something broke"
        assert err.provider == ""
        assert err.job_id == ""

    def test_with_context(self):
        err = CIProviderError(
            "auth failed",
            provider="github-actions",
            job_id="123",
        )
        assert err.provider == "github-actions"
        assert err.job_id == "123"


# ---------------------------------------------------------------------------
# Integration: target registered in TargetRegistry
# ---------------------------------------------------------------------------


class TestRemoteCITargetRegistration:
    def test_register_in_registry(self):
        from test_runner.execution.targets import TargetRegistry

        registry = TargetRegistry()
        provider = MockCIProvider()
        target = RemoteCIExecutionTarget(provider=provider)
        registry.register(target)

        assert "remote-ci:mock-ci" in registry.available_targets
        assert registry.get("remote-ci:mock-ci") is target

    def test_multiple_providers_registered(self):
        from test_runner.execution.targets import TargetRegistry

        registry = TargetRegistry()

        gh_config = CIProviderConfig(
            base_url="https://api.github.com",
            organization="org",
            repository="repo",
            workflow_id="test.yml",
        )
        gh_target = RemoteCIExecutionTarget(
            provider=GitHubActionsProvider(gh_config)
        )

        jenkins_config = CIProviderConfig(
            base_url="https://jenkins.example.com",
            repository="my-job",
        )
        jenkins_target = RemoteCIExecutionTarget(
            provider=JenkinsProvider(jenkins_config)
        )

        registry.register(gh_target)
        registry.register(jenkins_target)

        assert "remote-ci:github-actions" in registry.available_targets
        assert "remote-ci:jenkins" in registry.available_targets


# ---------------------------------------------------------------------------
# Status mapping tests
# ---------------------------------------------------------------------------


class TestStatusMapping:
    @pytest.mark.asyncio
    async def test_success_maps_to_passed(self):
        provider = MockCIProvider(
            statuses=[
                CIJobResult(job_id="1", status=CIJobStatus.COMPLETED, conclusion="success")
            ]
        )
        target = RemoteCIExecutionTarget(provider=provider, poll_interval=0.01)
        result = await target.execute(["test"])
        assert result.status == ExecutionStatus.PASSED

    @pytest.mark.asyncio
    async def test_failure_maps_to_failed(self):
        provider = MockCIProvider(
            statuses=[
                CIJobResult(job_id="1", status=CIJobStatus.FAILED, conclusion="failure")
            ]
        )
        target = RemoteCIExecutionTarget(provider=provider, poll_interval=0.01)
        result = await target.execute(["test"])
        assert result.status == ExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_cancelled_maps_to_skipped(self):
        provider = MockCIProvider(
            statuses=[
                CIJobResult(job_id="1", status=CIJobStatus.CANCELLED, conclusion="cancelled")
            ]
        )
        target = RemoteCIExecutionTarget(provider=provider, poll_interval=0.01)
        result = await target.execute(["test"])
        assert result.status == ExecutionStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_timed_out_maps_to_timeout(self):
        provider = MockCIProvider(
            statuses=[
                CIJobResult(job_id="1", status=CIJobStatus.TIMED_OUT, conclusion="timed_out")
            ]
        )
        target = RemoteCIExecutionTarget(provider=provider, poll_interval=0.01)
        result = await target.execute(["test"])
        assert result.status == ExecutionStatus.TIMEOUT
