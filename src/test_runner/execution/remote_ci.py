"""Remote CI execution targets for triggering and monitoring test runs.

Provides pluggable CI provider backends (GitHub Actions, Jenkins) that
trigger workflows/jobs on remote CI systems and poll for results.

Architecture:
    RemoteCIExecutionTarget  (ExecutionTarget subclass, main entry point)
        -> CIProvider (ABC)
            -> GitHubActionsProvider
            -> JenkinsProvider

Each provider encapsulates the API specifics of a particular CI system.
The RemoteCIExecutionTarget handles the common lifecycle (trigger, poll,
collect artifacts) and delegates CI-specific operations to the provider.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

from test_runner.execution.targets import (
    ExecutionResult,
    ExecutionStatus,
    ExecutionTarget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CI job status model
# ---------------------------------------------------------------------------


class CIJobStatus(str, Enum):
    """Normalized status of a CI job across providers."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"


@dataclass
class CIJobResult:
    """Normalized result from a CI job across providers.

    Attributes:
        job_id: Provider-specific job identifier (run ID, build number, etc.).
        status: Normalized job status.
        conclusion: Provider-specific conclusion string (e.g. 'success', 'failure').
        logs: Captured log output from the CI job.
        artifacts: Dictionary of artifact names to download URLs or content.
        duration_seconds: Wall-clock time from trigger to completion.
        url: Web URL to view the job in the CI system's UI.
        raw_response: Full provider-specific response for debugging.
    """

    job_id: str
    status: CIJobStatus
    conclusion: str = ""
    logs: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0
    url: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == CIJobStatus.COMPLETED and self.conclusion in (
            "success",
            "SUCCESS",
        )


# ---------------------------------------------------------------------------
# CI provider config
# ---------------------------------------------------------------------------


@dataclass
class CIProviderConfig:
    """Configuration for a CI provider connection.

    Attributes:
        base_url: Base URL for the CI system API.
        api_token: Authentication token.
        organization: Organization/owner (GitHub) or folder (Jenkins).
        repository: Repository name (GitHub) or job path (Jenkins).
        default_branch: Branch to trigger workflows against.
        workflow_id: Default workflow file or ID to trigger.
        poll_interval_seconds: How often to poll for job status.
        max_poll_attempts: Maximum number of poll iterations before timeout.
        extra_headers: Additional HTTP headers for API requests.
        verify_ssl: Whether to verify SSL certificates.
    """

    base_url: str = ""
    api_token: str = ""
    organization: str = ""
    repository: str = ""
    default_branch: str = "main"
    workflow_id: str = ""
    poll_interval_seconds: float = 10.0
    max_poll_attempts: int = 180  # 30 min at 10s intervals
    extra_headers: dict[str, str] = field(default_factory=dict)
    verify_ssl: bool = True


# ---------------------------------------------------------------------------
# Abstract CI provider
# ---------------------------------------------------------------------------


class CIProvider(ABC):
    """Abstract base for CI system integrations.

    Each provider implements the trigger-poll-collect lifecycle
    for a specific CI system (GitHub Actions, Jenkins, etc.).
    """

    def __init__(self, config: CIProviderConfig) -> None:
        self._config = config

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name (e.g. 'github-actions', 'jenkins')."""
        ...

    @abstractmethod
    async def trigger_job(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        workflow_id: str = "",
        branch: str = "",
    ) -> str:
        """Trigger a CI job and return its job ID.

        Args:
            command: Test command tokens to execute in the CI job.
            env: Environment variables to inject into the job.
            workflow_id: Override the default workflow/job to trigger.
            branch: Override the default branch.

        Returns:
            Provider-specific job identifier.

        Raises:
            CIProviderError: If the trigger fails.
        """
        ...

    @abstractmethod
    async def get_job_status(self, job_id: str) -> CIJobResult:
        """Query the current status of a CI job.

        Args:
            job_id: The job identifier returned by trigger_job.

        Returns:
            CIJobResult with current status and any available logs.
        """
        ...

    @abstractmethod
    async def get_job_logs(self, job_id: str) -> str:
        """Fetch the full log output of a completed CI job.

        Args:
            job_id: The job identifier.

        Returns:
            String containing the log output.
        """
        ...

    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running CI job.

        Args:
            job_id: The job identifier.

        Returns:
            True if cancellation was successful.
        """
        ...

    async def health_check(self) -> bool:
        """Verify connectivity to the CI system.

        Default implementation returns True. Override for actual checks.
        """
        return True

    async def get_artifacts(self, job_id: str) -> dict[str, str]:
        """Download artifacts from a completed CI job.

        Default returns empty dict. Override for providers that support artifacts.

        Args:
            job_id: The job identifier.

        Returns:
            Dictionary mapping artifact names to their content/paths.
        """
        return {}


class CIProviderError(Exception):
    """Raised when a CI provider operation fails."""

    def __init__(self, message: str, *, provider: str = "", job_id: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.job_id = job_id


# ---------------------------------------------------------------------------
# GitHub Actions provider
# ---------------------------------------------------------------------------


class GitHubActionsProvider(CIProvider):
    """CI provider for GitHub Actions.

    Triggers workflow dispatch events and polls workflow run status via the
    GitHub REST API (v3). Requires a personal access token or GitHub App
    token with ``actions:write`` and ``actions:read`` permissions.

    Example config::

        config = CIProviderConfig(
            base_url="https://api.github.com",
            api_token="ghp_xxxx",
            organization="my-org",
            repository="my-repo",
            workflow_id="tests.yml",
        )
        provider = GitHubActionsProvider(config)
    """

    @property
    def provider_name(self) -> str:
        return "github-actions"

    def _api_url(self, path: str) -> str:
        """Build a GitHub API URL."""
        base = self._config.base_url.rstrip("/") or "https://api.github.com"
        owner = self._config.organization
        repo = self._config.repository
        return f"{base}/repos/{owner}/{repo}{path}"

    def _headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            **self._config.extra_headers,
        }
        if self._config.api_token:
            headers["Authorization"] = f"Bearer {self._config.api_token}"
        return headers

    async def _http_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Make an HTTP request using asyncio-compatible HTTP.

        Uses aiohttp if available, falls back to urllib in a thread executor.
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                kwargs: dict[str, Any] = {
                    "headers": self._headers(),
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                    "ssl": self._config.verify_ssl,
                }
                if json_body is not None:
                    kwargs["json"] = json_body

                async with session.request(method, url, **kwargs) as resp:
                    body = await resp.json(content_type=None)
                    if resp.status >= 400:
                        raise CIProviderError(
                            f"GitHub API error {resp.status}: {body}",
                            provider=self.provider_name,
                        )
                    return body if isinstance(body, dict) else {"data": body}
        except ImportError:
            # Fallback: use urllib in a thread executor
            return await self._urllib_request(method, url, json_body=json_body)

    async def _urllib_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fallback HTTP using urllib (runs in thread executor)."""
        import json
        import urllib.request

        loop = asyncio.get_event_loop()

        def _do_request() -> dict[str, Any]:
            data = json.dumps(json_body).encode() if json_body else None
            req = urllib.request.Request(
                url,
                data=data,
                headers=self._headers(),
                method=method,
            )
            if data:
                req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
                return body if isinstance(body, dict) else {"data": body}

        return await loop.run_in_executor(None, _do_request)

    async def _http_get_text(self, url: str, *, timeout: float = 60.0) -> str:
        """Fetch text content (e.g., logs) from a URL."""
        try:
            import aiohttp

            headers = self._headers()
            headers["Accept"] = "application/vnd.github+json"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    ssl=self._config.verify_ssl,
                ) as resp:
                    return await resp.text()
        except ImportError:
            import urllib.request

            loop = asyncio.get_event_loop()

            def _fetch() -> str:
                req = urllib.request.Request(url, headers=self._headers())
                req.add_header("Accept", "application/vnd.github+json")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read().decode("utf-8", errors="replace")

            return await loop.run_in_executor(None, _fetch)

    async def trigger_job(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        workflow_id: str = "",
        branch: str = "",
    ) -> str:
        """Trigger a GitHub Actions workflow dispatch.

        The command is passed as a workflow input named ``test_command``.
        Environment variables are passed as the ``env_vars`` input (JSON-encoded).
        """
        import json as json_mod

        wf = workflow_id or self._config.workflow_id
        ref = branch or self._config.default_branch

        if not wf:
            raise CIProviderError(
                "No workflow_id configured for GitHub Actions",
                provider=self.provider_name,
            )

        url = self._api_url(f"/actions/workflows/{wf}/dispatches")
        inputs: dict[str, str] = {
            "test_command": " ".join(command),
        }
        if env:
            inputs["env_vars"] = json_mod.dumps(env)

        body = {"ref": ref, "inputs": inputs}

        logger.info(
            "GitHubActions: triggering workflow %s on %s (ref=%s)",
            wf,
            self._config.repository,
            ref,
        )

        await self._http_request("POST", url, json_body=body)

        # workflow_dispatch doesn't return a run ID directly.
        # We need to find the most recent run for this workflow.
        await asyncio.sleep(2)  # Brief delay for run to appear
        run_id = await self._find_latest_run(wf, ref)

        logger.info("GitHubActions: triggered run %s", run_id)
        return run_id

    async def _find_latest_run(self, workflow_id: str, branch: str) -> str:
        """Find the most recently triggered workflow run."""
        url = self._api_url(
            f"/actions/workflows/{workflow_id}/runs"
            f"?branch={branch}&per_page=1&event=workflow_dispatch"
        )
        data = await self._http_request("GET", url)
        runs = data.get("workflow_runs", [])
        if not runs:
            raise CIProviderError(
                "Could not find triggered workflow run",
                provider=self.provider_name,
            )
        return str(runs[0]["id"])

    async def get_job_status(self, job_id: str) -> CIJobResult:
        """Get the status of a GitHub Actions workflow run."""
        url = self._api_url(f"/actions/runs/{job_id}")
        data = await self._http_request("GET", url)

        status_map = {
            "queued": CIJobStatus.QUEUED,
            "in_progress": CIJobStatus.IN_PROGRESS,
            "completed": CIJobStatus.COMPLETED,
        }

        gh_status = data.get("status", "unknown")
        gh_conclusion = data.get("conclusion", "") or ""
        normalized_status = status_map.get(gh_status, CIJobStatus.UNKNOWN)

        # Map conclusion to final status
        if normalized_status == CIJobStatus.COMPLETED:
            if gh_conclusion == "cancelled":
                normalized_status = CIJobStatus.CANCELLED
            elif gh_conclusion == "timed_out":
                normalized_status = CIJobStatus.TIMED_OUT
            elif gh_conclusion not in ("success", "neutral", "skipped"):
                normalized_status = CIJobStatus.FAILED

        return CIJobResult(
            job_id=job_id,
            status=normalized_status,
            conclusion=gh_conclusion,
            url=data.get("html_url", ""),
            raw_response=data,
        )

    async def get_job_logs(self, job_id: str) -> str:
        """Download logs for a GitHub Actions workflow run."""
        url = self._api_url(f"/actions/runs/{job_id}/logs")
        try:
            return await self._http_get_text(url, timeout=120)
        except Exception as exc:
            logger.warning("GitHubActions: failed to fetch logs for %s: %s", job_id, exc)
            return f"[Failed to fetch logs: {exc}]"

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a GitHub Actions workflow run."""
        url = self._api_url(f"/actions/runs/{job_id}/cancel")
        try:
            await self._http_request("POST", url)
            logger.info("GitHubActions: cancelled run %s", job_id)
            return True
        except CIProviderError as exc:
            logger.warning("GitHubActions: failed to cancel %s: %s", job_id, exc)
            return False

    async def get_artifacts(self, job_id: str) -> dict[str, str]:
        """List artifacts from a GitHub Actions workflow run."""
        url = self._api_url(f"/actions/runs/{job_id}/artifacts")
        try:
            data = await self._http_request("GET", url)
            artifacts = {}
            for artifact in data.get("artifacts", []):
                artifacts[artifact["name"]] = artifact.get("archive_download_url", "")
            return artifacts
        except CIProviderError:
            return {}

    async def health_check(self) -> bool:
        """Verify GitHub API connectivity."""
        try:
            url = self._api_url("")
            await self._http_request("GET", url)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Jenkins provider
# ---------------------------------------------------------------------------


class JenkinsProvider(CIProvider):
    """CI provider for Jenkins.

    Triggers builds via the Jenkins REST API and polls for results.
    Requires a Jenkins API token with build and read permissions.

    Example config::

        config = CIProviderConfig(
            base_url="https://jenkins.example.com",
            api_token="user:api_token",
            repository="my-test-job",  # Jenkins job name/path
        )
        provider = JenkinsProvider(config)
    """

    @property
    def provider_name(self) -> str:
        return "jenkins"

    def _job_url(self, path: str = "") -> str:
        """Build a Jenkins job API URL."""
        base = self._config.base_url.rstrip("/")
        job_path = self._config.repository.strip("/")
        # Handle nested folders: a/b/c -> job/a/job/b/job/c
        parts = job_path.split("/")
        jenkins_path = "/".join(f"job/{p}" for p in parts)
        return f"{base}/{jenkins_path}{path}"

    def _headers(self) -> dict[str, str]:
        """Build request headers with authentication."""
        import base64

        headers = {
            "Content-Type": "application/json",
            **self._config.extra_headers,
        }
        if self._config.api_token:
            # Jenkins uses Basic auth with user:token
            encoded = base64.b64encode(
                self._config.api_token.encode()
            ).decode()
            headers["Authorization"] = f"Basic {encoded}"
        return headers

    async def _http_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
        expect_json: bool = True,
    ) -> dict[str, Any]:
        """Make an HTTP request to Jenkins API."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                kwargs: dict[str, Any] = {
                    "headers": self._headers(),
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                    "ssl": self._config.verify_ssl,
                }
                if json_body is not None:
                    kwargs["json"] = json_body

                async with session.request(method, url, **kwargs) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        raise CIProviderError(
                            f"Jenkins API error {resp.status}: {text}",
                            provider=self.provider_name,
                        )
                    # Jenkins queue responses return location header
                    if resp.status == 201:
                        location = resp.headers.get("Location", "")
                        return {"location": location, "status": resp.status}
                    if expect_json:
                        try:
                            return await resp.json(content_type=None)
                        except Exception:
                            text = await resp.text()
                            return {"text": text, "status": resp.status}
                    text = await resp.text()
                    return {"text": text, "status": resp.status}
        except ImportError:
            return await self._urllib_request(
                method, url, json_body=json_body, expect_json=expect_json
            )

    async def _urllib_request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        expect_json: bool = True,
    ) -> dict[str, Any]:
        """Fallback HTTP using urllib."""
        import json
        import urllib.request

        loop = asyncio.get_event_loop()

        def _do_request() -> dict[str, Any]:
            data = json.dumps(json_body).encode() if json_body else None
            req = urllib.request.Request(
                url,
                data=data,
                headers=self._headers(),
                method=method,
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if expect_json:
                    try:
                        return json.loads(resp.read().decode())
                    except Exception:
                        return {"text": resp.read().decode(), "status": resp.status}
                return {"text": resp.read().decode(), "status": resp.status}

        return await loop.run_in_executor(None, _do_request)

    async def trigger_job(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        workflow_id: str = "",
        branch: str = "",
    ) -> str:
        """Trigger a Jenkins build with parameters.

        The test command is passed as a ``TEST_COMMAND`` build parameter.
        """
        import json as json_mod
        import urllib.parse

        # Build parameters
        params: dict[str, str] = {
            "TEST_COMMAND": " ".join(command),
        }
        if branch:
            params["BRANCH"] = branch
        if env:
            params["ENV_VARS"] = json_mod.dumps(env)

        # Jenkins parameterized build URL
        param_string = "&".join(
            f"{urllib.parse.quote(k)}={urllib.parse.quote(v)}"
            for k, v in params.items()
        )
        url = self._job_url(f"/buildWithParameters?{param_string}")

        logger.info(
            "Jenkins: triggering build for %s",
            self._config.repository,
        )

        response = await self._http_request("POST", url, expect_json=False)

        # Jenkins returns a queue item location
        queue_url = response.get("location", "")
        if queue_url:
            build_number = await self._resolve_queue_item(queue_url)
            logger.info("Jenkins: triggered build #%s", build_number)
            return build_number

        # Fallback: get latest build number
        await asyncio.sleep(2)
        build_number = await self._get_latest_build_number()
        logger.info("Jenkins: triggered build #%s (via latest lookup)", build_number)
        return build_number

    async def _resolve_queue_item(self, queue_url: str) -> str:
        """Poll a Jenkins queue item until it yields a build number."""
        api_url = queue_url.rstrip("/") + "/api/json"

        for _ in range(30):  # Max 5 minutes in queue
            try:
                data = await self._http_request("GET", api_url)
                executable = data.get("executable")
                if executable and "number" in executable:
                    return str(executable["number"])
            except CIProviderError:
                pass
            await asyncio.sleep(10)

        raise CIProviderError(
            "Timed out waiting for Jenkins queue item to resolve",
            provider=self.provider_name,
        )

    async def _get_latest_build_number(self) -> str:
        """Get the latest build number for the configured job."""
        url = self._job_url("/lastBuild/api/json")
        data = await self._http_request("GET", url)
        return str(data.get("number", "0"))

    async def get_job_status(self, job_id: str) -> CIJobResult:
        """Get the status of a Jenkins build."""
        url = self._job_url(f"/{job_id}/api/json")
        data = await self._http_request("GET", url)

        building = data.get("building", False)
        result_str = data.get("result") or ""

        if building:
            status = CIJobStatus.IN_PROGRESS
        elif result_str == "SUCCESS":
            status = CIJobStatus.COMPLETED
        elif result_str == "ABORTED":
            status = CIJobStatus.CANCELLED
        elif result_str in ("FAILURE", "UNSTABLE"):
            status = CIJobStatus.FAILED
        else:
            status = CIJobStatus.UNKNOWN

        duration_ms = data.get("duration", 0)

        return CIJobResult(
            job_id=job_id,
            status=status,
            conclusion=result_str.lower() if result_str else "",
            duration_seconds=duration_ms / 1000.0,
            url=data.get("url", ""),
            raw_response=data,
        )

    async def get_job_logs(self, job_id: str) -> str:
        """Fetch console output of a Jenkins build."""
        url = self._job_url(f"/{job_id}/consoleText")
        try:
            data = await self._http_request(
                "GET", url, expect_json=False
            )
            return data.get("text", "")
        except CIProviderError as exc:
            logger.warning("Jenkins: failed to fetch logs for #%s: %s", job_id, exc)
            return f"[Failed to fetch logs: {exc}]"

    async def cancel_job(self, job_id: str) -> bool:
        """Stop/abort a Jenkins build."""
        url = self._job_url(f"/{job_id}/stop")
        try:
            await self._http_request("POST", url, expect_json=False)
            logger.info("Jenkins: stopped build #%s", job_id)
            return True
        except CIProviderError as exc:
            logger.warning("Jenkins: failed to stop #%s: %s", job_id, exc)
            return False

    async def health_check(self) -> bool:
        """Verify Jenkins API connectivity."""
        try:
            url = self._job_url("/api/json")
            await self._http_request("GET", url)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# RemoteCIExecutionTarget — the main ExecutionTarget implementation
# ---------------------------------------------------------------------------


class RemoteCIExecutionTarget(ExecutionTarget):
    """Executes test commands on remote CI systems.

    This target triggers CI jobs (GitHub Actions workflows, Jenkins builds, etc.),
    polls for completion, and collects results including logs and artifacts.

    The target delegates to a pluggable CIProvider for CI-system-specific
    operations, keeping the polling and lifecycle logic generic.

    Lifecycle:
        1. setup() — verify CI connectivity
        2. execute() — trigger job, poll status, collect logs
        3. collect_results() — download artifacts, enrich metadata
        4. teardown() — cancel any in-flight jobs

    Example::

        config = CIProviderConfig(
            base_url="https://api.github.com",
            api_token="ghp_xxxx",
            organization="my-org",
            repository="my-repo",
            workflow_id="tests.yml",
        )
        target = RemoteCIExecutionTarget(
            provider=GitHubActionsProvider(config),
        )
        await target.setup()
        result = await target.execute(["pytest", "-v", "tests/"])
        await target.teardown()
    """

    def __init__(
        self,
        provider: CIProvider,
        *,
        poll_interval: float | None = None,
        max_poll_attempts: int | None = None,
    ) -> None:
        self._provider = provider
        self._poll_interval = (
            poll_interval
            if poll_interval is not None
            else provider._config.poll_interval_seconds
        )
        self._max_poll_attempts = (
            max_poll_attempts
            if max_poll_attempts is not None
            else provider._config.max_poll_attempts
        )
        self._active_jobs: list[str] = []
        self._setup_complete = False

    @property
    def name(self) -> str:
        return f"remote-ci:{self._provider.provider_name}"

    # ------------------------------------------------------------------
    # Lifecycle: setup
    # ------------------------------------------------------------------

    async def setup(
        self,
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
    ) -> None:
        """Verify CI system connectivity."""
        logger.info(
            "RemoteCITarget (%s): verifying connectivity",
            self._provider.provider_name,
        )
        healthy = await self._provider.health_check()
        if not healthy:
            logger.warning(
                "RemoteCITarget (%s): health check failed — "
                "execution may fail",
                self._provider.provider_name,
            )
        self._setup_complete = True
        logger.info(
            "RemoteCITarget (%s): setup complete (healthy=%s)",
            self._provider.provider_name,
            healthy,
        )

    # ------------------------------------------------------------------
    # Lifecycle: execute — trigger, poll, collect logs
    # ------------------------------------------------------------------

    async def execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Trigger a CI job, poll for completion, and return results.

        Args:
            command: Test command tokens.
            working_directory: Not directly used (CI runs in its own env).
            env: Environment variables to inject into the CI job.
            timeout: Overall timeout in seconds. Overrides poll-based timeout.

        Returns:
            ExecutionResult with CI job logs and metadata.
        """
        start = time.monotonic()
        command_display = " ".join(command)

        # Calculate effective poll limits
        if timeout is not None:
            effective_max_polls = max(
                1, int(timeout / max(self._poll_interval, 1))
            )
        else:
            effective_max_polls = self._max_poll_attempts

        # --- Phase 1: Trigger ---
        try:
            job_id = await self._provider.trigger_job(command, env=env)
            self._active_jobs.append(job_id)
        except CIProviderError as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "RemoteCITarget: failed to trigger job: %s", exc
            )
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                exit_code=-1,
                stdout="",
                stderr=f"Failed to trigger CI job: {exc}",
                duration_seconds=elapsed,
                command_display=command_display,
                metadata={
                    "target": self.name,
                    "provider": self._provider.provider_name,
                    "phase": "trigger",
                },
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "RemoteCITarget: unexpected error triggering job: %s", exc
            )
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                exit_code=-1,
                stdout="",
                stderr=f"Unexpected error triggering CI job: {exc}",
                duration_seconds=elapsed,
                command_display=command_display,
                metadata={
                    "target": self.name,
                    "provider": self._provider.provider_name,
                    "phase": "trigger",
                },
            )

        # --- Phase 2: Poll ---
        logger.info(
            "RemoteCITarget: polling job %s (interval=%.1fs, max_polls=%d)",
            job_id,
            self._poll_interval,
            effective_max_polls,
        )

        ci_result: CIJobResult | None = None
        for poll_num in range(1, effective_max_polls + 1):
            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    logger.warning(
                        "RemoteCITarget: timeout after %.1fs", elapsed
                    )
                    await self._provider.cancel_job(job_id)
                    return ExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        exit_code=-1,
                        stdout="",
                        stderr=f"CI job timed out after {elapsed:.0f}s",
                        duration_seconds=elapsed,
                        command_display=command_display,
                        metadata={
                            "target": self.name,
                            "provider": self._provider.provider_name,
                            "job_id": job_id,
                            "phase": "poll",
                            "polls": poll_num,
                        },
                    )

            try:
                ci_result = await self._provider.get_job_status(job_id)
            except CIProviderError as exc:
                logger.warning(
                    "RemoteCITarget: poll error (attempt %d): %s",
                    poll_num,
                    exc,
                )
                await asyncio.sleep(self._poll_interval)
                continue

            # Check terminal states
            if ci_result.status in (
                CIJobStatus.COMPLETED,
                CIJobStatus.FAILED,
                CIJobStatus.CANCELLED,
                CIJobStatus.TIMED_OUT,
            ):
                logger.info(
                    "RemoteCITarget: job %s finished with status=%s conclusion=%s",
                    job_id,
                    ci_result.status.value,
                    ci_result.conclusion,
                )
                break

            # Log progress
            if poll_num % 6 == 0:  # Every ~60s at 10s interval
                logger.info(
                    "RemoteCITarget: job %s still %s (poll %d/%d)",
                    job_id,
                    ci_result.status.value,
                    poll_num,
                    effective_max_polls,
                )

            await asyncio.sleep(self._poll_interval)
        else:
            # Exhausted all poll attempts
            elapsed = time.monotonic() - start
            logger.warning(
                "RemoteCITarget: max poll attempts exhausted for job %s",
                job_id,
            )
            await self._provider.cancel_job(job_id)
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                exit_code=-1,
                stdout="",
                stderr=f"CI job polling exhausted after {effective_max_polls} attempts",
                duration_seconds=elapsed,
                command_display=command_display,
                metadata={
                    "target": self.name,
                    "provider": self._provider.provider_name,
                    "job_id": job_id,
                    "phase": "poll_exhausted",
                    "polls": effective_max_polls,
                },
            )

        # --- Phase 3: Collect logs ---
        logs = ""
        try:
            logs = await self._provider.get_job_logs(job_id)
        except Exception as exc:
            logger.warning(
                "RemoteCITarget: failed to fetch logs for %s: %s",
                job_id,
                exc,
            )
            logs = f"[Failed to fetch logs: {exc}]"

        elapsed = time.monotonic() - start

        # Remove from active jobs
        if job_id in self._active_jobs:
            self._active_jobs.remove(job_id)

        # Map CI result to ExecutionResult
        assert ci_result is not None
        exec_status = self._map_ci_status(ci_result)

        return ExecutionResult(
            status=exec_status,
            exit_code=0 if ci_result.success else 1,
            stdout=logs,
            stderr="" if ci_result.success else f"CI job {ci_result.conclusion}",
            duration_seconds=elapsed,
            command_display=command_display,
            metadata={
                "target": self.name,
                "provider": self._provider.provider_name,
                "job_id": job_id,
                "job_url": ci_result.url,
                "conclusion": ci_result.conclusion,
                "ci_status": ci_result.status.value,
                "ci_duration_seconds": ci_result.duration_seconds,
            },
        )

    def _map_ci_status(self, ci_result: CIJobResult) -> ExecutionStatus:
        """Map a CIJobResult to an ExecutionStatus."""
        if ci_result.success:
            return ExecutionStatus.PASSED
        status_map = {
            CIJobStatus.FAILED: ExecutionStatus.FAILED,
            CIJobStatus.CANCELLED: ExecutionStatus.SKIPPED,
            CIJobStatus.TIMED_OUT: ExecutionStatus.TIMEOUT,
        }
        return status_map.get(ci_result.status, ExecutionStatus.ERROR)

    # ------------------------------------------------------------------
    # Lifecycle: collect_results
    # ------------------------------------------------------------------

    async def collect_results(
        self,
        result: ExecutionResult,
        *,
        working_directory: str = "",
    ) -> ExecutionResult:
        """Enrich results with CI artifacts."""
        job_id = result.metadata.get("job_id")
        if job_id:
            try:
                artifacts = await self._provider.get_artifacts(job_id)
                if artifacts:
                    result.metadata["artifacts"] = artifacts
            except Exception as exc:
                logger.warning(
                    "RemoteCITarget: failed to collect artifacts: %s", exc
                )
        result.metadata.setdefault("target", self.name)
        return result

    # ------------------------------------------------------------------
    # Lifecycle: teardown
    # ------------------------------------------------------------------

    async def teardown(self) -> None:
        """Cancel any in-flight CI jobs and clean up."""
        logger.info(
            "RemoteCITarget (%s): tearing down (%d active jobs)",
            self._provider.provider_name,
            len(self._active_jobs),
        )
        for job_id in list(self._active_jobs):
            try:
                await self._provider.cancel_job(job_id)
                logger.info("RemoteCITarget: cancelled job %s", job_id)
            except Exception as exc:
                logger.warning(
                    "RemoteCITarget: failed to cancel job %s: %s",
                    job_id,
                    exc,
                )
        self._active_jobs.clear()
        self._setup_complete = False
        logger.info("RemoteCITarget: teardown complete")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream CI job progress by polling status and yielding updates.

        Since CI jobs run remotely, true line-by-line streaming isn't possible.
        Instead, yields periodic status updates and final logs.
        """
        command_display = " ".join(command)
        yield f"[ci] Triggering {self._provider.provider_name} job: {command_display}"

        try:
            job_id = await self._provider.trigger_job(command, env=env)
            self._active_jobs.append(job_id)
            yield f"[ci] Job triggered: {job_id}"
        except Exception as exc:
            yield f"[ci:error] Failed to trigger job: {exc}"
            return

        # Poll for completion
        effective_max_polls = self._max_poll_attempts
        if timeout is not None:
            effective_max_polls = max(
                1, int(timeout / max(self._poll_interval, 1))
            )

        for poll_num in range(1, effective_max_polls + 1):
            await asyncio.sleep(self._poll_interval)
            try:
                ci_result = await self._provider.get_job_status(job_id)
            except Exception as exc:
                yield f"[ci:warning] Poll error: {exc}"
                continue

            if ci_result.status in (
                CIJobStatus.COMPLETED,
                CIJobStatus.FAILED,
                CIJobStatus.CANCELLED,
                CIJobStatus.TIMED_OUT,
            ):
                yield f"[ci] Job {job_id} finished: {ci_result.status.value} ({ci_result.conclusion})"
                if ci_result.url:
                    yield f"[ci] View: {ci_result.url}"

                # Fetch and stream logs
                try:
                    logs = await self._provider.get_job_logs(job_id)
                    for line in logs.splitlines():
                        yield line
                except Exception as exc:
                    yield f"[ci:warning] Failed to fetch logs: {exc}"

                if job_id in self._active_jobs:
                    self._active_jobs.remove(job_id)
                return

            if poll_num % 3 == 0:
                yield f"[ci] Job {job_id}: {ci_result.status.value} (poll {poll_num}/{effective_max_polls})"

        yield f"[ci:timeout] Polling exhausted after {effective_max_polls} attempts"
        await self._provider.cancel_job(job_id)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check CI provider connectivity."""
        return await self._provider.health_check()
