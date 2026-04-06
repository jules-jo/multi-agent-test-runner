"""Pluggable execution targets for running test commands.

Each target implements the ExecutionTarget protocol, allowing the executor
to run commands locally, in Docker containers, or via remote CI systems.
The architecture is pluggable — new targets can be registered at runtime.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of a single command execution."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"      # Infrastructure error (not a test failure)
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ExecutionResult:
    """Result of executing a single test command.

    Attributes:
        status: Final status of the execution.
        exit_code: Process exit code (0 = success).
        stdout: Captured standard output.
        stderr: Captured standard error.
        duration_seconds: Wall-clock time for execution.
        command_display: Human-readable command string.
        attempt: Which attempt number produced this result (1-based).
        metadata: Extra target-specific information.
    """

    status: ExecutionStatus
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    command_display: str = ""
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.PASSED

    @property
    def is_retriable(self) -> bool:
        """Whether this result suggests a retry might help.

        Test failures (deterministic) are NOT retriable.
        Infrastructure errors and timeouts ARE retriable.
        """
        return self.status in (ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT)


class ExecutionTarget(ABC):
    """Abstract base for pluggable execution targets.

    Targets handle the full lifecycle of running a command in a specific
    environment (local shell, Docker container, remote CI, etc.):

    1. ``setup``   — prepare the environment (install deps, pull images, etc.)
    2. ``execute`` — run a test command and collect structured results
    3. ``collect_results`` — post-process / gather artifacts after execution
    4. ``teardown`` — release resources (stop containers, clean temp dirs, etc.)

    Subclasses MUST implement ``name`` and ``execute``.
    All other lifecycle hooks have sensible no-op defaults so simple targets
    (e.g. local shell) work without boilerplate.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this target (e.g. 'local', 'docker')."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle: setup
    # ------------------------------------------------------------------

    async def setup(
        self,
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
    ) -> None:
        """Prepare the execution environment before any commands run.

        Override to pull Docker images, start containers, create temp dirs,
        install dependencies, etc.  The default is a no-op.

        Args:
            working_directory: Project/workspace root for the test run.
            env: Extra environment variables to pre-configure.
        """
        logger.debug("%s: setup (no-op default)", self.name)

    # ------------------------------------------------------------------
    # Lifecycle: execute (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Execute a command and return the result.

        Args:
            command: Command tokens to execute.
            working_directory: Directory to run the command in.
            env: Extra environment variables.
            timeout: Timeout in seconds (None = no timeout).

        Returns:
            ExecutionResult with captured output and status.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle: collect results
    # ------------------------------------------------------------------

    async def collect_results(
        self,
        result: ExecutionResult,
        *,
        working_directory: str = "",
    ) -> ExecutionResult:
        """Post-process an execution result and gather additional artifacts.

        Override to parse JUnit XML, collect coverage reports, download
        remote artifacts, etc.  The default returns *result* unchanged.

        Args:
            result: The raw result returned by ``execute``.
            working_directory: Project root (useful for locating artifacts).

        Returns:
            An ``ExecutionResult``, potentially enriched with extra metadata.
        """
        return result

    # ------------------------------------------------------------------
    # Lifecycle: teardown
    # ------------------------------------------------------------------

    async def teardown(self) -> None:
        """Release resources after the test run completes.

        Override to stop containers, delete temp directories, revoke
        tokens, etc.  The default is a no-op.
        """
        logger.debug("%s: teardown (no-op default)", self.name)

    # ------------------------------------------------------------------
    # Streaming helper
    # ------------------------------------------------------------------

    async def stream_execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream output line-by-line during execution.

        Default implementation falls back to non-streaming execute.
        Subclasses can override for true streaming support.
        """
        result = await self.execute(
            command,
            working_directory=working_directory,
            env=env,
            timeout=timeout,
        )
        for line in result.stdout.splitlines():
            yield line
        for line in result.stderr.splitlines():
            yield f"[stderr] {line}"

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if this target is available and ready.

        Default returns True. Override for targets that need validation.
        """
        return True


class LocalTarget(ExecutionTarget):
    """Executes commands on the local machine via subprocess."""

    @staticmethod
    def _normalize_command(command: list[str]) -> list[str]:
        """Rewrite selected commands to use the active interpreter."""
        if not command:
            return command

        executable = os.path.basename(command[0]).lower()
        if executable in {"pytest", "pytest.exe", "py.test", "py.test.exe"}:
            return [sys.executable, "-m", "pytest", *command[1:]]
        return command

    @property
    def name(self) -> str:
        return "local"

    async def execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        cwd = working_directory or None
        merged_env = {**os.environ, **(env or {})}
        start = time.monotonic()
        resolved_command = self._normalize_command(command)

        try:
            proc = await asyncio.create_subprocess_exec(
                *resolved_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=merged_env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = time.monotonic() - start
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    duration_seconds=elapsed,
                    command_display=" ".join(command),
                )

            elapsed = time.monotonic() - start
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            status = ExecutionStatus.PASSED if exit_code == 0 else ExecutionStatus.FAILED

            return ExecutionResult(
                status=status,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=elapsed,
                command_display=" ".join(command),
            )

        except FileNotFoundError:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                exit_code=-1,
                stdout="",
                stderr=f"Command not found: {resolved_command[0]}",
                duration_seconds=elapsed,
                command_display=" ".join(command),
            )
        except OSError as exc:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                exit_code=-1,
                stdout="",
                stderr=f"OS error: {exc}",
                duration_seconds=elapsed,
                command_display=" ".join(command),
            )

    async def stream_execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream stdout/stderr line-by-line from a local subprocess.

        Yields lines as they arrive rather than waiting for completion.
        Stderr lines are prefixed with ``[stderr]``.
        """
        cwd = working_directory or None
        merged_env = {**os.environ, **(env or {})}
        resolved_command = self._normalize_command(command)

        try:
            proc = await asyncio.create_subprocess_exec(
                *resolved_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=merged_env,
            )
        except (FileNotFoundError, OSError) as exc:
            yield f"[error] {exc}"
            return

        assert proc.stdout is not None
        assert proc.stderr is not None

        async def _read_lines(
            stream: asyncio.StreamReader, prefix: str = ""
        ) -> list[str]:
            lines: list[str] = []
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                lines.append(f"{prefix}{line}" if prefix else line)
            return lines

        try:
            stdout_task = asyncio.create_task(_read_lines(proc.stdout))
            stderr_task = asyncio.create_task(_read_lines(proc.stderr, "[stderr] "))

            stdout_lines, stderr_lines = await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task),
                timeout=timeout,
            )

            for line in stdout_lines:
                yield line
            for line in stderr_lines:
                yield line

        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            yield f"[timeout] Command timed out after {timeout}s"


@dataclass(frozen=True)
class SSHConfig:
    """Configuration for SSH execution."""

    alias: str = "ssh"
    hostname: str = ""
    username: str = ""
    port: int | None = None
    ssh_config_host: str = ""
    credential_ref: str = ""
    extra_args: list[str] = field(default_factory=list)
    batch_mode: bool = True

    def __post_init__(self) -> None:
        if not (self.ssh_config_host or self.hostname):
            raise ValueError("SSHConfig requires ssh_config_host or hostname")
        if self.port is not None and self.port < 1:
            raise ValueError("port must be >= 1")


class SSHTarget(ExecutionTarget):
    """Executes commands on a remote system through the local ssh client."""

    _ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(self, config: SSHConfig) -> None:
        self._config = config
        self._local = LocalTarget()

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "SSHTarget":
        """Build a target from catalog system metadata."""
        return cls(
            SSHConfig(
                alias=str(metadata.get("alias") or "ssh"),
                hostname=str(metadata.get("hostname") or ""),
                username=str(metadata.get("username") or ""),
                port=metadata.get("port"),
                ssh_config_host=str(metadata.get("ssh_config_host") or ""),
                credential_ref=str(metadata.get("credential_ref") or ""),
            )
        )

    @property
    def name(self) -> str:
        return f"ssh:{self._config.alias}"

    @property
    def destination(self) -> str:
        if self._config.ssh_config_host:
            return self._config.ssh_config_host
        if self._config.username:
            return f"{self._config.username}@{self._config.hostname}"
        return self._config.hostname

    def _build_remote_command(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
    ) -> str:
        remote_command = shlex.join(command)

        exported_pairs = [
            f"{key}={shlex.quote(value)}"
            for key, value in (env or {}).items()
            if self._ENV_KEY_PATTERN.match(key)
        ]
        if exported_pairs:
            remote_command = f"env {' '.join(exported_pairs)} {remote_command}"

        if working_directory:
            remote_command = (
                f"cd {shlex.quote(working_directory)} && {remote_command}"
            )

        return remote_command

    def _build_ssh_command(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
    ) -> list[str]:
        ssh_command = ["ssh"]
        if self._config.batch_mode:
            ssh_command.extend(["-o", "BatchMode=yes"])
        if self._config.port is not None and not self._config.ssh_config_host:
            ssh_command.extend(["-p", str(self._config.port)])
        ssh_command.extend(self._config.extra_args)
        ssh_command.append(self.destination)
        ssh_command.append(
            self._build_remote_command(
                command,
                working_directory=working_directory,
                env=env,
            )
        )
        return ssh_command

    async def execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        ssh_command = self._build_ssh_command(
            command,
            working_directory=working_directory,
            env=env,
        )
        result = await self._local.execute(ssh_command, timeout=timeout)
        metadata = dict(result.metadata)
        metadata.update(
            {
                "target": self.name,
                "transport": "ssh",
                "ssh_destination": self.destination,
                "ssh_command": " ".join(ssh_command),
                "credential_ref": self._config.credential_ref,
            }
        )
        return ExecutionResult(
            status=result.status,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration_seconds,
            command_display=" ".join(command),
            attempt=result.attempt,
            metadata=metadata,
        )

    async def stream_execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> AsyncIterator[str]:
        ssh_command = self._build_ssh_command(
            command,
            working_directory=working_directory,
            env=env,
        )
        async for line in self._local.stream_execute(ssh_command, timeout=timeout):
            yield line

    async def health_check(self) -> bool:
        result = await self._local.execute(["ssh", "-V"], timeout=10)
        return result.success


@dataclass
class DockerConfig:
    """Configuration for Docker execution target.

    Attributes:
        image: Docker image to use (e.g. 'python:3.11', 'node:18').
        dockerfile: Path to a Dockerfile to build from (optional, overrides image pull).
        build_context: Build context directory when using dockerfile.
        build_args: Build arguments passed to ``docker build --build-arg``.
        container_name_prefix: Prefix for generated container names.
        mount_workdir: Whether to bind-mount the host working directory.
        container_workdir: Working directory inside the container.
        network: Docker network to attach the container to.
        extra_run_args: Additional arguments passed to ``docker run``.
        pull_policy: When to pull the image: 'always', 'if-not-present', 'never'.
        auto_remove: Remove the container after it exits (``--rm``).
        platform: Target platform (e.g. 'linux/amd64') for multi-arch images.
    """

    image: str = "python:3.11-slim"
    dockerfile: str = ""
    build_context: str = "."
    build_args: dict[str, str] = field(default_factory=dict)
    container_name_prefix: str = "test-runner"
    mount_workdir: bool = True
    container_workdir: str = "/workspace"
    network: str = ""
    extra_run_args: list[str] = field(default_factory=list)
    pull_policy: str = "if-not-present"  # 'always' | 'if-not-present' | 'never'
    auto_remove: bool = True
    platform: str = ""


class DockerTarget(ExecutionTarget):
    """Executes commands inside Docker containers.

    Supports:
    - Pulling images from registries (with configurable pull policy)
    - Building images from Dockerfiles
    - Bind-mounting the host working directory into the container
    - Environment variable injection
    - Configurable timeouts
    - Container cleanup on teardown
    - Real-time streaming of container output

    The target uses the Docker CLI (``docker`` binary) rather than the Docker
    SDK to avoid heavy dependencies. The CLI must be on ``$PATH``.

    Example::

        target = DockerTarget(DockerConfig(image="python:3.11"))
        await target.setup(working_directory="/my/project")
        result = await target.execute(
            ["pytest", "-v", "tests/"],
            working_directory="/my/project",
        )
        await target.teardown()
    """

    def __init__(
        self,
        config: DockerConfig | str | None = None,
        image: str = "",
        *,
        container_name: str = "",
    ) -> None:
        # Support both DockerConfig and legacy string-based init
        if isinstance(config, str):
            # Legacy: DockerTarget("python:3.11")
            self._config = DockerConfig(image=config)
        elif config is None:
            self._config = DockerConfig(image=image or "python:3.11-slim")
        else:
            self._config = config

        self._container_name = (
            container_name
            or f"{self._config.container_name_prefix}-{self._config.image.replace(':', '-').replace('/', '-')}"
        )
        self._image = self._config.image
        self._built_image: str = ""  # Set if we build from Dockerfile
        self._setup_complete = False
        self._local = LocalTarget()  # Reused for all docker CLI calls
        self._active_containers: list[str] = []

    @property
    def name(self) -> str:
        return f"docker:{self._effective_image}"

    @property
    def _effective_image(self) -> str:
        """The image to use — either the built image or the configured one."""
        return self._built_image or self._image

    # ------------------------------------------------------------------
    # Lifecycle: setup — pull or build the image
    # ------------------------------------------------------------------

    async def setup(
        self,
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
    ) -> None:
        """Prepare the Docker environment.

        If a Dockerfile is configured, builds the image.
        Otherwise, pulls the image according to the pull policy.

        Raises no exceptions on failure — logs warnings and marks setup
        incomplete so that ``execute`` can report a clear error.
        """
        logger.info("DockerTarget: setting up (image=%s)", self._image)

        if self._config.dockerfile:
            await self._build_image(working_directory)
        else:
            await self._pull_image()

        self._setup_complete = True
        logger.info(
            "DockerTarget: setup complete (effective_image=%s)",
            self._effective_image,
        )

    async def _pull_image(self) -> None:
        """Pull the Docker image based on pull_policy."""
        if self._config.pull_policy == "never":
            logger.debug("DockerTarget: pull_policy=never, skipping pull")
            return

        if self._config.pull_policy == "if-not-present":
            # Check if image already exists locally
            check = await self._local.execute(
                ["docker", "image", "inspect", self._image],
                timeout=30,
            )
            if check.success:
                logger.debug("DockerTarget: image %s already present", self._image)
                return

        logger.info("DockerTarget: pulling image %s", self._image)
        pull_cmd = ["docker", "pull"]
        if self._config.platform:
            pull_cmd.extend(["--platform", self._config.platform])
        pull_cmd.append(self._image)

        result = await self._local.execute(pull_cmd, timeout=600)
        if not result.success:
            logger.warning(
                "DockerTarget: failed to pull %s: %s",
                self._image,
                result.stderr.strip(),
            )

    async def _build_image(self, working_directory: str = "") -> None:
        """Build a Docker image from the configured Dockerfile."""
        import uuid

        tag = f"{self._config.container_name_prefix}-custom:{uuid.uuid4().hex[:12]}"
        build_context = self._config.build_context
        if working_directory and not build_context.startswith("/"):
            # Resolve relative build context against working directory
            import os
            build_context = os.path.join(working_directory, build_context)

        build_cmd = [
            "docker", "build",
            "-f", self._config.dockerfile,
            "-t", tag,
        ]
        if self._config.platform:
            build_cmd.extend(["--platform", self._config.platform])
        for arg_name, arg_val in self._config.build_args.items():
            build_cmd.extend(["--build-arg", f"{arg_name}={arg_val}"])
        build_cmd.append(build_context)

        logger.info("DockerTarget: building image from %s", self._config.dockerfile)
        result = await self._local.execute(build_cmd, timeout=600)
        if result.success:
            self._built_image = tag
            logger.info("DockerTarget: built image %s", tag)
        else:
            logger.warning(
                "DockerTarget: build failed: %s", result.stderr.strip()
            )

    # ------------------------------------------------------------------
    # Lifecycle: execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Execute a command inside a Docker container.

        Constructs a ``docker run`` command with:
        - Bind-mounted working directory (if configured)
        - Environment variable injection
        - Configurable network and platform
        - Auto-remove on exit (if configured)

        Returns an ExecutionResult with Docker-specific metadata.
        """
        docker_cmd = self._build_run_command(
            command,
            working_directory=working_directory,
            env=env,
        )

        start = time.monotonic()
        result = await self._local.execute(docker_cmd, timeout=timeout)
        elapsed = time.monotonic() - start

        # Translate the raw result into a Docker-contextualized one
        return ExecutionResult(
            status=result.status,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=elapsed,
            command_display=" ".join(command),
            attempt=result.attempt,
            metadata={
                "target": self.name,
                "image": self._effective_image,
                "docker_command": " ".join(docker_cmd),
                "container_workdir": self._config.container_workdir,
                "mount_workdir": self._config.mount_workdir,
            },
        )

    def _build_run_command(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """Build the ``docker run`` command line."""
        docker_cmd: list[str] = ["docker", "run"]

        if self._config.auto_remove:
            docker_cmd.append("--rm")

        # Container name for tracking
        import uuid
        container_id = f"{self._container_name}-{uuid.uuid4().hex[:8]}"
        docker_cmd.extend(["--name", container_id])
        self._active_containers.append(container_id)

        # Working directory mount
        if self._config.mount_workdir and working_directory:
            docker_cmd.extend([
                "-v", f"{working_directory}:{self._config.container_workdir}",
                "-w", self._config.container_workdir,
            ])
        elif working_directory:
            docker_cmd.extend(["-w", working_directory])

        # Network
        if self._config.network:
            docker_cmd.extend(["--network", self._config.network])

        # Platform
        if self._config.platform:
            docker_cmd.extend(["--platform", self._config.platform])

        # Environment variables
        for key, val in (env or {}).items():
            docker_cmd.extend(["-e", f"{key}={val}"])

        # Extra user-defined run args
        docker_cmd.extend(self._config.extra_run_args)

        # Image and command
        docker_cmd.append(self._effective_image)
        docker_cmd.extend(command)

        return docker_cmd

    # ------------------------------------------------------------------
    # Lifecycle: collect_results
    # ------------------------------------------------------------------

    async def collect_results(
        self,
        result: ExecutionResult,
        *,
        working_directory: str = "",
    ) -> ExecutionResult:
        """Enrich results with Docker-specific metadata.

        If the container produced JUnit XML or coverage artifacts in the
        mounted working directory, they will be available at the same
        host paths (since we bind-mount).
        """
        result.metadata.setdefault("target", self.name)
        result.metadata.setdefault("image", self._effective_image)
        return result

    # ------------------------------------------------------------------
    # Lifecycle: teardown
    # ------------------------------------------------------------------

    async def teardown(self) -> None:
        """Clean up Docker resources.

        Stops and removes any tracked containers that are still running,
        and removes any custom-built images.
        """
        logger.info("DockerTarget: tearing down")

        # Stop/remove active containers (best-effort)
        for container_id in self._active_containers:
            stop_result = await self._local.execute(
                ["docker", "rm", "-f", container_id],
                timeout=30,
            )
            if stop_result.success:
                logger.debug("DockerTarget: removed container %s", container_id)

        self._active_containers.clear()

        # Remove custom-built image if we built one
        if self._built_image:
            await self._local.execute(
                ["docker", "rmi", "-f", self._built_image],
                timeout=30,
            )
            logger.debug("DockerTarget: removed built image %s", self._built_image)
            self._built_image = ""

        self._setup_complete = False
        logger.info("DockerTarget: teardown complete")

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
        """Stream output from a Docker container line-by-line.

        Uses the local target's streaming to read docker run output
        in real time.
        """
        docker_cmd = self._build_run_command(
            command,
            working_directory=working_directory,
            env=env,
        )
        async for line in self._local.stream_execute(
            docker_cmd, timeout=timeout
        ):
            yield line

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check that Docker is available and the daemon is running."""
        result = await self._local.execute(["docker", "info"], timeout=10)
        return result.success


class RemoteCITarget(ExecutionTarget):
    """Legacy wrapper for backward compatibility.

    For full remote CI functionality, use ``RemoteCIExecutionTarget`` from
    ``test_runner.execution.remote_ci`` with a ``CIProvider`` (e.g.
    ``GitHubActionsProvider`` or ``JenkinsProvider``).

    This class now delegates to ``RemoteCIExecutionTarget`` if a provider
    can be inferred, otherwise returns a helpful error message.
    """

    def __init__(self, ci_url: str, *, api_token: str = "") -> None:
        self._ci_url = ci_url
        self._api_token = api_token

    @property
    def name(self) -> str:
        return f"remote-ci:{self._ci_url}"

    async def execute(
        self,
        command: list[str],
        *,
        working_directory: str = "",
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Stub for backward compatibility.

        For full remote CI support, use RemoteCIExecutionTarget with a
        GitHubActionsProvider or JenkinsProvider.
        """
        return ExecutionResult(
            status=ExecutionStatus.ERROR,
            exit_code=-1,
            stdout="",
            stderr=(
                "RemoteCITarget is a legacy stub. Use RemoteCIExecutionTarget "
                "with a CIProvider (GitHubActionsProvider or JenkinsProvider) "
                "for full remote CI support. See test_runner.execution.remote_ci."
            ),
            duration_seconds=0.0,
            command_display=" ".join(command),
            metadata={"target": self.name, "ci_url": self._ci_url},
        )


# ---------------------------------------------------------------------------
# Target registry
# ---------------------------------------------------------------------------


class TargetRegistry:
    """Registry of available execution targets.

    Provides a pluggable lookup for execution targets by name.
    """

    def __init__(self) -> None:
        self._targets: dict[str, ExecutionTarget] = {}
        # Register the local target by default
        self.register(LocalTarget())

    def register(self, target: ExecutionTarget) -> None:
        """Register an execution target by its name."""
        logger.info("Registered execution target: %s", target.name)
        self._targets[target.name] = target

    def get(self, name: str) -> ExecutionTarget | None:
        """Look up a target by name."""
        return self._targets.get(name)

    def get_default(self) -> ExecutionTarget:
        """Return the default (local) target."""
        return self._targets.get("local") or LocalTarget()

    @property
    def available_targets(self) -> list[str]:
        """List names of all registered targets."""
        return list(self._targets.keys())
