"""Tests for pluggable execution targets."""

from __future__ import annotations

import sys

import pytest

from test_runner.execution.targets import (
    DockerConfig,
    DockerTarget,
    ExecutionResult,
    ExecutionStatus,
    ExecutionTarget,
    LocalTarget,
    RemoteCITarget,
    SSHConfig,
    SSHTarget,
    TargetRegistry,
)


class TestLocalTarget:
    """Tests for the local subprocess execution target."""

    def test_normalize_pytest_uses_active_interpreter(self, monkeypatch):
        target = LocalTarget()
        monkeypatch.setattr(sys, "executable", "/tmp/python-test")

        command = target._normalize_command(["pytest", "tests/test_cli.py"])

        assert command == ["/tmp/python-test", "-m", "pytest", "tests/test_cli.py"]

    def test_normalize_non_pytest_leaves_command_unchanged(self):
        target = LocalTarget()

        command = target._normalize_command(["echo", "hello"])

        assert command == ["echo", "hello"]

    @pytest.mark.asyncio
    async def test_successful_command(self):
        target = LocalTarget()
        result = await target.execute(["echo", "hello"])
        assert result.success
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.status == ExecutionStatus.PASSED

    @pytest.mark.asyncio
    async def test_failing_command(self):
        target = LocalTarget()
        result = await target.execute(["python", "-c", "raise SystemExit(1)"])
        assert not result.success
        assert result.exit_code == 1
        assert result.status == ExecutionStatus.FAILED

    @pytest.mark.asyncio
    async def test_command_not_found(self):
        target = LocalTarget()
        result = await target.execute(["nonexistent_command_xyz"])
        assert result.status == ExecutionStatus.ERROR
        assert "not found" in result.stderr.lower() or "No such file" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout(self):
        target = LocalTarget()
        result = await target.execute(
            ["python", "-c", "import time; time.sleep(10)"],
            timeout=1,
        )
        assert result.status == ExecutionStatus.TIMEOUT
        assert result.exit_code == -1

    @pytest.mark.asyncio
    async def test_captures_stderr(self):
        target = LocalTarget()
        result = await target.execute(
            ["python", "-c", "import sys; sys.stderr.write('err_msg')"]
        )
        assert "err_msg" in result.stderr

    @pytest.mark.asyncio
    async def test_duration_tracked(self):
        target = LocalTarget()
        result = await target.execute(["echo", "fast"])
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_env_injection(self):
        target = LocalTarget()
        result = await target.execute(
            ["python", "-c", "import os; print(os.environ.get('TEST_VAR', ''))"],
            env={"TEST_VAR": "injected_value"},
        )
        assert "injected_value" in result.stdout

    @pytest.mark.asyncio
    async def test_health_check(self):
        target = LocalTarget()
        assert await target.health_check()

    def test_name(self):
        assert LocalTarget().name == "local"

    @pytest.mark.asyncio
    async def test_stream_execute_yields_lines(self):
        target = LocalTarget()
        lines = []
        async for line in target.stream_execute(
            ["python", "-c", "print('line1'); print('line2')"]
        ):
            lines.append(line)
        assert "line1" in lines
        assert "line2" in lines

    @pytest.mark.asyncio
    async def test_stream_execute_captures_stderr(self):
        target = LocalTarget()
        lines = []
        async for line in target.stream_execute(
            ["python", "-c", "import sys; sys.stderr.write('err_line\\n')"]
        ):
            lines.append(line)
        assert any("[stderr]" in l and "err_line" in l for l in lines)

    @pytest.mark.asyncio
    async def test_stream_execute_command_not_found(self):
        target = LocalTarget()
        lines = []
        async for line in target.stream_execute(["nonexistent_cmd_xyz"]):
            lines.append(line)
        assert any("[error]" in l for l in lines)

    @pytest.mark.asyncio
    async def test_stream_execute_timeout(self):
        target = LocalTarget()
        lines = []
        async for line in target.stream_execute(
            ["python", "-c", "import time; time.sleep(10)"],
            timeout=1,
        ):
            lines.append(line)
        assert any("[timeout]" in l for l in lines)

    @pytest.mark.asyncio
    async def test_working_directory(self):
        target = LocalTarget()
        result = await target.execute(
            ["python", "-c", "import os; print(os.getcwd())"],
            working_directory="/tmp",
        )
        assert result.success
        # /tmp may resolve to /private/tmp on macOS
        assert "tmp" in result.stdout.lower()


class TestExecutionTargetLifecycle:
    """Tests for the setup / collect_results / teardown lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_setup_is_noop_by_default(self):
        target = LocalTarget()
        # Should complete without error
        await target.setup(working_directory="/tmp", env={"A": "1"})

    @pytest.mark.asyncio
    async def test_teardown_is_noop_by_default(self):
        target = LocalTarget()
        await target.teardown()

    @pytest.mark.asyncio
    async def test_collect_results_returns_unchanged_by_default(self):
        target = LocalTarget()
        result = await target.execute(["echo", "hi"])
        collected = await target.collect_results(result, working_directory="/tmp")
        assert collected is result

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Run the complete setup → execute → collect → teardown cycle."""
        target = LocalTarget()
        await target.setup(working_directory="/tmp")
        result = await target.execute(["echo", "lifecycle"])
        result = await target.collect_results(result)
        assert result.success
        assert "lifecycle" in result.stdout
        await target.teardown()

    @pytest.mark.asyncio
    async def test_custom_collect_results(self):
        """Subclass can enrich results via collect_results."""

        class EnrichingTarget(LocalTarget):
            async def collect_results(self, result, *, working_directory=""):
                result.metadata["enriched"] = True
                return result

        target = EnrichingTarget()
        result = await target.execute(["echo", "test"])
        result = await target.collect_results(result)
        assert result.metadata["enriched"] is True


class TestSSHTarget:
    """Tests for the SSH execution target."""

    def test_name_uses_alias(self):
        target = SSHTarget(
            SSHConfig(alias="lab-a", hostname="lab-a.internal.example"),
        )
        assert target.name == "ssh:lab-a"

    def test_build_ssh_command_with_hostname_and_user(self):
        target = SSHTarget(
            SSHConfig(
                alias="lab-a",
                hostname="lab-a.internal.example",
                username="runner",
                port=2222,
            )
        )

        command = target._build_ssh_command(
            ["python", "scripts/check.py", "--quick"],
            working_directory="/opt/test-suite",
            env={"MODE": "smoke"},
        )

        assert command[:5] == ["ssh", "-o", "BatchMode=yes", "-p", "2222"]
        assert "runner@lab-a.internal.example" in command
        assert command[-1] == (
            "cd /opt/test-suite && env MODE=smoke "
            "python scripts/check.py --quick"
        )

    def test_build_ssh_command_prefers_ssh_config_host(self):
        target = SSHTarget(
            SSHConfig(
                alias="lab-a",
                hostname="ignored.example",
                ssh_config_host="lab-a",
            )
        )

        command = target._build_ssh_command(["./bin/device-check"])

        assert "lab-a" in command
        assert "ignored.example" not in command

    @pytest.mark.asyncio
    async def test_execute_wraps_local_ssh_invocation(self, monkeypatch):
        target = SSHTarget(
            SSHConfig(alias="lab-a", hostname="lab-a.internal.example"),
        )

        seen_commands: list[list[str]] = []

        async def fake_execute(command, *, working_directory="", env=None, timeout=None):
            seen_commands.append(command)
            if command[:2] == ["ssh", "-V"]:
                return ExecutionResult(
                    status=ExecutionStatus.PASSED,
                    exit_code=0,
                    stdout="",
                    stderr="OpenSSH_9.0",
                    duration_seconds=0.1,
                    command_display="ssh -V",
                )
            assert command[0] == "ssh"
            if command[-1] == "true":
                return ExecutionResult(
                    status=ExecutionStatus.PASSED,
                    exit_code=0,
                    stdout="",
                    stderr="",
                    duration_seconds=0.1,
                    command_display=" ".join(command),
                )
            return ExecutionResult(
                status=ExecutionStatus.PASSED,
                exit_code=0,
                stdout="remote-ok",
                stderr="",
                duration_seconds=0.5,
                command_display=" ".join(command),
            )

        monkeypatch.setattr(target._local, "execute", fake_execute)

        result = await target.execute(["./bin/device-check"], timeout=30)

        assert result.success
        assert result.stdout == "remote-ok"
        assert result.metadata["target"] == "ssh:lab-a"
        assert result.metadata["transport"] == "ssh"
        assert seen_commands[0] == ["ssh", "-V"]

    @pytest.mark.asyncio
    async def test_execute_reports_missing_ssh_client_as_preflight_error(self, monkeypatch):
        target = SSHTarget(
            SSHConfig(alias="lab-a", hostname="lab-a.internal.example"),
        )

        async def fake_execute(command, *, working_directory="", env=None, timeout=None):
            if command[:2] == ["ssh", "-V"]:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    exit_code=-1,
                    stdout="",
                    stderr="Command not found: ssh",
                    duration_seconds=0.1,
                    command_display="ssh -V",
                )
            raise AssertionError("preflight should stop before remote execution")

        monkeypatch.setattr(target._local, "execute", fake_execute)

        result = await target.execute(["./bin/device-check"], timeout=30)

        assert result.status == ExecutionStatus.ERROR
        assert "SSH preflight failed" in result.stderr
        assert result.metadata["ssh_preflight"] == "client"

    @pytest.mark.asyncio
    async def test_health_check_fails_when_destination_is_unreachable(self, monkeypatch):
        target = SSHTarget(
            SSHConfig(alias="lab-a", hostname="lab-a.internal.example"),
        )

        async def fake_execute(command, *, working_directory="", env=None, timeout=None):
            if command[:2] == ["ssh", "-V"]:
                return ExecutionResult(
                    status=ExecutionStatus.PASSED,
                    exit_code=0,
                    stdout="",
                    stderr="OpenSSH_9.0",
                    duration_seconds=0.1,
                    command_display="ssh -V",
                )
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                exit_code=255,
                stdout="",
                stderr="ssh: connect to host lab-a.internal.example port 22: Connection timed out",
                duration_seconds=0.1,
                command_display=" ".join(command),
            )

        monkeypatch.setattr(target._local, "execute", fake_execute)

        assert not await target.health_check()


class TestDockerTarget:
    """Tests for the Docker execution target."""

    def test_name_includes_image(self):
        target = DockerTarget("python:3.11")
        assert "python:3.11" in target.name

    def test_name_with_config(self):
        config = DockerConfig(image="node:18-alpine")
        target = DockerTarget(config)
        assert "node:18-alpine" in target.name

    def test_default_image(self):
        target = DockerTarget()
        assert "python:3.11-slim" in target.name

    def test_legacy_string_init(self):
        """Backward-compatible string-based init."""
        target = DockerTarget("python:3.11")
        assert target._image == "python:3.11"

    def test_config_init(self):
        config = DockerConfig(
            image="ruby:3.2",
            mount_workdir=False,
            container_workdir="/app",
            network="my-net",
        )
        target = DockerTarget(config)
        assert target._image == "ruby:3.2"
        assert target._config.mount_workdir is False
        assert target._config.container_workdir == "/app"
        assert target._config.network == "my-net"

    def test_build_run_command_basic(self):
        target = DockerTarget("python:3.11")
        cmd = target._build_run_command(["pytest", "-v"])
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--rm" in cmd
        assert "python:3.11" in cmd
        assert cmd[-2:] == ["pytest", "-v"]

    def test_build_run_command_with_workdir_mount(self):
        config = DockerConfig(image="python:3.11", mount_workdir=True)
        target = DockerTarget(config)
        cmd = target._build_run_command(
            ["pytest"],
            working_directory="/my/project",
        )
        # Should contain volume mount
        assert "-v" in cmd
        mount_idx = cmd.index("-v")
        assert "/my/project:/workspace" in cmd[mount_idx + 1]
        assert "-w" in cmd
        w_idx = cmd.index("-w")
        assert cmd[w_idx + 1] == "/workspace"

    def test_build_run_command_no_mount(self):
        config = DockerConfig(image="python:3.11", mount_workdir=False)
        target = DockerTarget(config)
        cmd = target._build_run_command(
            ["pytest"],
            working_directory="/my/project",
        )
        # Should set -w but NOT -v
        assert "-w" in cmd
        # No volume mount
        assert "-v" not in cmd

    def test_build_run_command_with_env(self):
        target = DockerTarget("python:3.11")
        cmd = target._build_run_command(
            ["pytest"],
            env={"CI": "true", "TOKEN": "abc"},
        )
        assert "-e" in cmd
        env_values = []
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                env_values.append(cmd[i + 1])
        assert "CI=true" in env_values
        assert "TOKEN=abc" in env_values

    def test_build_run_command_with_network(self):
        config = DockerConfig(image="python:3.11", network="test-net")
        target = DockerTarget(config)
        cmd = target._build_run_command(["pytest"])
        assert "--network" in cmd
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "test-net"

    def test_build_run_command_with_platform(self):
        config = DockerConfig(image="python:3.11", platform="linux/amd64")
        target = DockerTarget(config)
        cmd = target._build_run_command(["pytest"])
        assert "--platform" in cmd
        plat_idx = cmd.index("--platform")
        assert cmd[plat_idx + 1] == "linux/amd64"

    def test_build_run_command_with_extra_args(self):
        config = DockerConfig(
            image="python:3.11",
            extra_run_args=["--memory", "512m", "--cpus", "2"],
        )
        target = DockerTarget(config)
        cmd = target._build_run_command(["pytest"])
        assert "--memory" in cmd
        assert "512m" in cmd
        assert "--cpus" in cmd

    def test_build_run_command_no_auto_remove(self):
        config = DockerConfig(image="python:3.11", auto_remove=False)
        target = DockerTarget(config)
        cmd = target._build_run_command(["pytest"])
        # --rm should NOT be present after "docker run"
        # (but it might appear in container names etc.)
        # Check the first few args
        assert cmd[0:2] == ["docker", "run"]
        assert cmd[2] != "--rm"

    @pytest.mark.asyncio
    async def test_execute_returns_docker_metadata(self):
        """Execute includes Docker-specific metadata in the result."""
        target = DockerTarget("python:3.11")
        result = await target.execute(["echo", "hello"])
        assert "target" in result.metadata
        assert "image" in result.metadata
        assert "docker_command" in result.metadata
        assert result.metadata["image"] == "python:3.11"

    @pytest.mark.asyncio
    async def test_execute_preserves_command_display(self):
        target = DockerTarget("python:3.11")
        result = await target.execute(["pytest", "-v", "tests/"])
        assert result.command_display == "pytest -v tests/"

    @pytest.mark.asyncio
    async def test_execute_runs_docker(self):
        """Docker target wraps command in docker run (may fail if Docker not installed)."""
        target = DockerTarget("python:3.11")
        result = await target.execute(["echo", "hello"])
        assert result.status in (
            ExecutionStatus.PASSED,
            ExecutionStatus.ERROR,
            ExecutionStatus.FAILED,
        )

    @pytest.mark.asyncio
    async def test_health_check(self):
        target = DockerTarget("python:3.11")
        # Returns True if Docker is available, False otherwise
        check = await target.health_check()
        assert isinstance(check, bool)

    @pytest.mark.asyncio
    async def test_setup_teardown_lifecycle(self):
        """Setup and teardown complete without error."""
        config = DockerConfig(image="python:3.11", pull_policy="never")
        target = DockerTarget(config)
        await target.setup(working_directory="/tmp")
        assert target._setup_complete
        await target.teardown()
        assert not target._setup_complete

    @pytest.mark.asyncio
    async def test_collect_results_enriches_metadata(self):
        config = DockerConfig(image="python:3.11", pull_policy="never")
        target = DockerTarget(config)
        raw = ExecutionResult(
            status=ExecutionStatus.PASSED,
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.0,
        )
        enriched = await target.collect_results(raw, working_directory="/tmp")
        assert enriched.metadata["target"] == target.name
        assert enriched.metadata["image"] == "python:3.11"

    @pytest.mark.asyncio
    async def test_teardown_clears_active_containers(self):
        config = DockerConfig(image="python:3.11", pull_policy="never")
        target = DockerTarget(config)
        target._active_containers = ["test-container-1", "test-container-2"]
        await target.teardown()
        assert target._active_containers == []

    @pytest.mark.asyncio
    async def test_teardown_clears_built_image(self):
        config = DockerConfig(image="python:3.11", pull_policy="never")
        target = DockerTarget(config)
        target._built_image = "custom:abc123"
        await target.teardown()
        assert target._built_image == ""

    def test_effective_image_uses_built_image(self):
        target = DockerTarget("python:3.11")
        assert target._effective_image == "python:3.11"
        target._built_image = "custom:abc123"
        assert target._effective_image == "custom:abc123"

    def test_container_name_generated(self):
        target = DockerTarget("python:3.11")
        assert target._container_name.startswith("test-runner-")

    def test_container_name_custom(self):
        target = DockerTarget("python:3.11", container_name="my-container")
        assert target._container_name == "my-container"


class TestDockerConfig:
    """Tests for DockerConfig dataclass."""

    def test_defaults(self):
        config = DockerConfig()
        assert config.image == "python:3.11-slim"
        assert config.dockerfile == ""
        assert config.mount_workdir is True
        assert config.container_workdir == "/workspace"
        assert config.pull_policy == "if-not-present"
        assert config.auto_remove is True
        assert config.network == ""
        assert config.platform == ""
        assert config.extra_run_args == []
        assert config.build_args == {}

    def test_custom_values(self):
        config = DockerConfig(
            image="node:18",
            dockerfile="Dockerfile.test",
            build_context="./docker",
            build_args={"NODE_ENV": "test"},
            mount_workdir=False,
            container_workdir="/app",
            network="host",
            extra_run_args=["--privileged"],
            pull_policy="always",
            auto_remove=False,
            platform="linux/arm64",
        )
        assert config.image == "node:18"
        assert config.dockerfile == "Dockerfile.test"
        assert config.build_context == "./docker"
        assert config.build_args == {"NODE_ENV": "test"}
        assert config.mount_workdir is False
        assert config.container_workdir == "/app"
        assert config.network == "host"
        assert config.extra_run_args == ["--privileged"]
        assert config.pull_policy == "always"
        assert config.auto_remove is False
        assert config.platform == "linux/arm64"


class TestRemoteCITarget:
    """Tests for the remote CI target stub."""

    def test_name_includes_url(self):
        target = RemoteCITarget("https://ci.example.com")
        assert "ci.example.com" in target.name

    @pytest.mark.asyncio
    async def test_returns_not_implemented_error(self):
        target = RemoteCITarget("https://ci.example.com")
        result = await target.execute(["echo", "hello"])
        assert result.status == ExecutionStatus.ERROR
        assert "legacy stub" in result.stderr.lower() or "not yet implemented" in result.stderr.lower()


class TestTargetRegistry:
    """Tests for the pluggable target registry."""

    def test_local_registered_by_default(self):
        registry = TargetRegistry()
        assert "local" in registry.available_targets
        target = registry.get("local")
        assert target is not None
        assert target.name == "local"

    def test_register_custom_target(self):
        registry = TargetRegistry()
        docker = DockerTarget("node:18")
        registry.register(docker)
        assert docker.name in registry.available_targets
        assert registry.get(docker.name) is docker

    def test_get_unknown_returns_none(self):
        registry = TargetRegistry()
        assert registry.get("nonexistent") is None

    def test_get_default_returns_local(self):
        registry = TargetRegistry()
        default = registry.get_default()
        assert default.name == "local"

    def test_register_overrides_existing(self):
        registry = TargetRegistry()
        local1 = LocalTarget()
        registry.register(local1)
        assert registry.get("local") is local1
