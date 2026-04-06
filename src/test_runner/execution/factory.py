"""ExecutionTargetFactory — registry pattern for discovering, selecting,
and instantiating execution targets by name or configuration.

The factory maintains a registry of *target creators* (callable factories)
keyed by target name.  Built-in targets (``local``, ``docker``, ``remote-ci``)
are pre-registered; third-party or custom targets can be added at runtime via
:meth:`ExecutionTargetFactory.register` or the ``@execution_target`` decorator.

Design decisions:
  - **Lazy instantiation**: creators are stored, not instances.  A new target
    is created on each :meth:`create` call so callers get independent state.
  - **Configuration-driven**: :meth:`create` accepts an optional ``config``
    dict forwarded to the creator, enabling declarative target setup from
    YAML/JSON/env without touching code.
  - **Singleton registry**: :func:`get_factory` returns a module-level
    singleton pre-loaded with built-in targets so the rest of the codebase
    can share a single registry without passing it around.
  - **Discovery helpers**: :meth:`available`, :meth:`get_info`, and
    :meth:`list_targets` support introspection for the orchestrator to
    present choices to the user or LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Type, runtime_checkable

from test_runner.execution.targets import (
    DockerConfig,
    DockerTarget,
    ExecutionTarget,
    LocalTarget,
    RemoteCITarget,
    SSHConfig,
    SSHTarget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target creator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TargetCreator(Protocol):
    """Callable that produces an :class:`ExecutionTarget` from config kwargs."""

    def __call__(self, **config: Any) -> ExecutionTarget: ...


# ---------------------------------------------------------------------------
# Target metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetInfo:
    """Metadata about a registered execution target.

    Attributes:
        name: Canonical name used for lookup (e.g. ``"local"``).
        description: Human-readable one-liner.
        creator: The callable that produces target instances.
        config_schema: Optional dict describing accepted config keys.
    """

    name: str
    description: str
    creator: TargetCreator
    config_schema: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in creator functions
# ---------------------------------------------------------------------------


def _create_local(**config: Any) -> ExecutionTarget:
    """Create a :class:`LocalTarget`."""
    return LocalTarget()


def _create_docker(**config: Any) -> ExecutionTarget:
    """Create a :class:`DockerTarget` from config.

    Accepted config keys mirror :class:`DockerConfig` fields:
    ``image``, ``dockerfile``, ``mount_workdir``, ``container_workdir``,
    ``network``, ``platform``, ``pull_policy``, ``auto_remove``,
    ``extra_run_args``, ``build_args``, ``build_context``,
    ``container_name_prefix``.
    """
    docker_cfg_keys = {
        "image", "dockerfile", "build_context", "build_args",
        "container_name_prefix", "mount_workdir", "container_workdir",
        "network", "extra_run_args", "pull_policy", "auto_remove", "platform",
    }
    docker_kwargs = {k: v for k, v in config.items() if k in docker_cfg_keys}
    remaining = {k: v for k, v in config.items() if k not in docker_cfg_keys}

    docker_config = DockerConfig(**docker_kwargs) if docker_kwargs else DockerConfig()
    return DockerTarget(docker_config, **{k: v for k, v in remaining.items() if k == "container_name"})


def _create_remote_ci(**config: Any) -> ExecutionTarget:
    """Create a :class:`RemoteCITarget` (legacy stub) from config.

    For full remote CI support, register a custom creator that uses
    :class:`RemoteCIExecutionTarget` with a :class:`CIProvider`.

    Accepted config keys: ``ci_url``, ``api_token``.
    """
    ci_url = config.get("ci_url", "")
    api_token = config.get("api_token", "")
    return RemoteCITarget(ci_url, api_token=api_token)


def _create_ssh(**config: Any) -> ExecutionTarget:
    """Create an :class:`SSHTarget` from config."""
    ssh_cfg_keys = {
        "alias", "hostname", "username", "port",
        "ssh_config_host", "auth_method", "password_env_var",
        "credential_ref", "extra_args", "batch_mode",
    }
    ssh_kwargs = {k: v for k, v in config.items() if k in ssh_cfg_keys}
    return SSHTarget(SSHConfig(**ssh_kwargs))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class ExecutionTargetFactory:
    """Registry-based factory for execution targets.

    Provides a pluggable mechanism to discover, select, and instantiate
    execution targets by name or configuration dict.

    Built-in targets are pre-registered on construction.  Additional targets
    can be registered via :meth:`register` (programmatic) or the module-level
    :func:`execution_target` decorator.

    Example::

        factory = ExecutionTargetFactory()

        # Create by name (defaults)
        local = factory.create("local")

        # Create by name + config
        docker = factory.create("docker", image="node:18", network="host")

        # List available targets
        print(factory.available())   # ['local', 'docker', 'remote-ci']

        # Register a custom target
        factory.register("k8s", my_k8s_creator, description="Kubernetes pods")
    """

    def __init__(self, *, register_builtins: bool = True) -> None:
        self._registry: dict[str, TargetInfo] = {}

        if register_builtins:
            self._register_builtins()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        creator: TargetCreator,
        *,
        description: str = "",
        config_schema: dict[str, str] | None = None,
    ) -> None:
        """Register a target creator under *name*.

        If a target with the same name already exists it is silently replaced
        (useful for overriding built-ins in tests or plugins).

        Args:
            name: Canonical lookup name (case-insensitive, stored lowercase).
            creator: Callable ``(**config) -> ExecutionTarget``.
            description: Human-readable one-liner (for listing/help).
            config_schema: Optional mapping of config key → description.
        """
        canonical = name.strip().lower()
        info = TargetInfo(
            name=canonical,
            description=description or f"Execution target: {canonical}",
            creator=creator,
            config_schema=config_schema or {},
        )
        self._registry[canonical] = info
        logger.info("Registered execution target: %s", canonical)

    def unregister(self, name: str) -> bool:
        """Remove a target from the registry.

        Returns True if the target was found and removed, False otherwise.
        """
        canonical = name.strip().lower()
        if canonical in self._registry:
            del self._registry[canonical]
            logger.info("Unregistered execution target: %s", canonical)
            return True
        return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def available(self) -> list[str]:
        """Return sorted list of registered target names."""
        return sorted(self._registry.keys())

    def has(self, name: str) -> bool:
        """Check if a target with *name* is registered."""
        return name.strip().lower() in self._registry

    def get_info(self, name: str) -> TargetInfo | None:
        """Return metadata for *name*, or ``None`` if not registered."""
        return self._registry.get(name.strip().lower())

    def list_targets(self) -> list[TargetInfo]:
        """Return metadata for all registered targets, sorted by name."""
        return [self._registry[k] for k in sorted(self._registry)]

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def create(self, name: str, **config: Any) -> ExecutionTarget:
        """Instantiate a target by name, optionally passing config kwargs.

        Args:
            name: Registered target name (case-insensitive).
            **config: Keyword arguments forwarded to the target creator.

        Returns:
            A fresh :class:`ExecutionTarget` instance.

        Raises:
            KeyError: If *name* is not registered.
        """
        canonical = name.strip().lower()
        info = self._registry.get(canonical)
        if info is None:
            available = ", ".join(self.available()) or "(none)"
            raise KeyError(
                f"Unknown execution target {name!r}. "
                f"Available targets: {available}"
            )
        logger.debug("Creating execution target %r with config %s", canonical, config)
        return info.creator(**config)

    def create_from_config(self, config: dict[str, Any]) -> ExecutionTarget:
        """Instantiate a target from a configuration dictionary.

        The dict must contain a ``"target"`` key with the target name.
        All other keys are forwarded as config kwargs.

        Example config::

            {"target": "docker", "image": "python:3.11", "network": "host"}

        Args:
            config: Dict with ``"target"`` key and optional config keys.

        Returns:
            A fresh :class:`ExecutionTarget` instance.

        Raises:
            KeyError: If ``"target"`` key is missing or name is not registered.
        """
        config = dict(config)  # shallow copy
        name = config.pop("target", None)
        if name is None:
            raise KeyError(
                "Configuration dict must contain a 'target' key "
                "specifying the execution target name."
            )
        return self.create(name, **config)

    def get_or_create(
        self, name: str, *, default: str = "local", **config: Any
    ) -> ExecutionTarget:
        """Like :meth:`create` but falls back to *default* if *name* unknown.

        This is useful when the orchestrator receives a user-specified target
        name that might not be registered — rather than erroring, it falls
        back gracefully.
        """
        canonical = name.strip().lower()
        if canonical not in self._registry:
            logger.warning(
                "Target %r not found, falling back to %r", name, default
            )
            canonical = default
        return self.create(canonical, **config)

    # ------------------------------------------------------------------
    # Built-in registration
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        """Register the built-in execution targets."""
        self.register(
            "local",
            _create_local,
            description="Execute commands on the local machine via subprocess",
            config_schema={},
        )
        self.register(
            "docker",
            _create_docker,
            description="Execute commands inside Docker containers",
            config_schema={
                "image": "Docker image to use (default: python:3.11-slim)",
                "dockerfile": "Path to Dockerfile to build from",
                "mount_workdir": "Bind-mount host working directory (default: true)",
                "container_workdir": "Working directory inside container (default: /workspace)",
                "network": "Docker network to attach to",
                "platform": "Target platform (e.g. linux/amd64)",
                "pull_policy": "When to pull: always, if-not-present, never",
                "auto_remove": "Remove container after exit (default: true)",
                "extra_run_args": "Additional docker run arguments",
                "build_args": "Build arguments for Dockerfile builds",
            },
        )
        self.register(
            "remote-ci",
            _create_remote_ci,
            description="Trigger tests on a remote CI system (legacy stub)",
            config_schema={
                "ci_url": "URL of the remote CI system",
                "api_token": "API authentication token",
            },
        )
        self.register(
            "ssh",
            _create_ssh,
            description="Execute commands on a remote host through ssh",
            config_schema={
                "alias": "Saved system alias for reporting",
                "hostname": "Remote host name or address",
                "username": "Remote ssh username",
                "port": "Optional ssh port",
                "ssh_config_host": "Optional host alias from ~/.ssh/config",
                "auth_method": "SSH authentication method: ssh_key or password",
                "password_env_var": "Env var name that stores the SSH password for password auth",
                "credential_ref": "Pointer to external credentials or ssh config",
            },
        )


# ---------------------------------------------------------------------------
# Decorator for third-party target registration
# ---------------------------------------------------------------------------

# Module-level singleton, lazily initialized
_default_factory: ExecutionTargetFactory | None = None


def get_factory() -> ExecutionTargetFactory:
    """Return the module-level singleton :class:`ExecutionTargetFactory`.

    Creates it on first call with built-in targets pre-registered.
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = ExecutionTargetFactory()
    return _default_factory


def reset_factory() -> None:
    """Reset the singleton factory (primarily for testing)."""
    global _default_factory
    _default_factory = None


def execution_target(
    name: str,
    *,
    description: str = "",
    config_schema: dict[str, str] | None = None,
) -> Callable[[TargetCreator], TargetCreator]:
    """Decorator to register a target creator on the default factory.

    Usage::

        @execution_target("k8s", description="Run tests in K8s pods")
        def create_k8s(**config):
            return K8sTarget(namespace=config.get("namespace", "default"))

    The decorated function is registered immediately on the singleton
    factory returned by :func:`get_factory`.
    """

    def decorator(creator: TargetCreator) -> TargetCreator:
        get_factory().register(
            name,
            creator,
            description=description,
            config_schema=config_schema or {},
        )
        return creator

    return decorator
