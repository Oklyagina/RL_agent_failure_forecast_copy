"""Shared, calculation-free support for pipeline stages.

This module centralizes the path, policy-discovery, line-parsing, and logging
conventions that were previously repeated by executable scripts.  It does not
contain forecasting, ENN, classifier, or explainability calculations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable


PathValidator = Callable[[Path], bool]


def agent_artifact_dir(
    artifacts_dir: Path, env_name: str, agent_name: str
) -> Path:
    """Return the artifact root for one environment/agent pair."""
    return Path(artifacts_dir) / env_name / agent_name


def rollout_dir(artifacts_dir: Path, env_name: str, agent_name: str) -> Path:
    """Return the standard ENN rollout directory."""
    return agent_artifact_dir(artifacts_dir, env_name, agent_name) / "rollouts"


def model_dir(artifacts_dir: Path, env_name: str, agent_name: str) -> Path:
    """Return the standard ENN model directory."""
    return agent_artifact_dir(artifacts_dir, env_name, agent_name) / "model"


def failure_dir(artifacts_dir: Path, env_name: str, agent_name: str) -> Path:
    """Return the standard failure-forecast directory."""
    return agent_artifact_dir(artifacts_dir, env_name, agent_name) / "failure_forecast"


def policy_candidates(
    assets_dir: Path,
    env_name: str,
    *,
    extra: Iterable[Path] = (),
) -> list[Path]:
    """Return policy candidates in the repository's established priority."""
    return [
        *(Path(path) for path in extra),
        Path(assets_dir) / env_name,
        Path(assets_dir) / "network36",
    ]


def has_policy_layout(path: Path) -> bool:
    """Check the legacy directory-level policy contract."""
    path = Path(path)
    return (path / "model").is_dir() and (path / "actions").is_dir()


def has_policy_entrypoints(path: Path) -> bool:
    """Check the files used by the legacy forecasting preflight."""
    path = Path(path)
    return (
        (path / "model" / "saved_model.pb").is_file()
        and (path / "actions" / "actions.npy").is_file()
    )


def has_valid_saved_model(path: Path) -> bool:
    """Check the non-empty TensorFlow files required by runtime discovery."""
    model_dir = Path(path) / "model"
    variables_dir = model_dir / "variables"
    required_files = (
        model_dir / "saved_model.pb",
        variables_dir / "variables.index",
        variables_dir / "variables.data-00000-of-00001",
    )
    return all(item.is_file() and item.stat().st_size > 0 for item in required_files)


def has_complete_policy(path: Path) -> bool:
    """Check the full bundled-policy contract used by the policy loader."""
    path = Path(path)
    actions = path / "actions" / "actions.npy"
    return (
        actions.is_file()
        and actions.stat().st_size > 0
        and has_valid_saved_model(path)
    )


def resolve_policy_dir(
    assets_dir: Path,
    env_name: str,
    *,
    extra: Iterable[Path] = (),
    validator: PathValidator = has_policy_layout,
) -> Path:
    """Select the first usable policy, preserving the historical fallback.

    When no candidate passes ``validator``, scripts historically returned
    ``assets/<environment>`` and let the policy loader emit the detailed error.
    That behavior is intentionally retained.
    """
    candidates = policy_candidates(assets_dir, env_name, extra=extra)
    for path in candidates:
        if validator(path):
            return path
    return Path(assets_dir) / env_name


def parse_lines(raw: str | None, env: Any | None = None) -> list[Any]:
    """Parse comma-separated lines or use all names exposed by an environment."""
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    if env is None:
        return []
    names = getattr(env, "name_line", None)
    if names is None:
        raise ValueError(
            "FAILURE_LINES must be set when env.name_line is unavailable."
        )
    return list(names)


def configure_quiet_logging() -> None:
    """Keep long-running stages focused on warnings and progress output."""
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("curriculumagent").setLevel(logging.WARNING)
