"""Utilities for loading and calling the policy used by the ENN pipeline.

The ENN data collector only requires an object exposing ``act``.  A project can
provide any agent factory through ``AGENT_FACTORY=package.module:function``.
When no factory is configured, the repository keeps backwards compatibility
with the bundled CurriculumAgent assets (``model/`` + ``actions/``).
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable, Optional


AgentFactory = Callable[[Any], Any]


def call_agent(agent: Any, obs: Any, reward: float = 0.0, done: bool = False) -> Any:
    """Call a Grid2Op-like agent while tolerating common ``act`` signatures."""
    try:
        return agent.act(obs, reward=reward, done=done)
    except TypeError:
        try:
            return agent.act(obs, reward, done)
        except TypeError:
            return agent.act(obs)


def import_agent_factory(spec: str) -> AgentFactory:
    """Import ``module:function`` and return it as a callable agent factory.

    The callable receives the already-created Grid2Op environment and must
    return an object exposing ``act``.
    """
    if not spec or ":" not in spec:
        raise ValueError(
            "AGENT_FACTORY must use 'module:function' syntax, for example "
            "'my_agents.grid_agent:make_agent'."
        )
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError(f"Configured agent factory {spec!r} is not callable.")
    return factory


def _valid_curriculum_asset_dir(path: Path) -> bool:
    model_dir = path / "model"
    variables = model_dir / "variables"
    required = (
        path / "actions" / "actions.npy",
        model_dir / "saved_model.pb",
        variables / "variables.index",
        variables / "variables.data-00000-of-00001",
    )
    return all(p.is_file() and p.stat().st_size > 0 for p in required)


def build_agent(
    env: Any,
    *,
    factory: Optional[AgentFactory] = None,
    factory_spec: Optional[str] = None,
    agent_path: Optional[str | Path] = None,
) -> Any:
    """Build the policy used for rollout collection.

    Resolution order:
      1. explicit ``factory`` callable;
      2. ``factory_spec`` / ``AGENT_FACTORY`` (``module:function``);
      3. bundled CurriculumAgent assets at ``agent_path``.

    This makes rollout-based ENN training independent of tutor files while
    keeping the policy implementation itself pluggable.
    """
    if factory is not None:
        agent = factory(env)
    else:
        spec = factory_spec if factory_spec is not None else os.environ.get("AGENT_FACTORY", "").strip()
        if spec:
            agent = import_agent_factory(spec)(env)
        else:
            if agent_path is None:
                raise ValueError(
                    "No agent factory was provided and no agent_path is available. "
                    "Set AGENT_FACTORY=module:function or pass an explicit factory."
                )
            path = Path(agent_path)
            if not _valid_curriculum_asset_dir(path):
                raise FileNotFoundError(
                    f"Agent assets at {path} are incomplete. Expected model/ and actions/ "
                    "with a valid SavedModel, or configure AGENT_FACTORY for another agent."
                )
            from curriculumagent.submission.my_agent import make_agent

            agent = make_agent(env, str(path))

    if not hasattr(agent, "act") or not callable(agent.act):
        raise TypeError("Agent factory must return an object exposing an act(...) method.")
    return agent
