"""CSV dataset helpers for future RL-agent failure classification.

The target label is intentionally isolated in ``map_failure_label`` because the
project does not yet have a final definition of agent failure.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .agent_runtime import call_agent


FEATURE_ARRAYS = (
    "gen_p",
    "gen_q",
    "gen_v",
    "load_p",
    "load_q",
    "load_v",
    "p_or",
    "q_or",
    "v_or",
    "a_or",
    "p_ex",
    "q_ex",
    "v_ex",
    "a_ex",
    "rho",
    "line_status",
    "timestep_overflow",
    "topo_vect",
    "time_before_cooldown_line",
    "time_before_cooldown_sub",
    "time_next_maintenance",
    "duration_next_maintenance",
)

TIME_FEATURES = (
    "month",
    "day",
    "hour_of_day",
    "minute_of_hour",
    "day_of_week",
)


def map_failure_label(done_after: bool, next_obs: Any, info: Mapping[str, Any] | None = None) -> int:
    """Temporary rule: failure means Grid2Op ended before natural max_step.

    ``info`` is accepted so this function can later use Grid2Op termination
    details without changing the dataset collection code.
    """
    del info
    if not done_after:
        return 0

    current_step = getattr(next_obs, "current_step", None)
    max_step = getattr(next_obs, "max_step", None)
    if current_step is None or max_step is None:
        return 1
    return int(current_step < max_step)


def _as_scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        if isinstance(item, np.generic):
            return item.item()
        return item
    return value


def _add_array_features(row: dict[str, Any], prefix: str, values: Any) -> None:
    array = np.asarray(values).reshape(-1)
    for index, value in enumerate(array):
        row[f"{prefix}_{index}"] = _as_scalar(value)


def extract_observation_features(obs: Any) -> dict[str, Any]:
    """Return explicit, CSV-friendly features from a Grid2Op observation."""
    row: dict[str, Any] = {}
    for name in TIME_FEATURES:
        if hasattr(obs, name):
            row[name] = _as_scalar(getattr(obs, name))

    for name in FEATURE_ARRAYS:
        if hasattr(obs, name):
            _add_array_features(row, name, getattr(obs, name))

    return row


def extract_vector_features(vector: Sequence[float], example_obs: Any | None = None) -> dict[str, Any]:
    """Convert a saved observation vector into feature columns.

    If a Grid2Op observation object is provided, the vector is decoded back into
    an observation so the CSV gets semantic column names such as ``rho_0``.  If
    decoding is unavailable, the vector is still exported as stable
    ``obs_vect_*`` columns.
    """
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if example_obs is not None and hasattr(example_obs, "from_vect"):
        try:
            obs = example_obs.copy() if hasattr(example_obs, "copy") else example_obs
            obs = obs.from_vect(array.copy())
            return extract_observation_features(obs)
        except Exception:
            pass
    return {f"obs_vect_{index}": _as_scalar(value) for index, value in enumerate(array)}


def rollout_artifacts_available(rollout_dir: Path) -> bool:
    """Return whether the existing ENN rollout bundle is complete enough to reuse."""
    required = ("observations.npy", "labels.npy", "actions.npy")
    return all((rollout_dir / name).is_file() for name in required)


def load_rollout_artifacts(rollout_dir: Path, *, mmap_mode: str | None = "r") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load existing ENN rollout arrays.

    The returned ``labels`` are action-class labels, not failure labels.
    """
    observations = np.load(rollout_dir / "observations.npy", mmap_mode=mmap_mode, allow_pickle=False)
    labels = np.load(rollout_dir / "labels.npy", mmap_mode=mmap_mode, allow_pickle=False)
    actions = np.load(rollout_dir / "actions.npy", mmap_mode=mmap_mode, allow_pickle=False)
    if len(observations) != len(labels):
        raise ValueError(
            f"Rollout observations and labels have different lengths: "
            f"{len(observations)} != {len(labels)}"
        )
    return observations, labels, actions


def _chronic_name(env: Any) -> str | None:
    handler = getattr(env, "chronics_handler", None)
    if handler is None or not hasattr(handler, "get_name"):
        return None
    try:
        return str(handler.get_name())
    except Exception:
        return None


def _base_row(sample_index: int, episode: int | None, step: int | None, obs: Any) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "episode": episode,
        "step": step,
        "grid2op_current_step": getattr(obs, "current_step", None),
        "grid2op_max_step": getattr(obs, "max_step", None),
    }


def collect_fresh_failure_rows(
    env: Any,
    agent: Any,
    *,
    episodes: int,
    seed: int = 0,
    max_steps: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    progress_every: int = 25,
) -> list[dict[str, Any]]:
    """Run the configured agent and return one labeled row per timestep."""
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if max_steps is not None and max_steps <= 0:
        max_steps = None

    rows: list[dict[str, Any]] = []
    sample_index = 0
    for episode in range(episodes):
        if progress_callback is not None:
            progress_callback("episode_start", {"episode": episode, "episodes": episodes})
        try:
            obs = env.reset(seed=seed + episode)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed + episode)
            obs = env.reset()

        if hasattr(agent, "reset"):
            try:
                agent.reset(obs)
            except Exception:
                pass

        reward = getattr(env, "reward_range", (0.0, 0.0))[0]
        done = False
        step = 0
        chronic = _chronic_name(env)
        if progress_callback is not None:
            progress_callback(
                "episode_ready",
                {
                    "episode": episode,
                    "chronic_name": chronic,
                    "grid2op_current_step": getattr(obs, "current_step", None),
                    "grid2op_max_step": getattr(obs, "max_step", None),
                },
            )
        while not done and (max_steps is None or step < max_steps):
            if progress_callback is not None and (step == 0 or step % progress_every == 0):
                progress_callback(
                    "step_start",
                    {
                        "episode": episode,
                        "step": step,
                        "sample_index": sample_index,
                        "grid2op_current_step": getattr(obs, "current_step", None),
                        "max_rho": float(np.nanmax(obs.rho)) if hasattr(obs, "rho") else None,
                    },
                )
            action = call_agent(agent, obs, reward=float(reward), done=done)
            row = _base_row(sample_index, episode, step, obs)
            row["chronic_name"] = chronic
            row["reward_before"] = float(reward)
            row.update(extract_observation_features(obs))

            next_obs, next_reward, done, info = env.step(action)
            row["reward_after"] = float(next_reward)
            row["done_after"] = bool(done)
            row["failure"] = map_failure_label(done, next_obs, info)
            rows.append(row)
            if progress_callback is not None and row["failure"]:
                progress_callback(
                    "failure",
                    {
                        "episode": episode,
                        "step": step,
                        "sample_index": sample_index,
                        "grid2op_current_step": getattr(next_obs, "current_step", None),
                        "grid2op_max_step": getattr(next_obs, "max_step", None),
                    },
                )

            obs = next_obs
            reward = next_reward
            step += 1
            sample_index += 1
            if progress_callback is not None:
                progress_callback(
                    "step_done",
                    {
                        "episode": episode,
                        "step": step,
                        "sample_index": sample_index,
                        "total_rows": len(rows),
                    },
                )
        if progress_callback is not None:
            progress_callback(
                "episode_end",
                {
                    "episode": episode,
                    "steps": step,
                    "total_rows": len(rows),
                    "done": bool(done),
                },
            )

    return rows


def collect_replayed_failure_rows(
    env: Any,
    rollout_dir: Path,
    *,
    episodes: int,
    seed: int = 0,
    max_steps: int | None = None,
    atol: float = 1e-5,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    progress_every: int = 25,
) -> list[dict[str, Any]]:
    """Replay saved rollout actions if live observations match saved vectors.

    Raises ``ValueError`` on any alignment mismatch.  Callers can catch this and
    fall back to fresh collection.
    """
    observations, action_labels, action_vectors = load_rollout_artifacts(rollout_dir)
    if max_steps is not None and max_steps <= 0:
        max_steps = None

    rows: list[dict[str, Any]] = []
    sample_index = 0
    for episode in range(episodes):
        if progress_callback is not None:
            progress_callback("episode_start", {"episode": episode, "episodes": episodes})
        try:
            obs = env.reset(seed=seed + episode)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed + episode)
            obs = env.reset()

        reward = getattr(env, "reward_range", (0.0, 0.0))[0]
        done = False
        step = 0
        chronic = _chronic_name(env)
        if progress_callback is not None:
            progress_callback(
                "episode_ready",
                {
                    "episode": episode,
                    "chronic_name": chronic,
                    "grid2op_current_step": getattr(obs, "current_step", None),
                    "grid2op_max_step": getattr(obs, "max_step", None),
                },
            )
        while not done and (max_steps is None or step < max_steps):
            if sample_index >= len(observations):
                raise ValueError("Saved rollout ended before replay completed.")
            if progress_callback is not None and (step == 0 or step % progress_every == 0):
                progress_callback(
                    "step_start",
                    {
                        "episode": episode,
                        "step": step,
                        "sample_index": sample_index,
                        "grid2op_current_step": getattr(obs, "current_step", None),
                        "max_rho": float(np.nanmax(obs.rho)) if hasattr(obs, "rho") else None,
                    },
                )

            saved_obs = np.asarray(observations[sample_index], dtype=np.float32)
            live_obs = np.asarray(obs.to_vect(), dtype=np.float32)
            if saved_obs.shape != live_obs.shape or not np.allclose(saved_obs, live_obs, atol=atol, rtol=0.0):
                raise ValueError(
                    "Saved rollout is not aligned with this environment/seed at "
                    f"sample_index={sample_index}, episode={episode}, step={step}."
                )

            action_class = int(action_labels[sample_index])
            action_vector = np.asarray(action_vectors[action_class], dtype=np.float32)
            action = env.action_space.from_vect(action_vector)

            row = _base_row(sample_index, episode, step, obs)
            row["chronic_name"] = chronic
            row["reward_before"] = float(reward)
            row["agent_action_class"] = action_class
            row.update(extract_observation_features(obs))

            next_obs, next_reward, done, info = env.step(action)
            row["reward_after"] = float(next_reward)
            row["done_after"] = bool(done)
            row["failure"] = map_failure_label(done, next_obs, info)
            rows.append(row)
            if progress_callback is not None and row["failure"]:
                progress_callback(
                    "failure",
                    {
                        "episode": episode,
                        "step": step,
                        "sample_index": sample_index,
                        "grid2op_current_step": getattr(next_obs, "current_step", None),
                        "grid2op_max_step": getattr(next_obs, "max_step", None),
                    },
                )

            obs = next_obs
            reward = next_reward
            step += 1
            sample_index += 1
            if progress_callback is not None:
                progress_callback(
                    "step_done",
                    {
                        "episode": episode,
                        "step": step,
                        "sample_index": sample_index,
                        "total_rows": len(rows),
                    },
                )
        if progress_callback is not None:
            progress_callback(
                "episode_end",
                {
                    "episode": episode,
                    "steps": step,
                    "total_rows": len(rows),
                    "done": bool(done),
                },
            )

    return rows


def iter_artifact_feature_rows(rollout_dir: Path, example_obs: Any | None = None) -> Iterator[dict[str, Any]]:
    """Yield feature-only rows from existing rollout arrays.

    These rows deliberately do not invent a failure label.  They are useful for
    quickly inspecting saved observations or for later joining with labels.
    """
    observations, action_labels, _actions = load_rollout_artifacts(rollout_dir)
    for sample_index, vector in enumerate(observations):
        row: dict[str, Any] = {
            "sample_index": sample_index,
            "agent_action_class": int(action_labels[sample_index]),
        }
        row.update(extract_vector_features(vector, example_obs))
        yield row


def write_failure_dataset_csv(rows: Iterable[Mapping[str, Any]], output_path: Path) -> int:
    """Write rows to CSV using first-seen stable column ordering."""
    materialized = list(rows)
    if not materialized:
        raise RuntimeError("No failure dataset rows were produced.")

    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "sample_index",
        "episode",
        "step",
        "grid2op_current_step",
        "grid2op_max_step",
        "chronic_name",
        "reward_before",
        "reward_after",
        "done_after",
        "agent_action_class",
        "failure",
    ]
    for name in preferred:
        if any(name in row for row in materialized):
            fieldnames.append(name)
            seen.add(name)
    for row in materialized:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)
