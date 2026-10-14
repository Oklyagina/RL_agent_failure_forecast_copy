"""One-hour-ahead failure forecasting for Grid2Op RL agents.

Pipeline (all labels come from historical RL-agent episodes, no extra
simulations):

1. ``run_historical_episode`` plays the agent and records every observation.
2. Forecaster data: pairs ``(features at t, injections at t+12)`` taken from the
   same episode, twelve real environment steps apart.
3. Failure labels: each *opponent attack that disconnected a line* is an event.
   The label says whether the agent survived the following
   ``outcome_steps`` steps (0) or the episode ended in a blackout (1).
   Events whose outcome cannot be observed are skipped, never labeled 0.
4. HGB features for an event at step ``s`` are built at ``t = s - 12`` from the
   forecast of the t+12 injections plus the grid state at ``t`` only.

Time convention: one step is ``STEP_MINUTES`` minutes; ``observations[k]`` is
the state at step ``k`` of the episode, before the agent acts.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .agent_runtime import call_agent


HORIZON_STEPS = 12
STEP_MINUTES = 5
DEFAULT_OUTCOME_STEPS = 12

LABEL_BLACKOUT = "blackout_after_attack"
LABEL_SURVIVED = "survived_after_attack"

CLASSIFIER_FEATURES = [
    "line_id_encoded",
    "sum_load_p",
    "sum_load_q",
    "sum_gen_p",
    "var_line_rho",
    "avg_line_rho",
    "max_line_rho",
    "nb_rho_ge_0.95",
    "load_gen_ratio",
    "fcast_sum_load_p",
    "fcast_sum_load_q",
    "fcast_sum_gen_p",
    "aleatoric_load_p_mean",
    "aleatoric_load_q_mean",
    "aleatoric_gen_p_mean",
    "epistemic_before",
]

# Columns derived inside ``prepare_classifier_dataframe`` / the predictor.
_DERIVED_FEATURES = {"line_id_encoded", "load_gen_ratio"}

REQUIRED_DATA_COLUMNS = {
    "line_disconnected",
    "failed",
    *(feature for feature in CLASSIFIER_FEATURES if feature not in _DERIVED_FEATURES),
}


@dataclass
class FailureForecastConfig:
    """Grid dimensions, line names and paths needed by the failure forecaster."""

    env_name: str
    agent_name: str
    no_loads: int
    no_gens: int
    line_names: Sequence[str]
    lines_to_test: Sequence[Any]
    gen_min: np.ndarray
    gen_max: np.ndarray
    artifact_dir: Path

    @classmethod
    def from_env(cls, env: Any, *, env_name: str, agent_name: str,
                 artifact_dir: str | Path, lines_to_test: Sequence[Any]) -> "FailureForecastConfig":
        """Build configuration from a live Grid2Op environment instance."""
        no_gens = int(getattr(env, "n_gen"))
        return cls(
            env_name=str(env_name),
            agent_name=str(agent_name),
            no_loads=int(getattr(env, "n_load")),
            no_gens=no_gens,
            line_names=[str(name) for name in getattr(env, "name_line")],
            lines_to_test=list(lines_to_test),
            gen_min=np.asarray(getattr(env, "gen_pmin", np.zeros(no_gens)), dtype=float),
            gen_max=np.asarray(getattr(env, "gen_pmax", np.full(no_gens, np.inf)), dtype=float),
            artifact_dir=Path(artifact_dir),
        )

    def line_id(self, line: Any) -> int:
        """Resolve a line index or name (with or without ``line_``) to its index."""
        if isinstance(line, (int, np.integer)):
            return int(line)
        wanted = normalize_line_name(line)
        for idx, name in enumerate(self.line_names):
            if normalize_line_name(name) == wanted:
                return idx
        raise ValueError(f"Unknown line: {line!r}")

    def line_name(self, line: Any) -> str:
        """Return the environment name of a line given its index or name."""
        return str(self.line_names[self.line_id(line)])


def normalize_line_name(line: Any) -> str:
    """Normalize Grid2Op line names for stable classifier metadata keys."""
    return str(line).replace("line_", "")


# --------------------------------------------------------------------------
# Features from observations
# --------------------------------------------------------------------------

def convert_to_cos_sin(value: float, period: float) -> Tuple[float, float]:
    """Encode a cyclic scalar as cosine/sine features."""
    value_cos = np.cos(2 * np.pi * value / period)
    value_sin = np.sin(2 * np.pi * value / period)
    return float(value_cos), float(value_sin)


def get_features_with_history(observations: Sequence[Any]) -> np.ndarray:
    """Build the forecaster input for the LAST observation of ``observations``.

    ``observations`` is the episode history up to and including the current
    step ``t`` (index ``t`` is the last element); nothing after ``t`` is used.
    Features: current load_p/load_q/gen_p, the same values 1h/1d/1w earlier
    (NaN when the episode is not that old yet) and calendar encodings.
    """
    if not observations:
        raise ValueError("observations must contain at least the current observation")
    obs = observations[-1]
    current_idx = len(observations) - 1

    def values(lag_steps: int, attr: str) -> np.ndarray:
        """Return ``attr`` at a fixed lag, or NaN padding before history exists."""
        idx = current_idx - lag_steps
        size = len(getattr(obs, attr))
        if idx < 0:
            return np.full(size, np.nan, dtype=float)
        return np.asarray(getattr(observations[idx], attr), dtype=float)

    feature_vec = np.concatenate([
        obs.load_p, obs.load_q, obs.gen_p,
        *(values(lag, attr)
          for lag in (12, 288, 2016)
          for attr in ("load_p", "load_q", "gen_p")),
    ]).astype(float)

    hour_cos, hour_sin = convert_to_cos_sin(getattr(obs, "hour_of_day", 0), 24)
    minute_cos, minute_sin = convert_to_cos_sin(getattr(obs, "minute_of_hour", 0), 60)
    dow_cos, dow_sin = convert_to_cos_sin(getattr(obs, "day_of_week", 0), 7)
    temporal = np.asarray([
        getattr(obs, "day", 1),
        hour_cos, hour_sin,
        minute_cos, minute_sin,
        dow_cos, dow_sin,
    ], dtype=float)
    return np.concatenate([feature_vec, temporal]).flatten()


def injections_vector(obs: Any) -> np.ndarray:
    """Return ``[load_p, load_q, gen_p]`` of one observation (forecast target)."""
    return np.concatenate([obs.load_p, obs.load_q, obs.gen_p]).astype(np.float32)


def compute_grid_stats(obs: Any) -> Dict[str, float]:
    """Compact grid statistics (injection sums and line-loading summary)."""
    rho = np.asarray(obs.rho, dtype=float)
    return {
        "sum_load_p": float(np.sum(obs.load_p)),
        "sum_load_q": float(np.sum(obs.load_q)),
        "sum_gen_p": float(np.sum(obs.gen_p)),
        "var_line_rho": float(np.var(rho)),
        "avg_line_rho": float(np.mean(rho)),
        "max_line_rho": float(np.max(rho)),
        "nb_rho_ge_0.95": int(np.sum(rho >= 0.95)),
    }


def eligible_lines(cfg: FailureForecastConfig, obs: Any, lines: Sequence[Any]) -> List[Any]:
    """Keep requested lines that are connected in ``obs`` (shared by train/inference)."""
    return [line for line in lines if bool(obs.line_status[cfg.line_id(line)])]


def _get_uncertainty(model_enn: Any, obs: Any,
                     get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]]) -> float:
    """ENN epistemic uncertainty of ``obs``; NaN only when no ENN is supplied."""
    if model_enn is None:
        return float("nan")
    if get_uncertainty_fn is None:
        raise ValueError("get_uncertainty_fn is required when model_enn is given")
    input_dim = int(getattr(model_enn, "input_dim", len(obs.to_vect())))
    vect = np.asarray(obs.to_vect(), dtype=np.float32)[:input_dim].reshape(1, -1)
    return float(get_uncertainty_fn(model_enn, vect))


# --------------------------------------------------------------------------
# Historical episodes
# --------------------------------------------------------------------------

@dataclass
class EpisodeTrace:
    """Everything observed while the RL agent played one episode.

    ``observations[k]`` is the state at step ``k`` (before the agent acts).
    ``attacked_lines[k]`` holds the lines the opponent attacked in the env step
    that produced ``observations[k]`` (empty for ``k == 0``).  If the episode
    ended in a blackout, the game-over step is NOT in ``observations`` (its
    state is invalid); it has index ``len(observations)`` and its attacked
    lines are in ``terminal_attacked_lines``.
    """

    observations: List[Any]
    attacked_lines: List[frozenset]
    ended_in_blackout: bool
    terminal_attacked_lines: frozenset = frozenset()


def reset_environment(env: Any, seed: int) -> Any:
    """Reset the environment with a seed across Grid2Op versions."""
    try:
        return env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
        return env.reset()


def is_blackout(done: bool, info: Optional[dict]) -> bool:
    """True when a finished env step is a game over (blackout).

    Assumption: Grid2Op reports a game over with a non-empty ``info["exception"]``;
    reaching the end of the scenario ends the episode with no exception.
    """
    return bool(done) and bool((info or {}).get("exception"))


def _attacked_lines(info: Optional[dict]) -> frozenset:
    """Line indices attacked by the opponent in one env step (from ``info``)."""
    attack = (info or {}).get("opponent_attack_line")
    if attack is None:
        return frozenset()
    return frozenset(int(idx) for idx in np.flatnonzero(np.asarray(attack)))


def run_historical_episode(env: Any, agent: Any, seed: int,
                           max_steps: Optional[int] = None) -> EpisodeTrace:
    """Play one episode with the agent and record what happened.

    At most ``max_steps`` environment steps are played (``None`` = until the
    episode ends).  If the step limit is hit the trace is simply shorter: the
    episode is neither a blackout nor completed.
    """
    obs = reset_environment(env, seed)
    if hasattr(agent, "reset"):
        agent.reset(obs)
    observations = [obs]
    attacked = [frozenset()]
    reward = getattr(env, "reward_range", (0.0, 0.0))[0]
    steps = 0
    while max_steps is None or steps < max_steps:
        action = call_agent(agent, observations[-1], reward=float(reward), done=False)
        obs, reward, done, info = env.step(action)
        steps += 1
        if done:
            return EpisodeTrace(
                observations, attacked,
                ended_in_blackout=is_blackout(done, info),
                terminal_attacked_lines=_attacked_lines(info),
            )
        observations.append(obs)
        attacked.append(_attacked_lines(info))
    return EpisodeTrace(observations, attacked, ended_in_blackout=False)


# --------------------------------------------------------------------------
# Forecaster training data (t -> t+12)
# --------------------------------------------------------------------------

def build_forecast_pairs(observations: Sequence[Any], *, stride: int,
                         offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(X_t, injections_{t+12})`` pairs from ONE episode.

    Origins are ``offset, offset + stride, ...`` and need ``t + 12`` to exist in
    the same episode, so no pair crosses an episode boundary.  ``X_t`` only uses
    observations up to ``t``.  Returns arrays with zero rows if nothing fits.
    """
    last_origin = len(observations) - 1 - HORIZON_STEPS
    origins = range(offset, last_origin + 1, stride)
    rows_x = [get_features_with_history(observations[: t + 1]) for t in origins]
    rows_y = [injections_vector(observations[t + HORIZON_STEPS]) for t in origins]
    if not rows_x:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    return (np.stack(rows_x).astype(np.float32),
            np.stack(rows_y).astype(np.float32))


def collect_forecaster_training_data(
    env: Any,
    agent: Any,
    *,
    episodes: int,
    seed: int = 0,
    max_steps: Optional[int] = None,
    sampling_stride: int = HORIZON_STEPS,
    return_episode_ids: bool = False,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> Any:
    """Collect hourly-stratified pairs ``X_t -> injections at t+12``.

    One origin is kept per ``sampling_stride`` steps.  The origin offset
    rotates by episode so every five-minute phase is represented without
    storing twelve overlapping rows per hour.  ``progress_callback`` receives
    ``(episode, steps_played, total_rows)``.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if sampling_stride <= 0:
        raise ValueError("sampling_stride must be positive")
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    episode_ids: List[np.ndarray] = []
    total_rows = 0
    for episode in range(episodes):
        trace = run_historical_episode(env, agent, seed + episode, max_steps)
        X, y = build_forecast_pairs(
            trace.observations, stride=sampling_stride,
            offset=episode % sampling_stride,
        )
        if len(X):
            all_x.append(X)
            all_y.append(y)
            episode_ids.append(np.full(len(X), episode, dtype=np.int32))
            total_rows += len(X)
        if progress_callback:
            progress_callback(episode, len(trace.observations), total_rows)
    if not all_x:
        raise RuntimeError("Forecast data collection produced no rows.")
    X = np.concatenate(all_x).astype(np.float32)
    y = np.concatenate(all_y).astype(np.float32)
    if return_episode_ids:
        return X, y, np.concatenate(episode_ids)
    return X, y


# --------------------------------------------------------------------------
# Forecasts and HGB feature rows
# --------------------------------------------------------------------------

def forecast_injections(
    observations: Sequence[Any],
    mean_model: Any,
    aleatoric_model: Any,
    cfg: FailureForecastConfig,
    *,
    model_enn: Any = None,
    get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """Forecast load_p, load_q and gen_p at ``t + 12`` from history up to ``t``.

    ``observations[-1]`` is the current observation ``t``.  No power flow is
    run.  Returns the forecast, its uncertainty summaries and the ENN
    uncertainty of the current observation.
    """
    x_t = get_features_with_history(observations)
    y_pred = np.asarray(mean_model.predict([x_t])[0], dtype=float)
    z_pred = np.asarray(aleatoric_model.predict([x_t])[0], dtype=float)
    n_load = cfg.no_loads
    sigma = np.sqrt(np.expm1(np.clip(z_pred, 0.0, None)) + 1e-12)
    return {
        "load_p": y_pred[:n_load],
        "load_q": y_pred[n_load: 2 * n_load],
        "gen_p": np.clip(y_pred[2 * n_load:], cfg.gen_min, cfg.gen_max),
        "aleatoric_load_p_mean": float(np.mean(sigma[:n_load])),
        "aleatoric_load_q_mean": float(np.mean(sigma[n_load: 2 * n_load])),
        "aleatoric_gen_p_mean": float(np.mean(sigma[2 * n_load:])),
        "epistemic_before": _get_uncertainty(model_enn, observations[-1], get_uncertainty_fn),
    }


def measured_future_as_forecast(
    observations: Sequence[Any],
    origin: int,
    *,
    model_enn: Any = None,
    get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """EXPERIMENTAL benchmark (supervisor option 2): perfect forecast.

    Uses the measured injections at ``origin + 12`` as the "forecast".  This
    reads the future, so it must never be used for real training or inference.
    """
    future = observations[origin + HORIZON_STEPS]
    return {
        "load_p": np.asarray(future.load_p, dtype=float),
        "load_q": np.asarray(future.load_q, dtype=float),
        "gen_p": np.asarray(future.gen_p, dtype=float),
        "aleatoric_load_p_mean": 0.0,
        "aleatoric_load_q_mean": 0.0,
        "aleatoric_gen_p_mean": 0.0,
        "epistemic_before": _get_uncertainty(model_enn, observations[origin], get_uncertainty_fn),
    }


def build_feature_row(obs: Any, line: Any, forecast: Dict[str, Any],
                      cfg: FailureForecastConfig) -> Dict[str, Any]:
    """Combine the state at ``t`` with the t+12 forecast for one candidate line.

    Shared by training and inference so both see identical features.
    """
    current = compute_grid_stats(obs)
    sum_load = current["sum_load_p"]
    sum_gen = current["sum_gen_p"]
    return {
        "horizon_steps": HORIZON_STEPS,
        "horizon_minutes": HORIZON_STEPS * STEP_MINUTES,
        "line_disconnected": cfg.line_name(line),
        "line_id": cfg.line_id(line),
        "sum_load_p": sum_load,
        "sum_load_q": current["sum_load_q"],
        "sum_gen_p": sum_gen,
        "var_line_rho": current["var_line_rho"],
        "avg_line_rho": current["avg_line_rho"],
        "max_line_rho": current["max_line_rho"],
        "nb_rho_ge_0.95": float(current["nb_rho_ge_0.95"]),
        "load_gen_ratio": sum_load / (sum_gen + 1e-6),
        "fcast_sum_load_p": float(np.sum(forecast["load_p"])),
        "fcast_sum_load_q": float(np.sum(forecast["load_q"])),
        "fcast_sum_gen_p": float(np.sum(forecast["gen_p"])),
        "aleatoric_load_p_mean": forecast["aleatoric_load_p_mean"],
        "aleatoric_load_q_mean": forecast["aleatoric_load_q_mean"],
        "aleatoric_gen_p_mean": forecast["aleatoric_gen_p_mean"],
        "epistemic_before": forecast["epistemic_before"],
    }


# --------------------------------------------------------------------------
# Historical failure labels
# --------------------------------------------------------------------------

def find_attack_events(trace: EpisodeTrace, cfg: FailureForecastConfig,
                       lines: Sequence[Any]) -> List[Tuple[int, int]]:
    """Return ``(step, line_id)`` for each attack that really disconnected a line.

    An event needs the opponent to attack the line at ``step`` AND the line to
    go from connected (step - 1) to disconnected (step).  Plain trips, planned
    maintenance and the agent's own switching are not events.
    """
    wanted = {cfg.line_id(line) for line in lines}
    events = []
    for step in range(1, len(trace.observations)):
        before, after = trace.observations[step - 1], trace.observations[step]
        for line_id in sorted(trace.attacked_lines[step] & wanted):
            if bool(before.line_status[line_id]) and not bool(after.line_status[line_id]):
                events.append((step, line_id))
    return events


def label_attack_event(trace: EpisodeTrace, step: int, line_id: int,
                       outcome_steps: int) -> Tuple[Optional[int], str]:
    """Label one attack event from what the agent did next in the episode.

    The agent sees the attacked grid at ``step`` and acts at ``step`` ..
    ``step + outcome_steps - 1``.  Returns ``(label, reason)``:

    * ``(1, blackout_after_attack)``: the episode ended in a blackout within
      ``outcome_steps`` steps after the attack;
    * ``(0, survived_after_attack)``: the episode was still running (no
      blackout) ``outcome_steps`` steps after the attack;
    * ``(None, reason)``: no reliable label - another attack interferes, or the
      episode stopped (end of data / step limit) before the outcome was known.
    """
    n_obs = len(trace.observations)
    window_end = step + outcome_steps
    for other_step in range(step, min(window_end, n_obs - 1) + 1):
        others = set(trace.attacked_lines[other_step])
        if other_step == step:
            others.discard(line_id)
        if others:
            return None, "overlapping_attack"
    blackout_step = n_obs  # index of the game-over step, if any

    if trace.ended_in_blackout and blackout_step <= window_end:
        if trace.terminal_attacked_lines:
            return None, "overlapping_attack"
        return 1, LABEL_BLACKOUT
    if window_end <= n_obs - 1:
        return 0, LABEL_SURVIVED
    return None, "outcome_not_observed"


def collect_failure_forecast_rows(
    agent: Any,
    env: Any,
    mean_model: Any,
    aleatoric_model: Any,
    cfg: FailureForecastConfig,
    *,
    episodes: int,
    seed: int = 0,
    outcome_steps: int = DEFAULT_OUTCOME_STEPS,
    max_steps: Optional[int] = None,
    model_enn: Any = None,
    get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
    perfect_forecast: bool = False,
    progress_callback: Optional[Callable[[int, int, Counter], None]] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Collect labeled HGB rows from historical agent episodes.

    For every labeled attack event at step ``s`` on line ``l`` one row is built
    at the forecast origin ``t = s - 12`` using only observations up to ``t``
    plus the forecast of the t+12 injections.  ``perfect_forecast=True`` is the
    experimental "measured values as forecasts" benchmark.

    Returns ``(rows, stats)``; ``stats`` counts episodes, attack events, rows,
    positives/negatives and every skip reason.  ``progress_callback`` receives
    ``(episode, steps_played, stats)``.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if outcome_steps <= 0:
        raise ValueError("outcome_steps must be positive")
    rows: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    for episode in range(episodes):
        trace = run_historical_episode(env, agent, seed + episode, max_steps)
        stats["episodes"] += 1
        stats["blackout_episodes"] += int(trace.ended_in_blackout)
        for step, line_id in find_attack_events(trace, cfg, cfg.lines_to_test):
            stats["attack_events"] += 1
            origin = step - HORIZON_STEPS
            label, reason = label_attack_event(trace, step, line_id, outcome_steps)
            if label is None:
                stats["skipped_" + reason] += 1
                continue
            if origin < 0:
                stats["skipped_no_forecast_origin"] += 1
                continue
            if not bool(trace.observations[origin].line_status[line_id]):
                stats["skipped_line_down_at_origin"] += 1
                continue
            history = trace.observations[: origin + 1]
            if perfect_forecast:
                forecast = measured_future_as_forecast(
                    trace.observations, origin,
                    model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn)
            else:
                forecast = forecast_injections(
                    history, mean_model, aleatoric_model, cfg,
                    model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn)
            row = build_feature_row(history[-1], line_id, forecast, cfg)
            row.update({
                "episode": episode,
                "step": origin,
                "event_step": step,
                "failed": label,
                "label_reason": reason,
                "outcome_steps": outcome_steps,
                "forecast_source": "measured_future" if perfect_forecast else "forecaster",
            })
            rows.append(row)
            stats["rows"] += 1
            stats["positive" if label else "negative"] += 1
        if progress_callback:
            progress_callback(episode, len(trace.observations), stats)
    return rows, stats


def save_forecaster_training_data(
    out_dir: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    episode_ids: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Persist forecast training arrays beside the downstream classifier data."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x_path = out / "forecast_X.npy"
    y_path = out / "forecast_y.npy"
    np.save(x_path, np.asarray(X, dtype=np.float32))
    np.save(y_path, np.asarray(y, dtype=np.float32))
    paths = {"X": str(x_path), "y": str(y_path)}
    if episode_ids is not None:
        episode_path = out / "forecast_episode_ids.npy"
        np.save(episode_path, np.asarray(episode_ids, dtype=np.int32))
        paths["episode_ids"] = str(episode_path)
    return paths


def _fixed_tuning_indices(
    row_count: int,
    episode_ids: Optional[np.ndarray],
    *,
    max_subsample: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create one deterministic tuning split, grouped by episode when available."""
    if row_count < 2:
        raise ValueError("Optuna tuning requires at least two forecast rows.")
    if max_subsample <= 0:
        raise ValueError("max_subsample must be positive.")
    rng = np.random.default_rng(seed)

    if episode_ids is not None:
        groups = np.asarray(episode_ids)
        if groups.ndim != 1 or len(groups) != row_count:
            raise ValueError("episode_ids must be one-dimensional and match X/y rows.")
        unique_episodes = np.unique(groups)
        if len(unique_episodes) < 2:
            raise ValueError("Episode-separated tuning requires at least two episodes.")
        shuffled = rng.permutation(unique_episodes)
        validation_episode_count = min(
            len(shuffled) - 1,
            max(1, int(round(0.2 * len(shuffled)))),
        )
        validation_episodes = shuffled[:validation_episode_count]
        validation_mask = np.isin(groups, validation_episodes)
        train_pool = np.flatnonzero(~validation_mask)
        validation_pool = np.flatnonzero(validation_mask)
    else:
        shuffled_rows = rng.permutation(row_count)
        validation_count = min(
            row_count - 1,
            max(1, int(round(0.2 * row_count))),
        )
        validation_pool = shuffled_rows[:validation_count]
        train_pool = shuffled_rows[validation_count:]

    train_count = min(int(max_subsample), len(train_pool))
    validation_count = min(max(1, int(max_subsample) // 5), len(validation_pool))
    train_idx = rng.choice(train_pool, size=train_count, replace=False)
    validation_idx = rng.choice(
        validation_pool,
        size=validation_count,
        replace=False,
    )
    return np.asarray(train_idx, dtype=int), np.asarray(validation_idx, dtype=int)


def train_mean_forecaster(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_path: str | Path,
    n_trials: int = 20,
    max_subsample: int = 5000,
    seed: int = 42,
    episode_ids: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    patience: int = 10,
    min_improvement: float = 0.005,
    n_jobs: Optional[int] = -1,
) -> Any:
    """Train the mean forecaster for injections twelve steps ahead.

    Optuna tuning stops early once the best validation RMSE has not improved
    by more than ``min_improvement`` (relative) for ``patience`` trials; set
    ``patience`` to 0 to always run ``n_trials``. ``n_jobs`` parallelises the
    per-output regressors (-1 = all cores) and does not change the fitted models.
    """
    import time
    import joblib
    from tqdm import tqdm
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.multioutput import MultiOutputRegressor

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if n_trials > 0:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tune_train_idx, tune_validation_idx = _fixed_tuning_indices(
            len(X),
            episode_ids,
            max_subsample=int(max_subsample),
            seed=int(seed),
        )

        def objective(trial: Any) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
                "early_stopping": True,
                "random_state": seed,
            }
            model = MultiOutputRegressor(
                HistGradientBoostingRegressor(**params), n_jobs=n_jobs
            )
            model.fit(X[tune_train_idx], y[tune_train_idx])
            pred = model.predict(X[tune_validation_idx])
            return float(np.sqrt(mean_squared_error(y[tune_validation_idx], pred)))

        study = optuna.create_study(direction="minimize")
        total_trials = int(n_trials)
        stall = {"best": float("inf"), "trials": 0}
        bar = tqdm(total=total_trials, desc="[mean] Optuna", unit="trial")

        def report_progress(study: Any, trial: Any) -> None:
            """Update the progress bar and stop the study once tuning plateaus."""
            best = float(study.best_value)
            if best < stall["best"] * (1.0 - float(min_improvement)):
                stall["trials"] = 0
            else:
                stall["trials"] += 1
            stall["best"] = min(stall["best"], best)
            bar.update(1)
            bar.set_postfix(
                last=f"{trial.value:.4f}",
                best=f"{best:.4f}",
                no_gain=f"{stall['trials']}/{patience}" if patience > 0 else "off",
            )
            if progress_callback is not None:
                progress_callback(trial.number + 1, total_trials, best)
            if patience > 0 and stall["trials"] >= patience:
                bar.set_description("[mean] Optuna (early stop)")
                study.stop()

        try:
            study.optimize(
                objective,
                n_trials=total_trials,
                show_progress_bar=False,
                callbacks=[report_progress],
            )
        finally:
            bar.close()
        params = dict(study.best_params)
        params["early_stopping"] = True
        params["random_state"] = seed
    else:
        params = {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_depth": 6,
            "l2_regularization": 2.0,
            "early_stopping": True,
            "random_state": seed,
        }

    print(f"[mean] final fit on {len(X)} rows, {y.shape[1]} outputs...", flush=True)
    started = time.perf_counter()
    model = MultiOutputRegressor(HistGradientBoostingRegressor(**params), n_jobs=n_jobs)
    model.fit(X, y)
    print(f"[mean] final fit done in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def train_aleatoric_forecaster(
    X: np.ndarray,
    y: np.ndarray,
    mean_model: Any,
    *,
    model_path: str | Path,
) -> Any:
    """Train the legacy residual-variance forecaster."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    y_pred = np.asarray(mean_model.predict(X), dtype=np.float32)
    squared_residuals = (y - y_pred) ** 2
    upper = np.percentile(squared_residuals, 99.9, axis=0)
    squared_residuals = np.clip(squared_residuals, 0.0, upper)
    z = np.log1p(squared_residuals)
    base = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_depth=6,
        l2_regularization=2.0,
        early_stopping=True,
    )
    model = MultiOutputRegressor(base)
    model.fit(X, z)
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def write_rows_csv(rows: Iterable[Dict[str, Any]], output_path: str | Path) -> int:
    """Write heterogeneous feature rows with a stable preferred column order."""
    rows = list(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return 0
    preferred = [
        "episode", "step", "event_step", "horizon_steps", "horizon_minutes",
        "line_disconnected", "line_id", "failed", "label_reason",
    ]
    fieldnames = preferred + sorted(
        key for row in rows for key in row if key not in preferred
    )
    fieldnames = list(dict.fromkeys(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _line_map_from_series(values: Any) -> Dict[str, int]:
    """Build a first-seen line-name encoding compatible with HGB features."""
    mapping: Dict[str, int] = {}
    for value in values:
        name = normalize_line_name(value)
        if name not in mapping:
            mapping[name] = len(mapping)
    return mapping


def prepare_classifier_dataframe(data: str | Path | pd.DataFrame,
                                 line_map: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Validate raw rows and derive classifier-only numeric columns."""
    import pandas as pd

    df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data.copy()
    missing = sorted(REQUIRED_DATA_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    line_map = _line_map_from_series(df["line_disconnected"]) if line_map is None else line_map
    df["line_id_encoded"] = [
        int(line_map.get(normalize_line_name(value), -1))
        for value in df["line_disconnected"]
    ]
    if (df["line_id_encoded"] < 0).any():
        raise ValueError("Dataset contains line names absent from line_map.")
    df["load_gen_ratio"] = (
        pd.to_numeric(df["sum_load_p"], errors="coerce")
        / (pd.to_numeric(df["sum_gen_p"], errors="coerce") + 1e-6)
    )
    df["label"] = pd.to_numeric(df["failed"], errors="raise").astype(int)
    invalid = sorted(set(df["label"].unique()) - {0, 1})
    if invalid:
        raise ValueError(f"'failed' must be binary 0/1; found {invalid}")
    for feature in CLASSIFIER_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    df.attrs["line_map"] = {str(k): int(v) for k, v in line_map.items()}
    return df


def _failure_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Extract the probability column corresponding to failure class 1."""
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    matches = np.where(classes == 1)[0]
    if len(matches) != 1:
        raise ValueError("Classifier does not expose failure class 1.")
    return proba[:, int(matches[0])]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray,
             probs: np.ndarray) -> Dict[str, Any]:
    """Compute the legacy false-alarm, oversight, AUC, and confusion metrics."""
    from sklearn.metrics import confusion_matrix, roc_auc_score

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fa = (fp / (tn + fp) * 100.0) if (tn + fp) else 0.0
    oversight = (fn / (tp + fn) * 100.0) if (tp + fn) else 0.0
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = float("nan")
    return {
        "false_alarm_pct": float(fa),
        "oversight_pct": float(oversight),
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "test_rows": int(len(y_true)),
    }


def _split_classifier_rows(
    dataframe: pd.DataFrame,
    *,
    seed: int,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Split classifier rows while keeping each episode in exactly one set."""
    from sklearn.model_selection import train_test_split

    X = dataframe[CLASSIFIER_FEATURES].copy()
    y = dataframe["label"].astype(int)
    if "episode" in dataframe.columns and dataframe["episode"].nunique() >= 2:
        episodes = pd.to_numeric(dataframe["episode"], errors="raise").astype(int)
        unique_episodes = np.asarray(sorted(episodes.unique()), dtype=int)
        episode_has_failure = (
            pd.DataFrame({"episode": episodes, "label": y})
            .groupby("episode", sort=True)["label"]
            .max()
            .reindex(unique_episodes)
            .astype(int)
        )
        stratify = None
        class_counts = episode_has_failure.value_counts()
        if episode_has_failure.nunique() > 1 and int(class_counts.min()) >= 2:
            stratify = episode_has_failure.to_numpy()
        train_episodes, test_episodes = train_test_split(
            unique_episodes,
            test_size=float(test_size),
            random_state=int(seed),
            stratify=stratify,
        )
        train_mask = episodes.isin(train_episodes)
        test_mask = episodes.isin(test_episodes)
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
        split_info = {
            "strategy": "episode_grouped",
            "test_fraction": float(test_size),
            "train_episodes": int(len(train_episodes)),
            "test_episodes": int(len(test_episodes)),
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
        }
    else:
        stratify = y if int(y.value_counts().min()) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=int(seed),
            stratify=stratify,
        )
        split_info = {
            "strategy": "row_stratified",
            "test_fraction": float(test_size),
            "train_episodes": None,
            "test_episodes": None,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }

    if y_train.nunique() < 2:
        raise ValueError("Classifier training split does not contain both classes.")
    if y_test.nunique() < 2:
        raise ValueError("Classifier test split does not contain both classes.")
    return X_train, X_test, y_train, y_test, split_info


def train_failure_classifier(
    dataframe: pd.DataFrame,
    *,
    model_path: str | Path,
    threshold: float = 0.5,
    seed: int = 42,
    env_name: str = "",
    agent_name: str = "",
    max_iter: int = 300,
) -> Tuple[Any, Dict[str, Any]]:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    if "label" not in dataframe.columns:
        raise ValueError("Use prepare_classifier_dataframe before training.")
    y = dataframe["label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Failure classifier requires both classes.")
    if int(y.value_counts().min()) < 2:
        raise ValueError("Failure classifier needs at least two samples per class.")

    X_train, X_test, y_train, y_test, split_info = _split_classifier_rows(
        dataframe,
        seed=seed,
    )
    cat_indices = [CLASSIFIER_FEATURES.index("line_id_encoded")]
    model = HistGradientBoostingClassifier(
        categorical_features=cat_indices,
        class_weight="balanced",
        learning_rate=0.05,
        max_iter=int(max_iter),
        random_state=seed,
    )
    model.fit(X_train, y_train)
    probs = _failure_probability(model, X_test)
    preds = (probs >= float(threshold)).astype(int)
    metrics = _metrics(y_test.to_numpy(), preds, probs)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "environment": env_name,
        "agent": agent_name,
        "features": CLASSIFIER_FEATURES,
        "line_map": dataframe.attrs.get("line_map", {}),
        "failure_class": 1,
        "decision_threshold": float(threshold),
        "forecast_horizon_steps": HORIZON_STEPS,
        "forecast_horizon_minutes": HORIZON_STEPS * STEP_MINUTES,
        "split": split_info,
        "outcome_steps": (
            int(dataframe["outcome_steps"].iloc[0])
            if "outcome_steps" in dataframe.columns and len(dataframe) else None
        ),
        "label_definition": (
            "1 = opponent attack disconnected the line and the agent's episode "
            "ended in a blackout within outcome_steps; 0 = agent still running "
            "after outcome_steps; unknown outcomes are excluded"
        ),
        "label_counts": {
            str(label): int(count) for label, count in y.value_counts().items()
        },
        "label_reason_counts": {
            str(reason): int(count)
            for reason, count in (
                dataframe["label_reason"].value_counts().items()
                if "label_reason" in dataframe.columns
                else []
            )
        },
        "model_params": {
            "class_weight": "balanced",
            "learning_rate": 0.05,
            "max_iter": int(max_iter),
            "random_state": int(seed),
        },
        "metrics": metrics,
    }
    metadata_path = model_path.with_name("failure_classifier_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, {"metadata_path": str(metadata_path), **metadata}



class FailureForecastPredictor:
    """Runtime predictor returning a binary failure class and probability.

    Scores "the agent fails if line l is lost around t+12".  It uses only the
    observation history up to ``t`` and the saved forecasting models; no
    simulation and no agent call is made.
    """

    def __init__(self, model: Any, metadata: Dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.features = list(metadata.get("features", []))
        if self.features != CLASSIFIER_FEATURES:
            raise ValueError(
                "Classifier features do not match this code version; "
                "retrain the failure classifier."
            )
        self.line_map = {
            str(k): int(v) for k, v in metadata.get("line_map", {}).items()
        }
        self.threshold = float(metadata.get("decision_threshold", 0.5))

    @classmethod
    def load(cls, model_path: str | Path, metadata_path: Optional[str | Path] = None) -> "FailureForecastPredictor":
        """Load the persisted classifier and its adjacent metadata JSON."""
        import joblib

        model_path = Path(model_path)
        if metadata_path is None:
            metadata_path = model_path.with_name("failure_classifier_metadata.json")
        model = joblib.load(model_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        return cls(model, metadata)

    def features_from_row(self, row: Dict[str, Any] | pd.Series) -> pd.DataFrame:
        """Convert one feature row into the exact dataframe the model expects.

        Raises if a feature is missing (NaN is only a valid *value*, e.g. when
        no ENN is used, never a stand-in for a missing column).
        """
        import pandas as pd

        row_dict = dict(row)
        line_name = normalize_line_name(row_dict.get("line_disconnected", ""))
        if "line_id_encoded" not in row_dict:
            if line_name not in self.line_map:
                raise ValueError(f"Unknown line for failure classifier: {line_name!r}")
            row_dict["line_id_encoded"] = self.line_map[line_name]
        if "load_gen_ratio" not in row_dict and {"sum_load_p", "sum_gen_p"} <= set(row_dict):
            row_dict["load_gen_ratio"] = (
                float(row_dict["sum_load_p"]) / (float(row_dict["sum_gen_p"]) + 1e-6)
            )
        missing = [feature for feature in self.features if feature not in row_dict]
        if missing:
            raise ValueError("Feature row is missing: " + ", ".join(missing))
        return pd.DataFrame([{feature: row_dict[feature] for feature in self.features}])

    def predict_from_features(self, row: Dict[str, Any] | pd.Series) -> Dict[str, Any]:
        """Predict failure directly from an already-built feature row."""
        X = self.features_from_row(row)
        prob = float(_failure_probability(self.model, X)[0])
        return {
            "failure_prediction": 1 if prob >= self.threshold else 0,
            "failure_probability": prob,
        }

    def predict(
        self,
        obs: Any,
        observations: Sequence[Any],
        line: Any,
        mean_model: Any,
        aleatoric_model: Any,
        cfg: FailureForecastConfig,
        *,
        model_enn: Any = None,
        get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
    ) -> Dict[str, Any]:
        """Score one line; see ``predict_many``."""
        prediction = self.predict_many(
            obs, observations, [line], mean_model, aleatoric_model, cfg,
            model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn,
        )[0]
        return {
            "failure_prediction": prediction["failure_prediction"],
            "failure_probability": prediction["failure_probability"],
        }

    def predict_many(
        self,
        obs: Any,
        observations: Sequence[Any],
        lines: Sequence[Any],
        mean_model: Any,
        aleatoric_model: Any,
        cfg: FailureForecastConfig,
        *,
        model_enn: Any = None,
        get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
    ) -> List[Dict[str, Any]]:
        """Score each requested, currently connected line from one t+12 forecast.

        ``observations`` is the episode history up to the current step and its
        last element must be ``obs``.  Lines the classifier never saw during
        training raise an error.
        """
        if not observations or observations[-1] is not obs:
            raise ValueError("observations must end with the current observation")
        forecast = forecast_injections(
            observations, mean_model, aleatoric_model, cfg,
            model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn,
        )
        predictions: List[Dict[str, Any]] = []
        for line in eligible_lines(cfg, obs, lines):
            row = build_feature_row(obs, line, forecast, cfg)
            predictions.append({
                "line": cfg.line_name(line),
                **self.predict_from_features(row),
            })
        return predictions
