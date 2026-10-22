"""One-hour-ahead failure forecasting for Grid2Op RL agents.

Pipeline (legacy logic, horizon corrected from t+1 to t+12):

1. ``run_historical_episode`` plays the agent and records every observation.
2. Forecaster data: pairs ``(features at t, injections at t+12)`` for every
   step ``t`` of the episode (sampling stride 1, as in the legacy code).
3. Failure labels (``analyze_disconnection_effect``): at sampled steps ``t`` the
   agent plays 12 real steps in a copy of the environment; at t+12 each
   candidate line is disconnected in a further copy and ``failed`` is the
   Grid2Op ``done`` flag of that step (``failed=1`` for every line when the
   episode already ended before t+12).
4. HGB features: grid state at ``t``, summary of the powerflow simulated with
   the forecast t+12 injections, forecaster aleatoric uncertainty and ENN
   epistemic uncertainty before/after.

Time convention: one step is ``STEP_MINUTES`` minutes; ``observations[k]`` is
the state at step ``k`` of the episode, before the agent acts.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .agent_runtime import call_agent


HORIZON_STEPS = 12
STEP_MINUTES = 5
DEFAULT_SAMPLE_EVERY = 20
DEFAULT_MIN_STEP = 12
DEFAULT_CLASSIFIER_TRIALS = 500
DEFAULT_SEED_START = 700
MIN_GROUPED_EPISODES = 6
LABEL_DISCONNECTION_TEST = "forced_disconnection_at_t12"
LABEL_EPISODE_ENDED = "episode_ended_before_t12"
MEAN_BEST_PARAMS_FILENAME = "mean_forecaster_best_params.json"

# Legacy simulation phases: (name, opponent attack cooldown or None = env default).
LEGACY_PHASES: Tuple[Tuple[str, Optional[int]], ...] = (
    ("Standard", None),
    ("HeavyAttack_1", 288),
    ("HeavyAttack_2", 128),
)

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
    "fcast_var_line_rho",
    "fcast_avg_line_rho",
    "fcast_max_line_rho",
    "fcast_nb_rho_ge_0.95",
    "aleatoric_load_p_mean",
    "aleatoric_load_q_mean",
    "aleatoric_gen_p_mean",
    "epistemic_before",
    "epistemic_after",
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
    sampling_stride: int = 1,
    return_episode_ids: bool = False,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> Any:
    """Collect pairs ``X_t -> injections at t+12`` for every step ``t``.

    Legacy behaviour: ``sampling_stride=1`` keeps every origin.  A larger
    stride keeps one origin per ``sampling_stride`` steps with the offset
    rotating by episode.  ``progress_callback`` receives
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


def simulate_forecast_state(
    obs: Any,
    forecast: Dict[str, Any],
    env: Any,
    *,
    model_enn: Any = None,
    get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """Run a powerflow with the forecast t+12 injections (legacy behaviour).

    The forecast is injected through ``obs._forecasted_inj`` (timestamp
    ``t + 60 min``) and a do-nothing action is simulated.  The resulting
    observation gives the ``fcast_*`` grid statistics and ``epistemic_after``.
    Any failure leaves them NaN, as in the legacy code.
    """
    stats: Dict[str, Any] = {
        "fcast_sum_load_p": np.nan,
        "fcast_sum_load_q": np.nan,
        "fcast_sum_gen_p": np.nan,
        "fcast_var_line_rho": np.nan,
        "fcast_avg_line_rho": np.nan,
        "fcast_max_line_rho": np.nan,
        "fcast_nb_rho_ge_0.95": np.nan,
        "epistemic_after": float("nan"),
    }
    try:
        now_ts = obs.get_time_stamp()
        next_ts = now_ts + datetime.timedelta(minutes=HORIZON_STEPS * STEP_MINUTES)
        obs_copy = obs.copy()
        obs_copy._forecasted_inj = [
            (now_ts, {"injection": {
                "load_p": obs.load_p, "load_q": obs.load_q,
                "prod_p": obs.gen_p, "prod_v": obs.gen_v}}),
            (next_ts, {"injection": {
                "load_p": forecast["load_p"], "load_q": forecast["load_q"],
                "prod_p": forecast["gen_p"], "prod_v": obs.gen_v}}),
        ]
        sim_obs, _, _, _ = obs_copy.simulate(env.action_space({}))
        sim_stats = compute_grid_stats(sim_obs)
        for key, value in sim_stats.items():
            stats["fcast_" + key] = value
        stats["epistemic_after"] = _get_uncertainty(model_enn, sim_obs, get_uncertainty_fn)
    except Exception as exc:  # legacy: report and continue with NaN statistics
        print(f"[ERROR] Forecast / Powerflow Injection: {exc}")
    return stats


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
        "fcast_sum_load_p": forecast["fcast_sum_load_p"],
        "fcast_sum_load_q": forecast["fcast_sum_load_q"],
        "fcast_sum_gen_p": forecast["fcast_sum_gen_p"],
        "fcast_var_line_rho": forecast["fcast_var_line_rho"],
        "fcast_avg_line_rho": forecast["fcast_avg_line_rho"],
        "fcast_max_line_rho": forecast["fcast_max_line_rho"],
        "fcast_nb_rho_ge_0.95": forecast["fcast_nb_rho_ge_0.95"],
        "aleatoric_load_p_mean": forecast["aleatoric_load_p_mean"],
        "aleatoric_load_q_mean": forecast["aleatoric_load_q_mean"],
        "aleatoric_gen_p_mean": forecast["aleatoric_gen_p_mean"],
        "epistemic_before": forecast["epistemic_before"],
        "epistemic_after": forecast["epistemic_after"],
    }


# --------------------------------------------------------------------------
# Legacy failure labels: forced line disconnection at t+12
# --------------------------------------------------------------------------

def analyze_disconnection_effect(
    env: Any,
    agent: Any,
    obs: Any,
    observations: Sequence[Any],
    episode: int,
    mean_model: Any,
    aleatoric_model: Any,
    cfg: FailureForecastConfig,
    lines: Sequence[Any],
    *,
    model_enn: Any = None,
    get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
) -> List[Dict[str, Any]]:
    """Build one labeled row per candidate line for the current step ``t``.

    1. Forecast t+12 injections and simulate a powerflow with them
       (``fcast_*`` statistics and ``epistemic_after``).
    2. Let the agent play ``HORIZON_STEPS`` real steps in ``env.copy()``.  At
       t+12 disconnect each line in a further copy; ``failed`` is the ``done``
       flag of that step.  If the episode ended before t+12 every line gets
       ``failed=1`` (legacy behaviour).

    ``observations`` is the episode history up to ``t`` (last element ``obs``).
    Lines already disconnected at ``t`` are skipped.
    """
    try:
        forecast = forecast_injections(
            observations, mean_model, aleatoric_model, cfg,
            model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn)
    except Exception as exc:  # legacy: continue with zero sigma / NaN stats
        print(f"[ERROR] Forecast: {exc}")
        forecast = {
            "load_p": np.zeros(cfg.no_loads), "load_q": np.zeros(cfg.no_loads),
            "gen_p": np.zeros(cfg.no_gens),
            "aleatoric_load_p_mean": 0.0, "aleatoric_load_q_mean": 0.0,
            "aleatoric_gen_p_mean": 0.0,
            "epistemic_before": float("nan"),
        }
        forecast["epistemic_before"] = _get_uncertainty_safe(model_enn, obs, get_uncertainty_fn)
    else:
        forecast.update(simulate_forecast_state(
            obs, forecast, env, model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn))
    for key in ("fcast_sum_load_p", "fcast_sum_load_q", "fcast_sum_gen_p",
                "fcast_var_line_rho", "fcast_avg_line_rho", "fcast_max_line_rho",
                "fcast_nb_rho_ge_0.95", "epistemic_after"):
        forecast.setdefault(key, float("nan"))

    agent_env = env.copy()
    current_obs = obs
    reached_t12 = True
    reward = env.reward_range[0]
    done = False
    for _ in range(HORIZON_STEPS):
        try:
            action = call_agent(agent, current_obs, reward=float(reward), done=done)
        except Exception:
            action = agent_env.action_space({})
        current_obs, reward, done, _ = agent_env.step(action)
        if done:
            reached_t12 = False
            break

    rows: List[Dict[str, Any]] = []
    for line in eligible_lines(cfg, obs, lines):
        line_id = cfg.line_id(line)
        if reached_t12:
            test_env = agent_env.copy()
            _, _, failed, _ = test_env.step(
                env.action_space({"set_line_status": [(line_id, -1)]}))
            test_env.close()
            reason = LABEL_DISCONNECTION_TEST
        else:
            failed = True
            reason = LABEL_EPISODE_ENDED
        row = build_feature_row(obs, line_id, forecast, cfg)
        row.update({
            "episode": episode,
            "step": int(obs.current_step),
            "event_step": int(obs.current_step) + HORIZON_STEPS,
            "failed": 1 if failed else 0,
            "label_reason": reason,
        })
        rows.append(row)
    agent_env.close()
    return rows


def _get_uncertainty_safe(model_enn: Any, obs: Any,
                          get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]]) -> float:
    try:
        return _get_uncertainty(model_enn, obs, get_uncertainty_fn)
    except Exception as exc:
        print(f"[ERROR] ENN Epistemic Calculation (Before): {exc}")
        return float("nan")


def opponent_env_kwargs(lines: Sequence[Any], cooldown: int, seed: int = 42) -> Dict[str, Any]:
    """Grid2Op ``make`` kwargs of the legacy attack phases (random line opponent)."""
    from grid2op.Action import PowerlineSetAction
    from grid2op.Opponent import BaseActionBudget, RandomLineOpponent

    return {
        "opponent_action_class": PowerlineSetAction,
        "opponent_class": RandomLineOpponent,
        "opponent_budget_class": BaseActionBudget,
        "kwargs_opponent": {"lines_attacked": list(lines), "seed": seed},
        "opponent_attack_cooldown": int(cooldown),
    }


def collect_failure_forecast_rows(
    agent: Any,
    env: Any,
    mean_model: Any,
    aleatoric_model: Any,
    cfg: FailureForecastConfig,
    *,
    episodes: int,
    seed: int = 0,
    max_steps: Optional[int] = None,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    min_step: int = DEFAULT_MIN_STEP,
    phase: str = "Standard",
    model_enn: Any = None,
    get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
    progress_callback: Optional[Callable[[int, int, Counter], None]] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Collect labeled HGB rows for ONE simulation phase (legacy procedure).

    Episode ``seed + k`` is played by the agent; at every step with
    ``current_step > min_step`` and ``current_step % sample_every == 0`` the
    forced-disconnection analysis of ``analyze_disconnection_effect`` runs for
    ``cfg.lines_to_test``.  Observations whose vector length differs from the
    ENN input dimension are skipped (legacy sanity check).

    Returns ``(rows, stats)`` with counters for episodes, sample points, rows
    and positives/negatives; ``progress_callback`` receives
    ``(episode, steps_played, stats)``.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    expected_dim = getattr(model_enn, "input_dim", None) if model_enn is not None else None
    rows: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    for episode in range(episodes):
        obs = reset_environment(env, seed + episode)
        if hasattr(agent, "reset"):
            agent.reset(obs)
        observations = [obs]
        reward = getattr(env, "reward_range", (0.0, 0.0))[0]
        steps = 0
        stats["episodes"] += 1
        while max_steps is None or steps < max_steps:
            try:
                action = call_agent(agent, obs, reward=float(reward), done=False)
            except Exception:
                action = env.action_space({})
            if obs.current_step > min_step and obs.current_step % sample_every == 0:
                if expected_dim is not None and len(obs.to_vect()) != int(expected_dim):
                    stats["skipped_dimension_mismatch"] += 1
                else:
                    stats["sample_points"] += 1
                    new_rows = analyze_disconnection_effect(
                        env, agent, obs, observations, seed + episode,
                        mean_model, aleatoric_model, cfg, cfg.lines_to_test,
                        model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn)
                    for row in new_rows:
                        row["phase"] = phase
                        stats["rows"] += 1
                        stats["positive" if row["failed"] else "negative"] += 1
                    rows.extend(new_rows)
            obs, reward, done, info = env.step(action)
            steps += 1
            if done:
                stats["blackout_episodes"] += int(is_blackout(done, info))
                break
            observations.append(obs)
        if progress_callback:
            progress_callback(episode, steps, stats)
    return rows, stats


def collect_failure_forecast_phases(
    agent: Any,
    make_env: Callable[..., Any],
    mean_model: Any,
    aleatoric_model: Any,
    cfg: FailureForecastConfig,
    *,
    episodes: int,
    seed: int = DEFAULT_SEED_START,
    max_steps: Optional[int] = None,
    phases: Sequence[Tuple[str, Optional[int]]] = LEGACY_PHASES,
    model_enn: Any = None,
    get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
    progress_callback: Optional[Callable[[int, int, Counter], None]] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Run every legacy phase (Standard + random-opponent heavy attacks).

    ``make_env(**kwargs)`` must return a fresh Grid2Op environment; kwargs are
    empty for the standard phase and ``opponent_env_kwargs`` otherwise.  As in
    the legacy code every phase replays the same episode seeds.  The progress
    callback receives cumulative stats over all phases.
    """
    all_rows: List[Dict[str, Any]] = []
    total: Counter = Counter()
    for name, cooldown in phases:
        kwargs = {} if cooldown is None else opponent_env_kwargs(cfg.lines_to_test, cooldown)
        phase_env = make_env(**kwargs)
        try:
            done_before = Counter(total)

            def relay(episode: int, steps: int, stats: Counter) -> None:
                if progress_callback:
                    progress_callback(episode, steps, done_before + stats)

            rows, stats = collect_failure_forecast_rows(
                agent, phase_env, mean_model, aleatoric_model, cfg,
                episodes=episodes, seed=seed, max_steps=max_steps, phase=name,
                model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn,
                progress_callback=relay,
            )
        finally:
            phase_env.close()
        all_rows.extend(rows)
        total.update(stats)
    return all_rows, total


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


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_best_params(path: Path, seed: int) -> Optional[Dict[str, Any]]:
    """Load a completed tuning result; invalid files simply trigger retuning."""
    if not path.is_file():
        return None
    try:
        params = json.loads(path.read_text(encoding="utf-8"))["best_params"]
        params.update({"early_stopping": True, "random_state": int(seed)})
        return params
    except (OSError, KeyError, AttributeError, TypeError, json.JSONDecodeError):
        return None


def _load_model(path: str | Path) -> Optional[Any]:
    import joblib

    if not Path(path).is_file():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _fit_multioutput_with_progress(
    base_estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    native_threads: int,
    description: str,
    show_progress: bool = True,
) -> Any:
    """Fit independent outputs with completion progress and bounded threads."""
    from sklearn.base import clone
    from sklearn.multioutput import MultiOutputRegressor
    from threadpoolctl import threadpool_limits
    from tqdm import tqdm

    if int(native_threads) <= 0:
        raise ValueError("native_threads must be positive")
    output_count = int(y.shape[1])
    if show_progress:
        print(
            f"[{description}] rows={len(X)} outputs={output_count} "
            f"output_jobs=1 hgb_threads={native_threads}",
            flush=True,
        )

    fitted = []
    started = time.perf_counter()
    with tqdm(
        total=output_count, desc=f"[{description}] outputs",
        unit="output", disable=not show_progress,
    ) as bar:
        for index in range(output_count):
            estimator = clone(base_estimator)
            with threadpool_limits(limits=int(native_threads), user_api="openmp"):
                with threadpool_limits(limits=1, user_api="blas"):
                    estimator.fit(X, y[:, index])
            fitted.append(estimator)
            bar.update(1)
            bar.set_postfix(
                avg=f"{(time.perf_counter() - started) / (index + 1):.1f}s/output",
                refresh=False,
            )
    model = MultiOutputRegressor(base_estimator, n_jobs=1)
    model.estimators_ = fitted
    return model


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
    native_threads: int = 4,
    best_params_path: Optional[str | Path] = None,
    force_retune: bool = False,
) -> Any:
    """Train the mean forecaster for injections twelve steps ahead.

    Optuna tuning stops early once the best validation RMSE has not improved
    by more than ``min_improvement`` (relative) for ``patience`` trials; set
    ``patience`` to 0 to always run ``n_trials``. Output regressors are fitted
    sequentially to keep peak memory bounded.
    """
    from tqdm import tqdm
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    checkpoint_path = Path(best_params_path) if best_params_path else (
        Path(model_path).with_name(MEAN_BEST_PARAMS_FILENAME)
    )
    params = None if force_retune else _load_best_params(checkpoint_path, seed)
    reused_params = params is not None
    print(
        "[mean] tuning checkpoint: "
        + ("reusing saved best_params" if params else "not reusable; tuning required"),
        flush=True,
    )

    best_value = None
    completed_trials = 0
    if params is None and n_trials > 0:
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
            model = _fit_multioutput_with_progress(
                HistGradientBoostingRegressor(**params),
                X[tune_train_idx],
                y[tune_train_idx],
                native_threads=int(native_threads),
                description="mean tune",
                show_progress=False,
            )
            pred = np.column_stack([
                estimator.predict(X[tune_validation_idx])
                for estimator in model.estimators_
            ])
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
        best_value = float(study.best_value)
        completed_trials = len(study.trials)
    elif params is None:
        params = {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_depth": 6,
            "l2_regularization": 2.0,
            "early_stopping": True,
            "random_state": seed,
        }

    params.update({"early_stopping": True, "random_state": int(seed)})
    payload = {
        "completed_trials": int(completed_trials),
        "best_value": best_value,
        "best_params": params,
    }
    if not reused_params:
        _atomic_write_json(checkpoint_path, payload)
        print(f"[mean] saved tuning checkpoint -> {checkpoint_path}", flush=True)

    print(f"[mean] final fit on {len(X)} rows, {y.shape[1]} outputs...", flush=True)
    started = time.perf_counter()
    model = _fit_multioutput_with_progress(
        HistGradientBoostingRegressor(**params), X, y,
        native_threads=int(native_threads),
        description="mean final",
    )
    print(f"[mean] final fit done in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    path = Path(model_path)
    _atomic_joblib_dump(model, path)
    return model


def train_aleatoric_forecaster(
    X: np.ndarray,
    y: np.ndarray,
    mean_model: Any,
    *,
    model_path: str | Path,
    native_threads: int = 4,
) -> Any:
    """Train the legacy residual-variance forecaster."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=int(native_threads), user_api="openmp"):
        with threadpool_limits(limits=1, user_api="blas"):
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
    model = _fit_multioutput_with_progress(
        base, X, z, native_threads=int(native_threads),
        description="aleatoric final",
    )
    path = Path(model_path)
    _atomic_joblib_dump(model, path)
    return model


def load_or_train_forecasters(
    X: np.ndarray,
    y: np.ndarray,
    episode_ids: np.ndarray,
    *,
    mean_path: str | Path,
    aleatoric_path: str | Path,
    best_params_path: str | Path,
    n_trials: int,
    max_subsample: int = 5000,
    seed: int = 42,
    native_threads: int = 4,
    reuse: bool = True,
    force_retune: bool = False,
    on_train_start: Optional[Callable[[], None]] = None,
) -> Tuple[Any, Any, bool]:
    """Reuse saved models and train only the missing forecaster stages."""
    mean = _load_model(mean_path) if reuse and not force_retune else None
    aleatoric = _load_model(aleatoric_path) if reuse and not force_retune else None
    print(f"[preflight] best params: {'found' if Path(best_params_path).is_file() else 'missing'}")
    print(f"[preflight] mean model: {'found' if mean else 'missing'}")
    print(f"[preflight] aleatoric model: {'found' if aleatoric else 'missing'}")
    if mean is not None and aleatoric is not None:
        return mean, aleatoric, False

    if on_train_start is not None:
        on_train_start()
    mean_was_trained = mean is None
    if mean_was_trained:
        mean = train_mean_forecaster(
            X, y, model_path=mean_path, n_trials=n_trials,
            max_subsample=max_subsample, seed=seed, episode_ids=episode_ids,
            native_threads=native_threads,
            best_params_path=best_params_path, force_retune=force_retune,
        )
    if aleatoric is None or mean_was_trained:
        aleatoric = train_aleatoric_forecaster(
            X, y, mean, model_path=aleatoric_path,
            native_threads=native_threads,
        )
    return mean, aleatoric, True


def write_rows_csv(rows: Iterable[Dict[str, Any]], output_path: str | Path) -> int:
    """Write heterogeneous feature rows with a stable preferred column order."""
    rows = list(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return 0
    preferred = [
        "phase", "episode", "step", "event_step", "horizon_steps", "horizon_minutes",
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
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series,
           Optional[pd.Series], Dict[str, Any]]:
    """Legacy 70/15/15 train/validation/test split, episode-grouped when possible.

    Episode grouping (every episode in exactly one set) is kept from the
    current implementation; with fewer than ``MIN_GROUPED_EPISODES`` episodes
    the legacy row-level stratified split is used.  Returns
    ``(X_train, X_val, X_test, y_train, y_val, y_test, groups_train, info)``.
    """
    from sklearn.model_selection import train_test_split

    X = dataframe[CLASSIFIER_FEATURES].copy()
    y = dataframe["label"].astype(int)
    holdout = float(val_size) + float(test_size)
    test_share = float(test_size) / holdout

    def stratified(values: np.ndarray, labels: Optional[np.ndarray], fraction: float,
                   ) -> Tuple[np.ndarray, np.ndarray]:
        counts = pd.Series(labels).value_counts() if labels is not None else None
        use = labels is not None and len(counts) > 1 and int(counts.min()) >= 2
        return train_test_split(values, test_size=fraction, random_state=int(seed),
                                stratify=labels if use else None)

    if "episode" in dataframe.columns and dataframe["episode"].nunique() >= MIN_GROUPED_EPISODES:
        episodes = pd.to_numeric(dataframe["episode"], errors="raise").astype(int)
        unique_episodes = np.asarray(sorted(episodes.unique()), dtype=int)
        episode_has_failure = (
            pd.DataFrame({"episode": episodes, "label": y})
            .groupby("episode", sort=True)["label"]
            .max()
            .reindex(unique_episodes)
            .astype(int)
        )
        flags = episode_has_failure.to_numpy()
        train_ep, temp_ep = stratified(unique_episodes, flags, holdout)
        temp_flags = episode_has_failure.reindex(temp_ep).to_numpy()
        val_ep, test_ep = stratified(np.asarray(temp_ep), temp_flags, test_share)
        masks = {
            "train": episodes.isin(train_ep),
            "val": episodes.isin(val_ep),
            "test": episodes.isin(test_ep),
        }
        X_train, X_val, X_test = (X.loc[masks[k]] for k in ("train", "val", "test"))
        y_train, y_val, y_test = (y.loc[masks[k]] for k in ("train", "val", "test"))
        groups_train = episodes.loc[masks["train"]]
        split_info = {
            "strategy": "episode_grouped",
            "val_fraction": float(val_size),
            "test_fraction": float(test_size),
            "train_episodes": int(len(train_ep)),
            "val_episodes": int(len(val_ep)),
            "test_episodes": int(len(test_ep)),
        }
    else:
        index = np.arange(len(X))
        labels = y.to_numpy()
        train_idx, temp_idx = stratified(index, labels, holdout)
        val_idx, test_idx = stratified(np.asarray(temp_idx), labels[temp_idx], test_share)
        X_train, X_val, X_test = (X.iloc[i] for i in (train_idx, val_idx, test_idx))
        y_train, y_val, y_test = (y.iloc[i] for i in (train_idx, val_idx, test_idx))
        groups_train = None
        split_info = {
            "strategy": "row_stratified",
            "val_fraction": float(val_size),
            "test_fraction": float(test_size),
            "train_episodes": None,
            "val_episodes": None,
            "test_episodes": None,
        }
    split_info.update({
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    })

    for name, labels_part in (("training", y_train), ("validation", y_val), ("test", y_test)):
        if labels_part.nunique() < 2:
            raise ValueError(f"Classifier {name} split does not contain both classes.")
    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, split_info


def _classifier_class_weight(choice: str) -> Any:
    """Decode the legacy string choices of the class-weight search space."""
    weights = {"none": None, "balanced": "balanced",
               "w3": {0: 1, 1: 3}, "w5": {0: 1, 1: 5}, "w10": {0: 1, 1: 10}}
    if choice not in weights:
        raise ValueError(f"Unknown class_weight_choice: {choice}")
    return weights[choice]


def _classifier_risk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Legacy risk: 0.4 * false-alarm % + 0.6 * oversight % (oversights hurt more)."""
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fa = (fp / (tn + fp) * 100.0) if (tn + fp) else 0.0
    ovr = (fn / (tp + fn) * 100.0) if (tp + fn) else 0.0
    return 0.4 * fa + 0.6 * ovr


def _optimize_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: Optional[pd.Series],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_indices: List[int],
    *,
    n_trials: int,
    seed: int,
    n_jobs: int,
) -> Tuple[Dict[str, Any], float, float]:
    """Legacy Optuna search over HGB parameters, class weight and threshold.

    Each trial is fitted on the five folds of the TRAINING split and scored on
    the validation split with the legacy risk (the sampled threshold is tuned
    but, as in the legacy code, not used by the risk).  Folds are built from
    the training split only and grouped by episode when available.
    """
    import optuna
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    min_class = int(y_train.value_counts().min())
    n_splits = max(2, min(5, min_class))
    if groups_train is not None and groups_train.nunique() >= n_splits:
        folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                     .split(X_train, y_train, groups_train))
    else:
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                     .split(X_train, y_train))

    def objective(trial: Any) -> float:
        choice = trial.suggest_categorical(
            "class_weight_choice", ["none", "balanced", "w3", "w5", "w10"])
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 800),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 5.0, log=True),
            "class_weight": _classifier_class_weight(choice),
            "random_state": seed,
        }
        trial.suggest_float("threshold", 0.10, 0.90)
        scores = []
        for train_idx, _ in folds:
            model = HistGradientBoostingClassifier(categorical_features=cat_indices, **params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            scores.append(_classifier_risk(y_val.to_numpy(), model.predict(X_val)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(n_trials), n_jobs=int(n_jobs))
    best = dict(study.best_params)
    optuna_threshold = float(best.pop("threshold"))
    best["class_weight"] = _classifier_class_weight(best.pop("class_weight_choice"))
    best["random_state"] = seed
    return best, optuna_threshold, float(study.best_value)


def train_failure_classifier(
    dataframe: pd.DataFrame,
    *,
    model_path: str | Path,
    threshold: float = 0.5,
    seed: int = 42,
    env_name: str = "",
    agent_name: str = "",
    n_trials: int = DEFAULT_CLASSIFIER_TRIALS,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any]]:
    """Train the failure classifier with the legacy Optuna procedure.

    ``n_trials <= 0`` skips the search and fits fixed balanced HGB parameters.
    The final model is fitted on the training split only (as in the legacy
    code) and evaluated on the test split with ``threshold``.
    """
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    if "label" not in dataframe.columns:
        raise ValueError("Use prepare_classifier_dataframe before training.")
    y = dataframe["label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Failure classifier requires both classes.")
    if int(y.value_counts().min()) < 2:
        raise ValueError("Failure classifier needs at least two samples per class.")

    (X_train, X_val, X_test, y_train, y_val, y_test,
     groups_train, split_info) = _split_classifier_rows(dataframe, seed=seed)
    cat_indices = [CLASSIFIER_FEATURES.index("line_id_encoded")]
    optuna_threshold = None
    optuna_best_risk = None
    if n_trials > 0:
        params, optuna_threshold, optuna_best_risk = _optimize_classifier(
            X_train, y_train, groups_train, X_val, y_val, cat_indices,
            n_trials=int(n_trials), seed=int(seed), n_jobs=int(n_jobs))
    else:
        params = {"class_weight": "balanced", "learning_rate": 0.05,
                  "max_iter": 300, "random_state": int(seed)}
    model = HistGradientBoostingClassifier(categorical_features=cat_indices, **params)
    model.fit(X_train, y_train)
    probs = _failure_probability(model, X_test)
    preds = (probs >= float(threshold)).astype(int)
    metrics = _metrics(y_test.to_numpy(), preds, probs)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    serializable = dict(params)
    if isinstance(serializable.get("class_weight"), dict):
        serializable["class_weight"] = {
            str(k): v for k, v in serializable["class_weight"].items()}
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
        "label_definition": (
            "1 = Grid2Op episode ended (done) on the step that forcibly "
            "disconnects the line after the agent has played 12 steps (t+12), "
            "or the episode ended before t+12; 0 = the grid survived the "
            "disconnection"
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
        "model_params": serializable,
        "optuna": {
            "n_trials": int(n_trials),
            "best_risk": optuna_best_risk,
            "best_threshold_unused": optuna_threshold,
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
    agent call is made; one do-nothing powerflow with the forecast injections
    supplies the ``fcast_*`` features (legacy behaviour).
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
        env: Any,
        model_enn: Any = None,
        get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
    ) -> Dict[str, Any]:
        """Score one line; see ``predict_many``."""
        prediction = self.predict_many(
            obs, observations, [line], mean_model, aleatoric_model, cfg,
            env=env, model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn,
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
        env: Any,
        model_enn: Any = None,
        get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
    ) -> List[Dict[str, Any]]:
        """Score each requested, currently connected line from one t+12 forecast.

        ``observations`` is the episode history up to the current step and its
        last element must be ``obs``.  ``env`` supplies the action space for the
        forecast-injection powerflow (``simulate_forecast_state``).  Lines the
        classifier never saw during training raise an error.
        """
        if not observations or observations[-1] is not obs:
            raise ValueError("observations must end with the current observation")
        forecast = forecast_injections(
            observations, mean_model, aleatoric_model, cfg,
            model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn,
        )
        forecast.update(simulate_forecast_state(
            obs, forecast, env,
            model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn,
        ))
        predictions: List[Dict[str, Any]] = []
        for line in eligible_lines(cfg, obs, lines):
            row = build_feature_row(obs, line, forecast, cfg)
            predictions.append({
                "line": cfg.line_name(line),
                **self.predict_from_features(row),
            })
        return predictions
