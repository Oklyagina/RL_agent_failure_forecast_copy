"""One-hour-ahead failure forecasting for Grid2Op RL agents.

This module ports the legacy failure-forecast feature path from
``archive/legacy/old_version/src`` into the active package:

* historical forecast features use current, 1h, 1d, and 1w observation lags;
* mean and aleatoric forecasters predict only the final t+12 step;
* predicted injections are written to ``_forecasted_inj`` before a Grid2Op
  power-flow simulation;
* labels come from branch simulations of a line disconnection followed by the
  configured agent response.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .agent_runtime import call_agent


HORIZON_STEPS = 12
STEP_MINUTES = 5

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

REQUIRED_DATA_COLUMNS = {
    "line_disconnected",
    "failed",
    *(feature for feature in CLASSIFIER_FEATURES
      if feature not in {"line_id_encoded", "load_gen_ratio"}),
}


@dataclass
class FailureForecastConfig:
    """Grid dimensions and paths needed by the failure forecaster."""

    env_name: str
    agent_name: str
    no_loads: int
    no_gens: int
    lines_to_test: Sequence[Any]
    gen_min: np.ndarray
    gen_max: np.ndarray
    artifact_dir: Path

    @classmethod
    def from_env(cls, env: Any, *, env_name: str, agent_name: str,
                 artifact_dir: str | Path, lines_to_test: Sequence[Any]) -> "FailureForecastConfig":
        """Build configuration from a live Grid2Op environment instance."""
        no_gens = int(getattr(env, "n_gen"))
        gen_min = np.asarray(getattr(env, "gen_pmin", np.zeros(no_gens)), dtype=float)
        gen_max = np.asarray(getattr(env, "gen_pmax", np.full(no_gens, np.inf)), dtype=float)
        return cls(
            env_name=str(env_name),
            agent_name=str(agent_name),
            no_loads=int(getattr(env, "n_load")),
            no_gens=no_gens,
            lines_to_test=list(lines_to_test),
            gen_min=gen_min,
            gen_max=gen_max,
            artifact_dir=Path(artifact_dir),
        )


def convert_to_cos_sin(value: float, period: float) -> Tuple[float, float]:
    """Encode a cyclic scalar as cosine/sine features."""
    value_cos = np.cos(2 * np.pi * value / period)
    value_sin = np.sin(2 * np.pi * value / period)
    return float(value_cos), float(value_sin)


def get_features_with_history(observations: Sequence[Any], obs: Any) -> np.ndarray:
    """Legacy forecast input: current values plus 1h/1d/1w lags and time."""
    if not observations:
        observations = [obs]

    sz_load_p = len(obs.load_p)
    sz_load_q = len(obs.load_q)
    sz_gen_p = len(obs.gen_p)
    current_idx = len(observations) - 1

    def past(lag_steps: int) -> Optional[Any]:
        """Return the observation at a fixed lag, or None before history exists."""
        idx = current_idx - lag_steps
        return observations[idx] if idx >= 0 else None

    def values(past_obs: Optional[Any], attr: str, size: int) -> np.ndarray:
        """Extract an array from a lagged observation, padding missing history."""
        if past_obs is None:
            return np.full(size, np.nan, dtype=float)
        return np.asarray(getattr(past_obs, attr), dtype=float)

    past_hour = past(12)
    past_day = past(288)
    past_week = past(2016)
    feature_vec = np.concatenate([
        obs.load_p, obs.load_q, obs.gen_p,
        values(past_hour, "load_p", sz_load_p),
        values(past_hour, "load_q", sz_load_q),
        values(past_hour, "gen_p", sz_gen_p),
        values(past_day, "load_p", sz_load_p),
        values(past_day, "load_q", sz_load_q),
        values(past_day, "gen_p", sz_gen_p),
        values(past_week, "load_p", sz_load_p),
        values(past_week, "load_q", sz_load_q),
        values(past_week, "gen_p", sz_gen_p),
    ]).astype(float)

    hour_cos, hour_sin = convert_to_cos_sin(getattr(obs, "hour_of_day", 0), 23)
    minute_cos, minute_sin = convert_to_cos_sin(getattr(obs, "minute_of_hour", 0), 59)
    dow_cos, dow_sin = convert_to_cos_sin(getattr(obs, "day_of_week", 0), 6)
    temporal = np.asarray([
        getattr(obs, "day", 1),
        hour_cos,
        hour_sin,
        minute_cos,
        minute_sin,
        dow_cos,
        dow_sin,
    ], dtype=float)
    return np.concatenate([feature_vec, temporal]).flatten()


def compute_grid_stats(obs: Any) -> Dict[str, float]:
    """Legacy compact grid statistics used by the failure classifier."""
    rho = getattr(obs, "rho", None)
    if rho is None:
        var_rho = avg_rho = max_rho = float("nan")
        nb_rho_095 = 0
    else:
        rho = np.asarray(rho, dtype=float)
        var_rho = float(np.var(rho))
        avg_rho = float(np.mean(rho))
        max_rho = float(np.max(rho))
        nb_rho_095 = int(np.sum(rho >= 0.95))
    return {
        "sum_load_p": float(np.sum(obs.load_p)),
        "sum_load_q": float(np.sum(obs.load_q)),
        "sum_gen_p": float(np.sum(obs.gen_p)),
        "sum_gen_q": float(np.sum(obs.gen_q)) if hasattr(obs, "gen_q") else float("nan"),
        "var_line_rho": var_rho,
        "avg_line_rho": avg_rho,
        "max_line_rho": max_rho,
        "nb_rho_ge_0.95": nb_rho_095,
    }


def _failure_from_done(done: bool, obs: Any, info: Optional[dict]) -> bool:
    """Map Grid2Op branch termination to the binary blackout/failure label."""
    if not done:
        return False
    if info and info.get("exception"):
        return True
    current_step = getattr(obs, "current_step", None)
    max_step = getattr(obs, "max_step", None)
    if current_step is not None and max_step is not None:
        return int(current_step) < int(max_step)
    return True


def _resolve_line_id(env: Any, line: Any) -> int:
    """Resolve a line id or name into the integer index expected by Grid2Op."""
    if isinstance(line, (int, np.integer)):
        return int(line)
    normalized = str(line).replace("line_", "")
    names = getattr(env, "name_line", None)
    if names is not None:
        for idx, name in enumerate(names):
            if str(name).replace("line_", "") == normalized:
                return int(idx)
    return int(line)


def normalize_line_name(line: Any) -> str:
    """Normalize Grid2Op line names for stable classifier metadata keys."""
    return str(line).replace("line_", "")


def _line_name(env: Any, line: Any) -> str:
    """Return the environment line name when a numeric line id was supplied."""
    if isinstance(line, (int, np.integer)) and hasattr(env, "name_line"):
        line_id = int(line)
        if 0 <= line_id < len(env.name_line):
            return str(env.name_line[line_id])
    return str(line)


def make_disconnect_action(env: Any, line: Any) -> Any:
    """Create a Grid2Op action that disconnects exactly one candidate line."""
    line_id = _resolve_line_id(env, line)
    try:
        return env.action_space({"set_line_status": [(line_id, -1)]})
    except Exception:
        status = np.zeros(int(getattr(env, "n_line")), dtype=int)
        status[line_id] = -1
        return env.action_space({"set_line_status": status})


def _do_nothing_action(env: Any, obs: Any) -> Any:
    """Create a no-op action across Grid2Op versions and observation contexts."""
    try:
        return env.action_space({})
    except Exception:
        return obs._obs_env._helper_action_env({})


def _simulate(obs: Any, action: Any) -> Tuple[Any, float, bool, dict]:
    """Run observation-branch simulation for a single action."""
    return obs.simulate(action)


def _simulator_result(simulator: Any) -> Tuple[Any, float, bool, dict]:
    """Convert a Grid2Op Simulator state to the observation simulation contract."""
    converged = bool(simulator.converged)
    error = getattr(simulator, "error", None)
    exceptions = [] if converged or error is None else [error]
    return simulator.current_obs, 0.0, not converged, {"exception": exceptions}


def _get_uncertainty(model_enn: Any, obs: Any,
                     get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]]) -> float:
    """Compute optional ENN epistemic uncertainty, returning NaN if unavailable."""
    if model_enn is None:
        return float("nan")
    input_dim = int(getattr(model_enn, "input_dim", len(obs.to_vect())))
    vect = np.asarray(obs.to_vect(), dtype=np.float32)[:input_dim].reshape(1, -1)
    if get_uncertainty_fn is not None:
        return float(get_uncertainty_fn(model_enn, vect))
    try:
        import torch

        with torch.no_grad():
            out = model_enn(torch.as_tensor(vect, dtype=torch.float32))
        if isinstance(out, dict) and "uncertainty" in out:
            return float(np.asarray(out["uncertainty"].detach().cpu()).reshape(-1)[0])
    except Exception:
        pass
    return float("nan")


def run_t12_forecast(
    env: Any,
    obs: Any,
    observations: Sequence[Any],
    mean_model: Any,
    aleatoric_model: Any,
    cfg: FailureForecastConfig,
    *,
    model_enn: Any = None,
    get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """Run the legacy t+12 forecast and return forecast state/features."""
    x_t12 = get_features_with_history(observations, obs)
    y_pred = np.asarray(mean_model.predict([x_t12])[0], dtype=float)
    z_pred = np.asarray(aleatoric_model.predict([x_t12])[0], dtype=float)

    load_p_t12 = y_pred[:cfg.no_loads]
    load_q_t12 = y_pred[cfg.no_loads: cfg.no_loads * 2]
    gen_p_t12 = np.clip(y_pred[cfg.no_loads * 2:], cfg.gen_min, cfg.gen_max)
    sigma = np.sqrt(np.expm1(np.clip(z_pred, 0.0, None)) + 1e-12)

    now_ts = obs.get_time_stamp()
    next_ts = now_ts + _dt.timedelta(minutes=HORIZON_STEPS * STEP_MINUTES)
    simulator_factory = getattr(obs, "get_simulator", None)
    simulator = None
    if callable(simulator_factory):
        simulator = simulator_factory().predict(
            _do_nothing_action(env, obs),
            new_gen_p=gen_p_t12,
            new_gen_v=obs.gen_v,
            new_load_p=load_p_t12,
            new_load_q=load_q_t12,
        )
        forecast_obs, _, done, info = _simulator_result(simulator)
    else:
        obs_copy = obs.copy()
        obs_copy._forecasted_inj = [
            (now_ts, {"injection": {
                "load_p": obs.load_p,
                "load_q": obs.load_q,
                "prod_p": obs.gen_p,
                "prod_v": obs.gen_v,
            }}),
            (next_ts, {"injection": {
                "load_p": load_p_t12,
                "load_q": load_q_t12,
                "prod_p": gen_p_t12,
                "prod_v": obs.gen_v,
            }}),
        ]
        forecast_obs, _, done, info = _simulate(
            obs_copy, _do_nothing_action(env, obs)
        )
    return {
        "simulator": simulator,
        "forecast_obs": forecast_obs,
        "forecast_done": bool(done),
        "forecast_failed": _failure_from_done(bool(done), forecast_obs, info),
        "forecast_info": info or {},
        "forecast_grid_stats": compute_grid_stats(forecast_obs),
        "aleatoric_load_p_mean": float(np.mean(sigma[:cfg.no_loads])),
        "aleatoric_load_q_mean": float(np.mean(sigma[cfg.no_loads: cfg.no_loads * 2])),
        "aleatoric_gen_p_mean": float(np.mean(sigma[cfg.no_loads * 2:])),
        "epistemic_before": _get_uncertainty(model_enn, obs, get_uncertainty_fn),
        "epistemic_after": _get_uncertainty(model_enn, forecast_obs, get_uncertainty_fn),
        "forecast_input": x_t12,
        "predicted_load_p": load_p_t12,
        "predicted_load_q": load_q_t12,
        "predicted_gen_p": gen_p_t12,
    }


def build_feature_row(env: Any, obs: Any, line: Any, forecast: Dict[str, Any],
                      cfg: FailureForecastConfig) -> Dict[str, Any]:
    """Combine current, forecast, uncertainty, and line features into one row."""
    current = compute_grid_stats(obs)
    fcast = forecast.get("forecast_grid_stats", {})
    sum_load = float(current.get("sum_load_p", np.nan))
    sum_gen = float(current.get("sum_gen_p", np.nan))
    row = {
        "step": int(getattr(obs, "current_step", -1)),
        "horizon_steps": HORIZON_STEPS,
        "horizon_minutes": HORIZON_STEPS * STEP_MINUTES,
        "line_disconnected": _line_name(env, line),
        "line_id": _resolve_line_id(env, line),
        "aleatoric_load_p_mean": float(forecast.get("aleatoric_load_p_mean", np.nan)),
        "aleatoric_load_q_mean": float(forecast.get("aleatoric_load_q_mean", np.nan)),
        "aleatoric_gen_p_mean": float(forecast.get("aleatoric_gen_p_mean", np.nan)),
        "epistemic_before": float(forecast.get("epistemic_before", np.nan)),
        "epistemic_after": float(forecast.get("epistemic_after", np.nan)),
        "sum_load_p": sum_load,
        "sum_load_q": float(current.get("sum_load_q", np.nan)),
        "sum_gen_p": sum_gen,
        "var_line_rho": float(current.get("var_line_rho", np.nan)),
        "avg_line_rho": float(current.get("avg_line_rho", np.nan)),
        "max_line_rho": float(current.get("max_line_rho", np.nan)),
        "nb_rho_ge_0.95": float(current.get("nb_rho_ge_0.95", np.nan)),
        "load_gen_ratio": sum_load / (sum_gen + 1e-6),
        "fcast_sum_load_p": float(fcast.get("sum_load_p", np.nan)),
        "fcast_sum_load_q": float(fcast.get("sum_load_q", np.nan)),
        "fcast_sum_gen_p": float(fcast.get("sum_gen_p", np.nan)),
        "fcast_var_line_rho": float(fcast.get("var_line_rho", np.nan)),
        "fcast_avg_line_rho": float(fcast.get("avg_line_rho", np.nan)),
        "fcast_max_line_rho": float(fcast.get("max_line_rho", np.nan)),
        "fcast_nb_rho_ge_0.95": float(fcast.get("nb_rho_ge_0.95", np.nan)),
    }
    return row


def evaluate_line_failure(
    env: Any,
    agent: Any,
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
    """Create one labeled legacy-compatible row for one line contingency."""
    forecast = run_t12_forecast(
        env, obs, observations, mean_model, aleatoric_model, cfg,
        model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn,
    )
    row = build_feature_row(env, obs, line, forecast, cfg)
    row["forecast_failed"] = bool(forecast["forecast_failed"])
    if forecast["forecast_failed"]:
        row.update({"failed": 1, "label_reason": "forecast_failed_before_contingency"})
        return row

    disconnect_action = make_disconnect_action(env, line)
    simulator = forecast.get("simulator")
    if simulator is not None:
        attacked_simulator = simulator.predict(disconnect_action)
        attacked_obs, _, disconnect_done, disconnect_info = _simulator_result(
            attacked_simulator
        )
    else:
        attacked_simulator = None
        attacked_obs, _, disconnect_done, disconnect_info = _simulate(
            forecast["forecast_obs"], disconnect_action
        )
    if _failure_from_done(bool(disconnect_done), attacked_obs, disconnect_info):
        row.update({"failed": 1, "label_reason": "line_disconnection_blackout"})
        return row

    # Evaluate the live recommendation against the forecasted contingency. A
    # Simulator observation is detached from Grid2Op's nested obs.simulate API,
    # which some policies (including CurriculumAgent) use internally.
    agent_action = call_agent(agent, obs, reward=0.0, done=False)
    if attacked_simulator is not None:
        action_obs, _, action_done, action_info = _simulator_result(
            attacked_simulator.predict(agent_action)
        )
    else:
        action_obs, _, action_done, action_info = _simulate(
            attacked_obs, agent_action
        )
    failed = _failure_from_done(bool(action_done), action_obs, action_info)
    row.update({
        "failed": 1 if failed else 0,
        "label_reason": "agent_action_blackout" if failed else "survived_agent_action",
    })
    return row


def collect_failure_forecast_rows(
    env: Any,
    agent: Any,
    mean_model: Any,
    aleatoric_model: Any,
    cfg: FailureForecastConfig,
    *,
    episodes: int,
    seed: int = 0,
    sampling_stride: int = 20,
    max_steps: Optional[int] = None,
    model_enn: Any = None,
    get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> List[Dict[str, Any]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if sampling_stride <= 0:
        raise ValueError("sampling_stride must be positive")
    rows: List[Dict[str, Any]] = []
    for episode in range(episodes):
        try:
            obs = env.reset(seed=seed + episode)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed + episode)
            obs = env.reset()
        observations: List[Any] = []
        reward = getattr(env, "reward_range", (0.0, 0.0))[0]
        done = False
        step = 0
        if hasattr(agent, "reset"):
            try:
                agent.reset(obs)
            except Exception:
                pass
        while not done and (max_steps is None or step < max_steps):
            observations.append(obs)
            current_step = int(getattr(obs, "current_step", step))
            if current_step >= HORIZON_STEPS and current_step % sampling_stride == 0:
                for line in cfg.lines_to_test:
                    row = evaluate_line_failure(
                        env, agent, obs, observations, line, mean_model,
                        aleatoric_model, cfg, model_enn=model_enn,
                        get_uncertainty_fn=get_uncertainty_fn,
                    )
                    row["episode"] = int(episode)
                    rows.append(row)
            action = call_agent(agent, obs, reward=float(reward), done=done)
            obs, reward, done, _ = env.step(action)
            step += 1
        if progress_callback:
            progress_callback(episode, step, len(rows))
    return rows


def collect_forecaster_training_data(
    env: Any,
    agent: Any,
    *,
    episodes: int,
    seed: int = 0,
    max_steps: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect legacy forecast-model pairs ``X -> next(load_p, load_q, gen_p)``."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rows_x: List[np.ndarray] = []
    rows_y: List[np.ndarray] = []
    for episode in range(episodes):
        try:
            obs = env.reset(seed=seed + episode)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed + episode)
            obs = env.reset()
        observations: List[Any] = []
        reward = getattr(env, "reward_range", (0.0, 0.0))[0]
        done = False
        step = 0
        if hasattr(agent, "reset"):
            try:
                agent.reset(obs)
            except Exception:
                pass
        while not done and (max_steps is None or step < max_steps):
            observations.append(obs)
            x = get_features_with_history(observations, obs)
            action = call_agent(agent, obs, reward=float(reward), done=done)
            obs_next, reward, done, _ = env.step(action)
            if done:
                break
            y = np.concatenate([obs_next.load_p, obs_next.load_q, obs_next.gen_p], axis=0)
            rows_x.append(np.asarray(x, dtype=np.float32))
            rows_y.append(np.asarray(y, dtype=np.float32))
            obs = obs_next
            step += 1
        if progress_callback:
            progress_callback(episode, step, len(rows_x))
    if not rows_x:
        raise RuntimeError("Forecast data collection produced no rows.")
    return np.stack(rows_x).astype(np.float32), np.stack(rows_y).astype(np.float32)


def save_forecaster_training_data(out_dir: str | Path, X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
    """Persist forecast training arrays beside the downstream classifier data."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x_path = out / "forecast_X.npy"
    y_path = out / "forecast_y.npy"
    np.save(x_path, np.asarray(X, dtype=np.float32))
    np.save(y_path, np.asarray(y, dtype=np.float32))
    return {"X": str(x_path), "y": str(y_path)}


def train_mean_forecaster(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_path: str | Path,
    n_trials: int = 20,
    max_subsample: int = 5000,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Any:
    """Train the legacy mean forecaster for next-step injections."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.multioutput import MultiOutputRegressor

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    rng = np.random.default_rng(seed)

    if n_trials > 0:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
                "early_stopping": True,
                "random_state": seed,
            }
            sub_n = min(int(max_subsample), len(X))
            sub_idx = rng.choice(len(X), size=sub_n, replace=False)
            model = MultiOutputRegressor(HistGradientBoostingRegressor(**params))
            model.fit(X[sub_idx], y[sub_idx])
            pred = model.predict(X[sub_idx])
            return float(np.sqrt(mean_squared_error(y[sub_idx], pred)))

        study = optuna.create_study(direction="minimize")
        total_trials = int(n_trials)

        def report_progress(study: Any, trial: Any) -> None:
            """Report compact Optuna progress without enabling Optuna logs."""
            if progress_callback is not None:
                progress_callback(trial.number + 1, total_trials, float(study.best_value))

        study.optimize(
            objective,
            n_trials=total_trials,
            show_progress_bar=False,
            callbacks=[report_progress],
        )
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

    model = MultiOutputRegressor(HistGradientBoostingRegressor(**params))
    model.fit(X, y)
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
        "episode", "step", "horizon_steps", "horizon_minutes",
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
    from sklearn.model_selection import train_test_split

    if "label" not in dataframe.columns:
        raise ValueError("Use prepare_classifier_dataframe before training.")
    X = dataframe[CLASSIFIER_FEATURES].copy()
    y = dataframe["label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Failure classifier requires both classes.")
    if int(y.value_counts().min()) < 2:
        raise ValueError("Failure classifier needs at least two samples per class.")

    stratify = y if int(y.value_counts().min()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
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
    """Runtime predictor returning binary failure class and probability."""

    def __init__(self, model: Any, metadata: Dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.features = list(metadata.get("features", CLASSIFIER_FEATURES))
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
        """Convert one feature row into the exact dataframe expected by the model."""
        import pandas as pd

        row_dict = dict(row)
        line_name = normalize_line_name(row_dict.get("line_disconnected", ""))
        if "line_id_encoded" not in row_dict:
            if line_name not in self.line_map:
                raise ValueError(f"Unknown line for failure classifier: {line_name!r}")
            row_dict["line_id_encoded"] = self.line_map[line_name]
        if "load_gen_ratio" not in row_dict:
            row_dict["load_gen_ratio"] = (
                float(row_dict["sum_load_p"]) / (float(row_dict["sum_gen_p"]) + 1e-6)
            )
        return pd.DataFrame([{feature: row_dict.get(feature, np.nan)
                              for feature in self.features}])

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
        env: Any,
        agent: Any,
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
        """Run t+12 forecasting, build features, and return failure prediction."""
        forecast = run_t12_forecast(
            env, obs, observations, mean_model, aleatoric_model, cfg,
            model_enn=model_enn, get_uncertainty_fn=get_uncertainty_fn,
        )
        row = build_feature_row(env, obs, line, forecast, cfg)
        return self.predict_from_features(row)
