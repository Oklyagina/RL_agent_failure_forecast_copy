from __future__ import annotations

import os
import sys
from pathlib import Path
import joblib
import optuna
import numpy as np
import grid2op
from typing import Any, Dict, List, Sequence, Tuple
from tqdm import tqdm

from grid2op.Reward import L2RPNReward
from lightsim2grid import LightSimBackend

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
from config import CFG, TRAIN_MODE
from utils import append_to_npy
from curriculumagent.baseline.baseline import CurriculumAgent


# =============================================================================
# Feature Extraction (History + Current State)
# =============================================================================

def convert_to_cos_sin(value: float, max_value: float) -> Tuple[float, float]:
    """
    Encodes cyclical continuous features (like time) into sine and cosine components.
    """
    value_cos = np.cos(2 * np.pi * value / max_value)
    value_sin = np.sin(2 * np.pi * value / max_value)
    return float(value_cos), float(value_sin)


def get_features_with_history(observations_array: List[Any], obs: Any) -> np.ndarray:
    """
    Extracts current and historical state features from the environment observations.
    """
    sz_load_p = len(obs.load_p)
    sz_load_q = len(obs.load_q)
    sz_gen_p = len(obs.gen_p)

    def extract_features(past_obs, attr, size):
        if past_obs is not None:
            return getattr(past_obs, attr)
        return np.full(size, np.nan)

    current_idx = len(observations_array) - 1

    # Historical lookbacks (assuming 5-min intervals: 12=1h, 288=1d, 2016=1w)
    past_obs_hour = observations_array[current_idx - 12] if current_idx >= 12 else None
    past_obs_day = observations_array[current_idx - 288] if current_idx >= 288 else None
    past_obs_week = observations_array[current_idx - 2016] if current_idx >= 2016 else None

    # Current and historical grid states
    feature_vec = np.concatenate([
        obs.load_p, obs.load_q, obs.gen_p,

        extract_features(past_obs_hour, 'load_p', sz_load_p),
        extract_features(past_obs_hour, 'load_q', sz_load_q),
        extract_features(past_obs_hour, 'gen_p', sz_gen_p),

        extract_features(past_obs_day, 'load_p', sz_load_p),
        extract_features(past_obs_day, 'load_q', sz_load_q),
        extract_features(past_obs_day, 'gen_p', sz_gen_p),

        extract_features(past_obs_week, 'load_p', sz_load_p),
        extract_features(past_obs_week, 'load_q', sz_load_q),
        extract_features(past_obs_week, 'gen_p', sz_gen_p),
    ])

    # Temporal cyclical features
    hour_cos, hour_sin = convert_to_cos_sin(obs.hour_of_day, 23)
    minute_cos, minute_sin = convert_to_cos_sin(obs.minute_of_hour, 59)
    dow_cos, dow_sin = convert_to_cos_sin(obs.day_of_week, 6)

    temporal_features = [obs.day, hour_cos, hour_sin, minute_cos, minute_sin, dow_cos, dow_sin]

    return np.concatenate([feature_vec, temporal_features]).flatten()


# =============================================================================
# Data Collection (Target: t -> t+1)
# =============================================================================

def collect_data(env: grid2op.Environment, agent: CurriculumAgent, episode_seeds: Sequence[int],
                 x_path: str, y_path: str) -> None:
    """
    Executes episodes using the agent and collects state transitions for supervised learning.
    """
    print(f"[INFO] Collecting data to {x_path} and {y_path}...")

    if os.path.exists(x_path): os.remove(x_path)
    if os.path.exists(y_path): os.remove(y_path)

    for idx, ep_seed in enumerate(tqdm(episode_seeds, desc="Collecting forecast data", unit="episode")):
        obs = env.reset(seed=int(ep_seed))
        done = False

        observations: List[Any] = []
        X_ep: List[np.ndarray] = []
        y_ep: List[np.ndarray] = []

        while not done:
            observations.append(obs)
            x = get_features_with_history(observations, obs)

            action = agent.act(obs, env.reward_range[0], done)
            obs_next, _, done, _ = env.step(action)

            if done: break

            y = np.concatenate([obs_next.load_p, obs_next.load_q, obs_next.gen_p], axis=0)

            X_ep.append(x)
            y_ep.append(np.asarray(y))
            obs = obs_next

        if len(X_ep) > 0:
            append_to_npy(x_path, np.asarray(X_ep))
            append_to_npy(y_path, np.asarray(y_ep))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(episode_seeds):
            print(f"[DATA] Saved episode {idx + 1}/{len(episode_seeds)} to {x_path}")


# =============================================================================
# Model Training
# =============================================================================

def train_mean_model(x_path: str, y_path: str, n_trials: int = 20, max_subsample: int = 5000,
                     random_state: int = 42) -> None:
    """
    Optimizes and trains the primary mean forecasting model.
    """
    print(f"[TRAIN] Training Mean Model using {x_path}...")
    X_train = np.load(x_path, allow_pickle=True)
    y_train = np.load(y_path, allow_pickle=True)

    rng = np.random.default_rng(random_state)

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
            "early_stopping": True,
        }

        n = len(X_train)
        sub_n = min(max_subsample, n)
        sub_idx = rng.choice(n, size=sub_n, replace=False)

        model = MultiOutputRegressor(HistGradientBoostingRegressor(**params))
        model.fit(X_train[sub_idx], y_train[sub_idx])

        preds = model.predict(X_train[sub_idx])
        rmse = float(np.sqrt(mean_squared_error(y_train[sub_idx], preds)))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = dict(study.best_params)
    best_params["early_stopping"] = True

    best_model = MultiOutputRegressor(HistGradientBoostingRegressor(**best_params))
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, CFG.MODEL_MEAN_PATH)
    print(f"[SAVE] Mean Model saved to {CFG.MODEL_MEAN_PATH}")
    print(f"[TRAIN] Best params: {best_params}")


def train_aleatoric_model(x_path: str, y_path: str) -> None:
    """
    Trains the variance (aleatoric uncertainty) model based on the mean model's residuals.
    """
    print(f"[TRAIN] Training Aleatoric Model using {x_path}...")
    X_train = np.load(x_path, allow_pickle=True)
    y_train = np.load(y_path, allow_pickle=True)

    mean_model = joblib.load(CFG.MODEL_MEAN_PATH)
    y_pred = mean_model.predict(X_train)

    squared_residuals = (y_train - y_pred) ** 2
    upper = np.percentile(squared_residuals, 99.9, axis=0)
    squared_residuals = np.clip(squared_residuals, 0.0, upper)
    z = np.log1p(squared_residuals)

    base = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=300,
        max_depth=6, l2_regularization=2.0, early_stopping=True,
    )
    var_model = MultiOutputRegressor(base)
    var_model.fit(X_train, z)

    joblib.dump(var_model, CFG.MODEL_ALEATORIC_PATH)
    print(f"[SAVE] Aleatoric Model saved to {CFG.MODEL_ALEATORIC_PATH}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    if not TRAIN_MODE:
        print("[INFO] TRAIN_MODE is False. Skipping training.")
        raise SystemExit(0)

    env: grid2op.Environment = grid2op.make(
        CFG.ENV_NAME,
        backend=LightSimBackend(),
        reward_class=L2RPNReward,
    )

    agent = CurriculumAgent(env.action_space, env.observation_space, name="CA")
    try:
        agent.load(CFG.AGENT_PATH)
        print("[INFO] CurriculumAgent loaded.")
    except Exception:
        print("[WARN] Agent could not be loaded. Defaulting to untrained behavior.")

    train_seeds: List[int] = list(range(0, 900))
    test_seeds: List[int] = list(range(900, 970))

    # 1) Collect Data
    collect_data(env, agent, train_seeds, CFG.X_TRAIN_PATH, CFG.Y_TRAIN_PATH)
    collect_data(env, agent, test_seeds, CFG.X_TEST_PATH, CFG.Y_TEST_PATH)

    # 2) Train Models
    train_mean_model(CFG.X_TRAIN_PATH, CFG.Y_TRAIN_PATH, n_trials=500)
    train_aleatoric_model(CFG.X_TRAIN_PATH, CFG.Y_TRAIN_PATH)

    print("[INFO] Forecast training pipeline completed successfully.")
