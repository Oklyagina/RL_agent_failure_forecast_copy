import os
import numpy as np
import datetime
from typing import Tuple, List, Dict, Any, Optional

# Import the active configuration
from config import CFG

def convert_to_cos_sin(value: int, period: int) -> Tuple[float, float]:
    val_cos = np.cos(2 * np.pi * value / period)
    val_sin = np.sin(2 * np.pi * value / period)
    return val_cos, val_sin

def get_features(observations_array: List[Any], obs: Any, step: int) -> Tuple[np.ndarray, int]:
    t = obs.current_step

    def get_past(array: List[Any], idx: int, lag_steps: int) -> Any:
        past_index = idx - lag_steps
        if 0 <= past_index < len(array):
            return array[past_index]
        return None

    def extract_features(past_obs: Any, attr: str, size: int) -> np.ndarray:
        if past_obs is not None:
            return getattr(past_obs, attr)
        return np.zeros(size, dtype=float)

    # lags em passos de 5 min: 12=1h, 288=1d, 2016=1w
    past_obs_hour = get_past(observations_array, t, 12)
    past_obs_day  = get_past(observations_array, t, 288)
    past_obs_week = get_past(observations_array, t, 2016)

    feature_vec = np.concatenate([
        extract_features(past_obs_week, 'load_p', CFG.NO_LOADS),
        extract_features(past_obs_day,  'load_p', CFG.NO_LOADS),
        extract_features(past_obs_hour, 'load_p', CFG.NO_LOADS),

        extract_features(past_obs_week, 'load_q', CFG.NO_LOADS),
        extract_features(past_obs_day,  'load_q', CFG.NO_LOADS),
        extract_features(past_obs_hour, 'load_q', CFG.NO_LOADS),

        extract_features(past_obs_week, 'gen_p',  CFG.NO_GENS),
        extract_features(past_obs_day,  'gen_p',  CFG.NO_GENS),
        extract_features(past_obs_hour, 'gen_p',  CFG.NO_GENS),
    ])

    atual_ts = obs.get_time_stamp()
    new_ts = atual_ts + datetime.timedelta(minutes=step * 5)

    temporal_features = [
        new_ts.day,
        *convert_to_cos_sin(new_ts.hour, 24),
        *convert_to_cos_sin(new_ts.minute, 60),
        *convert_to_cos_sin(new_ts.weekday(), 7),
        step
    ]

    return np.concatenate([feature_vec, temporal_features]).reshape(1, -1).astype(float), 0


def compute_grid_stats(obs: Any) -> Dict[str, float]:
    """Calculates basic grid statistics from an observation.

    Works with both real Grid2Op observations and MockObs instances.
    MockObs does not have rho/gen_q, so those fields fall back to safe defaults.
    """
    # rho is only available on real Grid2Op observations
    if hasattr(obs, "rho"):
        rho = obs.rho
        var_rho    = float(np.var(rho))
        avg_rho    = float(np.mean(rho))
        max_rho    = float(np.max(rho))
        nb_rho_095 = int(np.sum(rho >= 0.95))
    else:
        # MockObs: rho not available — fill with NaN so downstream code
        # can distinguish "unknown" from a real zero.
        var_rho    = float("nan")
        avg_rho    = float("nan")
        max_rho    = float("nan")
        nb_rho_095 = 0

    # gen_q is also absent from MockObs
    sum_gen_q = float(np.sum(obs.gen_q)) if hasattr(obs, "gen_q") else float("nan")

    stats = {
        "sum_load_p":    float(np.sum(obs.load_p)),
        "sum_load_q":    float(np.sum(obs.load_q)),
        "sum_gen_p":     float(np.sum(obs.gen_p)),
        "sum_gen_q":     sum_gen_q,
        "var_line_rho":  var_rho,
        "avg_line_rho":  avg_rho,
        "max_line_rho":  max_rho,
        "nb_rho_ge_0.95": nb_rho_095,
    }
    return stats

def append_to_npy(file_path: str, array: np.ndarray) -> None:
    """Appends data to an .npy file, creating it if it doesn't exist."""
    if os.path.exists(file_path):
        old_data = np.load(file_path, allow_pickle=True)
        new_data = np.concatenate([old_data, array], axis=0)
    else:
        new_data = array
    np.save(file_path, new_data)