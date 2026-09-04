"""
data_collection.py
==================
Pipeline for executing Grid2Op simulations, forecasting grid states, assessing
epistemic/aleatoric uncertainty, and simulating line disconnections to generate
the final dataset for the downstream Gradient Boosting classifier.
"""

import os
import sys
import datetime
import joblib
import numpy as np
import pandas as pd
import torch
from typing import List, Any, Dict
from pathlib import Path
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import grid2op
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import BaseActionBudget, RandomLineOpponent
from grid2op.Exceptions import Grid2OpException
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNReward

# Local Imports
from config import CFG, DEVICE, TRAIN_MODE, TEST_SINGLE_EPISODE
from training_enn import get_uncertainty, load_trained_enn, _scaler_path
from utils import get_features, compute_grid_stats
from curriculumagent.baseline.baseline import CurriculumAgent

VERBOSE = os.environ.get("RUN_PIPELINE_VERBOSE", "1") == "1"


# =============================================================================
# Helper Classes & Functions
# =============================================================================

def convert_to_cos_sin(value: float, max_value: float) -> tuple[float, float]:
    """Encodes cyclical continuous features (e.g., time) into sine/cosine components."""
    value_cos = np.cos(2 * np.pi * value / max_value)
    value_sin = np.sin(2 * np.pi * value / max_value)
    return float(value_cos), float(value_sin)


class MockObs:
    """
    Lightweight stand-in for a Grid2Op observation used during autoregressive forecasting.

    Attributes mirror only what is needed by `get_features_with_history` and
    `compute_grid_stats`. `gen_v` and `gen_q` are stored so that `_forecasted_inj`
    injection dicts can reference them without raising AttributeError.
    """
    def __init__(self, load_p, load_q, gen_p, dt, gen_v=None, gen_q=None):
        self.load_p = np.asarray(load_p, dtype=np.float32)
        self.load_q = np.asarray(load_q, dtype=np.float32)
        self.gen_p  = np.asarray(gen_p,  dtype=np.float32)
        self.gen_v  = np.asarray(gen_v,  dtype=np.float32) if gen_v is not None else np.zeros(len(gen_p), dtype=np.float32)
        self.gen_q  = np.asarray(gen_q,  dtype=np.float32) if gen_q is not None else np.zeros(len(gen_p), dtype=np.float32)
        self.hour_of_day    = dt.hour
        self.minute_of_hour = dt.minute
        self.day_of_week    = dt.weekday()
        self.day            = dt.day
        # Note: 'rho' is intentionally absent — compute_grid_stats handles it gracefully.


def get_features_with_history(observations_array: List[Any], obs: Any) -> np.ndarray:
    """Extracts the state feature vector including historical lags (1h, 1d, 1w)."""
    sz_load_p = len(obs.load_p)
    sz_load_q = len(obs.load_q)
    sz_gen_p = len(obs.gen_p)

    def extract_features(past_obs, attr, size):
        if past_obs is not None:
            return getattr(past_obs, attr)
        return np.full(size, np.nan)

    current_idx = len(observations_array) - 1

    # Fetch historical observations (assumes 5-minute intervals)
    past_obs_hour = observations_array[current_idx - 12] if current_idx >= 12 else None
    past_obs_day = observations_array[current_idx - 288] if current_idx >= 288 else None
    past_obs_week = observations_array[current_idx - 2016] if current_idx >= 2016 else None

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

    hour_cos, hour_sin = convert_to_cos_sin(obs.hour_of_day, 23)
    minute_cos, minute_sin = convert_to_cos_sin(obs.minute_of_hour, 59)
    dow_cos, dow_sin = convert_to_cos_sin(obs.day_of_week, 6)

    temporal_features = [obs.day, hour_cos, hour_sin, minute_cos, minute_sin, dow_cos, dow_sin]

    return np.concatenate([feature_vec, temporal_features]).flatten()


def save_incremental(df: pd.DataFrame, path: str) -> None:
    """
    Saves the DataFrame to a CSV file incrementally.
    Creates a header if the file does not exist, otherwise appends data.
    """
    if df is None or df.empty:
        return

    file_exists = os.path.isfile(path)
    df.to_csv(path, mode='a', index=False, header=not file_exists)


# =============================================================================
# Core Analysis Logic
# =============================================================================

def analyze_disconnection_effect(
        env,
        model_predict,
        model_aleatoric,
        model_enn,
        obs,
        observations_array,
        ep,
        agent,
        scaler
) -> pd.DataFrame:
    """
    Pipeline:
      1. Forecast state at t+12 using mean & aleatoric models.
         Injects predicted values into an env.copy() to obtain a real powerflow
         at t+12 -> yields grid stats and epistemic_after.
      2. Advances 12 real steps in an env.copy() using the CurriculumAgent.
         At t+12, tests the disconnection of each critical line -> logs failure/success.
    """
    results = []

    # ------------------------------------------------------------------ #
    # Epistemic Uncertainty at t=0                                    #
    # ------------------------------------------------------------------ #
    try:
        obs_vect = obs.to_vect()[:CFG.ENN_INPUT_DIM].reshape(1, -1)
        unc_epistemic_before = float(get_uncertainty(model_enn, obs_vect))
    except Exception as e:
        print(f"[ERROR] ENN Epistemic Calculation (Before): {e}")
        unc_epistemic_before = float("nan")

    # ------------------------------------------------------------------ #
    # Forecast t+12 & Powerflow Simulation                            #
    # ------------------------------------------------------------------ #
    final_sigma_load_p = np.zeros(CFG.NO_LOADS)
    final_sigma_load_q = np.zeros(CFG.NO_LOADS)
    final_sigma_gen_p  = np.zeros(CFG.NO_GENS)
    fcast_grid_stats   = {}
    unc_epistemic_after = float("nan")

    try:
        x_t12  = get_features_with_history(observations_array, obs)
        y_pred = model_predict.predict([x_t12])[0]
        z_pred = model_aleatoric.predict([x_t12])[0]

        load_p_t12 = y_pred[:CFG.NO_LOADS]
        load_q_t12 = y_pred[CFG.NO_LOADS: CFG.NO_LOADS * 2]
        gen_p_t12  = np.clip(y_pred[CFG.NO_LOADS * 2:], CFG.GEN_MIN, CFG.GEN_MAX)

        y_var  = np.expm1(np.clip(z_pred, 0.0, None))
        y_sigma = np.sqrt(y_var + 1e-12)

        final_sigma_load_p = y_sigma[:CFG.NO_LOADS]
        final_sigma_load_q = y_sigma[CFG.NO_LOADS: CFG.NO_LOADS * 2]
        final_sigma_gen_p  = y_sigma[CFG.NO_LOADS * 2:]

        # Obtain real powerflow with predicted values via _forecasted_inj
        now_ts  = obs.get_time_stamp()
        next_ts = now_ts + datetime.timedelta(minutes=60)  # t+12 steps = 60 min

        obs_copy = obs.copy()
        obs_copy._forecasted_inj = [
            (now_ts,  {"injection": {"load_p": obs.load_p, "load_q": obs.load_q, "prod_p": obs.gen_p, "prod_v": obs.gen_v}}),
            (next_ts, {"injection": {"load_p": load_p_t12, "load_q": load_q_t12, "prod_p": gen_p_t12, "prod_v": obs.gen_v}}),
        ]

        do_nothing = env.action_space({})
        sim_obs_forecast, _, _, _ = obs_copy.simulate(do_nothing)

        fcast_grid_stats = compute_grid_stats(sim_obs_forecast)
        sim_vect = sim_obs_forecast.to_vect()[:CFG.ENN_INPUT_DIM].reshape(1, -1)
        unc_epistemic_after = float(get_uncertainty(model_enn, sim_vect))

    except Exception as e:
        print(f"[ERROR] Forecast / Powerflow Injection: {e}")

    # ------------------------------------------------------------------ #
    # 2. Real Test: Agent acts for 12 steps + Line Disconnections        #
    # ------------------------------------------------------------------ #
    grid_stats_obs = compute_grid_stats(obs)

    # Advance 12 steps once with the agent in a copied environment
    agent_env = env.copy()
    sim_obs_t12 = obs
    reached_t12 = True
    reward_agent = env.reward_range[0]
    done_agent = False

    for _ in range(12):
        try:
            a = agent.act(sim_obs_t12, reward_agent, done_agent)
        except Exception:
            a = agent_env.action_space({})

        sim_obs_t12, reward_agent, done_agent, _ = agent_env.step(a)
        if done_agent:
            reached_t12 = False
            break

    # Test disconnection for each critical line independently
    for line_target in CFG.LINES_TO_TEST:
        try:
            line_id = line_target
            if isinstance(line_target, str) and hasattr(env, "name_line"):
                if line_target in env.name_line:
                    line_id = list(env.name_line).index(line_target)

            action_disc = env.action_space({"set_line_status": [(line_id, -1)]})

            if reached_t12:
                # Branch off the t+12 environment state and apply the disconnection
                test_env = agent_env.copy()
                _, _, failed, _ = test_env.step(action_disc)
                test_env.close()
            else:
                failed = True  # Episode terminated before reaching t+12

            results.append({
                "episode":               ep,
                "step":                  obs.current_step,
                "line_disconnected":     line_target,
                "failed":                1 if failed else 0,

                # Uncertainties
                "aleatoric_load_p_mean": float(np.mean(final_sigma_load_p)),
                "aleatoric_load_q_mean": float(np.mean(final_sigma_load_q)),
                "aleatoric_gen_p_mean":  float(np.mean(final_sigma_gen_p)),
                "epistemic_before":      float(unc_epistemic_before),
                "epistemic_after":       float(unc_epistemic_after),

                # Observed Grid Stats (t=0)
                "sum_load_p":            grid_stats_obs["sum_load_p"],
                "sum_load_q":            grid_stats_obs["sum_load_q"],
                "sum_gen_p":             grid_stats_obs["sum_gen_p"],
                "var_line_rho":          grid_stats_obs["var_line_rho"],
                "avg_line_rho":          grid_stats_obs["avg_line_rho"],
                "max_line_rho":          grid_stats_obs["max_line_rho"],
                "nb_rho_ge_0.95":        grid_stats_obs["nb_rho_ge_0.95"],
                "load_gen_ratio":        grid_stats_obs["sum_load_p"] / (grid_stats_obs["sum_gen_p"] + 1e-6),

                # Forecasted Grid Stats (t+12)
                "fcast_sum_load_p":      fcast_grid_stats.get("sum_load_p",     np.nan),
                "fcast_sum_load_q":      fcast_grid_stats.get("sum_load_q",     np.nan),
                "fcast_sum_gen_p":       fcast_grid_stats.get("sum_gen_p",      np.nan),
                "fcast_var_line_rho":    fcast_grid_stats.get("var_line_rho",   np.nan),
                "fcast_avg_line_rho":    fcast_grid_stats.get("avg_line_rho",   np.nan),
                "fcast_max_line_rho":    fcast_grid_stats.get("max_line_rho",   np.nan),
                "fcast_nb_rho_ge_0.95":  fcast_grid_stats.get("nb_rho_ge_0.95", np.nan),
            })

        except Exception as e:
            print(f"[ERROR] Evaluating Line {line_target}: {e}")
            continue

    agent_env.close()
    return pd.DataFrame(results)


# =============================================================================
# Execution Phases
# =============================================================================

def run_single_episode_test(
        episode_id: int,
        model_predict: Any,
        model_aleatoric: Any,
        model_enn: Any,
        scaler: Any,
) -> None:
    """Runs prediction and analysis for a single specific episode for debugging."""
    print(f"\n[TEST] Running Single Episode Test. ID: {episode_id}")

    env = grid2op.make(CFG.ENV_NAME, reward_class=L2RPNReward, backend=LightSimBackend())
    agent = CurriculumAgent(env.action_space, env.observation_space, name="CA")

    try:
        agent.load(CFG.AGENT_PATH)
    except Exception:
        print("[WARNING] Agent could not be loaded, using random/do-nothing strategy.")

    obs = env.reset(seed=episode_id)
    done = False
    observations_array = []
    results_list = []
    try:
        max_steps = env.max_episode_duration()
    except Exception:
        max_steps = None
    progress = tqdm(
        total=max_steps if isinstance(max_steps, int) and max_steps > 0 else None,
        desc="Running test episode",
        unit="step",
    )

    while not done:
        if VERBOSE:
            tqdm.write(f"[TEST] Episode {episode_id} | Step {obs.current_step}")
        observations_array.append(obs)

        action = agent.act(obs, 0.0, done)

        if obs.current_step > 12 and obs.current_step % 20 == 0:
            df = analyze_disconnection_effect(
                env, model_predict, model_aleatoric, model_enn,
                obs, observations_array, episode_id, agent, scaler
            )
            if not df.empty:
                df['phase'] = "SingleTest"
                results_list.append(df)

                failures = df[df['failed'] == 1]
                if not failures.empty:
                    print(f"    -> [WARNING] Potential failures detected on lines: {failures['line_disconnected'].tolist()}")

        obs, _, done, _ = env.step(action)
        progress.update(1)

    progress.close()
    if results_list:
        full_df = pd.concat(results_list)
        save_incremental(full_df, CFG.CSV_OUTPUT_PATH)
        print(f"[TEST] Episode {episode_id} complete. Results saved successfully.")


def run_simulation_phase(
        phase_name: str,
        env_params: Dict[str, Any],
        ep_start: int,
        ep_end: int,
        model_predict: Any,
        model_aleatoric: Any,
        model_enn: Any,
        scaler: Any
) -> None:
    """Executes a batch of episodes to collect bulk data."""
    print(f"\n>>> STARTING PHASE: {phase_name} (Episodes {ep_start} to {ep_end})")

    env = grid2op.make(CFG.ENV_NAME, backend=LightSimBackend(), **env_params)
    agent = CurriculumAgent(env.action_space, env.observation_space, name="CA")

    try:
        agent.load(CFG.AGENT_PATH)
    except Exception:
        print("[WARNING] CurriculumAgent not loaded. Proceeding with fallback.")

    for ep in tqdm(range(ep_start, ep_end), desc=f"{phase_name} episodes", unit="episode"):
        obs = env.reset(seed=ep)
        done = False
        observations_array = []

        while not done:
            observations_array.append(obs)
            try:
                action = agent.act(obs, 0.0, done)
            except Exception:
                action = env.action_space({})

            if obs.current_step > 12 and obs.current_step % 20 == 0:
                # Dimension sanity check to prevent tensor shape crashes if grid topology varies
                if len(obs.to_vect()) == CFG.ENN_INPUT_DIM:
                    df = analyze_disconnection_effect(
                        env, model_predict, model_aleatoric, model_enn,
                        obs, observations_array, ep, agent, scaler
                    )
                    if not df.empty:
                        df['phase'] = phase_name
                        save_incremental(df, CFG.CSV_OUTPUT_PATH)
                else:
                    print(f"[SKIP] Step {obs.current_step} skipped due to dimension mismatch.")

            obs, _, done, _ = env.step(action)

    env.close()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    output_dir = os.path.dirname(CFG.CSV_OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INIT] Output directory verified: {output_dir}")

    if TRAIN_MODE and os.path.exists(CFG.CSV_OUTPUT_PATH):
        os.remove(CFG.CSV_OUTPUT_PATH)
        print(f"[INIT] Cleared previous dataset at {CFG.CSV_OUTPUT_PATH}")

    print(">>> Loading Trained Models (Forecaster, Aleatoric, ENN)...")
    try:
        model_predict   = joblib.load(CFG.MODEL_MEAN_PATH)
        model_aleatoric = joblib.load(CFG.MODEL_ALEATORIC_PATH)

        # Delegate ENN and Scaler loading to the robust functions in training_enn.py
        scaler = joblib.load(_scaler_path())
        model_enn = load_trained_enn()
        print(">>> All models and scaler loaded successfully.")

    except Exception as e:
        print(f"[FATAL ERROR] Model loading failed: {e}")
        print("Ensure the prerequisite training pipelines (train_forecast.py and train_enn.py) were executed.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Execution Workflow
    # -------------------------------------------------------------------------
    if TEST_SINGLE_EPISODE:
        run_single_episode_test(
            CFG.PROBA_TEST_EPISODE_ID,
            model_predict, model_aleatoric, model_enn, scaler
        )
    else:
        # Phase 1: Baseline behavior
        run_simulation_phase("Standard", {}, 700, 800, model_predict, model_aleatoric, model_enn, scaler)

        # Phase 2 & 3: Attack Scenarios
        opp_kwargs = {
            "opponent_action_class": PowerlineSetAction,
            "opponent_class": RandomLineOpponent,
            "opponent_budget_class": BaseActionBudget,
            "kwargs_opponent": {"lines_attacked": CFG.LINES_TO_TEST, "seed": 42}
        }

        run_simulation_phase("HeavyAttack_1", {**opp_kwargs, "opponent_attack_cooldown": 288},
                             700, 800, model_predict, model_aleatoric, model_enn, scaler)

        run_simulation_phase("HeavyAttack_2", {**opp_kwargs, "opponent_attack_cooldown": 128},
                             700, 800, model_predict, model_aleatoric, model_enn, scaler)
