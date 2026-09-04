import os
import sys
import subprocess
import time
import logging
import argparse
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from project_config import (AGENT_NAME, ARTIFACTS_DIR, ASSETS_DIR, ENV_DIR,
                            ENV_NAME)
from src import config as src_config
from src.config import CFG, TRAIN_MODE, PREDICT_PROBA_MODE, TEST_SINGLE_EPISODE

sys.modules["config"] = src_config
os.environ.setdefault("GRID2OP_DATA_PATH", str(ENV_DIR.parent))

PIPELINE_DIR = ARTIFACTS_DIR / ENV_NAME / AGENT_NAME
PIPELINE_DATA_DIR = PIPELINE_DIR / "data"
PIPELINE_MODEL_DIR = PIPELINE_DIR / "model"

agent_path = ASSETS_DIR / ENV_NAME
if not ((agent_path / "model").is_dir() and (agent_path / "actions").is_dir()):
    for candidate in (ASSETS_DIR / "network36", ROOT / "src" / "models" / "network36"):
        if (candidate / "model").is_dir() and (candidate / "actions").is_dir():
            agent_path = candidate
            break

for path in (PIPELINE_DATA_DIR, PIPELINE_MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)

CFG.ENV_NAME = str(ENV_DIR)
CFG.MODEL_MEAN_PATH = str(PIPELINE_MODEL_DIR / "HBGB_36.pkl")
CFG.MODEL_ALEATORIC_PATH = str(PIPELINE_MODEL_DIR / "HBGB_36_aleatoric.pkl")
CFG.MODEL_ENN_PATH = str(PIPELINE_MODEL_DIR / "enn_36.pth")
CFG.MODEL_CLASSIFIER_PATH = str(PIPELINE_MODEL_DIR / "final_classifier_36.pkl")
CFG.AGENT_PATH = str(agent_path)
CFG.X_TRAIN_PATH = str(PIPELINE_DATA_DIR / "X_train_36.npy")
CFG.Y_TRAIN_PATH = str(PIPELINE_DATA_DIR / "y_train_36.npy")
CFG.X_TEST_PATH = str(PIPELINE_DATA_DIR / "X_test36.npy")
CFG.Y_TEST_PATH = str(PIPELINE_DATA_DIR / "Y_test36.npy")
CFG.CSV_OUTPUT_PATH = str(PIPELINE_DATA_DIR / "uncertainty_disconnection_analysis.csv")
CFG.TUTOR_DIR = str(Path(CFG.AGENT_PATH) / "tutor" / "junior_data")
CFG.TRAIN_FILE = str(Path(CFG.TUTOR_DIR) / "test_train.npz")
CFG.VAL_FILE = str(Path(CFG.TUTOR_DIR) / "test_val.npz")
CFG.TEST_FILE = str(Path(CFG.TUTOR_DIR) / "test_test.npz")

# LLM_RULE_MODE is the new flag for symbolic rule inference.
# If it does not yet exist in config.py, it defaults to False.
try:
    from src.config import LLM_RULE_MODE
except ImportError:
    LLM_RULE_MODE = False


def configure_verbosity(verbose: bool = False) -> None:
    """Configure logging verbosity for this runner and imported modules."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, force=True)
    logging.getLogger().setLevel(level)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    except Exception:
        pass


# =============================================================================
# Subprocess executor (unchanged from original)
# =============================================================================

def execute_module(module_path: str, verbose: bool = False) -> None:
    """
    Executes a Python module as a subprocess, ensuring the project root
    is correctly appended to the PYTHONPATH to prevent module resolution errors.

    Args:
        module_path (str): The relative path to the Python script to execute.

    Raises:
        SystemExit: If the subprocess fails, it exits with the same error code.
    """
    print(f"\n{'=' * 60}")
    print(f" EXECUTING MODULE: {module_path}")
    print(f"{'=' * 60}\n")

    env = os.environ.copy()
    python_path = os.pathsep.join((str(ROOT), str(SRC_DIR)))
    env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["RUN_PIPELINE_VERBOSE"] = "1" if verbose else "0"

    start_time = time.time()
    bootstrap = """
import runpy
import sys
import run_pipeline

module_path = sys.argv[1]
run_pipeline.configure_verbosity(run_pipeline.os.environ.get("RUN_PIPELINE_VERBOSE") == "1")

if module_path.endswith("training_enn.py"):
    run_pipeline.CFG.ENV_NAME = run_pipeline.ENV_NAME
else:
    run_pipeline.CFG.ENV_NAME = str(run_pipeline.ENV_DIR)

if module_path.endswith("collect_data.py"):
    import training_enn
    training_enn._scaler_path = lambda: str(run_pipeline.PIPELINE_MODEL_DIR / ("scaler_" + run_pipeline.ENV_NAME + "_enn.pkl"))
    training_enn._best_weights_path = lambda: str(run_pipeline.PIPELINE_MODEL_DIR / ("enn_best_" + run_pipeline.ENV_NAME + ".pth"))
    training_enn._meta_path = lambda: str(run_pipeline.PIPELINE_MODEL_DIR / ("enn_meta_" + run_pipeline.ENV_NAME + ".json"))

runpy.run_path(module_path, run_name="__main__")
"""
    command = [sys.executable, "-c", bootstrap, module_path]

    try:
        if verbose:
            subprocess.run(command, check=True, env=env, cwd=str(ROOT))
        else:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                if "[INFO]" not in line:
                    print(line, end="")
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
        elapsed = time.time() - start_time
        print(f"\n  SUCCESS: {module_path} completed in {elapsed:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: {module_path} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)


# =============================================================================
# Training pipeline (unchanged from original)
# =============================================================================

def run_training_pipeline(verbose: bool = False) -> None:
    """
    Manages the execution flow of the full training pipeline, checking
    for existing artifacts to avoid redundant and expensive computations.
    """
    print("\n[MODE] INITIALIZING FULL TRAINING PIPELINE")

    if not os.path.exists(CFG.MODEL_MEAN_PATH):
        execute_module("src/train_forecast.py", verbose=verbose)
    else:
        print(f"  SKIP: Forecast model already exists at {CFG.MODEL_MEAN_PATH}")

    if not os.path.exists(CFG.MODEL_ENN_PATH):
        execute_module("src/training_enn.py", verbose=verbose)
    else:
        print(f"  SKIP: ENN model already exists at {CFG.MODEL_ENN_PATH}")

    if not os.path.exists(CFG.CSV_OUTPUT_PATH):
        execute_module("src/collect_data.py", verbose=verbose)
    else:
        print(f"  SKIP: Data already collected at {CFG.CSV_OUTPUT_PATH}")

    execute_module("src/train_classifier.py", verbose=verbose)


# =============================================================================
# LLM Rule Inference mode  ← NEW
# =============================================================================

def run_llm_rule_inference(verbose: bool = False) -> None:
    """
    Loads the trained models and the LLM symbolic rules, then runs one
    simulation episode. For each monitored line, at every analysis step it:

      1. Runs the forecast pipeline (t+12) — same logic as collect_data.py.
      2. Applies the symbolic rule for that line.
      3. Prints the binary prediction (0 = OK, 1 = predicted failure) and
         the natural-language explanation sentence (LaTeX-ready for the paper).

    Configuration in src/config.py:
      LLM_RULES_DIR      : path to the folder with line_*/best_rule.py files
                           e.g. "llm_rules_results/temp_0.5"
      LLM_RULES_EPISODE  : episode seed to simulate (default: CFG.PROBA_TEST_EPISODE_ID)

    The sentences for all available lines are also printed at startup so you
    can use them directly in the paper table.
    """
    configure_verbosity(verbose)

    import joblib
    import numpy as np
    import grid2op
    from lightsim2grid import LightSimBackend
    from grid2op.Reward import L2RPNReward
    from curriculumagent.baseline.baseline import CurriculumAgent

    # Local imports (src/ is on PYTHONPATH when run via run_pipeline.py)
    import training_enn
    training_enn._scaler_path = lambda: str(PIPELINE_MODEL_DIR / ("scaler_" + ENV_NAME + "_enn.pkl"))
    training_enn._best_weights_path = lambda: str(PIPELINE_MODEL_DIR / ("enn_best_" + ENV_NAME + ".pth"))
    training_enn._meta_path = lambda: str(PIPELINE_MODEL_DIR / ("enn_meta_" + ENV_NAME + ".json"))
    load_trained_enn = training_enn.load_trained_enn
    get_uncertainty = training_enn.get_uncertainty
    from src.utils import compute_grid_stats
    from src.collect_data import get_features_with_history
    from src.rule_predictor import RulePredictor

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    rules_dir      = getattr(CFG, "LLM_RULES_DIR",     "llm_rules_results/temp_0.5")
    episode_seed   = getattr(CFG, "LLM_RULES_EPISODE",  getattr(CFG, "PROBA_TEST_EPISODE_ID", 50))
    analysis_every = getattr(CFG, "ANALYSIS_STEP",       20)   # analyse every N steps

    print(f"\n[MODE] LLM RULE INFERENCE")
    print(f"  Rules directory : {rules_dir}")
    print(f"  Episode seed    : {episode_seed}")
    print(f"  Analysis every  : {analysis_every} steps")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("\n  Loading models...")
    try:
        model_predict   = joblib.load(CFG.MODEL_MEAN_PATH)
        model_aleatoric = joblib.load(CFG.MODEL_ALEATORIC_PATH)
        model_enn       = load_trained_enn()
        print("  All models loaded successfully.")
    except Exception as e:
        print(f"  [ERROR] Could not load models: {e}")
        print("  Run the training pipeline first (TRAIN_MODE=True).")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Instantiate RulePredictor  (once — reused across all steps)
    # ------------------------------------------------------------------
    observations_array = []   # accumulated observations for the episode

    rule_predictor = RulePredictor(
        rules_dir=rules_dir,
        model_predict=model_predict,
        model_aleatoric=model_aleatoric,
        model_enn=model_enn,
        observations_array=observations_array,
        compute_grid_stats_fn=compute_grid_stats,
        get_uncertainty_fn=get_uncertainty,
        get_features_with_history_fn=get_features_with_history,
    )

    # Print all rule sentences at startup (useful for the paper table)
    sentences = rule_predictor.all_sentences()
    if sentences:
        print(f"\n{'=' * 70}")
        print("  SYMBOLIC RULE SENTENCES (all available lines)")
        print(f"{'=' * 70}")
        for line_name, sentence in sentences.items():
            print(f"\n  [{line_name}]\n  {sentence}")
        print(f"\n{'=' * 70}\n")
    else:
        print(f"  [WARN] No rules found in '{rules_dir}'. Check the path.")

    # ------------------------------------------------------------------
    # Simulation episode
    # ------------------------------------------------------------------
    env   = grid2op.make(CFG.ENV_NAME, reward_class=L2RPNReward, backend=LightSimBackend())
    agent = CurriculumAgent(env.action_space, env.observation_space, name="CA")
    try:
        agent.load(CFG.AGENT_PATH)
    except Exception:
        print("  [WARN] Agent could not be loaded — using do-nothing fallback.")

    obs  = env.reset(seed=episode_seed)
    done = False
    reward = env.reward_range[0]

    print(f"  Starting episode (seed={episode_seed})...\n")

    try:
        max_steps = env.max_episode_duration()
    except Exception:
        max_steps = None
    progress = tqdm(
        total=max_steps if isinstance(max_steps, int) and max_steps > 0 else None,
        desc="Running LLM episode",
        unit="step",
    )

    while not done:
        observations_array.append(obs)
        # observations_array is the same list object referenced by rule_predictor,
        # so the predictor always sees the latest history automatically.

        if obs.current_step > 12 and obs.current_step % analysis_every == 0:
            print(f"  --- Step {obs.current_step} ---")

            for line_name in CFG.LINES_TO_TEST:
                # Normalise: LINES_TO_TEST may contain int IDs or string names
                if isinstance(line_name, int):
                    line_str = _line_id_to_name(line_name, env)
                else:
                    line_str = str(line_name)

                result = rule_predictor.predict(obs=obs, line_name=line_str)

                status = "FAILURE PREDICTED" if result["prediction"] == 1 else "OK"
                print(f"    Line {result['line_name']:15s} -> {status}")
                if result["prediction"] == 1:
                    print(f"      {result['sentence']}")

        try:
            action = agent.act(obs, reward, done)
        except Exception:
            action = env.action_space({})

        obs, reward, done, _ = env.step(action)
        progress.update(1)

    progress.close()
    env.close()
    print("\n  Episode finished.")


def _line_id_to_name(line_id: int, env) -> str:
    """Converts an integer line ID to its string name using env.name_line."""
    try:
        return str(env.name_line[line_id])
    except Exception:
        return str(line_id)


# =============================================================================
# Main entry point
# =============================================================================

def main(verbose: bool = False) -> None:
    """
    Routes execution based on the configuration flags in src/config.py:

      TRAIN_MODE=True      -> Full training pipeline (forecasters + ENN + data + classifier)
      TEST_SINGLE_EPISODE  -> Single episode simulation and data collection
      PREDICT_PROBA_MODE   -> Probabilistic inference on a single observation
      LLM_RULE_MODE=True   -> LLM symbolic rule inference: runs a simulation episode
                              and reports per-line failure predictions with
                              natural-language explanations (LaTeX-ready for the paper)

    To activate LLM rule inference, set in src/config.py:
        TRAIN_MODE          = False
        TEST_SINGLE_EPISODE = False
        PREDICT_PROBA_MODE  = False
        LLM_RULE_MODE       = True
    """
    configure_verbosity(verbose)

    environment_name = getattr(CFG, "ENV_NAME", "UNKNOWN")
    print(f"  CONFIGURATION: ENV={environment_name}")

    if TRAIN_MODE:
        run_training_pipeline(verbose=verbose)
    elif TEST_SINGLE_EPISODE:
        execute_module("src/collect_data.py", verbose=verbose)
    elif PREDICT_PROBA_MODE:
        execute_module("src/train_classifier.py", verbose=verbose)
    elif LLM_RULE_MODE:
        run_llm_rule_inference(verbose=verbose)
    else:
        print("\n  WARNING: No active execution mode selected in src/config.py.")
        print("  Set one of: TRAIN_MODE, TEST_SINGLE_EPISODE, PREDICT_PROBA_MODE, LLM_RULE_MODE = True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="show INFO logs")
    args = parser.parse_args()
    main(verbose=args.verbose)
