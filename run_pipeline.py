import os
import sys
import subprocess
import time
import logging
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

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
from src.pipeline_artifacts import (
    ArtifactStatus,
    StageCheck,
    classify_outputs,
    mark_provenance_stale,
    provenance_path,
    write_provenance,
)

sys.modules["config"] = src_config
os.environ.setdefault("GRID2OP_DATA_PATH", str(ENV_DIR.parent))

PIPELINE_DIR = ARTIFACTS_DIR / ENV_NAME / AGENT_NAME
PIPELINE_DATA_DIR = PIPELINE_DIR / "data"
PIPELINE_MODEL_DIR = PIPELINE_DIR / "model"

agent_path = ASSETS_DIR / ENV_NAME

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

def execute_module(
    module_path: str,
    verbose: bool = False,
    module_args: Optional[Sequence[str]] = None,
) -> None:
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
module_args = sys.argv[2:]
sys.argv = [module_path, *module_args]
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
    command = [sys.executable, "-c", bootstrap, module_path, *(module_args or [])]

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
# Environment-aware resumable training pipeline
# =============================================================================

FORECAST_DATA_PATHS = (
    Path(CFG.X_TRAIN_PATH), Path(CFG.Y_TRAIN_PATH),
    Path(CFG.X_TEST_PATH), Path(CFG.Y_TEST_PATH),
)
FORECAST_DATA_PRIMARY = PIPELINE_DATA_DIR / "forecast_data"
MEAN_MODEL_PATH = Path(CFG.MODEL_MEAN_PATH)
ALEATORIC_MODEL_PATH = Path(CFG.MODEL_ALEATORIC_PATH)
ENN_MODEL_PATH = Path(CFG.MODEL_ENN_PATH)
ENN_SCALER_PATH = PIPELINE_MODEL_DIR / f"scaler_{ENV_NAME}_enn.pkl"
ENN_META_PATH = PIPELINE_MODEL_DIR / f"enn_meta_{ENV_NAME}.json"
ANALYSIS_PATH = Path(CFG.CSV_OUTPUT_PATH)
CLASSIFIER_PATH = Path(CFG.MODEL_CLASSIFIER_PATH)


def _inspect_environment() -> Dict[str, int]:
    import grid2op
    from lightsim2grid import LightSimBackend

    try:
        env = grid2op.make(str(ENV_DIR), backend=LightSimBackend())
        obs = env.reset()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load configured environment {ENV_NAME!r} from {ENV_DIR}: {exc}"
        ) from exc
    try:
        target_dim = 2 * len(obs.load_p) + len(obs.gen_p)
        dimensions = {
            "observation_dim": int(len(obs.to_vect())),
            "forecast_input_dim": int(4 * target_dim + 7),
            "forecast_target_dim": int(target_dim),
            "loads": int(len(obs.load_p)),
            "generators": int(len(obs.gen_p)),
        }
    finally:
        env.close()

    configured_target = 2 * CFG.NO_LOADS + CFG.NO_GENS
    if configured_target != dimensions["forecast_target_dim"]:
        raise RuntimeError(
            "Configured load/generator counts do not match the current environment: "
            f"CFG expects target width {configured_target}, environment has "
            f"{dimensions['forecast_target_dim']}."
        )
    return dimensions


def _validate_agent_assets() -> None:
    root = Path(CFG.AGENT_PATH)
    required = [
        root / "actions" / "actions.npy",
        root / "model" / "saved_model.pb",
        root / "model" / "variables" / "variables.index",
        root / "model" / "variables" / "variables.data-00000-of-00001",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "CurriculumAgent assets are incomplete for the current .env. Missing: "
            + ", ".join(missing)
        )


def _check_with_validator(
    outputs: Iterable[Path],
    sidecar: Path,
    validator: Callable[[], Dict[str, int]],
) -> StageCheck:
    check = classify_outputs(outputs, sidecar, ENV_NAME, AGENT_NAME)
    if check.status == ArtifactStatus.MISSING:
        return check
    try:
        check.dimensions = validator()
    except Exception as exc:
        return StageCheck(ArtifactStatus.INCOMPATIBLE, str(exc))
    return check


def _validate_forecast_data(expected: Dict[str, int]) -> Dict[str, int]:
    import numpy as np

    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in FORECAST_DATA_PATHS]
    x_train, y_train, x_test, y_test = arrays
    for name, array in zip(("X_train", "y_train", "X_test", "Y_test"), arrays):
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(f"{name} must be a non-empty 2-D array; got {array.shape}")
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("forecast feature/target row counts do not match")
    if x_train.shape[1] != expected["forecast_input_dim"] or x_test.shape[1] != expected["forecast_input_dim"]:
        raise ValueError(
            f"forecast input width must be {expected['forecast_input_dim']}; "
            f"got train={x_train.shape[1]}, test={x_test.shape[1]}"
        )
    if y_train.shape[1] != expected["forecast_target_dim"] or y_test.shape[1] != expected["forecast_target_dim"]:
        raise ValueError(
            f"forecast target width must be {expected['forecast_target_dim']}; "
            f"got train={y_train.shape[1]}, test={y_test.shape[1]}"
        )
    return {
        "input_dim": int(x_train.shape[1]),
        "target_dim": int(y_train.shape[1]),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }


def _forecast_pair_valid(x_path: Path, y_path: Path, expected: Dict[str, int]) -> bool:
    import numpy as np

    try:
        x = np.load(x_path, mmap_mode="r", allow_pickle=False)
        y = np.load(y_path, mmap_mode="r", allow_pickle=False)
        return bool(
            x.ndim == y.ndim == 2 and len(x) > 0 and len(x) == len(y)
            and x.shape[1] == expected["forecast_input_dim"]
            and y.shape[1] == expected["forecast_target_dim"]
        )
    except Exception:
        return False


def _validate_regressor(path: Path, expected: Dict[str, int]) -> Dict[str, int]:
    import joblib

    model = joblib.load(path)
    input_dim = int(getattr(model, "n_features_in_", -1))
    estimators = getattr(model, "estimators_", None)
    output_dim = len(estimators) if estimators is not None else -1
    if input_dim != expected["forecast_input_dim"] or output_dim != expected["forecast_target_dim"]:
        raise ValueError(
            f"{path.name} dimensions are input={input_dim}, output={output_dim}; expected "
            f"input={expected['forecast_input_dim']}, output={expected['forecast_target_dim']}"
        )
    return {"input_dim": input_dim, "output_dim": output_dim}


def _validate_tutor_data(observation_dim: int) -> None:
    import numpy as np

    expected_keys = (
        (Path(CFG.TRAIN_FILE), "s_train", "a_train"),
        (Path(CFG.VAL_FILE), "s_validate", "a_validate"),
        (Path(CFG.TEST_FILE), "s_test", "a_test"),
    )
    for path, state_key, action_key in expected_keys:
        if not path.is_file():
            raise FileNotFoundError(f"Required ENN tutor split is missing: {path}")
        with np.load(path, mmap_mode="r", allow_pickle=False) as data:
            if state_key not in data or action_key not in data:
                raise ValueError(
                    f"{path} must contain {state_key!r} and {action_key!r}; found {data.files}"
                )
            states, actions = data[state_key], data[action_key]
            if states.ndim != 2 or len(states) == 0 or len(states) != len(actions):
                raise ValueError(
                    f"{path} is invalid: state shape={states.shape}, action shape={actions.shape}"
                )
            if states.shape[1] != observation_dim:
                raise ValueError(
                    f"{path} has {states.shape[1]} observation features; current environment "
                    f"{ENV_NAME} has {observation_dim}."
                )


def _validate_enn(expected: Dict[str, int]) -> Dict[str, int]:
    import joblib
    import torch

    meta = json.loads(ENN_META_PATH.read_text(encoding="utf-8"))
    meta_env = meta.get("environment")
    if meta_env and Path(str(meta_env)).name != ENV_NAME:
        raise ValueError(f"ENN metadata environment is {meta_env!r}, expected {ENV_NAME!r}")

    scaler = joblib.load(ENN_SCALER_PATH)
    scaler_dim = int(getattr(scaler, "n_features_in_", -1))
    state = torch.load(ENN_MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    matrices = [value for value in state.values() if getattr(value, "ndim", 0) == 2]
    if not matrices:
        raise ValueError("ENN checkpoint contains no two-dimensional weight tensors")
    input_dim = int(matrices[0].shape[1])
    num_classes = int(matrices[-1].shape[0])
    if input_dim != expected["observation_dim"] or scaler_dim != input_dim:
        raise ValueError(
            f"ENN/scaler dimensions are model={input_dim}, scaler={scaler_dim}; current "
            f"environment requires {expected['observation_dim']}"
        )
    meta_input = int(meta.get("input_dim", input_dim))
    meta_classes = int(meta.get("num_classes", num_classes))
    if meta_input != input_dim or meta_classes != num_classes:
        raise ValueError("ENN metadata dimensions do not match its checkpoint")
    return {"input_dim": input_dim, "num_classes": num_classes}


def _validate_analysis_csv() -> Dict[str, int]:
    import pandas as pd
    from src.train_classifier import ANALYSIS_REQUIRED_COLUMNS

    frame = pd.read_csv(ANALYSIS_PATH)
    if frame.empty:
        raise ValueError("analysis CSV is empty")
    missing = sorted(ANALYSIS_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("analysis CSV is missing columns: " + ", ".join(missing))
    return {"rows": int(len(frame)), "columns": int(len(frame.columns))}


def _validate_classifier() -> Dict[str, int]:
    import joblib
    from src.train_classifier import CLASSIFIER_FEATURES

    model = joblib.load(CLASSIFIER_PATH)
    input_dim = int(getattr(model, "n_features_in_", -1))
    if input_dim != len(CLASSIFIER_FEATURES):
        raise ValueError(
            f"classifier input width is {input_dim}; expected {len(CLASSIFIER_FEATURES)}"
        )
    return {"input_dim": input_dim}


def _write_stage_provenance(
    stage: str,
    primary: Path,
    outputs: Iterable[Path],
    dimensions: Dict[str, int],
    adopted: bool = False,
) -> None:
    write_provenance(
        provenance_path(primary), stage, ENV_NAME, AGENT_NAME,
        outputs, dimensions, adopted=adopted,
    )


def _print_preflight(rows: List[tuple[str, ArtifactStatus, str]]) -> None:
    print("\n  PIPELINE PREFLIGHT")
    print(f"  Environment : {ENV_NAME} ({ENV_DIR.resolve()})")
    print(f"  Agent       : {AGENT_NAME}")
    print(f"  Assets      : {Path(CFG.AGENT_PATH).resolve()}")
    print(f"  Artifacts   : {PIPELINE_DIR.resolve()}")
    print(f"\n  {'Stage':24} {'Status':20} Action")
    print(f"  {'-' * 24} {'-' * 20} {'-' * 24}")
    for stage, status, action in rows:
        print(f"  {stage:24} {status.value:20} {action}")


def run_training_pipeline(verbose: bool = False) -> None:
    """Validate, adopt, reuse, or train each pipeline artifact stage."""
    print("\n[MODE] INITIALIZING ENVIRONMENT-AWARE TRAINING PIPELINE")
    _validate_agent_assets()
    expected = _inspect_environment()

    data_sidecar = provenance_path(FORECAST_DATA_PRIMARY)
    data_check = _check_with_validator(
        FORECAST_DATA_PATHS, data_sidecar, lambda: _validate_forecast_data(expected)
    )
    mean_check = _check_with_validator(
        [MEAN_MODEL_PATH], provenance_path(MEAN_MODEL_PATH),
        lambda: _validate_regressor(MEAN_MODEL_PATH, expected),
    )
    aleatoric_check = _check_with_validator(
        [ALEATORIC_MODEL_PATH], provenance_path(ALEATORIC_MODEL_PATH),
        lambda: _validate_regressor(ALEATORIC_MODEL_PATH, expected),
    )
    enn_outputs = [ENN_MODEL_PATH, ENN_SCALER_PATH, ENN_META_PATH]
    enn_check = _check_with_validator(
        enn_outputs, provenance_path(ENN_MODEL_PATH), lambda: _validate_enn(expected)
    )
    analysis_check = _check_with_validator(
        [ANALYSIS_PATH], provenance_path(ANALYSIS_PATH), _validate_analysis_csv
    )
    classifier_check = _check_with_validator(
        [CLASSIFIER_PATH], provenance_path(CLASSIFIER_PATH), _validate_classifier
    )

    train_mean = not mean_check.reusable
    train_aleatoric = not aleatoric_check.reusable or train_mean
    need_forecast_data = train_mean or train_aleatoric
    collect_forecast_data = need_forecast_data and not data_check.reusable
    train_enn = not enn_check.reusable
    regenerate_analysis = (
        not analysis_check.reusable or train_mean or train_aleatoric or train_enn
    )
    train_classifier = not classifier_check.reusable or regenerate_analysis

    def action(check: StageCheck, run: bool, verb: str) -> str:
        if run:
            return verb
        if check.status == ArtifactStatus.LEGACY_ADOPTABLE:
            return "validate and adopt"
        return "reuse"

    analysis_status = (
        ArtifactStatus.STALE_DEPENDENCY
        if analysis_check.reusable and (train_mean or train_aleatoric or train_enn)
        else analysis_check.status
    )
    classifier_status = (
        ArtifactStatus.STALE_DEPENDENCY
        if classifier_check.reusable and regenerate_analysis
        else classifier_check.status
    )
    data_action = action(data_check, collect_forecast_data, "collect")
    if not need_forecast_data and not data_check.reusable:
        data_action = "not required by reusable models"
    _print_preflight([
        ("Forecast data", data_check.status, data_action),
        ("Mean forecaster", mean_check.status, action(mean_check, train_mean, "train")),
        ("Aleatoric forecaster", aleatoric_check.status,
         action(aleatoric_check, train_aleatoric, "train")),
        ("ENN bundle", enn_check.status, action(enn_check, train_enn, "train")),
        ("Analysis CSV", analysis_status,
         action(analysis_check, regenerate_analysis, "regenerate")),
        ("Classifier", classifier_status,
         action(classifier_check, train_classifier, "train")),
    ])
    print(
        f"\n  Dimensions: observation={expected['observation_dim']}, "
        f"forecast_input={expected['forecast_input_dim']}, "
        f"forecast_target={expected['forecast_target_dim']}"
    )

    # Adopt structurally compatible legacy artifacts before executing stages.
    adoption_specs = [
        ("forecast_data", FORECAST_DATA_PRIMARY, FORECAST_DATA_PATHS, data_check, False),
        ("mean_forecaster", MEAN_MODEL_PATH, [MEAN_MODEL_PATH], mean_check, train_mean),
        ("aleatoric_forecaster", ALEATORIC_MODEL_PATH, [ALEATORIC_MODEL_PATH],
         aleatoric_check, train_aleatoric),
        ("enn", ENN_MODEL_PATH, enn_outputs, enn_check, train_enn),
        ("analysis", ANALYSIS_PATH, [ANALYSIS_PATH], analysis_check, regenerate_analysis),
        ("classifier", CLASSIFIER_PATH, [CLASSIFIER_PATH], classifier_check, train_classifier),
    ]
    for stage, primary, outputs, check, will_run in adoption_specs:
        if check.status == ArtifactStatus.LEGACY_ADOPTABLE and not will_run:
            print(f"[ADOPT] {stage}: structurally valid legacy artifact for {ENV_NAME}/{AGENT_NAME}")
            _write_stage_provenance(stage, primary, outputs, check.dimensions, adopted=True)

    scheduled = [
        ("forecast_data", FORECAST_DATA_PRIMARY, collect_forecast_data),
        ("mean_forecaster", MEAN_MODEL_PATH, train_mean),
        ("aleatoric_forecaster", ALEATORIC_MODEL_PATH, train_aleatoric),
        ("enn", ENN_MODEL_PATH, train_enn),
        ("analysis", ANALYSIS_PATH, regenerate_analysis),
        ("classifier", CLASSIFIER_PATH, train_classifier),
    ]
    for stage, primary, will_run in scheduled:
        if will_run:
            mark_provenance_stale(
                provenance_path(primary), stage, ENV_NAME, AGENT_NAME,
                "stage scheduled to run",
            )

    if collect_forecast_data:
        force_train = not _forecast_pair_valid(
            Path(CFG.X_TRAIN_PATH), Path(CFG.Y_TRAIN_PATH), expected
        ) or data_check.status == ArtifactStatus.INCOMPATIBLE
        force_test = not _forecast_pair_valid(
            Path(CFG.X_TEST_PATH), Path(CFG.Y_TEST_PATH), expected
        ) or data_check.status == ArtifactStatus.INCOMPATIBLE
        args = ["--stage", "data"]
        if force_train:
            args.append("--force-train-data")
        if force_test:
            args.append("--force-test-data")
        execute_module("src/train_forecast.py", verbose=verbose, module_args=args)
        dimensions = _validate_forecast_data(expected)
        _write_stage_provenance(
            "forecast_data", FORECAST_DATA_PRIMARY, FORECAST_DATA_PATHS, dimensions
        )
    elif need_forecast_data and not data_check.reusable:
        raise RuntimeError(f"Forecast data is required but invalid: {data_check.reason}")

    if train_mean:
        execute_module(
            "src/train_forecast.py", verbose=verbose, module_args=["--stage", "mean"]
        )
        dimensions = _validate_regressor(MEAN_MODEL_PATH, expected)
        _write_stage_provenance("mean_forecaster", MEAN_MODEL_PATH, [MEAN_MODEL_PATH], dimensions)
    else:
        print(f"  SKIP: valid mean forecaster at {MEAN_MODEL_PATH}")

    if train_aleatoric:
        execute_module(
            "src/train_forecast.py", verbose=verbose, module_args=["--stage", "aleatoric"]
        )
        dimensions = _validate_regressor(ALEATORIC_MODEL_PATH, expected)
        _write_stage_provenance(
            "aleatoric_forecaster", ALEATORIC_MODEL_PATH,
            [ALEATORIC_MODEL_PATH], dimensions,
        )
    else:
        print(f"  SKIP: valid aleatoric forecaster at {ALEATORIC_MODEL_PATH}")

    if train_enn:
        _validate_tutor_data(expected["observation_dim"])
        execute_module("src/training_enn.py", verbose=verbose)
        dimensions = _validate_enn(expected)
        _write_stage_provenance("enn", ENN_MODEL_PATH, enn_outputs, dimensions)
    else:
        print(f"  SKIP: valid ENN bundle at {PIPELINE_MODEL_DIR}")

    if regenerate_analysis:
        execute_module("src/collect_data.py", verbose=verbose)
        dimensions = _validate_analysis_csv()
        _write_stage_provenance("analysis", ANALYSIS_PATH, [ANALYSIS_PATH], dimensions)
    else:
        print(f"  SKIP: valid analysis data at {ANALYSIS_PATH}")

    if train_classifier:
        execute_module("src/train_classifier.py", verbose=verbose)
        dimensions = _validate_classifier()
        _write_stage_provenance(
            "classifier", CLASSIFIER_PATH, [CLASSIFIER_PATH], dimensions
        )
    else:
        print(f"  SKIP: valid classifier at {CLASSIFIER_PATH}")


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
