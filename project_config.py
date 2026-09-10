"""Shared project configuration loaded from .env and environment variables."""

import os
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULTS = {

#============= General defaults =================#
    "ENV_NAME": "ai4realnet_small",
    "ENV_LOCATION": "environment",

    "AGENT_NAME": "curriculum",
    "AGENT_FACTORY": "",
    "ASSETS_DIR": "assets",
    "ARTIFACTS_DIR": "artifacts",

#============= Agent defaults =================#
    "CURRICULUM_ITERATIONS": "50",
    "CURRICULUM_JOBS": "1",
    "CURRICULUM_TUTOR_DO_NOTHING_THRESHOLD": "0.85",
    "CURRICULUM_TUTOR_BEST_ACTION_THRESHOLD": "0.999",
    "CURRICULUM_TUTOR_MIN_UNIQUE_ROWS": "100",
    "ROLLOUT_EPISODES": "50",
    "ENN_ROLLOUT_MAX_STEPS": "0",

#============= ENN defaults =================#
    "ENN_EPOCHS": "100",
    "ENN_ANNEAL_EPOCHS": "10",
    "ENN_BATCH_SIZE": "512",
    "ENN_LR": "1e-3",
    "ENN_VAL_FRAC": "0.1",
    "CLASSIFIER_OPTUNA_TRIALS": "100",
    "EXAMPLE_N_STEPS": "5",
    "SEED": "0",
}

def _read_dotenv(path: Path) -> dict[str, str]:
    values = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values

_DOTENV = _read_dotenv(ROOT / ".env")

def get_config(name: str) -> str:
    return os.environ.get(name, _DOTENV.get(name, DEFAULTS[name]))


def get_int(name: str) -> int:
    return int(get_config(name))


def get_float(name: str) -> float:
    return float(get_config(name))


def get_path(name: str) -> Path:
    path = Path(get_config(name))
    if path.is_absolute():
        return path
    return ROOT / path


def configure_grid2op_warnings() -> None:
    """Suppress known non-actionable Grid2Op/lightsim2grid compatibility warnings."""
    warnings.filterwarnings(
        "ignore",
        message=r"You are using a legacy grid2op version, please upgrade grid2op\.",
        category=UserWarning,
        module=r"lightsim2grid\.lightSimBackend",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"There were some Nan in the pp_net\.trafo\[\"tap_step_degree\"\], they have been replaced by 0",
        category=UserWarning,
        module=r"lightsim2grid\.gridmodel\.from_pandapower\._aux_add_trafo",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"We found either some slack coefficient to be < 0\. or they were all 0\.We set them all to 1\.0 to avoid such issues",
        category=UserWarning,
        module=r"lightsim2grid\.gridmodel\.from_pandapower\._aux_add_slack",
    )


#============= General config =================#
ENV_NAME = get_config("ENV_NAME")
ENV_LOCATION = get_path("ENV_LOCATION")
ENV_DIR = ENV_LOCATION / ENV_NAME
AGENT_NAME = get_config("AGENT_NAME")
AGENT_FACTORY = get_config("AGENT_FACTORY")
ASSETS_DIR = get_path("ASSETS_DIR")
ARTIFACTS_DIR = get_path("ARTIFACTS_DIR")

#============= Agent config =================#
CURRICULUM_ITERATIONS = get_int("CURRICULUM_ITERATIONS")
CURRICULUM_JOBS = get_int("CURRICULUM_JOBS")
CURRICULUM_TUTOR_DO_NOTHING_THRESHOLD = get_float("CURRICULUM_TUTOR_DO_NOTHING_THRESHOLD")
CURRICULUM_TUTOR_BEST_ACTION_THRESHOLD = get_float("CURRICULUM_TUTOR_BEST_ACTION_THRESHOLD")
CURRICULUM_TUTOR_MIN_UNIQUE_ROWS = get_int("CURRICULUM_TUTOR_MIN_UNIQUE_ROWS")
ROLLOUT_EPISODES = get_int("ROLLOUT_EPISODES")
ENN_ROLLOUT_MAX_STEPS = get_int("ENN_ROLLOUT_MAX_STEPS")

#============= ENN config =================#
ENN_EPOCHS = get_int("ENN_EPOCHS")
ENN_ANNEAL_EPOCHS = get_int("ENN_ANNEAL_EPOCHS")
ENN_BATCH_SIZE = get_int("ENN_BATCH_SIZE")
ENN_LR = get_float("ENN_LR")
ENN_VAL_FRAC = get_float("ENN_VAL_FRAC")
CLASSIFIER_OPTUNA_TRIALS = get_int("CLASSIFIER_OPTUNA_TRIALS")
EXAMPLE_N_STEPS = get_int("EXAMPLE_N_STEPS")
SEED = get_int("SEED")
