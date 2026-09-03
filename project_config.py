"""Shared project configuration loaded from .env and environment variables."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULTS = {

#============= General defaults =================#
    "ENV_NAME": "ai4realnet_small",
    "ENV_LOCATION": r"C:\Users",
    "AGENT_NAME": "curriculum",
    "ASSETS_DIR": "assets",
    "ARTIFACTS_DIR": "artifacts",

#============= Agent defaults =================#
    "CURRICULUM_ITERATIONS": "50",
    "CURRICULUM_JOBS": "1",
    "ROLLOUT_EPISODES": "50",

#============= ENN defaults =================#
    "ENN_EPOCHS": "100",
    "ENN_ANNEAL_EPOCHS": "10",
    "ENN_BATCH_SIZE": "512",
    "ENN_LR": "1e-3",
    "ENN_VAL_FRAC": "0.1",
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


#============= General config =================#
ENV_NAME = get_config("ENV_NAME")
ENV_LOCATION = get_path("ENV_LOCATION")
ENV_DIR = ENV_LOCATION / ENV_NAME
AGENT_NAME = get_config("AGENT_NAME")
ASSETS_DIR = get_path("ASSETS_DIR")
ARTIFACTS_DIR = get_path("ARTIFACTS_DIR")

#============= Agent config =================#
CURRICULUM_ITERATIONS = get_int("CURRICULUM_ITERATIONS")
CURRICULUM_JOBS = get_int("CURRICULUM_JOBS")
ROLLOUT_EPISODES = get_int("ROLLOUT_EPISODES")

#============= ENN config =================#
ENN_EPOCHS = get_int("ENN_EPOCHS")
ENN_ANNEAL_EPOCHS = get_int("ENN_ANNEAL_EPOCHS")
ENN_BATCH_SIZE = get_int("ENN_BATCH_SIZE")
ENN_LR = get_float("ENN_LR")
ENN_VAL_FRAC = get_float("ENN_VAL_FRAC")
EXAMPLE_N_STEPS = get_int("EXAMPLE_N_STEPS")
SEED = get_int("SEED")
