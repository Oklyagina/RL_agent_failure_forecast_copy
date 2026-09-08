import os
import sys
from pathlib import Path

import torch
import numpy as np
from typing import List, Dict, Any

# ==============================================================================
# GLOBAL SETTINGS (Module Level)
# ==============================================================================
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from project_config import (
    AGENT_NAME,
    ARTIFACTS_DIR,
    ASSETS_DIR,
    ENV_DIR,
    ENV_NAME as PROJECT_ENV_NAME,
)

os.environ.setdefault("GRID2OP_DATA_PATH", str(ENV_DIR.parent))

PIPELINE_DIR = ARTIFACTS_DIR / PROJECT_ENV_NAME / AGENT_NAME
PIPELINE_DATA_DIR = PIPELINE_DIR / "data"
PIPELINE_MODEL_DIR = PIPELINE_DIR / "model"


def _resolve_agent_path() -> Path:
    agent_path = ASSETS_DIR / PROJECT_ENV_NAME
    if (agent_path / "model").is_dir() and (agent_path / "actions").is_dir():
        return agent_path
    for candidate in (ASSETS_DIR / "network36", ROOT / "src" / "models" / "network36"):
        if (candidate / "model").is_dir() and (candidate / "actions").is_dir():
            return candidate
    return agent_path


BASE_DIR = str(ROOT)
DATA_DIR = str(PIPELINE_DATA_DIR)
MODELS_DIR = str(PIPELINE_MODEL_DIR)
MODELS_FORECASTER = str(PIPELINE_MODEL_DIR)
AGENT_DIR = str(ASSETS_DIR)
ENN_DATA_DIR = os.path.join(MODELS_DIR, "enn_data")
OUTPUT_DIR_LLM = str(ROOT / "llm_rule_results")

# Create directories automatically
for d in [DATA_DIR, MODELS_DIR, ENN_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Execution Flags
TRAIN_MODE = True
TEST_SINGLE_EPISODE = False
PREDICT_PROBA_MODE = False
LLM_RULE_MODE = False


# ==============================================================================
# UNIFIED CONFIGURATION (CFG)
# ==============================================================================
class CFG:
    """Consolidated configuration for the 36-bus environment."""

    # Environment Metadata
    ENV_NAME = str(ENV_DIR)
    NO_LOADS = 37
    NO_GENS = 22
    NO_LINES = 59
    GEN_MAX = np.array([
        50.0, 67.2, 50.0, 250.0, 50.0, 33.6, 37.3, 37.3, 33.6, 74.7, 100.0,
        37.3, 37.3, 100.0, 74.7, 74.7, 150.0, 67.2, 74.7, 400.0, 300.0, 350.0
    ])
    GEN_MIN = np.zeros(22)

    LINES_TO_TEST = [
        "62_58_180", "62_63_160", "48_50_136", "48_53_141",
        "41_48_131", "39_41_121", "43_44_125", "44_45_126",
        "34_35_110", "54_58_154"
    ]

    # ENN Architecture
    ENN_INPUT_DIM = 1363
    ENN_NUM_CLASSES = 250
    ENN_HIDDEN_DIM = 256
    ENN_DROPOUT = 0.35

    # ENN Training Hyperparameters (Migrated from training_enn.py)
    ENN_TOP_K = 120
    ENN_EPOCHS = 300
    ENN_BATCH_SIZE = 128
    ENN_MAX_LR = 3e-4
    ENN_WEIGHT_DECAY = 5e-5
    ENN_EDL_LAMBDA = 0.01
    ENN_NOISE_STD = 0.02
    ENN_WARMUP = 10
    ENN_PATIENCE = 35

    # Model and Data Paths
    MODEL_MEAN_PATH = os.path.join(MODELS_FORECASTER, "HBGB_36.pkl")
    MODEL_ALEATORIC_PATH = os.path.join(MODELS_DIR, "HBGB_36_aleatoric.pkl")
    MODEL_ENN_PATH = os.path.join(MODELS_DIR, "enn_36.pth")
    MODEL_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "final_classifier_36.pkl")
    AGENT_PATH = str(_resolve_agent_path())

    X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train_36.npy")
    Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train_36.npy")
    X_TEST_PATH = os.path.join(DATA_DIR, "X_test36.npy")
    Y_TEST_PATH = os.path.join(DATA_DIR, "Y_test36.npy")
    CSV_OUTPUT_PATH = os.path.join(DATA_DIR, "uncertainty_disconnection_analysis.csv")

    # Tutor/Dataset Paths
    TUTOR_DIR = os.path.join(AGENT_PATH, "tutor", "junior_data")
    TRAIN_FILE = os.path.join(TUTOR_DIR, "test_train.npz")
    VAL_FILE = os.path.join(TUTOR_DIR, "test_val.npz")
    TEST_FILE = os.path.join(TUTOR_DIR, "test_test.npz")

    # LLM symbolic rule inference
    LLM_RULE_MODE = False
    # Set LLM_RULES_DIR to the folder containing line_*/best_rule.py files.
    # Leave it as None to let run_pipeline.py auto-detect dual_llm.py outputs.
    LLM_RULES_DIR = None
    LLM_RULES_EPISODE = 50
    LLM_ENN_BUNDLE = "refactored"


# Helper for Model Initialization
ENN_PARAMS: Dict[str, int] = {
    'input_dim': CFG.ENN_INPUT_DIM,
    'num_classes': CFG.ENN_NUM_CLASSES,
    'hidden_dim': CFG.ENN_HIDDEN_DIM,
}
