import os
import torch
import numpy as np
from typing import List, Dict, Any

# ==============================================================================
# GLOBAL SETTINGS (Module Level)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_FORECASTER = os.path.join(BASE_DIR, "forecasts")
AGENT_DIR = os.path.join(BASE_DIR, "agents")
ENN_DATA_DIR = os.path.join(MODELS_DIR, "enn_data")
OUTPUT_DIR_LLM = os.path.join(BASE_DIR, "llm_rule_results")

# Create directories automatically
for d in [DATA_DIR, MODELS_DIR, MODELS_FORECASTER, AGENT_DIR, ENN_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Execution Flags
TRAIN_MODE = True
TEST_SINGLE_EPISODE = False
PREDICT_PROBA_MODE = False


# ==============================================================================
# UNIFIED CONFIGURATION (CFG)
# ==============================================================================
class CFG:
    """Consolidated configuration for the 36-bus environment."""

    # Environment Metadata
    ENV_NAME = "l2rpn_icaps_2021_small"
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
    ENN_ANNEAL_EPOCHS = 20
    ENN_PATIENCE = 35
    ENN_ROLLOUT_EPISODES = 50
    ENN_ROLLOUT_MAX_STEPS = None
    CLASSIFIER_OPTUNA_TRIALS = 100
    SEED = 0

    # Model and Data Paths
    MODEL_MEAN_PATH = os.path.join(MODELS_FORECASTER, "HBGB_36.pkl")
    MODEL_ALEATORIC_PATH = os.path.join(MODELS_DIR, "HBGB_36_aleatoric.pkl")
    MODEL_ENN_PATH = os.path.join(MODELS_DIR, "enn_36.pth")
    MODEL_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "final_classifier_36.pkl")
    AGENT_PATH = os.path.join(AGENT_DIR, "network36")

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


# Helper for Model Initialization
ENN_PARAMS: Dict[str, int] = {
    'input_dim': CFG.ENN_INPUT_DIM,
    'num_classes': CFG.ENN_NUM_CLASSES,
    'hidden_dim': CFG.ENN_HIDDEN_DIM,
}
