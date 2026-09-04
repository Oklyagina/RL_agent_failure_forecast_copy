import os
import json
import joblib
import optuna
import pandas as pd
import numpy as np

from typing import Tuple, List, Any, Dict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# Local imports
from config import CFG, TRAIN_MODE, PREDICT_PROBA_MODE

# ==============================================================================
# DYNAMIC LINE MAPPING
# ==============================================================================
# Deterministic encoding: maps line names to integers based on the central config.
LINE_MAP: Dict[str, int] = {line: idx for idx, line in enumerate(CFG.LINES_TO_TEST)}

# ==============================================================================
# DATA PREPARATION & METRICS
# ==============================================================================

def load_and_prep_data(filepath: str) -> pd.DataFrame:
    """Loads the CSV dataset, handles missing values, and encodes categorical features."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    df["line_id_encoded"] = df["line_disconnected"].map(LINE_MAP)

    unknown_lines = df["line_id_encoded"].isna()
    if unknown_lines.any():
        unknown = df.loc[unknown_lines, "line_disconnected"].unique().tolist()
        print(f"[WARNING] Unmapped lines found (encoded as -1): {unknown}")
        df["line_id_encoded"] = df["line_id_encoded"].fillna(-1)

    df["line_id_encoded"] = df["line_id_encoded"].astype(int)
    df["load_gen_ratio"] = df["sum_load_p"] / (df["sum_gen_p"] + 1e-6)
    df["label"] = df["failed"].astype(int)

    cols_to_clean = [
        "epistemic_before", "sum_load_p", "sum_load_q", "sum_gen_p",
        "var_line_rho", "avg_line_rho", "max_line_rho", "nb_rho_ge_0.95",
        "aleatoric_load_p_mean", "aleatoric_load_q_mean", "aleatoric_gen_p_mean",
        "load_gen_ratio", "epistemic_after"
    ]

    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

    return df

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Calculates False Alarm (FA) rate, Oversight (OVR) rate, and Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    denom_fa = tn + fp
    fa_rate = (fp / denom_fa * 100.0) if denom_fa > 0 else 0.0

    denom_ovr = tp + fn
    ovr_rate = (fn / denom_ovr * 100.0) if denom_ovr > 0 else 0.0

    return fa_rate, ovr_rate, cm


# ==============================================================================
# OPTUNA HYPERPARAMETER TUNING
# ==============================================================================

def decode_class_weight(choice: str):
    """Safely decodes string representations of class weights into dictionaries."""
    if choice == "none": return None
    if choice == "balanced": return "balanced"
    if choice == "w3": return {0: 1, 1: 3}
    if choice == "w5": return {0: 1, 1: 5}
    if choice == "w10": return {0: 1, 1: 10}
    raise ValueError(f"Unknown class_weight_choice: {choice}")

def build_model_params(trial: optuna.Trial) -> Tuple[Dict[str, Any], float]:
    """Defines the search space for the HistGradientBoostingClassifier."""
    class_weight_choice = trial.suggest_categorical("class_weight_choice", ["none", "balanced", "w3", "w5", "w10"])

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 800),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 5.0, log=True),
        "class_weight": decode_class_weight(class_weight_choice),
        "random_state": 42
    }

    threshold = trial.suggest_float("threshold", 0.10, 0.90)
    return params, threshold

def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series,
              cat_indices: List[int], X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Optuna objective function. Trains models using Stratified K-Fold and
    evaluates using a custom risk metric: 0.4 * False Alarm + 0.6 * Oversight.
    """
    params, threshold = build_model_params(trial)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, _ in skf.split(X, y):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]

        model = HistGradientBoostingClassifier(categorical_features=cat_indices, **params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)

        # Calculate custom risk metric: Heavily penalize oversights (missing a failure)
        fa, ovr, _ = calculate_metrics(y_val.to_numpy(), preds)
        custom_loss = 0.4 * fa + 0.6 * ovr

        scores.append(custom_loss)

    return float(np.mean(scores))

def save_best_metadata(save_dir: str, config_name: str, model_params: Dict[str, Any], threshold: float) -> None:
    """Saves the best hyperparameters as a JSON artifact for traceability."""
    os.makedirs(save_dir, exist_ok=True)

    # class_weight might not be JSON serializable natively if it's a dict with int keys
    safe_params = model_params.copy()
    if isinstance(safe_params.get("class_weight"), dict):
        safe_params["class_weight"] = {str(k): v for k, v in safe_params["class_weight"].items()}

    metadata = {"best_model_params": safe_params, "best_threshold": threshold}
    save_path = os.path.join(save_dir, f"classifier_metadata_{config_name}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":

    if TRAIN_MODE:
        try:
            df = load_and_prep_data(CFG.CSV_OUTPUT_PATH)
        except FileNotFoundError:
            print(f"[ERROR] Data file not found at {CFG.CSV_OUTPUT_PATH}.")
            raise SystemExit(1)

        # Define the features
        features = [
            'line_id_encoded', 'sum_load_p', 'sum_load_q', 'sum_gen_p',
            'var_line_rho', 'avg_line_rho', 'max_line_rho', 'nb_rho_ge_0.95',
            'load_gen_ratio', 'fcast_sum_load_p', 'fcast_sum_load_q', 'fcast_sum_gen_p',
            'fcast_var_line_rho', 'fcast_avg_line_rho', 'fcast_max_line_rho', 'fcast_nb_rho_ge_0.95',
            "aleatoric_load_p_mean", "aleatoric_load_q_mean", "aleatoric_gen_p_mean",
            "epistemic_before", "epistemic_after"]

        config_name = "All features"

        models_dir = os.path.dirname(CFG.MODEL_CLASSIFIER_PATH)

        print(f"\n{'=' * 60}")
        print(f"[INFO] Optimizing and Training Final Production Strategy: {config_name}")
        print(f"{'=' * 60}")

        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"[FATAL ERROR] Missing features in dataset: {missing}")
            raise SystemExit(1)

        X = df[features].copy()
        y = df["label"].copy()

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y
        )

        cat_indices = [features.index('line_id_encoded')] if 'line_id_encoded' in features else None

        # --- OPTUNA OPTIMIZATION ---
        print(f"[OPTUNA] Starting hyperparameter search (Cross-Validation)")

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, cat_indices, X_val, y_val),
            n_trials=500,
            n_jobs=-1,
            show_progress_bar=True,
        )

        best_params = study.best_params.copy()
        best_threshold = best_params.pop("threshold")
        class_weight = decode_class_weight(best_params.pop("class_weight_choice"))

        final_params = best_params.copy()
        final_params["class_weight"] = class_weight

        print(f"[OPTUNA] Best Custom Loss: {study.best_value:.4f}")
        print(f"[OPTUNA] Best Parameters selected.")

        print(f"[TRAIN] Fitting final model of the data...")
        final_model = HistGradientBoostingClassifier(
            categorical_features=cat_indices, **final_params
        )
        final_model.fit(X_train, y_train)

        # --- SAVE ARTIFACTS ---
        model_path = os.path.join(models_dir, CFG.MODEL_CLASSIFIER_PATH)
        joblib.dump(final_model, model_path)

        # Optionally overwrite the generic path from config so other scripts find it easily
        joblib.dump(final_model, CFG.MODEL_CLASSIFIER_PATH)

        save_best_metadata(models_dir, config_name, final_params, best_threshold)
        print(f"[SAVE] Model and metadata saved successfully to {models_dir}")
        print("\n[INFO] Classification training pipeline completed.")

    elif PREDICT_PROBA_MODE:
        print("[INFO] PREDICT_PROBA_MODE is enabled. Single inference logic goes here.")
    else:
        print("[INFO] Please enable TRAIN_MODE or PREDICT_PROBA_MODE in config.py")
