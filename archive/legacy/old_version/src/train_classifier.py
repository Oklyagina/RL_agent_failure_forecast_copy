"""Training utilities for the line-disconnection failure classifier."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from .config import CFG, TRAIN_MODE, PREDICT_PROBA_MODE
except ImportError:  # pragma: no cover
    from config import CFG, TRAIN_MODE, PREDICT_PROBA_MODE

try:
    from project_config import AGENT_NAME, ENV_NAME
except ImportError:  # pragma: no cover
    AGENT_NAME, ENV_NAME = "agent", str(getattr(CFG, "ENV_NAME", "environment"))


LINE_MAP: Dict[str, int] = {str(line).replace("line_", ""): idx for idx, line in enumerate(CFG.LINES_TO_TEST)}

CLASSIFIER_FEATURES = [
    "line_id_encoded", "sum_load_p", "sum_load_q", "sum_gen_p",
    "var_line_rho", "avg_line_rho", "max_line_rho", "nb_rho_ge_0.95",
    "load_gen_ratio", "fcast_sum_load_p", "fcast_sum_load_q", "fcast_sum_gen_p",
    "fcast_var_line_rho", "fcast_avg_line_rho", "fcast_max_line_rho",
    "fcast_nb_rho_ge_0.95", "aleatoric_load_p_mean", "aleatoric_load_q_mean",
    "aleatoric_gen_p_mean", "epistemic_before", "epistemic_after",
]

ANALYSIS_REQUIRED_COLUMNS = {
    "line_disconnected", "failed",
    *(feature for feature in CLASSIFIER_FEATURES
      if feature not in {"line_id_encoded", "load_gen_ratio"}),
}


def normalize_line_name(line: Any) -> str:
    return str(line).replace("line_", "")


def encode_line(line: Any, line_map: Optional[Dict[str, int]] = None) -> int:
    mapping = LINE_MAP if line_map is None else line_map
    return int(mapping.get(normalize_line_name(line), -1))


def load_and_prep_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    missing = sorted(ANALYSIS_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    if df["line_disconnected"].isna().any():
        raise ValueError("'line_disconnected' contains missing values.")

    # Derive the encoding from the actual training data instead of assuming the
    # original 36-bus line list.  Stable first-seen order keeps artifacts
    # reproducible while allowing the classifier to be used on another Grid2Op
    # environment / contingency set.
    line_names = [normalize_line_name(value) for value in df["line_disconnected"]]
    line_map: Dict[str, int] = {}
    for name in line_names:
        if name not in line_map:
            line_map[name] = len(line_map)
    df["line_id_encoded"] = pd.Series(line_names, index=df.index).map(line_map).astype(int)
    df["load_gen_ratio"] = df["sum_load_p"] / (df["sum_gen_p"] + 1e-6)
    df["label"] = pd.to_numeric(df["failed"], errors="raise").astype(int)
    invalid_labels = sorted(set(df["label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"'failed' must be binary 0/1; found {invalid_labels}")

    # HistGradientBoosting supports NaNs natively; only remove infinities.
    for col in CLASSIFIER_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df.attrs["line_map"] = line_map
    return df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fa_rate = (fp / (tn + fp) * 100.0) if (tn + fp) else 0.0
    oversight_rate = (fn / (tp + fn) * 100.0) if (tp + fn) else 0.0
    return fa_rate, oversight_rate, cm


def decode_class_weight(choice: str):
    choices = {
        "none": None,
        "balanced": "balanced",
        "w3": {0: 1, 1: 3},
        "w5": {0: 1, 1: 5},
        "w10": {0: 1, 1: 10},
    }
    if choice not in choices:
        raise ValueError(f"Unknown class_weight_choice: {choice}")
    return choices[choice]


def build_model_params(trial: optuna.Trial) -> Tuple[Dict[str, Any], float]:
    class_weight_choice = trial.suggest_categorical(
        "class_weight_choice", ["none", "balanced", "w3", "w5", "w10"]
    )
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 800),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 5.0, log=True),
        "class_weight": decode_class_weight(class_weight_choice),
        "random_state": 42,
    }
    threshold = trial.suggest_float("threshold", 0.10, 0.90)
    return params, threshold


def _failure_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    matches = np.where(classes == 1)[0]
    if len(matches) != 1:
        raise ValueError(f"Classifier classes {classes.tolist()} do not contain failure class 1 exactly once.")
    return proba[:, int(matches[0])]


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_indices: Optional[List[int]],
) -> float:
    """Cross-validated risk objective using the *trial threshold* correctly."""
    params, threshold = build_model_params(trial)
    class_counts = y_train.value_counts()
    min_count = int(class_counts.min()) if len(class_counts) > 1 else 0
    n_splits = min(5, min_count)
    if n_splits < 2:
        raise ValueError("Classifier training needs at least two samples of each class for CV.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        model = HistGradientBoostingClassifier(categorical_features=cat_indices, **params)
        model.fit(X_tr, y_tr)
        probs = _failure_probability(model, X_va)
        preds = (probs >= threshold).astype(int)
        fa, oversight, _ = calculate_metrics(y_va.to_numpy(), preds)
        scores.append(0.4 * fa + 0.6 * oversight)
    return float(np.mean(scores))


def classifier_metadata_path(model_path: str | Path) -> Path:
    return Path(model_path).with_name("classifier_metadata.json")


def save_metadata(
    model_path: str | Path,
    model_params: Dict[str, Any],
    threshold: float,
    line_map: Optional[Dict[str, int]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    safe_params = dict(model_params)
    if isinstance(safe_params.get("class_weight"), dict):
        safe_params["class_weight"] = {str(k): v for k, v in safe_params["class_weight"].items()}
    payload = {
        "environment": str(ENV_NAME),
        "agent": str(AGENT_NAME),
        "features": CLASSIFIER_FEATURES,
        "line_map": LINE_MAP if line_map is None else line_map,
        "failure_class": 1,
        "decision_threshold": float(threshold),
        "model_params": safe_params,
        "metrics": metrics or {},
    }
    path = classifier_metadata_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def train_failure_classifier(
    dataframe: pd.DataFrame,
    *,
    model_path: str | Path,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[Any, dict]:
    missing = [feature for feature in CLASSIFIER_FEATURES if feature not in dataframe.columns]
    if missing:
        raise ValueError("Missing classifier features: " + ", ".join(missing))
    if "label" not in dataframe.columns:
        raise ValueError("Prepared classifier dataframe must contain 'label'.")

    X = dataframe[CLASSIFIER_FEATURES].copy()
    y = dataframe["label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Failure classifier requires both success (0) and failure (1) examples.")
    if int(y.value_counts().min()) < 3:
        raise ValueError(
            "Failure classifier needs at least three examples of each class so the "
            "hold-out split and cross-validation both contain success/failure examples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    cat_indices = [CLASSIFIER_FEATURES.index("line_id_encoded")]

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, cat_indices),
        n_trials=int(n_trials),
        n_jobs=1,
        show_progress_bar=False,
    )

    best = dict(study.best_params)
    threshold = float(best.pop("threshold"))
    class_weight = decode_class_weight(best.pop("class_weight_choice"))
    final_params = {**best, "class_weight": class_weight, "random_state": seed}

    model = HistGradientBoostingClassifier(
        categorical_features=cat_indices, **final_params
    )
    model.fit(X_train, y_train)
    probs = _failure_probability(model, X_test)
    preds = (probs >= threshold).astype(int)
    fa, oversight, cm = calculate_metrics(y_test.to_numpy(), preds)
    try:
        auc = float(roc_auc_score(y_test, probs))
    except ValueError:
        auc = float("nan")
    metrics = {
        "false_alarm_pct": fa,
        "oversight_pct": oversight,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "test_rows": int(len(y_test)),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    trained_line_map = {
        str(k): int(v) for k, v in dataframe.attrs.get("line_map", LINE_MAP).items()
    }
    metadata_path = save_metadata(
        model_path, final_params, threshold, line_map=trained_line_map, metrics=metrics
    )
    return model, {"threshold": threshold, "metadata_path": str(metadata_path), **metrics}


if __name__ == "__main__":
    if TRAIN_MODE:
        df = load_and_prep_data(CFG.CSV_OUTPUT_PATH)
        n_trials = int(getattr(CFG, "CLASSIFIER_OPTUNA_TRIALS", 100))
        _, info = train_failure_classifier(
            df,
            model_path=CFG.MODEL_CLASSIFIER_PATH,
            n_trials=n_trials,
        )
        print(f"[SAVE] Failure classifier: {CFG.MODEL_CLASSIFIER_PATH}")
        print(f"[SAVE] Metadata: {info['metadata_path']}")
        print(
            f"[TEST] FA={info['false_alarm_pct']:.2f}% | "
            f"oversight={info['oversight_pct']:.2f}% | AUC={info['roc_auc']:.3f}"
        )
    elif PREDICT_PROBA_MODE:
        print(
            "[INFO] Use src.failure_probability.FailureProbabilityPredictor for single-observation inference."
        )
    else:
        print("[INFO] No classifier training mode is active.")
