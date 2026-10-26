"""Distil the current HGB failure classifier into readable per-line rules.

This is the small, offline LLM step used by the runtime explainability layer.
The generator, critic, and repair roles use one configured chat model, matching
the useful part of the legacy implementation without its paper-evaluation
orchestration.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

from project_config import AGENT_NAME, ARTIFACTS_DIR, ENV_NAME
from src.explainability.llm_client import LLMClient, LLMConfig
from src.explainability.rule_predictor import compile_rule, validate_rule
from src.explainability.shap_cache import ShapSettings, validate_rankings_cache
from src.failure_forecast import (
    CLASSIFIER_FEATURES,
    normalize_line_name,
    prepare_classifier_dataframe,
)


HERE = Path(__file__).resolve().parent
FAILURE_FORECAST_DIR = ARTIFACTS_DIR / ENV_NAME / AGENT_NAME / "failure_forecast"
DATA_PATH = FAILURE_FORECAST_DIR / "failure_forecast_rows.csv"
HGB_MODEL_PATH = FAILURE_FORECAST_DIR / "failure_classifier.pkl"
HGB_METADATA_PATH = FAILURE_FORECAST_DIR / "failure_classifier_metadata.json"
SHAP_RANKINGS_FILE = FAILURE_FORECAST_DIR / "explainability" / "shap_rankings.json"
OUTPUT_DIR = FAILURE_FORECAST_DIR / "explainability" / "rules"
PROMPTS_DIR = HERE / "prompts"

FEATURES = list(CLASSIFIER_FEATURES)
FEATURES_FOR_LLM = [feature for feature in FEATURES if feature != "line_id_encoded"]

FEATURE_DESCRIPTIONS = {
    "sum_load_p": "total active load now",
    "sum_load_q": "total reactive load now",
    "sum_gen_p": "total generation now",
    "var_line_rho": "variance of line loading now",
    "avg_line_rho": "average line loading now",
    "max_line_rho": "maximum line loading now",
    "nb_rho_ge_0.95": "number of highly loaded lines now",
    "load_gen_ratio": "load-to-generation ratio now",
    "aleatoric_load_p_mean": "active-load forecast uncertainty",
    "aleatoric_load_q_mean": "reactive-load forecast uncertainty",
    "aleatoric_gen_p_mean": "generation forecast uncertainty",
    "epistemic_before": "epistemic uncertainty now",
    "epistemic_after": "epistemic uncertainty in the forecast state",
    "fcast_sum_load_p": "total active load one hour ahead",
    "fcast_sum_load_q": "total reactive load one hour ahead",
    "fcast_sum_gen_p": "total generation one hour ahead",
    "fcast_var_line_rho": "variance of forecast line loading",
    "fcast_avg_line_rho": "average forecast line loading",
    "fcast_max_line_rho": "maximum forecast line loading",
    "fcast_nb_rho_ge_0.95": "forecast number of highly loaded lines",
}


@dataclass(frozen=True)
class DistillationConfig:
    """Small set of controls for one rule-distillation run."""

    iterations: int = 20
    temperature: float = 0.7
    top_features: int = 8
    max_error_examples: int = 8


def load_teacher_data(
    rows_path: str | Path,
    classifier_path: str | Path,
    metadata_path: str | Path,
) -> tuple[pd.DataFrame, Any, dict[str, int]]:
    """Load current artifacts and label rows with the HGB teacher decision."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if list(metadata.get("features", [])) != FEATURES:
        raise ValueError("Classifier metadata feature order does not match current code.")
    line_map = {str(key): int(value) for key, value in metadata["line_map"].items()}
    dataframe = prepare_classifier_dataframe(rows_path, line_map=line_map)
    dataframe["line_disconnected"] = dataframe["line_disconnected"].map(
        normalize_line_name
    )
    model = joblib.load(classifier_path)
    dataframe["target"] = model.predict(dataframe[FEATURES]).astype(int)
    return dataframe, model, line_map


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Return compact fidelity metrics, emphasizing failure recall through F2."""
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "f2": float(fbeta_score(actual, predicted, beta=2, zero_division=0)),
        "false_alarm_rate": float(fp / (fp + tn + 1e-12)),
        "missed_failure_rate": float(fn / (fn + tp + 1e-12)),
    }


def evaluate_rule(rule_code: str, rows: pd.DataFrame) -> np.ndarray:
    """Apply one validated rule to classifier feature rows."""
    rule = compile_rule(rule_code)
    return np.asarray([int(rule(row)) for _, row in rows.iterrows()], dtype=int)


def _fill_template(path: Path, values: Mapping[str, Any]) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text.strip()


def _extract_rule(text: str) -> str:
    tagged = re.search(
        r"\[Start of Rule\](.*?)\[End of Rule\]", text, re.I | re.S
    )
    if tagged:
        text = tagged.group(1)
    for block in re.findall(r"```(?:python)?\s*(.*?)```", text, re.I | re.S):
        if "def rule" in block:
            return block.strip()
    match = re.search(r"(def\s+rule\s*\(\s*\w+\s*\)\s*:[\s\S]*)", text)
    return match.group(1).strip() if match else ""


def _feature_order(rows: pd.DataFrame, ranking: Sequence[str] | None) -> list[str]:
    if ranking:
        selected = [feature for feature in ranking if feature in FEATURES_FOR_LLM]
        if selected:
            return selected
    failed = rows[rows["target"] == 1]
    normal = rows[rows["target"] == 0]
    if failed.empty or normal.empty:
        return list(FEATURES_FOR_LLM)
    separation = {
        feature: abs(float(failed[feature].median() - normal[feature].median()))
        for feature in FEATURES_FOR_LLM
    }
    return sorted(separation, key=separation.get, reverse=True)


def _feature_summary(rows: pd.DataFrame, features: Sequence[str]) -> str:
    failed = rows[rows["target"] == 1]
    normal = rows[rows["target"] == 0]
    lines = []
    for feature in features:
        failure_median = failed[feature].median() if not failed.empty else float("nan")
        normal_median = normal[feature].median() if not normal.empty else float("nan")
        lines.append(
            f"- {feature} ({FEATURE_DESCRIPTIONS.get(feature, feature)}): "
            f"failure median={failure_median:.5g}; "
            f"no-failure median={normal_median:.5g}; "
            f"range=[{rows[feature].min():.5g}, {rows[feature].max():.5g}]"
        )
    return "\n".join(lines)


def _seed_rule(rows: pd.DataFrame, features: Sequence[str]) -> str:
    """Create a deterministic one-split baseline for the LLM to improve."""
    failed = rows[rows["target"] == 1]
    normal = rows[rows["target"] == 0]
    if failed.empty:
        return "def rule(x):\n    return 0\n"
    if normal.empty:
        return "def rule(x):\n    return 1\n"

    best_code = "def rule(x):\n    return 0\n"
    best_f2 = -1.0
    for feature in features:
        positive = float(failed[feature].median())
        negative = float(normal[feature].median())
        if not np.isfinite(positive) or not np.isfinite(negative) or positive == negative:
            continue
        threshold = (positive + negative) / 2.0
        operator = ">=" if positive > negative else "<="
        code = (
            "def rule(x):\n"
            f"    if x[{feature!r}] {operator} {threshold!r}:\n"
            "        return 1\n"
            "    else:\n"
            "        return 0\n"
        )
        score = compute_metrics(rows["target"], evaluate_rule(code, rows))["f2"]
        if score > best_f2:
            best_code, best_f2 = code, score
    return best_code


def _wrong_examples(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    features: Sequence[str],
    limit: int,
) -> str:
    wrong = rows.loc[predictions != rows["target"], ["target", *features]].head(limit)
    return "None" if wrong.empty else wrong.to_json(orient="records")


def _repair_rule(client: LLMClient, raw: str, error: Exception, temperature: float) -> str:
    prompt = _fill_template(
        PROMPTS_DIR / "repair_prompt.md",
        {"bad_output": raw, "error_msg": str(error)},
    )
    repaired = _extract_rule(client.complete(prompt, temperature=temperature))
    validate_rule(repaired)
    return repaired


def distill_line(
    rows: pd.DataFrame,
    line_name: str,
    client: LLMClient,
    output_dir: str | Path,
    *,
    ranking: Sequence[str] | None = None,
    config: DistillationConfig = DistillationConfig(),
) -> dict[str, Any]:
    """Generate, evaluate, and save one explanatory contingency rule."""
    features = _feature_order(rows, ranking)[: config.top_features]
    best_rule = _seed_rule(rows, features)
    best_predictions = evaluate_rule(best_rule, rows)
    best_metrics = compute_metrics(rows["target"], best_predictions)
    feedback = "Improve the baseline while keeping the rule compact."

    for _ in range(config.iterations):
        prompt = _fill_template(
            PROMPTS_DIR / "generator_prompt.md",
            {
                "line_name": line_name,
                "current_metrics": json.dumps(best_metrics, sort_keys=True),
                "feature_summary": _feature_summary(rows, features),
                "worst_cases": _wrong_examples(
                    rows, best_predictions, features, config.max_error_examples
                ),
                "feedback": feedback,
                "best_rule": best_rule,
            },
        )
        raw = client.complete(prompt, temperature=config.temperature)
        candidate = _extract_rule(raw)
        try:
            validate_rule(candidate)
        except Exception as exc:
            candidate = _repair_rule(client, raw, exc, config.temperature)

        predictions = evaluate_rule(candidate, rows)
        metrics = compute_metrics(rows["target"], predictions)
        if metrics["f2"] > best_metrics["f2"]:
            best_rule, best_predictions, best_metrics = candidate, predictions, metrics

        critic_prompt = _fill_template(
            PROMPTS_DIR / "critic_prompt.md",
            {
                "current_rule": candidate,
                "current_metrics": json.dumps(metrics, sort_keys=True),
                "feature_summary": _feature_summary(rows, features),
                "worst_cases": _wrong_examples(
                    rows, predictions, features, config.max_error_examples
                ),
                "best_rule": best_rule,
            },
        )
        feedback = client.complete(critic_prompt, temperature=config.temperature)[:3000]

    line_dir = Path(output_dir) / f"line_{line_name}"
    line_dir.mkdir(parents=True, exist_ok=True)
    (line_dir / "best_rule.py").write_text(best_rule, encoding="utf-8")
    result = {"line": line_name, "features": features, **best_metrics}
    (line_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def distill_classifier(
    rows_path: str | Path,
    classifier_path: str | Path,
    metadata_path: str | Path,
    llm_config_path: str | Path,
    output_dir: str | Path,
    *,
    rankings_path: str | Path | None = None,
    config: DistillationConfig = DistillationConfig(),
    shap_settings: ShapSettings = ShapSettings(),
) -> list[dict[str, Any]]:
    """Distil one rule for every line represented by both teacher classes."""
    rankings = {}
    if rankings_path is not None:
        validation = validate_rankings_cache(
            rankings_path,
            rows_path,
            classifier_path,
            metadata_path,
            shap_settings,
        )
        if not validation.compatible:
            raise ValueError(
                f"Incompatible SHAP rankings cache {Path(rankings_path)}: "
                f"{validation.reason}. Recalculate the rankings before distillation."
            )
        rankings = validation.rankings or {}
    dataframe, _, line_map = load_teacher_data(
        rows_path, classifier_path, metadata_path
    )
    client = LLMClient(LLMConfig.from_json(llm_config_path))

    results = []
    for line_name in line_map:
        line_rows = dataframe[dataframe["line_disconnected"] == line_name].copy()
        if line_rows.empty:
            print(f"[skip] {line_name}: no classifier rows")
            continue
        print(f"[distil] {line_name}: {len(line_rows)} rows")
        results.append(
            distill_line(
                line_rows,
                line_name,
                client,
                output_dir,
                ranking=rankings.get(line_name),
                config=config,
            )
        )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distil the current failure classifier into per-line rules."
    )
    parser.add_argument("--rows", type=Path, default=DATA_PATH)
    parser.add_argument("--classifier", type=Path, default=HGB_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=HGB_METADATA_PATH)
    parser.add_argument("--llm-config", type=Path, required=True)
    parser.add_argument("--shap-rankings", type=Path, default=SHAP_RANKINGS_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--iterations", type=int, default=DistillationConfig().iterations
    )
    parser.add_argument("--shap-top-k", type=int, default=ShapSettings().top_k)
    parser.add_argument(
        "--shap-max-rows-per-line",
        type=int,
        default=ShapSettings().max_rows_per_line,
    )
    parser.add_argument("--shap-repeats", type=int, default=ShapSettings().repeats)
    parser.add_argument(
        "--shap-random-seed", type=int, default=ShapSettings().random_seed
    )
    args = parser.parse_args()
    distill_classifier(
        args.rows,
        args.classifier,
        args.metadata,
        args.llm_config,
        args.output_dir,
        rankings_path=args.shap_rankings,
        config=DistillationConfig(iterations=args.iterations),
        shap_settings=ShapSettings(
            top_k=args.shap_top_k,
            max_rows_per_line=args.shap_max_rows_per_line,
            repeats=args.shap_repeats,
            random_seed=args.shap_random_seed,
        ),
    )


if __name__ == "__main__":
    main()
