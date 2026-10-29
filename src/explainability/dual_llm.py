"""Distil the active HGB failure classifier into readable per-line rules.

The HGB remains authoritative. This optional offline stage builds compact
surrogate rules using a deterministic seed followed by generator/critic LLM
calls. Rule evaluation is vectorized because it is also the main inner loop of
the seed search.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from numpy import ndarray, dtype, bool_
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

from project_config import AGENT_NAME, ARTIFACTS_DIR, ENV_NAME
from src.explainability.llm_client import (
    LLMClient,
    LLMConfig,
    LLMUnavailableError,
)
from src.explainability.rule_predictor import compile_rule, validate_rule
from src.explainability.shap_cache import ShapSettings, validate_rankings_cache
from src.failure_forecast import (
    CLASSIFIER_FEATURES,
    normalize_line_name,
    prepare_classifier_dataframe,
)


LOGGER = logging.getLogger(__name__)
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
    """Controls that affect one rule-distillation run."""

    iterations: int = 20
    temperature: float = 0.7
    top_features: int = 8
    max_error_examples: int = 8

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.top_features < 1 or self.max_error_examples < 1:
            raise ValueError("iterations cannot be negative; feature/error limits must be positive")


@dataclass(frozen=True)
class _SeedResult:
    """Seed rule plus its already-computed predictions and metrics."""

    code: str
    predictions: np.ndarray
    metrics: dict[str, float]


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
    """Return fidelity metrics, emphasizing failure recall through F2."""
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "f2": float(fbeta_score(actual, predicted, beta=2, zero_division=0)),
        "false_alarm_rate": float(fp / (fp + tn + 1e-12)),
        "missed_failure_rate": float(fn / (fn + tp + 1e-12)),
    }


def _comparison(node: ast.Compare, rows: pd.DataFrame) -> bool | Any:
    feature = node.left.slice.value
    threshold_node = node.comparators[0]
    threshold = (
        -float(threshold_node.operand.value)
        if isinstance(threshold_node, ast.UnaryOp)
        else float(threshold_node.value)
    )
    values = rows[feature].to_numpy(copy=False)
    operator = node.ops[0]
    if isinstance(operator, ast.Gt):
        return values > threshold
    if isinstance(operator, ast.GtE):
        return values >= threshold
    if isinstance(operator, ast.Lt):
        return values < threshold
    if isinstance(operator, ast.LtE):
        return values <= threshold
    if isinstance(operator, ast.Eq):
        return values == threshold
    if isinstance(operator, ast.NotEq):
        return values != threshold
    raise ValueError(f"Unsupported comparison operator: {type(operator).__name__}")


def _condition(node: ast.expr, rows: pd.DataFrame) -> np.ndarray:
    if isinstance(node, ast.Compare):
        return np.asarray(_comparison(node, rows), dtype=bool)
    if isinstance(node, ast.BoolOp):
        values = [_condition(value, rows) for value in node.values]
        if isinstance(node.op, ast.And):
            return np.logical_and.reduce(values)
        return np.logical_or.reduce(values)
    raise ValueError(f"Unsupported condition: {type(node).__name__}")


def evaluate_rule(rule_code: str, rows: pd.DataFrame) -> np.ndarray:
    """Evaluate a validated rule with NumPy masks, without Python row loops."""
    function = validate_rule(rule_code)
    result = np.empty(len(rows), dtype=int)

    def run(statements: Sequence[ast.stmt], active: np.ndarray) -> np.ndarray:
        remaining = active
        for statement in statements:
            if not remaining.any():
                break
            if isinstance(statement, ast.Return):
                value_node = statement.value
                value = (
                    -int(value_node.operand.value)
                    if isinstance(value_node, ast.UnaryOp)
                    else int(value_node.value)
                )
                result[remaining] = value
                return np.zeros(len(rows), dtype=bool)
            if isinstance(statement, ast.If):
                condition = _condition(statement.test, rows)
                after_true = run(statement.body, remaining & condition)
                after_false = run(statement.orelse, remaining & ~condition)
                remaining = after_true | after_false
        return remaining

    unfinished = run(function.body, np.ones(len(rows), dtype=bool))
    if unfinished.any():
        raise ValueError("Rule does not return 0 or 1 for every input")
    return result


def _evaluate_rule_scalar(rule_code: str, rows: pd.DataFrame) -> np.ndarray:
    """Reference evaluator retained for equivalence tests and benchmarks."""
    rule = compile_rule(rule_code)
    return np.asarray([int(rule(row)) for _, row in rows.iterrows()], dtype=int)


@lru_cache(maxsize=3)
def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fill_template(path: Path, values: Mapping[str, Any]) -> str:
    text = _load_template(path)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text.strip()


def _extract_rule(text: str) -> str:
    tagged = re.search(r"\[Start of Rule\](.*?)\[End of Rule\]", text, re.I | re.S)
    if tagged:
        text = tagged.group(1)
    for block in re.findall(r"```(?:python)?\s*(.*?)```", text, re.I | re.S):
        if "def rule" in block:
            return block.strip()
    match = re.search(r"(def\s+rule\s*\(\s*\w+\s*\)\s*:[\s\S]*)", text)
    return match.group(1).strip() if match else ""


def _feature_order(rows: pd.DataFrame, ranking: Sequence[str] | None) -> list[str]:
    available = [feature for feature in FEATURES_FOR_LLM if feature in rows.columns]
    if ranking:
        selected = [feature for feature in ranking if feature in available]
        if selected:
            return selected
    failed = rows[rows["target"] == 1]
    normal = rows[rows["target"] == 0]
    if failed.empty or normal.empty:
        return available
    separation = {
        feature: abs(float(failed[feature].median() - normal[feature].median()))
        for feature in available
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


def _constant_rule(value: int, rows: pd.DataFrame) -> _SeedResult:
    code = f"def rule(x):\n    return {value}\n"
    predictions = np.full(len(rows), value, dtype=int)
    return _SeedResult(code, predictions, compute_metrics(rows["target"], predictions))


def _seed_rule(rows: pd.DataFrame, features: Sequence[str]) -> _SeedResult:
    """Select the best one-split seed using vectorized candidate evaluation.

    The returned predictions and metrics are reused by :func:`distill_line`, so
    the selected candidate is not evaluated a second time.
    """
    failed = rows[rows["target"] == 1]
    normal = rows[rows["target"] == 0]
    if failed.empty:
        return _constant_rule(0, rows)
    if normal.empty:
        return _constant_rule(1, rows)

    best = _constant_rule(0, rows)
    best_f2 = -1.0
    targets = rows["target"].to_numpy(dtype=int, copy=False)
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
        values = rows[feature].to_numpy(copy=False)
        predictions = np.asarray(
            values >= threshold if positive > negative else values <= threshold,
            dtype=int,
        )
        metrics = compute_metrics(targets, predictions)
        if metrics["f2"] > best_f2:
            best = _SeedResult(code, predictions, metrics)
            best_f2 = metrics["f2"]
    return best


def _wrong_examples(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    features: Sequence[str],
    limit: int,
) -> str:
    wrong = rows.loc[predictions != rows["target"], ["target", *features]].head(limit)
    return "None" if wrong.empty else wrong.to_json(orient="records")


def _progress(step: int, total: int, message: str) -> None:
    width = 12
    filled = min(width, round(width * step / max(total, 1)))
    print(f"[explainability] [{'#' * filled}{'-' * (width - filled)}] {message}")


def _repair_rule(client: LLMClient, raw: str, error: Exception, temperature: float) -> str:
    prompt = _fill_template(
        PROMPTS_DIR / "repair_prompt.md",
        {"bad_output": raw, "error_msg": str(error)},
    )
    return _extract_rule(client.complete(prompt, temperature=temperature))


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
    _progress(1, 3, f"{line_name}: evaluating deterministic seed candidates")
    seed = _seed_rule(rows, features)
    best_rule = seed.code
    best_predictions = seed.predictions
    best_metrics = seed.metrics
    feedback = "Improve the baseline while keeping the rule compact."
    feature_summary = _feature_summary(rows, features)

    for iteration in range(1, config.iterations + 1):
        prompt = _fill_template(
            PROMPTS_DIR / "generator_prompt.md",
            {
                "line_name": line_name,
                "current_metrics": json.dumps(best_metrics, sort_keys=True),
                "feature_summary": feature_summary,
                "worst_cases": _wrong_examples(
                    rows, best_predictions, features, config.max_error_examples
                ),
                "feedback": feedback,
                "best_rule": best_rule,
            },
        )
        _progress(2, 3, f"{line_name} iteration {iteration}/{config.iterations}: requesting a candidate rule; waiting")
        try:
            raw = client.complete(prompt, temperature=config.temperature)
        except LLMUnavailableError as exc:
            LOGGER.warning("LLM generator unavailable for %s: %s", line_name, exc)
            break
        candidate = _extract_rule(raw)
        try:
            try:
                predictions = evaluate_rule(candidate, rows)
            except (ValueError, TypeError, KeyError) as exc:
                candidate = _repair_rule(client, raw, exc, config.temperature)
                predictions = evaluate_rule(candidate, rows)
        except (LLMUnavailableError, ValueError, TypeError, KeyError) as exc:
            LOGGER.warning("Skipping invalid LLM rule for %s: %s", line_name, exc)
            feedback = f"Previous candidate failed validation or evaluation: {exc}"
            continue

        metrics = compute_metrics(rows["target"], predictions)
        if metrics["f2"] > best_metrics["f2"]:
            best_rule, best_predictions, best_metrics = candidate, predictions, metrics

        critic_prompt = _fill_template(
            PROMPTS_DIR / "critic_prompt.md",
            {
                "current_rule": candidate,
                "current_metrics": json.dumps(metrics, sort_keys=True),
                "feature_summary": feature_summary,
                "worst_cases": _wrong_examples(
                    rows, predictions, features, config.max_error_examples
                ),
                "best_rule": best_rule,
            },
        )
        _progress(2, 3, f"{line_name} iteration {iteration}/{config.iterations}: Requesting rule feedback; waiting")
        try:
            feedback = client.complete(
                critic_prompt, temperature=config.temperature
            )[:3000]
        except LLMUnavailableError as exc:
            LOGGER.warning("LLM critic unavailable for %s: %s", line_name, exc)
            feedback = "Critic unavailable; keep the best valid rule compact."

    line_dir = Path(output_dir) / f"line_{line_name}"
    line_dir.mkdir(parents=True, exist_ok=True)
    (line_dir / "best_rule.py").write_text(best_rule, encoding="utf-8")
    result = {"line": line_name, "features": features, **best_metrics}
    (line_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _progress(3, 3, f"{line_name}: explanatory rule saved")
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
    """Distil one best-effort rule for each configured contingency."""
    rankings: dict[str, list[str]] = {}
    if rankings_path is not None:
        validation = validate_rankings_cache(
            rankings_path, rows_path, classifier_path, metadata_path, shap_settings
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

    results: list[dict[str, Any]] = []
    for line_name in line_map:
        line_rows = dataframe[dataframe["line_disconnected"] == line_name].copy()
        if line_rows.empty:
            LOGGER.warning("No classifier rows for explainability line %s", line_name)
            continue
        print(f"[explainability] {line_name}: {len(line_rows)} rows")
        try:
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
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            LOGGER.warning("Could not distil rule for %s: %s", line_name, exc)

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
    parser.add_argument("--iterations", type=int, default=DistillationConfig().iterations)
    args = parser.parse_args()
    distill_classifier(
        args.rows,
        args.classifier,
        args.metadata,
        args.llm_config,
        args.output_dir,
        rankings_path=args.shap_rankings,
        config=DistillationConfig(iterations=args.iterations),
    )


if __name__ == "__main__":
    main()
