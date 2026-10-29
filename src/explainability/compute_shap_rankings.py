"""Rank current HGB features per contingency for the LLM rule generator."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.explainability import dual_llm
from src.explainability.shap_cache import (
    ShapSettings,
    build_provenance,
    write_rankings_cache,
)


DEFAULT_SHAP_SETTINGS = ShapSettings()
TOP_K = DEFAULT_SHAP_SETTINGS.top_k
MAX_ROWS_PER_LINE = DEFAULT_SHAP_SETTINGS.max_rows_per_line
REPEATS = DEFAULT_SHAP_SETTINGS.repeats
RANDOM_SEED = DEFAULT_SHAP_SETTINGS.random_seed


def feature_importance(
    model: Any,
    rows: Any,
    *,
    repeats: int = REPEATS,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, str]:
    """Return mean absolute SHAP values, with permutation importance fallback."""
    try:
        import shap

        background = (
            rows.sample(100, random_state=random_seed) if len(rows) > 100 else rows
        )
        explainer = shap.Explainer(model.predict_proba, background)
        try:
            values = explainer(rows, silent=True).values
        except TypeError:
            values = explainer(rows).values
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        return np.abs(values).mean(axis=0), "shap"
    except Exception as exc:
        from sklearn.inspection import permutation_importance

        labels = model.predict(rows)
        result = permutation_importance(
            model, rows, labels, n_repeats=repeats, random_state=random_seed
        )
        return result.importances_mean, f"permutation_fallback({type(exc).__name__})"


def compute_rankings(
    rows_path: str | Path,
    classifier_path: str | Path,
    metadata_path: str | Path,
    *,
    top_k: int = TOP_K,
    max_rows_per_line: int = MAX_ROWS_PER_LINE,
    repeats: int = REPEATS,
    random_seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """Compute feature rankings using the current classifier feature contract."""
    dataframe, model, line_map = dual_llm.load_teacher_data(
        rows_path, classifier_path, metadata_path
    )
    rankings: dict[str, list[str]] = {}
    progress = tqdm(
        line_map, desc="SHAP contingencies", unit="line", dynamic_ncols=True
    )
    for line_name in progress:
        line_rows = dataframe[dataframe["line_disconnected"] == line_name]
        progress.set_postfix(line=line_name, rows=len(line_rows))
        if len(line_rows) < 5:
            tqdm.write(f"  [skip] {line_name}: too few rows ({len(line_rows)})")
            continue
        features = line_rows[dual_llm.FEATURES].copy()
        if len(features) > max_rows_per_line:
            features = features.sample(
                max_rows_per_line, random_state=random_seed
            )
        importance, method = feature_importance(
            model,
            features,
            repeats=repeats,
            random_seed=random_seed,
        )
        order = np.argsort(importance)[::-1]
        ranked = [
            dual_llm.FEATURES[index]
            for index in order
            if dual_llm.FEATURES[index] in dual_llm.FEATURES_FOR_LLM
        ][:top_k]
        rankings[line_name] = ranked
        tqdm.write(f"  {line_name} [{method}]: {ranked}")
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=dual_llm.DATA_PATH)
    parser.add_argument("--classifier", type=Path, default=dual_llm.HGB_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=dual_llm.HGB_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=dual_llm.SHAP_RANKINGS_FILE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-rows-per-line", type=int, default=MAX_ROWS_PER_LINE)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    settings = ShapSettings(
        top_k=args.top_k,
        max_rows_per_line=args.max_rows_per_line,
        repeats=args.repeats,
        random_seed=args.random_seed,
    )
    provenance = build_provenance(
        args.rows, args.classifier, args.metadata, settings
    )
    rankings = compute_rankings(
        args.rows,
        args.classifier,
        args.metadata,
        top_k=settings.top_k,
        max_rows_per_line=settings.max_rows_per_line,
        repeats=settings.repeats,
        random_seed=settings.random_seed,
    )
    write_rankings_cache(
        args.output,
        rankings,
        args.rows,
        args.classifier,
        args.metadata,
        settings,
        expected_provenance=provenance,
    )
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
