"""
compute_shap_rankings.py
========================
Per-line feature ranking from the HGB teacher, used by dual_llm.py to focus each
interpretable rule on the most relevant variables for its contingency.

For every disconnected line, features are ranked by mean(|SHAP value|) computed on the HGB
classifier over that line's samples; the top features are written to shap_rankings.json:

    { "34_35_110": ["max_line_rho", "avg_line_rho", ...], ... }

If the SHAP library is unavailable or fails on the HistGradientBoosting model, the script
falls back to scikit-learn permutation importance, which yields a very similar ranking.

Run once before dual_llm.py. No LLM calls.
"""
import os
import json
import warnings

import numpy as np
import joblib

warnings.filterwarnings("ignore")

import dual_llm as Predictor   # reuse the same config, feature list and data loader

TOP_K = 8                 # features kept per line (dual_llm then sweeps k in {3..8} of these)
MAX_ROWS_PER_LINE = 200   # subsample per line for SHAP speed (the ranking is stable)


def feature_importance(model, X):
    """mean(|SHAP|) per feature; permutation-importance fallback for HGB."""
    try:
        import shap
        background = X.sample(min(100, len(X)), random_state=0) if len(X) > 100 else X
        explainer = shap.Explainer(model.predict_proba, background)
        try:
            values = explainer(X, silent=True).values
        except TypeError:
            values = explainer(X).values
        values = np.array(values)
        if values.ndim == 3:                 # (n, n_features, n_classes) -> positive class
            values = values[:, :, -1]
        return np.abs(values).mean(axis=0), "shap"
    except Exception as e:
        from sklearn.inspection import permutation_importance
        y = model.predict(X)
        r = permutation_importance(model, X, y, n_repeats=5, random_state=0)
        return r.importances_mean, f"permutation_fallback({type(e).__name__})"


def main():
    model = joblib.load(Predictor.HGB_MODEL_PATH)
    df = Predictor.prepare_dataframe()

    rankings = {}
    for line_name in Predictor.LINE_MAP:
        df_line = df[df["line_disconnected"] == line_name]
        if len(df_line) < 5:
            print(f"  [skip] {line_name}: too few rows ({len(df_line)})")
            continue
        X = df_line[Predictor.FEATURES].copy()
        if len(X) > MAX_ROWS_PER_LINE:
            X = X.sample(MAX_ROWS_PER_LINE, random_state=0)
        importance, method = feature_importance(model, X)
        order = np.argsort(importance)[::-1]
        ranked = [Predictor.FEATURES[i] for i in order if Predictor.FEATURES[i] in Predictor.FEATURES_FOR_LLM][:TOP_K]
        rankings[line_name] = ranked
        print(f"  {line_name} [{method}]: {ranked}")

    out = os.path.join(Predictor.HERE, "shap_rankings.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
