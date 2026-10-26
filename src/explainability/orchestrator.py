"""Thin bridge between the active failure forecast and explanatory rules."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from src.failure_forecast import (
    FailureForecastConfig,
    FailureForecastPredictor,
    build_feature_row,
    eligible_lines,
    forecast_injections,
    normalize_line_name,
    simulate_forecast_state,
)

from src.explainability.rule_predictor import RuleExplainer


class ExplainabilityOrchestrator:
    """Coordinate current HGB predictions with optional surrogate explanations.

    The class calls the existing forecast and feature-building functions; it
    does not reimplement them. The HGB output is always returned unchanged.
    """

    def __init__(
        self,
        predictor: FailureForecastPredictor,
        rules_dir: str | Path | None = None,
    ):
        self.predictor = predictor
        self.rule_explainer = RuleExplainer(rules_dir) if rules_dir else None

    def explain_from_features(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Score an existing feature row and attach a surrogate explanation."""
        result = dict(self.predictor.predict_from_features(row))
        line = normalize_line_name(row.get("line_disconnected", ""))
        output = {"line": line, **result}
        if self.rule_explainer is None or line not in self.rule_explainer.available_lines:
            return {
                **output,
                "explanation": None,
                "explanation_method": None,
                "surrogate_prediction": None,
                "surrogate_agrees": None,
            }

        explanation = self.rule_explainer.explain(line, row)
        return {
            **output,
            "explanation": explanation.text,
            "explanation_method": "llm_distilled_rule",
            "surrogate_prediction": explanation.prediction,
            "surrogate_agrees": explanation.prediction == result["failure_prediction"],
        }

    def predict_and_explain_many(
        self,
        obs: Any,
        observations: Sequence[Any],
        lines: Sequence[Any],
        mean_model: Any,
        aleatoric_model: Any,
        cfg: FailureForecastConfig,
        *,
        env: Any,
        model_enn: Any = None,
        get_uncertainty_fn: Optional[Callable[[Any, np.ndarray], float]] = None,
    ) -> list[dict[str, Any]]:
        """Build current feature rows once, then predict and explain each line."""
        if not observations or observations[-1] is not obs:
            raise ValueError("observations must end with the current observation")

        forecast = forecast_injections(
            observations,
            mean_model,
            aleatoric_model,
            cfg,
            model_enn=model_enn,
            get_uncertainty_fn=get_uncertainty_fn,
        )
        forecast.update(
            simulate_forecast_state(
                obs,
                forecast,
                env,
                model_enn=model_enn,
                get_uncertainty_fn=get_uncertainty_fn,
            )
        )
        return [
            self.explain_from_features(build_feature_row(obs, line, forecast, cfg))
            for line in eligible_lines(cfg, obs, lines)
        ]

    def model_descriptions(self) -> dict[str, str]:
        """Return the human-readable description of every loaded surrogate rule."""
        if self.rule_explainer is None:
            return {}
        return self.rule_explainer.all_descriptions()
