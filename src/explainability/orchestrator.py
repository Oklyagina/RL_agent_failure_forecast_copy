"""Thin bridge between the authoritative HGB predictor and optional rules."""
from __future__ import annotations

import logging
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


LOGGER = logging.getLogger(__name__)


class ExplainabilityOrchestrator:
    """Attach surrogate explanations without changing HGB results."""

    def __init__(
        self,
        predictor: FailureForecastPredictor,
        rules_dir: str | Path | None = None,
    ):
        self.predictor = predictor
        self.rule_explainer: RuleExplainer | None = None
        if rules_dir:
            try:
                self.rule_explainer = RuleExplainer(rules_dir)
            except (OSError, UnicodeError, ValueError, SyntaxError) as exc:
                LOGGER.warning("Explanatory rules are unavailable: %s", exc)

    @staticmethod
    def _unavailable(output: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return {
            **output,
            "explanation": None,
            "explanation_method": None,
            "explanation_status": "unavailable",
            "explanation_error": reason,
            "surrogate_prediction": None,
            "surrogate_agrees": None,
        }

    def explain_from_features(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Score a row first, then best-effort attach its optional explanation."""
        result = dict(self.predictor.predict_from_features(row))
        line = normalize_line_name(row.get("line_disconnected", ""))
        output = {"line": line, **result}
        if self.rule_explainer is None:
            return self._unavailable(output, "no explanatory rules were loaded")
        if line not in self.rule_explainer.available_lines:
            return self._unavailable(output, f"no explanatory rule for line {line}")

        try:
            explanation = self.rule_explainer.explain(line, row)
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            LOGGER.warning("Explanation failed for line %s: %s", line, exc)
            return self._unavailable(output, str(exc))
        return {
            **output,
            "explanation": explanation.text,
            "explanation_method": "llm_distilled_rule",
            "explanation_status": "available",
            "explanation_error": None,
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
        """Build the common forecast once, then score/explain every contingency."""
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
        """Return static descriptions for all successfully loaded rules."""
        if self.rule_explainer is None:
            return {}
        return self.rule_explainer.all_descriptions()
