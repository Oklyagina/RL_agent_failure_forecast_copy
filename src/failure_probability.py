"""Failure-probability inference for a Grid2Op observation + line contingency.

``FailureProbabilityPredictor.predict_proba(obs, line)`` is the minimal API.
The object owns the trained classifier and optional forecast/ENN dependencies,
so callers only provide the live observation and the line to disconnect.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    from .train_classifier import CLASSIFIER_FEATURES, LINE_MAP, encode_line
    from .utils import compute_grid_stats
except ImportError:  # pragma: no cover
    from train_classifier import CLASSIFIER_FEATURES, LINE_MAP, encode_line
    from utils import compute_grid_stats


class FailureProbabilityPredictor:
    """Predict the probability that the configured agent fails after a line contingency.

    The classifier is the model trained by ``src/train_classifier.py``.  When
    mean/aleatoric forecasters and an ENN are supplied, the predictor also
    computes the same t+12 features used during training.  Without them it still
    produces a probability using current-state features and NaNs for unavailable
    forecast features (HistGradientBoosting handles missing values natively).
    """

    def __init__(
        self,
        classifier: Any,
        *,
        metadata: Optional[dict] = None,
        model_predict: Any = None,
        model_aleatoric: Any = None,
        model_enn: Any = None,
        cfg: Any = None,
        get_uncertainty_fn: Optional[Callable] = None,
        get_features_with_history_fn: Optional[Callable] = None,
        compute_grid_stats_fn: Callable = compute_grid_stats,
        observations_array: Optional[List[Any]] = None,
    ):
        if not hasattr(classifier, "predict_proba"):
            raise TypeError("classifier must expose predict_proba(...).")
        self.classifier = classifier
        self.metadata = metadata or {}
        self.features = list(self.metadata.get("features", CLASSIFIER_FEATURES))
        if self.features != CLASSIFIER_FEATURES:
            # The builder currently guarantees the canonical training feature set.
            raise ValueError(
                "Classifier metadata feature order does not match CLASSIFIER_FEATURES. "
                "Retrain/export the classifier with this repository version."
            )
        self.line_map = {
            str(k).replace("line_", ""): int(v)
            for k, v in self.metadata.get("line_map", LINE_MAP).items()
        }
        self.threshold = float(self.metadata.get("decision_threshold", 0.5))
        self.failure_class = int(self.metadata.get("failure_class", 1))
        self.model_predict = model_predict
        self.model_aleatoric = model_aleatoric
        self.model_enn = model_enn
        self.cfg = cfg
        self._get_uncertainty = get_uncertainty_fn
        self._get_features_with_history = get_features_with_history_fn
        self._compute_grid_stats = compute_grid_stats_fn
        self.observations_array = observations_array if observations_array is not None else []

    @classmethod
    def from_artifacts(
        cls,
        classifier_path: str | Path,
        *,
        metadata_path: Optional[str | Path] = None,
        **kwargs,
    ) -> "FailureProbabilityPredictor":
        classifier_path = Path(classifier_path)
        classifier = joblib.load(classifier_path)
        if metadata_path is None:
            metadata_path = classifier_path.with_name("classifier_metadata.json")
        metadata = {}
        metadata_path = Path(metadata_path)
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(classifier, metadata=metadata, **kwargs)

    @classmethod
    def from_pipeline(cls, cfg: Any = None, *, observations_array: Optional[List[Any]] = None):
        """Load the complete trained pipeline and return a ready predictor.

        After the standard training pipeline has produced its artifacts, the
        call site is simply ``predictor.predict_proba(obs, line)``.
        """
        if cfg is None:
            try:
                from .config import CFG as cfg
            except ImportError:  # pragma: no cover
                from config import CFG as cfg
        try:
            from .training_enn import get_uncertainty, load_trained_enn
            from .collect_data import get_features_with_history
        except ImportError:  # pragma: no cover
            from training_enn import get_uncertainty, load_trained_enn
            from collect_data import get_features_with_history

        classifier_path = Path(cfg.MODEL_CLASSIFIER_PATH)
        return cls.from_artifacts(
            classifier_path,
            model_predict=joblib.load(cfg.MODEL_MEAN_PATH),
            model_aleatoric=joblib.load(cfg.MODEL_ALEATORIC_PATH),
            model_enn=load_trained_enn(),
            cfg=cfg,
            get_uncertainty_fn=get_uncertainty,
            get_features_with_history_fn=get_features_with_history,
            observations_array=observations_array,
        )

    def _remember_observation(self, obs: Any) -> None:
        """Append a live state once, enabling history-aware forecasts across calls."""
        if not self.observations_array:
            self.observations_array.append(obs)

    def _sync_cfg_dimensions(self, obs: Any) -> None:
        """Synchronize legacy CFG shape fields from the live observation.

        The forecast helper still uses ``CFG.NO_LOADS`` / ``CFG.NO_GENS`` for
        slicing model outputs.  Updating those values here keeps standalone
        inference generic even when it is not launched through ``run_pipeline``.
        """
        if self.cfg is None:
            return
        n_load = int(len(obs.load_p))
        n_gen = int(len(obs.gen_p))
        self.cfg.NO_LOADS = n_load
        self.cfg.NO_GENS = n_gen
        if hasattr(obs, "rho"):
            self.cfg.NO_LINES = int(len(obs.rho))
        if self.model_enn is not None:
            self.cfg.ENN_INPUT_DIM = int(
                getattr(self.model_enn, "input_dim", len(obs.to_vect()))
            )

        obs_env = getattr(obs, "_obs_env", None)
        for attr, fallback in (("GEN_MAX", np.inf), ("GEN_MIN", -np.inf)):
            source_name = "gen_pmax" if attr == "GEN_MAX" else "gen_pmin"
            candidate = getattr(obs, source_name, None)
            if candidate is None and obs_env is not None:
                candidate = getattr(obs_env, source_name, None)
            arr = None if candidate is None else np.asarray(candidate, dtype=float).reshape(-1)
            current = np.asarray(getattr(self.cfg, attr, []), dtype=float).reshape(-1)
            if arr is not None and len(arr) == n_gen:
                setattr(self.cfg, attr, arr.copy())
            elif len(current) != n_gen:
                setattr(self.cfg, attr, np.full(n_gen, fallback, dtype=float))
            return
        last = self.observations_array[-1]
        if last is obs:
            return
        current_step = getattr(obs, "current_step", None)
        last_step = getattr(last, "current_step", None)
        if current_step is not None and last_step == current_step:
            self.observations_array[-1] = obs
        else:
            self.observations_array.append(obs)

    def _line_name(self, obs: Any, line: Any) -> str:
        if isinstance(line, (int, np.integer)):
            numeric_name = str(int(line))
            # Data collection can be configured with integer line ids. In that
            # case classifier metadata stores the numeric category verbatim.
            if numeric_name in self.line_map:
                return numeric_name
            names = getattr(obs, "name_line", None)
            if names is None:
                obs_env = getattr(obs, "_obs_env", None)
                names = getattr(obs_env, "name_line", None)
            if names is not None and 0 <= int(line) < len(names):
                return str(names[int(line)]).replace("line_", "")
        return str(line).replace("line_", "")

    def _base_features(self, obs: Any, line: Any) -> Dict[str, float]:
        gs = self._compute_grid_stats(obs)
        sum_load_p = float(gs.get("sum_load_p", np.sum(getattr(obs, "load_p", np.nan))))
        sum_gen_p = float(gs.get("sum_gen_p", np.sum(getattr(obs, "gen_p", np.nan))))
        line_name = self._line_name(obs, line)
        encoded_line = encode_line(line_name, self.line_map)
        if encoded_line < 0:
            supported = ", ".join(sorted(self.line_map)) or "<none>"
            raise ValueError(
                f"Line {line_name!r} was not present in the classifier training data. "
                f"Supported lines: {supported}. Retrain the classifier with examples for "
                "this contingency before requesting a probability."
            )
        features = {
            "line_id_encoded": float(encoded_line),
            "sum_load_p": sum_load_p,
            "sum_load_q": float(gs.get("sum_load_q", np.nan)),
            "sum_gen_p": sum_gen_p,
            "var_line_rho": float(gs.get("var_line_rho", np.nan)),
            "avg_line_rho": float(gs.get("avg_line_rho", np.nan)),
            "max_line_rho": float(gs.get("max_line_rho", np.nan)),
            "nb_rho_ge_0.95": float(gs.get("nb_rho_ge_0.95", np.nan)),
            "load_gen_ratio": sum_load_p / (sum_gen_p + 1e-6),
            "fcast_sum_load_p": np.nan,
            "fcast_sum_load_q": np.nan,
            "fcast_sum_gen_p": np.nan,
            "fcast_var_line_rho": np.nan,
            "fcast_avg_line_rho": np.nan,
            "fcast_max_line_rho": np.nan,
            "fcast_nb_rho_ge_0.95": np.nan,
            "aleatoric_load_p_mean": np.nan,
            "aleatoric_load_q_mean": np.nan,
            "aleatoric_gen_p_mean": np.nan,
            "epistemic_before": np.nan,
            "epistemic_after": np.nan,
        }
        return features

    @property
    def _can_forecast(self) -> bool:
        return all([
            self.model_predict is not None,
            self.model_aleatoric is not None,
            self.model_enn is not None,
            self.cfg is not None,
            self._get_uncertainty is not None,
            self._get_features_with_history is not None,
        ])

    def build_features(self, obs: Any, line: Any) -> Dict[str, float]:
        self._remember_observation(obs)
        self._sync_cfg_dimensions(obs)
        features = self._base_features(obs, line)

        if self.model_enn is not None and self._get_uncertainty is not None:
            try:
                input_dim = int(getattr(self.model_enn, "input_dim", len(obs.to_vect())))
                vector = np.asarray(obs.to_vect(), dtype=np.float32)[:input_dim].reshape(1, -1)
                features["epistemic_before"] = float(
                    self._get_uncertainty(self.model_enn, vector)
                )
            except Exception:
                pass

        if self._can_forecast:
            try:
                try:
                    from .rule_predictor import _run_forecast
                except ImportError:  # pragma: no cover
                    from rule_predictor import _run_forecast
                fc = _run_forecast(
                    obs=obs,
                    observations_array=self.observations_array,
                    model_predict=self.model_predict,
                    model_aleatoric=self.model_aleatoric,
                    model_enn=self.model_enn,
                    cfg=self.cfg,
                    get_features_with_history_fn=self._get_features_with_history,
                    get_uncertainty_fn=self._get_uncertainty,
                    compute_grid_stats_fn=self._compute_grid_stats,
                )
                fgs = fc.get("fcast_grid_stats", {})
                features.update({
                    "epistemic_before": float(fc.get("epistemic_before", features["epistemic_before"])),
                    "epistemic_after": float(fc.get("epistemic_after", np.nan)),
                    "aleatoric_load_p_mean": float(fc.get("aleatoric_load_p_mean", np.nan)),
                    "aleatoric_load_q_mean": float(fc.get("aleatoric_load_q_mean", np.nan)),
                    "aleatoric_gen_p_mean": float(fc.get("aleatoric_gen_p_mean", np.nan)),
                    "fcast_sum_load_p": float(fgs.get("sum_load_p", np.nan)),
                    "fcast_sum_load_q": float(fgs.get("sum_load_q", np.nan)),
                    "fcast_sum_gen_p": float(fgs.get("sum_gen_p", np.nan)),
                    "fcast_var_line_rho": float(fgs.get("var_line_rho", np.nan)),
                    "fcast_avg_line_rho": float(fgs.get("avg_line_rho", np.nan)),
                    "fcast_max_line_rho": float(fgs.get("max_line_rho", np.nan)),
                    "fcast_nb_rho_ge_0.95": float(fgs.get("nb_rho_ge_0.95", np.nan)),
                })
            except Exception:
                # The classifier can still infer with missing forecast features.
                pass
        return features

    def predict_proba(self, obs: Any, line: Any) -> float:
        """Return P(failure=1) in [0, 1] for ``obs`` and ``line``."""
        features = self.build_features(obs, line)
        frame = pd.DataFrame([{name: features[name] for name in self.features}], columns=self.features)
        frame["line_id_encoded"] = frame["line_id_encoded"].astype(int)
        probabilities = np.asarray(self.classifier.predict_proba(frame), dtype=float)[0]
        classes = np.asarray(getattr(self.classifier, "classes_", [0, 1]))
        matches = np.where(classes == self.failure_class)[0]
        if len(matches) != 1:
            raise ValueError(
                f"Classifier classes {classes.tolist()} do not contain failure class "
                f"{self.failure_class} exactly once."
            )
        probability = float(probabilities[int(matches[0])])
        if not np.isfinite(probability):
            raise ValueError("Classifier returned a non-finite failure probability.")
        return float(np.clip(probability, 0.0, 1.0))

    def predict(self, obs: Any, line: Any) -> Dict[str, Any]:
        probability = self.predict_proba(obs, line)
        return {
            "line": self._line_name(obs, line),
            "failure_probability": probability,
            "failure_probability_pct": round(100.0 * probability, 2),
            "threshold": self.threshold,
            "predicted_failure": bool(probability >= self.threshold),
        }
