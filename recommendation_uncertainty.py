"""Epistemic-uncertainty KPI for Grid2Op policy recommendations.

The ENN produces a Dirichlet distribution over policy actions.  Its vacuity
``u = K / sum(alpha)`` is a bounded epistemic-uncertainty signal in [0, 1].
This module exposes both:

* ``epistemic_uncertainty_pct``: direct vacuity as a percentage (0-100);
* ``epistemic_uncertainty_total_pctile``: calibrated rank versus ENN training
  states (0-100), retained for backwards compatibility;
* categorical uncertainty/confidence bands derived from that calibrated rank.

The categorical confidence is intentionally called a *level/band*, not a
statistical confidence interval: high/medium/low are qualitative categories,
not coverage intervals around an estimator.

Calibration files are produced by the active refactored trainer
(`training/train_enn.py`) and loaded from the selected artifact bundle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


# =========================================================================== #
# Calibration
# =========================================================================== #

@torch.no_grad()
def _state_measures(enn, x_scaled, device, label=None):
    out = enn(torch.as_tensor(np.asarray(x_scaled, np.float32), device=device).unsqueeze(0))
    prob = out["prob"][0].detach().cpu().numpy()
    S = float(out["S"][0].item())
    u = float(out["uncertainty"][0].item())
    a = int(np.argmax(prob)) if label is None else int(label)
    if a < 0 or a >= len(prob):
        raise IndexError(f"ENN class label {a} is outside [0, {len(prob) - 1}].")
    p_a = float(prob[a])
    return u, p_a * (1.0 - p_a) / (S + 1.0), int(np.argmax(prob))


def build_calibration(enn, input_vectors_scaled):
    """Build reference distributions from scaled ENN-training/validation states."""
    enn.eval()
    device = next(enn.parameters()).device
    X = np.asarray(input_vectors_scaled, np.float32)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"Calibration input must be a non-empty 2-D array, got {X.shape}.")
    total_ref, action_ref = [], []
    for row in X:
        u, var_a, _ = _state_measures(enn, row, device)
        total_ref.append(u)
        action_ref.append(var_a)
    return np.sort(np.asarray(total_ref, float)), np.sort(np.asarray(action_ref, float))


def save_calibration(path, total_ref, action_ref):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, total_ref=np.asarray(total_ref, float), action_ref=np.asarray(action_ref, float))


class Calibration:
    def __init__(self, total_ref, action_ref, scaler=None, action_set=None, class_mapping=None):
        self.total_ref = np.sort(np.asarray(total_ref, float).reshape(-1))
        self.action_ref = np.sort(np.asarray(action_ref, float).reshape(-1))
        if len(self.total_ref) == 0 or len(self.action_ref) == 0:
            raise ValueError("Calibration reference arrays must be non-empty.")
        self.scaler = scaler
        self.action_set = None if action_set is None else np.asarray(action_set, dtype=float)
        if self.action_set is not None and self.action_set.ndim != 2:
            raise ValueError(f"action_set must be a 2-D array, got {self.action_set.shape}.")
        self.class_mapping = {str(k): int(v) for k, v in (class_mapping or {}).items()}


def load_calibration(calib_path, scaler=None, action_set=None, class_mapping=None):
    with np.load(calib_path, allow_pickle=False) as data:
        total_ref = data["total_ref"]
        action_ref = data["action_ref"]
    if isinstance(action_set, (str, Path)):
        action_set = np.load(action_set, allow_pickle=False)
    if isinstance(class_mapping, (str, Path)):
        payload = json.loads(Path(class_mapping).read_text(encoding="utf-8"))
        class_mapping = payload.get("class_mapping", {})
    return Calibration(total_ref, action_ref, scaler, action_set, class_mapping)


# =========================================================================== #
# Action matching
# =========================================================================== #

def _action_repr(action):
    return np.asarray(action.to_vect(), dtype=float)


def _find_action_index(action, action_set, atol: float = 1e-6):
    if action_set is None:
        return None
    v = _action_repr(action)
    if action_set.shape[1] != v.shape[0]:
        raise ValueError(
            f"Action-set rows have length {action_set.shape[1]} but action.to_vect() has "
            f"length {v.shape[0]}."
        )
    # max(abs(.)) is less sensitive to action-vector dimensionality than L1 sum.
    diffs = np.max(np.abs(action_set - v), axis=1)
    j = int(np.argmin(diffs))
    return j if diffs[j] <= atol else None


# =========================================================================== #
# KPI helpers
# =========================================================================== #

def _percentile(value, sorted_ref):
    ref = np.asarray(sorted_ref, dtype=float).reshape(-1)
    if len(ref) == 0:
        raise ValueError("Percentile reference array must be non-empty.")
    return float(100.0 * np.searchsorted(ref, value, side="right") / len(ref))


def uncertainty_level(percentile: float) -> str:
    """Map calibrated epistemic uncertainty percentile to low/medium/high."""
    p = float(percentile)
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p}")
    if p < 33.3333333333:
        return "low"
    if p < 66.6666666667:
        return "medium"
    return "high"


def confidence_level(percentile: float) -> str:
    """Inverse of epistemic uncertainty: low U -> high confidence, and vice versa."""
    return {"low": "high", "medium": "medium", "high": "low"}[uncertainty_level(percentile)]


def _call_agent(agent: Any, obs: Any):
    try:
        return agent.act(obs, reward=0.0, done=False)
    except TypeError:
        try:
            return agent.act(obs, 0.0, False)
        except TypeError:
            return agent.act(obs)


@torch.no_grad()
def assess_observation(obs, enn, calibration: Calibration, action: Optional[Any] = None):
    """Assess one observation and, optionally, one already-selected action.

    Passing the actual ``action`` is important for stochastic policies: the KPI
    then scores exactly the recommendation returned to the caller instead of
    invoking the policy a second time and potentially scoring another action.
    """
    enn.eval()
    device = next(enn.parameters()).device

    curated_id = _find_action_index(action, calibration.action_set) if action is not None else None

    x = np.asarray(obs.to_vect(), np.float32)
    input_dim = int(getattr(enn, "input_dim", x.size))
    if x.size != input_dim:
        if x.size < input_dim:
            raise ValueError(
                f"Observation vector has {x.size} features but ENN expects {input_dim}."
            )
        x = x[:input_dim]
    if calibration.scaler is not None:
        expected = int(getattr(calibration.scaler, "n_features_in_", input_dim))
        if expected != input_dim:
            raise ValueError(
                f"Calibration scaler expects {expected} features but ENN expects {input_dim}."
            )
        x = calibration.scaler.transform(x.reshape(1, -1)).astype(np.float32)[0]

    out = enn(torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0))
    prob = out["prob"][0].detach().cpu().numpy()
    S = float(out["S"][0].item())
    u = float(out["uncertainty"][0].item())
    # Numerical clipping protects the public percentage from tiny FP excursions.
    u_pct = 100.0 * float(np.clip(u, 0.0, 1.0))
    total_pctile = _percentile(u, calibration.total_ref)

    action_pctile = None
    if curated_id is not None:
        label = calibration.class_mapping.get(str(curated_id), curated_id if not calibration.class_mapping else None)
        if label is not None and 0 <= int(label) < len(prob):
            p_a = float(prob[int(label)])
            variance = p_a * (1.0 - p_a) / (S + 1.0)
            action_pctile = round(_percentile(variance, calibration.action_ref), 1)

    total_pctile = round(total_pctile, 1)
    return {
        "chosen_action_id": curated_id,
        "epistemic_uncertainty_pct": round(u_pct, 1),
        "epistemic_uncertainty_total_pctile": total_pctile,
        "epistemic_uncertainty_action_pctile": action_pctile,
        "epistemic_uncertainty_level": uncertainty_level(total_pctile),
        "epistemic_confidence_level": confidence_level(total_pctile),
    }


@torch.no_grad()
def assess_recommendation(obs, agent, enn, calibration, action=None):
    """KPI for an agent recommendation.

    ``action`` is optional for backwards compatibility.  New callers should
    pass the recommendation they already obtained from the policy.
    """
    if action is None:
        action = _call_agent(agent, obs)
    return assess_observation(obs, enn, calibration, action=action)
