"""
recommendation_uncertainty.py
=============================
Epistemic-uncertainty KPI for an RL recommendation.

Given a grid state, the RL agent, the trained Evidential Neural Network (ENN) and the
calibration, `assess_recommendation(...)` returns a dictionary with two percentiles (0-100):

    - epistemic_uncertainty_total_pctile  : total epistemic uncertainty of the situation
                                            (the ENN vacuity u = K/S);
    - epistemic_uncertainty_action_pctile : epistemic uncertainty of the action the agent chose
                                            (variance of that action's probability), or None
                                            for a do-nothing action (not in the curated set).

The agent returns a Grid2Op ACTION (not an index). To score it, the action is matched against
the curated action set (actions.npy): a do-nothing action is not in the set, so it gets no
per-action score; any other action is located in the set to obtain its index, which is then
mapped to the ENN's (remapped top-K) label via class_mapping.

Two steps:
  1) train/export the ENN bundle with training/train_enn.py, which writes the
     percentile calibration next to the model artifacts;
  2) at run time, call assess_recommendation(obs, agent, enn, calibration) per recommendation.

Per state, the ENN gives a Dirichlet over its K topology actions:
    e_k = evidence (>=0),  alpha_k = e_k + 1,  S = sum_k alpha_k,  p_k = alpha_k / S
    total epistemic uncertainty : u = K / S
    chosen action a uncertainty : Var(p_a) = p_a (1 - p_a) / (S + 1)
"""
import json

import numpy as np
import torch


# =========================================================================== #
#  Calibration: BUILD (run once, from the ENN training data)
# =========================================================================== #

@torch.no_grad()
def _state_measures(enn, x_scaled, device, label=None):
    """ENN forward on one scaled input vector -> (vacuity u, action variance, argmax label)."""
    out = enn(torch.as_tensor(np.asarray(x_scaled, np.float32), device=device).unsqueeze(0))
    prob = out["prob"][0].detach().cpu().numpy()
    S = float(out["S"][0].item())
    u = float(out["uncertainty"][0].item())
    a = int(np.argmax(prob)) if label is None else int(label)
    p_a = float(prob[a])
    return u, p_a * (1.0 - p_a) / (S + 1.0), int(np.argmax(prob))


def build_calibration(enn, input_vectors_scaled):
    """Compute the two reference distributions from scaled ENN-training inputs.

    Returns (total_ref, action_ref): sorted arrays of the vacuity and of the (top-action)
    probability variance over the training states."""
    enn.eval()
    device = next(enn.parameters()).device
    X = np.asarray(input_vectors_scaled, np.float32)
    tot, act = [], []
    for i in range(len(X)):
        u, var_a, _ = _state_measures(enn, X[i], device)
        tot.append(u); act.append(var_a)
    return np.sort(np.asarray(tot, float)), np.sort(np.asarray(act, float))


def save_calibration(path, total_ref, action_ref):
    np.savez(path, total_ref=total_ref, action_ref=action_ref)


# =========================================================================== #
#  Calibration: LOAD (bundles references + scaler + action set + action map)
# =========================================================================== #

class Calibration:
    """Holds the percentile references, the input scaler, the curated action set and the
    action label map."""

    def __init__(self, total_ref, action_ref, scaler=None, action_set=None, class_mapping=None):
        self.total_ref = np.sort(np.asarray(total_ref, float))
        self.action_ref = np.sort(np.asarray(action_ref, float))
        self.scaler = scaler                        # StandardScaler used in ENN training
        self.action_set = action_set                # (K_full, action_dim) curated actions (actions.npy)
        self.class_mapping = class_mapping or {}     # curated action id (str) -> ENN class label


def load_calibration(calib_path, scaler=None, action_set=None, class_mapping=None):
    """Load the precomputed references (.npz) and bundle the scaler, the curated action set and
    the action label map.

    scaler        : the StandardScaler used during ENN training.
    action_set    : the curated action set (actions.npy). A path to the .npy, or an array whose
                    rows are each action's vector (action.to_vect()); used to locate the agent's
                    returned action and obtain its index.
    class_mapping : {curated_action_id: ENN_label}, or a path to enn_meta_*.json (its
                    "class_mapping" field is used)."""
    data = np.load(calib_path)
    if isinstance(action_set, str):
        action_set = np.load(action_set)
    if isinstance(class_mapping, str):
        class_mapping = json.load(open(class_mapping))["class_mapping"]
    return Calibration(data["total_ref"], data["action_ref"], scaler, action_set, class_mapping)


# =========================================================================== #
#  Locating the agent's action in the curated set
# =========================================================================== #

def _action_repr(action):
    """Vector representation of a Grid2Op action, used to match it against actions.npy.
    Default: action.to_vect(). If your actions.npy stores a different encoding (e.g. set_bus
    arrays), change this and `action_set` to use that same encoding."""
    return np.asarray(action.to_vect(), dtype=float)


def _find_action_index(action, action_set):
    """Index of `action` within the curated action_set; None if it is not in the set
    (e.g. a do-nothing action)."""
    v = _action_repr(action)
    if action_set.shape[1] != v.shape[0]:
        raise ValueError(
            f"actions.npy rows have length {action_set.shape[1]} but action.to_vect() has "
            f"length {v.shape[0]}. Store the curated set as action.to_vect() vectors, or adapt "
            f"_action_repr().")
    diffs = np.abs(action_set - v).sum(axis=1)
    j = int(np.argmin(diffs))
    return j if diffs[j] < 1e-6 else None             # no match (do-nothing / outside set) -> None


# =========================================================================== #
#  The KPI: assess one recommendation
# =========================================================================== #

def _percentile(value, sorted_ref):
    return float(100.0 * np.searchsorted(sorted_ref, value, side="right") / len(sorted_ref))


@torch.no_grad()
def assess_recommendation(obs, agent, enn, calibration):
    """Epistemic-uncertainty KPI for the agent's recommendation in state `obs`.

    obs         : Grid2Op observation.
    agent       : the RL agent. Its chosen action is read via agent.act(obs, reward, done) and
                  is a Grid2Op action object (do-nothing or a topology action).
    enn         : the trained ENN (already loaded).
    calibration : a Calibration from load_calibration() (references + scaler + action set + map).

    Returns
    -------
    dict:
        {
          "chosen_action_id": int | None,                       # curated index; None if do-nothing
          "epistemic_uncertainty_total_pctile": float,          # 0-100, the situation
          "epistemic_uncertainty_action_pctile": float | None,  # 0-100; None for do-nothing
        }
    """
    enn.eval()
    device = next(enn.parameters()).device

    # 1) the action the agent chose (a Grid2Op action object)
    try:
        action = agent.act(obs, 0.0, False)            # standard Grid2Op agent signature
    except TypeError:
        action = agent.act(obs)

    # 2) locate it in the curated action set (do-nothing -> not found -> None)
    curated_id = (_find_action_index(action, calibration.action_set)
                  if calibration.action_set is not None else None)

    # 3) ENN forward on the scaled observation
    x = np.asarray(obs.to_vect(), np.float32)
    if calibration.scaler is not None:
        x = calibration.scaler.transform(x.reshape(1, -1)).astype(np.float32)[0]
    out = enn(torch.as_tensor(x, device=device).unsqueeze(0))
    prob = out["prob"][0].detach().cpu().numpy()
    S = float(out["S"][0].item())
    u = float(out["uncertainty"][0].item())

    # 4) total epistemic uncertainty -> percentile
    total_pctile = _percentile(u, calibration.total_ref)

    # 5) chosen-action epistemic uncertainty -> percentile (None for do-nothing)
    action_pctile = None
    if curated_id is not None:
        label = (calibration.class_mapping.get(str(curated_id))
                 if calibration.class_mapping else curated_id)
        if label is not None:
            p_a = float(prob[int(label)])
            action_pctile = round(_percentile(p_a * (1.0 - p_a) / (S + 1.0), calibration.action_ref), 1)

    return {
        "chosen_action_id": curated_id,
        "epistemic_uncertainty_total_pctile": round(total_pctile, 1),
        "epistemic_uncertainty_action_pctile": action_pctile,
    }
