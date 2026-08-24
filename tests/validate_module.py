"""End-to-end validation of the REAL recommendation_uncertainty.py using a
synthetic ENN with the same forward interface. Mirrors exactly what
run_example.py does, minus Grid2Op (fake obs/agent/action objects)."""
import json
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from enn_models_synthetic import EvidentialNetwork
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from recommendation_uncertainty import (
    build_calibration, save_calibration, load_calibration,
    assess_recommendation,
)

rng = np.random.RandomState(0)
torch.manual_seed(0)

OBS_DIM, ACT_DIM, K = 40, 25, 20

# --- fake Grid2Op objects ----------------------------------------------------
class FakeAction:
    def __init__(self, v): self.v = np.asarray(v, dtype=float)
    def to_vect(self): return self.v
    def as_serializable_dict(self): return {"set_bus": self.v.tolist()}

class FakeObs:
    def __init__(self, v): self.v = np.asarray(v, dtype="float32")
    def to_vect(self): return self.v

class FakeAgent:
    """act(obs, reward, done) -> Grid2Op-like action (as in the module)."""
    def __init__(self, action): self.action = action
    def act(self, obs, reward, done): return self.action

# --- curated action set + meta + scaler (as export_artifacts.py produces) ----
actions = rng.randn(K, ACT_DIM).round(3)
np.save("actions.npy", actions)

meta = {"input_dim": OBS_DIM, "num_classes": K, "n_curated_actions": K,
        "class_mapping": {str(k): k for k in range(K)}}
json.dump(meta, open("enn_meta.json", "w"))

X_train = rng.randn(500, OBS_DIM)
scaler = StandardScaler().fit(X_train)
json.dump({"type": "StandardScaler", "mean": scaler.mean_.tolist(),
           "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
           "n_features_in": int(scaler.n_features_in_)},
          open("scaler_params.json", "w"))

# --- scaler_from_json (same function as run_example.py) ----------------------
def scaler_from_json(path):
    p = json.load(open(path))
    s = StandardScaler()
    s.mean_ = np.asarray(p["mean"]); s.scale_ = np.asarray(p["scale"])
    s.var_ = np.asarray(p["var"]); s.n_features_in_ = int(p["n_features_in"])
    return s

# --- ENN + calibration (build once, save, reload) ----------------------------
enn = EvidentialNetwork(OBS_DIM, K); enn.eval()
X_scaled = scaler.transform(X_train).astype(np.float32)
total_ref, action_ref = build_calibration(enn, X_scaled)
save_calibration("calib.npz", total_ref, action_ref)

calibration = load_calibration(
    "calib.npz",
    scaler=scaler_from_json("scaler_params.json"),
    action_set="actions.npy",
    class_mapping="enn_meta.json",
)
print("refs:", len(calibration.total_ref), "states | class_mapping keys are",
      type(next(iter(calibration.class_mapping))).__name__)

ok = True

# --- case 1: action IN the curated set ---------------------------------------
obs = FakeObs(rng.randn(OBS_DIM))
agent = FakeAgent(FakeAction(actions[7]))
info = assess_recommendation(obs, agent, enn, calibration)
print("\n[1] curated action :", info)
ok &= info["chosen_action_id"] == 7
ok &= 0.0 <= info["epistemic_uncertainty_total_pctile"] <= 100.0
ok &= info["epistemic_uncertainty_action_pctile"] is not None
ok &= 0.0 <= info["epistemic_uncertainty_action_pctile"] <= 100.0

# --- case 2: do-nothing (not in the set) -------------------------------------
agent_dn = FakeAgent(FakeAction(np.zeros(ACT_DIM)))
info_dn = assess_recommendation(obs, agent_dn, enn, calibration)
print("[2] do-nothing     :", info_dn)
ok &= info_dn["chosen_action_id"] is None
ok &= info_dn["epistemic_uncertainty_action_pctile"] is None
ok &= 0.0 <= info_dn["epistemic_uncertainty_total_pctile"] <= 100.0

# --- case 3: percentile mapping is monotone and correctly bounded ------------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from recommendation_uncertainty import _percentile
lo = _percentile(calibration.total_ref[0] - 1.0, calibration.total_ref)
mid = _percentile(np.median(calibration.total_ref), calibration.total_ref)
hi = _percentile(calibration.total_ref[-1] + 1.0, calibration.total_ref)
vals = np.linspace(calibration.total_ref[0], calibration.total_ref[-1], 50)
pcts = [_percentile(v, calibration.total_ref) for v in vals]
mono = all(a <= b for a, b in zip(pcts, pcts[1:]))
print(f"[3] percentile map: below-min={lo:.1f}  median={mid:.1f}  "
      f"above-max={hi:.1f}  monotone={mono}")
ok &= lo == 0.0 and hi == 100.0 and 45.0 <= mid <= 55.0 and mono

# --- case 4: InteractiveAI wrapper (as in run_example.py) --------------------
def to_interactiveai(action, info):
    return {"title": "Topological recommendation", "use_case": "PowerGrid",
            "agent_type": 2, "actions": [action.as_serializable_dict()],
            "kpis": {"efficiency_of_the_reco": None,
                     "epistemic_uncertainty_total_pctile":
                         info["epistemic_uncertainty_total_pctile"],
                     "epistemic_uncertainty_action_pctile":
                         info["epistemic_uncertainty_action_pctile"]}}
rec = to_interactiveai(agent.action, info)
ok &= json.dumps(rec) is not None      # serialisable
print("[4] InteractiveAI kpis:", rec["kpis"])

print("\nVALIDATION:", "PASSED" if ok else "FAILED")
